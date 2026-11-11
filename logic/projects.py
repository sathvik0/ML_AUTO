"""Projects: where work is saved and who may see or change it.

Every read and write goes through here, and every function takes the
signed-in user (`utils.auth.user()`: email and global role), so access is
checked in one place:

- A project's members have a project role: owner, editor or viewer.
- What a user may do is the lower of their project role and their global
  role (a global viewer is a project viewer everywhere). Admins may see and
  change every project.
- Only owners share, rename, archive or delete; a project always keeps at
  least one owner.

Datasets are stored as Parquet, runs and registered models with joblib
(`utils.storage`); the database (`utils.db`) holds what they are and who
made them.
"""
import io

import joblib
import pandas as pd
from sqlalchemy import func, select

from logic import audit
from utils import db, storage

PROJECT_ROLES = ("viewer", "editor", "owner")


class AccessDenied(Exception):
    """The user may not do this, in words the page can show."""


def _email(user):
    return (user.get("email") or "").strip().lower()


def _admin(user):
    return user.get("role") == "admin"


def _global_can_edit(user):
    return user.get("role") in ("analyst", "admin")


def touch_user(user):
    """Record a signed-in user (for sharing pick-lists and audit)."""
    if not user.get("sub"):
        return
    with db.session() as s:
        row = s.get(db.User, user["sub"]) or db.User(sub=user["sub"])
        row.email, row.name, row.last_seen = (_email(user),
                                              user.get("name") or "",
                                              db.now())
        s.merge(row)
        audit.record(s, user, "auth.sign_in")
        s.commit()


# ── access ──────────────────────────────────────────────────────────────
def project_role(s, user, project_id):
    """The user's effective role on a project, or None."""
    if _admin(user):
        return "owner"
    member = s.scalar(select(db.Member).where(
        db.Member.project_id == project_id,
        db.Member.email == _email(user)))
    if member is None:
        return None
    if not _global_can_edit(user):
        return "viewer"
    return member.role


def _need(s, user, project_id, at_least):
    role = project_role(s, user, project_id)
    project = s.get(db.Project, project_id)
    if project is None or role is None:
        raise AccessDenied("That project does not exist or is not shared "
                           "with you.")
    if PROJECT_ROLES.index(role) < PROJECT_ROLES.index(at_least):
        raise AccessDenied(f"You are a {role} of {project.name}; this "
                           f"needs {at_least}.")
    return project, role


def can(user, project_id, at_least):
    with db.session() as s:
        role = project_role(s, user, project_id)
    return role is not None and \
        PROJECT_ROLES.index(role) >= PROJECT_ROLES.index(at_least)


# ── projects ────────────────────────────────────────────────────────────
def create_project(user, name, description=""):
    if not _global_can_edit(user):
        raise AccessDenied("Your role cannot create projects.")
    name = (name or "").strip()
    if not name:
        raise AccessDenied("Give the project a name.")
    with db.session() as s:
        p = db.Project(name=name[:200], description=description or "",
                       owner_email=_email(user))
        s.add(p)
        s.flush()
        s.add(db.Member(project_id=p.id, email=_email(user), role="owner",
                        added_by=_email(user)))
        s.add(db.Workspace(project_id=p.id, steps=[], config={},
                           updated_by=_email(user)))
        audit.record(s, user, "project.create", p.id, "project", p.id, name=p.name)
        s.commit()
        return {"id": p.id, "name": p.name, "role": "owner"}


def list_projects(user):
    """The projects the user can see, most recently changed first."""
    with db.session() as s:
        q = select(db.Project).where(db.Project.archived.is_(False))
        if not _admin(user):
            q = q.join(db.Member, db.Member.project_id == db.Project.id) \
                .where(db.Member.email == _email(user))
        rows = s.scalars(q.order_by(db.Project.updated_at.desc())).all()
        return [{"id": p.id, "name": p.name, "description": p.description,
                 "owner": p.owner_email, "updated_at": p.updated_at,
                 "role": project_role(s, user, p.id)} for p in rows]


def get_project(user, project_id):
    with db.session() as s:
        p, role = _need(s, user, project_id, "viewer")
        return {"id": p.id, "name": p.name, "description": p.description,
                "owner": p.owner_email, "role": role}


def rename(user, project_id, name):
    with db.session() as s:
        p, _ = _need(s, user, project_id, "owner")
        p.name, p.updated_at = name.strip()[:200] or p.name, db.now()
        audit.record(s, user, "project.rename", project_id, "project", project_id, name=p.name)
        s.commit()


def archive(user, project_id):
    with db.session() as s:
        p, _ = _need(s, user, project_id, "owner")
        p.archived, p.updated_at = True, db.now()
        audit.record(s, user, "project.archive", project_id, "project", project_id)
        s.commit()


def delete(user, project_id):
    """The project, its rows and its stored files, for good."""
    with db.session() as s:
        p, _ = _need(s, user, project_id, "owner")
        for model in (db.ModelVersion, db.Run, db.Workspace, db.Dataset,
                      db.Member):
            for row in s.scalars(select(model).where(
                    model.project_id == project_id)):
                s.delete(row)
        s.delete(p)
        audit.record(s, user, "project.delete", project_id, "project", project_id, name=p.name)
        s.commit()
    storage.delete_prefix(f"projects/{project_id}/")


# ── sharing ─────────────────────────────────────────────────────────────
def members(user, project_id):
    with db.session() as s:
        _need(s, user, project_id, "viewer")
        rows = s.scalars(select(db.Member).where(
            db.Member.project_id == project_id).order_by(db.Member.email))
        return [{"email": m.email, "role": m.role, "added_by": m.added_by}
                for m in rows]


def share(user, project_id, email, role):
    email = (email or "").strip().lower()
    if role not in PROJECT_ROLES:
        raise AccessDenied(f"Unknown role {role}.")
    if "@" not in email:
        raise AccessDenied("Enter an email address.")
    with db.session() as s:
        _need(s, user, project_id, "owner")
        m = s.scalar(select(db.Member).where(
            db.Member.project_id == project_id, db.Member.email == email))
        if m is None:
            s.add(db.Member(project_id=project_id, email=email, role=role,
                            added_by=_email(user)))
        else:
            if m.role == "owner" and role != "owner":
                _keep_an_owner(s, project_id, email)
            m.role = role
        audit.record(s, user, "project.share", project_id, "member", email, role=role)
        s.commit()


def unshare(user, project_id, email):
    email = (email or "").strip().lower()
    with db.session() as s:
        _need(s, user, project_id, "owner")
        m = s.scalar(select(db.Member).where(
            db.Member.project_id == project_id, db.Member.email == email))
        if m is None:
            return
        if m.role == "owner":
            _keep_an_owner(s, project_id, email)
        s.delete(m)
        audit.record(s, user, "project.unshare", project_id, "member", email)
        s.commit()


def _keep_an_owner(s, project_id, leaving):
    owners = s.scalars(select(db.Member.email).where(
        db.Member.project_id == project_id, db.Member.role == "owner")).all()
    if [o for o in owners if o != leaving] == []:
        raise AccessDenied("A project needs at least one owner.")


# ── datasets and the workspace ──────────────────────────────────────────
def _parquet(df):
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    # Mixed-type object columns (numbers and text) are stored as text:
    # Parquet needs one type per column.
    for c in out.columns:
        if out[c].dtype == object:
            kinds = {type(v) for v in out[c].dropna().head(1000)}
            if len(kinds) > 1:
                out[c] = out[c].map(lambda v: v if pd.isna(v) else str(v))
    buf = io.BytesIO()
    out.to_parquet(buf, index=False)
    return buf.getvalue()


def save_dataset(user, project_id, df, name, source):
    """Store a newly loaded file and make it the project's dataset: the
    preparation steps and choices start afresh with it."""
    with db.session() as s:
        _need(s, user, project_id, "editor")
        d = db.Dataset(project_id=project_id, name=(name or "data")[:300],
                       source=source, rows=len(df), columns=df.shape[1],
                       object_key="", created_by=_email(user))
        s.add(d)
        s.flush()
        d.object_key = f"projects/{project_id}/datasets/{d.id}.parquet"
        storage.put(d.object_key, _parquet(df))
        ws = s.get(db.Workspace, project_id) or db.Workspace(
            project_id=project_id)
        ws.dataset_id, ws.steps, ws.config = d.id, [], {}
        ws.updated_by, ws.updated_at = _email(user), db.now()
        s.merge(ws)
        s.get(db.Project, project_id).updated_at = db.now()
        audit.record(s, user, "dataset.save", project_id, "dataset", d.id, rows=len(df), columns=df.shape[1], source=source)
        s.commit()
        return d.id


def load_dataset(user, project_id, dataset_id):
    with db.session() as s:
        _need(s, user, project_id, "viewer")
        d = s.get(db.Dataset, dataset_id)
        if d is None or d.project_id != project_id:
            raise AccessDenied("That dataset is not in this project.")
        key, name = d.object_key, d.name
    return pd.read_parquet(io.BytesIO(storage.get(key))), name


def workspace(user, project_id):
    with db.session() as s:
        _need(s, user, project_id, "viewer")
        ws = s.get(db.Workspace, project_id)
        if ws is None:
            return {"dataset_id": None, "steps": [], "config": {}}
        return {"dataset_id": ws.dataset_id, "steps": list(ws.steps or []),
                "config": dict(ws.config or {}),
                "updated_by": ws.updated_by, "updated_at": ws.updated_at}


def save_workspace(user, project_id, steps, config):
    with db.session() as s:
        _need(s, user, project_id, "editor")
        ws = s.get(db.Workspace, project_id) or db.Workspace(
            project_id=project_id)
        ws.steps, ws.config = list(steps or []), dict(config or {})
        ws.updated_by, ws.updated_at = _email(user), db.now()
        s.merge(ws)
        s.get(db.Project, project_id).updated_at = db.now()
        s.commit()


# ── runs ────────────────────────────────────────────────────────────────
def _leaderboard(run):
    out = []
    for r in run.get("results") or []:
        score = None if r.get("error") else (r.get("test") or {}).get(
            run.get("metric"))
        out.append({"key": r["key"], "label": r.get("label", r["key"]),
                    "score": None if score is None else float(score),
                    "error": bool(r.get("error"))})
    return out


def save_run(user, project_id, run, dataset_id=None):
    """Store a finished run (models, predictions, everything the results
    page shows) and return its id."""
    best = next((r for r in run.get("results") or []
                 if r["key"] == run.get("best")), None)
    score = (best or {}).get("test", {}).get(run.get("metric")) \
        if best else None
    with db.session() as s:
        _need(s, user, project_id, "editor")
        r = db.Run(project_id=project_id, dataset_id=dataset_id,
                   task=run["task"], target=run.get("target"),
                   best_model=run.get("best"), metric=run.get("metric"),
                   score=None if score is None else float(score),
                   leaderboard=_leaderboard(run),
                   external=bool(run.get("external")), object_key="",
                   sklearn=str(run.get("sklearn") or ""),
                   created_by=_email(user))
        s.add(r)
        s.flush()
        r.object_key = f"projects/{project_id}/runs/{r.id}.joblib"
        buf = io.BytesIO()
        joblib.dump(run, buf, compress=3)
        storage.put(r.object_key, buf.getvalue())
        s.get(db.Project, project_id).updated_at = db.now()
        audit.record(s, user, "run.save", project_id, "run", r.id, task=run["task"], best=run.get("best"), external=bool(run.get("external")))
        s.commit()
        return r.id


def set_threshold(user, project_id, run_id, run, model_key, threshold):
    """A stored run with a model's cut-off changed: rewritten in place, so a
    project reopened later registers the model with the same cut-off."""
    with db.session() as s:
        _need(s, user, project_id, "editor")
        r = s.get(db.Run, run_id)
        if r is None or r.project_id != project_id:
            raise AccessDenied("That run is not in this project.")
        buf = io.BytesIO()
        joblib.dump(run, buf, compress=3)
        storage.put(r.object_key, buf.getvalue())
        audit.record(s, user, "run.threshold", project_id, "run", r.id,
                     model=model_key, threshold=threshold)
        s.commit()


def list_runs(user, project_id):
    with db.session() as s:
        _need(s, user, project_id, "viewer")
        rows = s.scalars(select(db.Run).where(
            db.Run.project_id == project_id)
            .order_by(db.Run.created_at.desc())).all()
        return [{"id": r.id, "task": r.task, "target": r.target,
                 "best_model": r.best_model, "metric": r.metric,
                 "score": r.score, "leaderboard": r.leaderboard,
                 "external": r.external, "created_by": r.created_by,
                 "created_at": r.created_at, "name": r.name or "",
                 "note": r.note or "", "dataset_id": r.dataset_id}
                for r in rows]


RUN_NAME_MAX = 200
RUN_NOTE_MAX = 4000


def describe_run(user, project_id, run_id, name, note):
    """Give a run a name and a note (what it tried, what was learnt)."""
    name, note = (name or "").strip(), (note or "").strip()
    if len(name) > RUN_NAME_MAX or len(note) > RUN_NOTE_MAX:
        raise AccessDenied(f"Names are up to {RUN_NAME_MAX} characters and "
                           f"notes up to {RUN_NOTE_MAX}.")
    with db.session() as s:
        _need(s, user, project_id, "editor")
        r = s.get(db.Run, run_id)
        if r is None or r.project_id != project_id:
            raise AccessDenied("That run is not in this project.")
        r.name, r.note = name, note
        audit.record(s, user, "run.describe", project_id, "run", r.id,
                     name=name, note_chars=len(note))
        s.commit()


def load_run(user, project_id, run_id):
    with db.session() as s:
        _need(s, user, project_id, "viewer")
        r = s.get(db.Run, run_id)
        if r is None or r.project_id != project_id:
            raise AccessDenied("That run is not in this project.")
        key = r.object_key
    # Written by this app to its own private storage, never uploaded.
    return joblib.load(io.BytesIO(storage.get(key)))


# ── the model registry ──────────────────────────────────────────────────
def register_model(user, project_id, run_id, model_bytes, *, name,
                   model_key, task, target=None, metric=None, score=None,
                   description=""):
    """A new version of the named model, from a stored run."""
    name = (name or "").strip()[:200]
    if not name:
        raise AccessDenied("Give the model a name.")
    with db.session() as s:
        _need(s, user, project_id, "editor")
        last = s.scalar(select(func.max(db.ModelVersion.version)).where(
            db.ModelVersion.project_id == project_id,
            db.ModelVersion.name == name)) or 0
        v = db.ModelVersion(project_id=project_id, run_id=run_id, name=name,
                            version=last + 1, model_key=model_key,
                            object_key="", task=task, target=target,
                            metric=metric, score=score,
                            description=description or "",
                            created_by=_email(user))
        s.add(v)
        s.flush()
        v.object_key = (f"projects/{project_id}/models/{v.id}/"
                        f"v{v.version}.joblib")
        storage.put(v.object_key, model_bytes)
        audit.record(s, user, "model.register", project_id, "model_version", v.id, model=name, version=v.version)
        s.commit()
        return {"id": v.id, "name": name, "version": v.version}


def list_models(user, project_id):
    with db.session() as s:
        _need(s, user, project_id, "viewer")
        rows = s.scalars(select(db.ModelVersion).where(
            db.ModelVersion.project_id == project_id).order_by(
            db.ModelVersion.name, db.ModelVersion.version.desc())).all()
        return [{"id": v.id, "name": v.name, "version": v.version,
                 "model_key": v.model_key, "task": v.task,
                 "target": v.target, "metric": v.metric, "score": v.score,
                 "stage": v.stage, "run_id": v.run_id,
                 "description": v.description, "created_by": v.created_by,
                 "created_at": v.created_at} for v in rows]


def model_file(user, project_id, version_id):
    with db.session() as s:
        _need(s, user, project_id, "viewer")
        v = s.get(db.ModelVersion, version_id)
        if v is None or v.project_id != project_id:
            raise AccessDenied("That model is not in this project.")
        key = v.object_key
    return storage.get(key)
