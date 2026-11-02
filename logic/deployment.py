"""Putting registered models to work: stages with approval, API keys for
the model API, and scheduled batch scoring.

Stages. A version is in one of none, staging, production, archived.
Editors move versions to staging, back to none, or to archived directly.
Production needs a request and a second person's approval: a project owner
(or an admin) who is not the requester. Approving a version archives the
model's previous production version, so a name has one production version.

API keys. Project owners create keys; a key is shown once and only its
SHA-256 is stored. A key reaches its own project's models and nothing else.

Batch jobs. A model name and stage (production by default), a source (a
project dataset, an uploaded file, or a Snowflake query), a destination
(project storage, or a Snowflake table) and an optional cron schedule.
`run_due()` runs the jobs whose time has come; on AWS a scheduled task calls
it every 15 minutes.

Functions that take `user` check access through `logic.projects`; those
without one are the service's own (the API and the scheduler).
"""
import hashlib
import io
import secrets
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select, update

from logic import projects, serving
from logic.projects import AccessDenied, _email, _need
from utils import db, storage

STAGES = ("none", "staging", "production", "archived")
DIRECT = ("none", "staging", "archived")      # no approval needed
KEY_PREFIX = "mlk_"


def _now():
    return datetime.now(timezone.utc)


def _aware(t):
    """SQLite hands back naive datetimes; everything here is UTC."""
    return t if t is None or t.tzinfo else t.replace(tzinfo=timezone.utc)


def _version(s, project_id, version_id):
    v = s.get(db.ModelVersion, version_id)
    if v is None or v.project_id != project_id:
        raise AccessDenied("That model version is not in this project.")
    return v


# ── stages ──────────────────────────────────────────────────────────────
def set_stage(user, project_id, version_id, stage, comment=""):
    """Staging, none or archived, at once (editors)."""
    if stage not in DIRECT:
        raise AccessDenied("Production needs a request and an approval.")
    with db.session() as s:
        _need(s, user, project_id, "editor")
        v = _version(s, project_id, version_id)
        s.add(db.Transition(project_id=project_id, version_id=v.id,
                            to_stage=stage, status="approved",
                            comment=comment, requested_by=_email(user),
                            decided_by=_email(user), decided_at=_now()))
        v.stage = stage
        s.commit()


def request_production(user, project_id, version_id, comment=""):
    with db.session() as s:
        _need(s, user, project_id, "editor")
        v = _version(s, project_id, version_id)
        if v.stage == "production":
            raise AccessDenied("That version is already in production.")
        pending = s.scalar(select(db.Transition).where(
            db.Transition.version_id == v.id,
            db.Transition.status == "pending"))
        if pending is not None:
            raise AccessDenied("A request for that version is already "
                               "waiting.")
        t = db.Transition(project_id=project_id, version_id=v.id,
                          to_stage="production", status="pending",
                          comment=comment or "", requested_by=_email(user))
        s.add(t)
        s.commit()
        return t.id


def _decide(user, project_id, transition_id, approve, note):
    with db.session() as s:
        _need(s, user, project_id, "owner")
        t = s.get(db.Transition, transition_id)
        if t is None or t.project_id != project_id or t.status != "pending":
            raise AccessDenied("That request is not waiting for a decision.")
        if t.requested_by == _email(user):
            raise AccessDenied("Someone other than the requester must "
                               "decide.")
        t.status = "approved" if approve else "rejected"
        t.decided_by, t.decided_at = _email(user), _now()
        t.decision_note = note or ""
        if approve:
            v = s.get(db.ModelVersion, t.version_id)
            for other in s.scalars(select(db.ModelVersion).where(
                    db.ModelVersion.project_id == project_id,
                    db.ModelVersion.name == v.name,
                    db.ModelVersion.stage == "production",
                    db.ModelVersion.id != v.id)):
                other.stage = "archived"
            v.stage = "production"
        s.commit()


def approve(user, project_id, transition_id, note=""):
    _decide(user, project_id, transition_id, True, note)


def reject(user, project_id, transition_id, note=""):
    _decide(user, project_id, transition_id, False, note)


def cancel(user, project_id, transition_id):
    with db.session() as s:
        _need(s, user, project_id, "viewer")
        t = s.get(db.Transition, transition_id)
        if t is None or t.project_id != project_id or t.status != "pending":
            raise AccessDenied("That request is not waiting.")
        if t.requested_by != _email(user):
            raise AccessDenied("Only the requester can withdraw it.")
        t.status = "cancelled"
        t.decided_by, t.decided_at = _email(user), _now()
        s.commit()


def transitions(user, project_id, pending_only=False):
    with db.session() as s:
        _need(s, user, project_id, "viewer")
        q = select(db.Transition, db.ModelVersion).join(
            db.ModelVersion, db.ModelVersion.id == db.Transition.version_id
        ).where(db.Transition.project_id == project_id)
        if pending_only:
            q = q.where(db.Transition.status == "pending")
        rows = s.execute(q.order_by(db.Transition.requested_at.desc()))
        return [{"id": t.id, "version_id": v.id, "model": v.name,
                 "version": v.version, "to_stage": t.to_stage,
                 "status": t.status, "comment": t.comment,
                 "requested_by": t.requested_by,
                 "requested_at": t.requested_at, "decided_by": t.decided_by,
                 "decided_at": t.decided_at, "note": t.decision_note}
                for t, v in rows]


# ── serving lookup (service-level) ──────────────────────────────────────
def resolve_version(project_id, name, stage="production", version=None):
    """The version a caller means: a number, or the one in a stage."""
    with db.session() as s:
        q = select(db.ModelVersion).where(
            db.ModelVersion.project_id == project_id,
            db.ModelVersion.name == name)
        if version is not None:
            q = q.where(db.ModelVersion.version == int(version))
        else:
            q = q.where(db.ModelVersion.stage == stage)
        v = s.scalars(q.order_by(db.ModelVersion.version.desc())).first()
        if v is None:
            return None
        return {"id": v.id, "name": v.name, "version": v.version,
                "stage": v.stage, "task": v.task, "target": v.target,
                "object_key": v.object_key}


def model_info(version):
    """The loaded model for a resolved version, cached per process."""
    return serving.load(version["id"],
                        lambda: storage.get(version["object_key"]))


# ── API keys ────────────────────────────────────────────────────────────
def _hash(key):
    return hashlib.sha256(key.encode()).hexdigest()


def create_key(user, project_id, name):
    """(plaintext key, id). The key is not stored; show it once."""
    name = (name or "").strip()[:200] or "API key"
    key = KEY_PREFIX + secrets.token_urlsafe(32)
    with db.session() as s:
        _need(s, user, project_id, "owner")
        k = db.ApiKey(project_id=project_id, name=name, prefix=key[:10],
                      key_hash=_hash(key), created_by=_email(user))
        s.add(k)
        s.commit()
        return key, k.id


def list_keys(user, project_id):
    with db.session() as s:
        _need(s, user, project_id, "owner")
        rows = s.scalars(select(db.ApiKey).where(
            db.ApiKey.project_id == project_id)
            .order_by(db.ApiKey.created_at.desc()))
        return [{"id": k.id, "name": k.name, "prefix": k.prefix,
                 "created_by": k.created_by, "created_at": k.created_at,
                 "last_used_at": k.last_used_at,
                 "revoked": k.revoked_at is not None} for k in rows]


def revoke_key(user, project_id, key_id):
    with db.session() as s:
        _need(s, user, project_id, "owner")
        k = s.get(db.ApiKey, key_id)
        if k is None or k.project_id != project_id:
            raise AccessDenied("That key is not in this project.")
        k.revoked_at = k.revoked_at or _now()
        s.commit()


def authenticate(key):
    """The project id a live key belongs to, or None. Marks it used at
    most once a minute (a write per request would be wasteful)."""
    if not key or not key.startswith(KEY_PREFIX):
        return None
    with db.session() as s:
        k = s.scalar(select(db.ApiKey).where(
            db.ApiKey.key_hash == _hash(key)))
        if k is None or k.revoked_at is not None:
            return None
        last = _aware(k.last_used_at)
        if last is None or (_now() - last).total_seconds() > 60:
            k.last_used_at = _now()
            s.commit()
        return k.project_id


# ── batch jobs ──────────────────────────────────────────────────────────
def next_run(schedule, after=None):
    """The next time a cron schedule fires after `after` (UTC), or None
    for a job that only runs on demand."""
    if not (schedule or "").strip():
        return None
    from croniter import croniter
    if not croniter.is_valid(schedule):
        raise AccessDenied(f"Not a valid schedule: {schedule}")
    return croniter(schedule, after or _now()).get_next(datetime) \
        .astimezone(timezone.utc)


SOURCES = ("dataset", "file", "snowflake")
DESTINATIONS = ("storage", "snowflake")


def save_input(user, project_id, df, name):
    """An uploaded file stored for a batch job; its source spec."""
    with db.session() as s:
        _need(s, user, project_id, "editor")
    key = f"projects/{project_id}/batch-inputs/{db.new_id()}.parquet"
    storage.put(key, projects._parquet(df))
    return {"type": "file", "object_key": key, "name": name,
            "rows": len(df)}


def create_job(user, project_id, *, name, model_name, source, destination,
               schedule="", stage="production"):
    if stage not in ("production", "staging"):
        raise AccessDenied("Batch jobs score the production or staging "
                           "version.")
    if source.get("type") not in SOURCES:
        raise AccessDenied("Unknown source.")
    if destination.get("type") not in DESTINATIONS:
        raise AccessDenied("Unknown destination.")
    with db.session() as s:
        _need(s, user, project_id, "editor")
        j = db.BatchJob(project_id=project_id,
                        name=(name or model_name)[:200],
                        model_name=model_name, stage=stage, source=source,
                        destination=destination,
                        schedule=(schedule or "").strip(),
                        next_run_at=next_run(schedule),
                        created_by=_email(user))
        s.add(j)
        s.commit()
        return j.id


def list_jobs(user, project_id):
    with db.session() as s:
        _need(s, user, project_id, "viewer")
        rows = s.scalars(select(db.BatchJob).where(
            db.BatchJob.project_id == project_id)
            .order_by(db.BatchJob.created_at.desc()))
        return [{"id": j.id, "name": j.name, "model_name": j.model_name,
                 "stage": j.stage, "source": j.source,
                 "destination": j.destination, "schedule": j.schedule,
                 "enabled": j.enabled, "next_run_at": j.next_run_at,
                 "last_run_at": j.last_run_at, "last_status": j.last_status,
                 "created_by": j.created_by} for j in rows]


def set_enabled(user, project_id, job_id, enabled):
    with db.session() as s:
        _need(s, user, project_id, "editor")
        j = s.get(db.BatchJob, job_id)
        if j is None or j.project_id != project_id:
            raise AccessDenied("That job is not in this project.")
        j.enabled = bool(enabled)
        j.next_run_at = next_run(j.schedule) if enabled else None
        s.commit()


def delete_job(user, project_id, job_id):
    with db.session() as s:
        _need(s, user, project_id, "editor")
        j = s.get(db.BatchJob, job_id)
        if j is None or j.project_id != project_id:
            raise AccessDenied("That job is not in this project.")
        for r in s.scalars(select(db.BatchRun).where(
                db.BatchRun.job_id == job_id)):
            s.delete(r)
        s.delete(j)
        s.commit()


def job_runs(user, project_id, job_id, limit=20):
    with db.session() as s:
        _need(s, user, project_id, "viewer")
        rows = s.scalars(select(db.BatchRun).where(
            db.BatchRun.job_id == job_id, db.BatchRun.project_id ==
            project_id).order_by(db.BatchRun.started_at.desc()).limit(limit))
        return [{"id": r.id, "status": r.status, "rows": r.rows,
                 "message": r.message, "triggered_by": r.triggered_by,
                 "started_at": r.started_at, "finished_at": r.finished_at,
                 "has_output": bool(r.output_key)} for r in rows]


def run_output(user, project_id, run_id):
    with db.session() as s:
        _need(s, user, project_id, "viewer")
        r = s.get(db.BatchRun, run_id)
        if r is None or r.project_id != project_id or not r.output_key:
            raise AccessDenied("That run has no stored output.")
        key = r.output_key
    return storage.get(key)


def _read_source(project_id, source, snowflake):
    kind = source["type"]
    if kind in ("dataset", "file"):
        if kind == "dataset":
            with db.session() as s:
                d = s.get(db.Dataset, source["dataset_id"])
                if d is None or d.project_id != project_id:
                    raise serving.ScoringError("The dataset is gone.")
                key = d.object_key
        else:
            key = source["object_key"]
            if not key.startswith(f"projects/{project_id}/"):
                raise serving.ScoringError("The file is not this project's.")
        return pd.read_parquet(io.BytesIO(storage.get(key)))
    conn = snowflake()
    from logic import snowflake_io as sf
    df, err = sf.run_query(conn, source["query"], source.get("database"),
                           source.get("schema"))
    if err:
        raise serving.ScoringError(f"Snowflake: {err}")
    return df


def _write(project_id, run_id, frame, destination, snowflake):
    if destination["type"] == "storage":
        key = f"projects/{project_id}/batch-outputs/{run_id}.csv"
        storage.put(key, frame.to_csv(index=False).encode("utf-8"))
        return key
    from logic import snowflake_io as sf
    rows, err = sf.write_table(snowflake(), frame, destination["database"],
                               destination["schema"], destination["table"],
                               destination.get("mode", "Replace table"))
    if err:
        raise serving.ScoringError(f"Snowflake: {err}")
    return ""


def _service_snowflake():
    """A connection with the deployment's own Snowflake credentials."""
    from logic import snowflake_io as sf
    conn, err = sf.connect(sf.resolve_config())
    if err:
        raise serving.ScoringError(f"Snowflake: {err}")
    return conn


def run_job(job_id, triggered_by="schedule", snowflake=None):
    """Run one job now and record it; returns the run's summary. Failures
    are recorded on the run, not raised."""
    snowflake = snowflake or _service_snowflake
    with db.session() as s:
        j = s.get(db.BatchJob, job_id)
        if j is None:
            raise AccessDenied("No such job.")
        job = {"project_id": j.project_id, "model_name": j.model_name,
               "stage": j.stage, "source": dict(j.source),
               "destination": dict(j.destination), "schedule": j.schedule}
        r = db.BatchRun(job_id=job_id, project_id=j.project_id,
                        status="running", triggered_by=triggered_by)
        s.add(r)
        s.commit()
        run_id = r.id
    status, rows, key, message, version_id = "failed", None, "", "", None
    try:
        version = resolve_version(job["project_id"], job["model_name"],
                                  job["stage"])
        if version is None:
            raise serving.ScoringError(
                f"No {job['stage']} version of {job['model_name']}.")
        version_id = version["id"]
        data = _read_source(job["project_id"], job["source"], snowflake)
        out = serving.score(model_info(version), data)
        key = _write(job["project_id"], run_id, out, job["destination"],
                     snowflake)
        status, rows = "succeeded", len(out)
        message = (f"Scored {rows:,} rows with {version['name']} "
                   f"v{version['version']}.")
    except serving.ScoringError as e:
        message = str(e)
    except Exception as e:
        message = f"{type(e).__name__}: the job could not run."
        from utils import log
        log.event(log.get("deployment"), "batch_job_failed",
                  level=40, exc_info=True, job=job_id)
    with db.session() as s:
        r = s.get(db.BatchRun, run_id)
        r.status, r.rows, r.output_key, r.message = status, rows, key, \
            message
        r.model_version_id, r.finished_at = version_id, _now()
        j = s.get(db.BatchJob, job_id)
        j.last_run_at, j.last_status = _now(), status
        # The scheduler's claim already set the next time; only a job with
        # none pending in the future (a manual run of a lapsed one) needs it.
        pending = _aware(j.next_run_at)
        if j.schedule and j.enabled and (pending is None
                                         or pending <= _now()):
            j.next_run_at = next_run(j.schedule)
        s.commit()
    return {"id": run_id, "status": status, "rows": rows,
            "message": message}


def run_now(user, project_id, job_id, snowflake=None):
    with db.session() as s:
        _need(s, user, project_id, "editor")
        j = s.get(db.BatchJob, job_id)
        if j is None or j.project_id != project_id:
            raise AccessDenied("That job is not in this project.")
    return run_job(job_id, triggered_by=_email(user), snowflake=snowflake)


def _claim(job_id, seen_next, schedule, now):
    """Advance a due job's next run time, only if no other runner has: the
    UPDATE matches the value read, so two overlapping runners cannot both
    claim (and run) the same job."""
    with db.session() as s:
        res = s.execute(update(db.BatchJob).where(
            db.BatchJob.id == job_id,
            db.BatchJob.next_run_at == seen_next).values(
            next_run_at=next_run(schedule, now)))
        s.commit()
        return res.rowcount == 1


def run_due(now=None):
    """Run every enabled job whose time has come; their summaries."""
    now = now or _now()
    with db.session() as s:
        due = [(j.id, j.next_run_at, j.schedule)
               for j in s.scalars(select(db.BatchJob).where(
                   db.BatchJob.enabled.is_(True),
                   db.BatchJob.next_run_at.is_not(None)))
               if _aware(j.next_run_at) <= now]
    return [run_job(job_id) for job_id, seen, schedule in due
            if _claim(job_id, seen, schedule, now)]
