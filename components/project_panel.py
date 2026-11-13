"""Projects in the app: open or create one, save work to it automatically,
share it, and browse its runs and registered models.

Session state (`proj_*`, not `ml_*`: a project stays open across new files):
  proj               {"id", "name", "role"} of the open project, or absent
  proj_dataset_id    the stored dataset the session's file came from
  proj_sig           what was last saved of the steps and the choices

With a project open and an editor's rights, these save themselves: a newly
loaded file (as the project's dataset), the preparation steps, the configure
page's choices, and every trained run. Without one, work is unsaved, and
the chip at the top offers to save it as a new project.
"""
import functools
import json
import logging

import pandas as pd
import streamlit as st

from logic import ml_eval, ml_upload, projects
from logic.projects import AccessDenied
from utils import auth, db, guard, log, scroll

_log = log.get("project_panel")

def _ss():
    return st.session_state


@functools.lru_cache(maxsize=1)
def _ready():
    """The database migrated, once per process; False (logged) when it
    cannot be reached — the app then works unsaved."""
    try:
        db.migrate()
        return True
    except Exception:
        log.event(_log, "projects_unavailable", level=logging.ERROR,
                  exc_info=True)
        return False


def available():
    return _ready()


def signed_in():
    """Once per session: record the user."""
    ss = _ss()
    if available() and not ss.get("proj_user_seen"):
        try:
            projects.touch_user(auth.user())
        except Exception:
            log.event(_log, "touch_user_failed", level=logging.WARNING,
                      exc_info=True)
        ss["proj_user_seen"] = True


def current():
    return _ss().get("proj")


def can_edit():
    p = current()
    return bool(p) and p["role"] in ("editor", "owner") and auth.can("edit")


def close():
    # The state only: the project widgets keep their own keys.
    for k in ("proj", "proj_dataset_id", "proj_sig"):
        _ss().pop(k, None)


def _sig():
    ss = _ss()
    return json.dumps({"steps": ss.get("ml_steps") or [],
                       "config": ss.get("ml_saved") or {}},
                      default=str, sort_keys=True)


# ── hooks the app calls ─────────────────────────────────────────────────
def on_new_dataset(df, name, source):
    """A file was just loaded. In an open project an editor saves it as
    the project's dataset; anyone else carries on unsaved."""
    p = current()
    if not p:
        return
    if not can_edit():
        st.toast(f"Opened unsaved: you cannot add data to {p['name']}.")
        close()
        return
    try:
        did = projects.save_dataset(auth.user(), p["id"], df, name, source)
    except Exception as e:
        _fail("Saving the dataset", e)
        return
    _ss()["proj_dataset_id"] = did
    _ss()["proj_sig"] = _sig()
    log.event(_log, "project_dataset_saved", rows=len(df),
              columns=df.shape[1])


def autosave():
    """Save what changed since the last run: the steps and choices, and a
    run not yet stored. Called at the end of every page."""
    if not can_edit() or "ml_df" not in _ss():
        return
    ss, p, user = _ss(), current(), auth.user()
    try:
        sig = _sig()
        if ss.get("proj_sig") != sig:
            projects.save_workspace(user, p["id"], ss.get("ml_steps") or [],
                                    ss.get("ml_saved") or {})
            ss["proj_sig"] = sig
        run = ss.get("ml_run")
        if run and not run.get("_project_run_id"):
            payload = dict(run)
            if payload.get("frame") is None:
                # The prepared file it was trained on: a run reopened later
                # shows its own data, not whatever is loaded then.
                payload["frame"] = ss["ml_df"]
            with log.timer() as t:
                rid = projects.save_run(user, p["id"], payload,
                                        ss.get("proj_dataset_id"))
            run["_project_run_id"] = rid
            log.event(_log, "project_run_saved", task=run["task"], ms=t.ms)
    except Exception as e:
        _fail("Saving to the project", e)


def _fail(what, e):
    if isinstance(e, AccessDenied):
        st.warning(str(e))
        return
    ref = log.incident_id()
    log.event(_log, "project_save_failed", level=logging.ERROR,
              exc_info=True, ref=ref)
    st.warning(f"{what} failed; your work here is unaffected. Reference "
               f"{ref}.")


def open_project(pid, start_ml):
    """Load a project's dataset, steps and choices into the session."""
    user = auth.user()
    p = projects.get_project(user, pid)
    ws = projects.workspace(user, pid)
    close()
    _ss()["proj"] = {"id": p["id"], "name": p["name"], "role": p["role"]}
    log.event(_log, "project_opened", role=p["role"])
    if not ws["dataset_id"]:
        _ss()["page"] = "start"        # nothing loaded yet: the data page
        return
    df, name = projects.load_dataset(user, pid, ws["dataset_id"])
    start_ml(df, "project", name=name, save=False)
    _ss()["proj_dataset_id"] = ws["dataset_id"]
    if ws["steps"]:
        from components import ml_eda
        try:
            ml_eda.set_steps(ws["steps"])
        except Exception:
            st.warning("The saved preparation steps no longer apply to the "
                       "data; it opened without them.")
            ml_eda.set_steps([])
    _ss()["ml_saved"] = ws["config"]
    _ss()["proj_sig"] = _sig()


# ── the chip next to the account menu ───────────────────────────────────
def chip():
    if not available():
        return
    p = current()
    with st.popover(p["name"] if p else "Not saved", width="stretch",
                    icon=":material/folder:" if p else
                    ":material/cloud_off:"):
        if p:
            st.markdown(f"**{p['name']}**  \nYour role: {p['role']}")
            st.caption("Saved automatically: the dataset, the preparation "
                       "steps, the configure choices and every run."
                       if can_edit() else "Read-only for you: changes here "
                       "are not saved to the project.")
            if st.button("Close project", width="stretch", key="proj_close"):
                close()
                st.rerun()
            return
        st.caption("This work is not saved. Save it as a project to keep "
                   "it, come back to it, and share it.")
        if not auth.can("edit"):
            st.caption(auth.why_not("edit"))
            return
        name = st.text_input("Project name", key="proj_new_name")
        if st.button("Save as project", type="primary", width="stretch",
                     key="proj_save_new", disabled=not name.strip()):
            guard.action("Saving", "The project could not be created.",
                         _save_as_new, name)


def _save_as_new(name):
    user = auth.user()
    p = projects.create_project(user, name)
    _ss()["proj"] = p
    ss = _ss()
    if "ml_source" in ss:
        did = projects.save_dataset(user, p["id"], ss["ml_source"],
                                    ss.get("ml_name") or "data", "upload")
        ss["proj_dataset_id"] = did
        projects.save_workspace(user, p["id"], ss.get("ml_steps") or [],
                                ss.get("ml_saved") or {})
        ss["proj_sig"] = _sig()
    autosave()
    log.event(_log, "project_created", from_session="ml_source" in ss)
    st.rerun()


# ── the start page ──────────────────────────────────────────────────────
def landing(start_ml):
    """Projects to open, and a new one, above the upload."""
    if not available():
        st.caption("Projects are unavailable right now (the database "
                   "cannot be reached); work here will not be saved.")
        return
    user = auth.user()
    mine = projects.list_projects(user)
    with st.container(border=True, key="card_projects"):
        st.markdown("#### Projects")
        if current():
            st.caption(f"Open: **{current()['name']}** — a file loaded "
                       f"below becomes its dataset.")
        if mine:
            for p in mine[:12]:
                c1, c2, c3 = st.columns([4, 2, 1.2],
                                        vertical_alignment="center")
                c1.markdown(f"**{p['name']}**"
                            + (f"  \n{p['description']}"
                               if p["description"] else ""))
                c2.caption(f"{p['role']} · updated "
                           f"{pd.Timestamp(p['updated_at']):%d %b %Y}")
                if c3.button("Open", key=f"proj_open_{p['id']}",
                             width="stretch"):
                    guard.action("Opening", "The project could not be "
                                 "opened.", _open_and_go, p["id"], start_ml)
        else:
            st.caption("No projects yet. Create one to keep your work, or "
                       "just load data below and save it later.")
        if auth.can("edit"):
            c1, c2 = st.columns([4, 1.2], vertical_alignment="bottom")
            name = c1.text_input("New project", key="proj_landing_name",
                                 placeholder="e.g. Churn model Q4")
            if c2.button("Create", key="proj_landing_create",
                         width="stretch", disabled=not name.strip()):
                guard.action("Creating", "The project could not be "
                             "created.", _create_and_stay, name)


def _open_and_go(pid, start_ml):
    open_project(pid, start_ml)
    scroll.request()
    st.rerun()


def _create_and_stay(name):
    _ss()["proj"] = projects.create_project(auth.user(), name)
    st.rerun()


# ── the results page ────────────────────────────────────────────────────
def save_threshold(run, res):
    """After a model's cut-off changes: rewrite the stored run, so the
    project keeps it. Nothing to do when the run is not stored (yet)."""
    rid, p = run.get("_project_run_id"), current()
    if not (available() and p and rid and can_edit()):
        return
    try:
        payload = dict(run)
        if payload.get("frame") is None:
            payload["frame"] = _ss()["ml_df"]
        projects.set_threshold(auth.user(), p["id"], rid, payload,
                               res["key"], res.get("threshold"))
    except Exception as e:
        _fail("Saving the cut-off", e)


def register(run, res):
    """Register the model on screen as a new version in the project."""
    p = current()
    if not available():
        return
    with st.container(border=True, key="card_registry"):
        st.markdown("#### Model registry")
        if not p:
            st.caption("Save this work as a project (the menu at the top) to "
                       "register models.")
            return
        if not can_edit():
            st.caption("Registering models needs an editor of the project.")
            return
        if run.get("external") and res["key"] != ml_upload.KEY:
            st.caption("Pick the uploaded model above to register it; the "
                       "baseline is only a yardstick.")
            return
        if run.get("external") and not run["upload"].get("file"):
            st.caption("This evaluation was made before uploaded models "
                       "could be registered: upload the file again on Use "
                       "own model to register it.")
            return
        if run.get("external"):
            st.caption("Registered as the file you uploaded, unchanged, "
                       "with the column mapping used here. It then goes "
                       "through stages, approvals, the model API, batch "
                       "jobs and monitoring like a model trained here; "
                       "monitoring compares new data with the file it was "
                       "evaluated on, and does not retrain it.")
        autosave()                       # the run must be stored first
        rid = run.get("_project_run_id")
        c1, c2 = st.columns([3, 1.2], vertical_alignment="bottom")
        name = c1.text_input("Register as", key="proj_reg_name",
                             value=(run.get("target") or run["task"])
                             .replace(" ", "_").lower() + "_model")
        if c2.button(f"Register {res['label']}", type="primary",
                     width="stretch", key=f"proj_reg_{res['key']}",
                     disabled=not (rid and name.strip())):
            score = (res.get("test") or {}).get(run.get("metric"))
            got = projects.register_model(
                auth.user(), p["id"], rid, ml_eval.model_bytes(run, res),
                name=name, model_key=res["key"], task=run["task"],
                target=run.get("target"), metric=run.get("metric"),
                score=None if score is None else float(score))
            log.event(_log, "model_registered", version=got["version"])
            st.success(f"Registered {got['name']} version {got['version']}.")


# ── the project page ────────────────────────────────────────────────────
def _fmt_time(t):
    return f"{pd.Timestamp(t):%d %b %Y %H:%M}"


def _run_label(r):
    return f"#{r['number']} · " + (f"{r['name']} · " if r["name"] else "") \
        + f"{_fmt_time(r['created_at'])} · {r['task']} · " \
        f"{r['target'] or ''} · {r['best_model'] or ''}"


def _runs(p, user):
    rows = projects.list_runs(user, p["id"])
    st.markdown("#### Runs")
    if not rows:
        st.caption("No runs yet: train on Configure & train and each run "
                   "is saved here.")
        return
    # Numbered oldest first, so a run keeps its number as others are added.
    for i, r in enumerate(rows):
        r["number"] = len(rows) - i
    from components import shell
    shell.table(
        [("#", "number"), ("Name", "name"), ("When", "when"),
         ("Task", "task"), ("Target", "target"), ("Best model", "best"),
         ("Metric", "metric"), ("Score", "score", "num"), ("By", "by"),
         ("Note", "note")],
        [{"number": r["number"], "name": r["name"],
          "when": _fmt_time(r["created_at"]),
          "by": r["created_by"],
          "task": r["task"] + (" (own model)" if r["external"] else ""),
          "target": r["target"], "best": r["best_model"],
          "metric": r["metric"],
          "score": None if r["score"] is None else f"{r['score']:.3f}",
          "note": (r["note"][:80] + "…") if len(r["note"]) > 80
          else r["note"]} for r in rows])
    labels = {r["id"]: _run_label(r) for r in rows}
    by_id = {r["id"]: r for r in rows}
    pick = st.selectbox("Run", list(labels), format_func=labels.get,
                        key="proj_run_pick")
    c1, c2, _ = st.columns([1, 1, 2.5])
    if c1.button("Open run", width="stretch", key="proj_run_open"):
        run = projects.load_run(user, p["id"], pick)
        run["_project_run_id"] = pick
        _ss()["ml_run"] = run
        _ss()["page"] = "ml_results"
        scroll.request()
        st.rerun()
    if can_edit() and c2.button(
            "Clone settings", width="stretch", key="proj_run_clone",
            help="Opens Configure & train with this run's data, preparation "
                 "and choices, to change one thing and train again."):
        guard.action("Cloning", "The run's settings could not be loaded.",
                     _clone, user, p, by_id[pick])
    if can_edit():
        with st.expander("Name and note"):
            r = by_id[pick]
            name = st.text_input("Name", value=r["name"],
                                 max_chars=projects.RUN_NAME_MAX,
                                 key=f"proj_run_name_{pick}",
                                 placeholder="Baseline with tenure bands")
            note = st.text_area("Note", value=r["note"], height=90,
                                max_chars=projects.RUN_NOTE_MAX,
                                key=f"proj_run_note_{pick}",
                                placeholder="What this run tried, and what "
                                            "it showed")
            if st.button("Save", key=f"proj_run_save_{pick}",
                         disabled=(name.strip(), note.strip()) ==
                         (r["name"], r["note"])):
                guard.action("Saving", "The note could not be saved.",
                             _describe, user, p, pick, name, note)
    if len(rows) >= 2:
        with st.expander("Compare two runs"):
            _compare(p, user, rows, labels)


def _describe(user, p, run_id, name, note):
    projects.describe_run(user, p["id"], run_id, name, note)
    log.event(_log, "run_described")
    st.rerun()


def _clone(user, p, r):
    """Load the run's dataset and preparation, put its configure choices
    back, and open Configure & train."""
    run = projects.load_run(user, p["id"], r["id"])
    widgets = (run.get("config") or {}).get("widgets")
    if run.get("external") or not widgets:
        st.warning("This run does not keep its configure choices (it was "
                   "trained before runs kept them, or it evaluated an "
                   "uploaded model). Open it to read its settings instead.")
        return
    ss = _ss()
    if r["dataset_id"] and r["dataset_id"] != ss.get("proj_dataset_id"):
        df, name = projects.load_dataset(user, p["id"], r["dataset_id"])
        for k in [k for k in ss.keys() if isinstance(k, str)
                  and k.startswith("ml_")]:
            del ss[k]
        ss["ml_source"], ss["ml_df"], ss["ml_name"] = df, df, name
        ss["ml_steps"], ss["ml_version"] = [], 0
        ss["proj_dataset_id"] = r["dataset_id"]
    elif "ml_source" not in ss:
        raise AccessDenied("The run's dataset is not stored in the project.")
    from components import ml_eda
    ml_eda.set_steps(run.get("prep_steps") or [])
    for k in [k for k in ss.keys() if isinstance(k, str)
              and k.startswith("ml_w_")]:
        del ss[k]
    ss["ml_saved"] = dict(widgets)
    ss["page"] = "ml_config"
    log.event(_log, "run_cloned")
    scroll.request()
    st.rerun()


def _compare(p, user, rows, labels):
    ids = list(labels)
    c1, c2 = st.columns(2)
    a = c1.selectbox("Run A", ids, index=1, format_func=labels.get,
                     key="proj_cmp_a")
    b = c2.selectbox("Run B", ids, index=0, format_func=labels.get,
                     key="proj_cmp_b")
    if a == b:
        st.caption("Pick two different runs.")
        return
    from components import shell
    from logic import run_compare
    ra = projects.load_run(user, p["id"], a)
    rb = projects.load_run(user, p["id"], b)
    got = run_compare.compare(ra, rb)
    shell.table([("", "k"), ("Run A", "a"), ("Run B", "b")],
                [{"k": k, "a": x, "b": y} for k, x, y in got["summary"]])
    f = got["features"]
    st.markdown("**Features**")
    st.caption(f"{f['shared']} shared"
               + (f" · only in A: {', '.join(f['only_a'])}"
                  if f["only_a"] else "")
               + (f" · only in B: {', '.join(f['only_b'])}"
                  if f["only_b"] else "")
               + ("" if f["only_a"] or f["only_b"] else " · the same in both"))
    s = got["steps"]
    if s["only_a"] or s["only_b"]:
        st.markdown("**Preparation steps**")
        for d in s["only_a"]:
            st.caption(f"Only in A: {d}")
        for d in s["only_b"]:
            st.caption(f"Only in B: {d}")
    st.markdown("**Settings that differ**")
    if got["settings"]:
        shell.table([("Setting", "k"), ("Run A", "a"), ("Run B", "b")],
                    [{"k": k, "a": x, "b": y}
                     for k, x, y in got["settings"]])
    else:
        st.caption("None: the same choices.")
    if got["models"]:
        st.markdown(f"**Each model's {got['metric']} on its test rows**")

        def num(v):
            return None if v is None else f"{v:.3f}"
        shell.table([("Model", "m"), ("Run A", "a", "num"),
                     ("Run B", "b", "num")],
                    [{"m": m, "a": num(x), "b": num(y)}
                     for m, x, y in got["models"]])
        st.caption("Each run is scored on its own test rows: when the data "
                   "or the split differs, so do the rows.")


def _models(p, user):
    rows = projects.list_models(user, p["id"])
    st.markdown("#### Registered models")
    if not rows:
        st.caption("None yet: register a model from the results page.")
        return
    from components import shell
    shell.table(
        [("Model", "model"), ("Version", "version"), ("Stage", "stage"),
         ("Algorithm", "algorithm"), ("Target", "target"),
         ("Metric", "metric"), ("Score", "score", "num"), ("By", "by"),
         ("Registered", "when")],
        [{"model": m["name"], "version": f"v{m['version']}",
          "stage": shell.status("Not staged" if m["stage"] == "none"
                                else m["stage"],
                                shell.TONES.get(m["stage"], "idle")),
          "algorithm": m["model_key"], "target": m["target"],
          "metric": m["metric"],
          "score": None if m["score"] is None else f"{m['score']:.3f}",
          "by": m["created_by"], "when": _fmt_time(m["created_at"])}
         for m in rows])
    labels = {m["id"]: f"{m['name']} v{m['version']} ({m['model_key']})"
              for m in rows}
    c1, c2 = st.columns([4, 1.2], vertical_alignment="bottom")
    pick = c1.selectbox("Download a version", list(labels),
                        format_func=labels.get, key="proj_model_pick")
    chosen = next(m for m in rows if m["id"] == pick)
    c2.download_button(
        "Download", width="stretch", key="proj_model_dl",
        data=lambda: projects.model_file(user, p["id"], pick),
        file_name=f"{chosen['name']}_v{chosen['version']}.joblib",
        mime="application/octet-stream")
    st.caption("Each version is this app's model file: it loads in "
               "**Use own model** and in Python with joblib.")


def _members(p, user):
    rows = projects.members(user, p["id"])
    st.markdown("#### Members")
    owner = p["role"] == "owner"
    for m in rows:
        c1, c2, c3 = st.columns([4, 1.5, 1], vertical_alignment="center")
        from components import shell
        c1.markdown(m["email"])
        c2.markdown(shell.status(m["role"]), unsafe_allow_html=True)
        if owner and c3.button("Remove", key=f"proj_rm_{m['email']}",
                               width="stretch"):
            guard.action("Removing", "The member could not be removed.",
                         _unshare, p, m["email"])
    if not owner:
        st.caption("Only owners change who has access.")
        return
    c1, c2, c3 = st.columns([4, 1.5, 1], vertical_alignment="bottom")
    email = c1.text_input("Share with (email)", key="proj_share_email")
    role = c2.selectbox("As", ["viewer", "editor", "owner"],
                        key="proj_share_role")
    if c3.button("Share", type="primary", width="stretch",
                 key="proj_share_go", disabled="@" not in email):
        guard.action("Sharing", "The project could not be shared.",
                     _share, p, email, role)
    st.caption("Viewers see the data, runs and models; editors also change "
               "and train; owners also share, rename and delete. A "
               "colleague can be added before their first sign-in.")


def _share(p, email, role):
    projects.share(auth.user(), p["id"], email, role)
    log.event(_log, "project_shared", role=role)
    st.rerun()


def _unshare(p, email):
    projects.unshare(auth.user(), p["id"], email)
    log.event(_log, "project_unshared")
    st.rerun()


def _settings(p, user):
    if p["role"] != "owner":
        return
    with st.expander("Project settings"):
        from logic import prediction_log
        keep = prediction_log.retention(p["id"])
        options = list(prediction_log.RETENTION)
        label = {d: "For good" if d is None else f"{d} days"
                 for d in options}
        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
        days = c1.selectbox("Keep the model API's requests for", options,
                            index=options.index(keep), format_func=label.get,
                            key="proj_log_days",
                            help="Older requests, their rows included, are "
                                 "deleted by the scheduled runner.")
        if c2.button("Save", key="proj_log_days_go", disabled=days == keep):
            prediction_log.set_retention(user, p["id"], days)
            st.rerun()
        st.divider()
        name = st.text_input("Name", value=p["name"], key="proj_rename")
        if st.button("Rename", key="proj_rename_go",
                     disabled=name.strip() in ("", p["name"])):
            projects.rename(user, p["id"], name)
            _ss()["proj"]["name"] = name.strip()
            st.rerun()
        st.divider()
        if st.button("Archive project", key="proj_archive",
                     help="Hidden from the project list; data is kept."):
            projects.archive(user, p["id"])
            close()
            _ss()["page"] = None
            st.rerun()
        confirm = st.text_input(f"Type {p['name']} to delete it for good",
                                key="proj_delete_confirm")
        if st.button("Delete project and its files", key="proj_delete",
                     disabled=confirm != p["name"]):
            projects.delete(user, p["id"])
            log.event(_log, "project_deleted")
            close()
            _ss()["page"] = None
            st.rerun()


def render():
    p = current()
    if not p:
        st.info("No project is open.")
        return
    user = auth.user()
    try:
        p = {**p, **projects.get_project(user, p["id"])}
    except AccessDenied as e:
        st.warning(str(e))
        close()
        return
    from components import (deploy_panel, governance_panel, jobs_panel,
                            monitor_panel, shell)
    # One section at a time, chosen in the sidebar: each has its own
    # queries, and the page used to run them all.
    which = shell.section()
    title = next(label for key, label, _ in shell.SECTIONS if key == which)
    st.markdown(f"## {title}")
    st.caption(f"{p['name']} · your role: {p['role']} · owner {p['owner']}")

    def card(name, fn, *args):
        with st.container(border=True,
                          key="card_" + name.lower().replace(" ", "_")):
            guard.section(name, fn, *args)

    if which == "runs":
        card("Training jobs", jobs_panel.project_section, p, user)
        card("Runs", _runs, p, user)
    elif which == "models":
        card("Registered models", _models, p, user)
        card("Stages and approvals", deploy_panel.stages, p, user)
        card("Fairness", governance_panel.fairness_section, p, user)
        card("Model API", deploy_panel.api_keys, p, user)
        card("Recent predictions", deploy_panel.prediction_logs, p, user)
    elif which == "operations":
        card("Monitoring", monitor_panel.render, p, user)
        card("Batch scoring", deploy_panel.batch, p, user)
    else:
        card("Members", _members, p, user)
        guard.section("Activity", governance_panel.activity_section, p, user)
        guard.section("Settings", _settings, p, user)
