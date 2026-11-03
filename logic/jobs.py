"""Background jobs: training that runs outside the web process.

The database is the source of truth for every job — status, progress,
result. A queue only says "there is work":

- `JOBS_QUEUE_URL` set (AWS): `submit` sends the job id to Amazon SQS and
  the worker service (`worker/trainer.py`) picks it up.
- Otherwise a small pool of threads in the app runs it (`JOBS_THREADS`, 1
  by default, so one long run cannot take every core from the page).
- `JOBS_MODE=none` queues without running (tests drive `run_job`).

A worker claims a job with one conditional update, so two workers can
never run the same job. While it runs it writes progress (at most once a
second), which doubles as a heartbeat and as the moment it learns of a
cancel request; training stops at the next model fit. A job whose
heartbeat goes quiet — its worker crashed or was replaced — is claimed
again and restarted from the beginning, up to MAX_ATTEMPTS times.
"""
import json
import logging
import os
import socket
import threading
import time
from datetime import timedelta

from sqlalchemy import or_, select, update

from logic import audit, projects
from logic.deployment import _aware, _now
from logic.projects import AccessDenied, _email, _need
from utils import db, log

_log = log.get("jobs")

MAX_ATTEMPTS = 3
STALE = timedelta(minutes=5)       # no heartbeat for this long: interrupted
PROGRESS_EVERY = 1.0               # seconds between progress writes
ACTIVE = ("queued", "running")


class Cancelled(BaseException):
    """Raised from the progress callback to stop training. A BaseException,
    like KeyboardInterrupt: training catches Exception around each model
    fit (one failing model must not sink the run), and a cancel has to get
    through that."""


def worker_id():
    return f"{socket.gethostname()}:{os.getpid()}"


# ── submitting and watching ─────────────────────────────────────────────
def submit_training(user, project_id, config, steps, dataset_id):
    """Queue a training run on the project's stored dataset; its id."""
    if not dataset_id:
        raise AccessDenied("Save the data to the project first.")
    with db.session() as s:
        _need(s, user, project_id, "editor")
        j = db.Job(project_id=project_id, kind="train", status="queued",
                   payload={"config": config, "steps": list(steps or []),
                            "dataset_id": dataset_id,
                            "role": user.get("role", "viewer")},
                   created_by=_email(user))
        s.add(j)
        s.flush()
        audit.record(s, user, "job.submit", project_id, "job", j.id, kind="train", models=len(config.get("models") or []))
        s.commit()
        job_id = j.id
    log.event(_log, "job_submitted", kind="train",
              models=len(config.get("models") or []))
    dispatch(job_id)
    return job_id


def _as_dict(j):
    return {"id": j.id, "kind": j.kind, "status": j.status,
            "progress": j.progress, "progress_text": j.progress_text,
            "message": j.message, "result": dict(j.result or {}),
            "cancel_requested": j.cancel_requested, "attempts": j.attempts,
            "created_by": j.created_by, "created_at": j.created_at,
            "started_at": j.started_at, "finished_at": j.finished_at,
            "task": (j.payload or {}).get("config", {}).get("task"),
            "target": (j.payload or {}).get("config", {}).get("target")}


def get(user, project_id, job_id):
    with db.session() as s:
        _need(s, user, project_id, "viewer")
        j = s.get(db.Job, job_id)
        if j is None or j.project_id != project_id:
            raise AccessDenied("That job is not in this project.")
        return _as_dict(j)


def list_jobs(user, project_id, limit=20):
    with db.session() as s:
        _need(s, user, project_id, "viewer")
        rows = s.scalars(select(db.Job).where(
            db.Job.project_id == project_id)
            .order_by(db.Job.created_at.desc()).limit(limit))
        return [_as_dict(j) for j in rows]


def request_cancel(user, project_id, job_id):
    """A queued job stops at once; a running one at its next model fit."""
    with db.session() as s:
        _need(s, user, project_id, "editor")
        j = s.get(db.Job, job_id)
        if j is None or j.project_id != project_id:
            raise AccessDenied("That job is not in this project.")
        if j.status == "queued":
            j.status, j.finished_at = "cancelled", _now()
            j.message = f"Cancelled by {_email(user)} before it started."
        elif j.status == "running":
            j.cancel_requested = True
        audit.record(s, user, "job.cancel", project_id, "job", job_id, was=j.status)
        s.commit()


def retry(user, project_id, job_id):
    """The same work as a new job."""
    with db.session() as s:
        _need(s, user, project_id, "editor")
        j = s.get(db.Job, job_id)
        if j is None or j.project_id != project_id:
            raise AccessDenied("That job is not in this project.")
        if j.status in ACTIVE:
            raise AccessDenied("That job is still going.")
        payload = {**j.payload, "role": user.get("role", "viewer")}
    return submit_training(user, project_id, payload["config"],
                           payload["steps"], payload["dataset_id"])


# ── the worker's side ───────────────────────────────────────────────────
def claim(job_id, worker, now=None):
    """Take a queued job, or one whose worker went quiet; False when
    someone else has it, it is finished, or it has run out of attempts."""
    now = now or _now()
    with db.session() as s:
        j = s.get(db.Job, job_id)
        if j is None or j.status not in ACTIVE:
            return False
        stale = (j.status == "running" and (
            j.heartbeat_at is None or _aware(j.heartbeat_at) < now - STALE))
        if j.status == "running" and not stale:
            return False
        if j.attempts >= MAX_ATTEMPTS:
            j.status, j.finished_at = "failed", now
            j.message = (f"Interrupted {j.attempts} times; not tried "
                         f"again. Retry it when the service is steady.")
            s.commit()
            return False
        res = s.execute(update(db.Job).where(
            db.Job.id == job_id, db.Job.attempts == j.attempts,
            or_(db.Job.status == "queued", db.Job.heartbeat_at.is_(None),
                db.Job.heartbeat_at == j.heartbeat_at)).values(
            status="running", attempts=j.attempts + 1, worker=worker,
            started_at=now, heartbeat_at=now, progress=0.0,
            progress_text="Starting"))
        s.commit()
        return res.rowcount == 1


def report(job_id, fraction, text):
    """Progress and heartbeat; True when a cancel has been asked for."""
    with db.session() as s:
        s.execute(update(db.Job).where(db.Job.id == job_id).values(
            progress=float(min(max(fraction, 0.0), 1.0)),
            progress_text=(text or "")[:300], heartbeat_at=_now()))
        s.commit()
        return bool(s.scalar(select(db.Job.cancel_requested).where(
            db.Job.id == job_id)))


def finish(job_id, status, message="", result=None):
    with db.session() as s:
        j = s.get(db.Job, job_id)
        j.status, j.message = status, message
        j.result = result or {}
        j.finished_at = _now()
        if status == "succeeded":
            j.progress, j.progress_text = 1.0, "Done"
        audit.record(s, "system:worker", "job.finish", j.project_id, "job", job_id, status=status)
        s.commit()


def state(job_id, now=None):
    """"done", "busy" (running, heartbeat fresh) or "open" (queued, or
    running with a quiet heartbeat): what a worker should do with a
    message for it."""
    now = now or _now()
    with db.session() as s:
        j = s.get(db.Job, job_id)
        if j is None or j.status not in ACTIVE:
            return "done"
        if j.status == "running" and j.heartbeat_at is not None \
                and _aware(j.heartbeat_at) >= now - STALE:
            return "busy"
        return "open"


def stale_jobs(now=None):
    """Running jobs whose worker went quiet: to be claimed again."""
    now = now or _now()
    with db.session() as s:
        rows = s.scalars(select(db.Job).where(db.Job.status == "running"))
        return [j.id for j in rows if j.heartbeat_at is None
                or _aware(j.heartbeat_at) < now - STALE]


def run_job(job_id, worker=None):
    """Claim and run one job to the end. Returns its final status, or
    None when it was not ours to run."""
    if not claim(job_id, worker or worker_id()):
        return None
    with db.session() as s:
        j = s.get(db.Job, job_id)
        project_id, payload, creator = j.project_id, dict(j.payload), \
            j.created_by
    user = {"email": creator, "role": payload.get("role", "viewer"),
            "sub": None}
    last = [0.0]

    def progress(fraction, text):
        if time.monotonic() - last[0] < PROGRESS_EVERY and fraction < 1:
            return
        last[0] = time.monotonic()
        if report(job_id, fraction, text):
            raise Cancelled()

    try:
        result = _train(user, project_id, payload, progress)
        finish(job_id, "succeeded", result["message"], result)
        status = "succeeded"
    except Cancelled:
        finish(job_id, "cancelled", "Cancelled while training.")
        status = "cancelled"
    except AccessDenied as e:
        finish(job_id, "failed", str(e))
        status = "failed"
    except Exception as e:
        log.event(_log, "job_failed", level=logging.ERROR, exc_info=True,
                  kind="train")
        finish(job_id, "failed",
               f"{type(e).__name__}: training could not finish.")
        status = "failed"
    log.event(_log, "job_finished", kind="train", status=status)
    return status


def _train(user, project_id, payload, progress):
    from logic import ml, ml_eval, ml_prep
    df, _ = projects.load_dataset(user, project_id, payload["dataset_id"])
    steps = payload.get("steps") or []
    progress(0.01, "Preparing the data")
    if steps:
        df = ml_prep.apply_steps(df, steps)
    run = ml.train(df, payload["config"], progress)
    run["prep_steps"], run["importance"] = list(steps), {}
    if run["task"] != "clustering" and run.get("best"):
        res = ml.result(run, run["best"])
        if ml_eval.importance_seconds(run, res) <= 60:
            progress(0.99, "Measuring what drives the best model")
            run["importance"][run["best"]] = \
                ml_eval.permutation_importance(run, res, df)
    run["frame"] = df
    rid = projects.save_run(user, project_id, run, payload["dataset_id"])
    best = ml.result(run, run["best"]) if run.get("best") else None
    score = (best or {}).get("test", {}).get(run.get("metric")) \
        if best else None
    label = best["label"] if best else "no model"
    return {"run_id": rid, "best": run.get("best"),
            "metric": run.get("metric"),
            "score": None if score is None else float(score),
            "message": f"Trained on {len(df):,} rows; best: {label}."}


# ── dispatch ────────────────────────────────────────────────────────────
_POOL = {"sem": None, "lock": threading.Lock()}


def _thread_slots():
    with _POOL["lock"]:
        if _POOL["sem"] is None:
            _POOL["sem"] = threading.BoundedSemaphore(
                max(1, int(os.environ.get("JOBS_THREADS", "1"))))
        return _POOL["sem"]


def dispatch(job_id):
    """Tell something to run the job."""
    queue_url = os.environ.get("JOBS_QUEUE_URL")
    mode = os.environ.get("JOBS_MODE", "queue" if queue_url else "thread")
    if mode == "none":
        return
    if mode == "queue":
        import boto3
        boto3.client("sqs").send_message(
            QueueUrl=queue_url, MessageBody=json.dumps({"job_id": job_id}))
        return

    def work():
        with _thread_slots():
            run_job(job_id)

    threading.Thread(target=work, name=f"job-{job_id[:8]}",
                     daemon=True).start()
