"""Every request the model API answers, kept: who called, which version
answered, how many rows, how many were flagged by the input checks, how
long it took, and the rows themselves with what the model said. That is
what an auditor asks for ("what did the model say about this customer, and
when?"), and what live accuracy needs once the outcomes are known.

The rows go to the project's storage (encrypted in S3 on AWS) as one
Parquet file per request; the database keeps a line per request.
"""
import io
from datetime import timedelta

import pandas as pd
from sqlalchemy import select

from logic import audit, projects
from logic.projects import AccessDenied, _need
from utils import db, log, storage

_log = log.get("prediction_log")

# How long a project keeps its requests (days); None keeps them for good.
RETENTION = (30, 90, 180, 365, 730, None)
DEFAULT_DAYS = 365
MAX_WINDOW_FILES = 2_000      # requests read into one monitor check
MAX_WINDOW_ROWS = 500_000


def record(project_id, version, rows, out, *, source, caller, ms):
    """Keep one answered request; its id. `rows` as received, `out` as the
    model returned them (its ML_ columns are kept beside the inputs)."""
    log_id = db.new_id()
    at = db.now()
    key = (f"projects/{project_id}/predictions/{at:%Y/%m}/{log_id}"
           f".parquet")
    kept = rows.reset_index(drop=True).copy()
    model_cols = [c for c in out.columns if str(c).startswith("ML_")]
    for c in model_cols:
        kept[c] = out[c].to_numpy()
    storage.put(key, projects._parquet(kept))
    flagged = (int(out["ML_Warnings"].notna().sum())
               if "ML_Warnings" in out.columns else 0)
    with db.session() as s:
        s.add(db.PredictionLog(
            id=log_id, project_id=project_id, version_id=version.get("id"),
            model_name=version["name"], version=int(version["version"]),
            stage=version["stage"], source=source, caller=caller[:200],
            rows=len(kept), flagged=flagged, latency_ms=int(ms),
            object_key=key, at=at))
        s.commit()
    return log_id


def window(project_id, model_name, days, now=None):
    """The rows logged for `model_name` in the last `days` days, newest
    request first, as sent (the model's ML_ columns left out), with
    `_requested_at`. Service-level: a monitor runs as the system."""
    since = (now or db.now()) - timedelta(days=int(days))
    with db.session() as s:
        logs = s.scalars(select(db.PredictionLog).where(
            db.PredictionLog.project_id == project_id,
            db.PredictionLog.model_name == model_name,
            db.PredictionLog.at >= since).order_by(
            db.PredictionLog.at.desc()).limit(MAX_WINDOW_FILES)).all()
        found = [(r.object_key, r.at) for r in logs]
    frames, total = [], 0
    for key, at in found:
        try:
            got = pd.read_parquet(io.BytesIO(storage.get(key)))
        except storage.StorageError:
            continue                    # purged meanwhile
        got = got[[c for c in got.columns if not str(c).startswith("ML_")]]
        got["_requested_at"] = at
        frames.append(got)
        total += len(got)
        if total >= MAX_WINDOW_ROWS:
            break
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def retention(project_id):
    """Days this project keeps its requests (None: for good)."""
    with db.session() as s:
        p = s.get(db.Project, project_id)
        settings = (p.settings or {}) if p else {}
    if "prediction_log_days" not in settings:
        return DEFAULT_DAYS
    return settings["prediction_log_days"]


def set_retention(user, project_id, days):
    if days not in RETENTION:
        raise AccessDenied("Choose one of the offered periods.")
    with db.session() as s:
        p, _ = _need(s, user, project_id, "owner")
        p.settings = {**(p.settings or {}), "prediction_log_days": days}
        audit.record(s, user, "project.retention", project_id, "project",
                     project_id, prediction_log_days=days)
        s.commit()


def purge(now=None):
    """Delete each project's requests older than it keeps them; how many."""
    now = now or db.now()
    with db.session() as s:
        ids = [p for (p,) in s.execute(select(db.PredictionLog.project_id)
                                       .distinct())]
    removed = 0
    for project_id in ids:
        days = retention(project_id)
        if days is None:
            continue
        with db.session() as s:
            old = s.scalars(select(db.PredictionLog).where(
                db.PredictionLog.project_id == project_id,
                db.PredictionLog.at < now - timedelta(days=int(days)))).all()
            for r in old:
                storage.delete(r.object_key)
                s.delete(r)
            s.commit()
            removed += len(old)
    if removed:
        log.event(_log, "prediction_logs_purged", requests=removed)
    return removed


def recent(user, project_id, limit=50):
    """The latest requests, newest first."""
    with db.session() as s:
        _need(s, user, project_id, "viewer")
        rows = s.scalars(select(db.PredictionLog).where(
            db.PredictionLog.project_id == project_id).order_by(
            db.PredictionLog.at.desc()).limit(limit)).all()
        return [{"id": r.id, "model": r.model_name, "version": r.version,
                 "stage": r.stage, "source": r.source, "caller": r.caller,
                 "rows": r.rows, "flagged": r.flagged,
                 "latency_ms": r.latency_ms, "at": r.at} for r in rows]


def rows(user, project_id, log_id):
    """One request's rows with the model's answers, as a frame."""
    with db.session() as s:
        _need(s, user, project_id, "viewer")
        r = s.get(db.PredictionLog, log_id)
        if r is None or r.project_id != project_id:
            raise AccessDenied("That request is not in this project.")
        key = r.object_key
    return pd.read_parquet(io.BytesIO(storage.get(key)))
