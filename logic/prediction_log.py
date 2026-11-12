"""Every request the model API answers, kept: who called, which version
answered, how many rows, how many were flagged by the input checks, how
long it took, and the rows themselves with what the model said. That is
what an auditor asks for ("what did the model say about this customer, and
when?"), and what live accuracy needs once the outcomes are known.

The rows go to the project's storage (encrypted in S3 on AWS) as one
Parquet file per request; the database keeps a line per request.
"""
import io

import pandas as pd
from sqlalchemy import select

from logic import projects
from logic.projects import AccessDenied, _need
from utils import db, storage


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
