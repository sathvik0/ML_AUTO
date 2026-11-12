"""The model API: predictions from registered models, for other systems.

Runs as its own service (`uvicorn api.main:app`) behind the same load
balancer, on /api/*. That path skips the browser sign-in; callers
authenticate with a project API key instead:

    Authorization: Bearer mlk_...        (or  X-API-Key: mlk_...)

A key reaches its own project's models only. Predictions come from the
model's production version unless `stage=staging` or `version=N` says
otherwise. Interactive docs: /api/docs.
"""
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select

from logic import deployment, prediction_log, serving
from utils import db, log

log.setup()
_log = log.get("api")


@asynccontextmanager
async def lifespan(app):
    db.migrate()
    yield


app = FastAPI(title="ML Automation model API", version="1",
              docs_url="/api/docs", openapi_url="/api/openapi.json",
              redoc_url=None, lifespan=lifespan)


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(
        ..., description="One object per row, keyed by column name — the "
                         "columns the model was trained on.")


class PredictResponse(BaseModel):
    model: str
    version: int
    stage: str
    task: str
    rows: int
    predictions: list[Any]
    outputs: list[dict[str, Any]] = Field(
        description="Every column the model adds, per row: the prediction; "
                    "for classifiers, each class's probability; "
                    "ML_Warnings when an input is outside what the model "
                    "was trained on (null otherwise); and with explain, "
                    "ML_Reasons.")
    request_id: str | None = Field(
        None, description="The id this request is kept under in the "
                          "project's prediction log (null if it could not "
                          "be kept).")


def api_caller(authorization: str | None = Header(None),
               x_api_key: str | None = Header(None)):
    """{"project_id", "name"} of the key the request carries."""
    key = x_api_key
    if not key and authorization and authorization.lower().startswith(
            "bearer "):
        key = authorization[7:].strip()
    got = deployment.caller(key)
    if got is None:
        log.event(_log, "api_unauthorised", level=logging.WARNING)
        raise HTTPException(401, "A valid API key is required.",
                            headers={"WWW-Authenticate": "Bearer"})
    return got


def project_for_key(got: dict = Depends(api_caller)):
    return got["project_id"]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/models")
def models(project_id: str = Depends(project_for_key)):
    """The project's registered models, each version with its stage."""
    with db.session() as s:
        rows = s.scalars(select(db.ModelVersion).where(
            db.ModelVersion.project_id == project_id).order_by(
            db.ModelVersion.name, db.ModelVersion.version.desc())).all()
        out = {}
        for v in rows:
            out.setdefault(v.name, []).append(
                {"version": v.version, "stage": v.stage, "task": v.task,
                 "target": v.target, "metric": v.metric, "score": v.score})
    return {"models": [{"name": n, "versions": vs} for n, vs in out.items()]}


def _json(frame):
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


@app.post("/api/v1/models/{name}/predict", response_model=PredictResponse)
def predict(name: str, body: PredictRequest,
            stage: str = Query("production", pattern="^(production|staging)$"),
            version: int | None = Query(None, ge=1),
            explain: int = Query(0, ge=0, le=serving.MAX_REASONS,
                                 description="Add ML_Reasons: the top N "
                                             "inputs behind each "
                                             "prediction (up to "
                                             f"{serving.MAX_EXPLAIN_ROWS:,} "
                                             "rows)."),
            who: dict = Depends(api_caller)):
    project_id = who["project_id"]
    t0 = time.perf_counter()
    if len(body.rows) > serving.MAX_API_ROWS:
        raise HTTPException(413, f"At most {serving.MAX_API_ROWS:,} rows "
                                 f"per request; send batches or use a "
                                 f"batch job.")
    v = deployment.resolve_version(project_id, name, stage, version)
    if v is None:
        what = f"version {version}" if version else f"a {stage} version"
        raise HTTPException(404, f"Model {name} has no {what}.")
    rows = pd.DataFrame(body.rows)
    try:
        out = serving.score(deployment.model_info(v), rows, explain=explain)
    except serving.ScoringError as e:
        raise HTTPException(422, str(e)) from None
    cols = serving.output_columns(out)
    main = "ML_Cluster" if v["task"] == "clustering" else "ML_Prediction"
    ms = round((time.perf_counter() - t0) * 1000)
    request_id = None
    try:
        request_id = prediction_log.record(
            project_id, v, rows, out, source="api",
            caller=f"key: {who['name']}", ms=ms)
    except Exception:
        # The answer still goes back; the gap is on record for operators.
        log.event(_log, "prediction_log_failed", level=logging.ERROR,
                  exc_info=True, ref=log.incident_id())
    log.event(_log, "api_predicted", rows=len(out), task=v["task"],
              version=v["version"], ms=ms,
              flagged=int(out["ML_Warnings"].notna().sum())
              if "ML_Warnings" in out.columns else 0)
    return {"model": v["name"], "version": v["version"],
            "stage": v["stage"], "task": v["task"], "rows": len(out),
            "predictions": [r[main] for r in _json(out[[main]])],
            "outputs": _json(out[cols]), "request_id": request_id}
