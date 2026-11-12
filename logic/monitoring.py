"""Monitoring models in production: scheduled checks, alerts, retraining.

A monitor watches one model name. Each check takes the model's production
version and compares the data it was trained on (the prepared frame stored
with its run) with newer rows (the project's dataset, an uploaded file or a
Snowflake query — sources as a batch job's):

- Drift (`logic.ml_drift`) on the model's own features: columns with major
  or moderate PSI, the drift classifier's AUC, the correlations that moved.
- Accuracy, where the newer rows carry the actual outcome: the model's
  metric on them against the version's test score.

Thresholds turn that into a status — ok, warning or alert. An alert is sent
to the deployment's SNS topic and the monitor's own chat webhook (`utils.
alerts`). With `retrain="staging"`, an alert also retrains the model on the
original data plus the newly labelled rows and registers the result as a
**staging** version: production still needs the usual request and
approval.
"""
import io
import os

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select, update

from logic import (audit, deployment, ml, ml_drift, ml_eval, ml_prep, ml_recipe,
                   projects, serving)
from logic.deployment import _aware, _now, next_run
from logic.projects import AccessDenied, _email, _need
from utils import alerts, db, log, storage

_log = log.get("monitoring")

DEFAULTS = {
    "major_columns": 1,      # alert at this many columns with major drift
    "drift_auc": 0.75,       # alert when the drift classifier reaches this
    "metric_drop": 0.05,     # alert: higher-is-better metric down this much
    "error_rise": 0.10,      # alert: error metric up this share
}
RETRAIN = ("notify", "staging")
SERVICE = {"sub": None, "email": "monitoring@ml-auto", "role": "admin"}
STATUS_ORDER = ("ok", "warning", "alert")


# ── monitors ────────────────────────────────────────────────────────────
# A monitor reads newer rows from where a batch job can, or from the
# requests the model API answered (logic.prediction_log).
SOURCES = deployment.SOURCES + ("logged",)


def create_monitor(user, project_id, *, name, model_name, source,
                   schedule="", thresholds=None, webhook="",
                   retrain="notify"):
    if source.get("type") not in SOURCES:
        raise AccessDenied("Unknown source.")
    outcomes = source.get("outcomes")
    if outcomes and (outcomes.get("source", {}).get("type")
                     not in deployment.SOURCES or not outcomes.get("id")):
        raise AccessDenied("Outcomes need a source and an ID column.")
    if retrain not in RETRAIN:
        raise AccessDenied("Unknown retraining policy.")
    why = alerts.webhook_problem(webhook)
    if why:
        raise AccessDenied(why)
    limits = {**DEFAULTS, **{k: float(v) for k, v in (thresholds or {})
                             .items() if k in DEFAULTS}}
    with db.session() as s:
        _need(s, user, project_id, "editor")
        m = db.Monitor(project_id=project_id,
                       name=(name or model_name)[:200],
                       model_name=model_name, source=source,
                       thresholds=limits, alert_webhook=webhook or "",
                       retrain=retrain, schedule=(schedule or "").strip(),
                       next_run_at=next_run(schedule),
                       created_by=_email(user))
        s.add(m)
        s.flush()
        audit.record(s, user, "monitor.create", project_id, "monitor", m.id, model=model_name, retrain=retrain, schedule=m.schedule, webhook=bool(webhook))
        s.commit()
        return m.id


def list_monitors(user, project_id):
    owner = projects.can(user, project_id, "owner")
    with db.session() as s:
        _need(s, user, project_id, "viewer")
        rows = s.scalars(select(db.Monitor).where(
            db.Monitor.project_id == project_id)
            .order_by(db.Monitor.created_at.desc()))
        return [{"id": m.id, "name": m.name, "model_name": m.model_name,
                 "source": m.source, "thresholds": m.thresholds,
                 # A webhook URL is a credential: owners only.
                 "webhook": m.alert_webhook if owner else
                 ("set" if m.alert_webhook else ""),
                 "retrain": m.retrain, "schedule": m.schedule,
                 "enabled": m.enabled, "next_run_at": m.next_run_at,
                 "last_run_at": m.last_run_at,
                 "last_status": m.last_status,
                 "created_by": m.created_by} for m in rows]


def set_enabled(user, project_id, monitor_id, enabled):
    with db.session() as s:
        _need(s, user, project_id, "editor")
        m = _monitor(s, project_id, monitor_id)
        m.enabled = bool(enabled)
        m.next_run_at = next_run(m.schedule) if enabled else None
        audit.record(s, user, "monitor.resume" if enabled else "monitor.pause", project_id, "monitor", monitor_id)
        s.commit()


def delete_monitor(user, project_id, monitor_id):
    with db.session() as s:
        _need(s, user, project_id, "editor")
        m = _monitor(s, project_id, monitor_id)
        for r in s.scalars(select(db.MonitorRun).where(
                db.MonitorRun.monitor_id == monitor_id)):
            s.delete(r)
        s.delete(m)
        audit.record(s, user, "monitor.delete", project_id, "monitor", monitor_id)
        s.commit()


def _monitor(s, project_id, monitor_id):
    m = s.get(db.Monitor, monitor_id)
    if m is None or m.project_id != project_id:
        raise AccessDenied("That monitor is not in this project.")
    return m


def monitor_runs(user, project_id, monitor_id, limit=30):
    with db.session() as s:
        _need(s, user, project_id, "viewer")
        rows = s.scalars(select(db.MonitorRun).where(
            db.MonitorRun.monitor_id == monitor_id,
            db.MonitorRun.project_id == project_id)
            .order_by(db.MonitorRun.started_at.desc()).limit(limit))
        return [{"id": r.id, "status": r.status, "summary": r.summary,
                 "message": r.message, "alerted": r.alerted,
                 "retrained_version_id": r.retrained_version_id,
                 "triggered_by": r.triggered_by,
                 "started_at": r.started_at,
                 "finished_at": r.finished_at} for r in rows]


# ── the check ───────────────────────────────────────────────────────────
class CheckError(ValueError):
    """A check that could not run, in words for the run's message."""


def _training_run(version):
    """The stored run a version came from (with its prepared frame)."""
    with db.session() as s:
        v = s.get(db.ModelVersion, version["id"])
        r = s.get(db.Run, v.run_id) if v.run_id else None
        if r is None:
            raise CheckError("The model's training run is not stored, so "
                             "there is nothing to compare against.")
        run_key, dataset_id = r.object_key, r.dataset_id
        d_key = (s.get(db.Dataset, dataset_id).object_key
                 if dataset_id else None)
    run = joblib.load(io.BytesIO(storage.get(run_key)))
    if run.get("frame") is None:
        if not d_key:
            raise CheckError("The training data is not stored.")
        source = pd.read_parquet(io.BytesIO(storage.get(d_key)))
        run["frame"] = ml_prep.apply_steps(source,
                                           run.get("prep_steps") or [])
    return run


def _prepared(info, rows):
    steps = info.get("prep_steps") or []
    if not steps:
        return rows.rename(columns=str)
    try:
        return ml_prep.apply_steps(rows.rename(columns=str), steps,
                                   rows=False)
    except ml_prep.StepError as e:
        raise CheckError(f"The model's preparation does not fit the new "
                         f"rows. {e}") from None


def _performance(info, version, current):
    """The model's metric on the labelled new rows, or None."""
    recipe = info["recipe"]
    target, task = recipe.get("target"), info["task"]
    if task == "clustering" or not target or target not in current.columns:
        return None
    y = ml_recipe.target_values(current, recipe)
    classes = list(info.get("classes") or [])
    keep = y.notna() & (y.isin(classes) if task == "classification"
                        else True)
    if keep.sum() < 20:
        return None
    rows = current[keep]
    scored = serving.score(info, rows)
    pred = scored["ML_Prediction"].to_numpy()
    proba = None
    if task == "classification":
        cols = [f"ML_Probability_{c}" for c in classes]
        if all(c in scored.columns for c in cols):
            proba = scored[cols].to_numpy(dtype=float)
    positive = info.get("positive") if len(classes) == 2 else None
    metrics = ml.score_predictions(task, y[keep].to_numpy(), pred, proba,
                                   classes, positive)
    metric = version.get("metric") or ml.SELECTABLE[
        ml.metric_family(task, len(classes) or None)][0]
    value = metrics.get(metric)
    if value is None or not np.isfinite(value):
        return None
    family = ml.metric_family(task, len(classes) or None)
    return {"metric": metric, "label": ml.metric_label(family, metric),
            "higher_is_better": ml.higher_is_better(family, metric),
            "baseline": version.get("score"), "current": float(value),
            "labelled": int(keep.sum())}


def assess(summary, limits):
    """(status, reasons) from a check's summary and the thresholds."""
    reasons, level = [], 0
    major, moderate = summary["major"], summary["moderate"]
    if len(major) >= limits["major_columns"]:
        level = 2
        reasons.append(f"major drift in {', '.join(major[:5])}")
    elif major or moderate:
        level = max(level, 1)
        reasons.append("drift in " + ", ".join((major + moderate)[:5]))
    auc = summary.get("drift_auc")
    if auc is not None:
        if auc >= limits["drift_auc"]:
            level = 2
            reasons.append(f"the new rows are easy to tell apart "
                           f"(drift AUC {auc:.2f})")
        elif auc >= 0.65:
            level = max(level, 1)
            reasons.append(f"some overall drift (drift AUC {auc:.2f})")
    perf = summary.get("performance")
    if perf and perf["baseline"] is not None:
        base, cur = perf["baseline"], perf["current"]
        if perf["higher_is_better"]:
            worse, limit = base - cur, limits["metric_drop"]
            how = f"{perf['label']} {base:.3f} → {cur:.3f}"
        else:
            worse = (cur - base) / abs(base) if base else 0.0
            limit = limits["error_rise"]
            how = f"{perf['label']} {base:,.4g} → {cur:,.4g}"
        if worse >= limit:
            level = 2
            reasons.append(f"accuracy fell: {how}")
        elif worse >= limit / 2:
            level = max(level, 1)
            reasons.append(f"accuracy slipping: {how}")
    return STATUS_ORDER[level], reasons


def check(version, info, train_run, current, limits):
    """The summary of one check (no side effects)."""
    reference = train_run["frame"]
    current = _prepared(info, current)
    columns = info.get("columns") or {f: f for f in info["features"]}
    features = [c for c in columns.values() if c in reference.columns
                and c in current.columns]
    target = info["recipe"].get("target")
    cols = features + ([target] if target and target in reference.columns
                       and target in current.columns else [])
    result = ml_drift.analyse(reference[cols], current[cols],
                              target if target in cols else None)
    cov = result["covariate"]
    cov = cov[cov["column"] != target] if not cov.empty else cov
    pairs = result["pairs"]
    summary = {
        "rows": len(current), "reference_rows": len(reference),
        "major": list(cov.loc[cov["severity"] == "Major", "column"]),
        "moderate": list(cov.loc[cov["severity"] == "Moderate", "column"]),
        "top_psi": [{"column": r.column, "psi": float(r.psi)}
                    for r in cov.head(8).itertuples()],
        "drift_auc": (result["classifier"] or {}).get("auc"),
        "pairs": [{"first": r.first, "second": r.second,
                   "change": float(r.change)}
                  for r in pairs.head(3).itertuples()]
        if not pairs.empty else [],
        "performance": _performance(info, version, current),
        "missing_features": [c for c in columns.values()
                             if c not in current.columns],
    }
    status, reasons = assess(summary, limits)
    summary["reasons"] = reasons
    return status, summary, current


def _retrain(project_id, version, train_run, current, monitor_name):
    """Train the model again on the original data plus the newly labelled
    rows, with the settings it won with; register it in staging."""
    if train_run.get("external"):
        raise CheckError("A model trained elsewhere is not retrained here: "
                         "retrain it where it was built and register the "
                         "new file.")
    target = train_run.get("target")
    reference = train_run["frame"]
    if target and target in current.columns:
        labelled = current[current[target].notna()]
        both = [c for c in reference.columns if c in labelled.columns]
        data = pd.concat([reference, labelled[both]], ignore_index=True)
    else:
        data = reference
    best = ml.result(train_run, train_run["best"])
    config = {**train_run["config"],
              "models": [{"key": best["key"], "params": best["params"],
                          "search": None}]}
    run = ml.train(data, config, lambda f, t: None)
    run["prep_steps"] = list(train_run.get("prep_steps") or [])
    run["importance"] = {}
    run["frame"] = data
    res = ml.result(run, best["key"])
    if res is None or res.get("error"):
        raise CheckError("Retraining failed.")
    rid = projects.save_run(SERVICE, project_id, run)
    score = (res.get("test") or {}).get(run.get("metric"))
    got = projects.register_model(
        SERVICE, project_id, rid, ml_eval.model_bytes(run, res),
        name=version["name"], model_key=best["key"], task=run["task"],
        target=run.get("target"), metric=run.get("metric"),
        score=None if score is None else float(score),
        description=f"Retrained by monitor {monitor_name} on "
                    f"{len(data):,} rows after an alert.")
    deployment.set_stage(SERVICE, project_id, got["id"], "staging",
                         "retrained by monitoring")
    return got


def _waiting_retrain(project_id, version):
    """The number of a version monitoring retrained after the current
    production one that is still in staging, or None: one waiting for
    review is enough — a daily alert must not stack a version a day."""
    with db.session() as s:
        prod = s.get(db.ModelVersion, version["id"])
        waiting = s.scalars(select(db.ModelVersion).where(
            db.ModelVersion.project_id == project_id,
            db.ModelVersion.name == version["name"],
            db.ModelVersion.stage == "staging",
            db.ModelVersion.created_by == SERVICE["email"],
            db.ModelVersion.version > prod.version)
            .order_by(db.ModelVersion.version.desc())).first()
        return waiting.version if waiting else None


def _message(m, version, status, summary, retrained):
    lines = [f"Model {version['name']} v{version['version']} "
             f"(production) — monitor {m['name']}: {status.upper()}."]
    lines += [f"• {r}" for r in summary.get("reasons", [])]
    perf = summary.get("performance")
    if perf:
        test = ("not recorded" if perf["baseline"] is None
                else f"{perf['baseline']:.4g}")
        lines.append(f"• {perf['label']} on {perf['labelled']:,} labelled "
                     f"rows: {perf['current']:.4g} (at training: {test})")
    lines.append(f"• Checked {summary['rows']:,} new rows against "
                 f"{summary['reference_rows']:,} training rows.")
    if retrained:
        lines.append(f"• Retrained: {version['name']} v{retrained['version']}"
                     f" is in staging. Review it and request production.")
    url = os.environ.get("APP_URL")
    if url:
        lines.append(f"Open the project: {url}")
    return "\n".join(lines)


def _rows(mon, version, snowflake):
    """The newer rows a check compares with the training data."""
    source = mon["source"]
    if source["type"] != "logged":
        return deployment._read_source(mon["project_id"], source, snowflake)
    from logic import prediction_log
    days = int(source.get("days") or 30)
    rows = prediction_log.window(mon["project_id"], mon["model_name"], days)
    if rows.empty:
        raise CheckError(f"No requests to {mon['model_name']} were logged "
                         f"in the last {days} days.")
    outcomes = source.get("outcomes")
    if outcomes:
        rows = _with_outcomes(mon, version, rows, outcomes, snowflake)
    return rows.drop(columns=["_requested_at"], errors="ignore")


def _with_outcomes(mon, version, rows, outcomes, snowflake):
    """The logged rows with the actual outcome where it has arrived: the
    latest request for each ID, joined on that ID. Accuracy is then
    measured on the rows that have one; drift uses them all."""
    key = outcomes["id"]
    target = (version.get("target") or (deployment.model_info(version)
                                        .get("recipe") or {}).get("target")
              or "").strip()
    actual = deployment._read_source(mon["project_id"], outcomes["source"],
                                     snowflake)
    for col, where in ((key, "the logged requests"), (key, "the outcomes"),
                       (target, "the outcomes")):
        frame = rows if where == "the logged requests" else actual
        if not col or col not in frame.columns:
            raise CheckError(f"{col or 'The target'} is not in {where}.")
    latest = rows.sort_values("_requested_at").drop_duplicates(key,
                                                                keep="last")
    latest = latest.drop(columns=[target], errors="ignore")
    actual = actual[[key, target]].drop_duplicates(key, keep="last")
    latest[key] = latest[key].astype(str)
    actual[key] = actual[key].astype(str)
    return latest.merge(actual, on=key, how="left")


def run_monitor(monitor_id, triggered_by="schedule", snowflake=None,
                sender=None):
    """One check, recorded; alerts and retraining as configured. Returns
    the run's summary. Failures are recorded, not raised."""
    snowflake = snowflake or deployment._service_snowflake
    with db.session() as s:
        m = s.get(db.Monitor, monitor_id)
        if m is None:
            raise AccessDenied("No such monitor.")
        mon = {"id": m.id, "project_id": m.project_id, "name": m.name,
               "model_name": m.model_name, "source": dict(m.source),
               "limits": {**DEFAULTS, **dict(m.thresholds or {})},
               "webhook": m.alert_webhook, "retrain": m.retrain}
        r = db.MonitorRun(monitor_id=m.id, project_id=m.project_id,
                          status="running", triggered_by=triggered_by)
        s.add(r)
        s.commit()
        run_id = r.id
    status, summary, message, version_id = "failed", {}, "", None
    alerted, retrained = False, None
    try:
        version = deployment.resolve_version(mon["project_id"],
                                             mon["model_name"], "production")
        if version is None:
            raise CheckError(f"{mon['model_name']} has no production "
                             f"version to monitor.")
        version_id = version["id"]
        with db.session() as s:
            v = s.get(db.ModelVersion, version_id)
            version.update(metric=v.metric, score=v.score)
        info = deployment.model_info(version)
        train_run = _training_run(version)
        rows = _rows(mon, version, snowflake)
        status, summary, current = check(version, info, train_run, rows,
                                         mon["limits"])
        waiting = (_waiting_retrain(mon["project_id"], version)
                   if status == "alert" and mon["retrain"] == "staging"
                   else None)
        if waiting:
            summary["reasons"].append(
                f"a retrained version (v{waiting}) is already waiting in "
                f"staging, so no new one was trained")
        elif status == "alert" and mon["retrain"] == "staging":
            try:
                retrained = _retrain(mon["project_id"], version, train_run,
                                     current, mon["name"])
            except Exception:
                log.event(_log, "monitor_retrain_failed", level=40,
                          exc_info=True)
                summary.setdefault("reasons", []).append(
                    "retraining was attempted and failed")
        message = "; ".join(summary.get("reasons") or ["no drift found"])
        if status == "alert":
            reached = alerts.send(
                f"ML alert: {version['name']} — {mon['name']}",
                _message(mon, version, status, summary, retrained),
                webhook=mon["webhook"] or None, sender=sender)
            alerted = bool(reached)
    except (CheckError, serving.ScoringError) as e:
        message = str(e)
    except Exception as e:
        message = f"{type(e).__name__}: the check could not run."
        log.event(_log, "monitor_failed", level=40, exc_info=True)
    with db.session() as s:
        r = s.get(db.MonitorRun, run_id)
        r.status, r.summary, r.message = status, _jsonable(summary), message
        r.alerted, r.model_version_id = alerted, version_id
        r.retrained_version_id = retrained["id"] if retrained else None
        r.finished_at = _now()
        m = s.get(db.Monitor, mon["id"])
        m.last_run_at, m.last_status = _now(), status
        pending = _aware(m.next_run_at)
        if m.schedule and m.enabled and (pending is None
                                         or pending <= _now()):
            m.next_run_at = next_run(m.schedule)
        audit.record(s, (triggered_by if "@" in triggered_by else "system:monitor"), "monitor.check", m.project_id, "monitor", m.id, status=status, alerted=alerted, retrained=retrained["id"] if retrained else None)
        s.commit()
    log.event(_log, "monitor_checked", status=status, alerted=alerted,
              retrained=bool(retrained))
    return {"id": run_id, "status": status, "message": message,
            "alerted": alerted,
            "retrained": retrained["version"] if retrained else None}


def _jsonable(x):
    """Numbers from numpy as plain ones, for the JSON column."""
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def run_now(user, project_id, monitor_id, snowflake=None, sender=None):
    with db.session() as s:
        _need(s, user, project_id, "editor")
        _monitor(s, project_id, monitor_id)
    return run_monitor(monitor_id, triggered_by=_email(user),
                       snowflake=snowflake, sender=sender)


def run_due(now=None):
    now = now or _now()
    with db.session() as s:
        due = [(m.id, m.next_run_at, m.schedule)
               for m in s.scalars(select(db.Monitor).where(
                   db.Monitor.enabled.is_(True),
                   db.Monitor.next_run_at.is_not(None)))
               if _aware(m.next_run_at) <= now]
    out = []
    for monitor_id, seen, schedule in due:
        with db.session() as s:
            claimed = s.execute(update(db.Monitor).where(
                db.Monitor.id == monitor_id,
                db.Monitor.next_run_at == seen).values(
                next_run_at=next_run(schedule, now))).rowcount == 1
            s.commit()
        if claimed:
            out.append(run_monitor(monitor_id))
    return out
