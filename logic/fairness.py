"""Fairness checks: does a model treat groups of a sensitive attribute
(gender, age band, region…) differently?

Measured on the model's **out-of-sample** predictions from its training run
— the test rows, and the out-of-fold rows when cross-validation was on —
never on rows it was fitted to. Per group:

- yes/no classification: selection rate (how often the model says yes),
  the actual rate, true-positive rate, false-positive rate, precision and
  accuracy. Flags: disparate impact below 0.8 (the four-fifths rule:
  lowest selection rate / highest), a true-positive-rate gap above 0.1
  (equal opportunity), a false-positive-rate gap above 0.1.
- several classes: accuracy per group; flag a gap above 0.1.
- regression: mean prediction and actual, MAE and mean error (bias).
  Flags: MAE ratio above 1.25, a bias gap above a tenth of the target's
  standard deviation.

A numeric attribute is cut into quartile bands; groups under MIN_GROUP rows
are listed but not judged. A difference can reflect a real difference in
the groups' actual rates (shown beside it) rather than bias in the model,
so a flagged report can be accepted by a project owner with a written
justification. A project can require a passing or accepted report before
production (`policy`).
"""
import io

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select

from logic import audit, ml
from logic.projects import AccessDenied, _email, _need
from utils import db, storage

MIN_GROUP = 30
MAX_LEVELS = 30
DI_MIN = 0.8
RATE_GAP = 0.10
MAE_RATIO = 1.25
BIAS_GAP_SD = 0.10
SENSITIVE_HINTS = ("gender", "sex", "age", "race", "ethnic", "religion",
                   "marital", "disab", "nationality", "citizen", "region",
                   "language")


# ── settings: the project's policy ──────────────────────────────────────
def policy(user, project_id):
    with db.session() as s:
        p, _ = _need(s, user, project_id, "viewer")
        return {"require_fairness": bool((p.settings or {}).get(
            "require_fairness"))}


def set_policy(user, project_id, require_fairness):
    with db.session() as s:
        p, _ = _need(s, user, project_id, "owner")
        p.settings = {**(p.settings or {}),
                      "require_fairness": bool(require_fairness)}
        audit.record(s, user, "project.policy", project_id, "project",
                     project_id, require_fairness=bool(require_fairness))
        s.commit()


def production_blocker(s, project_id, version_id):
    """Why a version may not be requested for production under the
    project's policy, or None."""
    p = s.get(db.Project, project_id)
    if not (p.settings or {}).get("require_fairness"):
        return None
    r = s.scalars(select(db.FairnessReport).where(
        db.FairnessReport.version_id == version_id)
        .order_by(db.FairnessReport.created_at.desc())).first()
    if r is None:
        return ("This project requires a fairness check before production: "
                "run one for this version on the project page.")
    if r.status == "pass" or r.accepted_by:
        return None
    return ("The latest fairness check for this version is flagged and "
            "not yet accepted by an owner.")


# ── the check ───────────────────────────────────────────────────────────
def suggest(columns):
    return [c for c in columns
            if any(h in str(c).lower() for h in SENSITIVE_HINTS)]


def groups_of(series):
    """Group labels: categories as they are (the rarest beyond MAX_LEVELS
    folded into "other"), numbers in quartile bands."""
    v = pd.to_numeric(series, errors="coerce")
    if v.notna().mean() > 0.9 and v.nunique() > 10:
        edges = np.unique(np.nanquantile(v, [0, .25, .5, .75, 1]))
        if len(edges) > 2:
            labels = [f"{edges[i]:g}–{edges[i + 1]:g}"
                      for i in range(len(edges) - 1)]
            bands = pd.cut(v, edges, labels=labels, include_lowest=True)
            return bands.astype(object).where(bands.notna(), "(missing)")
    text = series.astype(object).where(series.notna(), "(missing)") \
        .astype(str)
    keep = text.value_counts().index[:MAX_LEVELS]
    return text.where(text.isin(keep), "other")


def _out_of_sample(run):
    split = pd.Series(np.asarray(run["split"], dtype=object))
    return (split.eq("test") | split.str.contains("out of fold", na=False)) \
        .to_numpy()


def _classification(y, pred, groups, positive):
    rows = []
    for g in sorted(groups.unique(), key=str):
        m = (groups == g).to_numpy()
        yt, yp = y[m] == positive, pred[m] == positive
        n = int(m.sum())
        rows.append({
            "group": g, "rows": n,
            "selection_rate": float(yp.mean()) if n else None,
            "actual_rate": float(yt.mean()) if n else None,
            "tpr": float(yp[yt].mean()) if yt.any() else None,
            "fpr": float(yp[~yt].mean()) if (~yt).any() else None,
            "precision": float(yt[yp].mean()) if yp.any() else None,
            "accuracy": float((yt == yp).mean()) if n else None})
    return rows


def _gap(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return (max(vals) - min(vals)) if len(vals) > 1 else None


def _assess_classification(rows):
    judged = [r for r in rows if r["rows"] >= MIN_GROUP]
    if len(judged) < 2:
        return "insufficient", {}, []
    sel = [r["selection_rate"] for r in judged]
    di = (min(sel) / max(sel)) if max(sel) > 0 else None
    metrics = {"disparate_impact": di,
               "tpr_gap": _gap(judged, "tpr"),
               "fpr_gap": _gap(judged, "fpr"),
               "accuracy_gap": _gap(judged, "accuracy")}
    flags = []
    if di is not None and di < DI_MIN:
        lo = min(judged, key=lambda r: r["selection_rate"])["group"]
        hi = max(judged, key=lambda r: r["selection_rate"])["group"]
        flags.append(f"disparate impact {di:.2f} (below {DI_MIN}): {lo} is "
                     f"selected far less often than {hi}")
    for key, label in (("tpr_gap", "true-positive rate"),
                       ("fpr_gap", "false-positive rate")):
        if metrics[key] is not None and metrics[key] > RATE_GAP:
            flags.append(f"{label} differs by {metrics[key]:.2f} across "
                         f"groups (over {RATE_GAP})")
    return ("flag" if flags else "pass"), metrics, flags


def _multiclass(y, pred, groups):
    rows = []
    for g in sorted(groups.unique(), key=str):
        m = (groups == g).to_numpy()
        rows.append({"group": g, "rows": int(m.sum()),
                     "accuracy": float((y[m] == pred[m]).mean())
                     if m.any() else None})
    judged = [r for r in rows if r["rows"] >= MIN_GROUP]
    if len(judged) < 2:
        return rows, "insufficient", {}, []
    gap = _gap(judged, "accuracy")
    flags = ([f"accuracy differs by {gap:.2f} across groups"]
             if gap is not None and gap > RATE_GAP else [])
    return rows, ("flag" if flags else "pass"), {"accuracy_gap": gap}, flags


def _regression(y, pred, groups):
    y, pred = y.astype(float), pred.astype(float)
    sd = float(np.nanstd(y)) or 1.0
    rows = []
    for g in sorted(groups.unique(), key=str):
        m = (groups == g).to_numpy()
        err = pred[m] - y[m]
        rows.append({"group": g, "rows": int(m.sum()),
                     "mean_prediction": float(np.mean(pred[m])),
                     "mean_actual": float(np.mean(y[m])),
                     "mae": float(np.mean(np.abs(err))),
                     "mean_error": float(np.mean(err))})
    judged = [r for r in rows if r["rows"] >= MIN_GROUP]
    if len(judged) < 2:
        return rows, "insufficient", {}, []
    maes = [r["mae"] for r in judged]
    ratio = max(maes) / min(maes) if min(maes) > 0 else None
    bias_gap = _gap(judged, "mean_error")
    flags = []
    if ratio is not None and ratio > MAE_RATIO:
        flags.append(f"errors are {ratio:.2f}× larger for some groups "
                     f"(MAE ratio over {MAE_RATIO})")
    if bias_gap is not None and bias_gap > BIAS_GAP_SD * sd:
        flags.append(f"the model over- or under-predicts some groups more "
                     f"(bias gap {bias_gap:,.4g})")
    return rows, ("flag" if flags else "pass"), {
        "mae_ratio": ratio, "bias_gap": bias_gap}, flags


def assess(run, model_key, attributes):
    """The report for one trained model: per attribute, groups, metrics,
    flags and a status. No side effects."""
    res = ml.result(run, model_key)
    frame = run["frame"]
    y_all = run["y"]
    keep = _out_of_sample(run) & y_all.notna().to_numpy()
    y = y_all.to_numpy()[keep]
    pred = np.asarray(res["pred"], dtype=object)[keep]
    rows_frame = frame.loc[y_all.index[keep]]
    task, classes = run["task"], run.get("classes") or []
    out = {"rows": int(keep.sum()), "task": task, "attributes": {}}
    for attr in attributes:
        if attr not in rows_frame.columns:
            raise AccessDenied(f"{attr} is not in the training data.")
        groups = groups_of(rows_frame[attr])
        if task == "classification" and len(classes) == 2:
            rows = _classification(y, pred, groups, run["positive"])
            status, metrics, flags = _assess_classification(rows)
        elif task == "classification":
            rows, status, metrics, flags = _multiclass(y, pred, groups)
        else:
            rows, status, metrics, flags = _regression(
                pd.to_numeric(pd.Series(y)).to_numpy(),
                pd.to_numeric(pd.Series(pred)).to_numpy(), groups)
        out["attributes"][attr] = {"status": status, "groups": rows,
                                   "metrics": metrics, "flags": flags,
                                   "min_group": MIN_GROUP}
    statuses = [a["status"] for a in out["attributes"].values()]
    out["status"] = ("flag" if "flag" in statuses else
                     "pass" if "pass" in statuses else "insufficient")
    return out


def _version_run(s, project_id, version_id):
    v = s.get(db.ModelVersion, version_id)
    if v is None or v.project_id != project_id:
        raise AccessDenied("That model version is not in this project.")
    r = s.get(db.Run, v.run_id) if v.run_id else None
    if r is None:
        raise AccessDenied("The version's training run is not stored.")
    if v.task == "clustering":
        raise AccessDenied("Fairness checks need a model with a target.")
    return v, r.object_key


def training_columns(user, project_id, version_id):
    with db.session() as s:
        _need(s, user, project_id, "viewer")
        v, key = _version_run(s, project_id, version_id)
        target = v.target
    run = joblib.load(io.BytesIO(storage.get(key)))
    return [c for c in run["frame"].columns if c != target]


def check(user, project_id, version_id, attributes):
    """Run and store a report; its id and the report."""
    attributes = [a for a in dict.fromkeys(attributes or [])]
    if not attributes:
        raise AccessDenied("Choose at least one attribute.")
    with db.session() as s:
        _need(s, user, project_id, "editor")
        v, key = _version_run(s, project_id, version_id)
        model_key = v.model_key
    run = joblib.load(io.BytesIO(storage.get(key)))
    if run.get("frame") is None:
        raise AccessDenied("The version's training data is not stored.")
    report = assess(run, model_key, attributes)
    with db.session() as s:
        r = db.FairnessReport(project_id=project_id, version_id=version_id,
                              attributes=attributes, report=report,
                              status=report["status"], rows=report["rows"],
                              created_by=_email(user))
        s.add(r)
        s.flush()
        audit.record(s, user, "fairness.check", project_id,
                     "model_version", version_id, report=r.id,
                     status=report["status"], attributes=attributes)
        s.commit()
        return r.id, report


def accept(user, project_id, report_id, note):
    """An owner accepts a flagged report, with a written reason."""
    note = (note or "").strip()
    if len(note) < 20:
        raise AccessDenied("Explain why in a sentence or two (20 "
                           "characters at least): it is kept with the "
                           "report.")
    with db.session() as s:
        _need(s, user, project_id, "owner")
        r = s.get(db.FairnessReport, report_id)
        if r is None or r.project_id != project_id:
            raise AccessDenied("That report is not in this project.")
        if r.status != "flag":
            raise AccessDenied("Only a flagged report needs accepting.")
        r.accepted_by, r.accepted_at = _email(user), db.now()
        r.acceptance_note = note
        audit.record(s, user, "fairness.accept", project_id,
                     "model_version", r.version_id, report=r.id)
        s.commit()


def reports(user, project_id, version_id=None):
    with db.session() as s:
        _need(s, user, project_id, "viewer")
        q = select(db.FairnessReport).where(
            db.FairnessReport.project_id == project_id)
        if version_id:
            q = q.where(db.FairnessReport.version_id == version_id)
        rows = s.scalars(q.order_by(db.FairnessReport.created_at.desc()))
        return [{"id": r.id, "version_id": r.version_id,
                 "attributes": r.attributes, "report": r.report,
                 "status": r.status, "rows": r.rows,
                 "created_by": r.created_by, "created_at": r.created_at,
                 "accepted_by": r.accepted_by, "accepted_at": r.accepted_at,
                 "note": r.acceptance_note} for r in rows]
