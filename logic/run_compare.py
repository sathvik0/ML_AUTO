"""Two runs side by side: what was trained on, how, and how each model
scored. Only what differs is listed for settings and features, so the
answer to "what changed between these runs?" is the first thing read."""
from logic import ml, ml_prep

# (label, how to read it from a run) for the summary rows.
_VALIDATION = (("test_size", "Test share"), ("folds", "Folds"),
               ("stratify", "Stratified"), ("group", "Held-out groups"),
               ("time", "Split by time on"))
_EXTRAS = (("metric", "Best model chosen by"), ("seed", "Seed"),
           ("balance", "Class weights"), ("resample", "Resampling"),
           ("stack", "Stack of the best"), ("calibration", "Calibration"))


def _features(run):
    cfg = run.get("config") or {}
    return list(cfg.get("numeric") or []) + list(cfg.get("categorical") or [])


def _best_score(run):
    best = ml.result(run, run["best"]) if run.get("best") else None
    if not best or best.get("error"):
        return None
    return (best.get("test") or {}).get(run.get("metric"))


def summary(run):
    """{label: value} for one run, in the order the comparison shows."""
    best = ml.result(run, run["best"]) if run.get("best") else None
    score = _best_score(run)
    out = {
        "Task": ml.TASK_LABELS.get(run["task"], run["task"]),
        "Target": run.get("target") or "—",
        "Rows: train / test": f"{run.get('n_train', 0):,} / "
                              f"{run.get('n_test', 0):,}",
        "Features": len(_features(run)),
        "Preparation steps": len(run.get("prep_steps") or []),
        "Models compared": len([r for r in run["results"]
                                if r["key"] != ml.BASELINE]),
        "Best model": best["label"] if best else "—",
        "Best score": "—" if score is None else
        f"{ml.metric_label(run['family'], run['metric'])} {score:.3f}",
    }
    ts = run.get("time_split")
    if ts:
        out["Tested on"] = f"{ts['test_from']} to {ts['test_to']}"
    return out


def _value(v):
    if v is None or v == "" or v == "none":
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _settings(run):
    cfg = run.get("config") or {}
    val = cfg.get("validation") or {}
    out = {label: _value(val.get(k)) for k, label in _VALIDATION}
    out.update({label: _value(cfg.get(k)) for k, label in _EXTRAS})
    for k, v in sorted((cfg.get("prep") or {}).items()):
        out[f"Preparation: {k}"] = _value(v)
    for m in cfg.get("models") or []:
        name = ml.model_label(run["task"], m["key"])
        if m.get("search"):
            out[f"{name}: tuning"] = m["search"]["mode"]
        for k, v in sorted((m.get("params") or {}).items()):
            out[f"{name}: {k}"] = _value(v)
    return out


def compare(a, b):
    """What differs between runs `a` and `b`, as plain rows the page draws:

    - summary: (label, a, b) for every summary line
    - features: columns only in a, only in b, and how many are shared
    - steps: preparation steps (described) only in a, only in b
    - settings: (setting, a, b) where they differ; a model one run did not
      train shows "—" on that side
    - models: (model, score in a, score in b) on each run's own metric,
      when both runs chose by the same one
    """
    sa, sb = summary(a), summary(b)
    rows = [(k, sa.get(k, "—"), sb.get(k, "—"))
            for k in list(sa) + [k for k in sb if k not in sa]]
    fa, fb = _features(a), _features(b)
    da = [ml_prep.describe(s) for s in a.get("prep_steps") or []]
    db_ = [ml_prep.describe(s) for s in b.get("prep_steps") or []]
    ga, gb = _settings(a), _settings(b)
    settings = [(k, ga.get(k, "—"), gb.get(k, "—"))
                for k in list(ga) + [k for k in gb if k not in ga]
                if ga.get(k, "—") != gb.get(k, "—")]
    models = []
    if a.get("metric") == b.get("metric"):
        def scores(run):
            return {r["label"]: (r.get("test") or {}).get(run["metric"])
                    for r in run["results"] if not r.get("error")}
        xa, xb = scores(a), scores(b)
        for label in list(xa) + [k for k in xb if k not in xa]:
            models.append((label, xa.get(label), xb.get(label)))
    return {"summary": rows,
            "features": {"only_a": [c for c in fa if c not in fb],
                         "only_b": [c for c in fb if c not in fa],
                         "shared": len([c for c in fa if c in fb])},
            "steps": {"only_a": [d for d in da if d not in db_],
                      "only_b": [d for d in db_ if d not in da]},
            "settings": settings, "models": models,
            "metric": ml.metric_label(a["family"], a["metric"])
            if a.get("metric") == b.get("metric") else None}
