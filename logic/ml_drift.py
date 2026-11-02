"""Drift: how a newer file differs from the file a model learns from.

Two kinds, each measured on every column the two files share:

- **Covariate drift** — has each column's distribution moved? The
  Population Stability Index (PSI) on bins fixed from the reference file
  (deciles for a number, the categories for a category), with a
  Kolmogorov–Smirnov test for numbers and a chi-square test for categories,
  the change in missing values, and categories the reference never had. A
  drift classifier — a model asked to tell the two files apart — gives one
  number for the whole file (its ROC AUC: 0.5 when it cannot) and names the
  columns that give the new file away.
- **Correlational drift** — have the relationships moved? Spearman
  correlations between numeric columns in each file and their difference,
  and, when both files have the target, each feature's relationship with it
  (Spearman for a number, the correlation ratio for a category).

PSI bands follow the usual reading: below 0.1 stable, 0.1 to 0.25 moderate,
above 0.25 major.
"""
import numpy as np
import pandas as pd
from scipy import stats

from logic import ml, schema

PSI_MODERATE = 0.10
PSI_MAJOR = 0.25
BINS = 10
EPS = 1e-4                 # an empty bin's share, so the log is defined
MISSING = "(missing)"
MAX_ROWS_CLASSIFIER = 50_000   # per file, for the drift classifier only
SEVERITY = ("Stable", "Moderate", "Major")


def severity(psi):
    if psi is None or not np.isfinite(psi):
        return "Stable"
    return ("Major" if psi >= PSI_MAJOR else
            "Moderate" if psi >= PSI_MODERATE else "Stable")


def _psi(ref_share, cur_share):
    r = np.clip(np.asarray(ref_share, dtype=float), EPS, None)
    c = np.clip(np.asarray(cur_share, dtype=float), EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _numbers(s):
    v = pd.to_numeric(s, errors="coerce").astype("float64")
    return v.where(np.isfinite(v))


def numeric_bins(ref):
    """Inner edges from the reference's deciles; the outer bins are open."""
    v = ref.dropna()
    if v.empty:
        return np.array([])
    return np.unique(np.quantile(v, np.linspace(0, 1, BINS + 1)[1:-1]))


def _numeric_shares(v, edges):
    """Shares per bin, with the missing values as a bin of their own."""
    codes = np.digitize(v.dropna(), edges, right=True)
    counts = np.bincount(codes, minlength=len(edges) + 1).astype(float)
    counts = np.append(counts, v.isna().sum())
    return counts / max(len(v), 1)


def _labels(s):
    return s.astype(object).where(s.notna(), MISSING).astype(str)


def column_types(ref, columns):
    """"numeric" or "categorical" for each column, read from the reference."""
    return {c: schema.detect_dtype(ref[c]) for c in columns}


def shared_columns(ref, cur, prof=None):
    """(columns to compare, only in the reference, only in the new file).
    Identifiers and constant columns carry no distribution to compare."""
    prof = ml.profile(ref) if prof is None else prof
    skip = set(prof.loc[prof["kind"].isin(["id", "constant"]), "column"])
    both = [c for c in ref.columns if c in cur.columns and c not in skip]
    return (both, [c for c in ref.columns if c not in cur.columns],
            [c for c in cur.columns if c not in ref.columns])


def covariate(ref, cur, columns, types):
    """One row per column: PSI, its severity, the test, the shift in
    centre (numbers) or top category (categories), missing values before
    and after, and categories new to the file."""
    rows = []
    for c in columns:
        if types[c] == "numeric":
            r, n = _numbers(ref[c]), _numbers(cur[c])
            edges = numeric_bins(r)
            psi = _psi(_numeric_shares(r, edges), _numeric_shares(n, edges))
            if r.notna().sum() > 1 and n.notna().sum() > 1:
                ks = stats.ks_2samp(r.dropna(), n.dropna())
                stat, p = float(ks.statistic), float(ks.pvalue)
            else:
                stat, p = np.nan, np.nan
            change = (f"mean {r.mean():,.4g} → {n.mean():,.4g}; median "
                      f"{r.median():,.4g} → {n.median():,.4g}")
            new = ""
            test = "Kolmogorov–Smirnov"
        else:
            r, n = _labels(ref[c]), _labels(cur[c])
            cats = sorted(set(r) | set(n))
            rc = r.value_counts().reindex(cats, fill_value=0)
            nc = n.value_counts().reindex(cats, fill_value=0)
            psi = _psi(rc / max(len(r), 1), nc / max(len(n), 1))
            table = np.vstack([rc.to_numpy(), nc.to_numpy()])
            table = table[:, table.sum(axis=0) > 0]
            if table.shape[1] > 1:
                chi = stats.chi2_contingency(table)
                stat, p = float(chi[0]), float(chi[1])
            else:
                stat, p = np.nan, np.nan
            top = (nc / max(len(n), 1) - rc / max(len(r), 1)).abs() \
                .sort_values(ascending=False)
            k = top.index[0] if len(top) else None
            change = (f"{k}: {rc[k] / max(len(r), 1):.1%} → "
                      f"{nc[k] / max(len(n), 1):.1%}" if k is not None
                      else "")
            unseen = sorted(set(n) - set(r) - {MISSING})
            new = ", ".join(unseen[:5]) + (f" and {len(unseen) - 5} more"
                                           if len(unseen) > 5 else "")
            test = "Chi-square"
        rows.append({
            "column": c, "type": types[c], "psi": psi,
            "severity": severity(psi), "test": test, "statistic": stat,
            "p_value": p, "change": change,
            "missing_ref": float(ref[c].isna().mean()),
            "missing_cur": float(cur[c].isna().mean()),
            "new_categories": new})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("psi", ascending=False).reset_index(drop=True)


def drift_classifier(ref, cur, columns, types, seed=0):
    """{"auc", "importance"}: how well a model tells the new file from the
    reference (0.5: it cannot), and the columns it leans on. Out-of-fold
    predictions, so the AUC is honest; each file is capped at
    MAX_ROWS_CLASSIFIER rows here only."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    if not columns:
        return None
    rng = np.random.default_rng(seed)

    def cap(df):
        if len(df) <= MAX_ROWS_CLASSIFIER:
            return df
        return df.iloc[np.sort(rng.choice(len(df), MAX_ROWS_CLASSIFIER,
                                          replace=False))]

    a, b = cap(ref[columns]), cap(cur[columns])
    both = pd.concat([a, b], ignore_index=True)
    X = pd.DataFrame(index=both.index)
    for c in columns:
        if types[c] == "numeric":
            X[c] = _numbers(both[c]).fillna(-1e12)
        else:
            X[c] = pd.factorize(_labels(both[c]))[0]
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    model = RandomForestClassifier(n_estimators=150, max_depth=10,
                                   min_samples_leaf=20, n_jobs=-1,
                                   random_state=seed)
    folds = StratifiedKFold(3, shuffle=True, random_state=seed)
    proba = cross_val_predict(model, X, y, cv=folds,
                              method="predict_proba")[:, 1]
    model.fit(X, y)
    importance = pd.Series(model.feature_importances_, index=columns) \
        .sort_values(ascending=False)
    return {"auc": float(roc_auc_score(y, proba)), "importance": importance}


def classifier_reading(auc):
    if auc is None:
        return ""
    if auc < 0.55:
        return "the two files are hard to tell apart: little overall drift"
    if auc < 0.65:
        return "some drift: a model can partly tell the files apart"
    if auc < 0.80:
        return "clear drift: a model tells the files apart well"
    return "strong drift: the new file is easy to tell apart"


def correlations(ref, cur, numeric):
    """(reference matrix, new matrix, pairs) with Spearman correlations;
    `pairs` lists every pair with both values and the change, largest
    change first."""
    if len(numeric) < 2:
        return None, None, pd.DataFrame()
    r = ref[numeric].apply(_numbers).corr(method="spearman")
    n = cur[numeric].apply(_numbers).corr(method="spearman")
    rows = []
    for i, a in enumerate(numeric):
        for b in numeric[i + 1:]:
            ra, na = r.loc[a, b], n.loc[a, b]
            if np.isfinite(ra) and np.isfinite(na):
                rows.append({"first": a, "second": b, "ref_r": float(ra),
                             "cur_r": float(na), "change": float(na - ra)})
    pairs = pd.DataFrame(rows)
    if not pairs.empty:
        pairs = pairs.reindex(pairs["change"].abs()
                              .sort_values(ascending=False).index) \
            .reset_index(drop=True)
    return r, n, pairs


def _target_numbers(y):
    """The target as numbers: itself for a number, 1 for the rarer of two
    classes, None for several classes."""
    v = _numbers(y)
    if v.notna().mean() > 0.9 and v.nunique() > 2:
        return v
    labels = _labels(y)
    vals = labels[labels != MISSING].value_counts()
    if len(vals) != 2:
        return None
    return (labels == vals.index[-1]).astype(float).where(labels != MISSING)


def _eta(cat, y):
    """The correlation ratio: how much of y's spread the categories
    explain, from 0 to 1."""
    ok = y.notna()
    y, cat = y[ok], cat[ok]
    if len(y) < 3 or y.var() == 0:
        return np.nan
    means = y.groupby(cat).transform("mean")
    return float(np.sqrt(((means - y.mean()) ** 2).sum()
                         / ((y - y.mean()) ** 2).sum()))


def target_relationships(ref, cur, columns, types, target):
    """Each feature's relationship with the target in both files: Spearman
    for a number, the correlation ratio for a category. Empty when the
    target is missing from either file or has several classes."""
    if target is None or target not in ref.columns or \
            target not in cur.columns:
        return pd.DataFrame()
    yr, yc = _target_numbers(ref[target]), _target_numbers(cur[target])
    if yr is None or yc is None:
        return pd.DataFrame()
    rows = []
    for c in columns:
        if c == target:
            continue
        if types[c] == "numeric":
            a = _numbers(ref[c]).corr(yr, method="spearman")
            b = _numbers(cur[c]).corr(yc, method="spearman")
            how = "Spearman r"
        else:
            a, b = _eta(_labels(ref[c]), yr), _eta(_labels(cur[c]), yc)
            how = "correlation ratio"
        if np.isfinite(a) and np.isfinite(b):
            rows.append({"feature": c, "measure": how, "ref": float(a),
                         "cur": float(b), "change": float(b - a)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.reindex(out["change"].abs().sort_values(ascending=False)
                       .index).reset_index(drop=True)


def analyse(ref, cur, target=None, prof=None, seed=0):
    """Everything the Drift page shows, computed once."""
    columns, only_ref, only_cur = shared_columns(ref, cur, prof)
    types = column_types(ref, columns)
    features = [c for c in columns if c != target]
    numeric = [c for c in features if types[c] == "numeric"]
    ref_m, cur_m, pairs = correlations(ref, cur, numeric)
    return {
        "columns": columns, "types": types, "only_ref": only_ref,
        "only_cur": only_cur, "rows": (len(ref), len(cur)),
        "covariate": covariate(ref, cur, columns, types),
        "classifier": drift_classifier(ref, cur, features, types, seed),
        "corr_ref": ref_m, "corr_cur": cur_m, "pairs": pairs,
        "target": target,
        "target_rel": target_relationships(ref, cur, features, types, target),
    }
