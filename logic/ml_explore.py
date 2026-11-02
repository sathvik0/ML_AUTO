"""What the Explore & prepare page shows, computed over every row.

Pure functions of a frame and its profile (`ml.profile`), so each number on
the page can be checked a second way. Nothing here samples: the page only
thins what it draws.
"""
import numpy as np
import pandas as pd

from logic import ml, ml_recipe

TOP_LEVELS = 30          # categories drawn one bar each; the rest are summed
MAX_CORR_COLUMNS = 40    # a heatmap past this is unreadable
TARGET_LEVELS = 20       # a categorical target with more is not a class
BLANK = "(blank)"


def _numbers(series):
    v = pd.to_numeric(series, errors="coerce").astype("float64")
    return v.where(np.isfinite(v))


def numeric_columns(prof):
    return list(prof.loc[prof["type"] == "numeric", "column"])


# Columns whose spread or ranking means nothing: an NPI's outliers, a ZIP
# code's correlation, a constant's strength against the target.
UNMEASURED = ("id", "postal", "constant")


def measured_numeric(prof):
    """Numeric columns worth measuring: identifiers, postal codes and
    constants left out."""
    return list(prof.loc[(prof["type"] == "numeric")
                         & ~prof["kind"].isin(UNMEASURED), "column"])


def categorical_columns(prof):
    return list(prof.loc[prof["type"] == "categorical", "column"])


def overview(df, prof) -> dict:
    """The file at a glance. A gap is a missing cell, or an infinite number,
    which every model reads as missing too."""
    gaps = 0
    rows_with_gap = pd.Series(False, index=df.index)
    numeric = set(numeric_columns(prof))
    for c in df.columns:
        miss = _numbers(df[c]).isna() if c in numeric else df[c].isna()
        gaps += int(miss.sum())
        rows_with_gap |= miss
    cells = df.shape[0] * df.shape[1]
    return {"rows": len(df), "columns": df.shape[1],
            "numeric": len(numeric), "categorical": df.shape[1] - len(numeric),
            "gaps": gaps, "gap_share": gaps / cells if cells else 0.0,
            "rows_with_gap": int(rows_with_gap.sum()),
            "duplicate_rows": int(df.duplicated(keep="first").sum()),
            "memory_mb": float(df.memory_usage(deep=True).sum() / 1e6)}


def column_summary(df, prof) -> pd.DataFrame:
    """One row per column: gaps, and the spread of a number or the commonest
    value of a category."""
    rows = []
    kinds = prof.set_index("column")["type"]
    n = len(df)
    for c in df.columns:
        s = df[c]
        row = {"column": c, "type": kinds.get(c, "categorical"),
               "distinct": int(s.nunique(dropna=True))}
        if row["type"] == "numeric":
            v = _numbers(s)
            ok = v.dropna()
            row["gaps"] = int(v.isna().sum())
            if len(ok):
                q1, med, q3 = np.quantile(ok, [0.25, 0.5, 0.75])
                row.update(mean=float(ok.mean()),
                           sd=float(ok.std()) if len(ok) > 1 else 0.0,
                           min=float(ok.min()), q1=float(q1),
                           median=float(med), q3=float(q3),
                           max=float(ok.max()),
                           zeros=int((ok == 0).sum()),
                           skew=float(ok.skew()) if len(ok) > 2 else None)
        else:
            text = ml_recipe.label_text(s)
            row["gaps"] = int(text.isna().sum())
            counts = text.value_counts()
            if len(counts):
                row["top"] = str(counts.index[0])
                row["top_share"] = float(counts.iloc[0] / n) if n else 0.0
        row["gap_share"] = row["gaps"] / n if n else 0.0
        rows.append(row)
    cols = ["column", "type", "distinct", "gaps", "gap_share", "mean", "sd",
            "min", "q1", "median", "q3", "max", "zeros", "skew", "top",
            "top_share"]
    return pd.DataFrame(rows).reindex(columns=cols)


def value_counts(series, top=TOP_LEVELS) -> pd.DataFrame:
    """rows per value, commonest first; past `top`, the rest in one row, so
    the counts always add up to the file."""
    text = ml_recipe.label_text(series).fillna(BLANK)
    counts = text.value_counts()
    out = pd.DataFrame({"value": counts.index.astype(str),
                        "Rows": counts.to_numpy()})
    if len(out) > top:
        rest = out.iloc[top:]
        out = pd.concat([out.iloc[:top], pd.DataFrame(
            {"value": [f"{len(rest):,} other values"],
             "Rows": [int(rest["Rows"].sum())]})], ignore_index=True)
    out["share"] = out["Rows"] / max(len(series), 1)
    return out


def gaps_table(df, prof) -> pd.DataFrame:
    """Columns with gaps, most first."""
    numeric = set(numeric_columns(prof))
    rows = []
    for c in df.columns:
        miss = int((_numbers(df[c]).isna() if c in numeric
                    else df[c].isna()).sum())
        if miss:
            rows.append({"column": c, "gaps": miss,
                         "share": miss / max(len(df), 1)})
    out = pd.DataFrame(rows, columns=["column", "gaps", "share"])
    return out.sort_values("gaps", ascending=False).reset_index(drop=True)


def correlations(df, columns, method="pearson") -> pd.DataFrame:
    """Pairwise correlation over the rows where both values exist. Spearman
    ranks first, so one extreme value cannot drive it."""
    data = pd.DataFrame({c: _numbers(df[c]) for c in columns})
    return data.corr(method=method)


def strongest_pairs(corr, top=15) -> pd.DataFrame:
    """The column pairs that move together most, either way."""
    cols = list(corr.columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r):
                rows.append({"first": a, "second": b, "correlation": float(r)})
    out = pd.DataFrame(rows, columns=["first", "second", "correlation"])
    if not len(out):
        return out
    order = out["correlation"].abs().sort_values(ascending=False).index
    return out.loc[order].head(top).reset_index(drop=True)


def target_kind(df, prof, target):
    """"numeric" for a quantity to predict, "categorical" for classes, None
    for a column that is neither (one value, or more levels than classes)."""
    s = df[target]
    distinct = s.nunique(dropna=True)
    if distinct < 2:
        return None
    kind = prof.set_index("column")["type"].get(target)
    if kind == "numeric" and distinct > TARGET_LEVELS:
        return "numeric"
    if distinct <= TARGET_LEVELS:
        return "categorical"
    return None


def _auc(score, hit):
    """Mann-Whitney AUC of a score against a yes/no, gaps left out."""
    ok = ~np.isnan(score)
    score, hit = score[ok], hit[ok]
    n1, n0 = int(hit.sum()), int((~hit).sum())
    if not n1 or not n0:
        return None
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[hit].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _eta_squared(values, groups):
    """Share of a number's variation that lies between group averages."""
    ok = ~np.isnan(values) & pd.notna(groups)
    v, g = values[ok], np.asarray(groups)[ok]
    if len(v) < 2:
        return None
    total = float(((v - v.mean()) ** 2).sum())
    if total == 0:
        return 0.0
    means = pd.Series(v).groupby(g).transform("mean").to_numpy()
    return 1 - float(((v - means) ** 2).sum()) / total


def _cramers_v(a, b):
    table = pd.crosstab(a, b).to_numpy(dtype="float64")
    n = table.sum()
    r, k = table.shape
    if n == 0 or min(r, k) < 2:
        return None
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / n
    chi2 = float(((table - expected) ** 2
                  / np.where(expected > 0, expected, 1)).sum())
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def target_strength(df, prof, target, positive=None) -> pd.DataFrame:
    """How strongly each column goes with the target, on one 0-to-1 scale
    per kind of pair, strongest first.

    Number against two classes: |2 × AUC − 1|, how well the number alone
    ranks the class of interest. Number against more classes, or a category
    against a number: η², the share of variation between group averages.
    Category against classes: Cramér's V. Number against number: |Spearman ρ|.
    """
    kind = target_kind(df, prof, target)
    if kind is None:
        return pd.DataFrame(columns=["feature", "measure", "strength"])
    types = prof.set_index("column")["type"]
    kinds = prof.set_index("column")["kind"]
    skip = {c for c in df.columns if kinds.get(c) in UNMEASURED}
    rows = []
    if kind == "categorical":
        y = ml_recipe.label_text(df[target])
        classes = sorted(y.dropna().unique(), key=str)
        binary = len(classes) == 2
        pos = positive if positive in classes else (classes[-1] if classes
                                                   else None)
        for c in df.columns:
            if c == target or c in skip:
                continue
            if types.get(c) == "numeric":
                v = _numbers(df[c]).to_numpy()
                if binary:
                    keep = y.notna().to_numpy()
                    auc = _auc(v[keep], (y[keep] == pos).to_numpy())
                    if auc is not None:
                        rows.append({"feature": c, "measure": "|2·AUC − 1|",
                                     "strength": abs(2 * auc - 1)})
                else:
                    eta = _eta_squared(v, y.to_numpy())
                    if eta is not None:
                        rows.append({"feature": c, "measure": "η²",
                                     "strength": eta})
            else:
                text = ml_recipe.label_text(df[c]).fillna(BLANK)
                keep = y.notna()
                v = _cramers_v(text[keep], y[keep])
                if v is not None:
                    rows.append({"feature": c, "measure": "Cramér's V",
                                 "strength": v})
    else:
        t = _numbers(df[target])
        for c in df.columns:
            if c == target or c in skip:
                continue
            if types.get(c) == "numeric":
                r = t.corr(_numbers(df[c]), method="spearman")
                if pd.notna(r):
                    rows.append({"feature": c, "measure": "|Spearman ρ|",
                                 "strength": abs(float(r))})
            else:
                # a blank is a group of its own, as everywhere on the page
                text = ml_recipe.label_text(df[c]).fillna(BLANK).to_numpy()
                eta = _eta_squared(t.to_numpy(), text)
                if eta is not None:
                    rows.append({"feature": c, "measure": "η²",
                                 "strength": eta})
    out = pd.DataFrame(rows, columns=["feature", "measure", "strength"])
    return out.sort_values("strength", ascending=False).reset_index(drop=True)


def target_by_value(df, target, feature, kind, positive=None, top=TOP_LEVELS,
                    bands=10) -> pd.DataFrame:
    """The target across one column's values: per category (the `top`
    commonest), or per equal-count band of a number, in order. Classes give
    the share of the class of interest; a numeric target gives its average.
    Rows with no target are left out; rows with no value form "(blank)".
    """
    if kind == "categorical":
        y = ml_recipe.label_text(df[target])
        keep = y.notna()
        value = (y == positive).astype(float)
    else:
        value = _numbers(df[target])
        keep = value.notna()
    f = df[feature]
    banded = (pd.api.types.is_numeric_dtype(f)
              and f.nunique(dropna=True) > top)
    if banded:
        v = _numbers(f)
        ok = v.notna()
        edges = np.unique(np.quantile(v[ok], np.linspace(0, 1, bands + 1))) \
            if ok.any() else np.array([])
        if len(edges) >= 2:
            band = pd.cut(v, edges, include_lowest=True)
            frame = pd.DataFrame({"band": band, "y": value})[keep & ok]
            out = (frame.groupby("band", observed=True)
                   .agg(Rows=("y", "size"), target=("y", "mean"))
                   .reset_index())
            out["value"] = [f"{i.left:,.6g} to {i.right:,.6g}"
                            for i in out["band"]]
            out = out[["value", "Rows", "target"]]
        else:
            out = pd.DataFrame(columns=["value", "Rows", "target"])
        blank = keep & ~ok
        if blank.any():
            out = pd.concat([out, pd.DataFrame({
                "value": [BLANK], "Rows": [int(blank.sum())],
                "target": [float(value[blank].mean())]})], ignore_index=True)
        return out.reset_index(drop=True)
    group = ml_recipe.label_text(f).fillna(BLANK)
    out = (pd.DataFrame({"value": group[keep].to_numpy(),
                         "y": value[keep].to_numpy()})
           .groupby("value").agg(Rows=("y", "size"), target=("y", "mean"))
           .reset_index())
    return out.sort_values("Rows", ascending=False).head(top) \
        .reset_index(drop=True)


def outliers(df, columns, k=1.5) -> pd.DataFrame:
    """Per number: the quartile fences, and how many values lie beyond
    them or more than three standard deviations from the mean."""
    rows = []
    for c in columns:
        v = _numbers(df[c]).dropna()
        if not len(v):
            continue
        q1, q3 = np.quantile(v, [0.25, 0.75])
        lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        sd = v.std() if len(v) > 1 else 0.0
        z = int((np.abs(v - v.mean()) > 3 * sd).sum()) if sd > 0 else 0
        below, above = int((v < lo).sum()), int((v > hi).sum())
        rows.append({"column": c, "low_fence": float(lo),
                     "high_fence": float(hi), "below": below, "above": above,
                     "share": (below + above) / len(v), "beyond_3_sd": z})
    out = pd.DataFrame(rows, columns=["column", "low_fence", "high_fence",
                                      "below", "above", "share",
                                      "beyond_3_sd"])
    return out.sort_values("share", ascending=False).reset_index(drop=True)


def five_numbers(values) -> dict:
    """Quartiles and the furthest values inside the 1.5 × IQR fences: what a
    box draws, computed over every value."""
    v = _numbers(pd.Series(values)).dropna().to_numpy()
    if not len(v):
        return {}
    q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    inside = v[(v >= lo) & (v <= hi)]
    return {"q1": float(q1), "median": float(med), "q3": float(q3),
            "lowerfence": float(inside.min()) if len(inside) else float(q1),
            "upperfence": float(inside.max()) if len(inside) else float(q3),
            "mean": float(v.mean())}


def candidate_targets(prof) -> list:
    """Columns worth exploring a target against: the configure page's own
    proposals for both tasks, then everything else that is not an id."""
    first = (ml.target_candidates(prof, "classification")
             + ml.target_candidates(prof, "regression"))
    seen = list(dict.fromkeys(first))
    kinds = prof.set_index("column")["kind"]
    rest = [c for c in prof["column"] if c not in seen
            and kinds[c] not in ("id", "constant")]
    return seen + rest
