"""The Configure & train assistant: the user's objective, in plain words, in;
a recommendation (what to learn, the target, the models and their settings,
the metric to judge them by, validation and preparation) and its reasons out.

It sees the file's profile, the model catalogue with every setting's allowed
values, and the settings on screen, so a proposal names only things the page
can set. Nothing changes until the user applies it (`assistant_panel`), and
every value is checked there against the page's own options.
"""
import pandas as pd

from logic import claude, ml

MAX_COLUMNS = 80

_SYSTEM = """You are the machine-learning assistant inside an app where a user trains and compares models on a tabular file. The user describes their business objective in plain language; you turn it into a training configuration on the page and explain your reasoning briefly and concretely.

How to work:
- Decide what to learn: classification (predict a category, e.g. will a customer churn), regression (predict a number) or clustering (find groups, no target).
- Choose the target column from the candidates listed for that task. If the objective is ambiguous (several plausible targets, or unclear whether they want to predict or segment), ask one short clarifying question and make no proposal yet.
- Recommend 2 to 4 models from the catalogue that suit the data size, column types and objective, always including a fast interpretable one and a strong one (gradient boosting, XGBoost or LightGBM when available). Set a model's settings only where you have a reason; otherwise leave them at the defaults.
- Choose the metric to pick the best model by, matched to the objective: e.g. PR AUC or recall when the positive class is rare and missing it is costly, ROC AUC for ranking who to target, log loss when calibrated probabilities matter, MAE when errors cost in proportion, RMSE when large errors are especially bad.
- Recommend validation (cross-validation folds, test share), class balancing for imbalanced targets, and preparation: missing-value fills, scaling, categorical encoding (one-hot by default; target encoding for columns with many categories; ordinal only for trees), and hyperparameter search ("random" when there is time to tune).
- Warn about columns that could leak the answer (recorded after the outcome or derived from the target) if you see any.
- Use only the keys and values listed in the context. Leave a proposal field as an empty string to keep what is on the page.
- Keep the reply short: a few sentences or bullets, plain language, no tables. Explain the why behind the metric and the models."""

_FIELD = {"type": "string"}

_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string",
                  "description": "What you say to the user (Markdown)."},
        "has_proposal": {"type": "boolean",
                         "description": "True when `proposal` holds settings "
                                        "to apply; false when asking a "
                                        "question or just answering."},
        "proposal": {
            "type": "object",
            "properties": {
                "task": _FIELD, "target": _FIELD, "positive_class": _FIELD,
                "metric": _FIELD, "folds": _FIELD,
                "test_share_percent": _FIELD, "balance_classes": _FIELD,
                "num_missing": _FIELD, "cat_missing": _FIELD,
                "scale": _FIELD, "cat_encoding": _FIELD, "tune_mode": _FIELD,
                "models": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": _FIELD,
                            "settings": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"name": _FIELD,
                                                   "value": _FIELD},
                                    "required": ["name", "value"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["key", "settings"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["task", "target", "positive_class", "metric",
                         "folds", "test_share_percent", "balance_classes",
                         "num_missing", "cat_missing", "scale",
                         "cat_encoding", "tune_mode", "models"],
            "additionalProperties": False,
        },
    },
    "required": ["reply", "has_proposal", "proposal"],
    "additionalProperties": False,
}


def _columns(df, prof):
    lines = []
    for r in prof.head(MAX_COLUMNS).itertuples():
        bits = [r.type, f"{r.unique:,} distinct"]
        if r.missing:
            bits.append(f"{r.missing / len(df):.0%} missing")
        if r.type == "categorical" and r.unique <= 6:
            shares = df[r.column].astype(str).value_counts(normalize=True)
            bits.append(", ".join(f"{k} {v:.0%}" for k, v in
                                  shares.head(6).items()))
        elif r.type == "numeric":
            v = pd.to_numeric(df[r.column], errors="coerce")
            bits.append(f"{v.min():,.4g} to {v.max():,.4g}, median "
                        f"{v.median():,.4g}")
        if not r.use:
            bits.append(f"off by default: {r.note}")
        lines.append(f"- {r.column}: " + "; ".join(bits))
    if len(prof) > MAX_COLUMNS:
        lines.append(f"- …and {len(prof) - MAX_COLUMNS} more columns")
    return "\n".join(lines)


def _catalogue(task, n_rows):
    blocked = ml.unavailable(task, n_rows)
    lines = []
    for key, m in ml.MODELS[task].items():
        if key in blocked:
            continue
        params = "; ".join(
            f"{p['name']} ({p['label']}): "
            + ", ".join(ml.show_value(p, v) for v in p["values"])
            + f" [default {ml.show_value(p, p['default'])}]"
            for p in m["params"])
        lines.append(f"  - {key}: {m['label']}. {m['about']} Settings: "
                     f"{params or 'none'}")
    return "\n".join(lines)


def system_prompt(df, prof, options) -> str:
    """The instructions plus everything stable about this file: its
    columns, the target candidates, the models and the page's options.
    `options` maps each preparation/validation field to its allowed keys."""
    n = len(df)
    parts = [_SYSTEM, f"\n\n# The file\n{n:,} rows × {df.shape[1]} columns."
             f"\n{_columns(df, prof)}"]
    for task in ml.TASKS:
        cands = ml.target_candidates(prof, task) if task != "clustering" \
            else []
        parts.append(f"\n\n# {ml.TASK_LABELS[task]}"
                     + (f"\nTarget candidates: {', '.join(cands)}"
                        if task != "clustering" else
                        "\nNo target: groups rows by their features.")
                     + f"\nModels (key: label. about. settings with allowed "
                       f"values):\n{_catalogue(task, n)}")
    metrics = "\n".join(
        f"- {family}: " + ", ".join(f"{k} ({ml.metric_label(family, k)})"
                                    for k in keys)
        for family, keys in ml.SELECTABLE.items())
    parts.append("\n\n# Metrics by family (binary and multiclass are "
                 f"classification)\n{metrics}")
    opts = "\n".join(f"- {field}: {', '.join(map(str, keys))}"
                     for field, keys in options.items())
    parts.append(f"\n\n# Other settings and their allowed values\n{opts}\n"
                 "- balance_classes: yes, no")
    return "".join(parts)


def turn(system, history, user_text, current, *, api_key=None,
         client=None) -> dict:
    """One exchange: {reply, has_proposal, proposal}. `history` is the
    earlier exchanges as Messages API turns; `current` describes the
    settings on screen now."""
    messages = list(history) + [{
        "role": "user",
        "content": f"{user_text}\n\n(Settings on the page now: {current})"}]
    return claude.ask(system, messages, _SCHEMA, api_key=api_key,
                      client=client, effort="medium")
