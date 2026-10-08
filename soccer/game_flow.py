"""Annotation-only "game flow" features and their leak-safe encoding.

Each event is described by what happened before it in the same match: the
previous three event types, rolling counts over the last five minutes, the
half and the minute. Only past events are used, and they are the true labels
of those events (teacher forcing), so scores built on them are upper bounds
for a system that would have to predict the earlier events itself.

Categorical columns are encoded inside each model's pipeline, so the encoder
is fit on training rows only (docs/leak_audit.md, B11).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

NUMERIC = ["half", "minute", "recent_corners", "recent_shots", "recent_fouls",
           "recent_cards", "recent_attacks", "recent_stoppages"]
CATEGORICAL = ["prev_1", "prev_2", "prev_3", "prev_1_cat", "prev_2_cat", "prev_3_cat"]
FEATURES = NUMERIC + CATEGORICAL

# Never allowed as model inputs: they describe the future or the answer.
FORBIDDEN = {"label", "category", "next_event", "n_next_attacks", "text"}


def check_columns(columns) -> None:
    bad = FORBIDDEN & set(columns)
    if bad:
        raise ValueError(f"Columns {sorted(bad)} leak the label or the future and cannot be features")


def encoder(columns, one_hot: bool) -> ColumnTransformer:
    """Encode the categorical columns among `columns`; scale numeric ones.

    One-hot for linear and neural models, ordinal codes for tree models. Unseen
    categories at prediction time are tolerated.
    """
    check_columns(columns)
    cat = [c for c in columns if c in CATEGORICAL]
    num = [c for c in columns if c not in CATEGORICAL]
    cat_enc = (OneHotEncoder(handle_unknown="ignore") if one_hot
               else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    return ColumnTransformer([("cat", cat_enc, cat), ("num", StandardScaler(), num)])
