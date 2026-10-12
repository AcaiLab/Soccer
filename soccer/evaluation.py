"""Train / validate / test protocol shared by the event classifiers.

Splits are by game, from the manifest. Every candidate configuration is fit on
training games and compared on validation games; only the chosen configuration
is then scored on the test games, once. Choosing the best of many
configurations by test score would make the reported number optimistic
(docs/leak_audit.md, B5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from soccer.manifest import Game, assert_disjoint, game_nums, normalize_game_id
from soccer.metrics import bootstrap_ci, classification_scores


@dataclass
class Split:
    train: np.ndarray  # boolean masks over events
    val: np.ndarray
    test: np.ndarray
    test_games: list[str]


def split_by_game(games: np.ndarray, labels: np.ndarray, manifest: list[Game],
                  val_games=None, test_games=None) -> Split:
    """Masks for train / val / test events.

    Defaults come from the manifest; explicit game lists override them. Only
    manifest games are used: events of other games (e.g. extra games present
    in the features) belong to no split, rather than silently becoming
    training data. Test events whose label never occurs in training are
    dropped, since no classifier could predict them.
    """
    test_games = sorted(normalize_game_id(g) for g in (test_games or game_nums(manifest, "test")))
    val_games = sorted(normalize_game_id(g) for g in (val_games or game_nums(manifest, "val")))
    assert_disjoint(val=val_games, test=test_games)
    in_manifest = {g.game_num for g in manifest} | set(val_games) | set(test_games)
    test = np.isin(games, test_games)
    val = np.isin(games, val_games)
    train = np.isin(games, list(in_manifest)) & ~(test | val)
    outside = int((~np.isin(games, list(in_manifest))).sum())
    if outside:
        print(f"  {outside} events from games outside the manifest are not used")
    seen = set(labels[train])
    known = np.isin(labels, list(seen))
    dropped = int((test & ~known).sum())
    if dropped:
        print(f"  {dropped} test events have labels absent from training and are left out")
    assert_disjoint(train=set(games[train]), val=set(games[val]), test=set(games[test]))
    return Split(train, val & known, test & known, test_games)


def sklearn_classifiers(seed: int) -> dict:
    return {
        "LinearSVM": Pipeline([("scaler", StandardScaler()),
                               ("clf", LinearSVC(max_iter=5000, random_state=seed, C=1.0))]),
        "LogReg": Pipeline([("scaler", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=2000, random_state=seed, C=1.0))]),
        "MLP": Pipeline([("scaler", StandardScaler()),
                         ("clf", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=seed,
                                               early_stopping=True, validation_fraction=0.1))]),
    }


def select_on_validation(feature_sets: dict[str, np.ndarray], labels: np.ndarray, split: Split,
                         seed: int, games: np.ndarray | None = None) -> tuple[list[dict], dict]:
    """Fit every (features, classifier) pair on train, rank on validation, test the winner.

    Returns (validation rows for every candidate, result for the chosen one with
    test metrics, predictions and, if `games` is given, 95% intervals from
    resampling test games).
    """
    rows, best = [], None
    for feat_name, X in feature_sets.items():
        for clf_name, clf in sklearn_classifiers(seed).items():
            clf.fit(X[split.train], labels[split.train])
            val_pred = clf.predict(X[split.val])
            scores = classification_scores(labels[split.val], val_pred, labels[split.train])
            row = {"Features": feat_name, "Classifier": clf_name,
                   "Val_accuracy": scores["Accuracy"], "Val_macro_F1": scores["Macro_F1"]}
            rows.append(row)
            print(f"  {feat_name:<34s} {clf_name:<10s} val acc={scores['Accuracy']:.3f}  "
                  f"macro-F1={scores['Macro_F1']:.3f}")
            if best is None or row["Val_accuracy"] > best["row"]["Val_accuracy"]:
                best = {"row": row, "clf": clf, "X": X}

    test_pred = best["clf"].predict(best["X"][split.test])
    test_scores = classification_scores(labels[split.test], test_pred, labels[split.train])
    chosen = {**best["row"], **{f"Test_{k}": v for k, v in test_scores.items()},
              "y_true": labels[split.test], "y_pred": test_pred}
    if games is not None:
        chosen.update({f"Test_{k}": v for k, v in
                       bootstrap_ci(labels[split.test], test_pred, games[split.test], seed=seed).items()})
    return rows, chosen


def prediction_rows(games, halves, event_tags, y_true, y_pred, seed=None) -> pd.DataFrame:
    """One row per test event (game, half, event tag, true label, prediction), so results can be
    pooled across runs and folds and errors inspected event by event."""
    df = pd.DataFrame({"game": np.asarray(games).astype(str), "half": np.asarray(halves).astype(str),
                       "event_tag": np.asarray(event_tags).astype(str), "label": np.asarray(y_true),
                       "pred": np.asarray(y_pred)})
    if seed is not None:
        df.insert(0, "seed", seed)
    return df


def save_test_predictions(path, games, halves, event_tags, y_true, y_pred, seed=None) -> None:
    prediction_rows(games, halves, event_tags, y_true, y_pred, seed).to_csv(path, index=False)


def format_test_summary(chosen: dict, split: Split, n_events: int, kind: str = "events") -> str:
    """The lines every classifier report starts with."""
    ci = ""
    if "Test_Accuracy_CI" in chosen:
        (alo, ahi), (flo, fhi) = chosen["Test_Accuracy_CI"], chosen["Test_Macro_F1_CI"]
        ci = (f"95% interval, resampling test games: accuracy {alo:.3f}-{ahi:.3f}, "
              f"macro-F1 {flo:.3f}-{fhi:.3f}\n")
    return (f"Chosen on validation: {chosen['Features']} + {chosen['Classifier']}\n"
            f"Test games: {' '.join(split.test_games)}  ({n_events} {kind})\n"
            f"Test accuracy {chosen['Test_Accuracy']:.4f}  macro-F1 {chosen['Test_Macro_F1']:.4f}  "
            f"majority-class baseline {chosen['Test_Majority_baseline']:.4f}\n" + ci)
