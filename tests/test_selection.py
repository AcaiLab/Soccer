import numpy as np

from soccer import evaluation
from soccer.config import REPO_ROOT, encoder_label, load_config
from soccer.evaluation import Split, select_on_validation


class FixedPredictions:
    """A 'classifier' that returns a preset label for each event index in X."""

    def __init__(self, preds):
        self.preds = np.asarray(preds)

    def fit(self, X, y):
        return self

    def predict(self, X):
        return self.preds[X[:, 0].astype(int)]


def test_selection_follows_macro_f1_not_accuracy(monkeypatch):
    # 4 training events, then 10 validation events (8 "a", 2 "b"), then 2 test events.
    labels = np.array(["a", "a", "b", "b"] + ["a"] * 8 + ["b"] * 2 + ["a", "b"])
    always_a = ["a"] * len(labels)  # validation: accuracy 0.8, macro-F1 0.44
    finds_b = list(labels)
    finds_b[4:7] = ["b"] * 3  # validation: accuracy 0.7, macro-F1 0.67
    monkeypatch.setattr(evaluation, "sklearn_classifiers",
                        lambda seed: {"AlwaysA": FixedPredictions(always_a), "FindsB": FixedPredictions(finds_b)})
    idx = np.arange(len(labels))
    split = Split(train=idx < 4, val=(idx >= 4) & (idx < 14), test=idx >= 14, test_games=["99"])

    rows, chosen = select_on_validation({"features": idx[:, None].astype(float)}, labels, split, seed=0)

    by_name = {r["Classifier"]: r for r in rows}
    assert by_name["AlwaysA"]["Val_accuracy"] > by_name["FindsB"]["Val_accuracy"]
    assert chosen["Classifier"] == "FindsB"


def test_accuracy_breaks_macro_f1_ties():
    rows = [{"Val_macro_F1": 0.4, "Val_accuracy": 0.5}, {"Val_macro_F1": 0.4, "Val_accuracy": 0.6}]
    assert max(rows, key=evaluation.selection_key)["Val_accuracy"] == 0.6


def test_encoder_label_names_the_configured_encoder():
    assert encoder_label(load_config(REPO_ROOT / "configs" / "pipeline.yaml")) == "DINOv2"
    assert encoder_label(load_config(REPO_ROOT / "configs" / "pipeline_matchvision.yaml")) == "MatchVision"
