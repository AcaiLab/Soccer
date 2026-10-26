import pandas as pd
import pytest
from per_class import per_class_scores, per_class_table, to_markdown


def test_scores_are_averaged_over_seeds():
    preds = pd.DataFrame({"seed": [1, 1, 1, 2, 2, 2],
                          "label": ["goal", "goal", "card", "goal", "goal", "card"],
                          "pred": ["goal", "goal", "card", "goal", "card", "card"]})
    s = per_class_scores(preds)
    assert s.loc["goal", "recall"] == pytest.approx(0.75)  # 1.0 in seed 1, 0.5 in seed 2
    assert s.loc["card", "recall"] == pytest.approx(1.0)


def test_table_keeps_only_kept_labels_rarest_first():
    counts = pd.DataFrame({"label": ["goal", "card", "logo"], "status": ["kept", "kept", "excluded"],
                           "train": [50, 3, 9], "val": [5, 1, 1], "test": [6, 0, 1]})
    scores = pd.DataFrame({"precision": [1.0], "recall": [0.5], "f1": [0.667]}, index=pd.Index(["goal"], name="label"))
    table = per_class_table(counts, {"sequence": scores})
    assert list(table.label) == ["card", "goal"]
    text = to_markdown(table, ["sequence"])
    assert "| card | 3 | 1 | 0 | n/a |" in text and "| goal | 50 | 5 | 6 | 0.667 |" in text
