"""Grouped cross-validation: every match is a test game exactly once, never also trained on."""

import csv

import pandas as pd
import pytest
from make_cv_folds import main, make_folds
from report import main as report_main
from robustness import manifest_for

GAMES = [f"{n:02d}" for n in range(1, 66)]


def test_test_sets_partition_the_matches():
    folds = make_folds(GAMES, k=5, n_val=7, seed=42)
    tested = [g for f in folds for g, s in f.items() if s == "test"]
    assert sorted(tested) == GAMES  # each match tested once
    for f in folds:
        counts = pd.Series(f).value_counts()
        assert counts["test"] == 13 and counts["val"] == 7 and counts["train"] == 45
        assert set(f) == set(GAMES)  # every match has exactly one role per fold


def test_folds_are_reproducible_and_depend_on_the_seed():
    assert make_folds(GAMES, 5, 7, 42) == make_folds(GAMES, 5, 7, 42)
    assert make_folds(GAMES, 5, 7, 42) != make_folds(GAMES, 5, 7, 43)


def test_the_script_writes_valid_manifests(tmp_path):
    base = tmp_path / "games_x.csv"
    with open(base, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["game_num", "match_id", "league", "official_split", "split"])
        w.writerows([g, f"m{g}", "l", "train", "train"] for g in GAMES[:10])
    main(["--manifest", str(base), "--folds", "2", "--val-games", "2"])
    for k in (1, 2):
        rows = list(csv.DictReader(open(tmp_path / f"games_x_cv{k}.csv", encoding="utf-8")))
        assert len(rows) == 10 and sum(r["split"] == "test" for r in rows) == 5
        assert sum(r["split"] == "val" for r in rows) == 2


def test_robustness_finds_the_fold_manifests():
    assert manifest_for("cv3") == "configs/games_epl2122_cv3.csv"
    assert manifest_for("venue") == "configs/games_venue_split.csv"


def write_cv_results(root, overlap=False):
    rows = []
    for k in (1, 2):
        for seed in (1, 2):
            for metric, value in [("pooled_accuracy", 0.3 + 0.1 * k), ("pooled_macro_f1", 0.2),
                                  ("pooled_majority", 0.15), ("sequence_accuracy", 0.45 + 0.1 * k),
                                  ("sequence_macro_f1", 0.3), ("contrastive_heldout_before_ARI", 0.03),
                                  ("contrastive_heldout_after_ARI", 0.09)]:
                rows.append({"split": f"cv{k}", "seed": seed, "metric": metric, "value": value})
            games = ["01", "02"] if k == 1 else (["02", "03"] if overlap else ["03", "04"])
            for kind in ("pooled", "sequence"):
                d = root / "robustness" / f"cv{k}" / (f"pooled_s{seed}" if kind == "pooled" else "sequence")
                d.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"seed": seed, "game": games, "half": "1", "event_tag": "00-10.000",
                              "label": ["a", "b"], "pred": ["a", "a"]}).to_csv(
                    d / ("test_predictions.csv" if kind == "pooled" else "sequence_test_predictions.csv"), index=False)
    (root / "robustness").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "robustness" / "all_runs.csv", index=False)


def test_report_summarises_the_folds(tmp_path):
    write_cv_results(tmp_path)
    report_main(["--results", str(tmp_path), "--frames", str(tmp_path / "none")])
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## Cross-validation over all matches" in text
    assert "| Pooled frozen features | 0.450 ± 0.071 | 0.200 ± 0.000 | n/a | 0.500 (" in text
    assert text.count("| 4 |") == 2  # both models scored all four matches
    assert ") | 0.500 | 4 |" in text  # pooled balanced accuracy: recall of a = 1.0, of b = 0.0
    assert "sequence ahead in 2 of 2 folds" in text and "up in 2 of 2 folds" in text
    assert "match-level split" not in text  # folds are not repeated as ordinary splits


def test_report_refuses_a_match_tested_in_two_folds(tmp_path):
    write_cv_results(tmp_path, overlap=True)
    with pytest.raises(ValueError, match="more than one fold"):
        report_main(["--results", str(tmp_path), "--frames", str(tmp_path / "none")])


def test_balanced_accuracy_is_mean_recall_over_the_classes_present():
    from soccer.metrics import classification_scores

    y_true = ["a"] * 8 + ["b"] * 2
    y_pred = ["a"] * 8 + ["a", "b"]  # everything right except one of the two b events
    scores = classification_scores(y_true, y_pred, y_train=["a"] * 10)
    assert scores["Accuracy"] == pytest.approx(0.9)
    assert scores["Balanced_accuracy"] == pytest.approx((1.0 + 0.5) / 2)  # accuracy flatters the common class


def test_label_summary_documents_exclusions_and_counts():
    from label_summary import label_table, to_markdown

    from soccer.manifest import Game

    manifest = [Game("01", "m1", "l", "train", "train"), Game("02", "m2", "l", "valid", "val"),
                Game("03", "m3", "l", "test", "test")]
    labels = ["corner"] * 5 + ["goal"] * 2 + ["unknown", "show_added_time", "goal"]
    games = ["01", "01", "02", "03", "03", "01", "02", "03", "01", "02"]
    table = label_table(labels, games, manifest, {"junk": ["unknown"], "non_visual": ["show_added_time"]})
    rows = table.set_index("label")
    assert rows.loc["corner", ["train", "val", "test", "total"]].tolist() == [2, 1, 2, 5]
    assert rows.loc["goal", ["train", "val", "test", "total"]].tolist() == [1, 2, 0, 3]  # 3 events, none in test
    assert rows.loc["goal", "note"] == "fewer than 5 events" and rows.loc["corner", "note"] == ""
    assert (rows.loc["unknown", "status"], rows.loc["unknown", "reason"]) == ("excluded", "junk")
    assert rows.loc["show_added_time", "reason"] == "non-visual"
    text = to_markdown(table)
    assert "2 labels kept (8 events" in text and "2 excluded (2 events)" in text
