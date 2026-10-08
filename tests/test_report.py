import pandas as pd
import pytest
from report import main, paired


def write_results(root):
    (root / "dinov2-base").mkdir(parents=True)
    pd.DataFrame([{"Representation": "Event mean-pool", "Purity": 0.23, "Purity_random": 0.17, "ARI": 0.03,
                   "Game_ARI": 0.19}]).to_csv(root / "dinov2-base" / "clustering_metrics.csv", index=False)
    (root / "within_game").mkdir()
    pd.DataFrame({"ARI": [0.07, 0.08], "Purity": [0.44, 0.45], "Purity_random": [0.37, 0.38]}).to_csv(
        root / "within_game" / "within_game_metrics.csv", index=False)
    rows = []
    for seed, (before, after, pooled, seq) in {1: (0.02, 0.10, 0.30, 0.40), 2: (0.03, 0.08, 0.32, 0.38)}.items():
        rows += [
            {"split": "default", "seed": seed, "metric": "contrastive_heldout_before_ARI", "value": before},
            {"split": "default", "seed": seed, "metric": "contrastive_heldout_after_ARI", "value": after},
            {"split": "default", "seed": seed, "metric": "pooled_accuracy", "value": pooled},
            {"split": "default", "seed": seed, "metric": "pooled_macro_f1", "value": 0.25},
            {"split": "default", "seed": seed, "metric": "pooled_majority", "value": 0.16},
            {"split": "default", "seed": seed, "metric": "sequence_accuracy", "value": seq},
            {"split": "default", "seed": seed, "metric": "sequence_macro_f1", "value": 0.28},
        ]
    (root / "robustness" / "default" / "pooled_s1").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(root / "robustness" / "all_runs.csv", index=False)
    (root / "robustness" / "default" / "pooled_s1" / "classification_report.txt").write_text(
        "Test accuracy 0.3000  macro-F1 0.2500  majority-class baseline 0.1600\n"
        "95% interval, resampling test games: accuracy 0.250-0.350, macro-F1 0.200-0.300\n")
    return pd.DataFrame(rows)


def test_paired_differences_are_computed_within_seeds(tmp_path):
    df = write_results(tmp_path)
    mean, sd, n, up = paired(df, "default", "contrastive_heldout_after_ARI", "contrastive_heldout_before_ARI")
    assert mean == pytest.approx((0.08 + 0.05) / 2) and n == 2 and up == 2
    mean, _, _, up = paired(df, "default", "sequence_accuracy", "pooled_accuracy")
    assert mean == pytest.approx(0.08) and up == 2


def test_report_contains_every_section(tmp_path):
    write_results(tmp_path)
    main(["--results", str(tmp_path)])
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "| Event mean-pool | 0.030 | 0.190 |" in text
    assert "+0.065" in text and "(2/2 seeds up)" in text          # contrastive change, paired
    assert "0.250-0.350" in text                                   # game-resampled interval
    assert "sequence minus pooled accuracy +0.080" in text
