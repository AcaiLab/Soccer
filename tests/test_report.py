import json

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
    main(["--results", str(tmp_path), "--frames", str(tmp_path / "no_frames_here")])
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "| Event mean-pool | 0.030 | 0.190 |" in text
    assert "+0.065" in text and "(2/2 seeds up)" in text          # contrastive change, paired
    assert "0.250-0.350" in text                                   # game-resampled interval
    assert "sequence minus pooled accuracy +0.080" in text
    assert "Frame timing" not in text                              # no frames, no timing section


def test_report_states_the_frame_timing_check(tmp_path, monkeypatch):
    monkeypatch.delenv("SOCCER_MANIFEST", raising=False)  # default manifest: games 01-30
    write_results(tmp_path)
    frames = tmp_path / "frames"
    for name, error in [("game_01_half_1", None), ("game_01_half_2", 0.0), ("game_02_half_1", 45.08)]:
        (frames / name / "frames").mkdir(parents=True)
        info = {"events": 30} if error is None else {"events": 30, "max_time_error_sec": error}
        (frames / name / "extraction.json").write_text(json.dumps(info))
    main(["--results", str(tmp_path), "--frames", str(frames)])
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## Frame timing check" in text
    assert ("2 of 3 halves have a frame-timing check (1 extracted before the check existed); 1 above the "
            "tolerance, holding 30 of 90 events: game_02_half_1 45.1s.") in text


def test_report_shows_a_single_run_as_run_dino_pipeline_leaves_it(tmp_path):
    cls = tmp_path / "classification"
    cls.mkdir()
    (cls / "classification_report.txt").write_text(
        "Chosen on validation: DINOv2 max_pool + LogReg\nTest games: 31 32  (124 events)\n"
        "Test accuracy 0.2742  macro-F1 0.1438  majority-class baseline 0.2016\n"
        "95% interval, resampling test games: accuracy 0.254-0.292, macro-F1 0.100-0.200\n")
    (cls / "sequence_classification_report.txt").write_text(
        "Chosen on validation: TemporalCNN (mean val accuracy 0.2727 over 1 seed(s))\n"
        "Test accuracy 0.4929 +/- 0.0048 (sd over 1 seeds)  macro-F1 0.3096 +/- 0.0182 (sd over 1 seeds)  "
        "majority-class baseline 0.2016\n"
        "95% interval, resampling test games (all seeds pooled): accuracy 0.465-0.522, macro-F1 0.293-0.338\n")
    (tmp_path / "contrastive").mkdir()
    pd.DataFrame([{"Group": "held-out games", "Representation": "DINOv2 (before)", "ARI": 0.03, "Game_ARI": 0.40,
                   "Purity": 0.27}]).to_csv(tmp_path / "contrastive" / "contrastive_metrics.csv", index=False)
    main(["--results", str(tmp_path), "--frames", str(tmp_path / "none")])
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "| Pooled frozen features | DINOv2 max_pool + LogReg | 0.274 | 0.144 | 0.254-0.292 | 0.202 |" in text
    assert "| Sequence model | TemporalCNN (mean val accuracy 0.2727 over 1 seed(s)) | 0.493 | 0.310 | 0.465-0.522" in text
    assert "| held-out games | DINOv2 (before) | 0.030 | 0.400 | 0.270 |" in text
    assert "Visual + annotation" not in text  # no report, no row
