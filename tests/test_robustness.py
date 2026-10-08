from robustness import read_test_summary


def test_read_test_summary_parses_classifier_reports(tmp_path):
    report = tmp_path / "r.txt"
    report.write_text(
        "Chosen on validation: Contrastive (triplet head) + LogReg\n"
        "Test games: 25 26 27 28 29 30  (396 events)\n"
        "Test accuracy 0.3232  macro-F1 0.2529  majority-class baseline 0.1566\n"
        "95% interval, resampling test games: accuracy 0.281-0.366, macro-F1 0.210-0.290\n")
    s = read_test_summary(report)
    assert s == {"accuracy": 0.3232, "macro_f1": 0.2529, "majority": 0.1566, "ci_low": 0.281,
                 "ci_high": 0.366, "chosen": "Contrastive (triplet head) + LogReg"}


def test_read_test_summary_handles_sequence_reports(tmp_path):
    report = tmp_path / "r.txt"
    report.write_text(
        "Chosen on validation: BiLSTM (mean val accuracy 0.4100 over 5 seed(s))\n"
        "Test accuracy 0.4015 +/- 0.0100 (sd over 5 seeds)  macro-F1 0.2824 +/- 0.0050 (sd over 5 seeds)  "
        "majority-class baseline 0.1566\n")
    s = read_test_summary(report)
    assert s["accuracy"] == 0.4015 and s["majority"] == 0.1566
