"""
Build the results report from a run's result files.

Reads whatever exists under the results folder and writes report.md next to
it, so every number in the write-up comes from a file:

  dinov2-base/clustering_metrics.csv     do clusters follow the match or the event?
  within_game/within_game_metrics.csv    clustering inside each match
  robustness/all_runs.csv                every seed of every split
  robustness/<split>/*/..._report.txt    per-seed test scores and game-resampled intervals
  <frames>/game_XX_half_N/extraction.json  how exact each half's frame times are

Differences that matter (contrastive head after vs before, sequence vs pooled)
are computed per seed and reported as paired means with their spread.

Usage:
    python scripts/report.py
    SOCCER_RESULTS_DIR=~/soccer-data/runs/epl2122/results python scripts/report.py
"""

import argparse
import pathlib
import re

import numpy as np
import pandas as pd
from robustness import read_test_summary
from sklearn.metrics import balanced_accuracy_score

from soccer.config import data_path, encoder_label, load_config
from soccer.frames import describe_time_check, time_check
from soccer.manifest import manifest_games
from soccer.metrics import bootstrap_ci

SPLIT_NAMES = {"default": "match-level split", "venue": "leave-venue-out split",
               "epl2122": "65 games, official split"}


def fmt(mean, sd=None, digits=3):
    if sd is None or not np.isfinite(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def mean_sd_or_na(series, digits=3):
    """mean ± sd of a metric, or n/a when the runs predate the metric."""
    series = series.dropna()
    return "n/a" if series.empty else fmt(series.mean(), series.std() if len(series) > 1 else None, digits)


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def clustering_section(results: pathlib.Path) -> list[str]:
    out = []
    path = results / "dinov2-base" / "clustering_metrics.csv"
    if path.is_file():
        m = pd.read_csv(path)
        rows = [[r.Representation, f"{r.ARI:.3f}", f"{r.Game_ARI:.3f}" if "Game_ARI" in m else "n/a",
                 f"{r.Purity:.3f} (chance {r.Purity_random:.3f})"] for r in m.itertuples()]
        out += ["## Do clusters follow the match or the event?", "",
                "K-Means with one cluster per event label, over all events of the manifest's games. "
                "ARI: agreement with event labels; Game ARI: agreement with match identity "
                "(0 = chance, 1 = perfect).", "",
                table(["Representation", "ARI (event)", "Game ARI (match)", "Purity"], rows), ""]
    path = results / "within_game" / "within_game_metrics.csv"
    if path.is_file():
        w = pd.read_csv(path)
        out += [f"Within each match ({len(w)} matches, so the match signal cannot help): mean ARI "
                f"{w.ARI.mean():.3f} (sd {w.ARI.std():.3f}), purity {w.Purity.mean():.3f} against "
                f"{w.Purity_random.mean():.3f} for randomly shuffled clusters.", ""]
    return out


def paired(df: pd.DataFrame, split: str, a: str, b: str):
    """Per-seed difference a - b for one split: (mean, sd, n, n positive)."""
    s = df[df.split == split].pivot_table(index="seed", columns="metric", values="value")
    if a not in s or b not in s:
        return None
    d = (s[a] - s[b]).dropna()
    return d.mean(), d.std(ddof=1) if len(d) > 1 else float("nan"), len(d), int((d > 0).sum())


def read_ci(path: pathlib.Path):
    m = re.search(r"accuracy ([\d.]+)-([\d.]+), macro-F1", path.read_text()) if path.is_file() else None
    return (float(m.group(1)), float(m.group(2))) if m else None


CV_SPLIT = r"cv\d+$"
CV_MODELS = [("Pooled frozen features", "pooled", "pooled_s*/test_predictions.csv"),
             ("Sequence model", "sequence", "sequence/sequence_test_predictions.csv"),
             ("Visual + annotation", "multimodal", "multimodal_s*/multimodal_test_predictions.csv")]


def cv_section(results: pathlib.Path) -> list[str]:
    """Folds cv1..cvK (scripts/make_cv_folds.py): every match is a test game in exactly one fold."""
    path = results / "robustness" / "all_runs.csv"
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    folds = sorted({s for s in df.split.unique() if re.match(CV_SPLIT, s)}, key=lambda s: int(s[2:]))
    if not folds:
        return []

    def per_fold(metric):
        return pd.Series([df[(df.split == f) & (df.metric == metric)].value.mean() for f in folds], index=folds)

    out = ["## Cross-validation over all matches", "",
           f"{len(folds)} folds by whole match; each match is a test game in exactly one fold, and models are "
           "chosen on that fold's validation matches. Accuracy is mean ± sd over folds (each fold averaged over "
           "its seeds); the pooled column scores every test prediction of every fold together, with a 95% "
           "interval from resampling matches.", ""]
    rows = []
    for name, prefix, pattern in CV_MODELS:
        acc, f1 = per_fold(f"{prefix}_accuracy"), per_fold(f"{prefix}_macro_f1")
        if acc.isna().all():
            continue
        preds = pd.concat([pd.read_csv(p, dtype=str).assign(fold=f)
                           for f in folds for p in sorted((results / "robustness" / f).glob(pattern))])
        if (preds.groupby("game").fold.nunique() > 1).any():
            raise ValueError(f"{name}: a match was tested in more than one fold")
        ci = bootstrap_ci(preds.label, preds.pred, preds.game)["Accuracy_CI"]
        rows.append([name, fmt(acc.mean(), acc.std()), fmt(f1.mean(), f1.std()),
                     mean_sd_or_na(per_fold(f"{prefix}_balanced_accuracy")),
                     f"{(preds.label == preds.pred).mean():.3f} ({ci[0]:.3f}-{ci[1]:.3f})",
                     f"{balanced_accuracy_score(preds.label, preds.pred):.3f}", preds.game.nunique()])
    base = per_fold("pooled_majority")
    out += [table(["Model", "Accuracy", "Macro-F1", "Balanced accuracy", "Pooled accuracy (95% interval)",
                  "Pooled balanced accuracy", "Matches scored"], rows), "",
            f"Majority-class baseline: {fmt(base.mean(), base.std())}.", ""]
    diff = per_fold("sequence_accuracy") - per_fold("pooled_accuracy")
    if diff.notna().any():
        out += [f"Sequence minus pooled accuracy, per fold: {fmt(diff.mean(), diff.std())} "
                f"(sequence ahead in {int((diff > 0).sum())} of {int(diff.notna().sum())} folds).", ""]
    before, after = per_fold("contrastive_heldout_before_ARI"), per_fold("contrastive_heldout_after_ARI")
    if after.notna().any():
        change = after - before
        out += [f"Contrastive head, event ARI on each fold's held-out matches: {fmt(before.mean(), before.std())} "
                f"before, {fmt(after.mean(), after.std())} after (up in {int((change > 0).sum())} of "
                f"{int(change.notna().sum())} folds).", ""]
    return out


def robustness_section(results: pathlib.Path, encoder: str = "DINOv2") -> list[str]:
    path = results / "robustness" / "all_runs.csv"
    if not path.is_file():
        return []
    df = pd.read_csv(path)
    df = df[~df.split.str.match(CV_SPLIT)]  # folds have their own section
    if df.empty:
        return []
    out = [f"## Contrastive head (triplet loss on frozen {encoder})", "",
           "Held-out games are matches the head never saw. Values are mean ± sd over seeds; "
           "the change is computed within each seed.", ""]
    rows = []
    for split in df.split.unique():
        s = df[df.split == split].groupby("metric")["value"]
        mean, sd = s.mean(), s.std()
        for what, key in [("event ARI", "ARI"), ("match ARI", "GameARI"), ("purity", "purity")]:
            before, after = f"contrastive_heldout_before_{key}", f"contrastive_heldout_after_{key}"
            if before not in mean:
                continue
            p = paired(df, split, after, before)
            change = f"{p[0]:+.3f} ± {p[1]:.3f} ({p[3]}/{p[2]} seeds up)" if p else "n/a"
            rows.append([SPLIT_NAMES.get(split, split), what, fmt(mean[before], sd[before]),
                         fmt(mean[after], sd[after]), change])
    out += [table(["Split", "Held-out games", f"Before ({encoder})", "After (head)", "Change"], rows), ""]
    train_rows = []
    for split in df.split.unique():
        s = df[df.split == split].groupby("metric")["value"]
        mean = s.mean()
        if "contrastive_train_after_ARI" in mean:
            train_rows.append([SPLIT_NAMES.get(split, split), f"{mean['contrastive_train_before_ARI']:.3f}",
                               f"{mean['contrastive_train_after_ARI']:.3f}"])
    if train_rows:
        out += ["On the training games themselves (fit, not generalisation):", "",
                table(["Split", "Event ARI before", "Event ARI after"], train_rows), ""]

    out += ["## Event classification on held-out test games", "",
            "Accuracy and macro-F1 are mean ± sd over seeds; the interval is the 95% range from "
            "resampling whole test games, averaged over seeds; the baseline always predicts the most "
            "common training label.", ""]
    rows = []
    for split in df.split.unique():
        s = df[df.split == split].groupby("metric")["value"]
        mean, sd, n = s.mean(), s.std(), s.count()
        base = mean.get("pooled_majority", float("nan"))
        for model, prefix, report in [("Pooled frozen features (best on validation)", "pooled",
                                       "pooled_s*/classification_report.txt"),
                                      ("Sequence model (best on validation)", "sequence",
                                       "sequence/sequence_classification_report.txt"),
                                      ("Visual + annotation (best on validation)", "multimodal",
                                       "multimodal_s*/multimodal_classification_report.txt")]:
            key = f"{prefix}_accuracy"
            if key not in mean:
                continue
            cis = [c for c in (read_ci(p) for p in (results / "robustness" / split).glob(report)) if c]
            ci = (f"{np.mean([c[0] for c in cis]):.3f}-{np.mean([c[1] for c in cis]):.3f}" if cis else "n/a")
            rows.append([SPLIT_NAMES.get(split, split), model, fmt(mean[key], sd[key]),
                         fmt(mean[f"{prefix}_macro_f1"], sd[f"{prefix}_macro_f1"]),
                         fmt(mean[f"{prefix}_balanced_accuracy"], sd[f"{prefix}_balanced_accuracy"])
                         if f"{prefix}_balanced_accuracy" in mean else "n/a", ci,
                         f"{base:.3f}", int(n[key])])
    out += [table(["Split", "Model", "Accuracy", "Macro-F1", "Balanced accuracy", "95% interval (accuracy)",
                   "Majority baseline", "Seeds"], rows), ""]
    comparisons = []
    for split in df.split.unique():
        p = paired(df, split, "sequence_accuracy", "pooled_accuracy")
        if p:
            comparisons.append(f"- {SPLIT_NAMES.get(split, split)}: sequence minus pooled accuracy "
                               f"{p[0]:+.3f} (sd {p[1]:.3f}; sequence ahead in {p[3]} of {p[2]} seeds).")
    if comparisons:
        out += ["Paired by seed:", "", *comparisons, ""]
    return out


def timing_section(frames_root: pathlib.Path, games: set[str]) -> list[str]:
    if not frames_root.is_dir():
        return []
    check = time_check(frames_root, games)
    if not check["halves"]:
        return []
    return ["## Frame timing check", "",
            "Frames are cut by index (time × fps). Each half's extraction marker records the largest gap "
            "between the decoder's timestamp and that index time; above 0.5 s the saved frames may not be "
            "the ones at the annotated moments (`scripts/audit_video_timing.py` says why).", "",
            describe_time_check(check) + ".", ""]


SINGLE_RUN = [("Pooled frozen features", "classification_report.txt"),
              ("Sequence model", "sequence_classification_report.txt"),
              ("Visual + annotation", "multimodal_classification_report.txt")]


def single_run_section(results: pathlib.Path) -> list[str]:
    """The stages as run_dino_pipeline.py runs them: one seed, results/classification and results/contrastive."""
    out = []
    path = results / "contrastive" / "contrastive_metrics.csv"
    if path.is_file():
        m = pd.read_csv(path)
        rows = [[r.Group, r.Representation, f"{r.ARI:.3f}", f"{r.Game_ARI:.3f}", f"{r.Purity:.3f}"]
                for r in m.itertuples()]
        out += ["## Contrastive head, single run", "",
                "Held-out games are matches the head never saw; one seed (scripts/robustness.py repeats it).", "",
                table(["Games", "Representation", "Event ARI", "Match ARI", "Purity"], rows), ""]
    rows = []
    for name, file in SINGLE_RUN:
        if (results / "classification" / file).is_file():
            s = read_test_summary(results / "classification" / file)
            ci = f"{s['ci_low']:.3f}-{s['ci_high']:.3f}" if "ci_low" in s else "n/a"
            rows.append([name, s.get("chosen", "n/a"), f"{s['accuracy']:.3f}", f"{s['macro_f1']:.3f}",
                         f"{s['balanced_accuracy']:.3f}" if "balanced_accuracy" in s else "n/a", ci,
                         f"{s['majority']:.3f}"])
    if rows:
        out += ["## Event classification on held-out test games, single run", "",
                "Each model is chosen on validation games and scored once on test games; the interval resamples "
                "whole test games. One seed: scripts/robustness.py gives mean ± sd over seeds.", "",
                table(["Model", "Chosen on validation", "Accuracy", "Macro-F1", "Balanced accuracy",
                       "95% interval (accuracy)", "Majority baseline"], rows), ""]
    return out


def labels_section(results: pathlib.Path) -> list[str]:
    """Kept and excluded labels with their counts, from scripts/label_summary.py."""
    path = results / "labels" / "label_counts.md"
    if not path.is_file():
        return []
    return ["## Labels: what is kept and what is excluded", "", path.read_text(encoding="utf-8").strip(), ""]


def main(argv=None):
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Write report.md from a run's result files.")
    parser.add_argument("--results", default=str(data_path(cfg, "results")))
    parser.add_argument("--frames", default=str(data_path(cfg, "frames")),
                        help="extracted frames, for the frame-timing check (skipped if absent)")
    parser.add_argument("--title", default="Results")
    args = parser.parse_args(argv)

    results = pathlib.Path(args.results)
    sections = (clustering_section(results) + labels_section(results) + single_run_section(results)
                + cv_section(results)
                + robustness_section(results, encoder_label(cfg)))
    if not sections:
        raise SystemExit(f"No result files found under {results}")
    sections += timing_section(pathlib.Path(args.frames), set(manifest_games(cfg)))
    text = f"# {args.title}\n\nGenerated by scripts/report.py from {results}.\n\n" + "\n".join(sections)
    (results / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
