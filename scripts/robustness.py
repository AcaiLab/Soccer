"""
Repeat the contrastive and classification stages over seeds and splits.

For each manifest (default: the match-level split and the leave-venue-out
split) and each seed, trains and evaluates the contrastive head and reruns
the classifiers, then writes one summary:

    <results>/robustness/summary.csv   one row per split, metric: mean, sd, n
    <results>/robustness/summary.md    the same as a readable table

Uses the existing DINOv2 features (run the frames and features stages first)
and the existing game-flow features for the multimodal classifier.

Usage:
    python scripts/make_venue_split.py
    python scripts/robustness.py --device cuda
    python scripts/robustness.py --seeds 42 43 44 --splits default
"""

import argparse
import os
import pathlib
import re
import subprocess
import sys

import pandas as pd

from soccer.config import REPO_ROOT, data_path, load_config

SPLITS = {"default": "configs/games.csv", "venue": "configs/games_venue_split.csv",
          "epl2122": "configs/games_epl2122.csv"}


def run(script, args, manifest):
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / script), *map(str, args)]
    print("$ SOCCER_MANIFEST=" + manifest + " scripts/" + script + " " + " ".join(map(str, args)), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, "SOCCER_MANIFEST": manifest})


def read_test_summary(path: pathlib.Path) -> dict:
    """Pull accuracy, macro-F1 and the interval out of a classifier report."""
    text = path.read_text()
    out = {}
    m = re.search(r"Test accuracy ([\d.]+).*?macro-F1 ([\d.]+)", text)
    if m:
        out["accuracy"], out["macro_f1"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"majority-class baseline ([\d.]+)", text)
    if m:
        out["majority"] = float(m.group(1))
    m = re.search(r"accuracy ([\d.]+)-([\d.]+), macro-F1", text)
    if m:
        out["ci_low"], out["ci_high"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"Chosen on validation: (.+)", text)
    if m:
        out["chosen"] = m.group(1).strip()
    return out


def main(argv=None):
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Seeds x splits robustness run.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=["default", "venue"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    device = ["--device", args.device] if args.device else []
    features = data_path(cfg, "features")
    root = data_path(cfg, "results") / "robustness"
    rows = []
    for split in args.splits:
        manifest = SPLITS[split]
        if not (REPO_ROOT / manifest).is_file():
            raise SystemExit(f"{manifest} is missing; run scripts/make_venue_split.py first")
        out = root / split
        for seed in args.seeds:
            head = features / f"contrastive_{split}_s{seed}"
            run("contrastive_training.py", ["--seed", seed, "--out-dir", head, *device], manifest)
            run("evaluate_contrastive.py", ["--contrastive-features", head, "--out-dir", out / f"contrastive_s{seed}"],
                manifest)
            m = pd.read_csv(out / f"contrastive_s{seed}" / "contrastive_metrics.csv")
            for _, r in m.iterrows():
                tag = ("heldout" if r["Group"].startswith("held") else "train") + "_" + \
                      ("before" if "before" in r["Representation"] else "after")
                rows.append({"split": split, "seed": seed, "metric": f"contrastive_{tag}_ARI", "value": r["ARI"]})
                rows.append({"split": split, "seed": seed, "metric": f"contrastive_{tag}_purity", "value": r["Purity"]})
            run("event_classification.py", ["--contrastive-features", head, "--out-dir", out / f"pooled_s{seed}"],
                manifest)
            s = read_test_summary(out / f"pooled_s{seed}" / "classification_report.txt")
            rows += [{"split": split, "seed": seed, "metric": f"pooled_{k}", "value": s[k]}
                     for k in ("accuracy", "macro_f1", "majority") if k in s]
        run("sequence_classifier.py", ["--seeds", *args.seeds, "--out-dir", out / "sequence", *device], manifest)
        for _, r in pd.read_csv(out / "sequence" / "sequence_test_by_seed.csv").iterrows():
            rows.append({"split": split, "seed": int(r["seed"]), "metric": "sequence_accuracy", "value": r["Accuracy"]})
            rows.append({"split": split, "seed": int(r["seed"]), "metric": "sequence_macro_f1", "value": r["Macro_F1"]})
        run("multimodal_classification.py", ["--out-dir", out / "multimodal"], manifest)
        s = read_test_summary(out / "multimodal" / "multimodal_classification_report.txt")
        rows += [{"split": split, "seed": cfg["seed"], "metric": f"multimodal_{k}", "value": s[k]}
                 for k in ("accuracy", "macro_f1") if k in s]

    df = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / "all_runs.csv", index=False)
    summary = df.groupby(["split", "metric"])["value"].agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(root / "summary.csv", index=False)
    table = summary.pivot(index="metric", columns="split", values="mean").round(3)
    sd = summary.pivot(index="metric", columns="split", values="std").round(3)
    lines = ["| metric | " + " | ".join(table.columns) + " |", "|---|" + "---|" * len(table.columns)]
    for metric in table.index:
        cells = [f"{table.loc[metric, c]:.3f} ± {sd.loc[metric, c]:.3f}" if pd.notna(sd.loc[metric, c])
                 else f"{table.loc[metric, c]:.3f}" for c in table.columns]
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    md = (f"# Robustness: seeds {' '.join(map(str, args.seeds))}\n\n"
          "Mean ± sd over seeds; test games only. Sequence and pooled rows have their own "
          "game-resampled intervals in each report.\n\n" + "\n".join(lines) + "\n")
    (root / "summary.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
