"""
Per-class scores of the chosen classifiers next to each class's number of events.

Answers "which event types are too rare to learn?" from the saved test
predictions of a robustness run (every seed), so the decision to keep or merge
a class rests on its training count and its F1 together:

    <results>/per_class/per_class.csv   label, train/val/test events, precision/recall/F1 per model
    <results>/per_class/per_class.md    the same as a readable table

Scores are computed per seed and averaged. Run scripts/label_summary.py first
for the counts.

Usage:
    SOCCER_RESULTS_DIR=~/soccer-data/runs/epl1720_clf/mv/results_f1 python scripts/per_class.py --split epl1720
"""

import argparse
import pathlib

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from soccer.config import data_path, load_config

MODELS = {"pooled": "pooled_s*/test_predictions.csv", "sequence": "sequence/sequence_test_predictions.csv"}


def per_class_scores(preds: pd.DataFrame) -> pd.DataFrame:
    """Precision, recall and F1 per true label, averaged over the seeds in `preds` (columns seed, label, pred)."""
    rows = []
    for seed, p in preds.groupby("seed"):
        labels = sorted(set(p.label))
        prec, rec, f1, _ = precision_recall_fscore_support(p.label, p.pred, labels=labels, zero_division=0)
        rows += [{"seed": seed, "label": lab, "precision": a, "recall": b, "f1": c}
                 for lab, a, b, c in zip(labels, prec, rec, f1, strict=True)]
    return pd.DataFrame(rows).groupby("label")[["precision", "recall", "f1"]].mean()


def load_predictions(split_dir: pathlib.Path, pattern: str) -> pd.DataFrame | None:
    files = sorted(split_dir.glob(pattern))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def per_class_table(counts: pd.DataFrame, scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Kept labels with their counts, then one F1 (and recall) column per model, rarest training class first."""
    table = counts[counts.status == "kept"].set_index("label")[["train", "val", "test"]]
    for model, s in scores.items():
        table[f"{model}_f1"] = s["f1"]
        table[f"{model}_recall"] = s["recall"]
    return table.sort_values("train").reset_index()


def to_markdown(table: pd.DataFrame, models: list[str]) -> str:
    header = ["Label", "Train", "Val", "Test"] + [f"{m} F1" for m in models]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in table.itertuples(index=False):
        d = r._asdict()
        f1s = ["n/a" if pd.isna(d[f"{m}_f1"]) else f"{d[f'{m}_f1']:.3f}" for m in models]
        lines.append("| " + " | ".join([d["label"], str(d["train"]), str(d["val"]), str(d["test"]), *f1s]) + " |")
    return ("Per-class test F1 of the chosen models (mean over seeds), rarest training class first. "
            "n/a: the class has no test events.\n\n" + "\n".join(lines) + "\n")


def main(argv=None):
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Per-class scores next to class counts.")
    parser.add_argument("--results", default=str(data_path(cfg, "results")))
    parser.add_argument("--split", default="epl1720", help="robustness split folder")
    args = parser.parse_args(argv)

    results = pathlib.Path(args.results)
    counts_path = results / "labels" / "label_counts.csv"
    if not counts_path.is_file():
        raise SystemExit(f"{counts_path} is missing; run scripts/label_summary.py first")
    counts = pd.read_csv(counts_path)
    scores = {}
    for model, pattern in MODELS.items():
        preds = load_predictions(results / "robustness" / args.split, pattern)
        if preds is not None:
            scores[model] = per_class_scores(preds)
    if not scores:
        raise SystemExit(f"No test predictions under {results / 'robustness' / args.split}")
    table = per_class_table(counts, scores)
    out = results / "per_class"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "per_class.csv", index=False)
    text = to_markdown(table, list(scores))
    (out / "per_class.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
