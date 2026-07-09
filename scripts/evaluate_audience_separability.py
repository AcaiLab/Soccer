import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

from commentary_eval_utils import AUDIENCE_KEYWORDS, count_matches
from pipeline_utils import ensure_dir, read_jsonl, repo_root, write_json


def heuristic_predict(text: str) -> str:
    scores = {aud: count_matches(text, terms) for aud, terms in AUDIENCE_KEYWORDS.items()}
    return max(scores.items(), key=lambda item: item[1])[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/llm_commentary_samples_grounded.jsonl"))
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [r for r in read_jsonl(args.input) if r.get("commentary")]
    df = pd.DataFrame(rows)
    df["heuristic_pred"] = df["commentary"].map(heuristic_predict)
    heuristic_acc = accuracy_score(df["audience"], df["heuristic_pred"]) if len(df) else 0.0
    ml_acc = None
    if len(df) >= 12 and df["audience"].value_counts().min() >= 2:
        splits = min(3, int(df["audience"].value_counts().min()))
        clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1), LogisticRegression(max_iter=1000))
        pred = cross_val_predict(clf, df["commentary"], df["audience"], cv=StratifiedKFold(n_splits=splits, shuffle=True, random_state=7))
        df["tfidf_cv_pred"] = pred
        ml_acc = accuracy_score(df["audience"], pred)
    reports = ensure_dir(repo_root() / "reports")
    out_csv = args.out_csv or reports / f"{args.input.stem}_audience_separability.csv"
    out_summary = args.out_summary or reports / f"{args.input.stem}_audience_separability_summary.json"
    df.to_csv(out_csv, index=False)
    labels = sorted(df["audience"].unique()) if len(df) else []
    summary = {
        "input": str(args.input),
        "rows": int(len(df)),
        "heuristic_accuracy": float(heuristic_acc),
        "tfidf_cv_accuracy": ml_acc,
        "labels": labels,
        "heuristic_confusion": confusion_matrix(df["audience"], df["heuristic_pred"], labels=labels).tolist() if labels else [],
        "out_csv": str(out_csv),
    }
    write_json(out_summary, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
