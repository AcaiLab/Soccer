"""
Document which event labels are kept and which are excluded, with event counts per split.

Every experiment drops the labels listed under `labels:` in configs/pipeline.yaml
(scraper artefacts and broadcast graphics with no on-pitch event). This writes the
full table, so the exclusions are documented rather than silent:

    <results>/labels/label_counts.csv   label, status, reason, train, val, test, total
    <results>/labels/label_counts.md    the same as a readable table

Usage:
    python scripts/label_summary.py
    SOCCER_MANIFEST=configs/games_epl2122.csv python scripts/label_summary.py
"""

import argparse
import pathlib

import pandas as pd

from soccer.config import data_path, load_config, repo_path
from soccer.features import load_temporal
from soccer.manifest import load_manifest

RARE = 5  # fewer events than this: reported, but scores on it are unreliable


def label_table(labels, games, manifest, config_labels) -> pd.DataFrame:
    """One row per label: kept or excluded (and why), and its events in train / val / test."""
    split_of = {g.game_num: g.split for g in manifest}
    reason = {label: why for why, group in (("junk", config_labels.get("junk", [])),
                                            ("non-visual", config_labels.get("non_visual", [])))
              for label in group}
    events = pd.DataFrame({"label": list(labels), "split": [split_of[g] for g in games]})
    counts = pd.crosstab(events.label, events.split).reindex(columns=["train", "val", "test"], fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts.insert(0, "reason", [reason.get(label, "") for label in counts.index])
    counts.insert(0, "status", ["excluded" if label in reason else "kept" for label in counts.index])
    counts["note"] = ["fewer than %d events" % RARE if (s == "kept" and t < RARE) else ""
                      for s, t in zip(counts.status, counts.total, strict=True)]
    return counts.reset_index().sort_values(["status", "total"], ascending=[False, False]).reset_index(drop=True)


def to_markdown(table: pd.DataFrame) -> str:
    kept, dropped = table[table.status == "kept"], table[table.status == "excluded"]
    lines = [f"{len(kept)} labels kept ({int(kept.total.sum())} events: train {int(kept.train.sum())}, "
             f"val {int(kept.val.sum())}, test {int(kept.test.sum())}); {len(dropped)} excluded "
             f"({int(dropped.total.sum())} events).", "",
             "| Label | Status | Reason | Train | Val | Test | Total | Note |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r.label} | {r.status} | {r.reason or '-'} | {r.train} | {r.val} | {r.test} | {r.total} | "
              f"{r.note or '-'} |" for r in table.itertuples()]
    return "\n".join(lines) + "\n"


def main(argv=None):
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Write the table of kept and excluded labels with counts.")
    parser.add_argument("--features", default=str(data_path(cfg, "features") / "temporal_dinov2-base"))
    parser.add_argument("--out-dir", default=str(data_path(cfg, "results") / "labels"))
    args = parser.parse_args(argv)

    manifest = load_manifest(repo_path(cfg, "manifest"), repo_path(cfg, "excluded_matches"))
    tf = load_temporal(args.features, games={g.game_num for g in manifest})
    table = label_table(tf.labels, tf.games, manifest, cfg.get("labels", {}))
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "label_counts.csv", index=False)
    text = to_markdown(table)
    (out / "label_counts.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
