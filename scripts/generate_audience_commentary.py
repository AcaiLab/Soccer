import argparse
import json
from pathlib import Path

import numpy as np

from pipeline_utils import ensure_dir, load_config, repo_root, resolve_path, write_jsonl


EVENT_DESCRIPTIONS = {
    "goal": {
        "beginner": "A goal has been scored, which changes the score and usually shifts the match momentum.",
        "expert": "The attacking phase ends with a goal; inspect the buildup, defensive spacing, and final action.",
        "low_vision": "The ball has crossed the goal line into the net. Players are reacting and the crowd noise likely rises.",
    },
    "corner": {
        "beginner": "The attacking team has won a corner kick after the defending team last touched the ball over the goal line.",
        "expert": "This creates a set-piece chance, often with near-post runs, blockers, and second-ball pressure.",
        "low_vision": "Play is restarting from the corner flag. Players are gathering inside the penalty area.",
    },
    "yellow_card": {
        "beginner": "The referee has shown a yellow card, which is an official warning to a player.",
        "expert": "The booking can affect duel intensity and defensive choices for the cautioned player.",
        "low_vision": "The referee is displaying a yellow card. Players may be gathered around the official.",
    },
    "substitution": {
        "beginner": "A player is being replaced by a teammate, often to change tactics or manage fatigue.",
        "expert": "The substitution may alter formation, pressing intensity, or role assignments.",
        "low_vision": "The substitution board or sideline activity is visible as one player leaves and another enters.",
    },
    "shot_off_target": {
        "beginner": "A player attempted a shot, but the ball missed the goal.",
        "expert": "The chance ends without forcing a save; shot location and defensive pressure are key context.",
        "low_vision": "The ball is struck toward goal but travels wide or over the frame.",
    },
    "saved_by_goal-keeper": {
        "beginner": "The goalkeeper stopped the shot and prevented a goal.",
        "expert": "The keeper's intervention ends the immediate chance and may create a rebound or set piece.",
        "low_vision": "The goalkeeper reaches the ball and stops it near the goal area.",
    },
    "foul_with_no_card": {
        "beginner": "The referee has called a foul, so play stops and the other team gets a free kick.",
        "expert": "The foul interrupts the phase and may reset defensive shape or stop a transition.",
        "low_vision": "Play has stopped after contact or an illegal challenge. The referee indicates the direction of the restart.",
    },
    "off_side": {
        "beginner": "The attacker was offside, meaning they were too close to goal when the pass was played.",
        "expert": "The defensive line catches the runner ahead of the legal receiving position.",
        "low_vision": "The assistant referee or official signals offside, and play stops for the restart.",
    },
}


def default_description(label: str, audience: str) -> str:
    readable = label.replace("_", " ").replace("-", " ")
    if audience == "beginner":
        return f"The current event is {readable}. The important point is how this changes the next restart or phase of play."
    if audience == "expert":
        return f"The event is {readable}; the key analysis is the tactical context before and after the action."
    return f"The event is {readable}. The description should emphasize ball location, player movement, visible signals, and restart state."


def retrieve_neighbors(features: np.ndarray, labels: np.ndarray, games: np.ndarray, k: int) -> list[list[dict[str, object]]]:
    sims = features @ features.T
    np.fill_diagonal(sims, -np.inf)
    neighbors = []
    for i in range(len(features)):
        order = np.argsort(-sims[i])[:k]
        neighbors.append(
            [
                {
                    "rank": rank + 1,
                    "similarity": float(sims[i, j]),
                    "event_label": str(labels[j]),
                    "game_label": str(games[j]),
                }
                for rank, j in enumerate(order)
            ]
        )
    return neighbors


def make_commentary(label: str, audience: str, neighbors: list[dict[str, object]]) -> str:
    description = EVENT_DESCRIPTIONS.get(label, {}).get(audience, default_description(label, audience))
    same_label = sum(1 for item in neighbors if item["event_label"] == label)
    retrieval_note = f" Similar retrieved windows include {same_label} of the same event type among the nearest examples."
    if audience == "beginner":
        return description + retrieval_note
    if audience == "expert":
        return description + " Compare this with retrieved similar phases to judge whether the pattern is repeatable."
    return description + " Keep the narration concrete and spatial so the listener can follow without seeing the screen."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_dry_run.yaml"))
    parser.add_argument("--memory", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-windows", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    base = repo_root()
    output_root = resolve_path(cfg["project"]["output_root"], base) or base
    output_dir = ensure_dir(output_root / "outputs")
    memory_path = args.memory or output_root / "retrieval" / f"memory_{cfg['embeddings']['visual_model']}_mean_std.npz"
    out = args.out or output_dir / "audience_commentary_samples.jsonl"
    data = np.load(memory_path, allow_pickle=True)
    features = data["features"].astype(np.float32)
    labels = data["event_labels"]
    games = data["game_labels"]
    window_ids = data["window_ids"]
    k = int(cfg["generation"].get("max_examples_per_query", 5))
    audiences = cfg.get("audiences", ["beginner", "expert", "low_vision"])
    neighbors = retrieve_neighbors(features, labels, games, k=k)

    rows = []
    limit = min(args.max_windows, len(features))
    for i in range(limit):
        for audience in audiences:
            rows.append(
                {
                    "window_id": str(window_ids[i]),
                    "game_label": str(games[i]),
                    "event_label": str(labels[i]),
                    "audience": audience,
                    "retrieved_examples": neighbors[i],
                    "commentary": make_commentary(str(labels[i]), audience, neighbors[i]),
                }
            )
    write_jsonl(out, rows)
    print(json.dumps({"output": str(out), "rows": len(rows), "memory": str(memory_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
