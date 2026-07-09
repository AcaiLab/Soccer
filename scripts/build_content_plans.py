import argparse
import json
from pathlib import Path

import numpy as np

from pipeline_utils import ensure_dir, load_config, read_jsonl, repo_root, resolve_path, write_jsonl


AUDIENCE_PLANS = {
    "beginner": {
        "goal": "Help a viewer understand what happened and what it changes.",
        "must_include": ["event", "plain rule/context", "what happens next"],
        "avoid": ["unexplained tactical jargon", "assumed soccer knowledge"],
        "style": "plain, calm, short, explanatory",
    },
    "expert": {
        "goal": "Describe the tactical cause and likely consequence.",
        "must_include": ["event", "phase of play", "tactical implication"],
        "avoid": ["basic rule explanation", "overexplaining obvious restarts"],
        "style": "compact, tactical, analytical",
    },
    "low_vision": {
        "goal": "Make the visual state accessible without relying on sight.",
        "must_include": ["event", "field area if known", "player or referee visual state", "restart/action state"],
        "avoid": ["as you can see", "here/there without reference", "purely abstract analysis"],
        "style": "concrete, spatial, descriptive, not cluttered",
    },
}


EVENT_FACTS = {
    "goal": {
        "plain_event": "a goal has been scored",
        "rule_context": "the ball has crossed the goal line and the score changes",
        "next_state": "play will restart from the center circle",
        "tactical_context": "the attacking move has broken through the defensive block",
        "visual_state": "players react around the goal while the goalkeeper and defenders turn back toward the net",
    },
    "corner": {
        "plain_event": "the attacking team has won a corner",
        "rule_context": "the ball went out over the goal line after touching a defender",
        "next_state": "play restarts from the corner flag",
        "tactical_context": "this becomes a set-piece chance with runners attacking the box",
        "visual_state": "players gather in the penalty area while the taker moves toward the corner flag",
    },
    "yellow_card": {
        "plain_event": "the referee has shown a yellow card",
        "rule_context": "a yellow card is an official warning to a player",
        "next_state": "the booked player must be careful for the rest of the match",
        "tactical_context": "the caution can change how aggressively that player challenges",
        "visual_state": "the referee holds up the card while nearby players slow down or gather around",
    },
    "substitution": {
        "plain_event": "a substitution is being made",
        "rule_context": "one player leaves and a teammate enters",
        "next_state": "the team may be changing energy, roles, or tactics",
        "tactical_context": "the change may alter the shape, pressing, or attacking outlet",
        "visual_state": "activity shifts to the sideline as one player comes off and another prepares to enter",
    },
    "shot_off_target": {
        "plain_event": "a shot has missed the goal",
        "rule_context": "the attempt does not force the goalkeeper to make a save",
        "next_state": "play usually restarts with a goal kick if no defender touched it last",
        "tactical_context": "the chance ends without testing the goalkeeper, often from pressure or a difficult angle",
        "visual_state": "the ball travels wide or over the goal and players begin to reset",
    },
    "saved_by_goal-keeper": {
        "plain_event": "the goalkeeper has made a save",
        "rule_context": "the shot was on target but did not become a goal",
        "next_state": "the ball may stay in play, go out for a corner, or be held by the goalkeeper",
        "tactical_context": "the attack creates a real chance but the keeper prevents the finish",
        "visual_state": "the goalkeeper moves toward the ball and stops it near the goal area",
    },
    "foul_with_no_card": {
        "plain_event": "the referee has called a foul",
        "rule_context": "play stops because of illegal contact or a challenge",
        "next_state": "the other team gets a free kick",
        "tactical_context": "the foul breaks the rhythm of the attack or stops a transition",
        "visual_state": "players slow down as the referee signals the direction of the restart",
    },
    "off_side": {
        "plain_event": "the attacking player has been called offside",
        "rule_context": "the attacker was ahead of the allowed position when the pass was played",
        "next_state": "the defending team gets the restart",
        "tactical_context": "the defensive line holds its shape well enough to catch the runner early",
        "visual_state": "play stops as the official signals offside and players turn back upfield",
    },
    "free_kick": {
        "plain_event": "a free kick has been awarded",
        "rule_context": "play restarts from the spot of an infringement",
        "next_state": "the team can pass, cross, or shoot depending on distance",
        "tactical_context": "the restart gives the attacking side time to set a rehearsed pattern",
        "visual_state": "players pause around the ball while defenders organize their positions",
    },
    "penalty": {
        "plain_event": "a penalty has been awarded",
        "rule_context": "a foul or handball in the penalty area gives a direct shot from the spot",
        "next_state": "one player will take a high-value shot against the goalkeeper",
        "tactical_context": "this is one of the highest expected-goal situations in soccer",
        "visual_state": "players gather outside the penalty area while the ball is placed on the spot",
    },
}


def generic_event_facts(label: str) -> dict[str, str]:
    readable = label.replace("_", " ").replace("-", " ")
    return {
        "plain_event": f"a {readable} occurs",
        "rule_context": "this changes the current phase of play",
        "next_state": "players reset for the next action",
        "tactical_context": "the moment affects spacing, possession, or match rhythm",
        "visual_state": "players adjust their positions as the sequence changes",
    }


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
                    "similarity": round(float(sims[i, j]), 4),
                    "event_label": str(labels[j]),
                    "game_label": str(games[j]),
                }
                for rank, j in enumerate(order)
            ]
        )
    return neighbors


def make_plan(row: dict, audience: str, neighbors: list[dict[str, object]]) -> dict[str, object]:
    label = row["event_label"]
    event_facts = EVENT_FACTS.get(label, generic_event_facts(label))
    same_label_neighbors = sum(item["event_label"] == label for item in neighbors)
    audience_spec = AUDIENCE_PLANS[audience]
    return {
        "window_id": row["window_id"],
        "game_label": row["game_label"],
        "event_label": label,
        "audience": audience,
        "frame_count": row.get("frame_count"),
        "sampled_frame_count": row.get("sampled_frame_count"),
        "audience_goal": audience_spec["goal"],
        "style": audience_spec["style"],
        "must_include": audience_spec["must_include"],
        "avoid": audience_spec["avoid"],
        "facts": event_facts,
        "retrieval": {
            "neighbors": neighbors,
            "same_label_neighbors": same_label_neighbors,
            "neighbor_count": len(neighbors),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_dry_run.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--memory", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-windows", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    base = repo_root()
    output_root = resolve_path(cfg["project"]["output_root"], base) or base
    output_dir = ensure_dir(output_root / "outputs")
    manifest = args.manifest or output_root / "manifests" / "local_windows.jsonl"
    memory = args.memory or output_root / "retrieval" / f"memory_{cfg['embeddings']['visual_model']}_mean_std.npz"
    out = args.out or output_dir / "content_plans.jsonl"
    windows = read_jsonl(manifest)
    if args.max_windows:
        windows = windows[: args.max_windows]

    data = np.load(memory, allow_pickle=True)
    features = data["features"].astype(np.float32)
    labels = data["event_labels"]
    games = data["game_labels"]
    k = int(cfg["generation"].get("max_examples_per_query", 5))
    neighbors = retrieve_neighbors(features, labels, games, k=k)
    neighbors_by_window = {str(wid): neigh for wid, neigh in zip(data["window_ids"], neighbors, strict=True)}

    rows = []
    for row in windows:
        row_neighbors = neighbors_by_window[row["window_id"]]
        for audience in cfg.get("audiences", ["beginner", "expert", "low_vision"]):
            rows.append(make_plan(row, audience, row_neighbors))

    write_jsonl(out, rows)
    print(json.dumps({"output": str(out), "plans": len(rows), "windows": len(windows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
