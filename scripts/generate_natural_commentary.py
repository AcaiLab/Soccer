import argparse
import hashlib
import json
from pathlib import Path

from pipeline_utils import ensure_dir, load_config, read_jsonl, repo_root, resolve_path, write_jsonl


OPENERS = {
    "beginner": [
        "{plain_event_cap}.",
        "In this moment, {plain_event}.",
        "Here, {plain_event}.",
    ],
    "expert": [
        "{plain_event_cap}.",
        "The key moment: {plain_event}.",
        "The key moment is that {plain_event}.",
    ],
    "low_vision": [
        "{plain_event_cap}.",
        "Play has shifted: {plain_event}.",
        "The visible cue is that {plain_event}.",
    ],
}


def stable_choice(options: list[str], key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


def capitalize_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def beginner_commentary(plan: dict) -> str:
    facts = plan["facts"]
    opener = stable_choice(OPENERS["beginner"], plan["window_id"] + plan["audience"]).format(
        plain_event=facts["plain_event"],
        plain_event_cap=capitalize_first(facts["plain_event"]),
    )
    return " ".join(
        [
            opener,
            f"In simple terms, {facts['rule_context']}.",
            f"The next thing to watch is that {facts['next_state']}.",
        ]
    )


def expert_commentary(plan: dict) -> str:
    facts = plan["facts"]
    opener = stable_choice(OPENERS["expert"], plan["window_id"] + plan["audience"]).format(
        plain_event=facts["plain_event"],
        plain_event_cap=capitalize_first(facts["plain_event"]),
    )
    same = plan["retrieval"]["same_label_neighbors"]
    retrieval_clause = ""
    if same:
        noun = "example" if same == 1 else "examples"
        retrieval_clause = f" The retrieval memory finds {same} close {noun} of the same event type, so this pattern is worth comparing across matches."
    return " ".join(
        [
            opener,
            f"Tactically, {facts['tactical_context']}.",
            f"The immediate implication is that {facts['next_state']}.",
            retrieval_clause.strip(),
        ]
    ).strip()


def low_vision_commentary(plan: dict) -> str:
    facts = plan["facts"]
    visual_facts = plan.get("visual_facts") or {}
    camera = visual_facts.get("camera", {})
    scene = visual_facts.get("scene", {})
    visual_sentence = f"Visually, {facts['visual_state']}."
    if camera:
        shot_type = str(camera.get("shot_type", "unknown")).replace("_", " ")
        motion = scene.get("motion_intensity")
        visual_sentence = f"Visually, the broadcast appears to show a {shot_type}"
        if motion:
            visual_sentence += f" with {motion} motion"
        visual_sentence += "."
    opener = stable_choice(OPENERS["low_vision"], plan["window_id"] + plan["audience"]).format(
        plain_event=facts["plain_event"],
        plain_event_cap=capitalize_first(facts["plain_event"]),
    )
    return " ".join(
        [
            opener,
            visual_sentence,
            f"The important state change is that {facts['next_state']}.",
        ]
    )


def generate_commentary(plan: dict) -> str:
    audience = plan["audience"]
    if audience == "beginner":
        return beginner_commentary(plan)
    if audience == "expert":
        return expert_commentary(plan)
    if audience == "low_vision":
        return low_vision_commentary(plan)
    raise ValueError(f"Unknown audience: {audience}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_dry_run.yaml"))
    parser.add_argument("--plans", type=Path)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    base = repo_root()
    output_root = resolve_path(cfg["project"]["output_root"], base) or base
    output_dir = ensure_dir(output_root / "outputs")
    plans_path = args.plans or output_dir / "content_plans.jsonl"
    out = args.out or output_dir / "natural_commentary_samples.jsonl"

    plans = read_jsonl(plans_path)
    rows = []
    for plan in plans:
        rows.append(
            {
                "window_id": plan["window_id"],
                "game_label": plan["game_label"],
                "event_label": plan["event_label"],
                "audience": plan["audience"],
                "commentary": generate_commentary(plan),
                "content_plan": {
                    "must_include": plan["must_include"],
                    "avoid": plan["avoid"],
                    "facts": plan["facts"],
                    "retrieval": plan["retrieval"],
                },
            }
        )

    write_jsonl(out, rows)
    print(json.dumps({"output": str(out), "rows": len(rows), "plans": str(plans_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
