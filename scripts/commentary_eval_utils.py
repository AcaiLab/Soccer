import math
import re
from collections import Counter


EVENT_KEYWORDS = {
    "goal": ["goal", "scored", "net", "score"],
    "corner": ["corner", "corner flag", "set-piece", "set piece"],
    "free_kick": ["free kick", "restart", "foul"],
    "yellow_card": ["yellow card", "booking", "booked", "warning"],
    "red_card": ["red card", "sent off", "dismissed"],
    "substitution": ["substitution", "substitute", "replaced", "comes on", "comes off"],
    "shot_off_target": ["shot", "wide", "over", "missed"],
    "saved_by_goal-keeper": ["save", "goalkeeper", "keeper", "stopped"],
    "foul_with_no_card": ["foul", "free kick", "illegal challenge"],
    "foul_lead_to_penalty": ["foul", "penalty"],
    "penalty": ["penalty", "spot"],
    "off_side": ["offside", "off side"],
    "clearance": ["clearance", "cleared", "kicked away"],
    "ball_out_of_play": ["out of play", "went out"],
    "throw_in": ["throw-in", "throw in", "touchline"],
    "injury": ["injury", "injured", "medical"],
    "var": ["var", "review"],
    "show_added_time": ["added time", "stoppage time"],
    "start_of_half_game": ["start", "kickoff", "kick-off", "half begins"],
    "end_of_half_game": ["half ends", "end of the half", "full-time", "whistle"],
}

AUDIENCE_KEYWORDS = {
    "beginner": ["rule", "means", "according to", "in simple terms", "watch", "next"],
    "expert": ["tactical", "phase", "shape", "spacing", "overload", "press", "transition", "set-piece"],
    "low_vision": ["visually", "players", "ball", "referee", "moving", "gathering", "positioning", "visible"],
}

SPATIAL_WORDS = [
    "left",
    "right",
    "center",
    "central",
    "near",
    "far",
    "penalty area",
    "box",
    "corner flag",
    "goal line",
    "touchline",
    "sideline",
    "field",
]

MOTION_WORDS = [
    "moving",
    "runs",
    "run",
    "gathering",
    "spreading",
    "repositioning",
    "approaches",
    "preparing",
    "cross",
    "kick",
    "restart",
]

VAGUE_VISUAL_PHRASES = [
    "as you can see",
    "over there",
    "right there",
    "this here",
    "that area",
]

TACTICAL_WORDS = [
    "tactical",
    "shape",
    "spacing",
    "phase",
    "press",
    "overload",
    "transition",
    "defensive block",
    "set-piece",
    "second-ball",
    "build-up",
]

UNSUPPORTED_PATTERNS = [
    r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",  # likely player/person name
    r"\b\d+\s*-\s*\d+\b",  # score
    r"\b(left|right)\s+(flank|wing|side)\b",
    r"\boutside the box\b",
    r"\bdefensive wall\b",
    r"\breferee is nearby\b",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def count_matches(text: str, terms: list[str]) -> int:
    t = normalize(text)
    return sum(1 for term in terms if term in t)


def event_consistency(label: str, text: str) -> float:
    terms = EVENT_KEYWORDS.get(label, [label.replace("_", " ").replace("-", " ")])
    return 1.0 if count_matches(text, terms) > 0 else 0.0


def audience_style_score(audience: str, text: str) -> float:
    terms = AUDIENCE_KEYWORDS.get(audience, [])
    if not terms:
        return 0.0
    return min(1.0, count_matches(text, terms) / 2.0)


def low_vision_spatial_score(text: str) -> float:
    spatial = count_matches(text, SPATIAL_WORDS)
    motion = count_matches(text, MOTION_WORDS)
    return min(1.0, (spatial + motion) / 4.0)


def unsupported_detail_flags(text: str) -> list[str]:
    flags = []
    for pattern in UNSUPPORTED_PATTERNS:
        if re.search(pattern, text):
            flags.append(pattern)
    return flags


def vague_visual_flags(text: str) -> list[str]:
    t = normalize(text)
    return [phrase for phrase in VAGUE_VISUAL_PHRASES if phrase in t]


def sentence_count(text: str) -> int:
    parts = [p for p in re.split(r"[.!?]+", text) if p.strip()]
    return max(1, len(parts))


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def avg_sentence_length(text: str) -> float:
    return word_count(text) / sentence_count(text)


def readability_proxy(text: str) -> float:
    # Simple bounded score: shorter sentences are usually easier for live commentary.
    avg_len = avg_sentence_length(text)
    return max(0.0, min(1.0, 1.0 - max(0.0, avg_len - 18.0) / 22.0))


def slot_coverage(audience: str, text: str) -> float:
    t = normalize(text)
    if audience == "beginner":
        slots = [
            any(w in t for w in ["means", "because", "according to", "in simple terms"]),
            any(w in t for w in ["restart", "next", "now", "will"]),
            not any(w in t for w in ["overload", "half-space", "second-ball"]),
        ]
    elif audience == "expert":
        slots = [
            count_matches(text, TACTICAL_WORDS) > 0,
            any(w in t for w in ["phase", "shape", "spacing", "transition", "set-piece"]),
            "according to the rules" not in t,
        ]
    elif audience == "low_vision":
        slots = [
            count_matches(text, ["ball", "players", "referee"]) > 0,
            count_matches(text, SPATIAL_WORDS + MOTION_WORDS) > 0,
            len(vague_visual_flags(text)) == 0,
        ]
    else:
        slots = []
    return sum(bool(s) for s in slots) / max(1, len(slots))


def score_commentary_row(row: dict) -> dict[str, object]:
    text = row.get("commentary") or ""
    audience = row.get("audience", "")
    label = row.get("event_label", "")
    unsupported = unsupported_detail_flags(text)
    vague = vague_visual_flags(text)
    return {
        "window_id": row.get("window_id"),
        "event_label": label,
        "audience": audience,
        "model": row.get("model") or row.get("generator") or "",
        "word_count": word_count(text),
        "sentence_count": sentence_count(text),
        "avg_sentence_length": avg_sentence_length(text),
        "readability_proxy": readability_proxy(text),
        "event_consistency": event_consistency(label, text),
        "audience_style_score": audience_style_score(audience, text),
        "slot_coverage": slot_coverage(audience, text),
        "low_vision_spatial_score": low_vision_spatial_score(text) if audience == "low_vision" else math.nan,
        "unsupported_detail_count": len(unsupported),
        "unsupported_detail_flags": ";".join(unsupported),
        "vague_visual_count": len(vague),
        "vague_visual_flags": ";".join(vague),
        "commentary": text,
    }


def summarize_scores(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    numeric_keys = [
        "word_count",
        "readability_proxy",
        "event_consistency",
        "audience_style_score",
        "slot_coverage",
        "unsupported_detail_count",
        "vague_visual_count",
    ]
    summary: dict[str, object] = {"rows": len(rows)}
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if row.get(key) == row.get(key)]
        summary[f"{key}_mean"] = sum(vals) / len(vals) if vals else None
    by_audience = {}
    for audience, count in Counter(row["audience"] for row in rows).items():
        subset = [row for row in rows if row["audience"] == audience]
        by_audience[audience] = summarize_scores_shallow(subset)
        by_audience[audience]["count"] = count
    summary["by_audience"] = by_audience
    return summary


def summarize_scores_shallow(rows: list[dict[str, object]]) -> dict[str, float | None]:
    keys = ["event_consistency", "audience_style_score", "slot_coverage", "readability_proxy"]
    out = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if row.get(key) == row.get(key)]
        out[f"{key}_mean"] = sum(vals) / len(vals) if vals else None
    return out
