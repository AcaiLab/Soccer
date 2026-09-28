# Review: Audience-Adaptive Commentary (July 9 drop)

Reviewed 54,669 commentary records (18,223 windows x 3 audiences) across 256 games
from the shared Drive folder.

**Coverage caveat:** the folder contains 8 shards (0000-0007). Shards 0000-0003
downloaded fine; every file in shards 0004-0007 is blocked for programmatic
download ("cannot retrieve the public link — change permission to 'Anyone with
the link'"). This review therefore covers 4 of 8 shards (~half the data). Given
the extreme templating (see below), the findings almost certainly generalize,
but the remaining shards should be spot-checked once sharing is fixed.

## What works

1. **Grounding is faithful.** In the `low_vision` variant, the camera/motion
   phrases ("wide field view with medium motion") match the underlying
   `basic_visual_cues.jsonl` facts in **100% of 17,650 checked windows** — no
   hallucinated visual claims. (The remaining ~570 use a grounded
   "closeup or nonfield view" variant phrase.)
2. **Audience differentiation is real and consistent.** Beginner gets rule
   context ("a yellow card is an official warning"), expert gets tactical
   framing ("the caution can change how aggressively that player challenges"),
   low_vision gets explicit visual state. The register separation is clean.
3. **Event-level facts are accurate** for the well-defined labels (goal,
   corner, yellow_card, off_side, save): the stated rule/consequence is
   always correct for the label.

## Problems

### 1. Pipeline jargon leaks into the expert commentary (51.3% of expert records)
9,354 records — all of them in the `expert` variant, i.e. more than half of
all expert commentary — include sentences like *"The retrieval memory finds 1
close example of the same event type, so this pattern is worth comparing
across matches."* This is internal RAG bookkeeping, not commentary; no viewer
should see it. Should be stripped from the expert generation template.

### 2. Heavily templated — naturalness is low
Across 18,223 beginner texts there are only **78 unique strings (45 unique
sentence skeletons)**. Expert: 270 unique / 189 skeletons. Every text follows
the same 3-sentence frame ("X occurs. In simple terms, Y. The next thing to
watch is Z."). Accurate, but it reads as slot-filling, not commentary — over a
match this would feel robotic. Needs surface variation (paraphrase sampling,
temperature, or an LLM rewrite pass).

### 3. Fallback template produces incorrect/nonsense text for meta labels
Labels without a hand-written fact entry fall through to a generic template
that is *wrong* for them:

- `end_of_half_game` -> "players reset for the next action" (no — the half is over)
- `show_added_time`, `statistics_and_summary` -> described as gameplay phases
  (they are broadcast graphics)
- `unknown` -> *"a unknown occurs"*; `wcl-icon-settings-info-rounded` ->
  *"A wcl icon settings info rounded occurs"* — raw annotation junk read aloud.
  (Same two junk labels we flagged in the classification data; they should be
  filtered upstream.)

Also grammar slips in the openers: "a unknown", "a end of half game" (a/an).

### 4. low_vision variant doesn't yet serve its audience
All object-level visual facts (`ball_visible`, `referee_visible`,
`card_visible`) are **null in all 18,223 cue windows (100%)**, so the only visual detail
available is camera framing + motion intensity. "Wide field view with medium
motion" doesn't help a low-vision viewer follow the game — they need *what* is
happening (who has the ball, where on the pitch, what the referee is doing).
The template is fine; the extraction upstream needs to fill the object facts.

## Suggested priorities

1. Strip retrieval/memory sentences from all variants (one-line template fix).
2. Filter or remap the junk/meta labels before generation.
3. Write correct fact entries for `end_of_half_game`, `show_added_time`,
   `statistics_and_summary`, `throw_in`, `ball_out_of_play`.
4. Add surface variation (paraphrase pool or LLM rewrite) for naturalness.
5. Populate object-level visual facts so low_vision commentary can describe
   the actual scene.
