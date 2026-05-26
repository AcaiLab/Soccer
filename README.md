# Soccer Event Clustering with DINOv2 and Contrastive Learning

A research pipeline for extracting visual embeddings from soccer broadcast video, clustering events by type across games, and using contrastive learning (triplet loss) to disentangle event semantics from game-level visual bias.

**Dataset:** 10 EPL 2021-2022 matches | 627 events | 18,501 frames  
**Models:** DINOv2 (`facebook/dinov2-base`, 768-dim) · CLIP (`openai/clip-vit-base-patch32`, 512-dim)

---

## Results Summary

| Method | Purity | ARI | Silhouette |
|---|---|---|---|
| CLIP (baseline) | 0.250 | 0.026 | 0.184 |
| DINOv2 mean-pool | 0.271 | 0.036 | 0.059 |
| DINOv2 concat+PCA | 0.324 | 0.072 | 0.034 |
| Within-game (DINOv2) | 0.450 | 0.072 | 0.135 |
| **Contrastive (triplet loss)** | **0.973** | **0.802** | **0.490** |

Contrastive learning with triplet loss achieves a **+259% purity** and **+2,142% ARI** improvement over the raw DINOv2 baseline by training the model to pull same-type events together across games and push different events apart within the same game.

---

## Pipeline Overview

```
data/videos/{game}/
  *.mkv  (video halves)       +       *.json  (event annotations)
                    │
                    ▼
        preprocess/extract_clips_frames_label.py
        process_10_games.py  ── runs above for all 10 games ──▶
                    │
                    ▼
    results_by_label_epl_2021_2022/
      game_NN_half_N/frames/{event_type}/*.png   (18,501 frames)
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
  extract_frame_features.py    temporal_clip_features.py
  (flat: 18501 × 768)          (temporal: 627 × 60 × 768)
          │                    │
          └─────────┬──────────┘
                    ▼
          run_dino_pipeline.py   (orchestrates all 3 stages)
                    │
          ┌─────────┴──────────────────────┐
          ▼                                ▼
  cluster_analysis.py              contrastive_training.py
  (K-Means + t-SNE clustering)     (triplet loss projection head)
          │                                │
          ▼                                ▼
  clustering/dinov2-base/          evaluate_contrastive.py
  clustering/within_game/          clustering/contrastive/
```

---

## Repository Structure

```
Soccer/
├── preprocess/
│   ├── extract_clips_frames_label.py  # extract ±30s clips + frames per event
│   ├── extract_clips_frames.py        # original prototype (single game)
│   ├── plot_event_distribution.py     # visualize event label distribution
│   └── transcribe.py                  # Whisper audio transcription (optional)
│
├── process_10_games.py                # batch-process all 10 games
│
├── model_encoders.py                  # unified encoder: CLIP + DINOv2 via build_encoder()
├── extract_frame_features.py          # flat per-frame feature extraction (any model)
├── temporal_clip_features.py          # per-event temporal feature extraction (any model)
├── clip_feature_extraction.py         # CLIP-specific flat extractor (legacy)
│
├── run_dino_pipeline.py               # end-to-end reproducible pipeline
│
├── cluster_analysis.py                # K-Means + t-SNE across 4 representations
├── within_game_clustering.py          # per-game clustering (removes cross-game bias)
├── contrastive_training.py            # triplet loss training pipeline
├── evaluate_contrastive.py            # before/after contrastive evaluation
│
├── plot_dinov2_vs_clip.py             # CLIP vs DINOv2 comparison charts
│
├── clustering/                        # output visualizations (committed)
│   ├── dinov2-base/                   # DINOv2 K-Means + t-SNE plots
│   ├── within_game/                   # per-game clustering plots
│   ├── contrastive/                   # before/after contrastive plots
│   ├── dinov2_vs_clip_comparison.png
│   └── dinov2_improvement_heatmap.png
│
├── requirements.txt
└── README.md

# NOT committed (too large / generated):
# data/                               ~12 GB  raw videos + JSON annotations
# results_by_label_epl_2021_2022/     ~22 GB  extracted frames
# features/                           ~358 MB feature matrices + model weights
```

---

## Setup

```bash
git clone <repo-url>
cd Soccer
pip install -r requirements.txt
```

> **Data access:** Raw videos and JSON annotations are not stored in this repo due to size.
> Contact the team for access to the SoccerNet EPL 2021-2022 data or download from [SoccerNet](https://www.soccer-net.org/).

---

## Running the Pipeline

### Step 0 — Extract frames from video (requires raw data)
```bash
python process_10_games.py
```
Processes all 10 games and outputs frames to `results_by_label_epl_2021_2022/`.

### Step 1 — Extract DINOv2 features (full pipeline)
```bash
python run_dino_pipeline.py --model facebook/dinov2-base --batch-size 32
```
Runs all 3 stages: temporal features → flat features → clustering visualizations.

To re-run clustering only (features already extracted):
```bash
python run_dino_pipeline.py --skip-extraction
```

### Step 2 — Within-game clustering
```bash
python within_game_clustering.py
```
Clusters events separately within each game. Outputs to `clustering/within_game/`.

### Step 3 — Contrastive learning (triplet loss)
```bash
python contrastive_training.py --epochs 200 --lr 1e-3 --margin 0.5
```
Trains a projection head (768 → 512 → 256) using triplets:
- **Anchor:** Event A, Game X
- **Positive:** Event A, Game Y *(same event type, different game)*
- **Negative:** Event B, Game X *(different event type, same game)*

### Step 4 — Evaluate contrastive embeddings
```bash
python evaluate_contrastive.py
```
Produces before/after t-SNE plots and metrics comparison. Outputs to `clustering/contrastive/`.

---

## Key Design Decisions

### Triplet Construction
Game-level visual bias (stadium lighting, kit colors, camera angles) dominates raw embeddings. The triplet formulation explicitly forces the model to be **invariant to game identity** while being **sensitive to event type**:

```
distance(anchor, positive) < distance(anchor, negative) + margin
```

where the positive is the same event in a different game and the negative is a different event in the same game.

### Encoder Architecture
`model_encoders.py` provides a unified `BaseEncoder` interface with two implementations:
- `CLIPEncoder` — visual projection of CLIP ViT-B/32 (512-dim, L2-normalized)
- `DINOv2Encoder` — CLS token of DINOv2 ViT-B (768-dim, L2-normalized)

Switch models via `--model facebook/dinov2-large` (1024-dim) or `--model openai/clip-vit-base-patch32`.

### Representations Compared
| ID | Description | Shape |
|---|---|---|
| A | Individual frames | (18501, 768) |
| B | Event mean-pool | (627, 768) |
| C | Event concat + PCA | (627, 768) |
| F | Normalized mean-pool (per-game centering) | (627, 768) |
| Contrastive | Triplet-projected | (627, 256) |

---

## Visualizations (in `clustering/`)

| File | Description |
|---|---|
| `dinov2-base/A_individual_frames.png` | t-SNE of all 18,501 frames |
| `dinov2-base/B_event_meanpool.png` | t-SNE of 627 event embeddings |
| `dinov2-base/D_colorings_game_half.png` | 3-panel: by label, game, half |
| `dinov2-base/per_game/game_NN.png` | Each game highlighted on global t-SNE |
| `within_game/game_NN.png` | Within-game clustering per game |
| `contrastive/before_vs_after_labels.png` | DINOv2 vs contrastive, colored by label |
| `contrastive/before_vs_after_games.png` | DINOv2 vs contrastive, colored by game |
| `contrastive/contrastive_3panel.png` | Post-contrastive: label, game, K-Means |
| `contrastive/metrics_comparison.png` | Bar chart: before vs after |
| `dinov2_vs_clip_comparison.png` | CLIP vs DINOv2 across all metrics |
| `dinov2_improvement_heatmap.png` | % improvement heatmap |

---

## Event Labels (22 types)

`ball_out_of_play` · `ball_possession` · `clearance` · `corner` · `end_of_half_game` ·
`foul_lead_to_penalty` · `foul_with_no_card` · `free_kick` · `goal` · `injury` ·
`lead_to_corner` · `off_side` · `red_card` · `saved_by_goal-keeper` · `shot_off_target` ·
`show_added_time` · `start_of_half_game` · `statistics_and_summary` · `substitution` ·
`unknown` · `var` · `yellow_card`