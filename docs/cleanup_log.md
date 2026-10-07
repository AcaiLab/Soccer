# Cleanup log: cleanup/dino-contrastive

Cumulative record of everything removed or moved from `dino-contrastive-pipeline`.

- Branched from `dino-contrastive-pipeline` @ `dd3f65f`
- Backup tag: `backup/dino-contrastive-pre-cleanup` → `dd3f65f`
- Any removed file can be recovered with `git checkout backup/dino-contrastive-pre-cleanup -- <path>`

## Phase 1: review (2026-10-07)

No changes made.

## Phase 2: cleanup (2026-10-07)

Approved by Hasset Admasu on 2026-10-07: delete files that are not relevant at all.

### Commit 1: extend .gitignore

Nothing removed. Added ignore rules for credentials (`.env`, `*.pem`, `*.key`, `.aws/`, `token.txt`), raw media (`*.mkv`, `*.mp4`), model weights and arrays (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.npy`, `*.npz`, `*.pkl`) and experiment-tracking output (`wandb/`, `mlruns/`, `runs/`, `lightning_logs/`). Checked that none of the new rules match a file that is already tracked.

### Commit 2: remove superseded extraction scripts

| path | size | type | reason | confidence | recoverable? |
|---|---|---|---|---|---|
| `preprocess/extract_clips_frames.py` | 3.9 KB | code | Older ±15 s version of `extract_clips_frames_label.py` that does not sort frames by label. No script references it. The ±15 s version is also still on `main`. | high | yes, in git history |
| `preprocess/newExtractFrames.py` | 5.5 KB | code | Colab copy of the extractor with hardcoded `/content/` paths. No script references it. Its event handling (no `half` filter) is the same as the kept extractor, so it adds nothing to the half-bug investigation. | high | yes, in git history |
| `extract_remaining.py` | 5.3 KB | code | One-off rerun of frame extraction for games 34–40, duplicating `batch_pipeline.py`. No script references it. | high | yes, in git history |

Kept after review: `clip_feature_extraction.py`. Its outputs (`features/clip_features.npy`, `features/metadata.csv`) are the default inputs of `cluster_analysis.py`, which produces the CLIP baseline.

### Commit 3: remove duplicate result files

Each file was checked with `git hash-object` immediately before removal and is byte-identical to the copy that was kept.

| path | size | type | reason | confidence | recoverable? |
|---|---|---|---|---|---|
| `clustering/classification/multimodal_results.csv` | 0.6 KB | duplicate | Identical to `clustering/classification_30games/multimodal_results.csv`, the copy the course cites | high | yes, in git history |
| `clustering/classification/multimodal_classification_report.txt` | 2.1 KB | duplicate | Identical to the copy in `clustering/classification_30games/` | high | yes, in git history |
| `clustering/classification/multimodal_comparison.png` | 70.6 KB | duplicate | Identical to the copy in `clustering/classification_30games/` | high | yes, in git history |
| `clustering/classification/temporal_classification_results.csv` | 0.7 KB | duplicate | Identical to the copy in `clustering/classification_full_dataset/` | high | yes, in git history |
| `clustering/classification/temporal_classification_report.txt` | 2.2 KB | duplicate | Identical to the copy in `clustering/classification_full_dataset/` | high | yes, in git history |
| `clustering/classification/temporal_feature_comparison.png` | 66.4 KB | duplicate | Identical to the copy in `clustering/classification_full_dataset/` | high | yes, in git history |
| `clustering/classification/feature_importance.png` | 56.3 KB | duplicate | Identical to the copy in `clustering/classification_full_dataset/` | high | yes, in git history |

Note: by default the classifier scripts still write to `clustering/classification/`, so rerunning them will recreate these files there.

### Phase 2 totals

| files removed | size removed | tracked files remaining |
|---|---|---|
| 10 | about 0.22 MB | 113 (122 at the start, minus 10 removed, plus `docs/cleanup_log.md`) |
