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
