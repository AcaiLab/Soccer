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

## Phase 3: leak audit (2026-10-07)

Nothing removed or moved. Findings are in `docs/leak_audit.md`. Code change: `preprocess/extract_clips_frames_label.py` now extracts only the events of the half the video holds (commit `1f19f8a`).

| files removed | size removed | tracked files remaining |
|---|---|---|
| 0 | 0 | 114 |

## Phase 4: pipeline-ready refactor (2026-10-07)

Requested by Hasset Admasu on 2026-10-07 ("fix everything that you see wrong, especially conventional practices"). **No file was deleted or moved.** Code was fixed in place, and new files were added. Details are in each commit message (`86a0d9c`..`HEAD`) and the fixed findings are marked in `docs/leak_audit.md`.

### Files reduced to forwarders

These now only call the script that replaced them, so existing commands keep working. Their former logic is recoverable from the backup tag.

| path | now forwards to | why |
|---|---|---|
| `process_10_games.py` | `batch_pipeline.py --local-only --games 01..10` | duplicated extraction logic, string-patched the extractor |
| `process_new_games.py` | `batch_pipeline.py --local-only --games ...` (games looked up in the manifest) | same |
| `clip_feature_extraction.py` | `extract_frame_features.py --model openai/clip-vit-base-patch32` | duplicate of the model-agnostic extractor |

### Outputs no longer produced

| output | was produced by | why removed |
|---|---|---|
| `clip_temporal.pkl` | `temporal_clip_features.py` | nothing read it; pickle files can run code when loaded |
| "attention_pool" results | `event_classification.py`, `multimodal_classification.py` | mathematically equal to mean pooling for unit-length frames |
| `next_event`, `n_next_attacks` columns; pre-encoded `*_enc` columns and `game_flow_features.npz` | `game_flow_features.py` | future information; encoders must be fit on training rows only |
| `dinov2_improvement_heatmap.png` | `plot_dinov2_vs_clip.py` | percent change on near-zero ARI exaggerates differences |
| hardcoded external baselines (13.75%, 30.54%) on plots | classifier scripts | different protocol and test set; not comparable |

### Proposed for removal or move: awaiting approval, nothing done

| path | size | type | reason | confidence | recoverable? |
|---|---|---|---|---|---|
| `process_10_games.py` | 0.5 KB | code | forwarder; `batch_pipeline.py --local-only` replaces it | high | yes, in git history |
| `process_new_games.py` | 1.3 KB | code | forwarder; same | high | yes, in git history |
| `clip_feature_extraction.py` | 0.7 KB | code | forwarder; `extract_frame_features.py` replaces it | high | yes, in git history |
| `fast_batch_pipeline.py` | 4.6 KB | code | duplicates `batch_pipeline.py`; driven by a gitignored queue file | medium: keep if the queue workflow is still used for games 31–68 | yes, in git history |
| `soccerProjNewCode-1.py` | 9.0 KB | code | superseded by `soccer/extract.py`; hyphenated name is not importable; also on `main` | medium | yes, in git history and on `main` |
| `clustering/` (all files) | 15 MB | results | pre-fix results, several invalid (leak audit B1–B5) | low: the course cites them; suggest moving to `results/archive_pre_fix/` instead of deleting | yes, in git history |
| move root scripts into `scripts/` (or `src/`) | n/a | layout | conventional layout | low: changes every documented command | yes, via `git mv` |

### Phase 4 totals

| files removed | files moved | files added | tracked files |
|---|---|---|---|
| 0 | 0 | 28 | 142 (after the leak-audit and log updates) |

## Phase 4b: approved removals and moves (2026-10-07)

Approved by Hasset Admasu on 2026-10-07 ("make all the changes").

### Commit: remove superseded scripts

| path | size | type | reason | confidence | recoverable? |
|---|---|---|---|---|---|
| `process_10_games.py` | 0.5 KB | code | forwarder; `batch_pipeline.py --local-only --games 01 ... 10` replaces it | high | yes, in git history |
| `process_new_games.py` | 1.3 KB | code | forwarder; `batch_pipeline.py --local-only --games ...` replaces it | high | yes, in git history |
| `clip_feature_extraction.py` | 0.7 KB | code | forwarder; `extract_frame_features.py --model openai/clip-vit-base-patch32` replaces it | high | yes, in git history |
| `fast_batch_pipeline.py` | 4.6 KB | code | duplicates `batch_pipeline.py`; its queue file is replaced by `configs/games.csv` | medium | yes, in git history |
| `soccerProjNewCode-1.py` | 9.0 KB | code | superseded by `soccer/extract.py`; hyphenated name not importable | medium | yes, in git history and on `main` |

### Commit: move pre-fix results to an archive

| from | to | size | reason |
|---|---|---|---|
| `clustering/` (77 files) | `results/archive_pre_fix/` | 15 MB | pre-fix results, several invalid (leak audit B1-B5); kept because the write-up cites them. New runs write to `results/<name>/`, which is gitignored; the archive stays tracked and has a README naming what is invalid. |

### Commit: conventional layout

Moves with `git mv`, so history follows each file. Nothing removed.

| from | to | note |
|---|---|---|
| 17 stage scripts at the repo root | `scripts/` | run as `python scripts/<name>.py`; `run_dino_pipeline.py` stays at the root as the entry point |
| `model_encoders.py` | `soccer/encoders.py` | library code, not a script |
| frame-encoding helpers in `temporal_clip_features.py` | `soccer/encoding.py` | shared by three scripts; torch-free |
| `ec2-setup.sh` | `infra/ec2-setup.sh` | machine setup |

## Phase 4c: S3 staging and the GPU server workflow (2026-10-07)

Nothing removed or moved. Added: `soccer/transfer.py` and `scripts/transfer_dropbox_to_s3.py` (Dropbox → S3 streaming), `scripts/fetch_annotations.py`, S3 options on `batch_pipeline.py` and `run_dino_pipeline.py`, `infra/terraform/` (bucket, IAM, optional Lambda), `infra/gpu-server-setup.sh`, `infra/build_lambda.sh`, `docs/remote_gpu_server.md`.

## Phase 4d: Dropbox credentials that do not expire (2026-10-08)

Nothing removed or moved. Added `scripts/dropbox_auth.py` (a one-time PKCE authorisation that saves a refresh token to the gitignored `tmp_dbx/oauth.json`) and refresh-token support in `soccer/dropbox.py`: an access token is fetched when needed, renewed shortly before it expires, and renewed again if Dropbox refuses it. Fixed access tokens still work and, when given for one run, take priority. A missing permission (`missing_scope`) is now reported as such instead of as an expired token. `tests/test_dropbox_refresh.py` and `tests/test_dropbox_auth.py` cover this with a fake HTTP session, so no network is used.

## Phase 4e: corrected frames become the default (2026-10-11)

Approved by Hasset on 2026-10-11 ("i approve 1 and 2"). `configs/pipeline.yaml` now points `paths.frames` at `frames_v2`, the frames chosen by timestamp (docs/leak_audit.md, B19); docs, tests and setup scripts follow.

| path | size | type | reason | confidence | recoverable? |
|---|---|---|---|---|---|
| server `~/soccer-data/results_by_label_epl_2021_2022/` | 23 GB, 266,967 files (130 halves) | extracted frames | chosen by frame index: 41,509 frames from the wrong moment in 42 halves; replaced by `frames_v2` | high | only by re-extracting from video with the code before `ee6935f` |
| server `~/.config/soccer/dropbox_token` | 1.5 KB | expired 4-hour Dropbox access token | replaced by the refresh token in `dropbox_oauth.json`; the env line pointing to it is commented out, not removed | high | no, and not needed |
| PC `tmp_dbx/token.txt` (gitignored) | 1.5 KB | expired 4-hour Dropbox access token | same | high | no, and not needed |

Kept: the old features and results computed from the removed frames (`features/`, `results/`, `runs/epl2122/`), so before/after comparisons stay possible.
