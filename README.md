# Soccer event understanding with frozen vision models

Can a general-purpose vision model (DINOv2, CLIP), never trained on soccer, tell what kind of event happens in the 60 seconds around a commentary timestamp, in a match it has never seen? This repository turns broadcast video into per-event embeddings and studies them with clustering, a contrastive projection head, and classifiers.

> **Results in `results/archive_pre_fix/` predate the fixes on this branch.** Several of them are invalid: the contrastive head saw test-game labels, frames were cut from the wrong half, and models were picked on the test set. See [docs/leak_audit.md](docs/leak_audit.md) before quoting any number, and rerun the pipeline.

## Setup

Python 3.10+ (3.11 is what CI uses).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
# GPU machines: first install the torch build matching your CUDA driver, e.g.
#   pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .                                       # makes the shared `soccer` package importable
```

`preprocess/transcribe.py` additionally needs `openai-whisper` and ffmpeg; nothing else uses them.

## Data

The data is **SoccerReplay-1988** (1,988 matches, shared under an NDA; its official split is 1,488 train, 250 valid and 250 test).

| What | Where it comes from | Where it goes |
|---|---|---|
| Annotation JSONs | Hugging Face `Homie0609/SoccerReplay-1988` (gated) or the lab's Google Drive copy | `data/json/Soccer_Data_Json/{train,valid,test}/<league>/<match>/` |
| Match videos (`*_1.mkv`, `*_2.mkv`) | Password-protected Dropbox link, downloaded by `scripts/batch_pipeline.py` | `data/videos/<match>/` (deleted after extraction) |
| Extracted frames | `batch_pipeline.py` | `results_by_label_epl_2021_2022/game_XX_half_N/frames/<label>/` |
| Features, results | Pipeline stages | `features/`, `results/` |

`data/`, frames and features are gitignored. All paths are set in `configs/pipeline.yaml`; set `SOCCER_DATA_ROOT` to keep data elsewhere, e.g. on a mounted volume.

### Credentials

Nothing secret is stored in the repository. The Dropbox downloader reads, in order:

1. Environment variables `DROPBOX_TOKEN` (or `DROPBOX_TOKEN_FILE`, or the default `tmp_dbx/token.txt`), `DROPBOX_SHARED_URL` and `DROPBOX_LINK_PASSWORD`.
2. AWS SSM Parameter Store, if `SOCCER_SSM_PREFIX` is set: `<prefix>/dropbox/token`, `<prefix>/dropbox/shared_url` and `<prefix>/dropbox/link_password`, stored as SecureString.

Dropbox tokens are short-lived. When one expires, the scripts stop with exit code 2; set a new token and rerun, and finished games are kept. S3 access uses the standard AWS credential chain (instance or task role, `AWS_PROFILE`).

## Which games, and how they are split

`configs/games.csv` is the single source of truth: one row per match with its game number, match folder, league, official SoccerReplay split, and this project's `train` / `val` / `test` split. Every script reads it.

- **Splits are by whole match.** Events of one match share the stadium, light, kits and camera style, so an event-level split lets a model pass by recognising the match. Tests check that no match is in two splits.
- `val` games choose models, epochs and configurations; `test` games are scored once, at the end.
- `configs/excluded_matches.csv` lists dataset folders that must never be used (a duplicate, and two folders containing another match's annotations). The manifest refuses them.
- To add games, append rows to `configs/games.csv` with the next game numbers. Draw test games from SoccerReplay's official test split if results should be comparable with other work.

Alternative manifests, selected with `SOCCER_MANIFEST=<file>` for any script or run:

| Manifest | Games | Split |
|---|---|---|
| `configs/games.csv` (default) | 30 EPL 2021-22 | by match: train 01–20, val 21–24, test 25–30 |
| `configs/games_venue_split.csv` | the same 30 | leave-venue-out: no test or validation stadium appears in training (`scripts/make_venue_split.py`) |
| `configs/games_epl2122.csv` | all 65 EPL 2021-22 | SoccerReplay's official split: train 50, val 7, test 8 |

## Running the pipeline

```bash
python run_dino_pipeline.py                              # every stage, in order
python run_dino_pipeline.py --stages features cluster    # a subset
python run_dino_pipeline.py --stages frames --local-only # extract from videos already on disk
```

| Stage | Scripts | Hardware |
|---|---|---|
| `frames` | `batch_pipeline.py`: download, extract ±30 s at 1 fps per event (only events of the half each video holds), delete video | CPU, disk |
| `features` | `temporal_clip_features.py` (per-event `(60, 768)` DINOv2 tensors), `extract_frame_features.py` (per-frame) | GPU (CPU works, ~11 min per game) |
| `cluster` | `cluster_analysis.py`, `within_game_clustering.py` | CPU |
| `contrastive` | `contrastive_training.py` (train games only), `evaluate_contrastive.py` (held-out vs train reported separately) | GPU or CPU |
| `classify` | `event_classification.py`, `sequence_classifier.py`, `game_flow_features.py` + `multimodal_classification.py`, `temporal_event_classifier.py` | CPU (sequence models prefer GPU) |

The stage scripts live in `scripts/` and also run on their own; `python scripts/<name>.py --help` lists their options.

Every reported number comes with what it should be compared against:

- **Clustering** is scored against the event labels (ARI, NMI, purity next to its shuffled-cluster purity) *and* against game identity (`Game_ARI`, `Game_NMI`). `Game_ARI` well above `ARI` means the clusters follow the match more than the event.
- **Classifiers** report test accuracy and macro-F1 with the majority-class accuracy and a 95% interval from resampling whole test games.
- **Seeds**: `sequence_classifier.py --seeds 42 43 44` reports mean and sd over seeds. `scripts/robustness.py` repeats the contrastive head and all classifiers over 5 seeds for the match-level and leave-venue-out splits (add `--splits epl2122` for the 65-game set) and writes `results/robustness/summary.md`.

### Where it runs

All heavy work runs on the professor's GPU server; a personal machine only opens an SSH session. By default **the server downloads each game straight from Dropbox**, extracts its frames, deletes the video, then encodes and analyses everything. No AWS account is needed. Optionally, videos can first be staged in S3. Step by step, including SSH key handling: [docs/remote_gpu_server.md](docs/remote_gpu_server.md).

```bash
python scripts/fetch_annotations.py --splits train valid test     # public Google Drive copy
python run_dino_pipeline.py --device cuda                         # direct from Dropbox
# with S3 (needs AWS access):
python scripts/transfer_dropbox_to_s3.py && python run_dino_pipeline.py --from-s3 --upload --device cuda
```

- `infra/terraform/` (S3 option only) defines the bucket (private, KMS-encrypted, versioned, HTTPS-only) and the server's IAM identity, plus an optional Lambda for the transfer. No secret is stored in Terraform.
- `infra/gpu-server-setup.sh` sets up the server without sudo; `infra/ec2-setup.sh` does the same for an EC2 GPU instance.
- Lightweight data preparation, such as producing the train/test split manifest from data in S3, is planned as an AWS Lambda (Phase 5).

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

CI (`.github/workflows/ci.yml`) runs lint, tests and a gitleaks secret scan on every push. It installs the torch-free `requirements-ci.txt`, so the three tests that need torch are skipped there. The tests include an extraction test on a synthetic video, which checks that each saved frame comes from the time in its filename, and end-to-end runs of every analysis script on synthetic data.

## Layout

```
soccer/        shared package: config, manifest, annotations, extraction, Dropbox, storage,
               encoders, encoding, features, metrics, evaluation protocol
configs/       pipeline.yaml, games.csv (manifest), excluded_matches.csv
run_dino_pipeline.py  entry point: runs the stages in order
scripts/       pipeline stage scripts (frames, features, clustering, contrastive, classifiers, commentary)
infra/         Terraform (bucket, IAM, optional Lambda), GPU server and EC2 setup scripts
preprocess/    single-video extraction CLI, distribution plots, Whisper transcription
tests/         unit and smoke tests
docs/          remote_gpu_server.md, leak_audit.md, cleanup_log.md
results/       run outputs (gitignored); archive_pre_fix/ holds the old, partly invalid results
```
