# Leak audit: dino-contrastive-pipeline

- Audited: `dino-contrastive-pipeline` @ `dd3f65f`, its full history, and the other branches in the GitHub repo
- Date: 2026-10-07
- Working branch for fixes: `cleanup/dino-contrastive`

The audit has two parts: (A) secrets and personal data in git, and (B) machine-learning data leakage and data-correctness problems that affect whether the clusters and accuracies can be trusted. Secret values are redacted everywhere in this document.

---

## A. Secrets and personal data

### Method

- `gitleaks` 8.30.1, run over the history of `dino-contrastive-pipeline`, over all branches (`--all`), and over the current tracked files
- Grep over every commit on every branch for AWS keys (`AKIA…`, `ASIA…`, `aws_secret_access_key`), private keys, Hugging Face / OpenAI / GitHub / Slack / Dropbox token formats, and `password`, `secret`, `token`, `api_key` assignments
- File-name check over all history for `.env`, `*.pem`, `*.key`, credential files, notebooks, and `wandb/` or `mlruns/` output
- Email, personal-path, and AWS account ID check

### Findings

| # | Secret | File and line | First commit | Also in | Found by | Severity |
|---|---|---|---|---|---|---|
| A1 | Dropbox shared-link access key (`rlkey=` in `SHARED_URL`) | `download_dropbox_games.py:22` | `eaa1137` | `dd3f65f`, cleanup commits | gitleaks + grep | high |
| A2 | Same `rlkey` | `batch_pipeline.py:45`, `fast_batch_pipeline.py:34` | `dd3f65f` | cleanup commits | gitleaks + grep | high |
| A3 | Dropbox shared-link password (`LINK_PASSWORD = "…"`) | `download_dropbox_games.py:24`, `batch_pipeline.py:47`, `fast_batch_pipeline.py:36` | `eaa1137` (first file), `dd3f65f` (others) | cleanup commits | grep only; gitleaks does not flag it | high |
| A4 | AWS account ID inside the bucket name | `ec2-setup.sh:5` | `dd3f65f` | | grep | low: an identifier, not a credential |
| A5 | Dropbox API token synced from S3 by the setup script (`aws s3 sync $BUCKET/config/tmp_dbx/`) | `ec2-setup.sh:20` | `dd3f65f` | | review | medium: the token is stored in S3 under `config/tmp_dbx/` |

A1–A3 together give anyone who can read the GitHub repository download access to the shared SoccerReplay-1988 video folder. The dataset is distributed under an NDA, so this matters even though the link is not ours.

### Nothing found

- No AWS access keys, private keys, Hugging Face / OpenAI / GitHub / Slack tokens, or `.env` / key / credential files in any commit on any branch
- No notebooks, `wandb/` or `mlruns/` output in history
- No email addresses in any commit
- Personal data: only local usernames inside file paths (`/Users/yuntingyin/…`, `c:/Users/tsega/Downloads/…` in `fix_labeled_issues.py`). Low sensitivity. The match JSONs, which contain player and referee names, have never been committed.

### Remediation, in order

1. **Rotate first.** The link and its password are public in GitHub history, so removing them from the code does not revoke anything.
   - Ask the dataset provider (or whoever shared the folder) to disable this shared link and issue a new link and password.
   - The Dropbox API token that was pasted into the Claude Code session on 2026-10-07 should be revoked in the Dropbox app console once downloads are finished.
   - Rotate whatever Dropbox token is stored at `s3://…/config/tmp_dbx/`.
2. **Stop new copies.** Move `SHARED_URL` and `LINK_PASSWORD` out of the code into environment variables or AWS SSM Parameter Store (SecureString) or Secrets Manager, read at run time. This is planned for Phase 4 and needs your approval because it changes how the scripts are run.
3. **History removal, optional and only with explicit confirmation.**
   - `git filter-repo --replace-text` with a file listing the 2 secrets, run on a fresh mirror clone, then force-push every affected branch and tag.
   - Everyone with a clone must re-clone, and GitHub may keep cached views for a while.
   - Because rotation (step 1) is what actually removes the risk, history rewriting is clean-up, not protection. **No history has been rewritten.**

---

## B. Machine-learning leakage and data correctness

### Summary

| # | Finding | Type | Severity | Affects | Status |
|---|---|---|---|---|---|
| B1 | Contrastive head trained on the labels of the events it is evaluated on | leakage | critical | contrastive purity 0.973 / ARI 0.80; contrastive classifier 96.2% | open, fix in Phase 4 |
| B2 | `evaluate_contrastive.py` scores training and held-out events together | leakage | high | every contrastive metric, including `contrastive_30games_leakfree` | open, fix in Phase 4 |
| B3 | Frame extractor cuts every event from both halves' videos | data correctness (label noise) | critical | all 30- and 66-game features and results | **fixed** on `cleanup/dino-contrastive` |
| B4 | Pilot games 01–10 used a ±15 s window, later games ±30 s | per-game confound | high | 30- and 66-game clustering and classification | open: re-extract all games |
| B5 | Best model chosen by test accuracy, then reported on that same test | selection on test | medium | `event_classification.py`, `multimodal_classification.py`, `sequence_classifier.py` reports | open, fix in Phase 4 |
| B6 | 5 of 6 held-out games share a home stadium with training games | weak group split for the invariance claim | medium | every 30-game held-out result | recommendation below |
| B7 | Dataset duplicates and mismatched annotations in SoccerReplay-1988 | duplicates / wrong labels | medium | any game set that includes them | exclude in the games manifest |
| B8 | No games manifest; games numbered by download order; the JSON lookup searches the official `test/` and `valid/` folders too | reproducibility, possible split leak | medium | games 31–68 unverifiable | open, fix in Phase 4 |
| B9 | Game-flow features use the true labels of previous events (teacher forcing) | target information at inference | medium | annotation-only and multimodal results | report as upper bound |
| B10 | Future-information columns (`next_event`, `n_next_attacks`) computed next to the features | future leakage, latent | low | none today (not in the feature list) | guard in Phase 4 |
| B11 | Categorical encoders fit on all games before the split | preprocessing before split | low | game-flow features | fix in Phase 4 |

### B1. Contrastive head trained on test-game labels (critical)

- `contrastive_training.py` mines triplets from every game unless `--train-games` is given. Its own help text says "Default: all games (leaky for downstream eval)".
- The triplets are built from event labels: same label in a different game for the positive, different label in the same game for the negative. That makes the head a supervised model of the labels.
- With the default, it is trained on the labels of the very events that clustering and classification later score.
- **Evidence in committed results:**

  | Run | Result |
  |---|---|
  | `clustering/contrastive/contrastive_metrics.csv` (10 games, all games used for mining) | purity 0.973, ARI 0.802 |
  | `clustering/classification/classification_results.csv` | contrastive embeddings 96.2% vs raw DINOv2 27.4% on "held-out" games 09–10, whose labels the head had seen |
  | `clustering/contrastive_30games_leakfree/contrastive_metrics.csv` (mining restricted to training games) | purity 0.453, ARI 0.202 |
  | Reclustering of held-out games 25–30 only (reported in the project's course notes, from features not in git) | purity 0.217 after the head vs 0.231 before it |

- **Conclusion:** the head does not transfer to unseen matches. The 0.973, 0.80 and 96% figures should not be quoted.
- **Fix:** make `--train-games` required, or default it to the manifest's train split. Save the head's train game list next to its weights, and refuse to evaluate on any game in that list.

### B2. Evaluation mixes training and held-out events (high)

- `evaluate_contrastive.py` clusters all events, including those the head was trained on, and reports one number.
- Even the `contrastive_30games_leakfree` run therefore averages a near-perfect fit on training games (about 0.85) with no gain on held-out games (about 0.22). That is how 0.453 looks like a gain.
- **Fix:** report train-game and held-out-game metrics separately, and put held-out first.

### B3. Every event is also cut from the other half's video (critical, fixed)

**Cause:**
- Match JSON timestamps restart at 0:00 in each half. Confirmed on game 01: half 1 runs 00:01–46:53, half 2 runs 00:01–48:11.
- Each video file holds one half.
- `preprocess/extract_clips_frames_label.py` ignored the comment's `half` field. Every wrapper (`process_10_games.py`, `process_new_games.py`, `batch_pipeline.py`, `fast_batch_pipeline.py`) runs it once per half with the full JSON.
- So a second-half corner at 07:09 was also cut from 07:09 of the first-half video, under the label `corner`.

**Reproduced on game 01** (Arsenal v Chelsea, 2021-08-22), using the real video and JSON and the same patches `batch_pipeline.py` applies:

| | Before the fix | After the fix |
|---|---|---|
| Events in the JSON with a timestamp | 65 (30 + 35) | 65 |
| Events produced | 122 | 65 |
| Events cut from the wrong half's video | **57 (47%)** | **0** |
| Frames per event (mean) | 59.4 | 58.9 |

**Consistency with committed results:** games 11–30 average (3,309 − 627) / 20 = 134 events per game, against 63 for the pilot games 01–10. That matches the doubling.

**Effect:**
- About half the events in games 11 onward are a random moment from the same match carrying a wrong label.
- This is label noise, not train/test leakage, because the wrong frames come from the same match.
- It pushes purity and ARI against event labels down while leaving match identity untouched, so it exaggerates the "clusters by match, not by event" finding.

**Fix** (`cleanup/dino-contrastive`):
- The extractor reads the half from the video file name (`*_1.mkv` / `*_2.mkv`), or from an explicit `VIDEO_HALF`, and skips the other half's comments. All lines the wrappers patch are unchanged.
- Verified on game 01: 65 events, all from the correct half (the "after" column above). The extractor logs how many comments it skips because they belong to the other half.
- The existing frames in S3 for games 11 onward cannot be reused; they must be re-extracted from video.
- Frame seeking was checked and is accurate: frame-index seeks give pixel-identical frames at 95 s, 1,210 s and 2,600 s.

### B4. Mixed window sizes (high)

- The pilot features (games 01–10) have 63 events per game, the correct count, and about 29.5 frames per event (minimum 16). That fits a ±15 s window (at most 31 frames). `main` still has the ±15 s extractor.
- Games 11 onward use ±30 s, about 60 frames per event.
- No code in the repository reproduces the pilot: every wrapper has the B3 bug, which the pilot does not show.
- **Effect:** window length differs by game group, which is a per-game signal the embeddings can pick up and cluster on.
- **Fix:** re-extract all games with one window, recorded in config and in the features' metadata.

### B5. Model selection on the test set (medium)

- `event_classification.py` tries 5 pooling methods × 3 classifiers, `multimodal_classification.py` tries 3 feature sets × 3 classifiers, and `sequence_classifier.py` tries 4 models.
- Each sorts by **test** accuracy and reports the best as the headline, with a per-class report and confusion matrix.
- Choosing the maximum over 9–15 configurations on the test set is optimistic. `sequence_classifier.py` does use a validation split for early stopping, but still picks the reported model by test accuracy.
- **Fix:** choose the configuration on the validation games, then report the test score of that one configuration. Show all others as validation-only.

### B6. Stadium overlap in the held-out games (medium)

The held-out games 25–30 are separate matches, but 5 of 6 are at stadiums that also appear in training:

| Test game | Home team | Home team also in games 01–24? |
|---|---|---|
| 25 | West Ham | yes |
| 26 | Man City | yes |
| 27 | Tottenham | yes |
| 28 | Chelsea | no |
| 29 | Man City | yes |
| 30 | Brentford | yes |

- The project's central claim is that embeddings encode the venue rather than the event.
- A held-out set that shares venues with training can still reward venue recognition, so it is a weak test of that claim.
- See the split recommendation below.

### B7. Problems in the source dataset (medium)

Found by comparing every JSON file name in SoccerReplay-1988 (1,988 matches) with its folder:

| Problem | Folders | Split |
|---|---|---|
| Same match stored twice | `europe_champions-league_2018-2019/2018-11-27_aek-fc-afc-ajax-champions-league` and `…-champions-league-2018-2019` | both train |
| Folder holds another match's annotations (Bayern v Barcelona, 2022-09-13) | `europe_champions-league_2022-2023/2023-03-08_bayern-munchen-paris-saint-germain-champions-league-2022-2023` | train |
| Folder holds another match's annotations (Napoli v Inter, 2023-05-21) | `italy_serie-a_2022-2023/2023-05-22_empoli-fc-juventus-fc-serie-a-2022-2023` | train |

- The duplicate can end up on both sides of a split.
- The two mismatched folders would pair one match's video with another match's labels.
- Neither is in games 01–30. Exclude all three in the games manifest.
- Man City v Tottenham on 2019-04-17 (Champions League) and 2019-04-20 (Premier League) are different matches at the same stadium three days apart. Keep them on the same side of any split.

### B8. No games manifest; official splits mixed (medium)

- Games are numbered `game_NN` in download order. The mapping from numbers to matches for games 31–68 lives only in `tmp_dbx/download_queue.json`, which is gitignored and not available.
- `find_json_file` in `batch_pipeline.py` and `fast_batch_pipeline.py` looks in SoccerReplay's `train/`, `test/` and `valid/` folders.
- **Checked:** games 01–30 are all in the official **train** split, so there is no benchmark-test leak there.
- Games 31–68 cannot be checked without the queue file.
- **Fix:** one committed manifest (match ID, league, official split, our split, excluded with a reason), which every script reads.

### B9–B11. Annotation-feature issues

**B9. Teacher forcing (medium).** `prev_1`–`prev_3` and the rolling counts are computed from the true labels of earlier events. A deployed system would only have its own predictions. Report annotation-only and multimodal accuracies as upper bounds.

**B10. Future columns (low).** `game_flow_features.py` writes `next_event` and `n_next_attacks` into the same CSV as the features. They are not in today's feature lists. Add an assertion that rejects them.

**B11. Encoders fit before the split (low).**
- `LabelEncoder` for the `prev_*` columns is fit on all games, in `game_flow_features.py` and again in `multimodal_classification.py`.
- This leaks no target information: it only fixes integer codes for category names. But an unseen category in a new match would break it.
- Fit on training games, with an "unknown" code.

### Checked and found sound

- **Standardisation:** `StandardScaler` sits inside each scikit-learn `Pipeline`, so it is fit on training data only. `sequence_classifier.py` computes its normalisation statistics from training frames only.
- **Self-supervised pretraining:**
  - DINOv2 and CLIP are used frozen, with weights pretrained by Meta and OpenAI on their own data. Nothing in the repo pretrains or fine-tunes them on this dataset, so test data cannot have reached them.
  - B1 is the only trained component that saw test labels.
  - Any future fine-tuning must use training games only.
- **Augmentation:** none is used, so augmented views cannot cross a split.
- **Overlapping windows:** neighbouring events in the same match share frames. That stays inside one match and cannot cross a match-level split. It does make within-match events look more alike.

---

## Recommended split

**Group the split by match, and use SoccerReplay's official split as the outer split.**

- **Grouping by match** (`match_id`, the dataset folder name) is required.
  - Frames, windows, the stadium, light, kit and camera style are all shared within a match.
  - Any event-level or frame-level random split puts the same match on both sides and lets a model pass by recognising the match. That is exactly the shortcut this project studies.
  - Every match goes wholly to train, validation or test.
- **The official split as the outer split.**
  - Train on official `train`, select models on official `valid`, report once on official `test`.
  - Results are then leak-free by match and comparable with other SoccerReplay-1988 work.
  - For smaller experiments that stay inside official train, use `GroupKFold` or a fixed grouped split by match, with a seed recorded in the manifest.
- **For the invariance claim, add a stricter leave-venue-out evaluation.**
  - Group by home team, so no test stadium appears in training.
  - If accuracy holds up under that split, the model is recognising events rather than places.
- **A time-based split is not needed as the primary protocol.**
  - The task classifies a given event window; it doesn't forecast the future.
  - Time only matters inside the annotation features (B9, B10), which must use past events only.
  - Holding out later dates is a useful secondary check for drift, such as broadcaster graphics changing between seasons. The current test games 25–30 happen to be the latest dates in the 30-game set.
