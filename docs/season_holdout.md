# Season hold-out: EPL 2022-23

- Decided: 2026-10-26, following the professor's feedback on the Week 5 report
- Manifest: `configs/games_epl2223_holdout.csv` (115 matches, games 400-514, every one marked `test`)

## Rule

Train and validate on earlier seasons; score on 2022-23 once, at the end of the term, with models whose every choice (features, classifier, epochs, hyperparameters) was fixed before. Until then nothing reads its frames, features or labels: no extraction runs, no clustering, no "quick look".

## Why 2022-23

Premier League matches in the SoccerReplay-1988 share, with the dataset's official split:

| Season | Matches | Official train / val / test | Used in development |
|---|---|---|---|
| 2017-18 to 2019-20 | 255 | 182 / 35 / 38 | yes (Week 6) |
| 2020-21 | 91 | 68 / 13 / 10 | no |
| 2021-22 | 65 | 50 / 7 / 8 | yes (Weeks 2-5) |
| 2022-23 | 115 | 89 / 13 / 13 | no |
| 2023-24 | 26 | 21 / 3 / 2 | no |

2022-23 is the most recent season with enough matches for a test set and has never been used. 2023-24 is more recent but has only 26 matches. 2021-22 cannot be the hold-out: every design decision so far was made on it.

## MatchVision caveat

MatchVision was trained on SoccerReplay-1988's official training matches, so 89 of the 115 hold-out matches may have been in its training data. DINOv2 was not trained on this dataset. MatchVision results on the hold-out are therefore reported twice: on all 115 matches and on the 26 official val/test matches only (`official_split` column), which it did not train on.

## Before the final test

- Install the train-archive annotations for the 89 matches listed as official `train`.
- Extract frames and features for games 400-514 only after the final models are frozen.
