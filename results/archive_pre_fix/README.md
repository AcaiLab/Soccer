# Results from before the cleanup (archive)

Produced by `dino-contrastive-pipeline` @ `dd3f65f`, before the fixes on
`cleanup/dino-contrastive`. Kept because the project write-up cites them; do
not quote them as current results. See `docs/leak_audit.md`:

- `classification/classification_results.csv` (96.2% contrastive) and
  `contrastive/` (purity 0.973): the contrastive head had seen the test games' labels (B1).
- `contrastive_30games_leakfree/`: pools training and held-out games into one score (B2).
- Everything from the 30-game set: about half the events in games 11-30 are
  frames from the wrong half (B3), and games 01-10 used a different window (B4).
- Classifier "best" rows were chosen on the test set (B5).
- `within_game/`: purity without its ~0.38-0.41 chance level.
