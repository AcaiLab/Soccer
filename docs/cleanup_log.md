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
