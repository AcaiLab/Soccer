# Running the pipeline on the GPU server

Nothing heavy runs on a personal machine; it only opens an SSH session. There
are two ways to feed the server:

**A. Direct (no AWS needed, the default).** The server downloads one game from
Dropbox, extracts its frames, deletes the video, and moves on; then it encodes
and analyses everything.

```
Dropbox ──> GPU server: download game → frames → delete video → DINOv2 → analysis
```

**B. Staged in S3 (needs AWS access).** Videos are first streamed into a
bucket, then the server reads them from there. This is useful if several
machines need the videos, or to keep a durable copy. Sections 2 and 4 below
apply only to B.

| Step | Runs on | Load |
|---|---|---|
| Video download (A: Dropbox → server, B: Dropbox → S3 → server) | GPU server | about 1.2 GB per game, one game on disk at a time |
| Annotations (public Google Drive copy) | GPU server | about 20 MB |
| Frame extraction | GPU server CPU | |
| DINOv2 features, contrastive, sequence models | GPU server GPU | about 1 min per game for features |
| Clustering, classifiers | GPU server CPU | light |

Measured on game 01: video 1.2 GB, extracted frames 0.34 GB, features 11 MB.

## 1. Your SSH key (on your PC)

The private key is a password to the server. **Never paste it into a chat, an
email or a file in this repository**, and don't share it.

1. Put it at `C:\Users\<you>\.ssh\soccer_gpu`, with no file extension. The public key
   the server holds stays as it is.
2. Restrict it to your account, or OpenSSH will refuse to use it. In PowerShell:
   ```powershell
   icacls $env:USERPROFILE\.ssh\soccer_gpu /inheritance:r /grant:r "$($env:USERNAME):(R)"
   ```
3. Add a host entry to `C:\Users\<you>\.ssh\config`:
   ```
   Host soccer-gpu
       HostName <server address from your professor>
       User <your username on the server>
       Port 22
       IdentityFile ~/.ssh/soccer_gpu
       IdentitiesOnly yes
       ServerAliveInterval 60
   ```
4. Test it: `ssh soccer-gpu nvidia-smi`.

## 2. AWS, once (option B only)

From a machine with AWS admin credentials (after approval):

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars      # set region, profile, bucket name
terraform init && terraform plan                  # review, then: terraform apply
```

Then create the server's access key and the Dropbox settings by hand. Terraform
never stores secrets.

```bash
aws iam create-access-key --user-name soccer-vision-gpu-server     # shown once
aws ssm put-parameter --type SecureString --name /soccer/dropbox/shared_url    --value '<link>'
aws ssm put-parameter --type SecureString --name /soccer/dropbox/link_password --value '<password>'
aws ssm put-parameter --type SecureString --name /soccer/dropbox/token         --value '<token>' --overwrite
```

On the server, store the access key as profile `soccer`, readable only by you:

```bash
umask 077; mkdir -p ~/.aws
cat >> ~/.aws/credentials << 'EOF'
[soccer]
aws_access_key_id = ...
aws_secret_access_key = ...
EOF
```

The key can only touch this bucket's folders, the data key and the Dropbox
parameters. Deactivate it when the project ends:
`aws iam update-access-key --status Inactive`.

## 3. Code and environment on the server

```bash
ssh soccer-gpu
git clone -b cleanup/dino-contrastive https://github.com/AcaiLab/Soccer.git soccer   # if the branch is on GitHub
cd soccer
bash infra/gpu-server-setup.sh            # cu118 as argument for older drivers
nano ~/.config/soccer/env                 # fill in the REPLACE values
```

If the branch isn't on GitHub yet, copy it as a bundle instead, from your PC:
`git bundle create soccer.bundle cleanup/dino-contrastive`, then
`scp soccer.bundle soccer-gpu:`, then on the server
`git clone -b cleanup/dino-contrastive soccer.bundle soccer`.

### Dropbox credentials that do not expire

A Dropbox access token lasts four hours, which a long run cannot live with. Make a
refresh token once on your PC (README, Credentials), then copy the one file it
creates and tell the settings file where it is:

```bash
scp tmp_dbx/oauth.json soccer-gpu:.config/soccer/dropbox_oauth.json
ssh soccer-gpu 'chmod 600 ~/.config/soccer/dropbox_oauth.json'
ssh soccer-gpu 'echo "export DROPBOX_OAUTH_FILE=\$HOME/.config/soccer/dropbox_oauth.json" >> ~/.config/soccer/env'
ssh soccer-gpu 'cd soccer && source .venv/bin/activate && source ~/.config/soccer/env && python scripts/dropbox_auth.py check'
```

From then on every script fetches its own access tokens, and `DROPBOX_TOKEN_FILE`
is ignored. With option B the same two values can live in SSM instead:
`/soccer/dropbox/app_key` and `/soccer/dropbox/refresh_token`.

## 4. Copy the games from Dropbox to S3 (option B only)

Run long jobs inside `tmux`, so they survive a dropped SSH connection
(`tmux attach -t soccer` to come back).

```bash
tmux new -s soccer
cd ~/soccer && source .venv/bin/activate && source ~/.config/soccer/env
python scripts/transfer_dropbox_to_s3.py --workers 2 2>&1 | tee -a ~/transfer.log
```

The games are those in `configs/games.csv`. Rerunning skips files already in S3.
With a refresh token (above) the script renews its access token itself. A fixed
token lasts about 4 hours; when it expires the script stops with exit code 2:
put a new token in SSM (`--overwrite`) and rerun.

## 5. Annotations

```bash
python scripts/fetch_annotations.py --splits train valid test           # add --upload for option B
```

## 6. Run the pipeline

```bash
tmux new -s soccer          # or: tmux attach -t soccer
cd ~/soccer && source .venv/bin/activate && source ~/.config/soccer/env
python run_dino_pipeline.py --device cuda 2>&1 | tee -a ~/pipeline.log              # A: direct from Dropbox
python run_dino_pipeline.py --from-s3 --upload --device cuda 2>&1 | tee -a ~/pipeline.log   # B: via S3
```

This extracts frames (each game's video is downloaded, used and deleted),
encodes them on the GPU, clusters, trains the contrastive head on training
games and evaluates on held-out ones, and runs the classifiers. With B it
then copies `features/` and `results/` to S3. With A, fetch the (small)
results to your PC with `scp -r soccer-gpu:soccer-data/results .`. If a fixed
Dropbox token expires mid-run, put a new one in the token file and rerun
(a refresh token never needs this): finished games are skipped. Use `--stages` to run part of it, for
example `--stages cluster classify` after changing only the analysis.

Check on it from your PC without staying logged in:

```bash
ssh soccer-gpu "tail -n 30 ~/pipeline.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv"
```

## Shared-server etiquette and safety

- Use only the GPU you were given (`CUDA_VISIBLE_DEVICES` in the env file), and
  check `nvidia-smi` before starting.
- Data goes under `SOCCER_DATA_ROOT`. Videos are deleted after extraction;
  remove `frames` with `rm -r $SOCCER_DATA_ROOT/results_by_label_epl_2021_2022`
  once features are safely in S3.
- Credentials on the server (`~/.config/soccer/env`, the Dropbox token or OAuth
  file, and `~/.aws/credentials` for option B) are mode 600. Server administrators
  can still read them, so use narrowly scoped keys. The Dropbox refresh token works
  until revoked: give the app read-only scopes, and revoke it under Dropbox →
  Settings → Connected apps when the project ends.
- Nothing secret goes into the repository. CI scans every push with gitleaks.
