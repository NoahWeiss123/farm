# Pushing `farm_uf850_bottle` to the HuggingFace Hub

The dataset under `datasets_lerobot/farm_uf850_bottle/` is LeRobot-v2.0 shaped
and ready to upload. **200 episodes · 2 tasks · 59,183 frames · 30 fps.**
Total size ~796 MB (7 MB parquet + 788 MB videos).

> This is a **separate** repo from the older `NoahWeiss/farm_uf850` (the
> 66-episode rubik's-cube dataset), which is left untouched. The bottle
> dataset lives at its own repo id, **`NoahWeiss/farm_uf850_bottle`**.

> The CLI is now `hf` (the old `huggingface-cli` is deprecated and no longer
> works in recent `huggingface_hub`). All commands below use `hf`.

## 1. One-time setup

```bash
# In the project venv:
source .venv/bin/activate
pip install --upgrade huggingface_hub hf_transfer

# Log in. Creates a token at ~/.cache/huggingface/token.
# Use a token with "write" scope: https://huggingface.co/settings/tokens
hf auth login
hf auth whoami        # confirm you're logged in
```

The dataset repo id used everywhere downstream (openpi config, sbatch,
eval) is **`NoahWeiss/farm_uf850_bottle`** — keep it unless you change it in
`tools/cluster/patch_openpi_config*.py` and `tools/cluster/train_pi05.sbatch`
as well.

## 2. Create the dataset repo

```bash
hf repo create NoahWeiss/farm_uf850_bottle --repo-type dataset
# this dataset is PUBLIC (open-source); add --private if you ever want it private
```

(Skip if it already exists — the upload step just pushes into it.)

## 3. Upload

```bash
# From the project root. hf_transfer (installed above) parallelizes the
# 788 MB of video; expect a few minutes on a home connection. Resumable —
# re-run the same command if it drops.
HF_HUB_ENABLE_HF_TRANSFER=1 hf upload NoahWeiss/farm_uf850_bottle \
    datasets_lerobot/farm_uf850_bottle \
    . \
    --repo-type dataset \
    --commit-message "bottle pick-and-place: 200 episodes, 2 tasks, 59183 frames, 30 fps"
```

## 4. Verify

```bash
# Sanity-check the meta on the Hub:
hf download NoahWeiss/farm_uf850_bottle meta/info.json \
    --repo-type dataset --local-dir /tmp/farm_check
head -5 /tmp/farm_check/meta/info.json

# Load with LeRobot to confirm everything's wired up:
python - <<'EOF'
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("NoahWeiss/farm_uf850_bottle")   # streams from the Hub
print(f"episodes={ds.num_episodes}  frames={ds.num_frames}  fps={ds.fps}")
print(ds[0].keys())
print(ds[0]["task"])
print(ds[0]["observation.state"].shape)
print(ds[0]["observation.images.base"].shape)
EOF
```

Expected output:
```
episodes=200  frames=59183  fps=30
dict_keys(['observation.state', 'action', 'timestamp', 'frame_index',
           'episode_index', 'task_index', 'index', 'task',
           'observation.images.base', 'observation.images.wrist'])
'Picking up the bottle and placing it on the box'
torch.Size([7])
torch.Size([3, 480, 640])
```

## 5. Train

The repo id is already baked into the cluster config
(`tools/cluster/patch_openpi_config_pi05.py` → `repo_id="NoahWeiss/farm_uf850_bottle"`),
so once the dataset is on the Hub there's nothing else to wire up. Follow
`tools/cluster/README.md`:

```bash
# on the login pod, after staging the launcher files + running setup.sh:
sbatch ~/farm-train/train_pi05.sbatch     # π0.5 FULL fine-tune, 4× H100
```

The GPU node only **pulls** the dataset and runs `compute_norm_stats.py`
(~1 min) + training — all the heavy data prep (LeRobot formatting, video
encoding) is already done locally and shipped in this upload.

## Notes

* **Public dataset**: this repo is public (open-source), so it doesn't count
  against your private-storage quota and the training job can pull it without a
  token. The sbatch still sets `HF_TOKEN`, but that's for *pushing checkpoints*
  to the model repo (writing needs auth even for a public repo) — not for
  reading this dataset.
* **Updating after re-collection**: re-run
  `python tools/export_lerobot.py --src Dataset3 --out datasets_lerobot/farm_uf850_bottle --fps 30 --force`,
  then re-run the `hf upload` above (it overwrites).
