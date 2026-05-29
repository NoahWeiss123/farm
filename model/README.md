# tools/

The FARM model workstream: turn teleop recordings into a trained π0.5 policy and
run it on the arm. (The teleop **daemon** + dashboard live in `teleop/edge-agent/`;
the Quest client in `teleop/quest/`.)

## Pipeline

```
farm serve  ──teleop──▶  datasets/dataset3/episode_*        (recorded on the real UF850)
     │                         │
     │              export_lerobot.py
     ▼                         ▼
  review.html         datasets/lerobot/  ──push──▶  HF: NoahWeiss/farm_uf850_bottle
                                                          │
                                              cluster/ (train on H100s)
                                                          ▼
                                          HF: NoahWeiss/farm_uf850_pi05[_lora|_gse]
                                                          │
                                       serve_pi05.sbatch (WebSocket policy server)
                                                          ▼
                                   eval_pi05.py  ──actions──▶  farm serve  ──▶  arm
```

## Layout

| Path | What |
|---|---|
| `export_lerobot.py` | `datasets/dataset3/` raw episodes → LeRobot v2.0 dataset |
| `analyze_dataset.py` | audit a LeRobot dataset (action alignment, smoothness, gripper, tasks) — no GPU |
| `eval_pi05.py` | live eval client: reads `farm serve` obs, queries the policy server, drives the arm |
| `eval_pi05_episode_check.py` | open-loop check of one recorded episode through the served policy |
| `offline_eval.py` | open-loop pred-vs-recorded accuracy on the login pod (checkpoint selection) |
| `FINDINGS.md` | **diagnosis** of the deployed policy + ranked fixes + the 3-architecture plan |
| `HUGGINGFACE_UPLOAD.md` | how to push the dataset to the Hub |
| `cluster/` | everything that runs on the H100 cluster — see `cluster/README.md` |
| `rtc/` | Real-Time Chunking dev probes (smoothness across action-chunk seams) |

## The three fine-tuning architectures

Registered on the cluster by `cluster/setup.sh`, all directly comparable:

- **`pi05_farm_uf850`** — full fine-tune (8 GPUs). Max capacity; overfits the
  2-task data and erodes the base. This is the deployed ~30% model.
- **`pi05_farm_uf850_lora`** — LoRA (1 GPU). Freezes the backbone; preserves the
  base but can under-adapt.
- **`pi05_farm_uf850_gse`** — VLA-GSE (1 GPU). SVD-splits each weight into a
  preserved "generalized" expert + adapted "specialized" experts. The principled
  middle ground. See `cluster/openpi_gse.py`.

Start with `FINDINGS.md` for why and `cluster/README.md` for how.
