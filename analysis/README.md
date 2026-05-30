# Analysis — GSE vs full fine-tune

Offline **action-accuracy** + **inference-latency** evaluation of the two π0.5
FARM fine-tunes, rendered with ROOT. Open-loop, no robot: each model replays the
recorded observations from training episodes, and its predicted action chunk is
compared to the recorded ground truth (the recorder stores absolute joint
positions, so `action[t] == state[t+1]`).

## Models
| model | config | checkpoint |
|---|---|---|
| GSE | `pi05_farm_uf850_gse` | step-2999 — `NoahWeiss/farm_uf850_pi05_gse` |
| full FT | `pi05_farm_uf850` | step-19999 — `NoahWeiss/farm_uf850_pi05` |

6 training episodes (3 per bottle task), 16 frames each → 96 inferences/model.

## Results (training-episode fit)
| metric | GSE | full FT |
|---|---|---|
| overall joint MAE | 1.02° | **0.72°** |
| 10-step chunk MAE | 1.04° | **0.71°** |
| gripper MAE | 0.0033 | 0.0026 |
| latency (median / p90) | 51 / 51 ms | 48 / 48 ms |
| throughput | ~19.6 infers/s | ~20.9 infers/s |

Both fit the training data well (sub-1.1° joint error) at essentially the same
latency (~50 ms, ~20 chunk-inferences/s). The full FT fits tighter — but read
the caveats.

## Caveats
1. **Not effort-matched.** GSE trained 3,000 steps (≈6.5 epochs); the full FT
   trained 20,000 steps (≈21.6 epochs) — roughly 3× more data exposure. Part of
   the full FT's tighter fit is just more training, not the method itself.
2. **Open-loop on TRAINING data** — this measures fit / memorization, not
   generalization. A tighter training fit can mean *more* memorization, which is
   consistent with the full FT's known out-of-distribution failures.

## Figures
| file | shows |
|---|---|
| `cmp_per_joint_mae.png` | per-joint MAE (deg), GSE vs full |
| `cmp_headline_mae.png` | next-step vs 10-step-chunk joint MAE |
| `cmp_gripper_mae.png` | gripper error |
| `cmp_latency.png` | inference latency (median, p90) |
| `cmp_error_dist.png` | per-frame joint-error distribution |
| `cmp_per_episode.png` | per-episode joint MAE (consistency across tasks) |
| `gse_vs_full.pdf` | all six figures, multi-page |

`eval-gse.json` / `eval-full.json` hold the raw per-frame results.

## Regenerate
- **Eval** (cluster, 1 GPU): `model/cluster/eval_pi05.sbatch` drives
  `model/cluster/eval_offline.py` — runs both models on staged episodes.
- **Plots** (ROOT env): `python model/plot_eval_comparison.py
  analysis/eval-gse.json analysis/eval-full.json --outdir analysis`
