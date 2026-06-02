# FFT-multiobject vs other fine-tuning techniques (Phase C)

The π0.5 full fine-tune trained here (`farm_uf850_multiobject_fft_robust`, step 55999)
benchmarked against the other architectures on the held-out **eval_bench** set (15
separately-recorded episodes / 5 tasks, incl. an OOD `bottle→desk` task), offline
open-loop action accuracy — clean frames + 9 synthetic domain-shift conditions.

The FFT-multiobject and GSE-multiobject are the **controlled comparison**: same 424-ep
/ 4-task data, same heavy domain-randomization + prompt-aug toolkit — they differ *only*
in fine-tune method (full FT of all 3.3B params vs SVD-spectral GSE adapters on a frozen
backbone). LoRA (100-ep, 1 task) and the 2-task FFT (200-ep bottle) are single-/few-task
references.

## Results — joint MAE @ horizon end (deg; lower is better)

| Model | data | clean° | robust (mean)° | domain-shift° | bottle | bear | duck | hat |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **FFT-multiobj (full FT)** ★ | 424 ep / 4 task | **1.68** | **2.12** | **1.91** | 2.3 | 0.8 | 0.8 | 0.9 |
| GSE-multiobj (robust) | 424 ep / 4 task | 1.94 | 2.50 | 2.23 | 2.7 | 1.3 | 1.0 | 1.1 |
| LoRA | 100 ep / 1 task | 11.40 | 11.66 | 11.97 | 4.9 | 21.5 | 10.6 | 14.0 |
| 2-task FFT | 200 ep / 2 task | 5.57 | 7.03 | 8.45 | **0.7** | 11.9 | 3.8 | 10.8 |

## Findings

1. **The full fine-tune is the best all-rounder — and it beats GSE on robustness too.**
   FFT-multiobject is lowest on clean (1.68° vs 1.94°), mean-perturbed (2.12° vs 2.50°),
   AND the realistic room-change `domain_combo` (1.91° vs 2.23°). This is the notable
   result: on the *2-task bottle* data, earlier work found GSE degraded least under
   shift; on the **4-task multi-object data with heavy augmentation, the full FT wins on
   both fit and robustness**. With enough task diversity + domain randomization, full FT's
   extra capacity pays off without the over-memorization that sank the 2-task run (§3).

2. **Capability scales with data diversity (the dominant axis).** Both multi-object models
   handle all four objects (~0.8–2.7°); the single-/few-task baselines collapse on objects
   they never saw — LoRA is 21.5° on bear, the 2-task FFT 11.9° on bear / 10.8° on hat.
   No architecture recovers tasks absent from its data.

3. **A specialization↔generality trade-off is visible on bottle.** The 2-task FFT, a
   dedicated bottle specialist (200 bottle eps, 2 tasks), fits bottle tightest (0.7°) — but
   the multi-object FFT, despite seeing *more* bottle data (299 eps), is 2.3° on bottle
   because its capacity + heavy aug are spread across 4 tasks and invariance. It trades a
   little peak bottle accuracy for breadth + robustness — the right trade for a deployable
   multi-task policy.

4. **LoRA under-adapts even on its own task** (bottle 4.9° vs the FFTs' ≤2.3°): low-rank
   adapters on a frozen base don't move enough for this embodiment, consistent with the
   broader finding that LoRA is the weakest fit here.

## Takeaway

For a deployable, multi-task, scene-tolerant UF850 policy, **the heavy-aug full
fine-tune (FFT-multiobject, step 55999) is the pick** — best clean accuracy, best
robustness, all four tasks. GSE remains a close, cheaper-to-train second. The 2-task FFT
stays a strong single-environment bottle specialist. (Offline action error is a proxy;
confirm on-arm. eval_bench is 15 episodes incl. one OOD task.)

Data: `phaseC_compare.json`; baselines from the earlier `analysis/benchmark/` run (same
eval_bench), FFT@55999 from the checkpoint sweep (job 471).
