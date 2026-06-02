# FFT checkpoint generalization — which step is best?

**Question (user):** after the 56k full fine-tune, find the checkpoint with the best
generalization — test each against episodes and see which performs best, in case the
final step over-memorised.

**Method.** Two sweeps over all 7 saved checkpoints (8k,16k,24k,32k,40k,48k,55999),
each with a fixed seed so every checkpoint sees identical episodes/frames:
1. **Training-data fit** (`eval_fft_ckptgen`, job 470): open-loop action accuracy on
   48 multi-object episodes the FFT trained on. Measures FIT (rewards memorization).
2. **Held-out generalization** (`eval_fft_bench fftonly`, job 471): clean + 9
   domain-shift conditions on `eval_bench` — 15 separately-recorded episodes across 5
   tasks, including **bottle→desk which is NOT in the multi-object training set** (a
   genuinely out-of-distribution task), so this set is held out for the FFT.

Why both: the FFT trained on ALL 424 multi-object episodes, so training-data accuracy
favours the final step by construction; only held-out data reveals over-memorisation.

## Result (joint MAE @ horizon end, degrees)

| step | train° (fit) | train@5° | **held-out° (generalization)** | held-out robust° |
|---:|---:|---:|---:|---:|
| 8000  | 2.540 | 94.0% | 2.344 | 3.028 |
| 16000 | 1.925 | 97.7% | 2.136 | 2.615 |
| 24000 | 1.550 | 99.7% | 1.954 | 2.388 |
| 32000 | 1.392 | 99.7% | 1.802 | 2.245 |
| 40000 | 1.191 | 100%  | 1.741 | 2.162 |
| 48000 | 1.061 | 100%  | 1.727 | 2.153 |
| **55999** | **0.983** | 100% | **1.684** | **2.119** |

(See `clean_7_checkpoint.png` + `fft_sweep/fft_base_selection.json`.)

## Verdict: **step 55999 (the final) is best — on every metric.**

Both fit AND held-out generalization (clean and under domain shift) improve
**monotonically** to the end. The over-memorisation we guarded against did **not**
occur: the heavy domain-randomization augmentation + 4-task diversity + prompt-aug
regularised the full fine-tune enough that it kept generalising better all the way to
56k. The train↔held-out gap stays modest (0.98° vs 1.68°, ~1.7×), i.e. a healthy fit,
not overfitting.

Honest caveats: held-out is a 15-episode bench (one OOD task); the curve is flattening
near the end (40k→56k held-out improves only 1.74→1.68°), so an earlier checkpoint
(e.g. 40k) is nearly as good for ~30% less training — but 55999 is strictly best.

**Consequence:** the LoRA base (`lora_base`) is set to **55999**, and 55999 is the
checkpoint reported as "the FFT model" in the benchmark. The concern was worth
testing; the data cleared it.
