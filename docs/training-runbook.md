# Training runbook

What to do when the π0.5 fine-tune does not clear the 50% sim-success
threshold by end of training. This is the rollback-criteria decision tree
from DESIGN.md → Fine-Tuning Plan, expanded into actual diagnostic steps.

The 50% threshold is the checkpoint-acceptance bar, not a target. The
deployment target is ≥80% on the colored-block stacking eval (DESIGN.md →
Evaluation). A checkpoint below 50% on the held-out sim eval is treated
as a rejected fine-tune; do not promote it.

## When to consult this page

- `farm train eval` reports < 50% success across the held-out sim suite
  at end of training, or no intermediate checkpoint cleared the bar.
- A checkpoint that cleared the bar on the sim eval underperforms on the
  real arm by more than 20 percentage points.
- Sim success rate plateaus for ≥ 1500 steps with no recovery.

## Decision tree

Walk these causes in priority order. Most fine-tune misses are caused by
the first two; don't skip to the bottom.

### 1. Dataset too small

**Symptom.** Sim success plateaus early (within the first 1500 steps),
training loss still decreasing but slowly, eval curve flat or noisy.

**Check.**

- Total episode count per task. Phase-MVP target: 200–400 episodes per
  task. Fewer than 150 will not get you to 50%.
- Episode length distribution. Episodes shorter than 3 s usually mean
  the operator clipped them; episodes longer than 20 s usually mean a
  pause that should have been re-recorded.

**Fix.**

- Record another 100 episodes per failing task. Mix in lighting and
  background variation deliberately.
- Re-split: hold out by *collection session*, not by episode (see
  cause 4).

### 2. Dataset too noisy (bad demos)

**Symptom.** Loss curve looks fine, success plateaus around 30–45%.
Watching trained-policy rollouts shows the arm "almost" succeeding —
right gestures, wrong pose at the last moment.

**Check.**

- Rewatch a random 5% of your demos at 2× speed. Count "would I accept
  this on a real arm" pass/fail. If failure rate is > 10%, the dataset
  is contaminated.
- Look for systematic operator drift: a leader-arm teleop session where
  the wrist orientation slowly drifts off-axis can teach a wrong policy
  consistently.

**Fix.**

- Re-label demos with operator-pass/fail in the LeRobot metadata. Drop
  failed episodes from the training split; keep them in a "bad demos"
  shard for ablation.
- Re-record the worst session entirely.

### 3. Action-space mismatch

**Symptom.** Sim success is near-zero. Rollouts look completely
untrained — random motions, no convergence toward the target object.

**Check.**

- Episode metadata `action_space`. The Phase-MVP normative value is
  `ee_pose_delta_base_frame` (DESIGN.md → Open Questions #6). If your
  episodes encode joint-space deltas or absolute TCP poses, the model is
  training against the wrong action.
- The capability card the inference container loads
  ([capability-cards.md](capability-cards.md)) — its `action_space` must
  match the episodes' action space exactly.

**Fix.**

- Re-encode the dataset to ee-pose deltas in base frame. Do not retrain
  before this is consistent across every shard.

### 4. Calibration drift between collection sessions

**Symptom.** Sim eval looks acceptable (cleared 50%); real-arm eval is
much worse than the sim number. Or: the train and val curves diverge
sharply and never reconverge.

**Check.**

- Inspect the per-episode `calibration_hash` (see
  [safety.md](safety.md#calibration-drift-detection)). If two collection
  sessions have different hashes, you trained on a mixed extrinsic.
- Look at the dashboard's run-history view filtered by your training
  dataset; the calibration-hash grouping bands tell you exactly when the
  baseline shifted.

**Fix.**

- Re-collect any session whose hash differs from the canonical one. Or
  retrain on the subset that shares one hash, and re-evaluate.
- Lock the camera mount before the next collection run; document the
  mount procedure so a future session matches.

### 5. Train/val split leaks

**Symptom.** Sim val success rate is high; held-out test rate is much
lower. The eval-cadence rollouts look great in training and bad on the
held-out scenes.

**Check.**

- The split is held out by *collection session*, not by episode. Splitting
  by episode means a single session's lighting and fixturing correlations
  contaminate both train and val.

**Fix.**

- Re-do the split at the session level. Re-train. The metric you select
  on must be the held-out-by-session eval.

### 6. Model architecture mismatch with the 850's joint count

**Symptom.** Sim success is exactly 0% or wildly inconsistent across
seeds. Inference logs show shape mismatch warnings.

**Check.**

- The π0.5 controller expects a 6-DOF + gripper action vector for the
  850. If the checkpoint was inadvertently exported with a different
  arm's head (e.g. 7-DOF Franka), the dimensions don't line up.
- The capability card's `embodiment.dof` must equal 6.

**Fix.**

- Re-export the checkpoint with the correct head. If the head was
  swapped accidentally during a sweep, the wandb config and the model
  card usually have the canonical value.

## After diagnosing

1. Write the cause and the fix into the model card on Hugging Face. Two
   lines is enough. Future-you debugging the next miss will thank you.
2. Re-run the eval cadence; do not promote a checkpoint that has not
   cleared the 50% bar on the held-out sim suite.
3. If two consecutive causes fixed do not lift sim success above 50%,
   stop training and ship Phase-MVP with the classical-planner fallback
   path as primary. See DESIGN.md → Recommended Approach for why this is
   not a failure.

## Adjacent docs

- DESIGN.md → Fine-Tuning Plan (dataset target, hyperparameters, eval
  cadence, checkpoint selection)
- [safety.md](safety.md#calibration-drift-detection) (calibration hash
  semantics)
- [capability-cards.md](capability-cards.md) (action space, embodiment
  fields)
- [faq.md](faq.md#my-fine-tune-sim-eval-is-below-50) (entry point from
  the FAQ)
