#!/usr/bin/env python3
"""Real-episode action-prediction eval for the multiobject GSE policy.

NO robot, NO sim, NO fabricated data. We replay REAL demonstrated episodes
through the policy exactly the way openpi trained on them and dump, for every
queried frame, the model's predicted joint-angle chunk next to the real
recorded joint angles. Two complementary passes:

  (1) RANDOM-FRAME accuracy  — random frames from the MIDDLE of many episodes,
      one inference each. Headline number: how well does the predicted chunk
      end ``pred[H-1]`` match the demonstrated joints ``state[t+H]`` ~1/3 s
      ahead? Plus the FULL chunk pred[0..H-1] vs gt[0..H-1] is saved so error
      can be resolved per-horizon-step and per-joint downstream.

  (2) DENSE ROLLOUT  — a handful of WHOLE episodes (spanning every task) swept
      frame-by-frame: at each (strided) frame we feed the real observation and
      record the predicted chunk. Overlaid on the real joint trajectory this
      shows, directly, whether the model's commanded next-angles track the
      demonstrated movement along the whole episode.

Step clock (model/FINDINGS.md + the LeRobot export):
  * dataset exported at ``fps`` with ``action[t] = state[t+1]`` (absolute);
  * openpi loads the chunk as consecutive frames
    ``delta_timestamps = [k/fps for k in range(H)]`` (data_loader.py);
  so chunk step ``k`` <-> frame ``t+1+k`` and the horizon end
  ``pred[H-1]`` <-> ``state[t+H]``  (H=10 @ 30 fps -> 333 ms lookahead).

We read the LeRobot training set the SAME way openpi training does
(``LeRobotDataset`` + ``delta_timestamps`` over ``action``), so the saved
ground-truth chunk IS the model's training target -- no manual indexing, no
off-by-one. The returned ``actions`` are absolute joint targets in the same
space as ``observation.state``, so pred-vs-state is apples-to-apples.

This is an OPEN-LOOP / teacher-forced metric: the model always sees the REAL
observation, so it measures single-shot action-prediction fidelity (the
standard offline VLA metric), not closed-loop drift (that needs a simulator
that can render the real scene, which we don't have offline).

Outputs (both into the run dir):
  * ``--out`` JSON         — summary stats + per-horizon-step curve + headline.
  * ``--raw-out`` NPZ      — every predicted/real chunk + the dense rollouts,
                             so all figures + derived metrics are computed
                             locally with no GPU.

Run in the openpi container (needs JAX + GPU + lerobot + ffmpeg):

  uv run python eval_train_endhorizon.py \
      --config pi05_farm_multiobject_gse \
      --checkpoint-dir .../farm_gse_multiobject_robust_190/5999 \
      --repo-id NoahWeiss/farm_uf850_multiobject \
      --n-episodes 64 --samples-per-episode 12 --horizon 10 \
      --roll-per-task 2 --roll-stride 2 \
      --out eval-train-endhorizon.json --raw-out eval-train-endhorizon-raw.npz
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

JN = ["j1", "j2", "j3", "j4", "j5", "j6"]
DEG = 180.0 / np.pi


def to_hwc_uint8(x) -> np.ndarray:
    """LeRobot video frame (torch CHW float[0,1] or uint8, or numpy) -> HWC uint8."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = np.transpose(x, (1, 2, 0))                       # CHW -> HWC
    if x.dtype != np.uint8:
        scale = 255.0 if float(np.nanmax(x)) <= 1.0 + 1e-6 else 1.0
        x = np.clip(x.astype(np.float32) * scale + 0.5, 0, 255).astype(np.uint8)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    return np.ascontiguousarray(x)


def episode_ranges(ds, meta):
    """Per-episode [from, to) global frame indices, robust across lerobot versions."""
    try:
        edi = ds.episode_data_index
        return [int(v) for v in edi["from"]], [int(v) for v in edi["to"]]
    except Exception:
        ep_from, ep_to, c = [], [], 0
        for i in range(meta.total_episodes):
            L = int(meta.episodes[i]["length"])
            ep_from.append(c); ep_to.append(c + L); c += L
        return ep_from, ep_to


def episode_tasks(ds, repo_id, n_eps):
    """episode_index -> task string, read straight from meta/episodes.jsonl."""
    try:
        root = str(ds.root)
    except Exception:
        root = os.path.expanduser(f"~/.cache/huggingface/lerobot/{repo_id}")
    out = {}
    path = os.path.join(root, "meta", "episodes.jsonl")
    try:
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                out[int(r["episode_index"])] = (r.get("tasks") or [""])[0]
    except Exception:
        pass
    return out


def episode_state_traj(ds, a, b):
    """Real per-frame state[a:b] (n,7) WITHOUT video decode (parquet column)."""
    try:
        col = ds.hf_dataset.select(range(a, b))["observation.state"]
        return np.asarray(col, dtype=np.float32).reshape(b - a, -1)
    except Exception:
        # fallback: decode each frame (slow, but only for the few rollout eps)
        return np.stack([np.asarray(ds[i]["observation.state"], dtype=np.float32).reshape(-1)
                         for i in range(a, b)])


def infer_chunk(policy, ds, idx, task, image_tools, H):
    """One inference at global frame idx -> (pred[:H] (H,7), state (7,))."""
    item = ds[idx]
    if not task:
        t_ = item.get("task")
        task = t_[0] if isinstance(t_, (list, tuple)) else (t_ if isinstance(t_, str) else "")
    base = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(to_hwc_uint8(item["observation.images.base"]), 224, 224))
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(to_hwc_uint8(item["observation.images.wrist"]), 224, 224))
    state = np.asarray(item["observation.state"], dtype=np.float32).reshape(-1)
    obs = {"observation/image": base, "observation/wrist_image": wrist,
           "observation/state": state, "prompt": task}
    pred = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)  # (Hc,7) absolute joints
    return pred[:H], state, task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="registered openpi TrainConfig name")
    ap.add_argument("--checkpoint-dir", required=True, help="dir containing params/ + assets/")
    ap.add_argument("--repo-id", required=True, help="LeRobot TRAINING dataset (the one the model fine-tuned on)")
    ap.add_argument("--n-episodes", type=int, default=64)
    ap.add_argument("--samples-per-episode", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--mid-lo", type=float, default=0.25, help="middle window start (fraction of episode)")
    ap.add_argument("--mid-hi", type=float, default=0.75, help="middle window end (fraction of episode)")
    ap.add_argument("--ep-range", default="", help="restrict to episode indices lo:hi (half-open, "
                    "in original dataset indexing); default '' = all. e.g. 100:299 = held-out bottle.")
    ap.add_argument("--tol-deg", default="2,5,10", help="accuracy thresholds (mean per-joint deg) at horizon end")
    ap.add_argument("--roll-per-task", type=int, default=2, help="whole episodes to densely roll out per task")
    ap.add_argument("--roll-stride", type=int, default=2, help="inference every Nth frame in a rollout")
    ap.add_argument("--roll-max-frames", type=int, default=700, help="cap rollout episode length (frames)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="multiobject")
    ap.add_argument("--split", default="train",
                    help="provenance label written to the JSON: held_out | train_fit | train")
    ap.add_argument("--out", default="")
    ap.add_argument("--raw-out", default="")
    args = ap.parse_args()

    H = args.horizon
    # Optional episode-index window (original dataset indexing), e.g. held-out bottle 100:299.
    _ep_lo, _ep_hi = 0, 10**9
    if args.ep_range:
        _lo, _hi = args.ep_range.split(":", 1)
        _ep_lo, _ep_hi = int(_lo), int(_hi)
    def _in_range(e):
        return _ep_lo <= e < _ep_hi
    from openpi.policies import policy_config
    from openpi.training import config as _config
    from openpi_client import image_tools
    from lerobot.common.datasets import lerobot_dataset

    cfg = _config.get_config(args.config)
    print(f">>> loading policy: config={args.config}\n    ckpt={args.checkpoint_dir}", flush=True)
    t0 = time.time()
    policy = policy_config.create_trained_policy(cfg, args.checkpoint_dir)
    print(f"    policy ready in {time.time() - t0:.0f}s", flush=True)

    meta = lerobot_dataset.LeRobotDatasetMetadata(args.repo_id)
    fps = float(meta.fps)
    ds = lerobot_dataset.LeRobotDataset(
        args.repo_id,
        delta_timestamps={"action": [k / fps for k in range(H)]},   # = openpi's training loader
    )
    ep_from, ep_to = episode_ranges(ds, meta)
    n_eps_total = len(ep_from)
    ep_task = episode_tasks(ds, args.repo_id, n_eps_total)
    print(f">>> dataset {args.repo_id}: {n_eps_total} eps · fps={fps:.0f} · "
          f"{len(ds)} frames · horizon={H} ({H / fps * 1000:.0f} ms lookahead)", flush=True)
    print(f">>> [pass 1] {args.samples_per_episode} RANDOM mid-frames "
          f"[{args.mid_lo:.0%}-{args.mid_hi:.0%}] from {args.n_episodes} random episodes (seed={args.seed})", flush=True)

    rng = np.random.default_rng(args.seed)
    elig = [e for e in range(n_eps_total) if (ep_to[e] - ep_from[e]) > (H + 8) and _in_range(e)]
    if args.ep_range:
        print(f">>> episode window {args.ep_range}: {len(elig)} eligible episodes in [{_ep_lo},{_ep_hi})", flush=True)
    rng.shuffle(elig)
    chosen = sorted(elig[: args.n_episodes])

    # raw per-sample storage (everything downstream needs)
    S_ep, S_loc, S_pred, S_gt, S_state, S_task = [], [], [], [], [], []
    per_task: dict[str, list[float]] = {}
    tinf: list[float] = []

    for e in chosen:
        a, b = ep_from[e], ep_to[e]
        n = b - a
        lo = max(0, int(args.mid_lo * n))
        hi = min(n - H - 1, int(args.mid_hi * n))                    # ensure state[t+H] exists (real, not pad)
        if hi <= lo:
            continue
        k = min(args.samples_per_episode, hi - lo + 1)
        locs = sorted(int(x) for x in rng.choice(np.arange(lo, hi + 1), size=k, replace=False))
        task = ep_task.get(e, "")
        for loc in locs:
            item = ds[a + loc]
            t_now = task
            if not t_now:
                t_ = item.get("task")
                t_now = t_[0] if isinstance(t_, (list, tuple)) else (t_ if isinstance(t_, str) else "")
            base = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(to_hwc_uint8(item["observation.images.base"]), 224, 224))
            wrist = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(to_hwc_uint8(item["observation.images.wrist"]), 224, 224))
            state = np.asarray(item["observation.state"], dtype=np.float32).reshape(-1)
            gt = np.asarray(item["action"], dtype=np.float32).reshape(H, -1)   # [state[t+1] … state[t+H]]
            obs = {"observation/image": base, "observation/wrist_image": wrist,
                   "observation/state": state, "prompt": t_now}
            tic = time.perf_counter()
            pred = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)[:H]  # (H,7) absolute joints
            tinf.append(time.perf_counter() - tic)

            S_ep.append(e); S_loc.append(loc); S_task.append(t_now)
            S_pred.append(pred); S_gt.append(gt); S_state.append(state)
            per_task.setdefault(t_now, []).append(float(np.abs(pred[H - 1, :6] - gt[H - 1, :6]).mean()))
        print(f"  ep{e:03d} ({task[:34]!r}): {k} mid-frames in [{lo}-{hi}] of {n}", flush=True)

    if not S_pred:
        print("!!! no samples evaluated (episodes too short?)")
        return

    S_pred = np.stack(S_pred)              # (N,H,7)
    S_gt = np.stack(S_gt)                  # (N,H,7)
    S_state = np.stack(S_state)            # (N,7)
    S_ep = np.array(S_ep); S_loc = np.array(S_loc)
    N = len(S_pred)

    # ----- summary metrics (full picture; figures recompute from raw NPZ) -----
    abserr = np.abs(S_pred[:, :, :6] - S_gt[:, :, :6])                 # (N,H,6) rad
    grip_abserr = np.abs(S_pred[:, :, 6] - S_gt[:, :, 6])              # (N,H)
    end_err = abserr[:, H - 1, :].mean(axis=1)                        # (N,) mean-joint err @ horizon end
    first_err = abserr[:, 0, :].mean(axis=1)
    chunk_err = abserr.mean(axis=(1, 2))
    step_curve_deg = np.degrees(abserr.mean(axis=(0, 2)))             # (H,) mean over samples+joints
    step_curve_perjoint_deg = np.degrees(abserr.mean(axis=0))        # (H,6)
    end_perjoint_deg = np.degrees(abserr[:, H - 1, :].mean(axis=0))  # (6,)
    end_grip = grip_abserr[:, H - 1]

    tols = [float(x) for x in args.tol_deg.split(",") if x]
    acc = {t: float((np.degrees(end_err) <= t).mean()) for t in tols}
    grip_acc = float((end_grip <= 0.1).mean())

    print(f"\n=== REAL-EPISODE ACTION PREDICTION · TRAINING set (open-loop / teacher-forced) ===")
    print(f"    {N} frames · {len(chosen)} episodes · horizon end = step {H} = state[t+{H}] "
          f"({H / fps * 1000:.0f} ms ahead)")
    print(f"\n  per-joint MAE at horizon end (pred[{H - 1}] vs state[t+{H}]):")
    for j in range(6):
        print(f"    {JN[j]}: {end_perjoint_deg[j]:5.2f}°")
    print(f"  -> overall joint MAE @ horizon end : {np.degrees(end_err.mean()):.2f}°  ({end_err.mean():.4f} rad)")
    print(f"  -> gripper MAE @ horizon end       : {end_grip.mean():.4f}")
    print(f"\n  ACCURACY (fraction of frames within tol at horizon end):")
    for t in tols:
        print(f"    within {t:>4.1f}° : {acc[t] * 100:5.1f}%")
    print(f"    gripper within 0.10 : {grip_acc * 100:5.1f}%")
    print(f"\n  error vs horizon step (mean over samples+joints):")
    for k in range(H):
        print(f"    step {k} (state[t+{k + 1}], {(k + 1) / fps * 1000:4.0f} ms): {step_curve_deg[k]:5.2f}°")
    print(f"\n  context:")
    print(f"    first step  (pred[0]  vs state[t+1])  : {np.degrees(first_err.mean()):.2f}°")
    print(f"    chunk mean  (pred[:]  vs state[t+1..]) : {np.degrees(chunk_err.mean()):.2f}°")
    print(f"    horizon end (pred[{H - 1}] vs state[t+{H}]) : {np.degrees(end_err.mean()):.2f}°   <-- headline")

    if per_task:
        print(f"\n  per-task (horizon-end MAE° · within-5° accuracy):")
        for tk in sorted(per_task):
            v = np.array(per_task[tk])
            print(f"    {np.degrees(v.mean()):5.2f}°  {float((v * DEG <= 5).mean()) * 100:5.1f}%   "
                  f"{tk[:46]!r}  (n={len(v)})")

    # ----- pass 2: dense whole-episode rollouts spanning every task -----
    print(f"\n>>> [pass 2] dense rollout: {args.roll_per_task} whole episode(s) per task, "
          f"stride {args.roll_stride}", flush=True)
    by_task: dict[str, list[int]] = {}
    for e in range(n_eps_total):
        if (ep_to[e] - ep_from[e]) > (H + 8) and _in_range(e):
            by_task.setdefault(ep_task.get(e, "?"), []).append(e)
    roll_eps = []
    for tk, eps in by_task.items():
        eps_sorted = sorted(eps, key=lambda e: ep_to[e] - ep_from[e], reverse=True)  # longest first
        roll_eps.extend(eps_sorted[: args.roll_per_task])

    rollouts = []          # metadata for JSON
    raw_roll = {}          # arrays for NPZ
    for ri, e in enumerate(roll_eps):
        a, b = ep_from[e], ep_to[e]
        n = min(b - a, args.roll_max_frames)
        real = episode_state_traj(ds, a, a + n)                       # (n,7)
        locs = list(range(0, n, args.roll_stride))
        preds = np.empty((len(locs), H, real.shape[1]), dtype=np.float32)
        tk = ep_task.get(e, "")
        for i, loc in enumerate(locs):
            tic = time.perf_counter()
            pred, _, tk2 = infer_chunk(policy, ds, a + loc, tk, image_tools, H)
            tinf.append(time.perf_counter() - tic)
            preds[i] = pred
            tk = tk or tk2
        raw_roll[f"roll{ri}_real"] = real
        raw_roll[f"roll{ri}_loc"] = np.array(locs, dtype=np.int32)
        raw_roll[f"roll{ri}_pred"] = preds
        rollouts.append({"idx": ri, "episode": int(e), "task": tk,
                         "n_frames": int(n), "n_infer": len(locs), "stride": args.roll_stride})
        print(f"  roll{ri}: ep{e:03d} {tk[:34]!r}  {n} frames, {len(locs)} inferences", flush=True)

    tarr = np.array(tinf) * 1000.0
    steady = tarr[1:] if len(tarr) > 1 else tarr
    lat = {"first_ms": float(tarr[0]), "median_ms": float(np.median(steady)),
           "mean_ms": float(steady.mean()), "p90_ms": float(np.percentile(steady, 90))}
    print(f"\n  infer latency: first {lat['first_ms']:.0f} ms (incl. JIT) · "
          f"steady median {lat['median_ms']:.0f} ms · {len(tarr)} inferences total")

    headline = (f"TRAIN-SET single-shot prediction: horizon-end MAE {np.degrees(end_err.mean()):.2f}°"
                + (f", within 5°: {acc.get(5.0, float('nan')) * 100:.0f}%" if 5.0 in acc else "")
                + f"; first-step MAE {np.degrees(first_err.mean()):.2f}°")
    print(f"\n>>> {headline}\n")

    if args.raw_out:
        np.savez_compressed(
            args.raw_out,
            sample_ep=S_ep, sample_loc=S_loc, sample_pred=S_pred, sample_gt=S_gt,
            sample_state=S_state, sample_task=np.array(S_task, dtype=object),
            ep_range=np.array(args.ep_range, dtype=object), split=np.array(args.split, dtype=object),
            step_curve_deg=step_curve_deg, step_curve_perjoint_deg=step_curve_perjoint_deg,
            fps=np.float32(fps), horizon=np.int32(H),
            joint_names=np.array(JN, dtype=object),
            roll_count=np.int32(len(rollouts)),
            roll_meta=np.array(json.dumps(rollouts), dtype=object),
            **raw_roll,
        )
        print(f"wrote {args.raw_out}  ({N} samples + {len(rollouts)} rollouts)")

    if args.out:
        out = {
            "model": args.model, "config": args.config, "checkpoint": args.checkpoint_dir,
            "repo_id": args.repo_id, "split": args.split, "metric": "open_loop_teacher_forced",
            "ep_range": args.ep_range, "ep_lo": (_ep_lo if args.ep_range else None),
            "ep_hi": (_ep_hi if args.ep_range else None),
            "episodes_used": sorted(int(e) for e in set(chosen)),
            "fps": fps, "n_episodes": len(chosen), "n_samples": N, "horizon": H,
            "lookahead_ms": H / fps * 1000.0,
            "mid_window": [args.mid_lo, args.mid_hi], "seed": args.seed,
            "headline": headline,
            "end_of_horizon": {
                "overall_joint_mae_rad": float(end_err.mean()),
                "overall_joint_mae_deg": float(np.degrees(end_err.mean())),
                "per_joint_mae_deg": [float(x) for x in end_perjoint_deg],
                "gripper_mae": float(end_grip.mean()),
                "accuracy_within_deg": {str(t): acc[t] for t in tols},
                "gripper_accuracy_within_0.1": grip_acc,
            },
            "horizon_step_curve_deg": [float(x) for x in step_curve_deg],
            "horizon_step_perjoint_deg": [[float(v) for v in row] for row in step_curve_perjoint_deg],
            "context": {
                "first_step_mae_deg": float(np.degrees(first_err.mean())),
                "chunk_mean_mae_deg": float(np.degrees(chunk_err.mean())),
            },
            "per_task_end_mae_deg": {tk: float(np.degrees(np.mean(v))) for tk, v in per_task.items()},
            "per_task_acc_within_5deg": {tk: float((np.array(v) * DEG <= 5).mean()) for tk, v in per_task.items()},
            "rollouts": rollouts,
            "latency_ms": lat,
            "raw_npz": os.path.basename(args.raw_out) if args.raw_out else "",
        }
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
