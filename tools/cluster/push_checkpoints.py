"""Background checkpoint pusher: watch openpi's checkpoint dir during a
training run and stream each new checkpoint to a HuggingFace model repo,
tagged with its step number.

Launched as a background process from inside the sbatch's container.
Exits cleanly when the parent process (training) exits.

Layout the pusher consumes::

    <checkpoint_dir>/
        1000/      ← step number
            params/
            tree.json
            ...
        2000/
        ...

Each step subdir is uploaded as a single commit, then tagged ``step-<N>``
so downstream eval can pin via ``hf download <repo>
--revision step-19999``.

Run::

    python push_checkpoints.py \\
        --checkpoint-dir /path/to/checkpoints/exp_name \\
        --repo-id NoahWeiss/farm_uf850_pi0_fast \\
        --poll-interval 60
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import sys
import time
from pathlib import Path

# Lazy import so the script doesn't blow up if huggingface_hub isn't on
# the system path (the sbatch sources the openpi venv before launching us).
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

STEP_RE = re.compile(r"^\d+$")


def discover_steps(root: Path) -> list[int]:
    if not root.is_dir():
        return []
    return sorted(
        int(p.name) for p in root.iterdir()
        if p.is_dir() and STEP_RE.match(p.name)
    )


def push_step(api: HfApi, repo_id: str, step_dir: Path, step: int) -> None:
    """Upload one step's directory + tag it. Idempotent on retry: if the
    tag already exists, skip — we've already pushed this step."""
    tag = f"step-{step}"
    # Check if this tag already exists; if so, skip.
    try:
        refs = api.list_repo_refs(repo_id, repo_type="model")
        if any(t.name == tag for t in refs.tags):
            print(f"[push] {tag}: already tagged, skipping", flush=True)
            return
    except Exception as exc:
        print(f"[push] list_refs failed (continuing): {exc}", flush=True)

    print(f"[push] {tag}: uploading {step_dir}", flush=True)
    t0 = time.time()
    api.upload_folder(
        folder_path=str(step_dir),
        path_in_repo=tag,        # land under step-<N>/ in the repo
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"checkpoint @ step {step}",
    )
    try:
        api.create_tag(repo_id, tag=tag, repo_type="model")
    except HfHubHTTPError as exc:
        # Conflict = already tagged from a prior partial run; not fatal.
        print(f"[push] tag create failed (likely already exists): {exc}", flush=True)
    elapsed = time.time() - t0
    print(f"[push] {tag}: done in {elapsed:.0f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--poll-interval", type=int, default=60)
    ap.add_argument("--keep-period", type=int, default=2000,
                    help="Only push steps where step %% keep_period == 0 "
                         "(must match openpi's TrainConfig.keep_period — "
                         "openpi deletes the other checkpoints).")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[push] FATAL: HF_TOKEN env var not set", file=sys.stderr)
        return 1
    api = HfApi(token=token)

    # Ensure the model repo exists before the first upload — upload_folder
    # does NOT auto-create, so without this the step-5000 push would fail
    # with "repo not found". Idempotent (exist_ok=True). Created PUBLIC:
    # full π0.5 checkpoints are tens of GB and a private repo would blow the
    # free-tier storage quota; public also matches the open-source dataset.
    # (If the repo already exists, exist_ok leaves its visibility untouched.)
    try:
        api.create_repo(args.repo_id, repo_type="model", private=False, exist_ok=True)
        print(f"[push] ensured model repo exists: {args.repo_id}", flush=True)
    except Exception as exc:
        print(f"[push] WARN: create_repo failed (continuing): {exc}", flush=True)

    # Race-safe shutdown: parent (training) exiting → we drain + exit.
    stop = {"flag": False}
    def _sig(_signum, _frame):
        stop["flag"] = True
        print("[push] received stop signal, draining once before exit", flush=True)
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    seen: set[int] = set()
    print(f"[push] watching {args.checkpoint_dir} → {args.repo_id} every {args.poll_interval}s", flush=True)

    while True:
        steps = discover_steps(args.checkpoint_dir)
        # Normal poll: push only the kept-forever checkpoints (multiples
        # of keep_period). openpi atomically renames the step dir on
        # save, so once we see it, it's complete. The other checkpoints
        # (e.g. step 1000, 3000, 5000) get deleted by openpi within ~15
        # min of being written, so pushing them is a race we'd lose.
        # Drain pass (on SIGTERM): push everything we haven't pushed,
        # including the final step if it isn't a keep_period multiple.
        if stop["flag"]:
            to_push = [s for s in steps if s not in seen]
        else:
            to_push = [s for s in steps if s % args.keep_period == 0 and s not in seen]

        for s in to_push:
            step_dir = args.checkpoint_dir / str(s)
            try:
                push_step(api, args.repo_id, step_dir, s)
                seen.add(s)
            except Exception as exc:
                # Don't crash on transient errors — log and retry next tick.
                print(f"[push] step {s} failed (will retry): {exc}", flush=True)

        if stop["flag"]:
            print(f"[push] drained {len(seen)} step(s); exiting", flush=True)
            return 0

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())
