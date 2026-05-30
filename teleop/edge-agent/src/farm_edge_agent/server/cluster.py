"""Cluster training bridge — launch + monitor π0.5 fine-tunes on the H100 cluster.

The daemon runs on the laptop, training runs on the SLURM cluster. This module
shells out to ``kubectl`` (which the laptop has) to submit and watch jobs, so the
``/train`` dashboard page can drive training without leaving the browser.

Everything degrades gracefully: if ``kubectl`` is missing or the login pod can't
be found, the calls return ``{"error": ...}`` and the page shows "cluster
unavailable" rather than breaking. Nothing here runs unless the user clicks
Launch (or the page polls status).

Configurable via env (sensible defaults for the CS153 setup):
  FARM_CLUSTER_USER     cluster username           (default: nhweiss)
  FARM_CLUSTER_NS       SLURM namespace            (default: slurm)
  FARM_CLUSTER_WORKDIR  staged launcher dir on pod (default: ~/farm-train)
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import time

log = logging.getLogger("farm.cluster")

USER = os.environ.get("FARM_CLUSTER_USER", "nhweiss")
NS = os.environ.get("FARM_CLUSTER_NS", "slurm")
WORKDIR = os.environ.get("FARM_CLUSTER_WORKDIR", "~/farm-train")
SELECTOR = os.environ.get("FARM_CLUSTER_SELECTOR", f"stanford/user={USER}")

# model key → (sbatch script, log-file prefix, openpi config name, default steps/gpus)
MODELS: dict[str, dict] = {
    "full": {"script": "train_pi05.sbatch", "log": "train", "config": "pi05_farm_uf850",
             "label": "Full fine-tune", "steps": 20000, "gpus": 8},
    "lora": {"script": "train_pi05_lora.sbatch", "log": "train-lora", "config": "pi05_farm_uf850_lora",
             "label": "LoRA", "steps": 12000, "gpus": 1},
    "gse": {"script": "train_pi05_gse.sbatch", "log": "train-gse", "config": "pi05_farm_uf850_gse",
            "label": "GSE", "steps": 3000, "gpus": 4},
}

# openpi emits two log shapes. Every step, a tqdm progress line:
#   "Progress on: 378it/3.00kit rate:1.5s/it remaining:1:05:10 elapsed:10:30 postfix:-"
# and — every log_interval, when not swallowed by openpi's tqdm→logging redirect
# — a metrics line: "Step 300: loss=0.0009, grad_norm=0.0334, param_norm=1806.0".
# Parse the metrics line order-independently (key order varies by config) and
# always read the tqdm line, which reliably carries step / rate / remaining.
_STEP_RE = re.compile(r"Step (\d+):\s*(.+)")
_KV_RE = re.compile(r"([A-Za-z_]+)=([0-9.eE+-]+)")
# Both counts can carry a 'k' suffix once past 1000 (e.g. "1.33kit/3.00kit").
_TQDM_RE = re.compile(
    r"Progress on:\s*([0-9.]+)(k?)it/([0-9.]+)(k?)it\s+rate:([0-9.]+)(s/it|it/s)\s+"
    r"remaining:([0-9:]+)\s+elapsed:([0-9:]+)(?:\s+postfix:(\S+))?"
)
_SUBMIT_RE = re.compile(r"Submitted batch job (\d+)")

# Pod-name cache so we don't re-discover every poll.
_pod_cache: dict[str, float | str] = {"name": "", "at": 0.0}
# Adopted-job cache (discover() is called on idle no-job polls; keep it cheap).
_discover_cache: dict[str, object] = {"val": None, "at": 0.0}


def _hms(s: str) -> int:
    """Colon-clock to seconds: '1:05:10'→4210, '10:30'→630, '09'→9."""
    sec = 0
    for part in s.split(":"):
        if part:
            sec = sec * 60 + int(part)
    return sec


def parse_log(text: str) -> dict:
    """Parse openpi train stdout into loss history + live progress. Pure — unit-tested.

    Returns parallel ``steps``/``loss``/``grad_norm`` lists from the metrics
    lines (often empty — openpi's tqdm→logging redirect can swallow them), plus
    ``progress`` from the last tqdm line, or ``None`` before training starts:
    ``{step, total, s_per_it, it_per_s, remaining_s, elapsed_s}``.
    """
    steps: list[int] = []
    loss: list[float] = []
    grad: list[float | None] = []
    for m in _STEP_RE.finditer(text):
        kv = dict(_KV_RE.findall(m.group(2)))
        if "loss" not in kv:
            continue
        steps.append(int(m.group(1)))
        loss.append(float(kv["loss"]))
        grad.append(float(kv["grad_norm"]) if "grad_norm" in kv else None)
    progress = None
    for m in _TQDM_RE.finditer(text):
        rate = float(m.group(5))
        s_per_it = rate if m.group(6) == "s/it" else (1.0 / rate if rate else 0.0)
        it_per_s = (1.0 / rate if rate else 0.0) if m.group(6) == "s/it" else rate
        progress = {
            "step": int(round(float(m.group(1)) * (1000 if m.group(2) == "k" else 1))),
            "total": int(round(float(m.group(3)) * (1000 if m.group(4) == "k" else 1))),
            "s_per_it": round(s_per_it, 3),
            "it_per_s": round(it_per_s, 3),
            "remaining_s": _hms(m.group(7)),
            "elapsed_s": _hms(m.group(8)),
        }
        # Loss occasionally rides in the tqdm postfix (set_postfix(loss=…)).
        post = m.group(9) or ""
        if post and post != "-":
            kv = dict(_KV_RE.findall(post))
            if "loss" in kv and (not steps or steps[-1] != progress["step"]):
                steps.append(progress["step"])
                loss.append(float(kv["loss"]))
                grad.append(None)
    return {"steps": steps, "loss": loss, "grad_norm": grad, "progress": progress}


def available() -> bool:
    """Is kubectl on PATH?"""
    from shutil import which
    return which("kubectl") is not None


def _kubectl(args: list[str], *, timeout: float = 15.0) -> tuple[int, str]:
    try:
        p = subprocess.run(  # noqa: S603
            ["kubectl", *args], capture_output=True, text=True, timeout=timeout,
        )
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, "kubectl not found on PATH"
    except subprocess.TimeoutExpired:
        return 124, "kubectl timed out"
    except Exception as exc:  # noqa: BLE001
        return 1, str(exc)


def _pod(*, force: bool = False) -> str | None:
    """Discover the login pod name (cached 60s)."""
    now = time.monotonic()
    if not force and _pod_cache["name"] and now - float(_pod_cache["at"]) < 60.0:
        return str(_pod_cache["name"])
    rc, out = _kubectl([
        "get", "pod", "-n", NS, "-l", SELECTOR,
        "-o", "jsonpath={.items[0].metadata.name}",
    ])
    name = out.strip()
    if rc != 0 or not name:
        return None
    _pod_cache.update(name=name, at=now)
    return name


def _exec(remote_cmd: str, *, timeout: float = 20.0) -> tuple[int, str]:
    """Run a shell command on the login pod as the cluster user."""
    pod = _pod()
    if pod is None:
        return 1, f"login pod not found (selector {SELECTOR!r} in ns {NS!r})"
    return _kubectl(
        ["exec", "-n", NS, pod, "-c", "login", "--",
         "runuser", "-u", USER, "--", "bash", "-lc", remote_cmd],
        timeout=timeout,
    )


def launch(model: str, steps: int, gpus: int) -> dict:
    """Submit a training job. Returns ``{"job_id", "model", ...}`` or ``{"error"}``."""
    spec = MODELS.get(model)
    if spec is None:
        return {"error": f"unknown model {model!r}"}
    steps = max(1, int(steps))
    gpus = max(1, int(gpus))
    cmd = (
        f"cd {WORKDIR} && NUM_TRAIN_STEPS={steps} "
        f"sbatch --gres=gpu:{gpus} {spec['script']} 2>&1"
    )
    rc, out = _exec(cmd, timeout=30.0)
    m = _SUBMIT_RE.search(out)
    if rc != 0 or not m:
        return {"error": f"sbatch failed: {out.strip()[:400]}"}
    return {"job_id": m.group(1), "model": model, "steps": steps, "gpus": gpus,
            "config": spec["config"]}


def status(job_id: str, model: str, total_steps: int) -> dict:
    """SLURM state + parsed loss history for a running/finished job."""
    spec = MODELS.get(model, {})
    logfile = f"{spec.get('log', 'train')}-{job_id}.out"
    # SLURM state (PENDING/RUNNING/COMPLETED/…) — sacct survives after the job ends.
    rc, st = _exec(
        f"squeue -j {job_id} -h -o '%T' 2>/dev/null | head -1; "
        f"sacct -j {job_id} -X -n -o State 2>/dev/null | head -1",
        timeout=15.0,
    )
    state = "UNKNOWN"
    for tok in st.split():
        tok = tok.strip()
        if tok and tok.upper() not in ("UNKNOWN",):
            state = tok.upper()
            break
    # Log tail → loss history + live progress (step / rate / ETA from tqdm).
    rc2, tail = _exec(f"tail -n 600 {WORKDIR}/{logfile} 2>/dev/null", timeout=20.0)
    hist = parse_log(tail)
    prog = hist.get("progress")
    metric_step = hist["steps"][-1] if hist["steps"] else 0
    last_step = max(metric_step, prog["step"] if prog else 0)
    # The tqdm line carries the real total (self-corrects an adopted default).
    total = prog["total"] if (prog and prog.get("total")) else int(total_steps)
    phase = _phase(state, last_step > 0)
    return {
        "job_id": job_id, "model": model, "state": state, "phase": phase,
        "total_steps": int(total), "step": last_step,
        "loss": hist["loss"], "steps": hist["steps"], "grad_norm": hist["grad_norm"],
        "s_per_it": prog["s_per_it"] if prog else None,
        "it_per_s": prog["it_per_s"] if prog else None,
        "remaining_s": prog["remaining_s"] if prog else None,
        "elapsed_s": prog["elapsed_s"] if prog else None,
        "log_tail": "\n".join(tail.strip().splitlines()[-12:]),
    }


def _phase(state: str, has_step: bool) -> str:
    if state in ("PENDING", "CONFIGURING"):
        return "queued"
    if state in ("COMPLETED",):
        return "done"
    if state in ("FAILED", "CANCELLED", "CANCELLED+", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"):
        return "stopped"
    if state == "RUNNING":
        return "training" if has_step else "starting"
    return "starting" if has_step is False else "training"


def _model_from_name(name: str) -> str | None:
    """Map a SLURM job name (e.g. ``farm-pi05-gse``) to a known model key."""
    n = name.lower()
    if "gse" in n:
        return "gse"
    if "lora" in n:
        return "lora"
    if n.startswith("farm-pi05") or n == "pi05":
        return "full"
    return None


def _elapsed_to_s(s: str) -> float:
    """SLURM elapsed ('D-HH:MM:SS' / 'HH:MM:SS' / 'MM:SS') to seconds."""
    s = s.strip()
    if not s or s == "-":
        return 0.0
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    return days * 86400 + _hms(s)


def discover() -> dict | None:
    """Find a running/queued FARM job to adopt, so the dashboard survives a
    daemon restart and reflects jobs launched out-of-band. Cached 10 s. Returns
    a job dict shaped like ``launch`` (plus ``adopted: True``), or ``None``."""
    now = time.monotonic()
    if _discover_cache["at"] and now - float(_discover_cache["at"]) < 10.0:  # type: ignore[arg-type]
        return _discover_cache["val"]  # type: ignore[return-value]
    _, out = _exec("squeue -u \"$USER\" -h -o '%i|%j|%T|%M' 2>/dev/null", timeout=15.0)
    found = None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        jid, name, st, elapsed = parts[0], parts[1], parts[2].upper(), parts[3]
        model = _model_from_name(name)
        if model and st in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "RESIZING"):
            spec = MODELS[model]
            found = {
                "job_id": jid, "model": model, "total_steps": spec["steps"],
                "gpus": spec["gpus"], "config": spec["config"],
                "started_at": time.time() - _elapsed_to_s(elapsed), "adopted": True,
            }
            break
    _discover_cache.update(val=found, at=now)
    return found


def parse_metrics(text: str) -> dict:
    """Parse the ``srun … nvidia-smi`` + loadavg blob. Pure — unit-tested.

    Expects nvidia-smi CSV rows ``index, util, mem_used, mem_total`` (no header,
    no units), then a ``CPU`` marker, a /proc/loadavg line, and an nproc count.
    Returns ``{"gpus": [{index, util, mem_used, mem_total, mem_pct}], "cpu": {...}}``.
    """
    gpus: list[dict] = []
    cpu: dict = {}
    section = "gpu"
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "CPU":
            section = "cpu"
            continue
        if section == "gpu":
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4 and all(p.lstrip("-").isdigit() for p in parts):
                idx, util, used, total = (int(p) for p in parts)
                gpus.append({
                    "index": idx, "util": util, "mem_used": used, "mem_total": total,
                    "mem_pct": round(100 * used / total, 1) if total else 0,
                })
        else:  # cpu section: loadavg line then nproc
            toks = line.split()
            if len(toks) >= 3 and _isfloat(toks[0]) and "load1" not in cpu:
                cpu["load1"] = float(toks[0])
            elif line.isdigit():
                cpu["ncpu"] = int(line)
    if "load1" in cpu and cpu.get("ncpu"):
        cpu["pct"] = round(min(100.0, 100.0 * cpu["load1"] / cpu["ncpu"]), 1)
    return {"gpus": gpus, "cpu": cpu}


def _isfloat(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def metrics(job_id: str) -> dict:
    """Per-GPU utilization + CPU load for a running job, via ``srun --overlap``
    into its allocation (the SLURM-blessed way to inspect a live job's node)."""
    remote = (
        f"srun --jobid={job_id} --overlap --quiet --ntasks=1 bash -lc "
        "'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader,nounits 2>/dev/null; echo CPU; "
        "cat /proc/loadavg 2>/dev/null; nproc 2>/dev/null'"
    )
    rc, out = _exec(remote, timeout=25.0)
    if rc != 0 and "nvidia-smi" not in out:
        return {"gpus": [], "cpu": {}, "error": out.strip()[:160]}
    return parse_metrics(out)


def stop(job_id: str) -> dict:
    rc, out = _exec(f"scancel {job_id} 2>&1", timeout=15.0)
    return {"ok": rc == 0, "out": out.strip()[:200]}
