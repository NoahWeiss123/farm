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

# Serve launchers (distinct from training): key → sbatch script + metadata. The
# serve binds :SERVE_PORT on its worker; the dashboard relays it to the laptop via
# a two-hop tunnel — login-pod ``socat`` → worker, then a laptop-side
# ``kubectl port-forward`` → the login pod (managed in server/app.py). The worker
# isn't a pod the laptop can see, so the login-pod socat hop is required.
SERVE_PORT = int(os.environ.get("FARM_SERVE_PORT", "8000"))
# Serve jobs request a SHORT walltime + modest memory on purpose: the cluster's
# priority is flat FIFO and idle GPUs get reserved for earlier jobs, so a long
# (4 h, whole-node-memory) serve queues for hours. A ~1.5 h / 96 G job slots into
# a backfill gap and usually starts within minutes. Re-launch from the dashboard
# to extend a session. Override via env if a node is wide open.
SERVE_WALLTIME = os.environ.get("FARM_SERVE_WALLTIME", "01:30:00")
SERVE_MEM = os.environ.get("FARM_SERVE_MEM", "96G")
SERVE_MODELS: dict[str, dict] = {
    "fft": {
        "script": "serve_fft_multiobject.sbatch", "log": "serve",
        "label": "FFT multiobject (all 4 objects)", "default_step": "latest",
        "steps": ["latest", "8000", "16000", "24000", "32000", "40000", "48000", "55999"],
        "job_name": "serve-fft-multiobj",
    },
    "fftlora_hotswap": {
        # FFT-56k base resident + every per-object LoRA preloaded; the serve
        # routes the incoming task prompt to a skill and hot-swaps just the
        # adapter leaves (no base reload, no recompile). Drives the /user page.
        "script": "serve_fftlora_hotswap.sbatch", "log": "serve",
        "label": "FFT-56k + LoRA hot-swap (all skills)", "default_step": "55999",
        "steps": ["55999"],
        "job_name": "serve-fftlora-hotswap",
    },
    "lora_gse": {
        "script": "serve_pi05_lora_gse.sbatch", "log": "serve",
        "label": "LoRA-off-GSE (bottle)", "default_step": "9999",
        "steps": ["2000", "4000", "6000", "8000", "9999"],
        "job_name": "serve-lora-gse",
    },
}
# Serve lifecycle markers in the job log. The sbatch echoes _LAUNCHED just
# BEFORE exec'ing serve_policy.py (params still loading + JIT); openpi logs
# _BOUND only once the websocket server is actually accepting — that's the real
# "ready to infer" signal. (A bare TCP probe to the forwarded port is NOT
# reliable: kubectl port-forward's local listener accepts before the upstream
# serve exists, so it reports reachable while the container is still building.)
_SERVE_LAUNCHED_RE = re.compile(r">>> serve_policy\.py on :\d+")
_SERVE_BOUND_RE = re.compile(r"server listening on|Creating server \(host")

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


# ── Serving the policy from the dashboard ──────────────────────────────────
# The serve job is submitted like training, but instead of tracking loss we
# track its lifecycle (queued → starting → loading → serving) and stand up the
# login-pod side of the tunnel. The laptop-side port-forward lives in app.py.

def serve_launch(model: str, step: str | None) -> dict:
    """Submit a serve job (``STEP=<step> sbatch <serve script>``). Returns
    ``{"job_id", "model", "step"}`` or ``{"error"}``."""
    spec = SERVE_MODELS.get(model)
    if spec is None:
        return {"error": f"unknown serve model {model!r}"}
    step = str(step or spec["default_step"])
    if spec.get("steps") and step not in spec["steps"]:
        return {"error": f"step {step!r} not in {spec['steps']}"}
    cmd = (
        f"cd {WORKDIR} && STEP={step} "
        f"sbatch --time={SERVE_WALLTIME} --mem={SERVE_MEM} {spec['script']} 2>&1"
    )
    rc, out = _exec(cmd, timeout=30.0)
    m = _SUBMIT_RE.search(out)
    if rc != 0 or not m:
        return {"error": f"sbatch failed: {out.strip()[:400]}"}
    return {"job_id": m.group(1), "model": model, "step": step}


def serve_state(job_id: str) -> tuple[str, str]:
    """``(state, node)`` for a serve job. ``node`` is empty until it RUNs.
    ``squeue`` ``%R`` is the nodelist when running, or the pend reason in parens."""
    rc, out = _exec(f"squeue -j {job_id} -h -o '%T|%R' 2>/dev/null | head -1", timeout=15.0)
    line = out.strip().splitlines()[0] if out.strip() else ""
    if "|" not in line:
        return ("GONE", "")  # not in queue (finished/cancelled/never-ran)
    state, node = (p.strip() for p in line.split("|", 1))
    if node.startswith("("):  # e.g. "(Resources)", "(Priority)" — still pending
        node = ""
    return (state.upper() or "UNKNOWN", node)


def serve_log_tail(job_id: str, n: int = 30) -> str:
    rc, out = _exec(f"tail -n {n} {WORKDIR}/serve-{job_id}.out 2>/dev/null || true", timeout=20.0)
    return out


def serve_markers(job_id: str) -> tuple[bool, bool]:
    """``(launched, bound)`` by grepping the WHOLE job log. The startup markers
    appear exactly once and the log only grows, so a 30-line tail scrolls past
    them and detection flickers — grep the full file instead (it's monotonic)."""
    f = f"{WORKDIR}/serve-{job_id}.out"
    rc, out = _exec(
        f"grep -qE '>>> serve_policy[.]py on :' {f} 2>/dev/null && echo LAUNCHED; "
        f"grep -qE 'server listening on|Creating server [(]host' {f} 2>/dev/null && echo BOUND; "
        f"true",
        timeout=15.0,
    )
    return ("LAUNCHED" in out, "BOUND" in out)


def serve_is_launched(log_text: str) -> bool:
    """serve_policy.py has been exec'd — params restoring / JIT (not yet bound)."""
    return bool(_SERVE_LAUNCHED_RE.search(log_text))


def serve_is_bound(log_text: str) -> bool:
    """The websocket server is actually accepting connections — ready to infer.
    This (not a TCP probe) is the authoritative 'serving' signal."""
    return bool(_SERVE_BOUND_RE.search(log_text))


_SOCAT_TMUX = "farm_serve_socat"


def serve_socat_up(node: str, *, port: int = SERVE_PORT) -> bool:
    """(Re)point the login-pod relay ``socat :port → <node>:port``, inside a
    DETACHED tmux session. A plain ``... &`` backgrounded forking listener holds
    the one-shot ``kubectl exec`` channel open and gets torn down (exit 143);
    tmux daemonizes it so the exec returns immediately and the relay survives.
    Idempotent (kills any prior session first). Returns True iff the relay is
    actually accepting connections afterward (real /dev/tcp probe, not pgrep)."""
    if not node:
        return False
    remote = (
        f"tmux kill-session -t {_SOCAT_TMUX} 2>/dev/null; "
        f"tmux new-session -d -s {_SOCAT_TMUX} "
        f"'socat TCP-LISTEN:{port},fork,reuseaddr TCP:{node}:{port}'; "
        f"sleep 0.8; "
        f"if tmux has-session -t {_SOCAT_TMUX} 2>/dev/null && "
        f"(exec 3<>/dev/tcp/127.0.0.1/{port}) 2>/dev/null; then echo SOCAT_UP; else echo SOCAT_FAIL; fi"
    )
    rc, out = _exec(remote, timeout=20.0)
    return "SOCAT_UP" in out


def serve_socat_down(*, port: int = SERVE_PORT) -> None:
    _exec(f"tmux kill-session -t {_SOCAT_TMUX} 2>/dev/null; "
          f"pkill -f 'socat .*TCP-LISTEN:{port}' 2>/dev/null; true", timeout=10.0)


def serve_stop(job_id: str, *, port: int = SERVE_PORT) -> dict:
    """Cancel the serve job and tear down the login-pod relay."""
    rc, out = _exec(f"scancel {job_id} 2>&1; tmux kill-session -t {_SOCAT_TMUX} 2>/dev/null; "
                    f"pkill -f 'socat .*TCP-LISTEN:{port}' 2>/dev/null; true",
                    timeout=15.0)
    return {"ok": rc == 0, "out": out.strip()[:200]}


def serve_discover() -> dict | None:
    """Find a running/queued serve job to adopt (so the dashboard survives a
    daemon restart). Matched by the serve job name."""
    names = {spec["job_name"]: key for key, spec in SERVE_MODELS.items()}
    _, out = _exec("squeue -u \"$USER\" -h -o '%i|%j|%T' 2>/dev/null", timeout=15.0)
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        jid, name, st = parts[0], parts[1], parts[2].upper()
        if name in names and st in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING"):
            return {"job_id": jid, "model": names[name], "adopted": True}
    return None
