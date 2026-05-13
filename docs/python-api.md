# Python API

The CLI is a thin wrapper over `farm.Client`. Research labs drive FARM from
notebooks and experiment scripts; the CLI is for interactive sessions. Every
subcommand in [cli-reference.md](cli-reference.md) maps 1:1 to a method on
`Client` or one of its sub-objects.

## Installation

```bash
pip install farm-edge-agent
```

The package ships both the CLI entrypoint (`farm`) and the importable
library (`farm`).

## Constructing a client

```python
from farm import Client

# Reads ~/.farm/config.yaml. Equivalent to running CLI commands.
client = Client()

# Or pass overrides explicitly.
client = Client(
    api_key="...",
    workspace="my-lab",
    config_path="./alt-config.yaml",
)
```

`Client()` honours the precedence rules described in
[config-reference.md](config-reference.md#precedence): explicit kwargs win
over env vars, which win over the config file, which wins over compiled-in
defaults.

## Dispatching a run

```python
run = client.run(
    "pick the red block and stack it on the blue one",
    task_id="exp_42_seed_7",
    backend="auto",
)
```

`run` returns a `Run` object immediately; dispatch is asynchronous on the
server side. The local handle is the cursor into that run's event stream.

### Run modes

`backend` is one of:

- `auto` — the LLM router picks a backend per plan node from the registered
  [capability cards](capability-cards.md). This is the default.
- `pi05-ft` — force the fine-tuned π0.5 controller.
- `gemini-robotics` — force the Gemini Robotics controller.
- `classical` — force the deterministic classical-planner backend. No cloud
  inference; runs entirely inside the Edge Agent.

`client.run(..., offline=True)` is equivalent to `backend="classical"` and
additionally suppresses all cloud calls (planner, critic, telemetry upload).
Use it on disconnected machines or for [offline mode](safety.md#offline-mode).

## Streaming events

```python
for event in run.events():
    if event.type == "router_decision":
        print(event.chosen_backend, event.reason)
    elif event.type == "action_chunk":
        print(event.timestamp, len(event.actions))
    elif event.type == "safety_event":
        print(event.rule, event.description)
    elif event.type == "critic_note":
        print(event.text)
```

`run.events()` yields `RunEvent` objects in dispatch order until the run
terminates. Each event corresponds to a line in the JSONL
[run record](#run-records).

Block until the run finishes:

```python
run.wait()                       # blocks until terminal state
run.wait(timeout=120)            # raises TimeoutError after 120s
status = run.status              # 'succeeded' | 'failed' | 'safety_stop' | 'timeout'
```

## Run records

```python
record = run.record()            # parsed JSON dict
df = run.to_dataframe()          # pandas DataFrame of LeRobot-shaped frames
```

The record includes the original prompt, the plan DAG (with router
reasoning per node), every action chunk, downsampled observations, every
[safety event](safety.md), [critic notes](safety.md#the-trailing-critic), and
the wall-clock + cost breakdown. Schema is stable across Phase-MVP wire
protocol versions; see [upgrading.md](upgrading.md).

## Listing and exporting past runs

```python
records = client.runs.list(workspace="my-lab", since="2026-05-01")

# Export by run id.
client.runs.export("r_8x2k", format="jsonl")
client.runs.export("r_8x2k", format="lerobot", out="./r_8x2k/")
```

`runs.list` accepts `since` / `until` ISO timestamps, `task_id` filters,
and a `limit`. `runs.export` accepts `format="jsonl" | "lerobot"`.

## Exporters

LeRobot-format export is the killer feature for the research-lab user: the
shards drop straight into a paper appendix without re-shaping.

```python
client.runs.export(
    "r_8x2k",
    format="lerobot",
    out="./datasets/exp_42/",
    include_frames=True,          # write decoded JPEGs alongside the trajectory
)
```

The exporter writes one episode per run, with the per-frame schema described
in DESIGN.md's Fine-Tuning Plan: wrist camera frame, optional overhead
frame, joint state, TCP pose, action (ee-pose delta in base frame), and the
canonical language instruction.

## Validating a capability card from Python

```python
from farm import card

result = card.validate_file("./mycard.yaml")
if not result.ok:
    for err in result.errors:
        print(err.path, err.message, err.suggestion)
```

`result.errors` is the same structured set the CLI's
[`farm card validate`](cli-reference.md#farm-card-validate) emits. See
[capability-cards.md](capability-cards.md).

## Errors

`farm.Client` raises typed exceptions whose names match the
[error catalog](errors.md):

```python
from farm import errors

try:
    run = client.run("...", backend="pi05-ft")
except errors.WatchdogTimeout as e:
    # FARM-E3002
    ...
except errors.ProtocolMismatch as e:
    # FARM-E1006
    ...
```

The full mapping from exception class to error code is in
[errors.md](errors.md).
