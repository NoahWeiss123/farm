# Getting started

Three minutes from `pip install` to a sim arm moving on the dashboard. This is the
load-bearing TTHW path; if anything in here takes longer than the headline number,
file an issue.

You do not need a robot arm for this walkthrough. The `lerobot-mock` driver
ships with the Edge Agent and runs entirely on your laptop.

## Prerequisites

- Python 3.11 or newer
- A working network connection (`farm doctor network` will check it)
- One free terminal window
- No GPU, no arm, no camera required for the sim path

If you are setting up a real UFactory 850 instead, jump to
[hardware.md](hardware.md) and [`farm doctor real-arm`](cli-reference.md#farm-doctor-real-arm)
after step 3 below.

## 1. Install the Edge Agent

```bash
pip install farm-edge-agent
farm version
```

`farm version` prints the agent version and the wire protocol versions it
supports. If this command fails, the install did not land on your `PATH` —
re-run `pip install` inside a fresh virtualenv.

## 2. Sign in and write a config

```bash
farm quickstart
```

`farm quickstart` is the one-shot setup. It:

1. Opens a browser tab and asks you to confirm a one-time device code.
2. Writes `~/.farm/config.yaml` populated with a sandbox API key and the
   `lerobot-mock` driver.
3. Connects to the dispatcher, negotiates the wire protocol, and runs the
   canned `wave_hello` task against the sim arm.
4. Prints a dashboard URL that streams the run live.

You should see something like:

```
> Opening https://farm.dev/quickstart/abc123 ... press Enter when you've signed in.
> Wrote ~/.farm/config.yaml with sandbox key + lerobot-mock driver.
> Connecting to dispatcher ... protocol v1.2 OK.
> Running canned task 'wave_hello' on sim arm ...
> Watch it: https://farm.dev/run/r_8x2k
```

## 3. Watch the sim arm move

Open the URL printed at the end of step 2. The dashboard shows the trajectory
ghost overlay, the router's reasoning, the chosen backend per plan node, and
the safety panel. The sim arm runs at the same 30 Hz cadence the real arm
would.

## 4. Run a task from the CLI

The `quickstart` script ran one canned task. To dispatch your own:

```bash
farm run "wave hello on the sim arm" --backend classical
```

`--backend classical` forces the deterministic
[classical-planner](safety.md#recovery-primitives) path so the first run does
not depend on a warmed GPU container. Drop the flag to let the router pick
([auto mode](python-api.md#run-modes)).

## 5. Inspect the run record

Every run writes a structured JSONL record to `~/.farm/runs/<run-id>/`.
Download a portable copy:

```bash
farm export r_8x2k --format lerobot --out ./r_8x2k/
```

The export contains the prompt, the plan DAG, every action chunk, every
observation (downsampled), every [safety event](safety.md), and the critic
notes. See [python-api.md](python-api.md#exporters) for programmatic access.

## 6. What's next

- Real arm setup: [hardware.md](hardware.md) + `farm doctor real-arm`
- All CLI subcommands: [cli-reference.md](cli-reference.md)
- All config keys: [config-reference.md](config-reference.md)
- Driving FARM from a notebook: [python-api.md](python-api.md)
- If something is broken: [faq.md](faq.md) and [errors.md](errors.md)
