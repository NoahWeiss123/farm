from __future__ import annotations

import click

from farm_edge_agent.cli.commands import stub


@click.command("run", help="One-shot dispatch from CLI; streams events to stdout.")
@click.argument("task", required=False)
@click.option("--offline", is_flag=True, default=False,
              help="Run using only the local classical-planner; no cloud.")
@click.option("--resume", "resume", default=None, metavar="RUN_ID",
              help="Resume an interrupted run by run-id.")
@click.option("--task-id", "task_id", default=None, metavar="ID",
              help="Client-supplied task id (idempotency key).")
def run(task: str | None, offline: bool, resume: str | None, task_id: str | None) -> None:
    stub("run")
