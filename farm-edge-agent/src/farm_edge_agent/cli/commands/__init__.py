from __future__ import annotations

import click


def stub(name: str) -> None:
    click.echo(f"[FARM] not implemented yet: {name}")
