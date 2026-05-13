from __future__ import annotations

import click

from farm_edge_agent.cli.commands import (
    calibrate,
    card,
    config,
    doctor,
    export,
    login,
    quickstart,
    run,
    start,
    verify,
    version,
)


@click.group(
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="FARM Edge Agent CLI.",
)
@click.option("--config", "config_path", type=click.Path(), default=None,
              help="Path to config file (overrides ~/.farm/config.yaml).")
@click.option("--workspace", default=None, help="Workspace name to use for this invocation.")
@click.option("--json", "json_output", is_flag=True, default=False,
              help="Emit machine-readable JSON instead of human text.")
@click.option("--quiet", is_flag=True, default=False, help="Suppress non-essential output.")
@click.option("--auto-update", is_flag=True, default=False,
              help="Pip-upgrade the Edge Agent on protocol mismatch.")
@click.option("--accept-calibration", is_flag=True, default=False,
              help="Run even if calibration is older than 24h.")
@click.pass_context
def cli(
    ctx: click.Context,
    config_path: str | None,
    workspace: str | None,
    json_output: bool,
    quiet: bool,
    auto_update: bool,
    accept_calibration: bool,
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    ctx.obj["workspace"] = workspace
    ctx.obj["json"] = json_output
    ctx.obj["quiet"] = quiet
    ctx.obj["auto_update"] = auto_update
    ctx.obj["accept_calibration"] = accept_calibration

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


cli.add_command(quickstart.quickstart)
cli.add_command(login.login)
cli.add_command(config.config)
cli.add_command(start.start)
cli.add_command(run.run)
cli.add_command(export.export)
cli.add_command(calibrate.calibrate)
cli.add_command(card.card)
cli.add_command(doctor.doctor)
cli.add_command(verify.verify)
cli.add_command(version.version)


def main() -> None:
    cli.main(prog_name="farm")


if __name__ == "__main__":
    main()
