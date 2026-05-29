from __future__ import annotations

import click

from farm_edge_agent.cli.commands import config, serve, version


@click.group(
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="FARM Edge Agent CLI — UF850 sim + ROS-TCP teleop bridge.",
)
@click.option("--config", "config_path", type=click.Path(), default=None,
              help="Path to config file (overrides ~/.farm/config.yaml).")
@click.option("--json", "json_output", is_flag=True, default=False,
              help="Emit machine-readable JSON instead of human text.")
@click.option("--quiet", is_flag=True, default=False, help="Suppress non-essential output.")
@click.pass_context
def cli(
    ctx: click.Context,
    config_path: str | None,
    json_output: bool,
    quiet: bool,
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    ctx.obj["json"] = json_output
    ctx.obj["quiet"] = quiet

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


cli.add_command(config.config)
cli.add_command(version.version)
cli.add_command(serve.serve)


def main() -> None:
    cli.main(prog_name="farm")


if __name__ == "__main__":
    main()
