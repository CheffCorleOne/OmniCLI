"""
CLI entry point for OmniCLI.

Uses Typer to expose a simple interface:
- `omnici run "<natural language intent>"`
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from ..app import OmniCLIConfig, create_app
from ..utils.logging_manager import configure_logging

app = typer.Typer(help="OmniCLI – offline AI assistant for your terminal.")
console = Console()


@app.command()
def run(
    intent: str = typer.Argument(..., help="Natural language description of what you want."),
    execute: bool = typer.Option(
        False,
        "--execute",
        "-x",
        help="Actually execute the generated plan (EXPERIMENTAL; default is dry-run).",
    ),
    log_level: str = typer.Option(
        "info",
        "--log-level",
        help="Logging verbosity: debug, info, warning, error.",
    ),
) -> None:
    """Process a natural-language intent and produce a safe execution plan."""
    # Configure logging once per CLI invocation.
    level_map = {
        "debug": 10,
        "info": 20,
        "warning": 30,
        "error": 40,
    }
    configure_logging(level=level_map.get(log_level.lower(), 20))

    config = OmniCLIConfig(dry_run=not execute)
    app_instance = create_app(config=config)

    result = app_instance.handle_intent(intent)

    console.print(
        Panel.fit(
            f"[bold cyan]Intent:[/bold cyan] {result.intent}\n"
            f"[bold magenta]Classified:[/bold magenta] {result.classified_intent}\n"
            f"[bold green]Safe:[/bold green] {result.safe}",
            title="OmniCLI",
        )
    )

    console.rule("[bold yellow]Proposed Plan[/bold yellow]")
    if not result.plan:
        console.print("[dim]No concrete commands yet – model stubs are still in development.[/dim]")
    else:
        for i, cmd in enumerate(result.plan, start=1):
            console.print(f"[bold]{i}.[/bold] [white]{cmd}[/white]")

    console.rule("[bold blue]Context Snapshot[/bold blue]")
    console.print(Pretty(result.system_snapshot.model_dump(), expand=False))


def main(argv: Optional[list[str]] = None) -> None:
    """Proxy for `python -m omnicli`."""
    app(standalone_mode=True)


if __name__ == "__main__":  # pragma: no cover
    main()



