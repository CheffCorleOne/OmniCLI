"""
Application composition root for OmniCLI.

This ties together:
- hardware & system discovery
- context snapshotting
- intent understanding
- safe plan generation

Early versions operate in dry-run mode only and never execute real commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .core.context.state_snapshot import SystemSnapshot, capture_system_snapshot
from .core.context.environment_scanner import HardwareProfile, discover_hardware
from .core.orchestrator.workflow_engine import WorkflowEngine
from .nlp.understanding.intent_classifier import classify_intent
from .nlp.generation.command_generator import generate_command_plan
from .execution.safety.command_validator import validate_plan
from .utils.logging_manager import get_logger


@dataclass
class OmniCLIConfig:
    dry_run: bool = True


@dataclass
class OmniCLIResult:
    intent: str
    classified_intent: str
    system_snapshot: SystemSnapshot
    hardware_profile: HardwareProfile
    plan: List[str]
    safe: bool


class OmniCLIApp:
    """Facade for the main OmniCLI interaction."""

    def __init__(self, config: OmniCLIConfig | None = None) -> None:
        self.config = config or OmniCLIConfig()
        self._log = get_logger("app")

        self._log.info("[bold cyan]Stage:[/bold cyan] discover_hardware")
        self.hardware_profile = discover_hardware()
        self._log.debug(f"Hardware profile: {self.hardware_profile.model_dump()}")

        self._log.info("[bold cyan]Stage:[/bold cyan] capture_system_snapshot")
        self.system_snapshot = capture_system_snapshot()
        self._log.debug(f"System snapshot users={self.system_snapshot.users}")

        self._log.info("[bold cyan]Stage:[/bold cyan] init_workflow_engine")
        self.workflow_engine = WorkflowEngine(
            hardware_profile=self.hardware_profile,
            system_snapshot=self.system_snapshot,
        )

    def handle_intent(self, user_intent: str) -> OmniCLIResult:
        """Process a natural-language intent and return a safe execution plan."""
        self._log.info("[bold magenta]Intent:[/bold magenta] %s", user_intent)

        self._log.info("[bold cyan]Stage:[/bold cyan] classify_intent")
        classified = classify_intent(user_intent)
        self._log.debug("Classified intent: %s", classified)

        self._log.info("[bold cyan]Stage:[/bold cyan] generate_plan")
        plan = generate_command_plan(
            user_intent=user_intent,
            classified_intent=classified,
            snapshot=self.system_snapshot,
            hardware=self.hardware_profile,
        )
        self._log.debug("Generated plan: %s", plan)

        self._log.info("[bold cyan]Stage:[/bold cyan] validate_plan")
        safe = validate_plan(plan, snapshot=self.system_snapshot)
        self._log.info("[bold green]Plan safety:[/bold green] %s", safe)

        # In early versions we never actually execute the plan; execution engine will be
        # wired here later (with sandbox + rollback).
        if not self.config.dry_run and safe:
            self._log.info("[bold cyan]Stage:[/bold cyan] execute_plan (experimental)")
            self.workflow_engine.execute_plan(plan)
        else:
            self._log.info("[yellow]Dry-run:[/yellow] plan will not be executed")

        return OmniCLIResult(
            intent=user_intent,
            classified_intent=classified,
            system_snapshot=self.system_snapshot,
            hardware_profile=self.hardware_profile,
            plan=plan,
            safe=safe,
        )


def create_app(config: OmniCLIConfig | None = None) -> OmniCLIApp:
    """Factory used by the CLI entry point."""
    return OmniCLIApp(config=config)



