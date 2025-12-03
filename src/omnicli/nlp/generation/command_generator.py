"""Rule-based command plan generator stub.

Eventually this will call into the AI engine; for now we provide simple,
transparent rules so the CLI is demonstrably useful from day one.
"""

from __future__ import annotations

from typing import List

from ...core.context.environment_scanner import HardwareProfile
from ...core.context.state_snapshot import SystemSnapshot


def generate_command_plan(
    user_intent: str,
    classified_intent: str,
    snapshot: SystemSnapshot,
    hardware: HardwareProfile,
) -> List[str]:
    """Produce a shell command plan for a given intent.

    NOTE: This is a placeholder – real planning will use models + retrieval.
    """
    if classified_intent == "backup":
        return [
            "tar -czf ~/backup_$(date +%Y%m%d).tar.gz ~/projects",
            "# TODO: add rsync/scp step based on detected servers",
        ]
    if classified_intent == "deploy":
        return [
            "sudo systemctl restart nginx",
            "# TODO: generate docker-compose or k8s manifests based on context",
        ]
    if classified_intent == "setup_monitoring":
        return [
            "# TODO: detect distro and choose Prometheus/node_exporter setup",
        ]
    if classified_intent == "diagnose":
        return [
            "top -b -n 1 | head -n 20",
            "df -h",
            "free -m",
        ]
    # Fallback: simply echo what the user asked for as a reminder.
    return [f"# Plan for: {user_intent}"]



