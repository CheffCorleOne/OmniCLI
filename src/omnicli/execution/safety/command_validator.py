"""Safety checks for generated command plans.

Initial implementation is intentionally conservative and mostly structural.
"""

from __future__ import annotations

from typing import Iterable, List

from ...core.context.state_snapshot import SystemSnapshot


FORBIDDEN_SUBSTRINGS = [
    "rm -rf /",
    ":(){",
]


def validate_plan(plan: Iterable[str], snapshot: SystemSnapshot) -> bool:
    """Return True if the plan passes basic safety checks."""
    for cmd in plan:
        if any(bad in cmd for bad in FORBIDDEN_SUBSTRINGS):
            return False
    return True



