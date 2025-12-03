"""System snapshot collection.

Captures a coarse but useful view of the current system state that can be fed
into intent understanding and command planning.
"""

from __future__ import annotations

from typing import Dict, List

import os
from pydantic import BaseModel


class SystemSnapshot(BaseModel):
    users: List[str]
    env_vars: Dict[str, str]
    mounts: List[str]
    # TODO: extend with processes, services, containers, etc.


def _list_users() -> List[str]:
    # Very coarse for now; proper implementation will vary by platform.
    try:
        return sorted({os.getlogin()})
    except Exception:
        return []


def _get_mounts() -> List[str]:
    # Placeholder: on Linux we would parse /proc/mounts or use psutil.disk_partitions.
    return []


def capture_system_snapshot() -> SystemSnapshot:
    """Collect a lightweight snapshot of the system state."""
    return SystemSnapshot(
        users=_list_users(),
        env_vars=dict(os.environ),
        mounts=_get_mounts(),
    )



