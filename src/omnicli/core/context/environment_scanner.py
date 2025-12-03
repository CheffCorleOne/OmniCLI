"""Environment and hardware discovery utilities.

These are intentionally lightweight and dependency-free where possible so that
OmniCLI can start quickly and then refine the profile asynchronously later.
"""

from __future__ import annotations

from typing import Optional

import psutil
from pydantic import BaseModel

try:  # Optional: GPUtil might be missing in some environments
    import GPUtil  # type: ignore[import]
except Exception:  # pragma: no cover - best-effort import
    GPUtil = None  # type: ignore[assignment]


class GPUInfo(BaseModel):
    name: str | None = None
    total_memory_gb: float | None = None
    vendor: str | None = None


class HardwareProfile(BaseModel):
    cpu_count: int
    total_memory_gb: float
    has_gpu: bool = False
    gpu: Optional[GPUInfo] = None
    # Placeholders for richer feature detection (AVX, tensor cores, etc.)
    cpu_features: list[str] = []
    accelerators: list[str] = []


def _detect_gpu() -> Optional[GPUInfo]:
    """Detect a primary GPU using GPUtil if available.

    Falls back to None if GPUtil is not installed or no GPU is found.
    """
    if GPUtil is None:
        return None

    try:
        gpus = GPUtil.getGPUs()
    except Exception:
        return None

    if not gpus:
        return None

    gpu0 = gpus[0]
    # GPUtil reports memory in MB
    total_gb = round(getattr(gpu0, "memoryTotal", 0.0) / 1024.0, 2)
    return GPUInfo(
        name=getattr(gpu0, "name", None),
        total_memory_gb=total_gb or None,
        vendor=None,  # Could be parsed from name later
    )


def discover_hardware() -> HardwareProfile:
    """Best-effort, fast hardware profile used for initial routing decisions."""
    vm = psutil.virtual_memory()
    gpu = _detect_gpu()
    return HardwareProfile(
        cpu_count=psutil.cpu_count(logical=True) or 1,
        total_memory_gb=round(vm.total / (1024**3), 2),
        has_gpu=gpu is not None,
        gpu=gpu,
        cpu_features=[],
        accelerators=[],
    )



