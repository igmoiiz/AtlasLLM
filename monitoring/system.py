"""Device and VRAM introspection: one source of truth for hardware reporting.

Training logs, the benchmark script, and reproducibility metadata all need the
same facts about the machine. They come from here, not from ad-hoc
``torch.cuda`` calls scattered across modules (AGENTS.md rule 6).

VRAM figures are MiB (2**20 bytes) read from PyTorch's caching allocator, so
they describe this process, not the whole GPU:

    allocated   bytes held by live tensors
    reserved    bytes held by the allocator cache (>= allocated)
    peak        high-water mark of allocated since the last reset
    total       physical VRAM of the device

On CPU there is no device memory and PyTorch has nothing to report, so
:func:`vram_usage` returns ``None``. Callers decide what to record; the trainer
keeps its historical ``0.0``.
"""

import platform
from dataclasses import asdict, dataclass

import torch

MB = 2**20


@dataclass(frozen=True)
class DeviceInfo:
    """Stable description of the compute device, suitable for JSON."""

    type: str
    name: str
    total_memory_mb: float
    cuda_version: str | None
    capability: tuple[int, int] | None
    torch_version: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class VramUsage:
    """Point-in-time allocator snapshot for one CUDA device."""

    allocated_mb: float
    reserved_mb: float
    peak_mb: float
    total_mb: float

    def to_dict(self) -> dict:
        return asdict(self)


def device_info(device: torch.device) -> DeviceInfo:
    """Describe ``device`` without initializing CUDA if it is not already up.

    CPU fields that do not apply (``total_memory_mb``, ``capability``) are 0.0
    and ``None`` rather than invented values.
    """
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return DeviceInfo(
            type="cuda",
            name=props.name,
            total_memory_mb=round(props.total_memory / MB, 1),
            cuda_version=torch.version.cuda,
            capability=(props.major, props.minor),
            torch_version=torch.__version__,
        )
    return DeviceInfo(
        type=device.type,
        name=platform.processor() or platform.machine() or device.type,
        total_memory_mb=0.0,
        cuda_version=None,
        capability=None,
        torch_version=torch.__version__,
    )


def vram_usage(device: torch.device) -> VramUsage | None:
    """Allocator snapshot for a CUDA device, or ``None`` when not on CUDA.

    Reading these counters does not allocate or synchronize, so it is cheap
    enough to call on every logging step.
    """
    if device.type != "cuda":
        return None
    return VramUsage(
        allocated_mb=round(torch.cuda.memory_allocated(device) / MB, 1),
        reserved_mb=round(torch.cuda.memory_reserved(device) / MB, 1),
        peak_mb=round(torch.cuda.max_memory_allocated(device) / MB, 1),
        total_mb=round(torch.cuda.get_device_properties(device).total_memory / MB, 1),
    )
