"""Hardware introspection: device facts and VRAM usage.

    device_info(device) -> DeviceInfo    name, total memory, capability, versions
    vram_usage(device)  -> VramUsage|None  allocated / reserved / peak / total MB

One implementation for the trainer, the benchmark script, and reproducibility
metadata (AGENTS.md rule 6).
"""

from monitoring.system import MB, DeviceInfo, VramUsage, device_info, vram_usage

__all__ = ["MB", "DeviceInfo", "VramUsage", "device_info", "vram_usage"]
