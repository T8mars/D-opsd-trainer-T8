from __future__ import annotations

from typing import Any


def clamp_offload_percent(value: float | int | str | None) -> float:
    if value is None:
        return 0.0
    try:
        percent = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, percent))


def attach_layer_offload(
    module: Any,
    device: Any,
    offload_percent: float | int | str | None,
    logger: Any | None = None,
    *,
    label: str = "module",
    ignore_modules: list[Any] | None = None,
) -> bool:
    percent = clamp_offload_percent(offload_percent)
    if percent <= 0:
        return False
    if getattr(module, "_dopsd_layer_offload_enabled", False):
        return True

    from .memory_management import MemoryManager

    MemoryManager.attach(
        module,
        device,
        offload_percent=percent,
        ignore_modules=ignore_modules or [],
    )
    module._dopsd_layer_offload_enabled = True
    module._dopsd_layer_offload_percent = percent
    if logger is not None:
        logger.info(
            f"AI Toolkit-style layer offloading enabled for {label}: "
            f"{percent:.0%} of eligible Linear/Conv layers"
        )
    return True

