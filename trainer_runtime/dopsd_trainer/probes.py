from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(command: list[str], timeout: int = 8) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
        )
        return {
            "ok": completed.returncode == 0,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
        }
    except Exception as exc:  # pragma: no cover - platform dependent
        return {"ok": False, "stdout": "", "stderr": str(exc), "returncode": -1}


def probe_gpu() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "gpus": [], "error": "nvidia-smi not found"}

    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if not result["ok"]:
        return {"available": False, "gpus": [], "error": result["stderr"]}

    gpus = []
    for line in result["stdout"].strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        index, name, driver, total, free = parts
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "driver": driver,
                "memory_total_mb": int(float(total)),
                "memory_free_mb": int(float(free)),
            }
        )
    return {"available": bool(gpus), "gpus": gpus}


def probe_wsl() -> dict[str, Any]:
    if platform.system().lower() != "windows":
        return {"available": False, "distributions": [], "error": "not running on Windows host"}
    result = _run(["wsl.exe", "-l", "-v"])
    clean = result["stdout"].replace("\x00", "")
    distributions = []
    for line in clean.splitlines():
        line = line.replace("*", "").strip()
        if not line or line.startswith("NAME"):
            continue
        distributions.append(line.split()[0])
    return {
        "available": result["ok"] and bool(distributions),
        "distributions": distributions,
        "error": None if result["ok"] else result["stderr"],
    }


def probe_conda() -> dict[str, Any]:
    executable = shutil.which("conda")
    return {"available": executable is not None, "path": executable}


def probe_disk(project_root: str | Path) -> dict[str, Any]:
    usage = shutil.disk_usage(project_root)
    return {
        "available": True,
        "total_gb": round(usage.total / 1024**3, 2),
        "free_gb": round(usage.free / 1024**3, 2),
    }


def probe_environment(project_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root or Path.cwd()).resolve()
    return {
        "project_root": str(root),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": sys.version.split()[0],
        },
        "gpu": probe_gpu(),
        "wsl": probe_wsl(),
        "conda": probe_conda(),
        "disk": probe_disk(root),
        "hf_token": {
            "present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")),
        },
    }


def main() -> None:
    print(json.dumps(probe_environment(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
