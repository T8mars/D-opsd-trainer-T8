from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .profiles import list_production_profiles


def _project_path(project_root: Path, *parts: str) -> Path:
    return (project_root / Path(*parts)).resolve()


def _path_item(identifier: str, label: str, path: Path, purpose: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "label": label,
        "path": str(path).replace("\\", "/"),
        "exists": path.exists(),
        "purpose": purpose,
    }


def _env_present(env: Mapping[str, str], *names: str) -> bool:
    return any(bool(env.get(name)) for name in names)


def build_settings_summary(project_root: str | Path, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    env_values = os.environ if env is None else env

    hf_home = _project_path(root, "trainer-data", "hf-home")
    jobs_root = _project_path(root, "trainer-data", "jobs")

    return {
        "project_root": str(root).replace("\\", "/"),
        "backend": {
            "host": "Windows",
            "preferred": "WSL2 Ubuntu",
            "distro": "Ubuntu-22.04",
            "venv": str(_project_path(root, "trainer-data", "venvs", "dopsd")).replace("\\", "/"),
            "env_script": str(_project_path(root, "scripts", "dopsd_wsl_env.sh")).replace("\\", "/"),
        },
        "hf_token": {
            "present": _env_present(env_values, "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"),
            "source": "HF_TOKEN or HUGGINGFACE_HUB_TOKEN",
            "display": "status-only",
        },
        "paths": [
            _path_item("project_root", "Project root", root, "Workspace and UI server root"),
            _path_item("wsl_venv", "WSL trainer venv", _project_path(root, "trainer-data", "venvs", "dopsd"), "Python training environment"),
            _path_item("hf_home", "Hugging Face home", hf_home, "Managed model cache and auth home"),
            _path_item("hf_cache", "Hugging Face hub cache", hf_home / "hub", "Downloaded model snapshots"),
            _path_item("jobs_ledger", "Jobs ledger", jobs_root / "jobs.json", "Durable UI/API job state"),
            _path_item("runner_root", "Runner scripts", jobs_root / "runner", "Per-job runner.sh, state, logs, and PID files"),
            _path_item("smoke_runs", "Smoke runs", _project_path(root, "trainer-data", "smoke-runs"), "Verified smoke outputs and future short runs"),
        ],
        "safety_defaults": [
            {"id": "launcher", "label": "Launcher", "value": "python", "reason": "Verified local path for 16GB profiles"},
            {"id": "low_vram", "label": "Low VRAM", "value": True, "reason": "Offloads frozen conditioner components between stages"},
            {"id": "block_offload", "label": "Block offload", "value": False, "reason": "Keep disabled for verified starter profiles; group block offload is still experimental"},
            {"id": "resolution_scale", "label": "Resolution scale", "value": "profile-specific", "reason": "Use the verified per-recipe 16GB profile instead of one global value"},
            {"id": "use_8bit_adam", "label": "8-bit Adam", "value": True, "reason": "Reduces optimizer memory for LoRA runs"},
            {"id": "save_samples", "label": "Save samples", "value": True, "reason": "Enabled in verified starter profiles with low-VRAM sample scaling where needed"},
            {"id": "save_checkpoints", "label": "Save checkpoints", "value": True, "reason": "Enabled in verified starter profiles"},
        ],
        "production_profiles": list_production_profiles(),
        "runner_policy": {
            "backend": "WSL detached runner",
            "max_active_jobs": 1,
            "queue_order": "fifo",
            "delete_running_jobs": False,
            "auto_promote_queued": True,
        },
    }
