from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    role: str
    default: bool
    gated_possible: bool = False
    experimental: bool = False


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "Tongyi-MAI/Z-Image-Turbo": ModelSpec(
        model_id="Tongyi-MAI/Z-Image-Turbo",
        role="Z-Image Turbo base model",
        default=True,
    ),
    "Qwen/Qwen3-VL-4B-Instruct": ModelSpec(
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        role="VLM context encoder for Z-Image D-OPSD",
        default=True,
    ),
    "black-forest-labs/FLUX.2-klein-4B": ModelSpec(
        model_id="black-forest-labs/FLUX.2-klein-4B",
        role="FLUX2 Klein 4B base model",
        default=True,
        gated_possible=True,
    ),
    "black-forest-labs/FLUX.2-klein-9B": ModelSpec(
        model_id="black-forest-labs/FLUX.2-klein-9B",
        role="FLUX2 Klein 9B experimental base model",
        default=False,
        gated_possible=True,
        experimental=True,
    ),
}


def huggingface_cache_root() -> Path:
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"]).expanduser().resolve()
    if os.environ.get("HF_HOME"):
        return (Path(os.environ["HF_HOME"]).expanduser() / "hub").resolve()
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def _model_cache_dir(model_id: str, cache_root: Path) -> Path:
    return cache_root / f"models--{model_id.replace('/', '--')}"


def primary_snapshot_path(model_id: str, cache_root: str | Path | None = None) -> Path | None:
    root = Path(cache_root).expanduser().resolve() if cache_root is not None else huggingface_cache_root()
    model_dir = _model_cache_dir(model_id, root)
    snapshots_dir = model_dir / "snapshots"
    refs_main = model_dir / "refs" / "main"

    if refs_main.exists():
        ref = refs_main.read_text(encoding="utf-8").strip()
        ref_path = snapshots_dir / ref
        if ref_path.is_dir():
            return ref_path.resolve()

    snapshot_paths = sorted([path for path in snapshots_dir.glob("*") if path.is_dir()]) if snapshots_dir.exists() else []
    return snapshot_paths[-1].resolve() if snapshot_paths else None


def model_path_for_command(model_id: str, cache_root: str | Path | None = None) -> str:
    snapshot_path = primary_snapshot_path(model_id, cache_root)
    return str(snapshot_path) if snapshot_path else model_id


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def inspect_model_cache(model_id: str, cache_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(cache_root).expanduser().resolve() if cache_root is not None else huggingface_cache_root()
    model_dir = _model_cache_dir(model_id, root)
    snapshots_dir = model_dir / "snapshots"
    snapshot_paths = sorted([path for path in snapshots_dir.glob("*") if path.is_dir()]) if snapshots_dir.exists() else []
    primary_snapshot = primary_snapshot_path(model_id, root)
    spec = MODEL_REGISTRY.get(model_id)
    return {
        "model_id": model_id,
        "registered": spec is not None,
        "spec": asdict(spec) if spec else None,
        "cache_root": str(root),
        "cache_dir": str(model_dir),
        "cached": bool(snapshot_paths),
        "snapshots": [str(path) for path in snapshot_paths],
        "snapshot_count": len(snapshot_paths),
        "size_bytes": _directory_size(model_dir),
        "primary_snapshot": str(primary_snapshot) if primary_snapshot else None,
    }


def download_model(model_id: str, cache_root: str | Path | None = None) -> dict[str, Any]:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        return {
            "ok": False,
            "model_id": model_id,
            "error": f"huggingface_hub is not installed: {exc}",
        }

    root = Path(cache_root).expanduser().resolve() if cache_root is not None else huggingface_cache_root()
    try:
        path = snapshot_download(repo_id=model_id, cache_dir=str(root), resume_download=True)
        return {"ok": True, "model_id": model_id, "path": path, "cache_root": str(root)}
    except Exception as exc:
        return {"ok": False, "model_id": model_id, "error": str(exc), "cache_root": str(root)}


def registry_json() -> str:
    return json.dumps({key: asdict(value) for key, value in MODEL_REGISTRY.items()}, indent=2)
