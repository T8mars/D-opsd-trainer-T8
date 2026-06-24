from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass
class LossPoint:
    step: int
    epoch: int | None
    loss_dopsd: float | None
    loss_total: float | None
    grad_norm: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactEntry:
    name: str
    relative_path: str
    size_bytes: int
    modified_at: str | None
    is_image: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunSummary:
    output_dir: str
    status: str
    has_args: bool
    has_log: bool
    has_loss_log: bool
    failure_reason: str | None
    error_tail: list[str]
    latest_step: int
    latest_loss: float | None
    trainable_params: int | None
    log_tail: list[str]
    loss_points: list[LossPoint]
    samples: list[ArtifactEntry]
    sample_trajectories: list[ArtifactEntry]
    checkpoints: list[ArtifactEntry]
    sample_count: int
    sample_trajectory_count: int
    checkpoint_count: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["loss_points"] = [point.to_dict() for point in self.loss_points]
        payload["samples"] = [item.to_dict() for item in self.samples]
        payload["sample_trajectories"] = [item.to_dict() for item in self.sample_trajectories]
        payload["checkpoints"] = [item.to_dict() for item in self.checkpoints]
        return payload


def _clean_log_line(line: str) -> str:
    return ANSI_RE.sub("", line).strip()


def parse_loss_jsonl(path: str | Path) -> list[LossPoint]:
    loss_path = Path(path)
    if not loss_path.exists():
        return []

    points: list[LossPoint] = []
    for raw_line in loss_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        points.append(
            LossPoint(
                step=int(payload.get("glo_s", payload.get("step", 0)) or 0),
                epoch=payload.get("epoch"),
                loss_dopsd=_maybe_float(payload.get("loss_dopsd")),
                loss_total=_maybe_float(payload.get("loss_total")),
                grad_norm=_maybe_float(payload.get("grad_n")),
            )
        )
    return points


def summarize_run(output_dir: str | Path) -> RunSummary:
    run_dir = Path(output_dir)
    args_path = run_dir / "args.json"
    log_path = run_dir / "log.txt"
    loss_path = run_dir / "loss_log" / "loss_gen_log.jsonl"

    loss_points = parse_loss_jsonl(loss_path)
    log_lines: list[str] = []
    if log_path.exists():
        log_lines = [_clean_log_line(line) for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()]
        log_lines = [line for line in log_lines if line]

    external_error_lines = _read_external_error_lines(run_dir)
    failure_source_lines = [*log_lines, *external_error_lines]
    failure_reason = _extract_failure_reason(failure_source_lines)
    error_tail = _select_error_tail(failure_source_lines) if failure_reason else []
    latest = loss_points[-1] if loss_points else None
    status = _infer_status(log_lines, loss_points, failure_reason)
    samples = _list_artifacts(run_dir, "samples")
    sample_trajectories = _list_artifacts(run_dir, "samples_trajectory")
    checkpoints = _list_artifacts(run_dir, "checkpoints")

    return RunSummary(
        output_dir=str(run_dir),
        status=status,
        has_args=args_path.exists(),
        has_log=log_path.exists(),
        has_loss_log=loss_path.exists(),
        failure_reason=failure_reason,
        error_tail=error_tail,
        latest_step=latest.step if latest else 0,
        latest_loss=latest.loss_total if latest else None,
        trainable_params=_parse_trainable_params(log_lines),
        log_tail=log_lines[-12:],
        loss_points=loss_points,
        samples=samples,
        sample_trajectories=sample_trajectories,
        checkpoints=checkpoints,
        sample_count=len(samples),
        sample_trajectory_count=len(sample_trajectories),
        checkpoint_count=len(checkpoints),
    )


def _infer_status(log_lines: list[str], loss_points: list[LossPoint], failure_reason: str | None = None) -> str:
    joined = "\n".join(log_lines).lower()
    if failure_reason:
        return "failed"
    if "training completed" in joined:
        return "completed"
    if "traceback" in joined or "outofmemory" in joined or "out of memory" in joined:
        return "failed"
    if loss_points:
        return "running"
    if log_lines:
        return "initialized"
    return "missing"


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_trainable_params(log_lines: list[str]) -> int | None:
    for line in log_lines:
        match = re.search(r"Total trainable parameters in gen_model:\s*(\d+)", line)
        if match:
            return int(match.group(1))
    return None


def _read_external_error_lines(run_dir: Path) -> list[str]:
    paths = [
        run_dir.parent / f"{run_dir.name}.runner.err.log",
        run_dir / "runner.err.log",
    ]
    lines: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        lines.extend(line for line in (_clean_log_line(raw_line) for raw_line in raw_lines) if line)
    return lines


def _extract_failure_reason(lines: list[str]) -> str | None:
    for line in reversed(lines):
        if _is_progress_line(line):
            continue
        if _is_failure_line(line):
            return line[-320:]
    return None


def _select_error_tail(lines: list[str], limit: int = 12) -> list[str]:
    filtered = [line for line in lines if line and not _is_progress_line(line)]
    if not filtered:
        return []

    failure_index = next((index for index in range(len(filtered) - 1, -1, -1) if _is_failure_line(filtered[index])), None)
    if failure_index is None:
        return filtered[-limit:]
    return filtered[max(0, failure_index - limit + 1) : failure_index + 1]


def _is_failure_line(line: str) -> bool:
    lowered = line.lower()
    return (
        "traceback" in lowered
        or "runtimeerror:" in lowered
        or "outofmemory" in lowered
        or "out of memory" in lowered
        or "cuda driver error" in lowered
        or "torch.cuda" in lowered
    )


def _is_progress_line(line: str) -> bool:
    return line.lstrip().startswith("Steps:")


def _list_artifacts(run_dir: Path, dirname: str, limit: int = 200) -> list[ArtifactEntry]:
    artifact_dir = run_dir / dirname
    if not artifact_dir.is_dir():
        return []

    entries: list[ArtifactEntry] = []
    for file_path in sorted(path for path in artifact_dir.rglob("*") if path.is_file()):
        try:
            stat = file_path.stat()
        except OSError:
            continue
        relative = file_path.relative_to(run_dir).as_posix()
        entries.append(
            ArtifactEntry(
                name=file_path.name,
                relative_path=relative,
                size_bytes=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                is_image=file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"},
            )
        )

    entries.sort(key=lambda item: (item.modified_at or "", item.relative_path), reverse=True)
    return entries[:limit]
