from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional before full dependency install
    Image = None


IMAGE_KEYS = ("local_path_list", "local_paths", "image_path_list", "image_paths", "image_path", "path")
PROMPT_KEYS = (
    "user_prompt_en",
    "short_en",
    "medium_en",
    "detailed_en",
    "user_prompt_zh",
    "short_zh",
    "medium_zh",
    "detailed_zh",
)

EDIT_RECIPE_IDS = {"flux2-klein-editing", "flux2-klein-edit-self-distill-gt-ref"}
RECIPE_WORK_DIRS = {
    "z-image-turbo-vlm": "z-image-turbo_self-distill-vlm",
    "flux2-klein-identity": "flux2-klein_self-distill-edit",
    "flux2-klein-editing": "flux2-klein-edit-self-distill-gt-ref",
    "flux2-klein-edit-self-distill-gt-ref": "flux2-klein-edit-self-distill-gt-ref",
}


@dataclass
class DatasetIssue:
    row: int
    severity: str
    message: str


@dataclass
class DatasetPreview:
    row: int
    class_name: str | None
    prompt_key: str | None
    prompt: str
    image_paths: list[str]
    image_sizes: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class BucketSummary:
    width: int
    height: int
    count: int


@dataclass
class DatasetSummary:
    path: str
    recipe_id: str
    rows: int
    valid_rows: int
    issues: list[DatasetIssue]
    previews: list[DatasetPreview]
    bucket_summary: list[BucketSummary] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"row {index}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"row {index}: expected JSON object")
            rows.append(value)
    return rows


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _extract_image_paths(item: dict[str, Any]) -> list[str]:
    for key in IMAGE_KEYS:
        if key in item:
            values = _as_list(item[key])
            return [str(value) for value in values if str(value).strip()]
    return []


def _resolve_path(raw_path: str, jsonl_path: Path, project_root: Path, recipe_root: Path | None) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    bases = [jsonl_path.parent]
    if recipe_root is not None:
        bases.append(recipe_root)
    bases.append(project_root)
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (jsonl_path.parent / path).resolve()


def _image_size(path: Path) -> tuple[int, int] | None:
    if Image is None:
        return None
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def _parse_declared_size(value: Any) -> tuple[int, int] | None:
    text = str(value).lower().replace(" ", "")
    if "*" not in text:
        return None
    height_text, width_text = text.split("*", 1)
    try:
        height = int(float(height_text))
        width = int(float(width_text))
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _declared_sizes(item: dict[str, Any]) -> list[tuple[int, int]]:
    raw = item.get("h*w") or item.get("height*width")
    if raw is None:
        return []
    sizes = []
    for value in _as_list(raw):
        size = _parse_declared_size(value)
        if size is not None:
            sizes.append(size)
    return sizes


def _pick_prompt(item: dict[str, Any]) -> tuple[str | None, str]:
    for key in PROMPT_KEYS:
        value = item.get(key)
        if value is not None and str(value).strip():
            return key, str(value)
    return None, ""


def validate_dataset(jsonl_path: str | Path, recipe_id: str, project_root: str | Path | None = None) -> DatasetSummary:
    root = Path(project_root or Path.cwd()).resolve()
    path = Path(jsonl_path).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()

    issues: list[DatasetIssue] = []
    previews: list[DatasetPreview] = []

    if not path.exists():
        return DatasetSummary(
            path=str(path),
            recipe_id=recipe_id,
            rows=0,
            valid_rows=0,
            issues=[DatasetIssue(row=0, severity="error", message="JSONL file does not exist")],
            previews=[],
        )

    try:
        rows = _read_jsonl(path)
    except ValueError as exc:
        return DatasetSummary(
            path=str(path),
            recipe_id=recipe_id,
            rows=0,
            valid_rows=0,
            issues=[DatasetIssue(row=0, severity="error", message=str(exc))],
            previews=[],
        )

    min_images = 2 if recipe_id in EDIT_RECIPE_IDS else 1
    recipe_work_dir = RECIPE_WORK_DIRS.get(recipe_id)
    recipe_root = (root / recipe_work_dir).resolve() if recipe_work_dir else None
    valid_rows = 0
    bucket_counts: dict[tuple[int, int], int] = {}

    for index, item in enumerate(rows, start=1):
        prompt_key, prompt = _pick_prompt(item)
        if prompt_key is None:
            issues.append(DatasetIssue(row=index, severity="error", message="missing prompt field"))

        raw_image_paths = _extract_image_paths(item)
        if len(raw_image_paths) < min_images:
            issues.append(
                DatasetIssue(
                    row=index,
                    severity="error",
                    message=f"expected at least {min_images} image path(s), found {len(raw_image_paths)}",
                )
            )

        resolved_paths = [_resolve_path(raw_path, path, root, recipe_root) for raw_path in raw_image_paths]
        for resolved in resolved_paths:
            if not resolved.exists():
                issues.append(DatasetIssue(row=index, severity="error", message=f"image not found: {resolved}"))

        sizes = []
        for resolved in resolved_paths:
            size = _image_size(resolved)
            if size is not None:
                sizes.append(size)
        row_sizes = sizes or _declared_sizes(item)
        for size in row_sizes:
            bucket_counts[size] = bucket_counts.get(size, 0) + 1

        has_row_error = any(issue.row == index and issue.severity == "error" for issue in issues)
        if not has_row_error:
            valid_rows += 1

        if len(previews) < 8:
            previews.append(
                DatasetPreview(
                    row=index,
                    class_name=item.get("class_name"),
                    prompt_key=prompt_key,
                    prompt=prompt[:240],
                    image_paths=[str(path) for path in resolved_paths],
                    image_sizes=sizes,
                )
            )

    return DatasetSummary(
        path=str(path),
        recipe_id=recipe_id,
        rows=len(rows),
        valid_rows=valid_rows,
        issues=issues,
        previews=previews,
        bucket_summary=[
            BucketSummary(width=width, height=height, count=count)
            for (width, height), count in sorted(bucket_counts.items(), key=lambda item: (-item[1], item[0][1], item[0][0]))
        ],
    )
