"""Normalize recorded trajectories and embed them in an offline HTML viewer.

This is a read-only projection: it never executes tools, replays episodes,
loads an Env, follows artifact paths, or contacts a model provider.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any

MAX_INPUT_BYTES = 128 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
_MIMES = {"image/png", "image/jpeg", "image/gif", "image/webp", "image/bmp"}
_PUBLIC_TASK_KEYS = {"schema_version", "task_id", "env_id", "messages", "judge_ref"}


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _object(value: Any, label: str) -> dict:
    value = _plain(value)
    if isinstance(value, str):
        try:
            value = _plain(json.loads(value))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} must be a JSON object, not a file path") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _is_trajectory(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("messages"), list) and isinstance(value.get("steps"), list)


def _public_task(value: Any, trajectory: dict | None, warnings: list[str]) -> dict | None:
    if value is None:
        return None
    task = _object(value, "task")
    for key in ("env_id", "task_id"):
        if trajectory and task.get(key) and task[key] != trajectory.get(key):
            warnings.append(f"补充 Task 的 {key} 不匹配，未附加。")
            return None
    # Scenario and verifier bindings are not solver-visible task information.
    return {key: item for key, item in task.items() if key in _PUBLIC_TASK_KEYS}


def _judge(row: dict, task: dict | None, prefix: str, score_key: str, has_original: bool) -> dict:
    fields = {
        "overall": row.get(score_key),
        "scores": row.get(prefix + "judge_scores"),
        "model_scores": row.get(prefix + "judge_model_scores"),
        "normalized_scores": row.get(prefix + "judge_normalized_scores"),
        "rationale": row.get(prefix + "rationale"),
    }
    rubric = row.get("judge_ref") or (task or {}).get("judge_ref")
    if isinstance(rubric, str):
        try:
            rubric = _plain(json.loads(rubric))
        except json.JSONDecodeError:
            pass
    return {
        **fields,
        "present": any(value is not None for value in fields.values()),
        "rubric": rubric,
        "score_key": score_key,
        "scope": "row",
        "scope_note": (
            "这是行级 Judge 记录，未声明绑定哪个 episode。当前行包含原始/修订轨迹；"
            "不要将该分数视为修订后的评分。现有 refiner 会保留修订前的分数，除非之后另行 Judge。"
            if has_original else "这是随该记录保存的 Judge 数据；不从 finish 或工具成功推导评分。"
        ),
    }


def _normalize(data: Any, *, task: Any, summary: Any, trajectory_key: str,
               score_key: str, score_prefix: str, task_resolver=None) -> list[dict]:
    data = _plain(data)
    rows = data if isinstance(data, list) else [data]
    if not rows:
        raise ValueError("Input contains no records")
    result = []
    for index, raw in enumerate(rows):
        row = _object(raw, f"record {index + 1}")
        bare = _is_trajectory(row)
        if not bare and trajectory_key not in row:
            raise ValueError(f"record {index + 1}: expected a trajectory or field {trajectory_key!r}")
        warnings: list[str] = []
        variants = []
        for key, label, value in [
            ("trajectory", "当前轨迹", row if bare else row.get(trajectory_key)),
            ("trajectory_original", "原始轨迹", None if bare else row.get("trajectory_original")),
        ]:
            if value is None:
                continue
            try:
                trajectory = _object(value, key)
                if not _is_trajectory(trajectory):
                    raise ValueError("messages/steps must be arrays")
            except ValueError as exc:
                warnings.append(f"{key} 无法展示：{exc}")
                continue
            if trajectory.get("schema_version") != 2:
                warnings.append(f"{key} 未声明 schema_version=2，按已知字段展示。")
            variants.append({"key": key, "label": label, "trajectory": trajectory})
        current = variants[0]["trajectory"] if variants else None
        task_value = row.get("task") if not bare else None
        if task_value is None:
            task_value = task
        if task_value is None and current and task_resolver:
            try:
                task_value = task_resolver.resolve(current["task_id"], env_id=current["env_id"])
            except (KeyError, TypeError, ValueError, OSError) as exc:
                warnings.append(f"未能补充 Task：{type(exc).__name__}")
        public_task = _public_task(task_value, current, warnings)
        details = row.get("summary", summary) if not bare else summary
        if details is not None:
            details = _object(details, "summary")
            if current and any(details.get(key) and details[key] != current.get(key)
                               for key in ("env_id", "task_id", "episode_id")):
                warnings.append("补充 summary 的任务/episode 标识不匹配，未附加。")
                details = None
        metadata = {} if bare else {
            key: value for key, value in row.items()
            if key not in {trajectory_key, "trajectory_original", "task", "summary", "scenario", "verification"}
        }
        result.append({
            "id": f"record-{index + 1}",
            "task_id": (current or row).get("task_id", "未知任务"),
            "env_id": (current or row).get("env_id", "未知 Env"),
            "variants": variants,
            "task": public_task,
            "summary": details,
            "judge": _judge({} if bare else row, public_task, score_prefix, score_key, len(variants) > 1),
            "replay": row.get("replay_verification", (details or {}).get("replay_verification")) if not bare
                      else (details or {}).get("replay_verification"),
            "refined": row.get("_refined") if not bare else None,
            "refine_note": row.get("_refine_note") if not bare else None,
            "metadata": metadata,
            "warnings": warnings,
        })
    return result


def _assets(value: Any, assets: dict) -> Any:
    if isinstance(value, list):
        return [_assets(item, assets) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("type") == "image" and "data" in value:
        mime, encoded = value.get("media_type"), value.get("data")
        try:
            if mime not in _MIMES or not isinstance(encoded, str):
                raise ValueError("只支持内嵌 PNG/JPEG/GIF/WEBP/BMP")
            if len(encoded) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
                raise ValueError("图片超过 32 MiB")
            decoded = base64.b64decode(encoded, validate=True)
            if not decoded:
                raise ValueError("空图片")
            # Validate both the MIME and image signature; never render SVG/HTML.
            from ..contracts import ImageContent, validate_content
            validate_content((ImageContent(mime, encoded),))
            digest = hashlib.sha256(mime.encode() + b"\0" + decoded).hexdigest()
            assets.setdefault(digest, {"media_type": mime, "data": encoded})
            return {"type": "image", "asset_id": digest, "media_type": mime,
                    "detail": value.get("detail", "auto")}
        except (ValueError, TypeError, binascii.Error) as exc:
            return {"type": "unavailable_image", "reason": str(exc)}
    return {key: _assets(item, assets) for key, item in value.items()}


def render_trajectory_html(data: Any, *, task: Any = None, summary: Any = None,
                           title: str = "Trajectory Explorer", trajectory_key: str = "trajectory",
                           score_key: str = "traj_overall", score_prefix: str = "traj_",
                           task_resolver=None) -> str:
    """Render a Trajectory, pipeline row, or sequence as a self-contained HTML.

    Optional task data is limited to public fields. Non-finite pipeline scores
    become null; unbound row-level scores are never attributed to a refinement.
    Recorded messages are not asserted to be exact provider-wire requests.
    """
    records = _normalize(data, task=task, summary=summary, trajectory_key=trajectory_key,
                         score_key=score_key, score_prefix=score_prefix, task_resolver=task_resolver)
    assets: dict = {}
    payload = {"title": str(title), "records": _assets(records, assets), "assets": assets}
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for char, escaped in [("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                          ("\u2028", "\\u2028"), ("\u2029", "\\u2029")]:
        encoded = encoded.replace(char, escaped)
    resources = files(__package__).joinpath("assets")
    script = resources.joinpath("viewer.js").read_text(encoding="utf-8")
    css = resources.joinpath("viewer.css").read_text(encoding="utf-8")
    template = resources.joinpath("viewer.html").read_text(encoding="utf-8")
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    return (template.replace("__SCRIPT_HASH__", digest).replace("__STYLES__", css)
            .replace("__SCRIPT__", script).replace("__DATA__", encoded))


def _read(path: Path) -> Any:
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError(f"Input exceeds 128 MiB: {path.name}")
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".jsonl":
        rows = []
        for number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {number}") from exc
        return rows
    return json.loads(text)


def export_trajectory_html(source: str | Path, output: str | Path, *,
                           task_file: str | Path | None = None,
                           summary_file: str | Path | None = None,
                           tasks_dir: str | Path | None = None,
                           force: bool = False, **options: Any) -> Path:
    """Read JSON/JSONL or a rollout directory and atomically create a report.

    Only trajectory.json's explicit task.json/summary.json siblings are detected.
    Paths inside a trajectory are never followed. Existing output needs force.
    """
    path = Path(source).resolve(strict=True)
    if path.is_dir():
        path = (path / "trajectory.json").resolve(strict=True)
    destination = Path(output).absolute()
    auto_sidecars = path.name == "trajectory.json"
    task_path = Path(task_file) if task_file else (path.with_name("task.json") if auto_sidecars else None)
    summary_path = Path(summary_file) if summary_file else (path.with_name("summary.json") if auto_sidecars else None)
    inputs = [path, *(p.resolve() for p in (task_path, summary_path) if p is not None and p.exists())]
    if destination.is_symlink() or destination.resolve() in inputs:
        raise ValueError("Output may not be a symlink or overwrite an input/sidecar")
    if destination.exists() and not force:
        raise FileExistsError(f"Output exists (use --force): {destination}")
    task = _read(task_path) if task_path and (task_file or task_path.is_file()) else None
    summary = _read(summary_path) if summary_path and (summary_file or summary_path.is_file()) else None
    resolver = None
    if tasks_dir:
        from ..storage import JsonTaskStore
        resolver = JsonTaskStore(tasks_dir)
    rendered = render_trajectory_html(_read(path), task=task, summary=summary,
                                      task_resolver=resolver, **options)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".trajectory-html-", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
        if force:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination
