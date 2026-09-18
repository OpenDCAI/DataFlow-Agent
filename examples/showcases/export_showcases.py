#!/usr/bin/env python3
"""Export selected pipeline rows as GitHub-native Markdown showcases."""

from __future__ import annotations

import argparse
import base64
import copy
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, ImageDraw, ImageFont, ImageOps


CASE_CONFIG: dict[str, dict[str, Any]] = {
    "geometry_proof": {
        "order": 1,
        "slug": "geometry_proof",
        "title": "Image-grounded olympiad geometry",
        "summary": (
            "The agent starts from independent points, progressively constructs the "
            "diagram, reasons over fresh renders, and finishes with a proof."
        ),
        "animation_speed": 1.3,
    },
    "pixel_game": {
        "order": 2,
        "slug": "pixel_game",
        "title": "Visual planning in a Pyxel game",
        "summary": (
            "The agent reads a rendered maze, plans around walls, collects five gems, "
            "and reaches the goal under a deterministic move budget."
        ),
        "animation_speed": 1.3,
    },
    "pptx": {
        "order": 3,
        "slug": "pptx",
        "title": "Editable PowerPoint reconstruction",
        "summary": (
            "The task supplies three visual references. The agent recreates their "
            "content and hierarchy as editable 16:9 slides and reviews the render."
        ),
        "animation_speed": 1.7,
    },
    "diagram": {
        "order": 4,
        "slug": "diagram",
        "title": "Document pages to an editable diagram",
        "summary": (
            "The agent synthesizes two incident-runbook pages into one operational "
            "flow. A pixel-level Judge catches clipped labels and ambiguous routing; "
            "Refine restores the existing artifact and repairs it through fresh "
            "visual observations."
        ),
        "animation_speed": 1.0,
    },
    "vlmgym_2048": {
        "order": 8,
        "slug": "vlmgym_2048",
        "title": "Reading 2048 from pixels (G1 VLM-Gym)",
        "summary": (
            "Starting from a fresh board, the agent reads every tile value from the "
            "rendered G1 VLM-Gym screen, plans merges over 44 moves, and builds a 64 tile."
        ),
        "animation_speed": 2.0,
    },
    "vlmgym_shisensho": {
        "order": 9,
        "slug": "vlmgym_shisensho",
        "title": "Shisen-Sho path planning on a 12x12 board (G1 VLM-Gym)",
        "summary": (
            "On G1's full 12x12 board of 36 color-shape tile types, the agent finds "
            "pairs joined by paths with at most two turns, including routes through "
            "the empty margin, and removes ten pairs before no connectable pair remains."
        ),
        "animation_speed": 1.3,
    },
    "vlmgym_shisensho_cifar10": {
        "order": 10,
        "slug": "vlmgym_shisensho_cifar10",
        "title": "Clearing a CIFAR-10 Shisen-Sho board (G1 VLM-Gym)",
        "summary": (
            "Tiles are CIFAR-10 photos and only the class matters: the agent groups "
            "four different photos per class and clears the whole 6x6 board."
        ),
        "animation_speed": 1.3,
    },
    "vlmgym_swap": {
        "order": 11,
        "slug": "vlmgym_swap",
        "title": "Match-3 swaps with cascades (G1 VLM-Gym)",
        "summary": (
            "The agent scans an 8x8 match-3 board for adjacent swaps that complete "
            "lines, survives automatic reshuffles, and reaches 150 points."
        ),
        "animation_speed": 1.3,
    },
}

REMOVED_SHOWCASE_ENVS = {"chart"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def image_suffix(media_type: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }.get(media_type, ".png")


def text_content(content: Iterable[Mapping[str, Any]], *, limit: int = 1800) -> str:
    value = "\n".join(
        str(item.get("text") or "")
        for item in content
        if item.get("type") == "text" and item.get("text")
    ).strip()
    if len(value) > limit:
        return value[: limit - 1].rstrip() + "…"
    return value


def first_task_message(trajectory: Mapping[str, Any]) -> Mapping[str, Any]:
    for message in trajectory.get("messages") or ():
        if message.get("role") != "user":
            continue
        content = message.get("content") or ()
        if text_content(content, limit=100_000) or any(
            item.get("type") == "image" for item in content
        ):
            return message
    scenario = trajectory.get("scenario")
    if isinstance(scenario, Mapping) and scenario.get("instruction"):
        return {
            "role": "user",
            "content": [{"type": "text", "text": scenario["instruction"]}],
        }
    return {"role": "user", "content": []}


def observation_message(
    trajectory: Mapping[str, Any], step: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    index = step.get("observation_message_index")
    messages = trajectory.get("messages") or ()
    if isinstance(index, int) and 0 <= index < len(messages):
        message = messages[index]
        return message if isinstance(message, Mapping) else None
    return None


def save_images(
    content: Iterable[Mapping[str, Any]],
    destination: Path,
    stem: str,
) -> list[str]:
    paths: list[str] = []
    image_index = 0
    for item in content:
        if item.get("type") != "image" or not item.get("data"):
            continue
        image_index += 1
        media_type = str(item.get("media_type") or "image/png")
        filename = f"{stem}-{image_index:02d}{image_suffix(media_type)}"
        (destination / filename).write_bytes(base64.b64decode(item["data"]))
        paths.append(filename)
    return paths


def quote_markdown(value: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in value.splitlines())


def gallery(items: list[tuple[str, str]]) -> str:
    if not items:
        return ""
    lines = ["<table>"]
    for offset in range(0, len(items), 2):
        pair = items[offset : offset + 2]
        lines.append("  <tr>")
        for path, caption in pair:
            lines.append(
                "    <td align=\"center\" width=\"50%\">"
                f"<img src=\"{html.escape(path, quote=True)}\" alt=\""
                f"{html.escape(caption, quote=True)}\"><br><sub>"
                f"{html.escape(caption)}</sub></td>"
            )
        if len(pair) == 1:
            lines.append("    <td width=\"50%\"></td>")
        lines.append("  </tr>")
    lines.append("</table>")
    return "\n".join(lines)


def evaluation(row: Mapping[str, Any]) -> tuple[str, str]:
    replay = row.get("replay_verification")
    replay_label = "not applicable"
    replay_detail = "This open-ended task has no deterministic verifier."
    if isinstance(replay, Mapping):
        replay_label = str(replay.get("status") or "unknown")
        replay_detail = str(replay.get("reason") or "")
        if replay.get("status") == "passed":
            replay_label += f" · reward {float(replay.get('reward', 0.0)):.1f}"
    judge = row.get("traj_overall")
    judge_label = f"{float(judge):.2f}" if isinstance(judge, (int, float)) else "n/a"
    return replay_label, judge_label + (f"\n\n{replay_detail}" if replay_detail else "")


def collect_observation_images(
    trajectory: Mapping[str, Any],
    assets: Path,
    *,
    prefix: str = "",
) -> dict[int, list[str]]:
    observation_paths: dict[int, list[str]] = {}
    for step in trajectory.get("steps") or ():
        step_index = int(step.get("index") or 0)
        observation = observation_message(trajectory, step)
        if observation is None:
            continue
        stem = f"{prefix}step-{step_index:03d}"
        paths = save_images(
            observation.get("content") or (), assets, stem
        )
        if paths:
            observation_paths[step_index] = paths
    return observation_paths


def fade_animation(
    observation_paths: Mapping[int, list[str]],
    assets: Path,
    filename: str,
    *,
    stage: str | None = None,
    speed: float = 1.0,
) -> str | None:
    """Create a compact, looping overview without replacing source PNGs."""
    items = [
        (step_index, image_index, assets / name)
        for step_index, names in observation_paths.items()
        for image_index, name in enumerate(names, start=1)
    ]
    destination = assets / filename
    if not items:
        destination.unlink(missing_ok=True)
        return None

    # Long authoring trajectories need a smaller preview. The full-resolution
    # observations remain available inside their corresponding tool calls.
    maximum_size = (640, 480) if len(items) > 30 else (760, 640)
    prepared: list[tuple[int, int, Image.Image]] = []
    for step_index, image_index, path in items:
        with Image.open(path) as source:
            frame = ImageOps.contain(
                source.convert("RGB"), maximum_size, Image.Resampling.LANCZOS
            )
        prepared.append((step_index, image_index, frame))

    canvas_width = max(frame.width for _, _, frame in prepared)
    image_height = max(frame.height for _, _, frame in prepared)
    header_height = 36
    canvas_height = image_height + header_height
    font_size = 18 if len(items) > 30 else 20
    try:
        font = ImageFont.load_default(size=font_size)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()

    base_frames: list[Image.Image] = []
    for step_index, image_index, frame in prepared:
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        canvas.paste(
            frame,
            (
                (canvas_width - frame.width) // 2,
                header_height + (image_height - frame.height) // 2,
            ),
        )
        label = f"{stage} · " if stage else ""
        label += f"Step {step_index}"
        if len(observation_paths.get(step_index, ())) > 1:
            label += f" · Image {image_index}"
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, canvas_width, header_height), fill=(17, 24, 39))
        draw.text((14, 7), label, fill="white", font=font)
        base_frames.append(canvas)

    if speed <= 0:
        raise ValueError("animation speed must be positive")
    hold_ms = 900 if len(items) <= 12 else 550 if len(items) <= 30 else 330
    transition_alphas = (0.5,) if len(items) > 30 else (0.33, 0.66)
    # GIF stores durations in centiseconds. Quantize deliberately so the encoded
    # playback rate stays close to the requested multiplier after saving.
    hold_ms = max(20, math.ceil(hold_ms / speed / 10) * 10)
    transition_ms = max(20, round(110 / speed / 10) * 10)
    frames: list[Image.Image] = []
    durations: list[int] = []
    for index, current in enumerate(base_frames):
        frames.append(current)
        durations.append(hold_ms)
        if len(base_frames) > 1:
            following = base_frames[(index + 1) % len(base_frames)]
            for alpha in transition_alphas:
                frames.append(Image.blend(current, following, alpha))
                durations.append(transition_ms)

    palette_frames = [
        frame.quantize(
            colors=64,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        for frame in frames
    ]
    palette_frames[0].save(
        destination,
        save_all=True,
        append_images=palette_frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    return filename


def compact_trajectory(
    row: Mapping[str, Any],
    *,
    prompt: str,
    slug: str,
    observation_paths: Mapping[int, list[str]],
) -> dict[str, Any]:
    trajectory = row["trajectory"]
    compact_steps: list[dict[str, Any]] = []
    for step in trajectory.get("steps") or ():
        observation = observation_message(trajectory, step)
        compact_steps.append({
            "index": step.get("index"),
            "action": copy.deepcopy(step.get("action")),
            "tool_ok": step.get("tool_ok"),
            "error_code": step.get("error_code"),
            "is_final": step.get("is_final"),
            "elapsed_ms": step.get("elapsed_ms"),
            "observation_text": text_content(
                observation.get("content") or () if observation else (),
                limit=20_000,
            ),
            "observation_images": [
                f"../assets/{slug}/{name}"
                for name in observation_paths.get(int(step.get("index") or 0), [])
            ],
        })
    return {
        "episode_id": trajectory.get("episode_id"),
        "success": trajectory.get("success"),
        "termination_reason": trajectory.get("termination_reason"),
        "steps": compact_steps,
        "final_answer": trajectory.get("final_answer"),
        "replay_verification": row.get("replay_verification"),
        "judge": {
            "overall": row.get("traj_overall"),
            "judge_ref": copy.deepcopy(row.get("judge_ref")),
            "scores": copy.deepcopy(row.get("traj_judge_scores")),
            "normalized_scores": copy.deepcopy(row.get("traj_judge_normalized_scores")),
            "goal_achievement": row.get("traj_goal_achievement"),
            "tool_use": row.get("traj_tool_use"),
            "efficiency": row.get("traj_efficiency"),
            "coherence": row.get("traj_coherence"),
            "rationale": row.get("traj_rationale"),
        },
    }


def trajectory_stage_lines(
    row: Mapping[str, Any],
    *,
    slug: str,
    observation_paths: Mapping[int, list[str]],
    stage: str | None,
    animation: str | None,
) -> list[str]:
    trajectory = row["trajectory"]
    qualifier = f"{stage} " if stage else ""
    visual_heading = f"{stage} visual checkpoints" if stage else "Visual checkpoints"
    trajectory_heading = f"{stage} trajectory" if stage else "Trajectory"
    answer_heading = f"{stage} final answer" if stage else "Final answer"
    evaluation_heading = f"{stage} evaluation" if stage else "Evaluation"
    lines: list[str] = []
    if animation:
        caption = (
            f"{stage} trajectory — every rendered observation in chronological order"
            if stage
            else "Full trajectory — every rendered observation in chronological order"
        )
        lines.extend([
            f"## {visual_heading}",
            "",
            '<p align="center">'
            f'<img src="assets/{slug}/{animation}" alt="{html.escape(caption, quote=True)}">'
            f"<br><sub>{html.escape(caption)}</sub></p>",
            "",
        ])
    tools = Counter(
        str(step.get("action", {}).get("tool") or "unknown")
        for step in trajectory.get("steps") or ()
    )
    tool_summary = ", ".join(
        f"`{name}` × {count}" for name, count in tools.most_common()
    )
    lines.extend([f"## {trajectory_heading}", ""])
    if tool_summary:
        lines.extend([f"Action mix: {tool_summary}.", ""])
    else:
        lines.extend([
            "_No actions were recorded in this trajectory._",
            "",
        ])
    for step in trajectory.get("steps") or ():
        action = step.get("action") or {}
        index = int(step.get("index") or 0)
        tool = str(action.get("tool") or "unknown")
        thought = str(action.get("thought") or "").strip()
        summary = " ".join(thought.split())
        if len(summary) > 110:
            summary = summary[:109].rstrip() + "…"
        lines.extend([
            "<details>",
            f"<summary><code>{index:02d} · {html.escape(tool)}</code> — {html.escape(summary)}</summary>",
            "",
        ])
        if thought:
            lines.extend(["**Reasoning**", "", html.escape(thought), ""])
        lines.extend([
            "**Arguments**",
            "",
            "<pre><code>" + html.escape(
                json.dumps(action.get("args") or {}, ensure_ascii=False, indent=2)
            ) + "</code></pre>",
            "",
        ])
        observation = observation_message(trajectory, step)
        observed = text_content(
            observation.get("content") or () if observation else (), limit=1800
        )
        if observed:
            lines.extend([
                "**Observation**",
                "",
                "<pre><code>" + html.escape(observed) + "</code></pre>",
                "",
            ])
        for name in observation_paths.get(index, []):
            lines.extend([
                f"![{qualifier}step {index} observation](assets/{slug}/{name})",
                "",
            ])
        lines.extend(["</details>", ""])

    replay_label, judge_value = evaluation(row)
    judge_label, _, replay_detail = judge_value.partition("\n\n")
    lines.extend([
        f"## {answer_heading}",
        "",
        str(trajectory.get("final_answer") or "No final answer was recorded."),
        "",
        f"## {evaluation_heading}",
        "",
        f"- ReplayVerify: **{replay_label}**",
        f"- Judge: **{judge_label}**",
    ])
    if replay_detail:
        lines.append(f"- Replay detail: {replay_detail}")
    rubric = row.get("judge_ref")
    scores = row.get("traj_judge_scores")
    normalized = row.get("traj_judge_normalized_scores") or {}
    if isinstance(rubric, Mapping) and isinstance(scores, Mapping):
        score_range = rubric.get("score_range") or {}
        lines.extend([
            "",
            f"Task-owned rubric range: {score_range.get('min')}–{score_range.get('max')}. "
            "Overall is the arithmetic mean of range-normalized criterion scores.",
            "",
            "| Criterion | Raw score | Normalized score |",
            "| --- | ---: | ---: |",
        ])
        for criterion in rubric.get("criteria") or ():
            key = criterion["id"]
            lines.append(f"| `{key}` | {scores.get(key, 'n/a')} | {normalized.get(key, 'n/a')} |")
        lines.extend([
            "",
            "<details>",
            "<summary>Task judge_ref used for this evaluation</summary>",
            "",
            "<pre><code>" + html.escape(json.dumps(rubric, ensure_ascii=False, indent=2)) + "</code></pre>",
            "",
            "</details>",
        ])
    rationale = row.get("traj_rationale")
    if rationale:
        lines.extend([
            "",
            "<details>",
            "<summary>Judge rationale</summary>",
            "",
            html.escape(str(rationale)),
            "",
            "</details>",
        ])
    lines.append("")
    return lines


def render_case(
    row: Mapping[str, Any],
    output_dir: Path,
    original_row: Mapping[str, Any] | None = None,
) -> None:
    env_id = str(row["env_id"])
    config = CASE_CONFIG[env_id]
    trajectory = row["trajectory"]
    slug = str(config["slug"])
    assets = output_dir / "assets" / slug
    assets.mkdir(parents=True, exist_ok=True)
    compact_dir = output_dir / "trajectories"
    compact_dir.mkdir(parents=True, exist_ok=True)

    # Generated visual records are rebuilt from the selected source row so stale
    # frames from an earlier export cannot survive in the published showcase.
    for pattern in (
        "input-reference-*",
        "step-*",
        "original-step-*",
        "trajectory.gif",
        "original-trajectory.gif",
        "refined-trajectory.gif",
    ):
        for generated in assets.glob(pattern):
            generated.unlink()

    task_message = first_task_message(trajectory)
    prompt = text_content(task_message.get("content") or (), limit=100_000)
    input_images = save_images(
        task_message.get("content") or (), assets, "input-reference"
    )

    observation_paths = collect_observation_images(trajectory, assets)
    is_refine_pair = bool(row.get("_refined") and original_row is not None)
    original_observation_paths: dict[int, list[str]] = {}
    if is_refine_pair:
        original_trajectory = original_row["trajectory"]
        original_observation_paths = collect_observation_images(
            original_trajectory,
            assets,
            prefix="original-",
        )

    if is_refine_pair:
        animation_speed = float(config.get("animation_speed", 1.0))
        original_animation = fade_animation(
            original_observation_paths,
            assets,
            "original-trajectory.gif",
            stage="Original",
            speed=animation_speed,
        )
        refined_animation = fade_animation(
            observation_paths,
            assets,
            "refined-trajectory.gif",
            stage="Refined",
            speed=animation_speed,
        )
    else:
        original_animation = None
        refined_animation = fade_animation(
            observation_paths,
            assets,
            "trajectory.gif",
            speed=float(config.get("animation_speed", 1.0)),
        )

    compact_refined = compact_trajectory(
        row, prompt=prompt, slug=slug, observation_paths=observation_paths
    )
    if is_refine_pair:
        compact = {
            "format": "dataflow-agentmm-showcase-refine-pair-v2",
            "task_id": row.get("task_id"),
            "env_id": env_id,
            "task_prompt": prompt,
            "original": compact_trajectory(
                original_row,
                prompt=prompt,
                slug=slug,
                observation_paths=original_observation_paths,
            ),
            "refined": compact_refined,
        }
    else:
        compact = {
            "format": "dataflow-agentmm-showcase-compact-v2",
            "task_id": row.get("task_id"),
            "env_id": env_id,
            "task_prompt": prompt,
            **compact_refined,
        }
    (compact_dir / f"{slug}.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    branch = "refined" if row.get("_refined") else "initial"
    lines = [
        f"# {config['title']}",
        "",
        "[← All showcases](README.md)",
        "",
    ]
    if is_refine_pair:
        lines.extend([
            "> **Refine comparison:** this case shows the complete original trajectory "
            "and the complete refined trajectory. Refine created a new rollout; it did "
            "not overwrite the original record.",
            "",
        ])
    lines.extend([
        "> **Complete trajectory imagery:** the animation is a compressed overview of "
        "every rendered observation. Every original image is also preserved inside its "
        "corresponding tool-call section.",
        "",
    ])
    lines.extend([str(config["summary"]), ""])
    if is_refine_pair:
        original_replay, original_judge_value = evaluation(original_row)
        original_judge, _, _ = original_judge_value.partition("\n\n")
        refined_replay, refined_judge_value = evaluation(row)
        refined_judge, _, _ = refined_judge_value.partition("\n\n")
        lines.extend([
            "## Original → Refined at a glance",
            "",
            "| Stage | Episode | Steps | ReplayVerify | Judge |",
            "| --- | --- | ---: | --- | ---: |",
            f"| Original | `{original_row['trajectory'].get('episode_id')}` | "
            f"{len(original_row['trajectory'].get('steps') or ())} | {original_replay} | {original_judge} |",
            f"| Refined | `{trajectory.get('episode_id')}` | "
            f"{len(trajectory.get('steps') or ())} | {refined_replay} | {refined_judge} |",
            "",
        ])
    else:
        replay_label, judge_value = evaluation(row)
        judge_label, _, _ = judge_value.partition("\n\n")
        lines.extend([
            "| Env | Task | Branch | Steps | ReplayVerify | Judge |",
            "| --- | --- | --- | ---: | --- | ---: |",
            f"| `{env_id}` | `{row.get('task_id')}` | {branch} | "
            f"{len(trajectory.get('steps') or ())} | {replay_label} | {judge_label} |",
            "",
        ])
    lines.extend(["## Task", "", quote_markdown(prompt), ""])
    if input_images:
        lines.extend([
            "## Input references",
            "",
            gallery([
                (f"assets/{slug}/{name}", f"Input reference {index}")
                for index, name in enumerate(input_images, start=1)
            ]),
            "",
        ])
    lines.extend([
        f"[Open compact {'original + refined ' if is_refine_pair else ''}trajectory JSON](trajectories/{slug}.json).",
        "",
    ])
    if is_refine_pair:
        lines.extend(trajectory_stage_lines(
            original_row,
            slug=slug,
            observation_paths=original_observation_paths,
            stage="Original",
            animation=original_animation,
        ))
        lines.extend([
            "## Why Refine ran",
            "",
            quote_markdown(str(original_row.get("traj_rationale") or "The original trajectory did not satisfy the quality gate.")),
            "",
        ])
        lines.extend(trajectory_stage_lines(
            row,
            slug=slug,
            observation_paths=observation_paths,
            stage="Refined",
            animation=refined_animation,
        ))
    else:
        lines.extend(trajectory_stage_lines(
            row,
            slug=slug,
            observation_paths=observation_paths,
            stage=None,
            animation=refined_animation,
        ))
    (output_dir / f"{config['order']:02d}_{slug}.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def select_task(rows: Iterable[dict[str, Any]], task_id: str) -> dict[str, Any]:
    return next(row for row in rows if row.get("task_id") == task_id)


def legacy_verifier(row: Mapping[str, Any]) -> Mapping[str, Any]:
    trajectory = row.get("trajectory") or {}
    for key in ("verification", "verifier"):
        value = trajectory.get(key)
        if isinstance(value, Mapping):
            return value
    replay = row.get("replay_verification")
    return replay if isinstance(replay, Mapping) else {}


def normalize_legacy_replay(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    verifier = legacy_verifier(row)
    normalized["replay_verification"] = {
        "status": "passed" if verifier.get("passed") else "failed",
        "passed": bool(verifier.get("passed")),
        "reward": float(verifier.get("reward") or 0.0),
        "reason": str(verifier.get("reason") or ""),
        "checks": copy.deepcopy(verifier.get("checks") or []),
    }
    return normalized


def render_verifier_case(
    false_positive: Mapping[str, Any],
    pipeline_initial: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    initial_by_task = {row["task_id"]: row for row in pipeline_initial}
    original = normalize_legacy_replay(initial_by_task["task0003"])
    refined = normalize_legacy_replay(false_positive)
    original_trajectory = original["trajectory"]
    refined_trajectory = refined["trajectory"]
    assets = output_dir / "assets" / "deterministic_verifier"
    assets.mkdir(parents=True, exist_ok=True)
    compact_dir = output_dir / "trajectories"
    compact_dir.mkdir(parents=True, exist_ok=True)
    # Remove the superseded final-only export now that this case is a pair.
    (compact_dir / "deterministic_verifier_false_positive.json").unlink(
        missing_ok=True
    )
    for pattern in (
        "step-*",
        "original-step-*",
        "refined-step-*",
        "original-trajectory.gif",
        "refined-trajectory.gif",
    ):
        for generated in assets.glob(pattern):
            generated.unlink()
    prompt = str((refined_trajectory.get("scenario") or {}).get("instruction") or "")
    if not prompt:
        prompt = text_content(
            first_task_message(refined_trajectory).get("content") or (), limit=100_000
        )

    original_image_paths = collect_observation_images(
        original_trajectory,
        assets,
        prefix="original-",
    )
    refined_image_paths = collect_observation_images(
        refined_trajectory,
        assets,
        prefix="refined-",
    )
    original_animation = fade_animation(
        original_image_paths,
        assets,
        "original-trajectory.gif",
        stage="Original",
        speed=1.3,
    )
    refined_animation = fade_animation(
        refined_image_paths,
        assets,
        "refined-trajectory.gif",
        stage="Refined",
        speed=1.3,
    )
    compact = {
        "format": "dataflow-agentmm-showcase-refine-pair-v2",
        "task_id": refined.get("task_id"),
        "env_id": refined.get("env_id"),
        "task_prompt": prompt,
        "original": compact_trajectory(
            original,
            prompt=prompt,
            slug="deterministic_verifier",
            observation_paths=original_image_paths,
        ),
        "refined": compact_trajectory(
            refined,
            prompt=prompt,
            slug="deterministic_verifier",
            observation_paths=refined_image_paths,
        ),
    }
    (compact_dir / "deterministic_verifier_refine_pair.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    original_verifier = legacy_verifier(original)
    refined_verifier = legacy_verifier(refined)
    checks = refined_verifier.get("checks") or ()
    check_text = "; ".join(
        f"{item.get('name')}: {'pass' if item.get('passed') else 'fail'}"
        for item in checks if isinstance(item, Mapping)
    )
    lines = [
        "# Why a deterministic verifier is necessary",
        "",
        "[← All showcases](README.md)",
        "",
        "> **Refine comparison:** this case shows the complete original trajectory "
        "and the complete refined trajectory. The refined rollout improved the VLM "
        "Judge score, but deterministic replay still rejected its exact final state.",
        "",
        "> **Complete trajectory imagery:** each animation includes every rendered "
        "observation in order, and every original image remains attached to its tool call.",
        "",
        "A VLM Judge evaluates whether a trajectory looks coherent and complete. It is "
        "not the source of truth for hidden or exact environment state.",
        "",
        "## Original → Refined at a glance",
        "",
        "| Stage | Episode | Steps | Finish called | Verifier | Judge |",
        "| --- | --- | ---: | --- | --- | ---: |",
        f"| Original | `{original_trajectory.get('episode_id')}` | "
        f"{len(original_trajectory.get('steps') or ())} | "
        f"{'yes' if original_trajectory.get('termination_reason') == 'finish' else 'no'} | "
        f"{'passed' if original_verifier.get('passed') else 'failed'} | "
        f"{float(original.get('traj_overall') or 0.0):.2f} |",
        f"| Refined | `{refined_trajectory.get('episode_id')}` | "
        f"{len(refined_trajectory.get('steps') or ())} | "
        f"{'yes' if refined_trajectory.get('termination_reason') == 'finish' else 'no'} | "
        f"{'passed' if refined_verifier.get('passed') else 'failed'} | "
        f"{float(refined.get('traj_overall') or 0.0):.2f} |",
        "",
        "## Task",
        "",
        quote_markdown(prompt),
        "",
        "[Open compact original + refined trajectory JSON](trajectories/deterministic_verifier_refine_pair.json).",
        "",
    ]
    lines.extend(trajectory_stage_lines(
        original,
        slug="deterministic_verifier",
        observation_paths=original_image_paths,
        stage="Original",
        animation=original_animation,
    ))
    lines.extend([
        "## Why Refine ran",
        "",
        quote_markdown(str(original.get("traj_rationale") or "The original trajectory did not complete the task.")),
        "",
    ])
    lines.extend(trajectory_stage_lines(
        refined,
        slug="deterministic_verifier",
        observation_paths=refined_image_paths,
        stage="Refined",
        animation=refined_animation,
    ))
    lines.extend([
        "## What deterministic replay found",
        "",
        "The refined agent called `finish`, and the Judge assigned 1.0 after reading "
        "the last screen as a successful return to START. Fresh deterministic replay "
        "found a different exact state: all gems were collected, but the player was "
        "not at START. The refined natural-language completion claim was still a false "
        "positive.",
        "",
        f"Exact checks: {check_text or 'see paired trajectory JSON'}.",
        "",
        "## Takeaway",
        "",
        "Judge and ReplayVerify answer different questions:",
        "",
        "- Judge: was the visual reasoning process useful, coherent, and well executed?",
        "- ReplayVerify: do the recorded actions reproduce, and does exact task state satisfy the contract?",
        "",
        "Open-ended authoring tasks correctly return `not_applicable`; deterministic "
        "verification is added only when the environment exposes a meaningful exact condition.",
        "",
    ])
    (output_dir / "12_why_deterministic_verifier.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--input", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--false-positive", type=Path, required=True)
    value.add_argument("--pipeline-initial", type=Path, required=True)
    value.add_argument(
        "--refine-original",
        type=Path,
        action="append",
        default=[],
        help=(
            "Judged JSONL containing original rows for selected refined cases; "
            "repeat for multiple pipeline output directories."
        ),
    )
    return value


def main() -> int:
    args = parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.input)
    unknown_envs = {
        str(row.get("env_id")) for row in rows
    } - set(CASE_CONFIG) - REMOVED_SHOWCASE_ENVS
    if unknown_envs:
        raise ValueError(
            "showcase input contains unconfigured environments: "
            + ", ".join(sorted(unknown_envs))
        )
    rows = [
        row for row in rows
        if str(row.get("env_id")) not in REMOVED_SHOWCASE_ENVS
    ]
    original_rows = [
        row
        for source in args.refine_original
        for row in load_jsonl(source)
    ]
    originals_by_case = {
        (str(item.get("env_id")), str(item.get("task_id"))): item
        for item in original_rows
    }
    for row in sorted(rows, key=lambda item: CASE_CONFIG[item["env_id"]]["order"]):
        original = originals_by_case.get(
            (str(row.get("env_id")), str(row.get("task_id")))
        )
        if row.get("_refined") and original is None:
            raise ValueError(
                "selected refined case is missing its original trajectory: "
                f"{row.get('env_id')}/{row.get('task_id')}"
            )
        render_case(row, args.output_dir, original)
    false_positive = select_task(load_jsonl(args.false_positive), "task0003")
    render_verifier_case(
        false_positive,
        load_jsonl(args.pipeline_initial),
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
