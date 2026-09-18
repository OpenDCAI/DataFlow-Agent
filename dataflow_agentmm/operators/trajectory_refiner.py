"""Official selective trajectory repair adapted to controlled MM rollouts."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

from ..contracts import Content, ImageContent, Message, TaskResolver, TextContent
from ..prompts import LANGUAGES, REFINE_TEXT, for_language
from ..serving import ModelServing
from .utils.trajectory import (
    as_trajectory_dict,
    normal_success,
    observation_content,
    step_error_code,
    step_thought,
    step_tool_ok,
    steps,
    task_text,
)
from .explore_generator import AgentMMExploreGenerator


@OPERATOR_REGISTRY.register()
class AgentMMTrajectoryRefiner(OperatorABC):
    """Repair only failed or below-threshold trajectories by re-exploring."""

    def __init__(
        self,
        llm_serving: ModelServing | None = None,
        max_steps: int = 64,
        max_workers: int = 8,
        task_resolver: TaskResolver | None = None,
        refine_failed: bool = True,
        score_threshold: float | None = 0.6,
        score_key: str = "traj_overall",
        diagnosis_key: str | None = "traj_rationale",
        suggestion_key: str | None = "traj_refine_suggestion",
        important_steps_key: str | None = "traj_important_steps",
        verification_key: str | None = "replay_verification",
        important_step_window: int = 2,
        original_key: str | None = "trajectory_original",
        system_prompt: str | None = None,
        language: str = "en",
        max_prior_chars: int = 2000,
        max_prior_images: int | None = 4,
        max_diagnosis_chars: int = 4000,
        validate_tool_names: bool = True,
        structured_actions: bool = False,
        action_format_retries: int = 0,
        include_host_tools: bool = True,
        workspace_root: str | Path | None = None,
        workspace_retention: str = "ephemeral",
    ):
        if max_prior_chars < 1:
            raise ValueError("max_prior_chars must be positive")
        if max_prior_images is not None and (
            isinstance(max_prior_images, bool)
            or not isinstance(max_prior_images, int)
            or max_prior_images < 1
        ):
            raise ValueError("max_prior_images must be positive or None")
        if max_diagnosis_chars < 1:
            raise ValueError("max_diagnosis_chars must be positive")
        if (
            isinstance(important_step_window, bool)
            or not isinstance(important_step_window, int)
            or important_step_window < 0
        ):
            raise ValueError("important_step_window must be a non-negative integer")
        if language not in LANGUAGES:
            raise ValueError(f"language must be one of {sorted(LANGUAGES)}")
        self.logger = get_logger()
        self.language = language
        self.text = for_language(REFINE_TEXT, language)
        self.llm_serving = llm_serving
        if task_resolver is None:
            raise ValueError("AgentMMTrajectoryRefiner requires task_resolver")
        self.task_resolver = task_resolver
        self.max_steps = max_steps
        self.max_workers = max_workers
        self.refine_failed = refine_failed
        self.score_threshold = score_threshold
        self.score_key = score_key
        self.diagnosis_key = diagnosis_key
        self.suggestion_key = suggestion_key
        self.important_steps_key = important_steps_key
        self.verification_key = verification_key
        self.important_step_window = important_step_window
        self.original_key = original_key
        self.system_prompt = system_prompt
        self.max_prior_chars = max_prior_chars
        self.max_prior_images = max_prior_images
        self.max_diagnosis_chars = max_diagnosis_chars
        self.validate_tool_names = validate_tool_names
        self.include_host_tools = include_host_tools
        self._generator = AgentMMExploreGenerator(
            serving=llm_serving,
            task_resolver=task_resolver,
            max_steps=max_steps,
            max_workers=max_workers,
            system_prompt=system_prompt,
            validate_tool_names=validate_tool_names,
            structured_actions=structured_actions,
            action_format_retries=action_format_retries,
            include_host_tools=include_host_tools,
            workspace_root=workspace_root,
            workspace_retention=workspace_retention,
        )

    def _diagnose(self, trajectory: dict[str, Any]) -> str:
        if not normal_success(trajectory):
            answer = trajectory.get("final_answer")
            if answer is None or (isinstance(answer, str) and not answer.strip()):
                return self.text["note_no_answer"]
        notes: list[str] = []
        seen_actions: dict[str, int] = {}
        for step in steps(trajectory):
            if step.get("parse_error"):
                notes.append(self.text["note_unparseable"])
            if step_error_code(step) == "unknown_tool":
                tool = (step.get("action") or {}).get("tool")
                notes.append(self.text["note_unknown_tool"].format(tool=tool))
            if step_tool_ok(step) is False:
                notes.append(self.text["note_tool_failed"].format(
                    code=step_error_code(step)
                ))
            action = step.get("action") or {}
            tool = action.get("tool")
            if tool and tool != "finish":
                try:
                    args = json.dumps(
                        action.get("args", {}), sort_keys=True, ensure_ascii=False
                    )
                except (TypeError, ValueError):
                    args = str(action.get("args"))
                key = f"{tool}:{args}"
                seen_actions[key] = seen_actions.get(key, 0) + 1
        if any(count > 1 for count in seen_actions.values()):
            notes.append(self.text["note_loop"])
        if not notes:
            return self.text["note_low_quality"]
        return "; ".join(dict.fromkeys(notes)) + "."

    @staticmethod
    def _step_headline(index: int, step: dict[str, Any]) -> str:
        action = step.get("action") or {}
        flag = ""
        if step.get("parse_error"):
            flag = " [PARSE_ERROR]"
        elif step_error_code(step) == "unknown_tool":
            flag = " [INVALID_TOOL]"
        elif step_tool_ok(step) is False:
            flag = " [TOOL_ERROR]"
        return (
            f"  {index}. tool={action.get('tool')} "
            f"args={json.dumps(action.get('args', {}), ensure_ascii=False)}{flag}"
        )

    def _important_indices(
        self,
        trajectory: dict[str, Any],
        important_steps: Sequence[int],
    ) -> list[int]:
        """Flagged steps plus ``important_step_window`` neighbours on each side."""

        total = len(steps(trajectory))
        selected: set[int] = set()
        for value in important_steps:
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if not 1 <= value <= total:
                continue
            low = max(1, value - self.important_step_window)
            high = min(total, value + self.important_step_window)
            selected.update(range(low, high + 1))
        return sorted(selected)

    def _important_step_messages(
        self,
        trajectory: dict[str, Any],
        important_steps: Sequence[int],
    ) -> tuple[Message, ...]:
        """Render flagged steps in full: no text truncation, images attached."""

        indices = self._important_indices(trajectory, important_steps)
        if not indices:
            return ()
        flagged = ", ".join(str(value) for value in sorted(set(important_steps)) if value in
                            range(1, len(steps(trajectory)) + 1))
        messages: list[Message] = [Message.text(
            "user",
            self.text["important_steps_header"].format(flagged=flagged or "(none)"),
            name="trajectory_refiner.important_steps",
        )]
        all_steps = steps(trajectory)
        for index in indices:
            step = all_steps[index - 1]
            content: list[Content] = [TextContent(
                f"{self._step_headline(index, step)}\n"
                f"     thought={json.dumps(step_thought(step), ensure_ascii=False)}\n"
                + self.text["observation_label"]
            )]
            content.extend(observation_content(trajectory, step))
            messages.append(Message.of(
                "observation",
                content,
                name=f"trajectory_refiner.step{index:03d}",
            ))
        return tuple(messages)

    def _render_prior(self, trajectory: dict[str, Any]) -> str:
        """Render a bounded text-only summary; images are attached separately.

        The tail matters most for repair, so an over-budget summary keeps the
        first and last steps and elides the middle.
        """
        lines: list[str] = []
        for index, step in enumerate(steps(trajectory), start=1):
            action = step.get("action") or {}
            tool = action.get("tool")
            args = action.get("args", {})
            flag = ""
            if step.get("parse_error"):
                flag = " [PARSE_ERROR]"
            elif step_error_code(step) == "unknown_tool":
                flag = " [INVALID_TOOL]"
            elif step_tool_ok(step) is False:
                flag = " [TOOL_ERROR]"
            lines.append(
                f"  {index}. tool={tool} "
                f"args={json.dumps(args, ensure_ascii=False)}{flag}"
            )
            observed = observation_content(trajectory, step)
            text = " ".join(
                item.text.strip()
                for item in observed
                if isinstance(item, TextContent) and item.text.strip()
            )
            image_count = sum(isinstance(item, ImageContent) for item in observed)
            if text:
                lines.append(f"     observation_text: {text}")
            if image_count:
                lines.append(
                    f"     observation_images: {image_count} "
                    "(attached separately as multimodal image content when selected)"
                )
        lines.append(f"  final_answer: {trajectory.get('final_answer')}")
        rendered = "\n".join(lines)
        if len(rendered) <= self.max_prior_chars:
            return rendered
        head_budget = self.max_prior_chars // 4
        head, tail, size = [], [], 0
        for line in lines:
            if size + len(line) + 1 > head_budget:
                break
            head.append(line)
            size += len(line) + 1
        size = 0
        for line in reversed(lines[len(head):]):
            if size + len(line) + 1 > self.max_prior_chars - head_budget:
                break
            tail.append(line)
            size += len(line) + 1
        omitted = len(lines) - len(head) - len(tail)
        return "\n".join([
            *head,
            self.text["omitted_lines"].format(count=omitted),
            *reversed(tail),
        ])

    def _prior_images(
        self,
        trajectory: dict[str, Any],
    ) -> list[tuple[int, ImageContent]]:
        images = [
            (index, item)
            for index, step in enumerate(steps(trajectory), start=1)
            for item in observation_content(trajectory, step)
            if isinstance(item, ImageContent)
        ]
        if self.max_prior_images is not None:
            images = images[-self.max_prior_images:]
        return images

    def _refine_instruction_messages(
        self,
        trajectory: dict[str, Any],
        *,
        diagnosis: str,
        task: str,
        verification: Any = None,
        suggestion: str = "",
        important_steps: Sequence[int] = (),
    ) -> tuple[Message, ...]:
        """Build repair context without reclassifying screenshots as task refs.

        Task-authored reference images stay in the original user message.  Prior
        trajectory screenshots use the observation role so serving adapters can
        preserve the task refs first and spend the remaining request budget on
        the newest visual checkpoints, exactly as they do during rollout.
        """

        messages: list[Message] = []
        messages.append(Message.text(
            "user",
            self.text["context_header"].format(
                prior=self._render_prior(trajectory),
                diagnosis=diagnosis,
            ),
            name="trajectory_refiner.diagnosis",
        ))
        if verification is not None:
            try:
                rendered = json.dumps(verification, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                rendered = str(verification)
            messages.append(Message.text(
                "user",
                self.text["verification_header"].format(verification=rendered),
                name="trajectory_refiner.verification",
            ))
        if suggestion:
            messages.append(Message.text(
                "user",
                self.text["suggestion_header"].format(suggestion=suggestion),
                name="trajectory_refiner.suggestion",
            ))
        important = self._important_step_messages(trajectory, important_steps)
        messages.extend(important)
        # Flagged steps already carry their own untruncated observations; only
        # fall back to recency-selected screenshots when none were flagged.
        images = [] if important else self._prior_images(trajectory)
        if images:
            visual_context: list[Content] = [
                TextContent(self.text["selected_images_start"])
            ]
            for step_index, image in images:
                visual_context.extend((
                    TextContent(self.text["selected_images_item"].format(index=step_index)),
                    image,
                ))
            visual_context.append(TextContent(self.text["selected_images_end"]))
            messages.append(Message.of(
                "observation",
                visual_context,
                name="trajectory_refiner.previous_observations",
            ))
        final_instruction: list[Content] = [
            TextContent(self.text["footer"].format(task=task))
        ]
        messages.append(Message.of(
            "user",
            final_instruction,
            name="trajectory_refiner.instruction",
        ))
        return tuple(messages)

    def _should_refine(self, trajectory: dict[str, Any] | None, score: Any) -> bool:
        if trajectory is None:
            return False
        if self.refine_failed and not normal_success(trajectory):
            return True
        if self.score_threshold is not None and score is not None:
            try:
                return float(score) < self.score_threshold
            except (TypeError, ValueError):
                return False
        return False

    def _refine_one(
        self,
        value: Any,
        score: Any,
        judge_diagnosis: Any = None,
        earliest_original: Any = None,
        verification: Any = None,
        suggestion: Any = None,
        important_steps: Any = None,
    ) -> dict[str, Any]:
        trajectory = as_trajectory_dict(value)
        if not self._should_refine(trajectory, score):
            return {
                "trajectory": value,
                "original": None,
                "refined": False,
                "note": "kept (passed quality / unparseable)",
                "improved": False,
            }
        try:
            task = self.task_resolver.resolve(
                str(trajectory["task_id"]), env_id=str(trajectory["env_id"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            return {
                "trajectory": value,
                "original": None,
                "refined": False,
                "note": f"refine_error: cannot resolve task: {exc}",
                "improved": False,
            }
        original_task = task_text(trajectory)
        diagnosis = self._diagnose(trajectory)
        if isinstance(judge_diagnosis, str) and judge_diagnosis.strip():
            feedback = judge_diagnosis.strip()
            if len(feedback) > self.max_diagnosis_chars:
                feedback = feedback[:self.max_diagnosis_chars] + "...[truncated]"
            diagnosis = f"{diagnosis}\n" + self.text["judge_feedback"].format(
                feedback=feedback
            )
        try:
            corrections = self._refine_instruction_messages(
                trajectory,
                diagnosis=diagnosis,
                task=original_task,
                verification=verification if isinstance(verification, Mapping) else None,
                suggestion=suggestion.strip() if isinstance(suggestion, str) else "",
                important_steps=(
                    important_steps
                    if isinstance(important_steps, Sequence)
                    and not isinstance(important_steps, (str, bytes))
                    else ()
                ),
            )
            # Repair always re-explores from a fresh Env for now. Restoring the
            # recorded prefix first will come back as an Env-side tool; see
            # AgentRollout.run_with_response_prefix.
            repaired = self._generator._runner().run(replace(
                task,
                messages=(*task.messages, *corrections),
            ))
        except Exception as exc:
            self.logger.error(f"[AgentMMTrajectoryRefiner] refine failed: {exc}")
            return {
                "trajectory": value,
                "original": None,
                "refined": False,
                "note": f"refine_error: {exc}",
                "improved": False,
            }
        improved = repaired.success and not normal_success(trajectory)
        return {
            "trajectory": repaired.to_dict(),
            "original": (
                earliest_original
                if as_trajectory_dict(earliest_original) is not None
                else trajectory
            ),
            "refined": True,
            "note": f"refined: {diagnosis}",
            "improved": improved,
        }

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "trajectory",
        output_key: str = "trajectory",
    ):
        if self.llm_serving is None:
            raise ValueError("AgentMMTrajectoryRefiner requires an llm_serving instance")
        dataframe = storage.read(output_type="dataframe")
        if input_key not in dataframe.columns:
            raise KeyError(
                f"input_key {input_key!r} not found in columns: "
                f"{list(dataframe.columns)}"
            )
        scores = (
            dataframe[self.score_key].tolist()
            if self.score_key in dataframe.columns
            else [None] * len(dataframe)
        )
        diagnoses = (
            dataframe[self.diagnosis_key].tolist()
            if self.diagnosis_key is not None
            and self.diagnosis_key in dataframe.columns
            else [None] * len(dataframe)
        )
        values = dataframe[input_key].tolist()

        def column(name: str | None) -> list[Any]:
            if name is not None and name in dataframe.columns:
                return dataframe[name].tolist()
            return [None] * len(dataframe)

        verifications = column(self.verification_key)
        suggestions = column(self.suggestion_key)
        important_steps = column(self.important_steps_key)
        earliest_originals = (
            dataframe[self.original_key].tolist()
            if self.original_key is not None
            and self.original_key in dataframe.columns
            else [None] * len(dataframe)
        )
        results: list[dict[str, Any] | None] = [None] * len(values)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    self._refine_one,
                    value,
                    score,
                    diagnoses[index],
                    earliest_originals[index],
                    verifications[index],
                    suggestions[index],
                    important_steps[index],
                ): index
                for index, (value, score) in enumerate(zip(values, scores))
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    results[index] = {
                        "trajectory": values[index],
                        "original": None,
                        "refined": False,
                        "note": f"crash: {exc}",
                        "improved": False,
                    }
        complete = [item for item in results if item is not None]
        dataframe[output_key] = [item["trajectory"] for item in complete]
        if self.original_key is not None:
            dataframe[self.original_key] = [item["original"] for item in complete]
        dataframe["_refined"] = [item["refined"] for item in complete]
        dataframe["_refine_note"] = [item["note"] for item in complete]
        storage.write(dataframe)
        return [output_key]
