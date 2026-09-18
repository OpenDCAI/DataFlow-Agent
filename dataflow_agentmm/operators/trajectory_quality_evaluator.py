"""Official trajectory-quality judge adapted to canonical multimodal messages."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Mapping, Sequence

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage

from ..contracts import (
    ImageContent,
    JudgeCriterion,
    JudgeReference,
    Message,
    TaskResolver,
    TextContent,
)
from ..prompts import (
    GENERIC_JUDGE_CRITERIA,
    JUDGE_LABELS,
    JUDGE_SYSTEM_PROMPT,
    LANGUAGES,
    for_language,
)
from ..serving import ModelServing
from .utils.trajectory import (
    as_trajectory_dict,
    normal_success,
    observation_content,
    step_error_code,
    step_thought,
    step_tool_ok,
    steps,
)


def generic_judge_ref(language: str = "en") -> JudgeReference:
    """Fallback rubric used when a Task declares no judge_ref."""

    return JudgeReference(
        score_min=1,
        score_max=5,
        criteria=tuple(
            JudgeCriterion(identifier, description)
            for identifier, description in for_language(GENERIC_JUDGE_CRITERIA, language)
        ),
    )


GENERIC_JUDGE_REF = generic_judge_ref("en")
AXES = GENERIC_JUDGE_REF.criterion_ids
DEFAULT_MAX_COMBINED_RUBRIC_CHARS = 16_000


def _judge_response_options(
    rubric: JudgeReference,
) -> dict[str, Any]:
    score_properties = {
        criterion.id: {
            "type": "number",
            "minimum": rubric.score_min,
            "maximum": rubric.score_max,
        }
        for criterion in rubric.criteria
    }
    result = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "trajectory_quality_verdict",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "rationale": {"type": "string"},
                        "scores": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": score_properties,
                            "required": list(rubric.criterion_ids),
                        },
                        # Repair evidence for Refine; empty when the judge sees
                        # no problem or cannot localize one.
                        "important_steps": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 1},
                        },
                        "refine_suggestion": {"type": "string"},
                    },
                    "required": [
                        "rationale", "scores", "important_steps", "refine_suggestion",
                    ],
                },
            },
        },
    }
    return result


@OPERATOR_REGISTRY.register()
class AgentMMTrajectoryQualityEvaluator(OperatorABC):
    """LLM/VLM Judge with Task rubrics and deterministic score aggregation."""

    def __init__(
        self,
        llm_serving: ModelServing | None = None,
        max_workers: int = 8,
        max_observation_chars: int = 1500,
        system_prompt: str | None = None,
        score_prefix: str = "traj_",
        task_resolver: TaskResolver | None = None,
        language: str = "en",
        max_combined_rubric_chars: int | None = (
            DEFAULT_MAX_COMBINED_RUBRIC_CHARS
        ),
    ):
        if (
            max_combined_rubric_chars is not None
            and max_combined_rubric_chars < 1
        ):
            raise ValueError(
                "max_combined_rubric_chars must be positive or None"
            )
        if language not in LANGUAGES:
            raise ValueError(f"language must be one of {sorted(LANGUAGES)}")
        self.logger = get_logger()
        self.language = language
        self.labels = for_language(JUDGE_LABELS, language)
        self.generic_rubric = generic_judge_ref(language)
        self.llm_serving = llm_serving
        self.task_resolver = task_resolver
        self.max_workers = max_workers
        self.max_observation_chars = max_observation_chars
        self.system_prompt = system_prompt or for_language(JUDGE_SYSTEM_PROMPT, language)
        self.score_prefix = score_prefix
        self.max_combined_rubric_chars = max_combined_rubric_chars

    @staticmethod
    def get_desc(lang: str = "zh") -> str:
        if lang == "zh":
            return "按 Task judge_ref 逐项评分并归一化取均值；缺省时使用通用四维 rubric。"
        return (
            "Scores each Task judge_ref criterion and averages normalized "
            "scores, with a generic four-axis fallback."
        )

    @staticmethod
    def _initial_task_text(trajectory: Mapping[str, Any]) -> str:
        """Return the original user instruction, excluding Refine prompts."""

        for raw in trajectory.get("messages") or []:
            if not isinstance(raw, Mapping) or raw.get("role") != "user":
                continue
            values = [
                str(item.get("text") or "")
                for item in raw.get("content") or []
                if isinstance(item, Mapping) and item.get("type") == "text"
            ]
            text = "\n".join(value for value in values if value)
            if text:
                return text
        return ""

    def _judge_messages(
        self,
        trajectory: dict[str, Any],
        rubric: JudgeReference,
        replay_verification: Any | None = None,
        *,
        focused_criterion_ids: tuple[str, ...] | None = None,
    ) -> tuple[Message, ...]:
        if focused_criterion_ids is not None:
            if not focused_criterion_ids:
                raise ValueError("focused_criterion_ids must not be empty")
            unknown = set(focused_criterion_ids).difference(rubric.criterion_ids)
            if unknown:
                raise ValueError(
                    f"unknown focused criterion ids: {sorted(unknown)}"
                )
        header: list[TextContent | ImageContent] = [TextContent(
            self.labels["task"].format(task=self._initial_task_text(trajectory))
        )]
        task_images: list[ImageContent] = []
        for raw in trajectory.get("messages") or []:
            if not isinstance(raw, Mapping) or raw.get("role") != "user":
                continue
            try:
                message = Message.from_dict(raw)
            except (KeyError, TypeError, ValueError):
                continue
            task_images.extend(
                item for item in message.content if isinstance(item, ImageContent)
            )
        if task_images:
            header.append(TextContent(
                self.labels["task_images"].format(count=len(task_images))
            ))
            header.extend(task_images)
        header.append(TextContent(self.labels["rubric"].format(
            rubric=json.dumps(rubric.to_dict(), ensure_ascii=False)
        )))
        if focused_criterion_ids is not None:
            header.append(TextContent(self.labels["shard"].format(
                ids=json.dumps(focused_criterion_ids, ensure_ascii=False)
            )))
        header.append(TextContent(self.labels["evidence_note"]))
        if replay_verification is not None:
            header.append(TextContent(self.labels["replay"].format(
                verification=json.dumps(
                    replay_verification, ensure_ascii=False, sort_keys=True
                )
            )))
        messages: list[Message] = [
            Message.text("system", self.system_prompt),
            Message.of("user", header),
        ]
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
            messages.append(Message.text("user", self.labels["step"].format(
                index=index,
                thought=step_thought(step),
                tool=tool,
                args=json.dumps(args, ensure_ascii=False),
                flag=flag,
            )))
            observed: list[TextContent | ImageContent] = []
            for item in observation_content(trajectory, step):
                if isinstance(item, TextContent):
                    text = item.text
                    if len(text) > self.max_observation_chars:
                        text = text[:self.max_observation_chars] + "...[truncated]"
                    observed.append(TextContent(text + "\n"))
                else:
                    observed.append(item)
            if observed:
                # Preserve the semantic role so multimodal serving adapters can
                # retain the newest visual checkpoints under an image budget.
                messages.append(Message.of(
                    "observation",
                    observed,
                    name=str(tool or "trajectory.step"),
                ))
        messages.append(Message.text("user", self.labels["final"].format(
            answer=trajectory.get("final_answer"),
            success=normal_success(trajectory),
            steps=len(steps(trajectory)),
        )))
        return tuple(messages)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        if not text:
            return None
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidate = fenced.group(1) if fenced else None
        if candidate is None:
            start = text.find("{")
            if start < 0:
                return None
            depth = 0
            in_string = False
            escaped = False
            for index in range(start, len(text)):
                character = text[index]
                if in_string:
                    if escaped:
                        escaped = False
                    elif character == "\\":
                        escaped = True
                    elif character == '"':
                        in_string = False
                    continue
                if character == '"':
                    in_string = True
                elif character == "{":
                    depth += 1
                elif character == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:index + 1]
                        break
        if candidate is None:
            return None
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _empty_verdict(rationale: str) -> dict[str, Any]:
        return {
            "scores": {},
            "model_scores": {},
            "normalized_scores": {},
            "overall": None,
            "rationale": rationale,
            "important_steps": [],
            "refine_suggestion": "",
        }

    @staticmethod
    def _coerce_rubric(value: Any) -> JudgeReference:
        if isinstance(value, JudgeReference):
            return value
        if isinstance(value, Mapping):
            return JudgeReference.from_dict(value)
        raise TypeError("judge_ref must be a JudgeReference or object")

    @staticmethod
    def _missing_rubric(value: Any) -> bool:
        return value is None or (
            isinstance(value, float) and value != value
        )

    def _rubric_for_row(
        self,
        record: Mapping[str, Any],
        trajectory_value: Any,
        rubric_key: str | None,
    ) -> JudgeReference:
        trajectory = as_trajectory_dict(trajectory_value)
        if self.task_resolver is not None:
            task_id = record.get("task_id")
            env_id = record.get("env_id")
            if trajectory is not None:
                task_id = task_id or trajectory.get("task_id")
                env_id = env_id or trajectory.get("env_id")
            if not isinstance(task_id, str) or not isinstance(env_id, str):
                raise ValueError(
                    "task_resolver Judge requires task_id and env_id"
                )
            task = self.task_resolver.resolve(task_id, env_id=env_id)
            return task.judge_ref or self.generic_rubric

        raw = record.get(rubric_key) if rubric_key is not None else None
        if self._missing_rubric(raw):
            return self.generic_rubric
        return self._coerce_rubric(raw)

    def _judge_one(
        self,
        value: Any,
        rubric: JudgeReference,
        replay_verification: Any | None = None,
    ) -> dict[str, Any]:
        trajectory = as_trajectory_dict(value)
        if trajectory is None:
            return self._empty_verdict("unparseable_trajectory")

        serialized_rubric = json.dumps(rubric.to_dict(), ensure_ascii=False)
        should_shard = (
            len(rubric.criteria) > 1
            and self.max_combined_rubric_chars is not None
            and len(serialized_rubric) > self.max_combined_rubric_chars
        )
        if should_shard:
            return self._judge_one_sharded(
                trajectory,
                rubric,
                replay_verification,
                reason=(
                    f"rubric_chars={len(serialized_rubric)} exceeds "
                    f"{self.max_combined_rubric_chars}"
                ),
            )
        try:
            messages = self._judge_messages(
                trajectory, rubric, replay_verification
            )
            generated = self.llm_serving.generate_messages_with_options(
                [messages],
                [_judge_response_options(rubric)],
            )
            if len(generated) != 1:
                raise RuntimeError("judge serving must return exactly one response")
            raw = generated[0]
        except Exception as exc:  # judge errors remain row-local
            self.logger.warning(
                f"[AgentMMTrajectoryQualityEvaluator] judge call failed: {exc}"
            )
            return self._empty_verdict(f"judge_error: {exc}")
        verdict = self._extract_json(raw)
        if verdict is None:
            if len(rubric.criteria) > 1:
                return self._judge_one_sharded(
                    trajectory,
                    rubric,
                    replay_verification,
                    reason="combined verdict was unparseable",
                )
            return self._empty_verdict("unparseable_verdict")
        try:
            model_scores = self._validated_scores(
                verdict, rubric
            )
        except (TypeError, ValueError) as exc:
            if len(rubric.criteria) > 1:
                return self._judge_one_sharded(
                    trajectory,
                    rubric,
                    replay_verification,
                    reason=f"combined verdict was invalid: {exc}",
                )
            return self._empty_verdict(f"invalid_verdict: {exc}")
        important_steps, refine_suggestion = self._repair_evidence(verdict, trajectory)
        return self._finalize_verdict(
            trajectory,
            rubric,
            model_scores,
            str(verdict.get("rationale", "")),
            important_steps,
            refine_suggestion,
        )

    @staticmethod
    def _validated_scores(
        verdict: Mapping[str, Any],
        rubric: JudgeReference,
    ) -> dict[str, float]:
        scores = verdict.get("scores")
        if not isinstance(scores, Mapping):
            raise TypeError("scores must be an object")
        expected = set(rubric.criterion_ids)
        if set(scores) != expected:
            raise ValueError(
                "scores must contain exactly the configured criterion ids"
            )
        model_scores: dict[str, float] = {}
        for criterion_id in rubric.criterion_ids:
            rubric.normalize(scores[criterion_id])
            model_scores[criterion_id] = float(scores[criterion_id])
        return model_scores

    def _judge_one_sharded(
        self,
        trajectory: dict[str, Any],
        rubric: JudgeReference,
        replay_verification: Any | None,
        *,
        reason: str,
    ) -> dict[str, Any]:
        model_scores: dict[str, float] = {}
        rationales = [f"[judge_sharded] {reason}"]
        important: list[int] = []
        suggestions: list[str] = []
        for criterion in rubric.criteria:
            shard = JudgeReference(
                score_min=rubric.score_min,
                score_max=rubric.score_max,
                criteria=(criterion,),
            )
            try:
                messages = self._judge_messages(
                    trajectory,
                    rubric,
                    replay_verification,
                    focused_criterion_ids=shard.criterion_ids,
                )
                generated = self.llm_serving.generate_messages_with_options(
                    [messages],
                    [_judge_response_options(shard)],
                )
                if len(generated) != 1:
                    raise RuntimeError(
                        "judge serving must return exactly one response"
                    )
                verdict = self._extract_json(generated[0])
                if verdict is None:
                    raise ValueError("unparseable verdict")
                model_scores.update(self._validated_scores(
                    verdict, shard
                ))
                rationales.append(
                    f"[{criterion.id}]\n{str(verdict.get('rationale', ''))}"
                )
                shard_steps, shard_suggestion = self._repair_evidence(verdict, trajectory)
                important.extend(step for step in shard_steps if step not in important)
                if shard_suggestion:
                    suggestions.append(f"[{criterion.id}] {shard_suggestion}")
            except Exception as exc:
                self.logger.warning(
                    "[AgentMMTrajectoryQualityEvaluator] judge shard "
                    f"{criterion.id!r} failed: {exc}"
                )
                return self._empty_verdict(
                    f"judge_shard_error[{criterion.id}]: {exc}"
                )
        return self._finalize_verdict(
            trajectory,
            rubric,
            model_scores,
            "\n\n".join(rationales),
            sorted(important),
            "\n".join(suggestions),
        )

    @staticmethod
    def _repair_evidence(
        verdict: Mapping[str, Any],
        trajectory: Mapping[str, Any],
    ) -> tuple[list[int], str]:
        """Keep only in-range step numbers and a plain-text suggestion."""

        total = len(steps(trajectory))
        raw_steps = verdict.get("important_steps")
        important: list[int] = []
        if isinstance(raw_steps, Sequence) and not isinstance(raw_steps, (str, bytes)):
            for value in raw_steps:
                if isinstance(value, bool) or not isinstance(value, int):
                    continue
                if 1 <= value <= total and value not in important:
                    important.append(value)
        suggestion = verdict.get("refine_suggestion")
        return sorted(important), suggestion.strip() if isinstance(suggestion, str) else ""

    def _finalize_verdict(
        self,
        trajectory: dict[str, Any],
        rubric: JudgeReference,
        model_scores: Mapping[str, float],
        rationale: str,
        important_steps: Sequence[int] = (),
        refine_suggestion: str = "",
    ) -> dict[str, Any]:
        raw_scores = dict(model_scores)
        normalized = {
            criterion_id: rubric.normalize(raw_scores[criterion_id])
            for criterion_id in rubric.criterion_ids
        }
        return {
            "scores": raw_scores,
            "model_scores": model_scores,
            "normalized_scores": normalized,
            "overall": sum(normalized.values()) / len(normalized),
            "rationale": rationale,
            "important_steps": list(important_steps),
            "refine_suggestion": refine_suggestion,
        }

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "trajectory",
        output_key: str = "traj_overall",
        rubric_key: str | None = "judge_ref",
        replay_key: str | None = "replay_verification",
    ):
        if self.llm_serving is None:
            raise ValueError(
                "AgentMMTrajectoryQualityEvaluator requires an llm_serving instance"
            )
        dataframe = storage.read(output_type="dataframe")
        if input_key not in dataframe.columns:
            raise KeyError(
                f"input_key {input_key!r} not found in columns: "
                f"{list(dataframe.columns)}"
            )
        if replay_key is not None and replay_key not in dataframe.columns:
            replay_key = None
        values = dataframe[input_key].tolist()
        records = dataframe.to_dict(orient="records")
        rubrics = [
            self._rubric_for_row(record, values[index], rubric_key)
            for index, record in enumerate(records)
        ]
        dataframe["judge_ref"] = [rubric.to_dict() for rubric in rubrics]
        replay_values = (
            dataframe[replay_key].tolist()
            if replay_key is not None
            else [None] * len(values)
        )
        verdicts: list[dict[str, Any] | None] = [None] * len(values)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    self._judge_one,
                    value,
                    rubrics[index],
                    replay_values[index],
                ): index
                for index, value in enumerate(values)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    verdicts[index] = future.result()
                except Exception as exc:
                    verdicts[index] = self._empty_verdict(f"error: {exc}")
        complete = [item or self._empty_verdict("missing_verdict") for item in verdicts]
        for axis in AXES:
            dataframe[f"{self.score_prefix}{axis}"] = [
                item["scores"].get(axis) for item in complete
            ]
        dataframe[f"{self.score_prefix}judge_scores"] = [
            item["scores"] for item in complete
        ]
        dataframe[f"{self.score_prefix}judge_model_scores"] = [
            item["model_scores"] for item in complete
        ]
        dataframe[f"{self.score_prefix}judge_normalized_scores"] = [
            item["normalized_scores"] for item in complete
        ]
        dataframe[output_key] = [item["overall"] for item in complete]
        dataframe[f"{self.score_prefix}rationale"] = [
            item["rationale"] for item in complete
        ]
        dataframe[f"{self.score_prefix}important_steps"] = [
            item["important_steps"] for item in complete
        ]
        dataframe[f"{self.score_prefix}refine_suggestion"] = [
            item["refine_suggestion"] for item in complete
        ]
        storage.write(dataframe)
        return [output_key]
