# Task authoring and synthesis v2

## Ownership

Task generation is a dataset-build operation. Runtime receives a reviewed,
reusable `Task`; replay resolves the same `(env_id, task_id)`. Do not ask an LLM
to regenerate init data, goals, or verifier bindings during rollout or replay.

## Strict materialized shape

```json
{
  "schema_version": 2,
  "task_id": "task0001",
  "env_id": "my_env",
  "messages": [
    {"role": "user", "content": [{"type": "text", "text": "Solve this."}]}
  ],
  "scenario": {
    "init": {"private_world": {}}
  },
  "verification": {"kind": "my_trusted_binding"},
  "judge_ref": {
    "score_range": {"min": 1, "max": 5},
    "criteria": [
      {"id": "accuracy", "description": "Check the task-specific result."}
    ]
  }
}
```

`scenario` may be absent or null. Put solver instructions only in `messages`.
Put private fresh-run data only in `scenario.init`; omit Scenario instead of
using an empty init. Runtime step budgets belong
to `RolloutConfig`. A reviewed task-specific Judge rubric belongs in optional
`judge_ref`; provenance still belongs to dataset/pipeline records.

`verification` is store-level binding data, not part of the core `Task`
contract. `Task.from_dict` rejects it, so task JSON files that include it must
be loaded through `JsonTaskStore`, which strips and resolves the binding
separately. To construct a `Task` directly in code, omit `verification` from
the dict you pass to `Task.from_dict` and wire the verifier through the
explicit resolver path instead.

`judge_ref` contains exactly `score_range` and `criteria`. Require finite
`min < max`, at least one criterion, unique stable criterion ids, and concrete
descriptions of observable evidence. Criteria are equally weighted: the Judge
operator normalizes every returned score with the configured range and takes
the arithmetic mean. Omit `judge_ref` to use the built-in generic rubric.

Solver-visible source material must be resource-closed at materialization
time. Small text may remain inline; store-owned images and UTF-8 text/Markdown
may use digest-pinned `image_ref` and `text_ref` message blocks. The task store
must confine each path, verify its SHA-256, and materialize canonical content
before rollout. Do not publish a task that still depends on a URL, an assumed
browser/search tool, an unavailable upload, or an unpinned local path. Missing
or stale resources are task-build failures, not instructions for the solver to
search the web.

Verification JSON is inert and independent of Scenario. Bind each allowed
`kind` through an explicit trusted builder passed to `JsonTaskStore`. Never
import a callable named by task data.

## LLM authoring loop

1. Supply solver-facing Env metadata, the real tool catalog, the Task v2
   schema, family brief, and reviewed examples.
2. Request strict JSON candidates with distinct ids and no Markdown.
3. Parse without semantic repair and reject unknown fields.
4. Validate message content, init domain constraints, ids, bounds, and private
   data separation.
5. Start a fresh Env and solve with a deterministic oracle/domain planner.
6. Replay the oracle trajectory and require the task-bound verifier to pass.
7. Corrupt every required outcome independently and require `failed`.
8. Tamper a control result and require `diverged` before verifier creation.
9. Review instruction clarity, diversity, reachability, leaks, and artifacts.
10. Verify every declared resource exists locally, is hash-pinned, and is
    sufficient to solve the task without undeclared external tools.
11. Materialize atomically without overwriting reviewed tasks by default.

For open-ended tasks, omit the verifier. Their ReplayVerify result should be
`not_applicable`; use a Judge as separate quality evidence.

## Prompt skeleton

```text
Author Tasks for Env {env_id}. Return a JSON array only.
Every item must follow Task schema_version 2.

Solver-facing Env metadata:
{environment_context}

Tool catalog:
{tools}

Task family:
{family_brief}

Requirements:
- put instructions in messages;
- add scenario only for private fresh-run transfer;
- use only approved inert verifier kinds;
- do not add budgets, provenance, code, credentials, or unconfined paths;
- add judge_ref only when a reviewed task-specific rubric is useful;
- do not expose hidden expected values in solver-visible messages.
```

Prompt compliance is not acceptance. Code owns every validation and oracle
gate.
