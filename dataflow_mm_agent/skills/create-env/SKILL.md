---
name: create-env
description: Audit an existing environment, MCP server, application, library, or empty product idea and turn it into a lightweight DataFlow-MM-Agent Env adapter, optional Task catalog, and independently replay-verifiable workflow.
---

# Create Agent-MM Env v2

Build the smallest controlled adapter that preserves the intended operation.
Do not impose state, tasks, Scenarios, or verifiers on an integration that does
not need them.

## Select the source mode

- Existing Agent-MM Env: audit its actual tool dispatcher and lifecycle.
- MCP server: inventory declarations, handlers, transport, session ownership,
  resources, filesystem/network access, and artifacts.
- Application/library: expose bounded user operations, not arbitrary code.
- No source: agree on a capability brief before choosing a state model.

For source-backed work, read
[references/source-audit.md](references/source-audit.md) and run
`scripts/scan_source.py`. The scan is routing evidence, not API truth.

## Workflow

### 1. Preserve and scope

Read repository instructions, inspect the worktree, and create the requested
recovery checkpoint before material edits. Define which tool, transition,
artifact, and UI/session parity levels are in scope. Record controlled
adaptations and explicit exclusions.

### 2. Audit the operation surface

Build a capability matrix with one row per operation:

`name | schema | reads | writes | state/session | observation | artifact | errors | dependency | decision | test`

Classify each row `implement`, `adapt`, `defer`, or `reject`. Read authoritative
registration and handler code, licenses, tests, and dependency manifests.

### 3. Choose the lightest contract

Read [references/contracts-and-layout.md](references/contracts-and-layout.md).
Every Env implements only `tools()` and `call()`. Add `start(init, workspace)`
only for episode initialization and `close()` only for cleanup. State and
snapshots remain optional implementation capabilities; `snapshot()` and
`verify_task(binding, rollout)` follow the signatures documented in
[references/contracts-and-layout.md](references/contracts-and-layout.md).

Rollout and replay complete optional `start` before reading `tools()`, then
retain that catalog for the episode. A session-based MCP can connect in
`start` and discover tools on the same session; do not require a pre-generated
catalog or a separate discovery session for this ordering.

`EnvironmentSpec` is solver-facing registration metadata, not an init/state
schema. Register only the factory and description. Never add runtime `finish`.

For a direct MCP bridge, write a small integration-specific SDK adapter and
call ordinary `register_env`; do not introduce a framework-level generic MCP
registry. Keep the MCP SDK and server dependencies in the Env package.

### 4. Implement and harden

Validate JSON arguments, return stable structured errors, confine filesystem
effects to the episode workspace, and make failed compound mutations atomic.
For visual Envs, return an image after relevant diagram mutations. Query calls
must not mutate domain state.

If the domain benefits from a canonical state/backend/renderer split, build
it. Do not manufacture a canonical model for a stateless remote operation just
to satisfy a pattern.

### 5. Add tasks only when requested

Read [references/task-generation.md](references/task-generation.md) when tasks
are in scope. A required reusable `Task` contains id, Env id, and model
messages. Its private `Scenario` is optional and exists only to transfer init
information into a fresh run. Replay verifier factories are resolved
independently by Task identity.

Use strict v2 task JSON with `JsonTaskStore`. Verifier descriptors are inert
data bound through an explicit trusted builder table. LLM task authoring is a
dataset-build step; rollout and replay never regenerate private task data.

### 6. Verify independently

The runner produces an unscored `Trajectory`. `ReplayVerify` resolves its Task,
strictly replays actions in a fresh Env, and only then creates the task-bound
verifier. Store the result in a sibling `replay_verification` pipeline field.
Do not mutate the trajectory or require a binary `reached_goal` convention.

Tasks without a verifier must yield `not_applicable` without Env creation.

### 7. Package and validate

Expose an idempotent `register()` through the
`dataflow_mm_agent.environments` entry-point group. Include task JSON/assets in
wheel and sdist. For process-isolated deployment, document the required
environment variables and the worker interpreter setup as described in
[references/contracts-and-layout.md](references/contracts-and-layout.md).
Read [references/validation.md](references/validation.md) and run applicable
static, domain, replay, isolation, package, privacy, and live canary gates.

## Completion report

Report capability counts, controlled differences, registration/package paths,
Task and replay coverage, tests/canaries, remaining gaps, and the recovery
checkpoint. Do not claim full parity from catalog equality or one happy path.
