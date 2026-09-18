# Env validation and release gates v2

## Contract and domain tests

- A fresh factory result structurally implements `tools` and `call`.
- Registry metadata contains no task, verifier, init-schema, or state-schema
  ownership.
- Tool names are unique and every in-scope tool dispatches.
- Unknown names, invalid JSON-schema arguments, and semantic errors fail with
  stable codes.
- Optional `start` accepts `None` for a Task without Scenario.
- Rollout and replay start the Env before discovering tools. A catalog that
  requires startup works; an Env with no startup hook still works.
- Startup errors (including invalid initial content or terminal results) stop
  discovery and execution. Startup and catalog failures still close the Env.
- Catalog discovery runs once per episode; the model sees the resulting tools
  before its first decision, with initial observations after task messages.
- Separate Env instances/workspaces do not share episode state.
- Query calls leave domain state unchanged; failed compound mutations are
  atomic; optional `close` is idempotent.
- Visual observations decode, and relevant mutations visibly change them.
- File paths reject absolute, traversal, and symlink escapes; exported
  artifacts reopen with an independent parser.

Catalog equality is not handler coverage. Execute each claimed operation with
meaningful preconditions.

## Task, rollout, and ReplayVerify tests

For every reviewed Task:

1. resolve the exact `(env_id, task_id)` from a v2 Task store;
2. run an oracle/scripted solution through `AgentRollout`;
3. prove the Trajectory contains neither Scenario nor verifier output;
4. replay in a fresh Env and require the expected domain result;
5. corrupt required domain outcomes and require `failed`;
6. corrupt `ok`, error code, or `is_final` and require `diverged` before the
   verifier factory is called;
7. change only observation text/image bytes and prove replay control remains
   exact;
8. reproduce parse, unknown-tool, and schema failures through the shared loop;
9. reject actions after a terminal action;
10. prove a no-verifier Task returns `not_applicable` without Env creation.

For isolated Envs, assert Task providers and snapshots are not worker RPCs.
Custom verification runs against the still-live worker Env.

## Live model canary

After deterministic gates pass, run a representative Task through the real
serving adapter with bounded steps and a fresh output directory. Inspect tool
arguments/errors, image turns, termination, artifacts, and the separate replay
result. A live success demonstrates usability but does not replace deterministic
tests.

## Packaging and privacy

Build wheel and sdist, inspect them, and install into a clean environment.
Confirm plugin entry points and optional task/assets. Scan tracked/package
files for secrets, private keys, internal hosts/users, absolute workspaces,
caches, previews, and generated trajectories. Report skipped gates and
controlled parity gaps exactly.
