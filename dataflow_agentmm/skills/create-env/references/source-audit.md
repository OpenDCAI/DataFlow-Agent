# Source and MCP audit

## Contents

1. Static inventory
2. Source modes
3. MCP-specific audit
4. Capability matrix
5. Security and provenance

## 1. Static inventory

Run the bundled scanner from the skill directory:

```bash
python scripts/scan_source.py /absolute/or/relative/source --format markdown
```

The scanner reports languages, manifests, documentation, MCP and DataFlow-AgentMM
signals, candidate tools, sensitive filenames, and a short recommended read
set. It deliberately does not print file contents or secret values. It is a
heuristic index, not proof of an API.

Confirm its results with fast source search:

```bash
rg --files SOURCE
rg -n "ToolSpec|list_tools|tools/list|FastMCP|server\\.tool|registerTool|inputSchema" SOURCE
rg -n "execute|handler|dispatch|call_tool|CallTool|EnvironmentSpec|class .*Env" SOURCE
```

Read in this order:

1. repository instructions and license;
2. package manifests and entry points;
3. server/tool registration;
4. input schemas and handlers;
5. state/session model;
6. import/export and rendering code;
7. tests and fixtures;
8. README claims;
9. generated output only when maintained source is absent.

Never infer tool completeness from filenames alone. A handler may register
tools dynamically, alias names, or gate capabilities behind optional modules.

## 2. Source modes

### Existing DataFlow-AgentMM Env

Compare `Env.tools()` with actual dispatch branches, then check:

- every `ToolSpec` has exactly one implementation;
- queries leave domain state equivalent when the Env owns state;
- mutations return observations from post-mutation state;
- optional `start` handles private init as well as `None`;
- optional snapshots are detached and never treated as a universal contract;
- `close` is safe before/after start and after partial failures;
- artifacts are invalidated or regenerated after state changes;
- task-bound verification still matches the canonical domain state.

### MCP server

Find protocol and server versions, all tool registrations, resources/prompts,
transport, process lifecycle, and any UI/session dependency. Separate protocol
surface from implementation helpers. MCP resources or prompts do not
automatically become Env tools; decide whether they are reset inputs, public
references, observations, or out of scope.

### Application or library

Start from user-visible operations rather than exporting every function.
Choose actions that have bounded JSON arguments, controlled state effects,
and observable results. Reject arbitrary shell, eval, unrestricted file paths,
credential stores, and ambient host access. Wrap an existing engine behind a
backend only if it can be isolated per episode when isolation is required.

### Empty concept

Write this brief before code:

```text
User outcome:
Canonical state:
Optional start inputs:
Queries:
Mutations:
Text/image observations:
Saved artifacts:
Deterministic success evidence:
Episode isolation and teardown:
Explicitly excluded powers:
```

Derive the first capability matrix from the brief, then review it with the
human before expanding into an expensive or externally stateful backend.

## 3. MCP-specific audit

For static source, identify:

- `tools/list` declarations and `tools/call` routing;
- exact tool names, descriptions, required/default fields, enums, and limits;
- error codes and partial-success behavior;
- server-global versus connection/session state;
- resources, templates, prompts, subscriptions, and notifications;
- stdio/HTTP transport assumptions;
- filesystem, browser, subprocess, network, and credential access;
- generated artifacts and their canonical format;
- startup and shutdown requirements.

If an approved MCP client is available, use `tools/list` as a cross-check.
Compare runtime discovery with static registrations and explain differences.
Do not send mutating calls during discovery unless the user requested an
integration test and the server is isolated.

For non-Python MCPs, do not translate implementation syntax line by line.
Extract schemas and behavior, then implement a Python Env adapter around a
canonical model or a narrow backend transport. Keep the original process only
when its artifact fidelity or domain engine is essential and isolation is
demonstrated.

## 4. Capability matrix

Use one row per public operation:

| Field | Meaning |
| --- | --- |
| name | Exact upstream name and aliases |
| schema | Required fields, defaults, bounds, unions |
| reads/writes | External and canonical state touched |
| state | Preconditions and postconditions |
| observation | Text, image, artifact, or structured payload |
| errors | Invalid input, missing state, dependency failure |
| dependency | Browser, binary, network, renderer, file format |
| decision | implement, adapt, defer, reject |
| test | Contract, round-trip, negative, or visual test |

Also record parity separately:

- **API:** names and schemas;
- **semantic:** state transitions and errors;
- **artifact:** read/write round trips and fidelity;
- **UI/session:** selection, tabs, events, or remote state.

An `adapt` decision must name the replacement semantics. A `defer` decision
must leave an extension boundary if full support is a stated goal. A `reject`
decision should identify the safety or reproducibility reason.

## 5. Security and provenance

- Read the upstream license before copying code, fixtures, fonts, icons, or
  schemas. Prefer a clean controlled rewrite when redistribution is unclear.
- Never copy `.env`, headers, tokens, cookies, user directories, trajectories,
  or internal URLs into source, tests, docs, or fixtures.
- Do not print suspect file contents during scanning. Report filenames only.
- Treat symlinks as escape paths when confining a workspace.
- Keep upstream snapshots as non-runtime provenance if the project needs them;
  prevent imports from the Env into the snapshot tree.
- Distinguish source facts from inferences in the parity report.
