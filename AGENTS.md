# DataFlow-AgentMM workspace

Treat this repository as a human–coding-agent workspace as well as a Python
distribution. Preserve user changes, keep credentials and local runtime output
out of source, and validate changes in proportion to their risk.

When asked to create, port, expand, or audit an DataFlow-AgentMM Env—or to translate an
MCP server, application, library, external process, or blank product idea into
an Env—read the complete
`dataflow_agentmm/skills/create-env/SKILL.md` before taking implementation
actions. Follow its routing instructions and read each referenced resource
required for the current starting mode.

Do not import concrete environments into the core package. Env implementations,
tasks, deterministic verifiers, and their concrete dependencies belong in
separate Env packs or isolated examples; the core package owns contracts,
registration, runtime components, serving adapters, operators, and storage.
