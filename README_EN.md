# DataFlow-Agent

DataFlow-Agent is a state-driven, modular AI agent framework with an extensible Agent/Workflow/Tool system. It ships with a CLI and Gradio UI pages to support operator recommendation, pipeline generation/debugging, operator QA, and web collection workflows.

[中文](README.md) | English

## Quickstart

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
python gradio_app/app.py --page_set data
```

## CLI

```bash
dfa --help
dfa create --agent_name my_agent
dfa create --wf_name my_workflow
```

## Docs

```bash
mkdocs serve
```

## License

Apache-2.0. See `LICENSE`.

