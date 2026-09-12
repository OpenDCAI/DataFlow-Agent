from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from copy import deepcopy

import pytest
from PIL import Image

from dataflow_mm_agent import Trajectory
from dataflow_mm_agent.visualization import export_trajectory_html, render_trajectory_html
from dataflow_mm_agent.visualization.__main__ import main


def trajectory():
    return {
        "schema_version": 2, "episode_id": "episode-1", "task_id": "task-1", "env_id": "any_env",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "SYSTEM AND TOOLS"}]},
            {"role": "user", "content": [{"type": "text", "text": "ACTUAL TASK"}]},
            {"role": "observation", "name": "env.start", "content": [{"type": "text", "text": "STARTED"}]},
            {"role": "assistant", "content": [{"type": "text", "text": '{"tool":"echo","args":{"text":"x"}}'}]},
            {"role": "observation", "name": "echo", "content": [{"type": "text", "text": "OBSERVATION"}]},
            {"role": "assistant", "content": [{"type": "text", "text": '{"tool":"finish","args":{"answer":"DONE"}}'}]},
        ],
        "steps": [
            {"index": 1, "response_message_index": 3, "observation_message_index": 4,
             "action": {"tool": "echo", "args": {"text": "x"}}, "parse_error": False,
             "elapsed_ms": 0, "tool_ok": True, "error_code": None, "retryable": None, "is_final": False},
            {"index": 2, "response_message_index": 5, "observation_message_index": None,
             "action": {"tool": "finish", "args": {"answer": "DONE"}}, "parse_error": False,
             "elapsed_ms": 10, "tool_ok": True, "error_code": None, "retryable": None, "is_final": True},
        ],
        "final_answer": "DONE", "termination_reason": "finish", "success": True, "num_steps": 2,
        "started_at": "2026-09-12T00:00:00+00:00", "completed_at": "2026-09-12T00:00:02+00:00",
        "metadata": {"structured_actions": False},
    }


def payload(html):
    return json.loads(re.search(r'<script id="trajectory-data" type="application/json">(.*?)</script>', html, re.S)[1])


def image_block():
    stream = io.BytesIO()
    Image.new("RGB", (8, 6), "#227799").save(stream, format="PNG")
    return {"type": "image", "media_type": "image/png", "data": base64.b64encode(stream.getvalue()).decode()}


def test_canonical_object_keeps_messages_steps_and_result():
    value = Trajectory.from_dict(trajectory())
    result = payload(render_trajectory_html(value))["records"][0]
    assert result["variants"][0]["trajectory"] == value.to_dict()
    assert not result["judge"]["present"]
    assert result["judge"]["overall"] is None


@pytest.mark.parametrize("encoded", [False, True])
def test_pipeline_row_and_json_string_cell(encoded):
    value = trajectory()
    original = deepcopy(value); original["episode_id"] = "before-refine"
    row = {"trajectory": json.dumps(value) if encoded else value, "trajectory_original": original,
           "_refined": True, "traj_overall": 0, "traj_judge_scores": {"accuracy": 0},
           "traj_judge_model_scores": {"accuracy": 0}, "traj_judge_normalized_scores": {"accuracy": 0},
           "traj_rationale": "REAL RATIONALE", "judge_ref": {"score_range": {"min": 0, "max": 4},
             "criteria": [{"id": "accuracy", "description": "ACCURACY CRITERION"}]}}
    record = payload(render_trajectory_html(row))["records"][0]
    assert len(record["variants"]) == 2
    assert record["judge"]["present"] and record["judge"]["overall"] == 0
    assert record["judge"]["scores"]["accuracy"] == 0
    assert record["judge"]["scope"] == "row"
    assert "不要将该分数视为修订后的评分" in record["judge"]["scope_note"]
    assert record["judge"]["rubric"] == row["judge_ref"]


def test_custom_pipeline_columns():
    record = payload(render_trajectory_html({"roll": trajectory(), "quality": .7, "score_rationale": "WHY"},
                                            trajectory_key="roll", score_key="quality", score_prefix="score_"))["records"][0]
    assert record["judge"]["overall"] == .7
    assert record["judge"]["rationale"] == "WHY"


def test_inline_images_are_validated_and_deduplicated():
    value = trajectory(); block = image_block()
    value["messages"][4]["content"].extend([block, deepcopy(block)])
    data = payload(render_trajectory_html([value, value]))
    assert len(data["assets"]) == 1
    result = data["records"][0]["variants"][0]["trajectory"]["messages"][4]["content"]
    assert result[1]["asset_id"] == result[2]["asset_id"]
    assert "data" not in result[1]


@pytest.mark.parametrize("mime,data", [("image/svg+xml", "PHN2Zy8+"), ("image/png", "not base64"),
                                      ("image/png", "PGh0bWw+"), ("image/png", "")])
def test_invalid_and_active_images_are_not_rendered(mime, data):
    value = trajectory(); value["messages"][4]["content"] = [{"type": "image", "media_type": mime, "data": data}]
    result = payload(render_trajectory_html(value))
    assert result["assets"] == {}
    assert result["records"][0]["variants"][0]["trajectory"]["messages"][4]["content"][0]["type"] == "unavailable_image"


def test_script_injection_is_data_not_markup_and_csp_hash_matches():
    attack = '</script><img src=x onerror="alert(1)"><script>alert("owned")</script>\u2028&'
    value = trajectory(); value["messages"][1]["content"][0]["text"] = attack
    html = render_trajectory_html(value, title=attack)
    assert attack not in html and html.count("</script>") == 2
    assert payload(html)["title"] == attack
    assert "connect-src 'none'" in html and "object-src 'none'" in html
    script = re.search(r"<script>(.*?)</script>", html, re.S)[1]
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    assert f"script-src 'sha256-{digest}'" in html
    assert ".innerHTML" not in script and "eval(" not in script and "fetch(" not in script


def test_private_task_state_is_excluded_and_mismatch_is_flagged():
    task = {"env_id": "any_env", "task_id": "task-1", "messages": [],
            "scenario": {"init": {"private": "HIDDEN_INIT"}}, "verification": {"kind": "HIDDEN_VERIFIER"}}
    html = render_trajectory_html(trajectory(), task=task)
    assert "HIDDEN_INIT" not in html and "HIDDEN_VERIFIER" not in html
    task["task_id"] = "wrong"
    record = payload(render_trajectory_html(trajectory(), task=task))["records"][0]
    assert record["task"] is None and record["warnings"]


def test_nan_is_missing_not_zero_and_summary_does_not_become_judge():
    row = {"trajectory": trajectory(), "traj_overall": float("nan")}
    record = payload(render_trajectory_html(row, summary={"total_tokens": 0, "model": "any-model"}))["records"][0]
    assert not record["judge"]["present"] and record["judge"]["overall"] is None
    assert record["summary"]["total_tokens"] == 0
    value = trajectory(); value["steps"][0]["elapsed_ms"] = float("nan")
    record = payload(render_trajectory_html({"trajectory": json.dumps(value)}))["records"][0]
    assert record["variants"][0]["trajectory"]["steps"][0]["elapsed_ms"] is None


def test_errors_invalid_indexes_and_unbound_messages_are_preserved():
    value = trajectory()
    value["steps"][0].update(response_message_index=-1, observation_message_index=999, tool_ok=False,
                              error_code="unknown_tool", parse_error=True, action=None)
    value["messages"].append({"role": "observation", "name": "runtime", "content": [{"type": "text", "text": "FATAL"}]})
    html = render_trajectory_html(value)
    assert payload(html)["records"][0]["variants"][0]["trajectory"] == value
    assert "value >= 0" in html  # Browser guards reject negative indexes, not Python-style indexing.


def test_null_trajectory_preserves_pipeline_diagnostics():
    record = payload(render_trajectory_html({"trajectory": None, "traj_rationale": "error: missing"}))["records"][0]
    assert record["variants"] == [] and record["judge"]["present"]


@pytest.mark.parametrize("data", [[], {}, {"messages": []}, "not a trajectory path"])
def test_bad_top_level_input_fails_clearly(data):
    with pytest.raises(ValueError):
        render_trajectory_html(data)


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_multi_record_export(tmp_path, suffix):
    source=tmp_path / ("input" + suffix)
    rows=[{"trajectory": trajectory()}, {"trajectory": json.dumps(trajectory()), "traj_overall": .5}]
    source.write_text(json.dumps(rows) if suffix == ".json" else "\n".join(json.dumps(r) for r in rows))
    output=tmp_path / "report.html"
    assert export_trajectory_html(source, output) == output
    assert len(payload(output.read_text())["records"]) == 2


def test_directory_sidecars_are_public_and_summary_ids_checked(tmp_path):
    (tmp_path/"trajectory.json").write_text(json.dumps(trajectory()))
    (tmp_path/"task.json").write_text(json.dumps({"env_id": "any_env", "task_id": "task-1", "messages": [], "scenario": "SECRET_INIT"}))
    (tmp_path/"summary.json").write_text(json.dumps({"task_id": "task-1", "model": "some-model", "replay_verification": {"status": "not_applicable"}}))
    output=export_trajectory_html(tmp_path,tmp_path/"view.html")
    record=payload(output.read_text())["records"][0]
    assert record["summary"]["model"] == "some-model"
    assert record["replay"]["status"] == "not_applicable"
    assert "SECRET_INIT" not in output.read_text()
    record=payload(render_trajectory_html(trajectory(),summary={"task_id":"another"}))["records"][0]
    assert record["summary"] is None and record["warnings"]


def test_explicit_task_store(tmp_path):
    tasks=tmp_path/"tasks"; tasks.mkdir()
    (tasks/"task-1.json").write_text(json.dumps({"schema_version":2,"env_id":"any_env","task_id":"task-1",
         "messages":[{"role":"user","content":[{"type":"text","text":"STORE TASK"}]}]}))
    source=tmp_path/"input.json"; source.write_text(json.dumps(trajectory()))
    output=export_trajectory_html(source,tmp_path/"view.html",tasks_dir=tasks)
    assert payload(output.read_text())["records"][0]["task"]["messages"][0]["content"][0]["text"] == "STORE TASK"


def test_never_follows_artifact_paths(tmp_path):
    secret=tmp_path/"private.txt"; secret.write_text("DO_NOT_READ_ME")
    value=trajectory(); value["steps"][0]["action"]["args"]["path"] = str(secret)
    assert "DO_NOT_READ_ME" not in render_trajectory_html(value)
    result = payload(render_trajectory_html({"trajectory": str(secret)}))["records"][0]
    assert result["variants"] == [] and result["warnings"]


def test_output_collision_force_and_input_protection(tmp_path):
    source=tmp_path/"input.json"; source.write_text(json.dumps(trajectory()))
    output=tmp_path/"view.html"; output.write_text("KEEP")
    with pytest.raises(FileExistsError): export_trajectory_html(source,output)
    assert output.read_text() == "KEEP"
    export_trajectory_html(source,output,force=True)
    assert output.read_text().startswith("<!doctype html>")
    with pytest.raises(ValueError): export_trajectory_html(source,source,force=True)
    link=tmp_path/"link.html"; link.symlink_to(output)
    with pytest.raises(ValueError): export_trajectory_html(source,link,force=True)
    assert not list(tmp_path.glob(".trajectory-html-*"))


def test_jsonl_line_error_and_input_size_limit(tmp_path,monkeypatch):
    source=tmp_path/"input.jsonl"; source.write_text(json.dumps(trajectory())+"\nBROKEN")
    with pytest.raises(ValueError,match="line 2"): export_trajectory_html(source,tmp_path/"x.html")
    monkeypatch.setattr("dataflow_mm_agent.visualization.report.MAX_INPUT_BYTES",1)
    with pytest.raises(ValueError,match="128 MiB"): export_trajectory_html(source,tmp_path/"x.html")


def test_cli_exports_and_reports_errors(tmp_path,capsys):
    source=tmp_path/"input.json"; source.write_text(json.dumps(trajectory()))
    output=tmp_path/"view.html"
    assert main([str(source),"-o",str(output),"--title","CLI REPORT"]) == 0
    assert str(output) in capsys.readouterr().out
    assert payload(output.read_text())["title"] == "CLI REPORT"
    with pytest.raises(SystemExit) as exc: main([str(source),"-o",str(output)])
    assert exc.value.code == 2
