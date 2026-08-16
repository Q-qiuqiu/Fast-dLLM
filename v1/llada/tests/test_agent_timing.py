import json

import pytest

from agent_timing import AgentTimingRecorder


def test_records_every_request_and_updates_current_session_summary(tmp_path):
    log_path = tmp_path / "agent_timings.jsonl"
    recorder = AgentTimingRecorder(str(log_path))
    metrics = {
        "generation_seconds": 10.0,
        "generated_tokens": 200,
        "tps": 20.0,
        "nfe": 100,
        "plan_json_repair": {
            "applied": True,
            "method": "minimal_syntax",
            "operations": ["close_object_and_add_comma:1"],
        },
        "agent_priority": {
            "catalog": ["search_agent", "math_agent"],
            "priority_slots": 4,
            "tracking_slots": 32,
            "agent_slots": [
                {
                    "slot": 0,
                    "priority": True,
                    "agent": "search_agent",
                    "recognized_seconds": 1.0,
                    "confirmed_seconds": 1.5,
                    "recognized_step": 10,
                    "confirmed_step": 12,
                    "probability": 0.9,
                    "margin": 0.8,
                    "confirmed": True,
                    "fuzzy_matched_from": None,
                },
                {
                    "slot": 1,
                    "priority": True,
                    "agent": "math_agent",
                    "recognized_seconds": 3.0,
                    "confirmed_seconds": None,
                    "recognized_step": 30,
                    "confirmed_step": None,
                    "probability": 0.8,
                    "margin": 0.6,
                    "confirmed": False,
                    "fuzzy_matched_from": None,
                },
                {"slot": 2, "agent": None},
            ],
        },
    }
    first = recorder.record(
        completion_id="chatcmpl-ok",
        created_unix=100,
        query="question one",
        model="llada",
        temperature=0.0,
        requested_max_tokens=1024,
        metrics=metrics,
    )
    recorder.record(
        completion_id="chatcmpl-error",
        created_unix=101,
        query="question two",
        model="llada",
        temperature=0.0,
        requested_max_tokens=1024,
        error="generation failed",
    )

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0] == first
    assert records[0]["request_index"] == 1
    assert records[0]["agent_decision_count"] == 2
    assert records[0]["agent_confirmation_count"] == 1
    assert records[0]["priority_slots"] == 4
    assert records[0]["tracking_slots"] == 32
    assert all(item["priority"] for item in records[0]["agents"])
    assert records[0]["all_agents_decided_seconds"] == 3.0
    assert records[0]["all_agents_confirmed_seconds"] == 1.5
    assert records[1]["request_index"] == 2
    assert records[1]["status"] == "error"
    assert records[0]["session_id"] == records[1]["session_id"]

    summary_path = tmp_path / "agent_timings.summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["session_id"] == recorder.session_id
    assert summary["requests"] == {
        "total": 2,
        "successful": 1,
        "failed": 1,
        "with_agent_decisions": 1,
        "with_agent_confirmations": 1,
        "plan_json_repaired": 1,
        "plan_json_repair_failed": 0,
    }
    assert summary["agents"] == {
        "decided": 2,
        "confirmed": 1,
        "decided_but_unconfirmed": 1,
    }
    assert summary["agent_decision_seconds"]["mean_seconds"] == pytest.approx(2.0)
    assert summary["agent_confirmation_seconds"]["mean_seconds"] == pytest.approx(1.5)
    assert summary["all_agents_decided_seconds_per_request"]["mean_seconds"] == 3.0
    assert summary["generation_seconds"]["mean_seconds"] == 10.0


def test_new_server_session_appends_jsonl_but_restarts_summary(tmp_path):
    log_path = tmp_path / "timings.jsonl"
    first = AgentTimingRecorder(str(log_path))
    first.record(
        completion_id="one",
        created_unix=1,
        query="one",
        model="llada",
        temperature=0.0,
        requested_max_tokens=None,
    )
    second = AgentTimingRecorder(str(log_path))
    second.record(
        completion_id="two",
        created_unix=2,
        query="two",
        model="llada",
        temperature=0.0,
        requested_max_tokens=None,
    )

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["session_id"] != records[1]["session_id"]
    summary = json.loads((tmp_path / "timings.summary.json").read_text())
    assert summary["session_id"] == second.session_id
    assert summary["requests"]["total"] == 1
