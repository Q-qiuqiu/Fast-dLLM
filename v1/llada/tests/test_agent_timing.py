import json

from agent_timing import AgentTimingRecorder


def _record(recorder, completion_id, created_unix, query, **kwargs):
    return recorder.record(
        completion_id=completion_id,
        created_unix=created_unix,
        query=query,
        model="llada",
        temperature=0.0,
        requested_max_tokens=1024,
        **kwargs,
    )


def test_records_requests_in_canonical_log_without_writing_summary(tmp_path):
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
    first = _record(recorder, "chatcmpl-ok", 100, "question one", metrics=metrics)
    _record(
        recorder,
        "chatcmpl-error",
        101,
        "question two",
        error="generation failed",
    )

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0] == first
    assert records[0]["request_index"] == 1
    assert records[0]["attempt_count"] == 1
    assert records[0]["agent_decision_count"] == 2
    assert records[0]["agent_confirmation_count"] == 1
    assert records[0]["priority_slots"] == 4
    assert records[0]["tracking_slots"] == 32
    assert all(item["priority"] for item in records[0]["agents"])
    assert records[0]["all_agents_decided_seconds"] == 3.0
    assert records[0]["all_agents_confirmed_seconds"] == 1.5
    assert records[1]["request_index"] == 2
    assert records[1]["status"] == "error"

    assert not (tmp_path / "agent_timings.summary.json").exists()


def test_restart_retry_upserts_by_model_and_query_not_request_index(tmp_path):
    log_path = tmp_path / "timings.jsonl"
    first = AgentTimingRecorder(str(log_path))
    _record(first, "one-error", 1, "query one", error="first attempt failed")
    _record(first, "two-ok", 2, "query two")

    second = AgentTimingRecorder(str(log_path))
    retried = _record(second, "one-ok", 3, "query one")
    records = [json.loads(line) for line in log_path.read_text().splitlines()]

    assert len(records) == 2
    assert retried["request_index"] == 1
    assert retried["attempt_count"] == 2
    assert retried["first_created_unix"] == 1
    assert records[0]["completion_id"] == "one-ok"
    assert records[0]["status"] == "ok"
    assert records[1]["completion_id"] == "two-ok"
    assert records[0]["session_id"] == second.session_id
    assert records[1]["session_id"] == first.session_id

    assert not (tmp_path / "timings.summary.json").exists()


def test_same_query_under_different_policies_keeps_both_records(tmp_path):
    log_path = tmp_path / "policy_timings.jsonl"
    recorder = AgentTimingRecorder(str(log_path))
    _record(
        recorder,
        "planreason",
        1,
        "same query",
        metrics={"agent_priority": {"policy": "planreason"}},
    )
    _record(
        recorder,
        "reasonplan",
        2,
        "same query",
        metrics={"agent_priority": {"policy": "reasonplan"}},
    )

    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(records) == 2
    assert [record["policy"] for record in records] == [
        "planreason",
        "reasonplan",
    ]
    assert records[0]["request_key"] != records[1]["request_key"]
    assert [record["request_index"] for record in records] == [1, 2]


def test_legacy_append_log_is_backed_up_deduplicated_and_reindexed(tmp_path):
    log_path = tmp_path / "legacy.jsonl"
    legacy = [
        {
            "schema_version": 1,
            "session_id": "old-a",
            "request_index": 1,
            "completion_id": "a-error",
            "created_unix": 10,
            "query": "query a",
            "model": "llada",
            "status": "error",
            "agents": [],
        },
        {
            "schema_version": 1,
            "session_id": "old-a",
            "request_index": 2,
            "completion_id": "b-ok",
            "created_unix": 11,
            "query": "query b",
            "model": "llada",
            "status": "ok",
            "agents": [],
        },
        {
            "schema_version": 1,
            "session_id": "old-b",
            "request_index": 1,
            "completion_id": "a-ok",
            "created_unix": 12,
            "query": "query a",
            "model": "llada",
            "status": "ok",
            "agents": [],
        },
        {
            "schema_version": 1,
            "session_id": "old-b",
            "request_index": 2,
            "completion_id": "c-ok",
            "created_unix": 13,
            "query": "query c",
            "model": "llada",
            "status": "ok",
            "agents": [],
        },
    ]
    log_path.write_text(
        "".join(json.dumps(record) + "\n" for record in legacy),
        encoding="utf-8",
    )

    recorder = AgentTimingRecorder(str(log_path))
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [record["completion_id"] for record in records] == [
        "a-ok",
        "b-ok",
        "c-ok",
    ]
    assert [record["request_index"] for record in records] == [1, 2, 3]
    assert records[0]["attempt_count"] == 2
    assert records[0]["first_created_unix"] == 10
    assert records[0]["status"] == "ok"

    backups = list(tmp_path.glob("legacy.jsonl.precanonical.*.bak"))
    assert len(backups) == 1
    assert len(backups[0].read_text().splitlines()) == 4
    assert recorder._migration_backup_path == str(backups[0])
