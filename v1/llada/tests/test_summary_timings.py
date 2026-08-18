import json

import pytest

from summary_timings import summarize_records, write_summary


def _timing_record(
    query,
    *,
    status="ok",
    attempt_count=1,
    session_id="session-a",
    decision=None,
    confirmation=None,
    generation_seconds=None,
    generated_tokens=None,
    tps=None,
    nfe=None,
):
    agents = []
    if decision is not None or confirmation is not None:
        agents.append(
            {
                "agent": "search_agent",
                "decision_seconds": decision,
                "confirmation_seconds": confirmation,
            }
        )
    return {
        "query": query,
        "model": "llada",
        "session_id": session_id,
        "status": status,
        "attempt_count": attempt_count,
        "agents": agents,
        "all_agents_decided_seconds": decision,
        "all_agents_confirmed_seconds": confirmation,
        "generation_seconds": generation_seconds,
        "generated_tokens": generated_tokens,
        "returned_tokens": generated_tokens,
        "tps": tps,
        "nfe": nfe,
    }


def test_summarizes_latest_canonical_records_and_metrics():
    records = [
        _timing_record(
            "one",
            decision=1.0,
            confirmation=1.5,
            generation_seconds=10.0,
            generated_tokens=200,
            tps=20.0,
            nfe=100,
        ),
        _timing_record("two", status="error"),
    ]
    summary = summarize_records(records)

    assert summary["requests"]["total"] == 2
    assert summary["requests"]["successful"] == 1
    assert summary["requests"]["failed"] == 1
    assert summary["agents"]["decided"] == 1
    assert summary["agent_decision_seconds"]["mean_seconds"] == 1.0
    assert summary["agent_confirmation_seconds"]["mean_seconds"] == 1.5
    assert summary["generation_seconds"]["mean_seconds"] == 10.0
    assert summary["generated_tokens"]["sum_tokens"] == 200.0
    assert summary["tps"]["mean_tps"] == 20.0
    assert summary["nfe"]["sum_nfe"] == 100.0


def test_deduplicates_legacy_retries_using_latest_model_query_record():
    records = [
        _timing_record("one", status="error", session_id="session-a"),
        _timing_record("two", session_id="session-a"),
        _timing_record(
            "one",
            status="ok",
            session_id="session-b",
            decision=2.0,
        ),
    ]
    summary = summarize_records(records)

    assert summary["input_record_count"] == 3
    assert summary["requests"]["total"] == 2
    assert summary["requests"]["successful"] == 2
    assert summary["requests"]["failed"] == 0
    assert summary["requests"]["updated_records"] == 1
    assert summary["requests"]["retry_attempts"] == 1
    assert summary["source_session_count"] == 2
    assert summary["agent_decision_seconds"]["mean_seconds"] == 2.0


def test_writes_summary_to_requested_path(tmp_path):
    input_path = tmp_path / "huskyqa_timings.jsonl"
    output_path = tmp_path / "huskyqa_timings_summary.json"
    input_path.write_text(
        json.dumps(_timing_record("one", generation_seconds=4.0)) + "\n",
        encoding="utf-8",
    )

    returned = write_summary(input_path, output_path)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == returned
    assert written["timing_log_path"] == str(input_path.resolve())
    assert written["generation_seconds"]["mean_seconds"] == pytest.approx(4.0)
