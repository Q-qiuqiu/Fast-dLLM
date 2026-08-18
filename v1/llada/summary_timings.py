#!/usr/bin/env python3
"""Build an aggregate summary from an Agent timing JSONL file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class _NumericStats:
    def __init__(self) -> None:
        self.values: List[float] = []

    def add(self, value: Optional[float]) -> None:
        if value is not None:
            self.values.append(float(value))

    def render(self, unit: str) -> Dict[str, Optional[float]]:
        total = sum(self.values)
        return {
            "count": len(self.values),
            f"sum_{unit}": total,
            f"mean_{unit}": total / len(self.values) if self.values else None,
            f"min_{unit}": min(self.values) if self.values else None,
            f"max_{unit}": max(self.values) if self.values else None,
        }


def _derived_request_key(record: Dict[str, Any]) -> str:
    existing = record.get("request_key")
    if existing:
        return str(existing)
    model = str(record.get("model") or "")
    query = record.get("query")
    if query is not None:
        identity = f"{model}\0{query}"
    else:
        query_hash = record.get("query_sha256")
        if not query_hash:
            raise ValueError("Timing record has neither query nor query_sha256.")
        identity = f"legacy\0{model}\0{query_hash}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Record at {path}:{line_number} must be an object.")
            records.append(value)
    return records


def canonical_latest(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the latest attempt for each model/query identity in first-seen order."""

    canonical: Dict[str, Dict[str, Any]] = {}
    for source in records:
        record = dict(source)
        key = _derived_request_key(record)
        previous = canonical.get(key)
        if previous is not None:
            previous_attempts = max(1, int(previous.get("attempt_count") or 1))
            record["attempt_count"] = max(
                previous_attempts + 1,
                int(record.get("attempt_count") or 1),
            )
        else:
            record["attempt_count"] = max(
                1, int(record.get("attempt_count") or 1)
            )
        canonical[key] = record
    return list(canonical.values())


def summarize_records(
    records: List[Dict[str, Any]], *, input_path: Optional[Path] = None
) -> Dict[str, Any]:
    canonical = canonical_latest(records)
    decision_stats = _NumericStats()
    confirmation_stats = _NumericStats()
    all_decided_stats = _NumericStats()
    all_confirmed_stats = _NumericStats()
    generation_stats = _NumericStats()
    generated_token_stats = _NumericStats()
    returned_token_stats = _NumericStats()
    tps_stats = _NumericStats()
    nfe_stats = _NumericStats()

    successful = 0
    failed = 0
    with_decisions = 0
    with_confirmations = 0
    repaired = 0
    repair_failures = 0
    agent_decisions = 0
    agent_confirmations = 0
    agent_unconfirmed = 0

    for record in canonical:
        if record.get("status") == "error":
            failed += 1
        else:
            successful += 1
        agents = record.get("agents") or []
        decisions = [
            float(item["decision_seconds"])
            for item in agents
            if item.get("decision_seconds") is not None
        ]
        confirmations = [
            float(item["confirmation_seconds"])
            for item in agents
            if item.get("confirmation_seconds") is not None
        ]
        if decisions:
            with_decisions += 1
        if confirmations:
            with_confirmations += 1
        agent_decisions += len(decisions)
        agent_confirmations += len(confirmations)
        agent_unconfirmed += sum(
            item.get("decision_seconds") is not None
            and item.get("confirmation_seconds") is None
            for item in agents
        )
        for value in decisions:
            decision_stats.add(value)
        for value in confirmations:
            confirmation_stats.add(value)
        all_decided_stats.add(record.get("all_agents_decided_seconds"))
        all_confirmed_stats.add(record.get("all_agents_confirmed_seconds"))
        generation_stats.add(record.get("generation_seconds"))
        generated_token_stats.add(record.get("generated_tokens"))
        returned_token_stats.add(record.get("returned_tokens"))
        tps_stats.add(record.get("tps"))
        nfe_stats.add(record.get("nfe"))

        repair = record.get("plan_json_repair") or {}
        if repair.get("applied"):
            repaired += 1
        if repair.get("method") == "failed":
            repair_failures += 1

    attempt_counts = [
        max(1, int(record.get("attempt_count") or 1)) for record in canonical
    ]
    source_sessions = {
        record.get("session_id")
        for record in canonical
        if record.get("session_id")
    }
    return {
        "schema_version": 2,
        "summary_scope": "canonical_latest_record_per_request_key",
        "generated_unix": time.time(),
        "timing_log_path": str(input_path.resolve()) if input_path else None,
        "input_record_count": len(records),
        "source_session_count": len(source_sessions),
        "requests": {
            "total": len(canonical),
            "successful": successful,
            "failed": failed,
            "with_agent_decisions": with_decisions,
            "with_agent_confirmations": with_confirmations,
            "plan_json_repaired": repaired,
            "plan_json_repair_failed": repair_failures,
            "updated_records": sum(count > 1 for count in attempt_counts),
            "retry_attempts": sum(count - 1 for count in attempt_counts),
        },
        "agents": {
            "decided": agent_decisions,
            "confirmed": agent_confirmations,
            "decided_but_unconfirmed": agent_unconfirmed,
        },
        "agent_decision_seconds": decision_stats.render("seconds"),
        "agent_confirmation_seconds": confirmation_stats.render("seconds"),
        "all_agents_decided_seconds_per_request": all_decided_stats.render(
            "seconds"
        ),
        "all_agents_confirmed_seconds_per_request": all_confirmed_stats.render(
            "seconds"
        ),
        "generation_seconds": generation_stats.render("seconds"),
        "generated_tokens": generated_token_stats.render("tokens"),
        "returned_tokens": returned_token_stats.render("tokens"),
        "tps": tps_stats.render("tps"),
        "nfe": nfe_stats.render("nfe"),
    }


def write_summary(input_path: Path, output_path: Path) -> Dict[str, Any]:
    summary = summarize_records(read_jsonl(input_path), input_path=input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, output_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read Agent timing JSONL and write an aggregate JSON summary."
    )
    parser.add_argument("timings_jsonl", type=Path, help="Input timing JSONL file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <input stem>_summary.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.timings_jsonl.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_summary.json")
    )
    summary = write_summary(input_path, output_path)
    requests = summary["requests"]
    print(
        f"summary={output_path} requests={requests['total']} "
        f"successful={requests['successful']} failed={requests['failed']} "
        f"retry_attempts={requests['retry_attempts']}"
    )


if __name__ == "__main__":
    main()
