"""Durable per-request Agent decision timing records for the LLaDA server."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


LOGGER = logging.getLogger("fastdllm.agent_timing")


class _RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.minimum: Optional[float] = None
        self.maximum: Optional[float] = None

    def add(self, value: Optional[float]) -> None:
        if value is None:
            return
        number = float(value)
        self.count += 1
        self.total += number
        self.minimum = number if self.minimum is None else min(self.minimum, number)
        self.maximum = number if self.maximum is None else max(self.maximum, number)

    def render(self) -> Dict[str, Optional[float]]:
        return {
            "count": self.count,
            "sum_seconds": self.total,
            "mean_seconds": (
                self.total / self.count if self.count else None
            ),
            "min_seconds": self.minimum,
            "max_seconds": self.maximum,
        }


class AgentTimingRecorder:
    """Append one JSONL record per API request and maintain session aggregates."""

    schema_version = 1

    def __init__(
        self,
        log_path: str,
        summary_path: Optional[str] = None,
    ) -> None:
        if not log_path:
            raise ValueError("Agent timing log path cannot be empty.")
        self.log_path = Path(log_path).expanduser().resolve()
        self.summary_path = (
            Path(summary_path).expanduser().resolve()
            if summary_path
            else self.log_path.with_suffix(".summary.json")
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        # Fail at server startup, rather than halfway through a benchmark, if
        # the configured record destination is not writable.
        with self.log_path.open("a", encoding="utf-8"):
            pass

        self.session_id = f"agent-session-{uuid.uuid4().hex}"
        self.started_unix = time.time()
        self._lock = threading.Lock()
        self._request_index = 0
        self._requests_successful = 0
        self._requests_failed = 0
        self._requests_with_decisions = 0
        self._requests_with_confirmations = 0
        self._requests_repaired = 0
        self._repair_failures = 0
        self._agent_decisions = 0
        self._agent_confirmations = 0
        self._agent_unconfirmed = 0
        self._decision_stats = _RunningStats()
        self._confirmation_stats = _RunningStats()
        self._all_decided_stats = _RunningStats()
        self._all_confirmed_stats = _RunningStats()
        self._generation_stats = _RunningStats()
        self._write_summary()

    @staticmethod
    def _query_metadata(query: str) -> Dict[str, str]:
        encoded = query.encode("utf-8")
        return {
            "query": query,
            "query_sha256": hashlib.sha256(encoded).hexdigest(),
        }

    def _summary(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "session_started_unix": self.started_unix,
            "updated_unix": time.time(),
            "timing_log_path": str(self.log_path),
            "requests": {
                "total": self._request_index,
                "successful": self._requests_successful,
                "failed": self._requests_failed,
                "with_agent_decisions": self._requests_with_decisions,
                "with_agent_confirmations": self._requests_with_confirmations,
                "plan_json_repaired": self._requests_repaired,
                "plan_json_repair_failed": self._repair_failures,
            },
            "agents": {
                "decided": self._agent_decisions,
                "confirmed": self._agent_confirmations,
                "decided_but_unconfirmed": self._agent_unconfirmed,
            },
            # Agent-level averages weight every (non-deduplicated) call equally.
            "agent_decision_seconds": self._decision_stats.render(),
            "agent_confirmation_seconds": self._confirmation_stats.render(),
            # Request-level values are the time at which the last recorded
            # Agent in that request was decided/confirmed.
            "all_agents_decided_seconds_per_request": self._all_decided_stats.render(),
            "all_agents_confirmed_seconds_per_request": self._all_confirmed_stats.render(),
            "generation_seconds": self._generation_stats.render(),
        }

    def _write_summary(self) -> None:
        temporary = self.summary_path.with_suffix(
            self.summary_path.suffix + ".tmp"
        )
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(self._summary(), file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary, self.summary_path)

    def record(
        self,
        *,
        completion_id: str,
        created_unix: int,
        query: str,
        model: str,
        temperature: float,
        requested_max_tokens: Optional[int],
        metrics: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        metrics = metrics or {}
        priority = metrics.get("agent_priority") or {}
        agents = []
        for slot in priority.get("agent_slots") or []:
            name = slot.get("agent")
            decision = slot.get("recognized_seconds")
            confirmation = slot.get("confirmed_seconds")
            if name is None and decision is None and confirmation is None:
                continue
            agents.append(
                {
                    "slot": slot.get("slot"),
                    "agent": name,
                    "priority": bool(slot.get("priority")),
                    # Decision is when the name first satisfies the configured
                    # probability/margin gates and can trigger prefetch.
                    "decision_seconds": decision,
                    # Confirmation additionally requires a naturally emitted
                    # JSON anchor and is the stricter post-verification time.
                    "confirmation_seconds": confirmation,
                    "decision_step": slot.get("recognized_step"),
                    "confirmation_step": slot.get("confirmed_step"),
                    "probability": slot.get("probability"),
                    "margin": slot.get("margin"),
                    "confirmed": bool(slot.get("confirmed")),
                    "fuzzy_matched_from": slot.get("fuzzy_matched_from"),
                }
            )

        decisions = [
            float(item["decision_seconds"])
            for item in agents
            if item["decision_seconds"] is not None
        ]
        confirmations = [
            float(item["confirmation_seconds"])
            for item in agents
            if item["confirmation_seconds"] is not None
        ]
        all_decided = max(decisions) if decisions else None
        all_confirmed = max(confirmations) if confirmations else None
        repair = metrics.get("plan_json_repair") or None
        record = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "request_index": None,
            "completion_id": completion_id,
            "created_unix": int(created_unix),
            **self._query_metadata(query),
            "model": model,
            "temperature": float(temperature),
            "requested_max_tokens": requested_max_tokens,
            "status": "error" if error else "ok",
            "error": error,
            "agent_registry": priority.get("catalog"),
            "priority_slots": priority.get("priority_slots"),
            "tracking_slots": priority.get("tracking_slots"),
            "agents": agents,
            "agent_decision_count": len(decisions),
            "agent_confirmation_count": len(confirmations),
            "all_agents_decided_seconds": all_decided,
            "all_agents_confirmed_seconds": all_confirmed,
            "generation_seconds": metrics.get("generation_seconds"),
            "generated_tokens": metrics.get("generated_tokens"),
            "returned_tokens": metrics.get("returned_tokens"),
            "tps": metrics.get("tps"),
            "nfe": metrics.get("nfe"),
            "plan_json_repair": repair,
        }

        with self._lock:
            self._request_index += 1
            record["request_index"] = self._request_index
            if error:
                self._requests_failed += 1
            else:
                self._requests_successful += 1
            if decisions:
                self._requests_with_decisions += 1
            if confirmations:
                self._requests_with_confirmations += 1
            if repair and repair.get("applied"):
                self._requests_repaired += 1
            if repair and repair.get("method") == "failed":
                self._repair_failures += 1
            self._agent_decisions += len(decisions)
            self._agent_confirmations += len(confirmations)
            self._agent_unconfirmed += sum(
                item["decision_seconds"] is not None
                and item["confirmation_seconds"] is None
                for item in agents
            )
            for value in decisions:
                self._decision_stats.add(value)
            for value in confirmations:
                self._confirmation_stats.add(value)
            self._all_decided_stats.add(all_decided)
            self._all_confirmed_stats.add(all_confirmed)
            self._generation_stats.add(metrics.get("generation_seconds"))

            with self.log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._write_summary()

        LOGGER.info(
            "agent_timing_saved request=%s decisions=%d confirmations=%d "
            "all_decided_seconds=%s",
            record["request_index"],
            len(decisions),
            len(confirmations),
            all_decided,
        )
        return record
