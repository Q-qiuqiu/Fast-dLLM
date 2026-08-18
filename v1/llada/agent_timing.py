"""Durable, retry-aware Agent decision timing records for the LLaDA server."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


LOGGER = logging.getLogger("fastdllm.agent_timing")


class AgentTimingRecorder:
    """Keep one latest JSONL record per stable model/query request identity.

    ``request_index`` is only a stable display/order field. The upsert key is a
    hash of ``model`` and ``query`` because the server-side index restarts when
    the server restarts, while a resumed benchmark can skip successful cases.
    """

    schema_version = 2

    def __init__(self, log_path: str) -> None:
        if not log_path:
            raise ValueError("Agent timing log path cannot be empty.")
        self.log_path = Path(log_path).expanduser().resolve()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Fail at server startup rather than halfway through a benchmark when
        # the configured destination cannot be created.
        with self.log_path.open("a", encoding="utf-8"):
            pass

        self.session_id = f"agent-session-{uuid.uuid4().hex}"
        self.started_unix = time.time()
        self._lock = threading.Lock()
        self._migration_backup_path: Optional[str] = None
        self._records = self._load_canonical_records()

    @staticmethod
    def _query_metadata(query: str) -> Dict[str, str]:
        encoded = query.encode("utf-8")
        return {
            "query": query,
            "query_sha256": hashlib.sha256(encoded).hexdigest(),
        }

    @staticmethod
    def _request_key(model: str, query: str) -> str:
        identity = f"{model}\0{query}".encode("utf-8")
        return hashlib.sha256(identity).hexdigest()

    @classmethod
    def _record_key(cls, record: Dict[str, Any]) -> str:
        model = str(record.get("model") or "")
        query = record.get("query")
        if query is not None:
            return cls._request_key(model, str(query))
        # Legacy fallback only. New records always contain the original query.
        query_hash = str(record.get("query_sha256") or "")
        if not query_hash:
            raise ValueError("Timing record has neither query nor query_sha256.")
        return hashlib.sha256(
            f"legacy\0{model}\0{query_hash}".encode("utf-8")
        ).hexdigest()

    def _read_records_from_disk(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with self.log_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid timing JSONL at {self.log_path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Timing JSONL record at {self.log_path}:{line_number} "
                        "must be a JSON object."
                    )
                records.append(value)
        return records

    def _atomic_write_log(self, records: List[Dict[str, Any]]) -> None:
        temporary = self.log_path.with_suffix(self.log_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary, self.log_path)

    def _backup_legacy_log(self) -> Path:
        suffix = self.session_id.rsplit("-", 1)[-1][:12]
        backup = self.log_path.with_name(
            f"{self.log_path.name}.precanonical.{suffix}.bak"
        )
        shutil.copy2(self.log_path, backup)
        self._migration_backup_path = str(backup)
        return backup

    def _load_canonical_records(self) -> Dict[str, Dict[str, Any]]:
        raw_records = self._read_records_from_disk()
        canonical: Dict[str, Dict[str, Any]] = {}
        migration_needed = False

        for raw in raw_records:
            record = dict(raw)
            key = self._record_key(record)
            previous = canonical.get(key)
            if previous is None:
                attempt_count = max(1, int(record.get("attempt_count") or 1))
                first_created = record.get("first_created_unix")
                if first_created is None:
                    first_created = record.get("created_unix")
            else:
                migration_needed = True
                previous_attempts = max(
                    1, int(previous.get("attempt_count") or 1)
                )
                attempt_count = max(
                    previous_attempts + 1,
                    int(record.get("attempt_count") or 1),
                )
                first_created = previous.get(
                    "first_created_unix", previous.get("created_unix")
                )

            expected_index = (
                int(previous["request_index"])
                if previous is not None
                else len(canonical) + 1
            )
            if (
                record.get("schema_version") != self.schema_version
                or record.get("request_key") != key
                or record.get("request_index") != expected_index
                or record.get("attempt_count") != attempt_count
                or record.get("first_created_unix") != first_created
            ):
                migration_needed = True

            record["schema_version"] = self.schema_version
            record["request_key"] = key
            record["request_index"] = expected_index
            record["attempt_count"] = attempt_count
            record["first_created_unix"] = first_created
            canonical[key] = record

        if migration_needed and raw_records:
            backup = self._backup_legacy_log()
            self._atomic_write_log(list(canonical.values()))
            LOGGER.info(
                "agent_timing_migrated input_records=%d canonical_records=%d "
                "backup=%s",
                len(raw_records),
                len(canonical),
                backup,
            )
        return canonical

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
                    "decision_seconds": decision,
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
        request_key = self._request_key(model, query)
        record = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "request_key": request_key,
            "request_index": None,
            "attempt_count": None,
            "first_created_unix": None,
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
            previous = self._records.get(request_key)
            if previous is None:
                record["request_index"] = len(self._records) + 1
                record["attempt_count"] = 1
                record["first_created_unix"] = int(created_unix)
            else:
                record["request_index"] = previous["request_index"]
                record["attempt_count"] = max(
                    1, int(previous.get("attempt_count") or 1)
                ) + 1
                record["first_created_unix"] = previous.get(
                    "first_created_unix", previous.get("created_unix")
                )
            self._records[request_key] = record
            self._atomic_write_log(list(self._records.values()))

        LOGGER.info(
            "agent_timing_saved request=%s attempt=%s updated=%s "
            "decisions=%d confirmations=%d all_decided_seconds=%s",
            record["request_index"],
            record["attempt_count"],
            previous is not None,
            len(decisions),
            len(confirmations),
            all_decided,
        )
        return record
