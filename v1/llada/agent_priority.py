"""Agent-name-first decoding for Fast-dLLM's LLaDA sampler.

The controller in this module is deliberately independent from the model and
cache implementations.  It owns a compact routing region in the first
generation block, scores complete catalog names, manages per-slot state, and
leaves task tokens to the normal Fast-dLLM transfer policy.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


LOGGER = logging.getLogger("fastdllm.agent_priority")
MAX_AGENT_SLOTS = 4


def configure_agent_file_logging(
    path: Optional[str],
    level: int = logging.INFO,
    max_bytes: int = 20 * 1024 * 1024,
    backup_count: int = 3,
) -> Optional[Path]:
    """Configure rotating Agent logs, or disable them when ``path`` is empty."""

    for handler in list(LOGGER.handlers):
        if getattr(handler, "_fastdllm_agent_file", False):
            LOGGER.removeHandler(handler)
            handler.close()
    if not path:
        LOGGER.setLevel(logging.NOTSET)
        LOGGER.propagate = True
        return None

    log_path = Path(path).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler._fastdllm_agent_file = True
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    LOGGER.addHandler(handler)
    LOGGER.setLevel(level)
    LOGGER.propagate = False
    return log_path


class AgentFieldState(str, Enum):
    MASKED = "MASKED"
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"


class AgentEventType(str, Enum):
    PRELOAD_START = "PRELOAD_START"
    PRELOAD_CANCEL = "PRELOAD_CANCEL"
    PRELOAD_SWITCH = "PRELOAD_SWITCH"
    AGENT_CONFIRMED = "AGENT_CONFIRMED"


@dataclass(frozen=True)
class AgentSpec:
    """One registered agent and the latency properties of its bound model."""

    name: str
    cold_start_seconds: float = 0.0
    wrong_preload_cost_seconds: float = 0.0
    resident: bool = False

    def __post_init__(self) -> None:
        if not self.name or self.name == "none":
            raise ValueError("Agent names must be non-empty; 'none' is reserved.")
        if self.cold_start_seconds < 0 or self.wrong_preload_cost_seconds < 0:
            raise ValueError("Agent latency values must be non-negative.")


@dataclass
class AgentPriorityConfig:
    """Configuration for catalog-constrained agent decoding.

    Passing no controller to ``generate*`` is the off switch and preserves the
    original decoder.  ``enabled`` provides a second explicit switch for callers
    that construct configuration dynamically.
    """

    catalog: Sequence[AgentSpec]
    enabled: bool = True
    slots: int = MAX_AGENT_SLOTS
    tentative_probability: float = 0.45
    tentative_margin: float = 0.10
    confirm_probability: float = 0.72
    confirm_margin: float = 0.25
    max_distribution_change: float = 0.08
    stable_steps: int = 2
    plateau_confirm_probability: float = 0.52
    plateau_confirm_margin: float = 0.20
    plateau_max_distribution_change: float = 0.02
    plateau_stable_steps: int = 4
    tentative_cancel_steps: int = 2
    min_preload_benefit_seconds: float = 0.0
    expected_step_seconds: float = 0.05
    step_time_ema_alpha: float = 0.20
    max_observed_step_multiplier: float = 4.0
    benefit_priority_weight: float = 0.0
    remask_task_on_switch: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.slots <= MAX_AGENT_SLOTS:
            raise ValueError(f"slots must be between 1 and {MAX_AGENT_SLOTS}.")
        names = [spec.name for spec in self.catalog]
        if len(names) != len(set(names)):
            raise ValueError("Agent Catalog contains duplicate names.")
        if not names:
            raise ValueError("Agent Catalog cannot be empty.")
        for value in (
            self.tentative_probability,
            self.confirm_probability,
            self.plateau_confirm_probability,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Probability thresholds must be in [0, 1].")
        if self.stable_steps < 1 or self.plateau_stable_steps < 1:
            raise ValueError("Stability step counts must be at least one.")
        if self.tentative_cancel_steps < 1:
            raise ValueError("tentative_cancel_steps must be at least one.")
        if self.expected_step_seconds <= 0 or self.max_observed_step_multiplier < 1:
            raise ValueError("Step timing configuration must be positive.")
        if not 0.0 < self.step_time_ema_alpha <= 1.0:
            raise ValueError("step_time_ema_alpha must be in (0, 1].")


@dataclass(frozen=True)
class AgentEvent:
    event_type: AgentEventType
    slot: int
    agent_name: Optional[str] = None
    previous_agent_name: Optional[str] = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class AsyncEventDispatcher:
    """Run preload callbacks away from the main decoding thread."""

    def __init__(
        self,
        callback: Optional[Callable[[AgentEvent], None]] = None,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self._callback = callback
        self._logger = logger
        self._queue: "queue.Queue[Optional[AgentEvent]]" = queue.Queue()
        self._worker = threading.Thread(
            target=self._run,
            name="fastdllm-agent-events",
            daemon=True,
        )
        self._worker.start()

    def emit(self, event: AgentEvent) -> None:
        self._queue.put_nowait(event)

    def _run(self) -> None:
        while True:
            event = self._queue.get()
            try:
                if event is None:
                    return
                self._logger.info(
                    "agent_event %s",
                    json.dumps(
                        {
                            "event": event.event_type.value,
                            "slot": event.slot,
                            "agent_name": event.agent_name,
                            "previous_agent_name": event.previous_agent_name,
                            "metadata": dict(event.metadata),
                        },
                        sort_keys=True,
                    ),
                )
                if self._callback is not None:
                    self._callback(event)
            except Exception:  # callbacks must never terminate decoding/event delivery
                self._logger.exception("Agent preload event callback failed")
            finally:
                self._queue.task_done()

    def drain(self) -> None:
        """Wait for callbacks; intended for tests and orderly process shutdown."""

        self._queue.join()

    def close(self, wait: bool = False) -> None:
        self._queue.put_nowait(None)
        if wait:
            self._worker.join()


@dataclass(frozen=True)
class AgentLayout:
    """Token positions for the compact route and four task regions."""

    prompt_length: int
    gen_length: int
    block_length: int
    agent_width: int
    agent_spans: Tuple[Tuple[int, int], ...]
    task_spans: Tuple[Tuple[int, int], ...]
    fixed_tokens: Mapping[int, int]
    filler_token_id: int

    @property
    def route_length(self) -> int:
        return self.task_spans[0][0] - self.prompt_length


@dataclass
class SlotRuntime:
    state: AgentFieldState = AgentFieldState.MASKED
    candidate: Optional[str] = None
    previous_distribution: Optional[Dict[str, float]] = None
    last_distribution: Optional[Dict[str, float]] = None
    top_probability: float = 0.0
    margin: float = 0.0
    distribution_change: float = 1.0
    consistent_steps: int = 0
    unreliable_steps: int = 0
    preload_agent: Optional[str] = None


@dataclass(frozen=True)
class AgentPlan:
    agents: Tuple[str, ...]
    tasks: Tuple[str, ...]

    def render(self) -> str:
        chunks = []
        for agent_name, task in zip(self.agents, self.tasks):
            chunks.append(
                "<subtask>\n"
                f"agent_name: {agent_name}\n"
                f"task: {task}\n"
                "</subtask>"
            )
        return "\n\n".join(chunks)


def _tokenize(tokenizer, text: str) -> List[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return list(ids)


def _choose_filler_token(tokenizer, mask_id: int) -> int:
    for attribute in ("pad_token_id", "eos_token_id", "bos_token_id"):
        token_id = getattr(tokenizer, attribute, None)
        if token_id is not None and token_id != mask_id:
            return int(token_id)
    return 0 if mask_id != 0 else 1


def build_agent_layout(
    tokenizer,
    config: AgentPriorityConfig,
    prompt_length: int,
    gen_length: int,
    block_length: int,
    mask_id: int,
) -> Tuple[AgentLayout, Dict[str, Tuple[int, ...]]]:
    """Put all catalog fields in block zero and divide the rest among tasks."""

    names = [spec.name for spec in config.catalog] + ["none"]
    token_ids = {name: tuple(_tokenize(tokenizer, name)) for name in names}
    if any(not ids for ids in token_ids.values()):
        raise ValueError("Every Agent Catalog name must tokenize to at least one token.")
    width = max(len(ids) for ids in token_ids.values())
    prefix = _tokenize(tokenizer, "<agents>")
    separator = _tokenize(tokenizer, "|")
    suffix = _tokenize(tokenizer, "</agents>")
    if not separator:
        raise ValueError("Tokenizer produced no token for the route separator.")

    local = 0
    fixed_tokens: Dict[int, int] = {}
    for token_id in prefix:
        fixed_tokens[prompt_length + local] = token_id
        local += 1

    agent_spans = []
    for slot in range(config.slots):
        start = prompt_length + local
        local += width
        agent_spans.append((start, prompt_length + local))
        if slot != config.slots - 1:
            for token_id in separator:
                fixed_tokens[prompt_length + local] = token_id
                local += 1
    for token_id in suffix:
        fixed_tokens[prompt_length + local] = token_id
        local += 1

    if local > block_length:
        raise ValueError(
            "Agent routing region does not fit in the first block: "
            f"needs {local} tokens, block_length={block_length}."
        )
    task_markers = [_tokenize(tokenizer, f"<task{slot}>") for slot in range(config.slots)]
    remaining = gen_length - local - sum(len(marker) for marker in task_markers)
    if remaining < config.slots:
        raise ValueError("gen_length leaves no task token for every Agent slot.")

    task_spans = []
    base, extra = divmod(remaining, config.slots)
    cursor = prompt_length + local
    for slot in range(config.slots):
        for token_id in task_markers[slot]:
            fixed_tokens[cursor] = token_id
            cursor += 1
        length = base + (1 if slot < extra else 0)
        task_spans.append((cursor, cursor + length))
        cursor += length

    layout = AgentLayout(
        prompt_length=prompt_length,
        gen_length=gen_length,
        block_length=block_length,
        agent_width=width,
        agent_spans=tuple(agent_spans),
        task_spans=tuple(task_spans),
        fixed_tokens=fixed_tokens,
        filler_token_id=_choose_filler_token(tokenizer, mask_id),
    )
    return layout, token_ids


class PreloadManager:
    """Reference-count tentative loads so duplicate agents start only once."""

    def __init__(
        self,
        specs: Mapping[str, AgentSpec],
        dispatcher: AsyncEventDispatcher,
    ) -> None:
        self.specs = specs
        self.dispatcher = dispatcher
        self._claims: Dict[str, set] = {name: set() for name in specs}
        self._slot_agent: Dict[int, str] = {}

    def transition(
        self,
        slot: int,
        new_agent: Optional[str],
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        old_agent = self._slot_agent.get(slot)
        if old_agent == new_agent:
            return
        metadata_dict = dict(metadata or {})
        cancel_old = False
        start_new = False
        if old_agent is not None:
            self._claims[old_agent].discard(slot)
            cancel_old = not self._claims[old_agent] and not self.specs[old_agent].resident
            self._slot_agent.pop(slot, None)
        if new_agent is not None:
            start_new = not self._claims[new_agent] and not self.specs[new_agent].resident
            self._claims[new_agent].add(slot)
            self._slot_agent[slot] = new_agent

        metadata_dict.update(
            {
                "cancel_previous_model": cancel_old,
                "start_new_model": start_new,
                "deduplicated": bool(new_agent is not None and not start_new),
            }
        )
        if old_agent is None and new_agent is not None:
            if not start_new:
                return
            event_type = AgentEventType.PRELOAD_START
        elif old_agent is not None and new_agent is None:
            if not cancel_old:
                return
            event_type = AgentEventType.PRELOAD_CANCEL
        else:
            event_type = AgentEventType.PRELOAD_SWITCH
        self.dispatcher.emit(
            AgentEvent(
                event_type=event_type,
                slot=slot,
                agent_name=new_agent,
                previous_agent_name=old_agent,
                metadata=metadata_dict,
            )
        )


class AgentDecodingController:
    """Field-level scorer and MASKED/TENTATIVE/CONFIRMED state machine."""

    def __init__(
        self,
        tokenizer,
        config: AgentPriorityConfig,
        prompt_length: int,
        gen_length: int,
        block_length: int,
        total_steps: int,
        mask_id: int,
        event_callback: Optional[Callable[[AgentEvent], None]] = None,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self.enabled = config.enabled
        self.config = config
        self.tokenizer = tokenizer
        self.mask_id = mask_id
        self.total_steps = total_steps
        self.logger = logger
        self.dispatcher = AsyncEventDispatcher(event_callback, logger=logger)
        if not self.enabled:
            self.layout = None
            self.catalog_token_ids = {}
            self.specs = {}
            self.preloads = None
            self.slots = []
            self._observed_steps = 0
            self._last_observe_at = None
            self._step_seconds_ema = config.expected_step_seconds
            return
        self.layout, self.catalog_token_ids = build_agent_layout(
            tokenizer, config, prompt_length, gen_length, block_length, mask_id
        )
        self.specs = {spec.name: spec for spec in config.catalog}
        self.preloads = PreloadManager(self.specs, self.dispatcher)
        self.slots = [SlotRuntime() for _ in range(config.slots)]
        self._observed_steps = 0
        self._last_observe_at = None
        self._step_seconds_ema = config.expected_step_seconds

    def initialize(self, x: torch.Tensor) -> None:
        if not self.enabled:
            return
        if x.shape[0] != 1:
            raise ValueError("Agent-priority decoding currently requires batch size 1.")
        # initialize() is called by generate* after model loading, so timing from
        # controller/catalog construction cannot pollute the decode-time estimate.
        self._observed_steps = 0
        self._last_observe_at = time.perf_counter()
        self._step_seconds_ema = self.config.expected_step_seconds
        for position, token_id in self.layout.fixed_tokens.items():
            x[:, position] = token_id

    def decoder_mask(
        self, mask_index: torch.Tensor, mask_start: int = 0
    ) -> torch.Tensor:
        """Remove controller-owned fields while retaining ordinary task tokens."""

        if not self.enabled:
            return mask_index
        result = mask_index.clone()
        mask_end = mask_start + result.shape[1]

        def clear_span(start: int, end: int) -> None:
            overlap_start = max(start, mask_start)
            overlap_end = min(end, mask_end)
            if overlap_start < overlap_end:
                result[:, overlap_start - mask_start : overlap_end - mask_start] = False

        for start, end in self.layout.agent_spans:
            clear_span(start, end)
        for slot, runtime in enumerate(self.slots):
            if runtime.state == AgentFieldState.CONFIRMED and runtime.candidate == "none":
                start, end = self.layout.task_spans[slot]
                clear_span(start, end)
        return result

    def has_unconfirmed_agents(self) -> bool:
        return any(slot.state != AgentFieldState.CONFIRMED for slot in self.slots)

    def _score_slot(
        self,
        logits: torch.Tensor,
        logits_start: int,
        slot: int,
    ) -> Optional[Dict[str, float]]:
        start, end = self.layout.agent_spans[slot]
        relative_start = start - logits_start
        if relative_start < 0 or end - logits_start > logits.shape[1]:
            return None
        field_logits = logits[0, relative_start : relative_start + self.layout.agent_width]
        log_probs = torch.log_softmax(field_logits.to(torch.float64), dim=-1)
        joint_scores = []
        names = list(self.catalog_token_ids)
        for name in names:
            ids = self.catalog_token_ids[name]
            positions = torch.arange(len(ids), device=logits.device)
            targets = torch.tensor(ids, device=logits.device, dtype=torch.long)
            joint_scores.append(log_probs[positions, targets].sum())
        probabilities = torch.softmax(torch.stack(joint_scores), dim=0)
        return {
            name: float(probabilities[index].detach().cpu())
            for index, name in enumerate(names)
        }

    def _remaining_decode_seconds(self, global_step: int) -> float:
        return max(self.total_steps - global_step - 1, 0) * self._step_seconds_ema

    def _update_step_time(self) -> None:
        now = time.perf_counter()
        if self._last_observe_at is not None:
            observed = now - self._last_observe_at
            cap = (
                self.config.expected_step_seconds
                * self.config.max_observed_step_multiplier
            )
            observed = min(max(observed, 0.0), cap)
            alpha = self.config.step_time_ema_alpha
            self._step_seconds_ema = (
                (1.0 - alpha) * self._step_seconds_ema + alpha * observed
            )
        self._last_observe_at = now

    def preload_benefit(
        self,
        agent_name: str,
        probability: float,
        remaining_decode_seconds: float,
    ) -> float:
        if agent_name == "none":
            return 0.0
        spec = self.specs[agent_name]
        if spec.resident:
            return 0.0
        overlap = min(spec.cold_start_seconds, remaining_decode_seconds)
        return (
            probability * overlap
            - (1.0 - probability) * spec.wrong_preload_cost_seconds
        )

    def _write_candidate(self, x: torch.Tensor, slot: int, name: str) -> None:
        start, end = self.layout.agent_spans[slot]
        x[:, start:end] = self.layout.filler_token_id
        ids = self.catalog_token_ids[name]
        x[:, start : start + len(ids)] = torch.tensor(
            ids, device=x.device, dtype=x.dtype
        )

    def _mask_candidate(self, x: torch.Tensor, slot: int) -> None:
        start, end = self.layout.agent_spans[slot]
        x[:, start:end] = self.mask_id

    def _remask_task(self, x: torch.Tensor, slot: int) -> None:
        if not self.config.remask_task_on_switch:
            return
        start, end = self.layout.task_spans[slot]
        x[:, start:end] = self.mask_id

    def _set_none_task(self, x: torch.Tensor, slot: int) -> None:
        start, end = self.layout.task_spans[slot]
        x[:, start:end] = self.layout.filler_token_id
        ids = self.catalog_token_ids["none"]
        length = min(len(ids), end - start)
        x[:, start : start + length] = torch.tensor(
            ids[:length], device=x.device, dtype=x.dtype
        )

    def _confirm(
        self,
        x: torch.Tensor,
        slot_index: int,
        candidate: str,
        forced: bool,
        reason: str,
    ) -> None:
        runtime = self.slots[slot_index]
        previous_state = runtime.state
        runtime.state = AgentFieldState.CONFIRMED
        runtime.candidate = candidate
        self._write_candidate(x, slot_index, candidate)
        if candidate == "none":
            self.preloads.transition(slot_index, None)
            self._set_none_task(x, slot_index)
        else:
            self.preloads.transition(
                slot_index,
                candidate,
                {"reason": reason, "forced": forced},
            )
        self.dispatcher.emit(
            AgentEvent(
                event_type=AgentEventType.AGENT_CONFIRMED,
                slot=slot_index,
                agent_name=candidate,
                metadata={
                    "forced": forced,
                    "confirmation_reason": reason,
                    "previous_state": previous_state.value,
                    "probability": runtime.top_probability,
                    "margin": runtime.margin,
                },
            )
        )

    def _update_slot(
        self,
        x: torch.Tensor,
        slot_index: int,
        distribution: Dict[str, float],
        global_step: int,
        is_last_agent_step: bool,
    ) -> None:
        runtime = self.slots[slot_index]
        ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
        candidate, top_probability = ranked[0]
        second_probability = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_probability - second_probability
        if runtime.previous_distribution is None:
            change = 1.0
        else:
            change = 0.5 * sum(
                abs(distribution[name] - runtime.previous_distribution[name])
                for name in distribution
            )
        if runtime.candidate == candidate:
            runtime.consistent_steps += 1
        else:
            runtime.consistent_steps = 1
        runtime.previous_distribution = dict(distribution)
        runtime.last_distribution = dict(distribution)
        runtime.top_probability = top_probability
        runtime.margin = margin
        runtime.distribution_change = change

        remaining = self._remaining_decode_seconds(global_step)
        benefits = {
            name: self.preload_benefit(name, probability, remaining)
            for name, probability in distribution.items()
        }
        top_benefit = benefits[candidate]
        priority_bonus = self.config.benefit_priority_weight * max(top_benefit, 0.0)
        tentative_ready = (
            min(top_probability + priority_bonus, 1.0)
            >= self.config.tentative_probability
            and margin >= self.config.tentative_margin
        )
        confirm_ready = (
            top_probability >= self.config.confirm_probability
            and margin >= self.config.confirm_margin
            and change <= self.config.max_distribution_change
            and runtime.consistent_steps >= self.config.stable_steps
        )
        plateau_ready = (
            top_probability >= self.config.plateau_confirm_probability
            and margin >= self.config.plateau_confirm_margin
            and change <= self.config.plateau_max_distribution_change
            and runtime.consistent_steps >= self.config.plateau_stable_steps
        )

        previous_state = runtime.state
        previous_candidate = runtime.candidate
        if runtime.state == AgentFieldState.MASKED:
            if tentative_ready:
                runtime.state = AgentFieldState.TENTATIVE
                runtime.candidate = candidate
                runtime.unreliable_steps = 0
                self._write_candidate(x, slot_index, candidate)
        elif runtime.state == AgentFieldState.TENTATIVE:
            if not tentative_ready:
                runtime.unreliable_steps += 1
                if runtime.unreliable_steps >= self.config.tentative_cancel_steps:
                    self.preloads.transition(
                        slot_index, None, {"reason": "sustained_confidence_drop"}
                    )
                    runtime.preload_agent = None
                    runtime.state = AgentFieldState.MASKED
                    runtime.candidate = None
                    self._mask_candidate(x, slot_index)
                    self._remask_task(x, slot_index)
            else:
                runtime.unreliable_steps = 0
                if previous_candidate != candidate:
                    runtime.candidate = candidate
                    self._remask_task(x, slot_index)
                self._write_candidate(x, slot_index, candidate)

        if runtime.state == AgentFieldState.TENTATIVE:
            desired_preload = None
            if runtime.unreliable_steps > 0 and runtime.preload_agent is not None:
                # Match the state-machine grace period: a single noisy step must
                # not tear down an already-running preload through the benefit gate.
                desired_preload = runtime.preload_agent
            elif (
                runtime.candidate != "none"
                and benefits[runtime.candidate] >= self.config.min_preload_benefit_seconds
            ):
                desired_preload = runtime.candidate
            self.preloads.transition(
                slot_index,
                desired_preload,
                {
                    "benefit_seconds": benefits.get(runtime.candidate, 0.0),
                    "remaining_decode_seconds": remaining,
                },
            )
            runtime.preload_agent = desired_preload

        candidate_is_current = runtime.candidate == candidate
        if (
            runtime.state == AgentFieldState.TENTATIVE
            and candidate_is_current
            and (confirm_ready or plateau_ready or is_last_agent_step)
        ):
            if confirm_ready:
                reason = "standard_stability"
            elif plateau_ready:
                reason = "stable_plateau"
            else:
                reason = "last_agent_step"
            self._confirm(
                x,
                slot_index,
                runtime.candidate,
                forced=is_last_agent_step and not (confirm_ready or plateau_ready),
                reason=reason,
            )
        elif runtime.state == AgentFieldState.MASKED and is_last_agent_step:
            runtime.candidate = candidate
            self._confirm(
                x, slot_index, candidate, forced=True, reason="last_agent_step"
            )

        transition = None
        if previous_state != runtime.state or previous_candidate != runtime.candidate:
            transition = (
                f"{previous_state.value}:{previous_candidate or '-'}->"
                f"{runtime.state.value}:{runtime.candidate or '-'}"
            )
        self.logger.info(
            "agent_step %s",
            json.dumps(
                {
                    "step": global_step,
                    "slot": slot_index,
                    "candidates": distribution,
                    "top1": candidate,
                    "top1_probability": top_probability,
                    "margin": margin,
                    "distribution_change": change,
                    "consistent_steps": runtime.consistent_steps,
                    "unreliable_steps": runtime.unreliable_steps,
                    "state": runtime.state.value,
                    "transition": transition,
                    "preload_benefit_seconds": top_benefit,
                    "remaining_decode_seconds": remaining,
                    "estimated_step_seconds": self._step_seconds_ema,
                },
                sort_keys=True,
            ),
        )

    def observe(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        logits_start: int,
        global_step: int,
        is_last_agent_step: bool = False,
    ) -> None:
        if not self.enabled:
            return
        self._update_step_time()
        self._observed_steps += 1
        for slot_index, runtime in enumerate(self.slots):
            distribution = self._score_slot(logits, logits_start, slot_index)
            if distribution is not None:
                self._update_slot(
                    x,
                    slot_index,
                    distribution,
                    global_step,
                    is_last_agent_step,
                )
            elif runtime.last_distribution is not None:
                self.logger.info(
                    "agent_step %s",
                    json.dumps(
                        {
                            "step": global_step,
                            "slot": slot_index,
                            "candidates": runtime.last_distribution,
                            "top1": runtime.candidate,
                            "top1_probability": runtime.top_probability,
                            "margin": runtime.margin,
                            "distribution_change": runtime.distribution_change,
                            "consistent_steps": runtime.consistent_steps,
                            "state": runtime.state.value,
                            "transition": None,
                            "cached_distribution": True,
                        },
                        sort_keys=True,
                    ),
                )

    def finalize(self, x: torch.Tensor) -> None:
        """Guarantee catalog-only names even if decoding stopped unexpectedly."""

        for slot_index, runtime in enumerate(self.slots):
            if runtime.state == AgentFieldState.CONFIRMED:
                continue
            if runtime.last_distribution:
                candidate = max(runtime.last_distribution, key=runtime.last_distribution.get)
            else:
                candidate = "none"
            self._confirm(
                x, slot_index, candidate, forced=True, reason="finalize"
            )

    def plan(self, x: torch.Tensor) -> AgentPlan:
        self.finalize(x)
        agents: List[str] = []
        tasks: List[str] = []
        pad_ids = {
            self.layout.filler_token_id,
            self.mask_id,
        }
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None:
            pad_ids.add(int(eos_id))
        for slot_index, runtime in enumerate(self.slots):
            agent = runtime.candidate or "none"
            agents.append(agent)
            if agent == "none":
                tasks.append("none")
                continue
            start, end = self.layout.task_spans[slot_index]
            ids = x[0, start:end].detach().cpu().tolist()
            while ids and ids[-1] in pad_ids:
                ids.pop()
            if self.mask_id in ids:
                ids = ids[: ids.index(self.mask_id)]
            task = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            for marker in ("<task0>", "<task1>", "<task2>", "<task3>"):
                if marker in task:
                    task = task.split(marker, 1)[0]
            task = task.replace("<subtask>", " ").replace("</subtask>", " ")
            task = " ".join(task.split())
            tasks.append(task or "none")
        while len(agents) < MAX_AGENT_SLOTS:
            agents.append("none")
            tasks.append("none")
        return AgentPlan(tuple(agents), tuple(tasks))


def catalog_from_dicts(items: Iterable[Mapping[str, object]]) -> List[AgentSpec]:
    """Convenience loader for JSON/YAML-style Agent Catalog entries."""

    return [
        AgentSpec(
            name=str(item["name"]),
            cold_start_seconds=float(item.get("cold_start_seconds", 0.0)),
            wrong_preload_cost_seconds=float(
                item.get("wrong_preload_cost_seconds", 0.0)
            ),
            resident=bool(item.get("resident", False)),
        )
        for item in items
    ]
