"""Priority decoding for Agent names embedded in a normal JSON response.

Unlike :mod:`agent_priority`, this controller does not reserve a routing region
or render a private output format.  It searches full-sequence logits for the
first Agent fields in a compact JSON plan and writes catalog-constrained Agent
names directly into those final response positions.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch


# Reuse the existing rotating Agent log configured by llada_server.
LOGGER = logging.getLogger("fastdllm.agent_priority")


def extract_agent_registry(
    messages: Sequence[Dict[str, str]], fallback: Sequence[str]
) -> List[str]:
    """Read ordered Agent definitions from an AOP system prompt.

    Both HuskyQA and IIRC define roles as ``- name_agent: description`` lines,
    but use different registries. Restrict extraction to system-message
    definition lines so an Agent-like string in the user query cannot expand
    the constrained catalog.
    """

    names = []
    for message in messages:
        if message.get("role") != "system":
            continue
        content = str(message.get("content") or "")
        for name in re.findall(
            r"(?m)^\s*-\s*([a-z][a-z0-9_]*_agent)\s*:", content
        ):
            if name not in names:
                names.append(name)
    return names or list(fallback)


@dataclass(frozen=True)
class JsonAgentPriorityConfig:
    catalog: Sequence[str]
    priority_slots: int = 4
    tracking_slots: Optional[int] = None
    anchor_min_logit_margin: float = -6.0
    anchor_stable_steps: int = 2
    tentative_probability: float = 0.45
    tentative_margin: float = 0.20
    confirm_probability: float = 0.72
    confirm_margin: float = 0.25
    confirm_stable_steps: int = 2
    discovery_steps: int = 4
    min_anchor_gap: int = 12

    def __post_init__(self) -> None:
        names = list(self.catalog)
        if not names or len(names) != len(set(names)):
            raise ValueError("Agent catalog must be non-empty and contain unique names.")
        if self.priority_slots < 1:
            raise ValueError("priority_slots must be positive.")
        if self.tracking_slots is not None and self.tracking_slots < self.priority_slots:
            raise ValueError("tracking_slots must be at least priority_slots.")
        if self.anchor_stable_steps < 1 or self.confirm_stable_steps < 1:
            raise ValueError("Stability step counts must be positive.")
        if self.discovery_steps < 1:
            raise ValueError("discovery_steps must be positive.")


@dataclass
class JsonAgentSlotRuntime:
    anchor_start: Optional[int] = None
    anchor_token_ids: Tuple[int, ...] = ()
    name_start: Optional[int] = None
    anchor_score: float = -math.inf
    anchor_observed_ratio: float = 0.0
    anchor_consistent_steps: int = 0
    candidate: Optional[str] = None
    recognized_candidate: Optional[str] = None
    candidate_probability: float = 0.0
    candidate_margin: float = 0.0
    recognized_probability: Optional[float] = None
    recognized_margin: Optional[float] = None
    candidate_consistent_steps: int = 0
    confirmed: bool = False
    field_written: bool = False
    fuzzy_matched_from: Optional[str] = None
    first_observed_seconds: Optional[float] = None
    recognized_seconds: Optional[float] = None
    confirmed_seconds: Optional[float] = None
    first_observed_step: Optional[int] = None
    recognized_step: Optional[int] = None
    confirmed_step: Optional[int] = None
    last_distribution: Optional[Dict[str, float]] = field(default=None)


class JsonAgentFieldController:
    """Locate and prioritize the first Agent fields in ordinary plan JSON."""

    def __init__(
        self,
        tokenizer,
        config: JsonAgentPriorityConfig,
        prompt_length: int,
        gen_length: int,
        mask_id: int,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self.enabled = True
        self.tokenizer = tokenizer
        self.config = config
        self.prompt_length = int(prompt_length)
        self.gen_length = int(gen_length)
        self.mask_id = int(mask_id)
        self.logger = logger
        # Dual Cache already performs one full-sequence forward per block.
        # Repeating the identical all-mask forward here neither reveals new
        # structure nor improves confidence, and speculative writes during that
        # phase can create self-fulfilling JSON anchors.  Reuse normal block
        # warm-ups for continuously improving anchor discovery instead.
        self.full_sequence_discovery_steps = 0
        self.tracking_slots = config.tracking_slots or config.priority_slots
        self.slots = [JsonAgentSlotRuntime() for _ in range(self.tracking_slots)]
        self._started_at: Optional[float] = None
        self._observed_steps = 0
        self._full_sequence_observations = 0

        # Do not include the opening brace, indentation, or opening key quote.
        # The LLaDA tokenizer merges leading whitespace with ``{``/``\"`` into
        # context-dependent tokens (for example ``Ġ{`` and ``Ġ\"``), so a
        # standalone encoding of a pretty-printed object never matches the
        # sequence.  This key suffix is context-stable and remains specific to
        # an Agent JSON field.
        canonical_anchor_texts = ['agent":"', 'agent": "']
        # Materialized output may vary in JSON whitespace. Keep speculative
        # discovery on the two canonical forms so tolerant variants cannot add
        # all-mask false positives.
        anchor_texts = list(canonical_anchor_texts)
        for before_colon in (" ", "\t", "\n", "\n  "):
            for after_colon in ("", " ", "\t", "\n", "\n  "):
                anchor_texts.append(f'agent"{before_colon}:{after_colon}"')
        anchor_texts = list(dict.fromkeys(anchor_texts))
        self.anchor_variants = tuple(
            tuple(self._encode(text)) for text in anchor_texts
        )
        self.speculative_anchor_variants = frozenset(
            tuple(self._encode(text)) for text in canonical_anchor_texts
        )
        if any(not value for value in self.anchor_variants):
            raise ValueError("Tokenizer produced an empty JSON Agent anchor.")

        space_ids = self._encode(" ")
        if len(space_ids) != 1 or not self.tokenizer.decode(space_ids).isspace():
            raise ValueError("JSON Agent padding requires a single whitespace token.")
        self.space_token_id = int(space_ids[0])

        comma_ids = self._encode(",")
        if len(comma_ids) != 1 or self.tokenizer.decode(comma_ids) != ",":
            raise ValueError("JSON Agent layout requires a single comma token.")
        self.comma_token_id = int(comma_ids[0])

        self.catalog_value_ids = {
            name: tuple(self._encode(name + '"')) for name in config.catalog
        }
        if any(not ids for ids in self.catalog_value_ids.values()):
            raise ValueError("Every Agent name must tokenize to at least one token.")
        self.value_width = max(len(ids) for ids in self.catalog_value_ids.values())
        self.padded_catalog_ids = {
            name: ids + (self.space_token_id,) * (self.value_width - len(ids))
            for name, ids in self.catalog_value_ids.items()
        }

    def _encode(self, text: str) -> List[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def initialize(self, x: torch.Tensor) -> None:
        if x.shape[0] != 1:
            raise ValueError("JSON Agent priority decoding requires batch size 1.")
        self._started_at = time.perf_counter()
        self._observed_steps = 0
        self._full_sequence_observations = 0

    def close(self) -> None:
        return None

    def _elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        return time.perf_counter() - self._started_at

    def _anchor_candidates(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        logits_start: int,
    ) -> List[Tuple[int, Tuple[int, ...], float, float]]:
        """Return ``(absolute_start, pattern, score, observed_ratio)`` candidates."""

        sequence_logits = logits[0]
        sequence_max = sequence_logits.amax(dim=-1)
        absolute_end = logits_start + sequence_logits.shape[0]
        generation_start = self.prompt_length
        generation_end = self.prompt_length + self.gen_length
        candidates: Dict[int, Tuple[int, Tuple[int, ...], float, float]] = {}

        for pattern in self.anchor_variants:
            width = len(pattern)
            start = max(generation_start, logits_start)
            end = min(generation_end, absolute_end) - width + 1
            if end <= start:
                continue
            relative = start - logits_start
            count = end - start
            scores = torch.zeros(count, device=logits.device, dtype=torch.float32)
            compatible = torch.ones(count, device=logits.device, dtype=torch.bool)
            observed = torch.zeros(count, device=logits.device, dtype=torch.float32)

            for offset, token_id in enumerate(pattern):
                row_start = relative + offset
                row_end = row_start + count
                target = sequence_logits[row_start:row_end, token_id].float()
                scores += target - sequence_max[row_start:row_end].float()
                absolute_positions = torch.arange(
                    start + offset,
                    end + offset,
                    device=x.device,
                )
                current = x[0, absolute_positions]
                compatible &= (current == self.mask_id) | (current == token_id)
                observed += (current == token_id).float()

            scores /= float(width)
            observed /= float(width)
            if pattern in self.speculative_anchor_variants:
                eligible = compatible & (
                    (scores >= self.config.anchor_min_logit_margin)
                    | (observed == 1.0)
                )
            else:
                # Tolerant whitespace forms are evidence only after the normal
                # decoder has actually emitted them.
                eligible = compatible & (observed == 1.0)
            observed_indices = torch.nonzero(
                eligible & (observed == 1.0), as_tuple=False
            ).flatten()
            speculative_indices = torch.nonzero(
                eligible & (observed != 1.0), as_tuple=False
            ).flatten()
            if observed_indices.numel() == 0 and speculative_indices.numel() == 0:
                continue
            # A fully materialized anchor is direct evidence from the normal
            # response and must never be displaced by higher-logit speculative
            # positions. Limit only the speculative host transfers.
            selected_indices = observed_indices.detach().cpu().tolist()
            if speculative_indices.numel() > 0:
                top_count = min(
                    int(speculative_indices.numel()),
                    self.config.priority_slots * 8,
                )
                top = torch.topk(scores[speculative_indices], k=top_count).indices
                selected_indices.extend(
                    speculative_indices[top].detach().cpu().tolist()
                )
            for local_index in selected_indices:
                absolute_start = start + int(local_index)
                candidate = (
                    absolute_start,
                    pattern,
                    float(scores[local_index].detach().cpu()),
                    float(observed[local_index].detach().cpu()),
                )
                previous = candidates.get(absolute_start)
                if previous is None or (candidate[3], candidate[2]) > (
                    previous[3], previous[2]
                ):
                    candidates[absolute_start] = candidate

        # Different whitespace variants can describe the same field with starts
        # a token or two apart.  Collapse those local alternatives by confidence
        # before applying occurrence order; otherwise a weaker pretty-print
        # variant immediately before a compact anchor can steal its slot.
        clustered = []
        cluster_radius = max(len(pattern) for pattern in self.anchor_variants)
        for candidate in sorted(candidates.values(), key=lambda item: item[0]):
            if not clustered or candidate[0] - clustered[-1][-1][0] >= cluster_radius:
                clustered.append([candidate])
            else:
                clustered[-1].append(candidate)
        ranked = [
            max(cluster, key=lambda item: (item[3], item[2]))
            for cluster in clustered
        ]
        ranked.sort(key=lambda item: item[0])

        # The policy is explicitly occurrence based: priority slots correspond
        # to the first four Agent fields in response order, not the four fields
        # with the highest confidence.  Confidence gates admission above; after
        # that, preserve textual order and keep duplicate Agent names separate.
        selected = []
        for candidate in ranked:
            position = candidate[0]
            if any(
                abs(position - existing[0]) < self.config.min_anchor_gap
                for existing in selected
            ):
                continue
            selected.append(candidate)
            if len(selected) == self.config.priority_slots:
                break

        # Timing-only slots observe every later field that has already been
        # materialized by the normal decoder. They never admit speculative
        # anchors and therefore cannot affect the first-four prefetch policy.
        tracked = {candidate[0]: candidate for candidate in selected}
        for candidate in ranked:
            if candidate[3] == 1.0:
                tracked[candidate[0]] = candidate
        selected = []
        for candidate in sorted(tracked.values(), key=lambda item: item[0]):
            position = candidate[0]
            if any(
                abs(position - existing[0]) < self.config.min_anchor_gap
                for existing in selected
            ):
                continue
            selected.append(candidate)
            if len(selected) == self.tracking_slots:
                break
        return selected

    def _assign_anchors(
        self,
        x: torch.Tensor,
        candidates: List[Tuple[int, Tuple[int, ...], float, float]],
    ) -> None:
        for slot_index, runtime in enumerate(self.slots):
            if slot_index >= len(candidates):
                # A later full-sequence observation can prove that an early
                # all-mask pass hallucinated more plan objects than the normal
                # response contains. Drop stale, unmaterialized slots instead
                # of reporting phantom Agent calls.
                if not runtime.confirmed and not runtime.field_written:
                    self.slots[slot_index] = JsonAgentSlotRuntime()
                continue
            anchor_start, pattern, score, observed_ratio = candidates[slot_index]
            if runtime.anchor_start == anchor_start and runtime.anchor_token_ids == pattern:
                runtime.anchor_consistent_steps += 1
                runtime.anchor_score = score
                runtime.anchor_observed_ratio = observed_ratio
                continue
            # A field is written only after its anchor occurs naturally in x.
            # Once written, never erase or relocate that normal response text.
            if runtime.confirmed or runtime.field_written:
                continue
            runtime.anchor_start = anchor_start
            runtime.anchor_token_ids = pattern
            runtime.name_start = anchor_start + len(pattern)
            runtime.anchor_score = score
            runtime.anchor_observed_ratio = observed_ratio
            runtime.anchor_consistent_steps = 1
            runtime.candidate = None
            runtime.recognized_candidate = None
            runtime.candidate_probability = 0.0
            runtime.candidate_margin = 0.0
            runtime.recognized_probability = None
            runtime.recognized_margin = None
            runtime.candidate_consistent_steps = 0
            runtime.last_distribution = None
            runtime.first_observed_seconds = None
            runtime.recognized_seconds = None
            runtime.confirmed_seconds = None
            runtime.first_observed_step = None
            runtime.recognized_step = None
            runtime.confirmed_step = None
            runtime.field_written = False
            runtime.fuzzy_matched_from = None

    def _score_catalog(
        self,
        logits: torch.Tensor,
        logits_start: int,
        runtime: JsonAgentSlotRuntime,
    ) -> Optional[Dict[str, float]]:
        if runtime.name_start is None:
            return None
        relative_start = runtime.name_start - logits_start
        relative_end = relative_start + self.value_width
        if relative_start < 0 or relative_end > logits.shape[1]:
            return None
        field_logits = logits[0, relative_start:relative_end].float()
        log_probs = torch.log_softmax(field_logits, dim=-1)
        positions = torch.arange(self.value_width, device=logits.device)
        scores = []
        names = list(self.padded_catalog_ids)
        for name in names:
            targets = torch.tensor(
                self.padded_catalog_ids[name], device=logits.device, dtype=torch.long
            )
            scores.append(log_probs[positions, targets].sum())
        probabilities = torch.softmax(torch.stack(scores), dim=0)
        return {
            name: float(probabilities[index].detach().cpu())
            for index, name in enumerate(names)
        }

    @staticmethod
    def _normalized_agent_name(value: str) -> str:
        return "".join(character for character in value.lower() if character.isalnum())

    @staticmethod
    def _edit_distance(left: str, right: str) -> int:
        previous = list(range(len(right) + 1))
        for left_index, left_character in enumerate(left, start=1):
            current = [left_index]
            for right_index, right_character in enumerate(right, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[right_index] + 1,
                        previous[right_index - 1]
                        + (left_character != right_character),
                    )
                )
            previous = current
        return previous[-1]

    def _observed_catalog_value(
        self,
        x: torch.Tensor,
        runtime: JsonAgentSlotRuntime,
    ) -> Optional[str]:
        """Return an Agent name already decoded in the normal response."""

        if runtime.name_start is None:
            return None
        for name, token_ids in self.catalog_value_ids.items():
            end = runtime.name_start + len(token_ids)
            if end > x.shape[1]:
                continue
            expected = torch.tensor(token_ids, device=x.device, dtype=x.dtype)
            if torch.equal(x[0, runtime.name_start:end], expected):
                runtime.fuzzy_matched_from = None
                return name

        # Map a naturally emitted near-match for prefetch reporting, without
        # rewriting the response. Separator differences normalize away; other
        # spelling perturbations are accepted only at unique edit distance one.
        window_end = min(x.shape[1], runtime.name_start + self.value_width + 5)
        token_ids = x[0, runtime.name_start:window_end].detach().cpu().tolist()
        visible = []
        for token_id in token_ids:
            if token_id == self.mask_id:
                break
            visible.append(token_id)
        if not visible:
            return None
        decoded = self.tokenizer.decode(visible, skip_special_tokens=True)
        match = re.match(r'^\s*([A-Za-z][A-Za-z0-9_ -]{2,31})\s*["\']', decoded)
        if match is None:
            return None
        raw_name = match.group(1).strip()
        normalized = self._normalized_agent_name(raw_name)
        matches = [
            name
            for name in self.catalog_value_ids
            if self._edit_distance(
                normalized, self._normalized_agent_name(name)
            ) <= 1
        ]
        if len(matches) == 1:
            runtime.fuzzy_matched_from = (
                raw_name
                if normalized != self._normalized_agent_name(matches[0])
                else None
            )
            return matches[0]
        return None

    def _write_candidate(
        self,
        x: torch.Tensor,
        runtime: JsonAgentSlotRuntime,
        candidate: str,
    ) -> None:
        if runtime.anchor_start is None or runtime.name_start is None:
            return
        # Never synthesize an anchor: doing that makes the next observation
        # treat controller-written text as model evidence and can duplicate or
        # truncate plan objects.  This method is called only for an anchor that
        # is already fully present in x.
        value = torch.tensor(
            self.padded_catalog_ids[candidate], device=x.device, dtype=x.dtype
        )
        x[:, runtime.name_start:runtime.name_start + self.value_width] = value
        x[:, runtime.name_start + self.value_width] = self.comma_token_id
        runtime.field_written = True

    def _update_slot(
        self,
        x: torch.Tensor,
        slot_index: int,
        distribution: Dict[str, float],
        global_step: int,
        allow_write: bool = True,
    ) -> None:
        runtime = self.slots[slot_index]
        ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
        candidate, probability = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = probability - second
        now = self._elapsed()
        if runtime.first_observed_seconds is None:
            runtime.first_observed_seconds = now
            runtime.first_observed_step = global_step
        if runtime.candidate == candidate:
            runtime.candidate_consistent_steps += 1
        else:
            runtime.candidate = candidate
            runtime.candidate_consistent_steps = 1
        runtime.candidate_probability = probability
        runtime.candidate_margin = margin
        runtime.last_distribution = dict(distribution)

        anchor_ready = (
            runtime.anchor_observed_ratio == 1.0
            or runtime.anchor_consistent_steps >= self.config.anchor_stable_steps
        )
        reliable = (
            anchor_ready
            and runtime.candidate_consistent_steps >= 2
            and probability >= self.config.tentative_probability
            and margin >= self.config.tentative_margin
        )
        if reliable:
            if runtime.recognized_candidate is None:
                runtime.recognized_candidate = candidate
                runtime.recognized_probability = probability
                runtime.recognized_margin = margin
                runtime.recognized_seconds = now
                runtime.recognized_step = global_step
            # A speculative recognition can be surfaced for prefetch, but it
            # must not mutate the response until the JSON anchor was naturally
            # decoded by the model.
            if (
                allow_write
                and runtime.anchor_observed_ratio == 1.0
                and candidate == runtime.recognized_candidate
            ):
                self._write_candidate(x, runtime, runtime.recognized_candidate)

        confirm = (
            reliable
            and candidate == runtime.recognized_candidate
            and runtime.anchor_observed_ratio == 1.0
            and probability >= self.config.confirm_probability
            and margin >= self.config.confirm_margin
            and runtime.candidate_consistent_steps >= self.config.confirm_stable_steps
        )
        if confirm and not runtime.confirmed:
            runtime.confirmed = True
            runtime.confirmed_seconds = now
            runtime.confirmed_step = global_step

        self.logger.info(
            "json_agent_step %s",
            json.dumps(
                {
                    "step": global_step,
                    "slot": slot_index,
                    "anchor_start": runtime.anchor_start,
                    "anchor_score": runtime.anchor_score,
                    "anchor_consistent_steps": runtime.anchor_consistent_steps,
                    "candidate": candidate,
                    "probability": probability,
                    "margin": margin,
                    "candidate_consistent_steps": runtime.candidate_consistent_steps,
                    "recognized_seconds": runtime.recognized_seconds,
                    "confirmed_seconds": runtime.confirmed_seconds,
                    "confirmed": runtime.confirmed,
                    "fuzzy_matched_from": runtime.fuzzy_matched_from,
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
        del is_last_agent_step
        self._observed_steps += 1
        covers_full_sequence = (
            logits_start <= self.prompt_length
            and logits_start + logits.shape[1]
            >= self.prompt_length + self.gen_length
        )
        if covers_full_sequence:
            self._full_sequence_observations += 1
            candidates = self._anchor_candidates(logits, x, logits_start)
            self._assign_anchors(x, candidates)
        for slot_index, runtime in enumerate(self.slots):
            if runtime.confirmed:
                continue
            observed_name = self._observed_catalog_value(x, runtime)
            if observed_name is not None:
                distribution = {
                    name: 1.0 if name == observed_name else 0.0
                    for name in self.padded_catalog_ids
                }
            else:
                distribution = self._score_catalog(logits, logits_start, runtime)
            if distribution is not None:
                self._update_slot(
                    x,
                    slot_index,
                    distribution,
                    global_step,
                    allow_write=(
                        slot_index < self.config.priority_slots
                        and observed_name is None
                    ),
                )

    def decoder_mask(
        self, mask_index: torch.Tensor, mask_start: int = 0
    ) -> torch.Tensor:
        result = mask_index.clone()
        mask_end = mask_start + result.shape[1]
        for runtime in self.slots:
            if runtime.name_start is None or not runtime.field_written:
                continue
            start = runtime.anchor_start
            end = runtime.name_start + self.value_width + 1
            overlap_start = max(start, mask_start)
            overlap_end = min(end, mask_end)
            if overlap_start < overlap_end:
                result[:, overlap_start - mask_start:overlap_end - mask_start] = False
        return result

    def has_unconfirmed_agents(self) -> bool:
        # Agent discovery must not extend a block's normal denoising loop.  The
        # next block warm-up supplies another full-sequence observation.
        return False

    def _materialized_anchor_candidates(
        self, x: torch.Tensor
    ) -> List[Tuple[int, Tuple[int, ...], float, float]]:
        """Find natural JSON Agent anchors without consulting model logits."""

        generation_start = self.prompt_length
        generation_end = min(
            x.shape[1], self.prompt_length + self.gen_length
        )
        sequence = x[0, generation_start:generation_end]
        candidates: Dict[int, Tuple[int, Tuple[int, ...], float, float]] = {}
        for pattern in self.anchor_variants:
            width = len(pattern)
            if sequence.shape[0] < width:
                continue
            target = torch.tensor(pattern, device=x.device, dtype=x.dtype)
            windows = sequence.unfold(0, width, 1)
            matches = torch.nonzero(
                (windows == target).all(dim=1), as_tuple=False
            ).flatten()
            for relative_start in matches.detach().cpu().tolist():
                absolute_start = generation_start + int(relative_start)
                candidate = (absolute_start, pattern, 0.0, 1.0)
                previous = candidates.get(absolute_start)
                if previous is None or len(pattern) > len(previous[1]):
                    candidates[absolute_start] = candidate

        clustered = []
        cluster_radius = max(len(pattern) for pattern in self.anchor_variants)
        for candidate in sorted(candidates.values(), key=lambda item: item[0]):
            if not clustered or candidate[0] - clustered[-1][-1][0] >= cluster_radius:
                clustered.append([candidate])
            else:
                clustered[-1].append(candidate)
        selected = [
            max(cluster, key=lambda item: len(item[1]))
            for cluster in clustered
        ]
        return selected[:self.tracking_slots]

    def finalize(self, x: torch.Tensor) -> None:
        # Capture a field that materialized in the last generation block, where
        # no subsequent full-sequence warm-up exists. Exact natural JSON is
        # definitive evidence, so this records an end-of-generation upper bound
        # without writing or otherwise changing the response canvas.
        self._assign_anchors(x, self._materialized_anchor_candidates(x))
        now = self._elapsed()
        final_step = self._observed_steps
        for runtime in self.slots:
            observed_name = self._observed_catalog_value(x, runtime)
            if observed_name is None:
                continue
            if runtime.first_observed_seconds is None:
                runtime.first_observed_seconds = now
                runtime.first_observed_step = final_step
            runtime.candidate = observed_name
            runtime.recognized_candidate = observed_name
            runtime.candidate_probability = 1.0
            runtime.candidate_margin = 1.0
            runtime.recognized_probability = 1.0
            runtime.recognized_margin = 1.0
            if runtime.recognized_seconds is None:
                runtime.recognized_seconds = now
                runtime.recognized_step = final_step
            if runtime.confirmed_seconds is None:
                runtime.confirmed_seconds = now
                runtime.confirmed_step = final_step
            runtime.confirmed = True

    def metrics(self) -> Dict[str, object]:
        slots = []
        for index, runtime in enumerate(self.slots):
            if (
                index >= self.config.priority_slots
                and runtime.anchor_start is None
                and runtime.recognized_candidate is None
            ):
                continue
            slots.append(
                {
                    "slot": index,
                    "priority": index < self.config.priority_slots,
                    "agent": runtime.recognized_candidate,
                    "anchor_start": runtime.anchor_start,
                    "anchor_score": (
                        runtime.anchor_score if math.isfinite(runtime.anchor_score) else None
                    ),
                    "anchor_observed_ratio": runtime.anchor_observed_ratio,
                    "first_observed_seconds": runtime.first_observed_seconds,
                    "recognized_seconds": runtime.recognized_seconds,
                    "confirmed_seconds": runtime.confirmed_seconds,
                    "first_observed_step": runtime.first_observed_step,
                    "recognized_step": runtime.recognized_step,
                    "confirmed_step": runtime.confirmed_step,
                    "probability": runtime.recognized_probability,
                    "margin": runtime.recognized_margin,
                    "confirmed": runtime.confirmed,
                    "fuzzy_matched_from": runtime.fuzzy_matched_from,
                }
            )
        recognized = [
            slot["recognized_seconds"]
            for slot in slots
            if slot["recognized_seconds"] is not None
        ]
        discovered_count = sum(slot["anchor_start"] is not None for slot in slots)
        recognized_count = len(recognized)
        all_recognized = (
            discovered_count > 0 and recognized_count == discovered_count
        )
        priority = slots[:self.config.priority_slots]
        priority_discovered = sum(
            slot["anchor_start"] is not None for slot in priority
        )
        priority_recognized = sum(
            slot["recognized_seconds"] is not None for slot in priority
        )
        return {
            "priority_slots": self.config.priority_slots,
            "tracking_slots": self.tracking_slots,
            "catalog": list(self.config.catalog),
            "observed_steps": self._observed_steps,
            "full_sequence_observations": self._full_sequence_observations,
            "discovered_agent_fields": discovered_count,
            "recognized_agent_fields": recognized_count,
            "all_priority_agents_recognized": (
                priority_discovered > 0
                and priority_recognized == priority_discovered
            ),
            "all_tracked_agents_recognized": all_recognized,
            "agent_slots": slots,
            # Do not label a partial result as "all recognized".  Keep the
            # partial timestamp separately so failed runs remain diagnosable.
            "all_recognized_seconds": (
                max(recognized) if all_recognized else None
            ),
            "latest_partial_recognized_seconds": (
                max(recognized) if recognized else None
            ),
        }
