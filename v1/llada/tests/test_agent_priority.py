import logging
import time

import torch

from agent_priority import (
    LOGGER,
    AgentDecodingController,
    AgentEvent,
    AgentEventType,
    AgentFieldState,
    AgentPriorityConfig,
    AgentSpec,
    AsyncEventDispatcher,
    configure_agent_file_logging,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9
    bos_token_id = 10

    def __init__(self):
        self.encoded = {
            "<agents>": [1],
            "|": [2],
            "</agents>": [3],
            "search_agent": [4, 5],
            "code_agent": [6],
            "summary_agent": [7],
            "none": [8],
            "<task0>": [14],
            "<task1>": [14],
            "<task2>": [14],
            "<task3>": [14],
        }
        self.decoded = {
            11: "search recent methods ",
            12: "modify decoder ",
            13: "summarize results ",
        }

    def encode(self, text, add_special_tokens=False):
        return self.encoded[text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.decoded.get(token_id, "") for token_id in ids)


def make_controller(callback=None, slots=1, **overrides):
    values = {
        "tentative_probability": 0.45,
        "tentative_margin": 0.10,
        "confirm_probability": 0.70,
        "confirm_margin": 0.20,
        "max_distribution_change": 0.05,
        "stable_steps": 2,
        "expected_step_seconds": 1.0,
    }
    values.update(overrides)
    config = AgentPriorityConfig(
        catalog=[
            AgentSpec("search_agent", 4.0, 0.5),
            AgentSpec("code_agent", 3.0, 0.5),
            AgentSpec("summary_agent", 0.0, 0.0, resident=True),
        ],
        slots=slots,
        **values,
    )
    return AgentDecodingController(
        tokenizer=FakeTokenizer(),
        config=config,
        prompt_length=2,
        gen_length=32,
        block_length=16,
        total_steps=8,
        mask_id=15,
        event_callback=callback,
    )


def logits_for(controller, winners, strength=8.0):
    logits = torch.zeros(1, 34, 16)
    for slot, winner in enumerate(winners):
        start, _ = controller.layout.agent_spans[slot]
        for offset, token_id in enumerate(controller.catalog_token_ids[winner]):
            logits[0, start + offset, token_id] = strength
    return logits


def test_field_probability_uses_joint_token_sequence():
    controller = make_controller()
    logits = logits_for(controller, ["search_agent"], strength=4.0)
    distribution = controller._score_slot(logits, logits_start=0, slot=0)

    start, _ = controller.layout.agent_spans[0]
    log_probs = torch.log_softmax(logits[0, start : start + 2].double(), dim=-1)
    expected_scores = torch.tensor(
        [
            log_probs[0, 4] + log_probs[1, 5],
            log_probs[0, 6],
            log_probs[0, 7],
            log_probs[0, 8],
        ]
    )
    expected = torch.softmax(expected_scores, dim=0)
    assert abs(sum(distribution.values()) - 1.0) < 1e-8
    assert distribution["search_agent"] == expected[0].item()


def test_tentative_then_confirmed_and_preload_is_deduplicated(caplog):
    events = []
    controller = make_controller(events.append, slots=4)
    x = torch.full((1, 34), 15, dtype=torch.long)
    controller.initialize(x)
    logits = logits_for(controller, ["search_agent"] * 4)

    with caplog.at_level(logging.INFO, logger="fastdllm.agent_priority"):
        controller.observe(logits, x, 0, global_step=0)
        assert all(slot.state == AgentFieldState.TENTATIVE for slot in controller.slots)
        controller.observe(logits, x, 0, global_step=1)

    controller.dispatcher.drain()
    assert all(slot.state == AgentFieldState.CONFIRMED for slot in controller.slots)
    assert sum(event.event_type == AgentEventType.PRELOAD_START for event in events) == 1
    assert sum(event.event_type == AgentEventType.AGENT_CONFIRMED for event in events) == 4
    assert "top1_probability" in caplog.text
    assert "distribution_change" in caplog.text


def test_tentative_switch_remasks_task_and_emits_switch():
    events = []
    controller = make_controller(
        events.append,
        confirm_probability=0.99999,
        max_distribution_change=0.0,
    )
    x = torch.full((1, 34), 15, dtype=torch.long)
    controller.initialize(x)
    controller.observe(logits_for(controller, ["search_agent"]), x, 0, 0)
    task_start, task_end = controller.layout.task_spans[0]
    x[:, task_start:task_end] = 11

    controller.observe(logits_for(controller, ["code_agent"]), x, 0, 1)
    controller.dispatcher.drain()

    assert controller.slots[0].state == AgentFieldState.TENTATIVE
    assert controller.slots[0].candidate == "code_agent"
    assert torch.all(x[:, task_start:task_end] == 15)
    assert any(event.event_type == AgentEventType.PRELOAD_SWITCH for event in events)


def test_confidence_drop_cancels_tentative_preload():
    events = []
    controller = make_controller(events.append)
    x = torch.full((1, 34), 15, dtype=torch.long)
    controller.initialize(x)
    controller.observe(logits_for(controller, ["search_agent"]), x, 0, 0)

    controller.observe(torch.zeros(1, 34, 16), x, 0, 1)
    assert controller.slots[0].state == AgentFieldState.TENTATIVE
    controller.observe(torch.zeros(1, 34, 16), x, 0, 2)
    controller.dispatcher.drain()

    assert controller.slots[0].state == AgentFieldState.MASKED
    assert any(event.event_type == AgentEventType.PRELOAD_CANCEL for event in events)


def test_single_step_confidence_dip_does_not_thrash_preload():
    events = []
    controller = make_controller(events.append)
    x = torch.full((1, 34), 15, dtype=torch.long)
    controller.initialize(x)
    strong = logits_for(controller, ["search_agent"])
    controller.observe(strong, x, 0, 0)
    controller.observe(torch.zeros(1, 34, 16), x, 0, 1)
    controller.observe(strong, x, 0, 2)
    controller.dispatcher.drain()

    assert controller.slots[0].candidate == "search_agent"
    assert not any(event.event_type == AgentEventType.PRELOAD_CANCEL for event in events)


def test_stable_probability_plateau_confirms_without_waiting_for_last_step():
    events = []
    controller = make_controller(
        events.append,
        confirm_probability=0.72,
        plateau_confirm_probability=0.52,
        plateau_confirm_margin=0.20,
        plateau_max_distribution_change=0.02,
        plateau_stable_steps=4,
    )
    x = torch.full((1, 34), 15, dtype=torch.long)
    controller.initialize(x)
    distribution = {
        "search_agent": 0.58,
        "code_agent": 0.20,
        "summary_agent": 0.10,
        "none": 0.12,
    }
    for step in range(4):
        controller._update_slot(x, 0, distribution, step, False)
    controller.dispatcher.drain()

    assert controller.slots[0].state == AgentFieldState.CONFIRMED
    confirmation = next(
        event for event in events if event.event_type == AgentEventType.AGENT_CONFIRMED
    )
    assert confirmation.metadata["confirmation_reason"] == "stable_plateau"
    assert confirmation.metadata["forced"] is False


def test_decode_timer_resets_and_clips_slow_first_forward():
    controller = make_controller(expected_step_seconds=0.05)
    x = torch.full((1, 34), 15, dtype=torch.long)
    controller.initialize(x)
    controller._last_observe_at = time.perf_counter() - 100.0
    controller._update_step_time()

    expected_upper_bound = 0.05 * (
        1.0 - controller.config.step_time_ema_alpha
        + controller.config.step_time_ema_alpha
        * controller.config.max_observed_step_multiplier
    )
    assert controller._step_seconds_ema <= expected_upper_bound + 1e-9
    assert controller._remaining_decode_seconds(0) < 2.0


def test_none_never_preloads_and_renders_four_slots():
    events = []
    controller = make_controller(events.append)
    x = torch.full((1, 34), 15, dtype=torch.long)
    controller.initialize(x)
    logits = logits_for(controller, ["none"])
    controller.observe(logits, x, 0, 0)
    controller.observe(logits, x, 0, 1)
    plan = controller.plan(x)
    controller.dispatcher.drain()

    assert plan.agents == ("none", "none", "none", "none")
    assert plan.tasks == ("none", "none", "none", "none")
    assert "agent_name: none\ntask: none" in plan.render()
    assert not any(event.event_type.value.startswith("PRELOAD") for event in events)


def test_preload_benefit_accounts_for_residency_and_error_cost():
    controller = make_controller()
    benefit = controller.preload_benefit("search_agent", 0.75, 2.0)
    assert benefit == 0.75 * 2.0 - 0.25 * 0.5
    assert controller.preload_benefit("summary_agent", 1.0, 20.0) == 0.0
    assert controller.preload_benefit("none", 1.0, 20.0) == 0.0


def test_event_callback_does_not_block_emitter():
    received = []

    def slow_callback(event):
        time.sleep(0.05)
        received.append(event)

    dispatcher = AsyncEventDispatcher(slow_callback)
    started = time.perf_counter()
    dispatcher.emit(AgentEvent(AgentEventType.PRELOAD_START, 0, "search_agent"))
    elapsed = time.perf_counter() - started
    dispatcher.drain()

    assert elapsed < 0.02
    assert len(received) == 1


def test_verbose_logs_can_be_written_to_rotating_file(tmp_path):
    old_handlers = list(LOGGER.handlers)
    old_level = LOGGER.level
    old_propagate = LOGGER.propagate
    try:
        path = configure_agent_file_logging(str(tmp_path / "agent.log"))
        LOGGER.info("agent_step test-record")
        for handler in LOGGER.handlers:
            handler.flush()
        assert path.read_text(encoding="utf-8").endswith("agent_step test-record\n")
        assert LOGGER.propagate is False
    finally:
        for handler in list(LOGGER.handlers):
            if handler not in old_handlers:
                LOGGER.removeHandler(handler)
                handler.close()
        LOGGER.handlers = old_handlers
        LOGGER.setLevel(old_level)
        LOGGER.propagate = old_propagate
