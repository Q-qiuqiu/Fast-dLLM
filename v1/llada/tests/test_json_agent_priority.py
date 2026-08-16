from types import SimpleNamespace

import torch

from generate import generate_with_dual_cache
from json_agent_priority import (
    JsonAgentFieldController,
    JsonAgentPriorityConfig,
    extract_agent_registry,
)


class CharacterTokenizer:
    mask_token_id = 255

    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) + 1 for character in text]

    @staticmethod
    def decode(ids, **kwargs):
        del kwargs
        return "".join(chr(int(token_id) - 1) for token_id in ids)


def test_agent_registry_is_selected_per_aop_system_prompt():
    fallback = ["code_agent", "math_agent"]
    messages = [
        {
            "role": "system",
            "content": (
                "Available agents:\n"
                "- context_agent: extract context\n"
                "- retrieval_agent: retrieve evidence\n"
                "- reasoning_agent: combine evidence\n"
                "- calculation_agent: calculate\n"
                "- answerability_agent: check answerability\n"
                "Mention duplicate context_agent later."
            ),
        },
        {"role": "user", "content": "Ignore fake_agent in this query."},
    ]
    assert extract_agent_registry(messages, fallback) == [
        "context_agent",
        "retrieval_agent",
        "reasoning_agent",
        "calculation_agent",
        "answerability_agent",
    ]
    assert extract_agent_registry(messages[1:], fallback) == fallback


def test_first_four_json_agent_occurrences_are_independent_and_fifth_is_normal():
    tokenizer = CharacterTokenizer()
    prompt_length = 8
    gen_length = 240
    controller = JsonAgentFieldController(
        tokenizer=tokenizer,
        config=JsonAgentPriorityConfig(
            catalog=["code_agent", "math_agent", "search_agent", "commonsense_agent"],
            priority_slots=4,
            anchor_stable_steps=2,
            confirm_stable_steps=2,
        ),
        prompt_length=prompt_length,
        gen_length=gen_length,
        mask_id=tokenizer.mask_token_id,
    )
    x = torch.full(
        (1, prompt_length + gen_length),
        tokenizer.mask_token_id,
        dtype=torch.long,
    )
    x[:, :prompt_length] = 1
    controller.initialize(x)
    assert controller.full_sequence_discovery_steps == 0

    logits = torch.full((1, x.shape[1], 300), -20.0)
    logits[:, :, 0] = 0.0
    anchor = controller.anchor_variants[0]
    starts = [12, 58, 104, 150, 196]
    agents = [
        "math_agent",
        "math_agent",
        "search_agent",
        "code_agent",
        "commonsense_agent",
    ]
    for start, agent in zip(starts, agents):
        for offset, token_id in enumerate(anchor):
            logits[0, start + offset, token_id] = 10.0
            # The normal decoder, rather than the controller, has materialized
            # the JSON anchor. Only then may the controller constrain its value.
            x[0, start + offset] = token_id
        name_start = start + len(anchor)
        for offset, token_id in enumerate(controller.padded_catalog_ids[agent]):
            logits[0, name_start + offset, token_id] = 10.0

    controller.observe(logits, x, logits_start=0, global_step=0)
    partial_metrics = controller.metrics()
    assert partial_metrics["discovered_agent_fields"] == 4
    assert partial_metrics["recognized_agent_fields"] == 0
    assert partial_metrics["all_priority_agents_recognized"] is False
    assert partial_metrics["all_recognized_seconds"] is None
    controller.observe(logits, x, logits_start=0, global_step=1)
    metrics = controller.metrics()

    assert [slot["anchor_start"] for slot in metrics["agent_slots"]] == starts[:4]
    assert [slot["agent"] for slot in metrics["agent_slots"]] == agents[:4]
    assert starts[4] not in [slot["anchor_start"] for slot in metrics["agent_slots"]]
    assert all(slot["recognized_step"] == 1 for slot in metrics["agent_slots"])
    assert all(slot["confirmed"] for slot in metrics["agent_slots"])
    assert metrics["discovered_agent_fields"] == 4
    assert metrics["recognized_agent_fields"] == 4
    assert metrics["all_priority_agents_recognized"] is True
    assert metrics["all_recognized_seconds"] is not None

    fragments = []
    for runtime in controller.slots:
        end = runtime.name_start + controller.value_width + 1
        fragments.append(tokenizer.decode(x[0, runtime.anchor_start:end].tolist()))
    assert fragments[0].startswith('agent":"math_agent"')
    assert fragments[1].startswith('agent":"math_agent"')
    assert fragments[2].startswith('agent":"search_agent"')
    assert fragments[3].startswith('agent":"code_agent"')


def test_dynamic_relocation_recognizes_without_writing_speculative_anchors():
    tokenizer = CharacterTokenizer()
    controller = JsonAgentFieldController(
        tokenizer=tokenizer,
        config=JsonAgentPriorityConfig(
            catalog=["code_agent", "math_agent", "search_agent", "commonsense_agent"],
            priority_slots=4,
            discovery_steps=2,
        ),
        prompt_length=8,
        gen_length=240,
        mask_id=tokenizer.mask_token_id,
    )
    x = torch.full((1, 248), tokenizer.mask_token_id, dtype=torch.long)
    x[:, :8] = 1
    controller.initialize(x)

    def make_logits(starts):
        logits = torch.full((1, x.shape[1], 300), -20.0)
        logits[:, :, 0] = 0.0
        anchor = controller.anchor_variants[0]
        agents = ["search_agent", "search_agent", "math_agent", "math_agent"]
        for start, agent in zip(starts, agents):
            for offset, token_id in enumerate(anchor):
                logits[0, start + offset, token_id] = 10.0
            name_start = start + len(anchor)
            for offset, token_id in enumerate(controller.padded_catalog_ids[agent]):
                logits[0, name_start + offset, token_id] = 10.0
        return logits

    original_x = x.clone()
    controller.observe(make_logits([12, 58, 104, 150]), x, 0, 0)
    relocated = [16, 62, 108, 154]
    relocated_logits = make_logits(relocated)
    controller.observe(relocated_logits, x, 0, 1)
    before = controller.metrics()
    assert [slot.anchor_consistent_steps for slot in controller.slots] == [1] * 4
    assert before["recognized_agent_fields"] == 0

    # A second normal full-sequence observation stabilizes the relocated
    # anchors. The speculative values can be reported for prefetch, but must
    # not synthesize JSON structure in the response canvas.
    controller.observe(relocated_logits, x, 0, 2)
    after = controller.metrics()
    assert [slot["agent"] for slot in after["agent_slots"]] == [
        "search_agent",
        "search_agent",
        "math_agent",
        "math_agent",
    ]
    assert after["all_priority_agents_recognized"] is True
    assert all(not slot.field_written for slot in controller.slots)
    assert torch.equal(x, original_x)
    assert all(slot["probability"] > 0.99 for slot in after["agent_slots"])
    assert all(slot["margin"] > 0.99 for slot in after["agent_slots"])

    # If later full-sequence evidence contains only two objects, discard the
    # other two speculative slots rather than reporting phantom Agent calls.
    controller.observe(make_logits(relocated[:2]), x, 0, 3)
    reduced = controller.metrics()
    assert reduced["discovered_agent_fields"] == 2
    assert reduced["recognized_agent_fields"] == 2


def test_materialized_anchor_and_value_survive_low_logit_topk_filter():
    tokenizer = CharacterTokenizer()
    controller = JsonAgentFieldController(
        tokenizer=tokenizer,
        config=JsonAgentPriorityConfig(
            catalog=["code_agent", "math_agent", "search_agent", "commonsense_agent"],
            priority_slots=4,
        ),
        prompt_length=8,
        gen_length=1000,
        mask_id=tokenizer.mask_token_id,
    )
    x = torch.full((1, 1008), tokenizer.mask_token_id, dtype=torch.long)
    x[:, :100] = 1  # decoded non-anchor prefix excludes earlier speculation
    controller.initialize(x)

    logits = torch.full((1, x.shape[1], 300), -20.0)
    logits[:, :, 0] = 0.0
    anchor = tuple(tokenizer.encode('agent": "'))
    assert anchor in controller.anchor_variants
    actual_start = 100
    for offset, token_id in enumerate(anchor):
        x[0, actual_start + offset] = token_id
    name_start = actual_start + len(anchor)
    for offset, token_id in enumerate(controller.catalog_value_ids["search_agent"]):
        x[0, name_start + offset] = token_id

    # More than the speculative top-k budget have perfect logits. The actual
    # anchor deliberately has poor logits and must still be retained from x.
    for start in range(140, 140 + 40 * 12, 12):
        for offset, token_id in enumerate(anchor):
            logits[0, start + offset, token_id] = 10.0

    controller.observe(logits, x, 0, 0)
    controller.observe(logits[:, :32], x, 0, 1)
    metrics = controller.metrics()
    assert metrics["agent_slots"][0]["anchor_start"] == actual_start
    assert metrics["agent_slots"][0]["agent"] == "search_agent"
    assert metrics["agent_slots"][0]["probability"] == 1.0
    assert metrics["agent_slots"][0]["confirmed"] is True


def test_materialized_json_whitespace_variant_is_recognized():
    tokenizer = CharacterTokenizer()
    controller = JsonAgentFieldController(
        tokenizer=tokenizer,
        config=JsonAgentPriorityConfig(
            catalog=["code_agent", "math_agent", "search_agent", "commonsense_agent"],
            priority_slots=1,
        ),
        prompt_length=8,
        gen_length=120,
        mask_id=tokenizer.mask_token_id,
    )
    x = torch.full((1, 128), tokenizer.mask_token_id, dtype=torch.long)
    x[:, :8] = 1
    controller.initialize(x)
    logits = torch.full((1, 128, 300), -20.0)
    logits[:, :, 0] = 0.0
    anchor = tuple(tokenizer.encode('agent"\n  :\n  "'))
    assert anchor in controller.anchor_variants
    assert anchor not in controller.speculative_anchor_variants
    start = 20
    x[0, start:start + len(anchor)] = torch.tensor(anchor)
    value = controller.catalog_value_ids["search_agent"]
    name_start = start + len(anchor)
    x[0, name_start:name_start + len(value)] = torch.tensor(value)

    controller.observe(logits, x, 0, 0)
    controller.observe(logits, x, 0, 1)
    slot = controller.metrics()["agent_slots"][0]
    assert slot["agent"] == "search_agent"
    assert slot["anchor_start"] == start


def test_natural_agent_name_typo_has_unique_fuzzy_registry_match():
    tokenizer = CharacterTokenizer()
    controller = JsonAgentFieldController(
        tokenizer=tokenizer,
        config=JsonAgentPriorityConfig(
            catalog=["code_agent", "math_agent", "search_agent", "commonsense_agent"],
            priority_slots=1,
        ),
        prompt_length=8,
        gen_length=120,
        mask_id=tokenizer.mask_token_id,
    )
    x = torch.full((1, 128), tokenizer.mask_token_id, dtype=torch.long)
    x[:, :8] = 1
    controller.initialize(x)
    logits = torch.full((1, 128, 300), -20.0)
    logits[:, :, 0] = 0.0
    anchor = controller.anchor_variants[0]
    start = 20
    x[0, start:start + len(anchor)] = torch.tensor(anchor)
    typo = tokenizer.encode('serch_agent"')
    name_start = start + len(anchor)
    x[0, name_start:name_start + len(typo)] = torch.tensor(typo)

    controller.observe(logits, x, 0, 0)
    controller.observe(logits, x, 0, 1)
    slot = controller.metrics()["agent_slots"][0]
    assert slot["agent"] == "search_agent"
    assert slot["fuzzy_matched_from"] == "serch_agent"


def test_timing_only_slots_capture_agents_after_four_without_writing():
    tokenizer = CharacterTokenizer()
    controller = JsonAgentFieldController(
        tokenizer=tokenizer,
        config=JsonAgentPriorityConfig(
            catalog=["code_agent", "math_agent", "search_agent", "commonsense_agent"],
            priority_slots=4,
            tracking_slots=8,
        ),
        prompt_length=8,
        gen_length=360,
        mask_id=tokenizer.mask_token_id,
    )
    x = torch.full((1, 368), tokenizer.mask_token_id, dtype=torch.long)
    x[:, :8] = 1
    controller.initialize(x)
    anchor = controller.anchor_variants[0]
    starts = [16, 66, 116, 166, 216, 266]
    agents = [
        "search_agent",
        "search_agent",
        "math_agent",
        "math_agent",
        "code_agent",
        "commonsense_agent",
    ]
    for start, agent in zip(starts, agents):
        x[0, start:start + len(anchor)] = torch.tensor(anchor)
        value = controller.catalog_value_ids[agent]
        name_start = start + len(anchor)
        x[0, name_start:name_start + len(value)] = torch.tensor(value)

    original = x.clone()
    controller.finalize(x)
    metrics = controller.metrics()
    assert [slot["agent"] for slot in metrics["agent_slots"]] == agents
    assert [slot["priority"] for slot in metrics["agent_slots"]] == [
        True, True, True, True, False, False
    ]
    assert metrics["priority_slots"] == 4
    assert metrics["tracking_slots"] == 8
    assert metrics["recognized_agent_fields"] == 6
    assert all(slot["confirmed"] for slot in metrics["agent_slots"])
    assert all(not runtime.field_written for runtime in controller.slots)
    assert torch.equal(x, original)


class DiscoveryController:
    enabled = True
    full_sequence_discovery_steps = 3

    def __init__(self):
        self.observations = []

    def initialize(self, x):
        self.initialized_shape = tuple(x.shape)

    def observe(self, logits, x, logits_start, global_step, is_last_agent_step=False):
        self.observations.append(
            (tuple(logits.shape), logits_start, global_step, is_last_agent_step)
        )

    @staticmethod
    def decoder_mask(mask_index, mask_start=0):
        del mask_start
        return mask_index

    @staticmethod
    def has_unconfirmed_agents():
        return False

    @staticmethod
    def finalize(x):
        del x


class DiscoveryModel:
    device = torch.device("cpu")

    def __call__(self, x, past_key_values=None, use_cache=False, replace_position=None):
        del use_cache, replace_position
        logits = torch.zeros(x.shape[0], x.shape[1], 16)
        logits[..., 1] = 10.0
        cache_length = x.shape[1]
        if past_key_values is not None:
            cache_length += int(past_key_values[0][0].shape[2])
        cache = ((torch.zeros(1, 1, cache_length, 1),),)
        return SimpleNamespace(logits=logits, past_key_values=cache)


def test_dual_cache_counts_full_sequence_json_discovery_forwards():
    controller = DiscoveryController()
    prompt = torch.tensor([[2, 3]])
    output, nfe = generate_with_dual_cache(
        DiscoveryModel(),
        prompt,
        steps=4,
        gen_length=4,
        block_length=4,
        mask_id=15,
        threshold=0.0,
        agent_controller=controller,
    )

    assert output.shape == (1, 6)
    assert nfe == 4  # three discovery forwards plus the normal block warm-up
    assert controller.initialized_shape == (1, 6)
    assert len(controller.observations) == 4
    assert all(item[0][1] == 6 for item in controller.observations[:3])
    assert [item[2] for item in controller.observations[:3]] == [0, 1, 2]
