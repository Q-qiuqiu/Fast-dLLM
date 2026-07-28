from types import SimpleNamespace

import torch

from agent_priority import (
    AgentDecodingController,
    AgentFieldState,
    AgentPriorityConfig,
    AgentSpec,
)
import pytest

from generate import generate, generate_with_dual_cache, generate_with_prefix_cache
from test_agent_priority import FakeTokenizer


class FakeModel:
    device = torch.device("cpu")

    def __init__(self, controller=None):
        self.controller = controller

    def __call__(self, x):
        logits = torch.zeros(x.shape[0], x.shape[1], 16)
        logits[..., 11] = 5.0
        if self.controller is not None:
            for start, _ in self.controller.layout.agent_spans:
                for offset, token_id in enumerate(
                    self.controller.catalog_token_ids["search_agent"]
                ):
                    logits[:, start + offset, token_id] = 10.0
        return SimpleNamespace(logits=logits)


class CacheFakeModel(FakeModel):
    def __call__(
        self,
        x,
        past_key_values=None,
        use_cache=False,
        replace_position=None,
    ):
        if replace_position is not None:
            logits_start = int(torch.nonzero(replace_position[0])[0])
        elif past_key_values is not None:
            logits_start = int(past_key_values[0][0].shape[2])
        else:
            logits_start = 0
        logits = torch.zeros(x.shape[0], x.shape[1], 16)
        logits[..., 11] = 5.0
        if self.controller is not None:
            for start, _ in self.controller.layout.agent_spans:
                relative = start - logits_start
                if relative < 0 or relative >= x.shape[1]:
                    continue
                for offset, token_id in enumerate(
                    self.controller.catalog_token_ids["search_agent"]
                ):
                    if relative + offset < x.shape[1]:
                        logits[:, relative + offset, token_id] = 10.0
        cache_length = x.shape[1] if past_key_values is None else logits_start + x.shape[1]
        cache = ((torch.zeros(1, 1, cache_length, 1),),)
        return SimpleNamespace(logits=logits, past_key_values=cache)


def build_controller(enabled=True):
    return AgentDecodingController(
        tokenizer=FakeTokenizer(),
        config=AgentPriorityConfig(
            catalog=[AgentSpec("search_agent", 2.0, 0.1)],
            enabled=enabled,
            slots=4,
            tentative_probability=0.4,
            tentative_margin=0.1,
            confirm_probability=0.6,
            confirm_margin=0.2,
            max_distribution_change=0.05,
            stable_steps=2,
        ),
        prompt_length=2,
        gen_length=32,
        block_length=16,
        total_steps=32,
        mask_id=15,
    )


def test_generate_integration_confirms_catalog_fields_and_decodes_tasks():
    controller = build_controller()
    model = FakeModel(controller)
    traced_steps = []
    output, nfe = generate(
        model,
        torch.tensor([[1, 2]]),
        steps=32,
        gen_length=32,
        block_length=16,
        mask_id=15,
        agent_controller=controller,
        step_callback=lambda nfe, block, step, state: traced_steps.append(
            (nfe, block, step)
        ),
    )
    plan = controller.plan(output)

    assert nfe > 0
    assert all(slot.state == AgentFieldState.CONFIRMED for slot in controller.slots)
    assert plan.agents == ("search_agent",) * 4
    assert all(task.startswith("search recent methods") for task in plan.tasks)
    assert plan.render().count("<subtask>") == 4
    assert len(traced_steps) == nfe
    assert [item[0] for item in traced_steps] == list(range(1, nfe + 1))


def test_disabled_controller_restores_original_generate_path():
    prompt = torch.tensor([[1, 2]])
    baseline, baseline_nfe = generate(
        FakeModel(),
        prompt,
        steps=16,
        gen_length=16,
        block_length=16,
        mask_id=15,
    )
    disabled = AgentDecodingController(
        tokenizer=FakeTokenizer(),
        config=AgentPriorityConfig(
            catalog=[AgentSpec("search_agent")],
            enabled=False,
            slots=1,
        ),
        prompt_length=2,
        gen_length=16,
        block_length=16,
        total_steps=16,
        mask_id=15,
    )
    actual, actual_nfe = generate(
        FakeModel(),
        prompt,
        steps=16,
        gen_length=16,
        block_length=16,
        mask_id=15,
        agent_controller=disabled,
    )

    assert torch.equal(actual, baseline)
    assert actual_nfe == baseline_nfe


@pytest.mark.parametrize(
    "generate_fn", [generate_with_prefix_cache, generate_with_dual_cache]
)
def test_agent_controller_runs_in_cache_decoders(generate_fn):
    controller = build_controller()
    traced_steps = []
    output, nfe = generate_fn(
        CacheFakeModel(controller),
        torch.tensor([[1, 2]]),
        steps=32,
        gen_length=32,
        block_length=16,
        mask_id=15,
        agent_controller=controller,
        step_callback=lambda nfe, block, step, state: traced_steps.append(
            (nfe, block, step)
        ),
    )

    plan = controller.plan(output)
    assert plan.agents == ("search_agent",) * 4
    assert all(slot.state == AgentFieldState.CONFIRMED for slot in controller.slots)
    assert len(traced_steps) == nfe
    assert [item[0] for item in traced_steps] == list(range(1, nfe + 1))
