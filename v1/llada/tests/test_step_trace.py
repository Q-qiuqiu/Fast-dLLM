import json

import torch

from step_trace import StepTraceWriter
from test_agent_priority import FakeTokenizer


def test_step_trace_writes_mask_ids_tokens_and_readable_response(tmp_path):
    state = torch.tensor([[90, 91, 11, 15, 12, 15]])
    path = tmp_path / "trace.jsonl"
    writer = StepTraceWriter(
        tokenizer=FakeTokenizer(),
        prompt_length=2,
        mask_id=15,
        path=str(path),
    )
    writer(1, 0, 0, state)
    writer.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1
    assert list(records[0]) == ["response"]
    assert records[0]["response"] == "search recent methods MASKmodify decoder MASK"


def test_step_trace_overwrites_an_existing_trace(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text("old data\n", encoding="utf-8")
    writer = StepTraceWriter(FakeTokenizer(), 1, 15, str(path))
    writer(3, 1, 2, torch.tensor([[90, 15]]))
    writer.close()

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record == {"response": "MASK"}
    assert "old data" not in path.read_text(encoding="utf-8")
