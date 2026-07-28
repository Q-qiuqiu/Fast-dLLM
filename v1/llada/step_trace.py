"""Optional per-forward response tracing for LLaDA diffusion decoding."""

from __future__ import annotations

import json
from pathlib import Path


class StepTraceWriter:
    """Write one JSONL record after every model-forward/transfer step.

    Each line contains only the readable generated suffix. Unresolved positions
    are shown as ``MASK``. JSONL is used so long generations do not accumulate
    in memory; the line number is the decoding-step order.
    """

    def __init__(
        self,
        tokenizer,
        prompt_length: int,
        mask_id: int,
        path: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = int(prompt_length)
        self.mask_id = int(mask_id)
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self._closed = False

    def _decode_with_masks(self, token_ids):
        pieces = []
        decoded_run = []

        def flush_run():
            if decoded_run:
                pieces.append(
                    self.tokenizer.decode(
                        decoded_run,
                        skip_special_tokens=False,
                    )
                )
                decoded_run.clear()

        for token_id in token_ids:
            if token_id == self.mask_id:
                flush_run()
                pieces.append("MASK")
            else:
                decoded_run.append(token_id)
        flush_run()
        return "".join(pieces)

    def __call__(self, nfe, block_index, block_step, state) -> None:
        if self._closed:
            return
        suffix_ids = (
            state[0, self.prompt_length :].detach().to(device="cpu").tolist()
        )
        record = {"response": self._decode_with_masks(suffix_ids)}
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
