# Fast-dLLM v1: Training-free Acceleration of Diffusion LLM

[![Project](https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages)](https://nvlabs.github.io/Fast-dLLM)
[![arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2505.22618)
<a href="https://fast-dllm.hanlab.ai"><img src="https://img.shields.io/static/v1?label=Demo&message=Fast-dLLM&color=yellow"></a> &ensp;

Fast-dLLM v1 is a **training-free** inference acceleration framework for diffusion-based Large Language Models (dLLMs). It supports efficient inference for models like **Dream** and **LLaDA** by enabling KV Cache and Parallel Decoding.

## Key Features

1. **Key-Value Cache for Block-Wise Decoding**
   We propose an efficient block-wise decoding KV Cache mechanism for Masked Diffusion Models (MDMs). By reusing attention Key-Value activations across multiple steps within each block, our approach avoids redundant computation and significantly accelerates inference. Furthermore, our DualCache extension also caches masked suffix tokens, enabling even greater speedup with negligible accuracy loss.

<div align="center">
  <img src="asset/kvcache.jpg" alt="KV Cache for block-wise decoding" width="800"/>
  <p>KV Cache for block-wise decoding</p>
</div>

2. **Confidence-Aware Parallel Decoding**
   Instead of decoding tokens sequentially, we introduce a confidence-aware parallel decoding scheme. At each step, only tokens with confidence over a threshold are unmasked in parallel, while uncertain ones remain masked for future steps. This selective approach effectively balances decoding efficiency and output quality.

<div align="center">
  <img src="asset/output.gif" alt="Decoding comparison" width="800"/>
  <p><b>Left:</b> Standard decoding (LLaDA). <b>Right:</b> Confidence-aware parallel decoding.</p>
</div>

<div align="center">
  <img src="asset/pseudo_code.jpg" alt="Pseudo code for our method" width="800"/>
  <p>Pseudo code for our method</p>
</div>

3. **Overall Performance**
   Overall, introducing the KV Cache mechanism yields significant speed improvements for all tasks and sequence lengths, typically achieving a 2x to 3.6x speedup compared to the vanilla backbone. When the parallel decoding strategy is applied individually, we see additional acceleration, often pushing speedups to 4x-6x for the evaluated settings, particularly as the generation length increases.

<div align="center">
  <img src="asset/overall_performance.jpg" alt="Overall performance" width="800"/>
  <p>Overall performance comparison</p>
</div>

## Demo

https://github.com/user-attachments/assets/32bbff97-6e60-4e14-95c0-2cbec136476f

<div align="center">
  <img src="asset/speedup.jpg" alt="End-to-end speedup over vanilla LLaDA baseline" width="800"/>
  <p>End-to-end speedup over vanilla LLaDA baseline</p>
</div>

## File Structure

```
v1/
├── README.md               # This file
├── requirements.txt        # Dependencies for inference & evaluation
├── dream/                  # Dream model related code
│   ├── model/              # Dream model definition
│   ├── eval.py             # Evaluation harness integration
│   ├── eval.md             # Evaluation guide
│   ├── eval_gsm8k.sh       # GSM8K evaluation script
│   ├── eval_humaneval.sh   # HumanEval evaluation script
│   └── demo_multiturn_chat.py  # Multi-turn chat demo
└── llada/                  # LLaDA model related code
    ├── model/              # LLaDA model definition
    ├── generate.py         # Core generation with cache & parallel decoding
    ├── eval_llada.py       # Evaluation harness integration
    ├── eval.md             # Evaluation guide
    ├── eval_gsm8k.sh       # GSM8K evaluation script
    ├── eval_humaneval.sh   # HumanEval evaluation script
    ├── chat.py             # Command-line chat interface
    └── app.py              # Gradio web demo
```

## Installation

```bash
cd v1
pip install -r requirements.txt
```

## Usage

### 1. Using LLaDA Model

#### Interactive Chat
```bash
python llada/chat.py --gen_length 128 --steps 128 --block_size 32
```

Parameter descriptions:
- `--gen_length`: Maximum length of generated text
- `--steps`: Number of sampling steps
- `--block_size`: Cache block size
- `--use_cache`: Whether to use cache
- `--if_cache_position`: Whether to use dual cache
- `--threshold`: Confidence threshold

#### Web Demo
```bash
pip install gradio
cd llada
python app.py
```

#### OpenAI-Compatible API Server

Start the Fast-dLLM v1 LLaDA server from the repository root:

```bash
python v1/llada/fastdllm_server.py \
  --gen-length 256 \
  --block-size 32 \
  --cache-mode dual
```

These are the only inference arguments needed at startup. The server follows
the original `generate.py` defaults: 128 total diffusion steps and
`threshold=None`. The generation functions divide the total steps by the number
of blocks internally. Model path, served model name, host, port, device, and
dtype use defaults in `fastdllm_server.py`; they can be overridden with the
corresponding `FASTDLLM_*` environment variables when necessary.

For example, set `FASTDLLM_MODEL_PATH` to use another local model, or set
`FASTDLLM_API_KEY` to require an `Authorization: Bearer ...` header. Without the
API key environment variable, authentication is disabled.

Call the non-streaming Chat Completions endpoint:

```bash
curl http://127.0.0.1:7004/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/data/labshare/Param/llada",
    "messages": [{"role": "user", "content": "Introduce Fast-dLLM."}],
    "max_tokens": 256,
    "temperature": 0
  }'
```

The server also provides `GET /health` and `GET /v1/models`. Requests with
`stream=true` return HTTP 400 because LLaDA masked-diffusion decoding does not
produce stable incremental tokens. The response's extra `fastdllm` object reports
the number of forward evaluations, generation time, and generation TPS.

Python clients can use the standard OpenAI SDK by setting the base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:7004/v1", api_key="empty")
response = client.chat.completions.create(
    model="/data/labshare/Param/llada",
    messages=[{"role": "user", "content": "Introduce Fast-dLLM."}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

#### Agent-name-first planner decoding

`llada/agent_priority.py` adds an optional catalog-constrained planner mode to
all three LLaDA generation functions (`generate`, Prefix Cache, and Dual
Cache). It reserves a compact routing region in the first block for up to four
complete Agent names, while the ordinary Fast-dLLM transfer policy continues
to decode task tokens. The public result is always rendered as four slots:

```text
<subtask>
agent_name: search_agent
task: search recent diffusion acceleration methods
</subtask>

<subtask>
agent_name: none
task: none
</subtask>
```

The other two slots follow the same format. `none` is built in and is never
preloaded. Every other name must appear in the configured Agent Catalog; see
`llada/config/agent_catalog.example.json` for cold-start, wrong-preload-cost, and
residency fields.

Run the one-shot example from `v1/llada`:

```bash
python inference.py \
  --catalog config/agent_catalog.example.json \
  --gen-length 128 \
  --steps 128 \
  --block-size 32 \
  --policy now \
  --cache-mode prefix \
  --agent-log-path logs/agent_decode.log \
  --save-step-trace \
  --step-trace-path traces/decode_trace.jsonl
```

Verbose output is written to `agent_decode.log` by default instead of being
printed to the terminal. `--agent-log-path` selects another file. Logs rotate
at 20 MiB and retain three backups. The file records every slot's field-level
candidate probabilities, top-1/top-2 margin, distribution change, consecutive
stable steps, state transition, preload benefit, and the non-blocking
`PRELOAD_START`, `PRELOAD_CANCEL`, `PRELOAD_SWITCH`, and `AGENT_CONFIRMED`
events. The planner server provides the equivalent `--agent_log_path` option.

The default state-machine configuration tolerates one transient confidence
drop before cancelling a tentative preload. A field below the normal
confirmation probability can also confirm through the stable-plateau path
after four consistent steps (`probability >= 0.52`, `margin >= 0.20`, and
distribution change `<= 0.02`). These values are configurable through
`AgentPriorityConfig`. Decode-time estimation is reset by `generate*` after
model loading and uses a clipped EMA so a slow first Dual Cache forward does
not dominate preload-benefit estimates.

`--save-step-trace` is disabled by default. When enabled, one JSON object is
written after every forward/transfer step. Each line contains only a readable
`response`; unresolved positions appear as `MASK`, and the line number is the
decoding-step order. The trace is streamed to JSONL and overwrites an existing
file at `--step-trace-path`. This diagnostic mode copies the state to CPU and
flushes every step, so disable it for latency/throughput benchmarks.

`--policy` selects the prompt/Agent strategy independently from `--cache-mode`:

| Policy | Prompt | Agent-priority controller | Output |
|---|---|---|---|
| `raw` | Original user query | Off | Original model text |
| `mid` | Compact planner prompt | Off | Original model text |
| `now` | Compact planner prompt | On | Four rendered `<subtask>` slots |

For strict original LLaDA use `--policy raw --cache-mode none`. For the
same-prompt Dual Cache baseline use `--policy mid --cache-mode dual`. The
current Agent-name-first implementation is `--policy now --cache-mode dual`.
The planner server exposes the same `--policy` option (with its existing
`--cache_mode` spelling). At the Python API level, omitting `agent_controller`
still preserves the original `generate*` behavior.

Run the CPU-only unit and integration tests without loading LLaDA weights:

```bash
cd v1/llada
pip install -r ../requirements-test.txt
python -m pytest -q tests/test_agent_priority.py tests/test_generate_agent_integration.py
```

#### Model Evaluation
| Benchmark         | Gen Length | LLaDA   | +Cache         | +Parallel      | +Cache+Parallel (Fast-dLLM) |
|-------------------|------------|---------|----------------|----------------|-----------------------------|
| **GSM8K (5-shot)**| 256        | 79.3<br>6.73<br>(1×) | 79.5<br>21.23<br>(3.2×) | 79.2<br>16.53<br>(2.5×) | 78.5<br>**54.4<br>(8.1×)** |
|                   | 512        | 77.5<br>3.23<br>(1×) | 77.0<br>10.43<br>(3.3×) | 77.6<br>18.63<br>(5.8×) | 77.2<br>**35.3<br>(11.0×)** |
| **HumanEval (0-shot)** | 256   | 41.5<br>30.5 (1×) | 42.7<br>40.73<br>(1.3×) | 43.9<br>101.53<br>(3.3×) | 43.3<br>**114.1<br>(3.7×)** |
|                   | 512        | 43.9<br>18.4 (1×) | 45.7<br>29.33<br>(1.6×) | 43.3<br>57.13<br>(3.1×) | 44.5<br>**73.7<br>(4.0×)** |

Each cell presents the accuracy (top row, in percentage) and the decoding throughput (middle row, in tokens per second) with relative speedup (bottom row) to the LLaDA baseline.

For detailed evaluation instructions, please refer to:
- [LLaDA Evaluation Guide](llada/eval.md)
- [Dream Evaluation Guide](dream/eval.md)

### 2. Using Dream Model

For detailed evaluation instructions on GSM8K and HumanEval benchmarks, please refer to [Dream Evaluation Guide](dream/eval.md).

## Citation

```bibtex
@misc{wu2025fastdllmtrainingfreeaccelerationdiffusion,
      title={Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding}, 
      author={Chengyue Wu and Hao Zhang and Shuchen Xue and Zhijian Liu and Shizhe Diao and Ligeng Zhu and Ping Luo and Song Han and Enze Xie},
      year={2025},
      eprint={2505.22618},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.22618}, 
}
```

## Acknowledgements

We would like to thank the authors of [LLaDA](https://github.com/llada-project/llada) and [Dream](https://github.com/dream-project/dream) for their excellent work and open-source contributions.
