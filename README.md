# QGRE Engine

**Quality-Gated Reward Escalation.** A single-GPU training engine that grounds LLM reasoning in novel domains — without supervised fine-tuning, without a critic network, without a cluster.

The core claim: SFT teaches models to *match outputs*. QGRE teaches models to *match inputs* — to ground their reasoning in the structure of the problem. A 1.7B model trained with QGRE on Hamiltonian mechanics writes correct physics derivations in 50 steps. Not because it memorized them. Because it learned the protocol.

```
generate → score → segment → advantages → loss → backward → update
One process. One GPU. ~20 seconds per step.
```

---

## The Problem QGRE Solves

Standard RL training assigns one reward per completion. The model cannot distinguish which part of its output earned the reward and which part lost it. For structured reasoning — where format, grounding, chain coherence, and accuracy are separate concerns — a single scalar is noise.

QGRE decomposes the reward into per-quality scores, assigns each to its region of the output, and gates them behind a curriculum. The model learns format before it learns grounding. It learns grounding before it learns accuracy. Each skill becomes the foundation for the next.

This is not prompt engineering. It is the training signal itself, decomposed.

## Results

**Hamiltonian mechanics** (Qwen3-1.7B, 4-bit, RTX 5080):

| Step | Avg Reward | Min | Max | What the model produces |
|------|-----------|-----|-----|------------------------|
| 0 | 0.61 | 0.40 | 0.96 | Guessing — some correct structure by chance |
| 3 | 0.93 | 0.85 | 0.98 | Identifies T, V, derives Hamilton's equations |
| 18 | 0.98 | 0.96 | 1.00 | Near-perfect derivations with `<think>` reasoning |
| 47 | 0.94 | 0.94 | 0.95 | Converged — consistent quality across prompts |

50 steps. No SFT warm-up. No hand-crafted examples. The curriculum *is* the warm-up.

```
SCORE: 0.98
<think>
Okay, so I need to derive the Hamiltonian H(x, p) from first principles
for a block attached to a spring on a frictionless surface. The block has
mass 3 kg, the spring constant is 6 N/m...
```

---

## Quick Start

```bash
pip install -e .

# Hamiltonian mechanics (SPO mode, verifiable via sympy)
python -m qgre train \
  --config examples/hamiltonian/config.yaml \
  --reward examples.hamiltonian.reward_fn:hamiltonian_reward

# Single-step math
python -m qgre train \
  --config examples/math/config.yaml \
  --reward examples.math.reward_fn:math_reward_fn
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     QGRETrainer.step()                          │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌─────────┐ │
│  │ Generate  │──▸│  Score   │──▸│  Advantages  │──▸│  Loss   │ │
│  │ (vLLM)   │   │(reward_fn)│  │  (SPO+GDPO   │   │(NeMo RL)│ │
│  │          │   │          │   │  +VPRM+phase) │   │         │ │
│  └──────────┘   └──────────┘   └──────────────┘   └────┬────┘ │
│       ▲                                                 │      │
│       │              ┌──────────────┐                   ▼      │
│       └──────────────│ LoRA Sync    │◂── backward + optimizer  │
│                      │ (save/load)  │         step             │
│                      └──────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

No Ray. No verl. No TRL. Direct function calls on a single GPU.

## How QGRE Works

### Four techniques, one pipeline

**1. SPO** — Single-stream Policy Optimization. A persistent EMA baseline tracks what each prompt usually scores, per step. `V = V + lr × (r − V)`. No groups needed. Works with n=1 — one completion per prompt, every completion teaches.

**2. GDPO** — Group Decomposed normalization. Each step's advantages are normalized independently across the batch. Format advantages don't drown grounding advantages. Every step gets equal gradient bandwidth.

**3. VPRM** — Verifiable Process Reward Mapping. A segmenter splits the completion into regions (THINK, STEP_1, STEP_2, FORMAT). Each token receives the advantage of its step. Think tokens receive zero advantage — reasoning is free.

**4. Phase Gating** — The curriculum. Phase 1 activates only step 1 qualities. When mastery exceeds the threshold (default 0.8 over 20 batches), phase 2 activates step 2 qualities cumulatively. The model masters format before grounding, grounding before accuracy.

```
Phase 1: [q_format]                    → mastery > 0.8 → advance
Phase 2: [q_format, q_grounding]       → mastery > 0.8 → advance
Phase 3: [q_format, q_grounding, ...]  → mastery > 0.8 → advance
Phase N: all qualities active          → full training
```

The engine manages this via `GameState`. No external curriculum logic needed.

### Stagnation detection

The engine monitors training progress per phase. If mastery plateaus (improvement < 0.02 over 50 steps) or a phase exceeds a timeout (default 200 steps), the engine logs a stagnation signal to MLflow. Detection only — intervention is left to the training run configuration.

## Bring Your Own Domain

Three things. That's it.

### 1. A reward function

```python
from qgre.types import RewardResult

def my_reward_fn(prompt: str, completion: str, metadata: dict | None = None) -> RewardResult:
    scores = {
        "q_format": check_format(completion),       # 0.0 – 1.0, always partial credit
        "q_grounding": check_grounding(completion),  # never binary
        "q_accuracy": check_accuracy(completion),
    }
    return RewardResult(reward=sum(scores.values()) / len(scores), scores=scores)
```

Every score gives partial credit. Binary 0/1 kills gradient signal.

### 2. A step_qualities mapping

```yaml
algorithm:
  step_qualities:
    1: [q_format]
    2: [q_grounding]
    3: [q_accuracy]
```

### 3. A segmenter (optional)

| Segmenter | Config value | Use case |
|-----------|-------------|----------|
| `uniform` | `segmenter: uniform` | Single-step domains (math, Q&A). All tokens → STEP_1. |
| `qwen3_xml` | `segmenter: qwen3_xml` | Multi-step XML tags (`<step1>...<step2>..`). Token ID pattern matching. |
| `hif_json` | `segmenter: hif_json` | HIF JSON output. Decode-and-regex on `nodes`, `edges`, `incidences`, `scan-results`. |
| Custom | `segmenter: my_module:my_fn` | Any `Callable[[list[int]], list[str]]`. |

## Full Config

```yaml
model:
  path: unsloth/Qwen3-1.7B-unsloth-bnb-4bit
  lora_rank: 8
  lora_alpha: 16
  load_in_4bit: true
  fast_inference: true
  gpu_memory_utilization: 0.35

data:
  train_files: [data/train.parquet]
  max_prompt_length: 3200
  train_batch_size: 16
  prompt_column: prompt
  metadata_columns: [ground_truth]

generation:
  temperature: 1.0
  top_p: 1.0
  max_tokens: 4096
  stop_token_ids: [151643, 151645]    # Qwen3: <|endoftext|> + <|im_end|>

algorithm:
  mode: spo                            # "spo" (n=1) or "grpo" (n=8)
  segmenter: uniform                   # "uniform", "qwen3_xml", "hif_json", or "module:fn"
  reference_policy_kl_type: k3         # "k1" (unbiased), "k2" (squared), "k3" (exponential)

  spo:
    lr: 0.1
    n: 1

  clip_ratio_low: 0.2
  clip_ratio_high: 0.28
  loss_mode: pg                        # "pg" (no KL) or "kl_cov"
  kl_cov_ratio: 0.0
  llds_coef: 0.05                      # LLDS collapse prevention (arXiv:2512.04220)
  loss_type: grpo                      # "grpo" or "dr_grpo" (unbiased, arXiv:2503.20783)

  kl_think_multiplier: 0.1            # Low KL on reasoning tokens
  kl_format_multiplier: 2.0           # High KL on structural tokens
  kl_step_multiplier: 1.0

  lambda_return: 0.0                   # Eligibility traces (0=off)
  length_penalty_coef: 0.0             # Dynamic length control (0=off)

  step_qualities:
    1: [q_format]
    2: [q_grounding]
    3: [q_accuracy]

training:
  total_steps: 800
  lr: 5.0e-6
  warmup_steps: 10
  lr_scheduler: cosine
  save_freq: 50
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  mastery_threshold: 0.8
  stagnation_timeout: 200              # Steps before STUCK signal
  plateau_window: 50                   # Steps to measure plateau slope
  plateau_threshold: 0.02              # Minimum improvement to avoid STAGNATING

logging:
  mlflow_experiment: my-experiment
  completion_dir: output/completions
  checkpoint_dir: output/checkpoints
```

## Programmatic API

```python
from qgre.config import QGREConfig
from qgre.generation import UnslothBackend
from qgre.trainer import QGRETrainer
from qgre.data import QGREDataLoader, load_prompts_from_parquet

cfg = QGREConfig.from_yaml("config.yaml")
backend = UnslothBackend(cfg.model, cfg.generation)
model, tokenizer = backend.load()

prompts = load_prompts_from_parquet("data/train.parquet")
loader = QGREDataLoader(
    prompts=prompts, tokenizer=tokenizer,
    max_prompt_length=cfg.data.max_prompt_length,
    train_batch_size=cfg.data.train_batch_size,
    n_completions=cfg.algorithm.spo.n,
    metadata_columns=cfg.data.metadata_columns,
)

trainer = QGRETrainer(
    model=model, tokenizer=tokenizer,
    reward_fn=my_reward_fn, config=cfg,
    generation_backend=backend,
)

trainer.train(loader, backend)
```

## What's Built In

Research-backed features. All opt-in via config.

| Feature | Config | Source |
|---------|--------|--------|
| SPO persistent baseline | `mode: spo` | SPO (Tencent, ICLR 2026) |
| GDPO per-step normalization | Always on | GDPO (NVIDIA, Jan 2026) |
| VPRM segment propagation | Via segmenter | VPRMs (IBM Research, Jan 2026) |
| Phase-gated curriculum | Via step_qualities | QGRE (this work) |
| Stagnation detection | `stagnation_timeout`, `plateau_window` | Scaf-GRPO informed |
| LLDS collapse prevention | `llds_coef: 0.05` | arXiv:2512.04220 |
| Dr.GRPO unbiased loss | `loss_type: dr_grpo` | arXiv:2503.20783 |
| KL estimator selection | `reference_policy_kl_type: k1` | Comedy of Estimators (ICLR 2026) |
| Region-specific KL | `kl_think_multiplier: 0.1` | Archer (ICLR 2026) |
| KL-adaptive SPO lr | `spo.kl_adaptive: true` | SPO Algorithm 1 |
| Prioritized sampling | Auto (SPO mode) | SPO Section 3.2 |
| Eligibility traces | `lambda_return: 0.95` | GRPO-λ (ICLR 2026) |
| Dynamic length control | `length_penalty_coef: 0.01` | Huawei |
| Completion length tracking | Always on | Verbosity drift detection |
| GDPO NaN guard | Always on | ms-swift #8123 |
| AdamW 8-bit | Automatic | bitsandbytes |
| selective_log_softmax | Always on | TRL PR #2799 |
| Triton fused logprobs | Auto (if available) | Custom kernel |

## Checkpoint & Resume

Full state saved and restored automatically:
- Model weights (LoRA adapters)
- Optimizer state (AdamW8bit)
- LR scheduler
- GameState (phase, mastery windows, stagnation counters, phase history)
- SPO value tracker (V per prompt per step)
- PyTorch + CUDA RNG state

Resume is automatic. `trainer.train()` finds the latest checkpoint and continues.

## File Structure

```
qgre/
  __init__.py          — Exports RewardResult, GameState, StagnationStatus
  __main__.py          — CLI: python -m qgre train --config --reward --segmenter
  types.py             — RewardResult, GameState, StagnationStatus
  config.py            — All config dataclasses + YAML loader
  segments.py          — Segmenters: qwen3_xml, hif_json, uniform, custom
  advantages.py        — QGREStepAdvantageEstimator (SPO+GDPO+VPRM+phase)
  data.py              — Parquet → tokenize → left-pad → batch → prioritized sampling
  checkpoint.py        — Save/resume full training state
  logging.py           — MLflow metrics + JSONL completion logs
  trainer.py           — QGRETrainer: the training loop
  generation.py        — UnslothBackend: vLLM colocated generation
  lora_verify.py       — LoRA weight sync verification
  fused_logprobs.py    — Chunked logprobs (no full logits materialization)
  triton_logprobs.py   — Triton fused lm_head→logprobs kernel
  nemo_extracted/      — ClippedPGLossFn, KL, logits (Apache-2.0 from NeMo RL)
examples/
  hamiltonian/         — Physics derivation (SPO, verifiable via sympy)
  hypergraph/          — Multi-step XML structured output (SPO)
  math/                — Single-step math
tests/                 — 130 CPU tests + 9 GPU tests
```

## Tests

```bash
python -m pytest tests/ -q                    # All CPU tests (~27s)
python -m pytest tests/test_segments.py -v    # Segmentation (XML + HIF JSON)
python -m pytest tests/test_advantages.py -v  # Advantage computation
python -m pytest tests/test_smoke.py --gpu -v # GPU smoke test (requires Qwen3-1.7B)
```

## Known Constraints

- **16GB VRAM**: Qwen3-1.7B 4-bit at `gpu_memory_utilization=0.35` peaks at 6.2GB. 8B requires 0.6+ and tighter micro-batching.
- **On-policy only**: `force_on_policy_ratio=True` means ratio clipping has no effect. The model trains on what it just generated. KL regularization requires stored generation-time logprobs (not yet implemented).
- **Segmenters are model-specific**: `qwen3_xml` uses Qwen3 token IDs. `hif_json` uses decoded text. Other models need custom segmenters or `uniform`.
- **vLLM recreation**: Engine recreates the vLLM backend every 50 steps to prevent VRAM leak (Unsloth #3864). Failures are logged, not silent.

## References

- QGRE paper: (forthcoming)
- [SPO](https://arxiv.org/abs/2509.13232) — Single-stream Policy Optimization (Tencent, ICLR 2026)
- [GDPO](https://arxiv.org/abs/2601.05242) — Group Decomposed Policy Optimization (NVIDIA, Jan 2026)
- [VPRMs](https://arxiv.org/abs/2601.17223) — Verifiable Process Rewards (IBM Research, Jan 2026)
- [Dr.GRPO](https://arxiv.org/abs/2503.20783) — Unbiased GRPO (Mar 2025)
- [LLDS](https://arxiv.org/abs/2512.04220) — Lazy Likelihood Displacement (Dec 2025)
- [Comedy of Estimators](https://arxiv.org/abs/2512.21852) — KL estimator analysis (Bengio et al., Dec 2025)
- [Archer](https://openreview.net/forum?id=ee326398473daf76d49b49cda4dea9d699fbf61b) — Dual-token KL constraints (ICLR 2026)
- [Scaf-GRPO](https://arxiv.org/abs/2510.19807) — Scaffolded progressive training (Feb 2026)
- [NeMo RL](https://github.com/NVIDIA-NeMo/RL) — Loss functions (Apache-2.0)

## License

Apache-2.0. NeMo RL extracted components retain their original Apache-2.0 headers.

---

Built by [Torad Labs](https://torad.ai). The engine behind the QGRE paper.
