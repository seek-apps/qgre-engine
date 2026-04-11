# QGRE Engine

**Quality-Gated Reward Escalation** — a single-GPU reinforcement learning engine for LLMs.

No TRL. No verl. No Ray. One process, one GPU, direct function calls. ~16,600 lines of Python.

## The Problem

Supervised fine-tuning matches outputs. The model learns to produce *text that looks like* the answer. It does not learn to produce *text that is* the answer. The distinction matters when the domain has verifiable correctness — when you can check whether a Hamiltonian satisfies Hamilton's equations, when you can differentiate symbolically to confirm self-consistency, when the ground truth is a mathematical object and not a string.

SFT teaches format. RL teaches grounding. QGRE is the RL engine.

## Current Results

A 1.7B parameter model (Qwen3-1.7B, 4-bit quantized) trained with QGRE on a single RTX 5080 (16 GB) derives Hamiltonian mechanics from first principles. The reward function uses [math-verify](https://github.com/huggingface/Math-Verify) and sympy to check symbolic equivalence — not string matching, not format scoring. The model writes kinetic energy, potential energy, composes the Hamiltonian, derives Hamilton's equations of motion, and maintains self-consistency. Verified by symbolic differentiation.

Step 3000+ (April 2026):

| Quality | Accuracy | Notes |
|---|---|---|
| V (potential energy) | ~95% | |
| T (kinetic energy) | ~72% | Remaining failures are correct generic formulas the reward function now recognizes |
| H (Hamiltonian) | ~85% | |
| Hamilton's equations | ~62-65% | dq/dt and dp/dt combined |
| Self-consistency | ~52% | Model's equations checked against its own H |
| Perfect score (6/6 = 1.0) | ~30% of completions | |

Typical reward per step: 0.667-0.833. VRAM: 11 GB steady, zero growth over 3000+ steps. LoRA rank 32, 4-bit quantization, cosine LR schedule.

The model started from zero. No SFT warmup. No Hamiltonian mechanics in pretraining. Phase-gated curriculum with a skill tree — freefall first, then spring, then compound systems, then boss problems (driven oscillator). The model learned what a Hamiltonian is through reinforcement alone.

## Why This Exists

Existing RL engines assume a specific deployment topology. verl requires 4-8 A100s and Ray. TRL requires a group of 8 completions per prompt. OpenRLHF requires a separate critic model. All three apply a single scalar reward uniformly across every token in the completion.

QGRE assumes one GPU and asks: what if credit assignment were the product, not an afterthought?

| | QGRE | verl | TRL | OpenRLHF |
|---|---|---|---|---|
| Gradient target | Per-token, per-quality, per-step | Per-completion scalar | Per-completion scalar | Per-completion + critic |
| Cold start | Phase gating + skill tree from zero | Requires SFT | Requires SFT | Requires SFT |
| Credit assignment | Segmented: each token gets the gradient for its quality | Uniform scalar | Uniform scalar | Learned critic |
| Completions per prompt | 1 (SPO) | 8-16 (GRPO) | 8 (GRPO) | 1 (PPO + critic) |
| GPU requirement | 1 x 16 GB consumer | 4-8 x A100 80 GB | 1-4 GPUs | 4-8 GPUs |
| Lines of code | ~16,600 | ~50,000 | ~30,000 | ~40,000 |

The existing engines were not built wrong. They were built for a different problem — RLHF with scalar preference signals. QGRE addresses the problem that appears when rewards are structured, verifiable, and decomposable into per-region per-quality signals.

---

## Quick Start

### Requirements

- Python >= 3.10
- PyTorch >= 2.4.0
- CUDA GPU with >= 16 GB VRAM (tested on RTX 5080, RTX 4090)
- [Unsloth](https://github.com/unslothai/unsloth) >= 2026.3.5 (provides vLLM integration and 4-bit quantization)

### Install

```bash
git clone https://github.com/torad-labs/qgre-engine.git
cd qgre-engine
pip install -e ".[unsloth,hamiltonian,dev]"
```

### Run the Hamiltonian Example

```bash
python -m qgre train \
  --config examples/hamiltonian/config.yaml \
  --reward examples.hamiltonian.reward_fn_v2:hamiltonian_reward
```

This trains Qwen3-1.7B (4-bit) to derive Hamiltonian mechanics. The reward function uses math-verify and sympy for symbolic equivalence checking. Training logs to MLflow.

### Minimal Config

```yaml
model:
  path: unsloth/Qwen3-1.7B-unsloth-bnb-4bit
  lora_rank: 32
  lora_alpha: 64
  load_in_4bit: true
  fast_inference: true
  gpu_memory_utilization: 0.25
  weight_sync_strategy: merge
  pad_token: "<|fim_pad|>"
  pad_token_id: 151662
  lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  modules_to_save: [lm_head]

data:
  train_files:
    - path/to/train.parquet
  max_prompt_length: 1024
  train_batch_size: 4
  prompt_column: prompt
  metadata_columns: [ground_truth]

generation:
  temperature: 0.7
  max_tokens: 4096
  stop_token_ids: [151643, 151645]

algorithm:
  mode: spo
  loss_type: dr_grpo
  use_fused_logprobs: true
  use_triton_logprobs: true
  step_qualities:
    1: [q_correctness]

training:
  total_steps: 5000
  lr: 5.0e-6
  lr_scheduler: cosine
  save_freq: 50
```

### Write a Reward Function

```python
from qgre.types import RewardResult

def my_reward(completion: str, metadata: dict) -> RewardResult:
    """Score a completion against ground truth.

    Returns RewardResult with per-quality scores in [0, 1].
    The scores dict keys must match the quality names in step_qualities.
    """
    correct = check_answer(completion, metadata["ground_truth"])
    return RewardResult(
        reward=1.0 if correct else 0.0,
        scores={"q_correctness": 1.0 if correct else 0.0},
    )
```

For fine-grained credit assignment, return `scored_spans` — character offsets where each quality was found:

```python
return RewardResult(
    reward=overall_score,
    scores={"q_step1": 0.8, "q_step2": 1.0},
    scored_spans={
        "q_step1": {"start": 45, "end": 120, "score": 0.8},
        "q_step2": {"start": 125, "end": 210, "score": 1.0},
    },
)
```

---

## Architecture

### Training Loop

One process. No distributed coordination. Each step:

1. **Sample** a prompt from the priority-weighted dataloader (curriculum-gated).
2. **Generate** a completion via vLLM (colocated on the same GPU, MERGE weight sync).
3. **Score** the completion with the user-provided reward function.
4. **Segment** the completion into regions (via segmenter or scored_spans).
5. **Estimate** per-token per-quality advantages (SPO baseline + VPRM critic + ERIC).
6. **Compute** clipped policy gradient loss with AlignedLossFrame.
7. **Backward** through Triton fused logprobs (forward) + PyTorch chunked (backward).
8. **Update** LoRA parameters via AdamW8bit with cosine schedule.
9. **Sync** weights to vLLM via MERGE (merge_adapter / unmerge_adapter).
10. **Checkpoint** every N steps with full state serialization.

### MERGE Weight Sync

Training model and vLLM share the same GPU. The MERGE strategy eliminates ~1.9 GB of duplicate weight storage:

- **Before generation**: `merge_adapter()` bakes LoRA weights into base model weights.
- **Generate**: vLLM reads the merged base weights directly. No separate LoRA buffer needed.
- **After generation**: `unmerge_adapter()` lifts LoRA weights back out for training.

This works because Unsloth's `fast_inference` colocates training and vLLM in the same process, sharing the same `nn.Module` memory. The alternative (DIRECT_COPY) copies LoRA A/B into vLLM's stacked buffers each step, requiring vLLM to maintain its own copy of base weights — 1.4 GB of duplication plus 500 MB for the LoRA buffer.

MERGE requires a Params4bit monkey-patch for PEFT 4-bit compatibility. The `weight_export.py` module handles this.

### VRAM Budget

Qwen3-1.7B, 4-bit, LoRA rank 32, MERGE strategy:

| Component | VRAM |
|---|---|
| Model body (NF4) | ~850 MB |
| lm_head + embeddings (bf16) | ~890 MB |
| vLLM (gpu_memory_utilization=0.25) | ~2.5 GB |
| LoRA adapters (rank 32) | ~48 MB |
| Optimizer (AdamW8bit) | ~96 MB |
| **Training peak** | **~11 GB** |
| Steady state | ~9 GB |
| VRAM growth over 3000+ steps | 0.0 GB |

VRAM stability comes from `torch.cuda.empty_cache()` after merge/unmerge cycles. No engine recreation needed.

---

## Core Techniques

### 1. SPO (Single-stream Policy Optimization)

One completion per prompt. Every completion teaches.

Standard GRPO generates 8-16 completions per prompt, uses the group mean as baseline, and discards completions below the mean. SPO generates one completion and maintains a persistent EMA baseline per prompt per quality per step. The baseline tracks the model's running performance — advantage is the delta between the current reward and what the model usually achieves on that prompt for that quality.

Variance-aware baseline: when reward variance drops (the model is converging), the baseline learning rate slows to preserve the gradient signal. Aspiration gap: when the baseline matches the reward, a target-aware term `beta * (reward - target)` preserves shaped gradients through the plateau.

### 2. VPRM (Verifiable Process Reward Mapping)

Two modes of credit assignment:

**Segmenter mode**: A segmenter function maps each token to a named region (e.g., `KINETIC`, `POTENTIAL`, `HAMILTONIAN`). Each region receives the advantage computed from its associated quality scores. Built-in segmenters: `uniform`, `qwen3_xml`, `label`, `hamiltonian`.

**Span mode**: The reward function returns `scored_spans` — character offsets where each quality expression was found. The spans module (`spans.py`) converts character offsets to per-token boolean masks using the tokenizer's decode mapping. Each token receives the advantage for the quality whose span contains it.

The VPRM critic (`critic.py`) learns a per-quality per-region value function. Architecture per quality: `mean_pool(hidden_states[region]) -> Linear -> ReLU -> Linear -> ReLU -> Linear(1)`. Polyak-averaged target network for stable baselines.

### 3. Phase-Gated Curriculum with Skill Tree

The curriculum is a DAG, not a linear sequence.

**Skill tree**: Each node defines a skill (e.g., `freefall`, `spring_only`, `gravity_spring`). Nodes declare prerequisites, mastery thresholds, regression thresholds, review probabilities, and learnability thresholds. Prompts are matched to skills via metadata (e.g., `match_metadata: {system: freefall}`).

**Advancement**: A skill advances when mastery exceeds the threshold AND learnability `p(1-p)` drops below the learnability threshold — the model must be both accurate and stable. Regression detection triggers review scheduling.

**Tier gating**: Orthogonal to the skill tree, a 2D mastery matrix gates prompt difficulty tiers. Tutorial-tracked prompts bypass tier gating (the skill tree is the authority for those prompts).

Example from the Hamiltonian config:

```
freefall (root) ──> spring_only ──> gravity_spring ──> driven_oscillator (boss)
                         └──────> damped_spring ──────┘
```

### 4. ERIC (Entropy-Regulated Importance Constraint)

Four-quadrant advantage modification based on (correct/wrong) x (confident/uncertain):

| | Confident (low entropy) | Uncertain (high entropy) |
|---|---|---|
| **Correct** | Q2: Leave alone (learned) | Q1: Reinforce (learning) |
| **Wrong** | Q3: Entropy boost (shake confidence) | Q4: Flag for hints (provide direction) |

ERIC uses a chunked lm_head entropy proxy — no attention matrices needed (Unsloth's fast_inference kernels block `output_attentions`). Entropy tracks model commitment: low-entropy tokens are "committed anchors." ERIC dampens positive advantage on committed correct tokens (they are already learned) and boosts entropy on committed wrong tokens (they need unlearning).

Combined with position-based causal weighting (`entropy_position` mode): earlier tokens get higher importance weight because downstream tokens condition on them.

### 5. AlignedLossFrame

All tensor shift operations for advantage-logprob alignment live in one place.

Policy gradient loss requires aligning advantages (computed on completion tokens) with logprobs (shifted by one for autoregressive prediction). This is the source of an entire class of off-by-one bugs. AlignedLossFrame centralizes the coordinate system: logprob space. Shape validation at construction time. A single object carries `logprobs`, `ref_logprobs`, `advantages`, `loss_mask`, and `completion_mask` — all guaranteed to be aligned.

### 6. Triton Fused Logprobs

Custom Triton kernel for the forward pass. One GPU launch replaces 16 Python-to-CUDA round-trips from the chunked lm_head + checkpoint path.

The kernel tiles along the vocab dimension (BLOCK_V=128), computing `hidden[t] @ lm_head.T -> logsumexp -> gather at label[t] -> logprob[t]` per sequence position. Peak VRAM: `hidden_dim x BLOCK_V` per thread block, not `seq x vocab`. Zero `[seq, vocab]` allocation.

Backward pass uses PyTorch's chunked path (autograd through lm_head chunks). Wrapped in `torch.autograd.Function` for clean integration with the training loop. Falls back to the PyTorch chunked path when Triton is unavailable or `vocab_size % 128 != 0`.

### Additional Techniques

**LoRA dropout**: Bernoulli masks on LoRA A matrices during generation. Partially reverts the model to base model behavior, letting suppressed knowledge surface. Linear annealing over configurable steps. Based on NoisyGRPO (NeurIPS 2025).

**Dr.GRPO**: Removes length normalization and standard deviation normalization from the GRPO loss (arXiv:2503.20783). Unbiased gradients.

**Region-specific KL**: THINK regions get low KL (explore freely), FORMAT regions get high KL (lock structure), STEP regions get normal KL.

**LLDS (Log-Likelihood Divergence Smoothing)**: Collapse prevention from NeMo RL. Penalizes divergence between current and reference policy logprobs.

**GRPO-lambda eligibility traces**: Lambda-return approximation for per-token credit assignment. Backward-accumulates advantages with decay factor, giving earlier tokens credit for downstream correctness.

**Length penalty**: Dynamic length control — penalizes length only when group accuracy exceeds a threshold. Prevents the model from gaming reward through verbosity.

**Frontier amplification**: Multiplies advantages for steps that block curriculum advancement. Mastered steps get weight 1.0, frontier (blocking) steps get amplified gradients.

**Aspiration gap**: When the SPO baseline matches the constant partial credit (e.g., the model always scores 0.5 on a quality), vanilla advantage goes to zero. The aspiration gap term `beta * (reward - target)` preserves gradient through converged baselines.

**LoRA-Pro**: Gradient adjustment so the low-rank update better approximates full fine-tuning (ICLR 2025, arXiv:2407.18242). Solves a Sylvester equation per LoRA layer after backward.

**Gradient coherence monitoring**: Temporal cosine similarity between consecutive steps' gradients (convergence signal), spatial cosine between adjacent layers (disagreement signal), LoRA weight norm tracking, turbulence detection.

---

## Config Reference

QGRE is configured via a single YAML file. Top-level sections:

### `model`

| Field | Type | Default | Description |
|---|---|---|---|
| `path` | str | `""` | HuggingFace model path (required) |
| `lora_rank` | int | `8` | LoRA rank |
| `lora_alpha` | int | `16` | LoRA alpha scaling |
| `load_in_4bit` | bool | `true` | 4-bit quantization (NF4) |
| `fast_inference` | bool | `true` | Unsloth fast inference (colocated vLLM) |
| `gpu_memory_utilization` | float | `0.35` | vLLM GPU memory fraction |
| `weight_sync_strategy` | str | `"direct_copy"` | `"direct_copy"` or `"merge"` |
| `pad_token` | str | `""` | Pad token string (required, must not be EOS) |
| `pad_token_id` | int | `-1` | Pad token ID (required) |
| `lora_target_modules` | list[str] | Qwen3 defaults | LoRA target modules |
| `modules_to_save` | list[str] | `["lm_head"]` | Modules trained in full precision |
| `max_lora_rank` | int | `0` | Max LoRA rank for vLLM (0 = auto) |

### `data`

| Field | Type | Default | Description |
|---|---|---|---|
| `train_files` | list[str] | `[]` | Parquet file paths (required) |
| `max_prompt_length` | int | `3200` | Maximum prompt token length |
| `train_batch_size` | int | `16` | Prompts per training step |
| `prompt_column` | str | `"prompt"` | Column name for prompt text |
| `metadata_columns` | list[str] | `["ground_truth", "extra_info"]` | Metadata columns passed to reward function |
| `system_prompt_column` | str \| None | `None` | Column for separate system message |
| `difficulty_column` | str \| None | `None` | Column for difficulty-gated curriculum |
| `tier_order` | list[str] \| None | `None` | Tier progression order |
| `tier_advance_threshold` | float | `0.85` | Mastery threshold for tier advancement |
| `tier_advance_quality_phase` | int | `3` | Quality phase required for advancement |
| `initial_tiers` | list[str] \| None | `None` | Starting tiers |

### `generation`

| Field | Type | Default | Description |
|---|---|---|---|
| `temperature` | float | `0.7` | Sampling temperature |
| `top_p` | float | `0.8` | Nucleus sampling threshold |
| `top_k` | int | `20` | Top-k sampling |
| `min_p` | float | `0.1` | Minimum probability threshold |
| `max_tokens` | int | `4096` | Maximum completion tokens |
| `repetition_penalty` | float | `1.0` | vLLM repetition penalty (1.0 = disabled) |
| `stop_token_ids` | list[int] | `[]` | Stop token IDs (required per model) |
| `max_logprobs` | int | `5` | vLLM max logprobs for LLDS |
| `lora_dropout_rate` | float | `0.0` | LoRA A dropout rate (0.15 recommended) |
| `lora_dropout_anneal_steps` | int | `500` | Linear anneal to zero over N steps |

### `algorithm`

| Field | Type | Default | Description |
|---|---|---|---|
| `mode` | str | `"spo"` | `"spo"` or `"grpo"` |
| `clip_ratio_low` | float | `0.2` | PPO clip lower bound |
| `clip_ratio_high` | float | `0.28` | PPO clip upper bound |
| `loss_type` | str | `"grpo"` | `"grpo"` or `"dr_grpo"` (unbiased) |
| `llds_coef` | float | `0.05` | LLDS collapse prevention coefficient |
| `step_qualities` | dict | `None` | `{step_num: [quality_names]}` mapping |
| `segmenter` | str | `"uniform"` | Segmenter: `"uniform"`, `"qwen3_xml"`, `"label"`, `"hamiltonian"`, or `"module:function"` |
| `use_fused_logprobs` | bool | `true` | Chunked lm_head logprob computation |
| `use_triton_logprobs` | bool | `true` | Triton kernel for forward pass |
| `advantage_scale` | float | `1.0` | Scale factor for advantages (0.1 recommended for 1-3B) |
| `min_completion_tokens` | int | `0` | Minimum tokens before negative floor (50 recommended) |
| `attention_constrained_advantage` | bool | `false` | Enable ERIC |
| `attention_constraint_strength` | float | `1.0` | ERIC dampening multiplier |
| `eric_mode` | str | `"entropy_position"` | `"entropy"`, `"position"`, or `"entropy_position"` |
| `kl_think_multiplier` | float | `0.1` | KL weight for THINK regions |
| `kl_format_multiplier` | float | `2.0` | KL weight for FORMAT regions |
| `kl_step_multiplier` | float | `1.0` | KL weight for STEP regions |
| `lambda_return` | float | `0.0` | Eligibility trace decay (0 = off, 0.95 = typical) |
| `length_penalty_coef` | float | `0.0` | Length penalty coefficient |
| `length_penalty_threshold` | float | `0.5` | Accuracy threshold for length penalty |
| `frontier_amplification` | float | `2.0` | Gradient boost for blocking steps |

### `algorithm.spo`

| Field | Type | Default | Description |
|---|---|---|---|
| `lr` | float | `0.1` | EMA baseline learning rate |
| `n` | int | `1` | Completions per prompt |
| `aspiration_beta` | float | `0.5` | Aspiration gap strength |
| `aspiration_target` | float | `0.8` | Target reward for aspiration |
| `var_aware` | bool | `true` | Variance-aware baseline |
| `var_threshold` | float | `0.01` | Variance threshold for slowdown |
| `staleness_window` | int | `50` | Steps before baseline decays to prior |

### `training`

| Field | Type | Default | Description |
|---|---|---|---|
| `total_steps` | int | `800` | Total training steps |
| `lr` | float | `5e-6` | Optimizer learning rate |
| `warmup_steps` | int | `10` | LR warmup steps |
| `lr_scheduler` | str | `"cosine"` | LR scheduler type |
| `save_freq` | int | `50` | Checkpoint frequency (0 = disabled) |
| `gradient_accumulation_steps` | int | `1` | Gradient accumulation |
| `max_grad_norm` | float | `1.0` | Gradient clipping |
| `mastery_threshold` | float | `0.8` | Phase advancement threshold |
| `stagnation_timeout` | int | `200` | Steps before stagnation detection |
| `embedding_lr_ratio` | float | `0.1` | lm_head LR = base LR x this |
| `kv_cache_flush_freq` | int | `50` | vLLM KV cache flush frequency (0 = disabled) |
| `quality_window_size` | int | `20` | Rolling window for mastery tracking |
| `seed` | int | `-1` | Random seed (-1 = time-based) |
| `log_attention_patterns` | bool | `false` | Log attention entropy and collapse |

### `vprm`

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable VPRM learned critic |
| `intermediate_dim` | int | `128` | MLP hidden dimension |
| `lr` | float | `1e-4` | Critic learning rate |
| `clip_advantage` | float | `5.0` | Per-quality advantage clipping |
| `spo_fallback_min_regions` | int | `2` | Min regions for critic (else SPO fallback) |
| `polyak_tau` | float | `0.01` | Target network update rate |
| `use_target_network` | bool | `true` | Enable Polyak-averaged target |

### `egrs`

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable ERIC/EGRS 4-quadrant system |
| `reward_threshold` | float | `0.5` | Score above this = "correct" |
| `entropy_threshold` | float | `0.5` | Normalized entropy below this = "confident" |
| `gate_temperature` | float | `0.1` | Sigmoid temperature for soft gating |
| `exploration_weight` | float | `0.1` | Entropy bonus for Q3 (confident+wrong) |
| `hint_enabled` | bool | `true` | Enable hint injection for Q4 |
| `hint_extractor` | str | `"none"` | `"hamiltonian"`, `"generic"`, or `"none"` |
| `mastery_threshold` | float | `0.8` | Mastery at which hints stop |

### `tutorial`

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable skill tree tutorial |
| `post_mastery_behavior` | str | `"review_only"` | `"review_only"`, `"pause"`, or `"continue_all"` |
| `untracked_always_active` | bool | `true` | Prompts not in any skill are always available |
| `sequential_mastery` | bool | `false` | Focus on one skill at a time |
| `skill_tree` | dict | `{}` | Skill definitions (see below) |

Each skill in `skill_tree`:

| Field | Type | Default | Description |
|---|---|---|---|
| `prompts` | list[str] | `[]` | Explicit prompt IDs |
| `match_metadata` | dict \| None | `None` | Match prompts by metadata column values |
| `prerequisites` | list[str] | `[]` | Skills that must be mastered first |
| `mastery_threshold` | float | `0.8` | Mastery score for advancement |
| `regression_threshold` | float | `0.6` | Score below this triggers regression |
| `mastery_window` | int | `20` | Rolling window for mastery score |
| `review_probability` | float | `0.15` | Probability of review after mastery |
| `score_key` | str \| None | `None` | Quality key to track (None = overall reward) |
| `learnability_threshold` | float | `0.10` | Advance when p(1-p) < this |

### `lora_pro`

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable LoRA-Pro gradient adjustment |
| `beta1` | float | `0.9` | Adam beta1 for equivalent gradient |
| `beta2` | float | `0.999` | Adam beta2 for equivalent gradient |
| `grad_scale` | float | `1.0` | Post-adjustment gradient multiplier |
| `grad_floor` | float | `1e-7` | Minimum gradient norm |

### `logging`

| Field | Type | Default | Description |
|---|---|---|---|
| `mlflow_experiment` | str | `"qgre-training"` | MLflow experiment name |
| `completion_dir` | str | `"output/completions"` | JSONL completion log directory |
| `checkpoint_dir` | str | `"output/checkpoints"` | Checkpoint directory |
| `log_freq` | int | `5` | Progress table frequency |
| `grad_log_freq` | int | `10` | Gradient flow log frequency |

---

## API

### RewardResult

The reward function contract. Return this from your reward function.

```python
@dataclass(frozen=True)
class RewardResult:
    reward: float                    # Overall reward in [0, 1]
    scores: dict[str, float]         # Per-quality scores {quality_name: score}
    scored_spans: dict | None = None # Optional: character offsets for span-based assignment
```

`scores` keys must match the quality names in `step_qualities`. The engine uses `scores` for per-quality advantage estimation and `scored_spans` for per-token credit assignment.

### GameState

Tracks curriculum state: current phase, mastery scores, step counters, skill tree state, tier state.

```python
@dataclass
class GameState:
    phase: int = 1
    mastery: dict[str, float]         # Per-quality rolling mastery scores
    phase_step: int = 0
    total_step: int = 0
    skill_mastery: dict[str, float]   # Per-skill mastery scores
    active_skills: set[str]           # Currently trainable skills
    unlocked_tiers: set[str]          # Currently accessible tiers
    ...
```

### AlignedLossFrame

The centralized coordinate system for advantage-logprob alignment.

```python
@dataclass
class AlignedLossFrame:
    logprobs: torch.Tensor           # [batch, seq] in logprob space
    ref_logprobs: torch.Tensor       # [batch, seq] reference policy
    advantages: torch.Tensor         # [batch, seq] aligned to logprobs
    loss_mask: torch.Tensor          # [batch, seq] valid positions
    completion_mask: torch.Tensor    # [batch, seq] completion tokens
```

Constructed with shape validation. All tensors guaranteed to share dimensions.

### GenerationBackend Protocol

```python
class GenerationBackend(Protocol):
    def generate(self, input_ids, attention_mask, **kwargs) -> GenerationOutput: ...
    def set_training_mode(self) -> None: ...
    def set_inference_mode(self) -> None: ...
    @property
    def weight_exporter(self) -> WeightExporter: ...
    @property
    def weight_loader(self) -> WeightLoader: ...
    @property
    def model(self) -> nn.Module: ...
    @property
    def tokenizer(self) -> Any: ...
```

The trainer interacts with generation through this protocol. `UnslothBackend` is the production implementation.

---

## Tests

504 tests: 492 CPU, 12 GPU.

```bash
# All CPU tests
pytest tests/ -m "not gpu"

# GPU tests (requires CUDA)
pytest tests/ -m "gpu"

# Specific module
pytest tests/test_advantages.py -v

# With coverage
pytest tests/ --cov=qgre --cov-report=html
```

Test coverage spans every module: advantages, attention constraints, checkpoint serialization, critic networks, data loading, EGRS integration, fused logprobs, gradient coherence, Hamiltonian reward, hardening regressions, hints, LLDS, logging, LoRA-Pro, LoRA verification, NeMo extracted, schema validation, segments, spans, sync state, trainer, Triton logprobs, tutorial skill tree, weight loader lifecycle, and wiring.

---

## Reward Function Design

The Hamiltonian reward function (`examples/hamiltonian/reward_fn_v2.py`) demonstrates the design principle: correctness-only scoring.

**No format scoring.** RL teaches WHAT (mathematical correctness). SFT teaches HOW (formatting). Format scoring is the primary reward hacking vector — the model learns to produce well-formatted wrong answers. This is documented in arXiv:2602.18037.

The reward function:

1. **Extracts** all math expressions from the completion using format-agnostic line-by-line parsing. Strips LaTeX delimiters, splits on `=`, handles both LaTeX and plain text.
2. **Parses** expressions using math-verify (HuggingFace's verification library).
3. **Verifies** symbolic equivalence via sympy. Not string matching — `p^2/(2m)` and `p**2/2/m` are the same expression.
4. **Falls back** to substitution: if the model writes a correct generic formula (e.g., `p^2/(2m)`) but the ground truth has specific values (e.g., `p^2/10`), substitutes known constants from the problem metadata and checks equivalence.
5. **Falls back** to velocity form: recognizes kinetic energy written as `mv^2/2` when ground truth uses momentum form `p^2/(2m)`.
6. **Returns** `scored_spans` mapping each quality to the character offsets where the expression was found.

Six qualities: `q_kinetic`, `q_potential`, `q_hamiltonian`, `q_dqdt`, `q_dpdt`, `q_consistency`.

---

## File Structure

```
qgre/
  __init__.py               Exports: RewardResult, GameState, Segmenter, segmenters
  __main__.py               CLI: python -m qgre train --config ... --reward ...
  types.py                  RewardResult (frozen), GameState, AlignedLossFrame,
                            TrainingContext, CheckpointState, SyncLifecycle states
  config.py                 All config dataclasses + YAML loader + validation
  trainer.py                QGRETrainer: training loop, Triton conditional path,
                            MERGE weight sync, ERIC integration
  generation.py             UnslothBackend: vLLM colocation, MERGE weight sync,
                            LoRA dropout, hint injection protocol
  advantages.py             SPO + Dr.GRPO + VPRM + phase gating + EGRS 4-quadrant +
                            frontier amplification + aspiration gap
  data.py                   Parquet loading, tokenization, padding, batching,
                            priority sampling, tier gating
  segments.py               Segmenters: qwen3_xml, uniform, label, custom
  spans.py                  Character-to-token mapping for scored_spans
  critic.py                 VPRM per-region per-dimension learned critic with
                            Polyak-averaged target network
  checkpoint.py             Full state save/resume with schema versioning and migration
  fused_logprobs.py         Chunked lm_head projection (PyTorch path)
  triton_logprobs.py        Triton kernel + torch.autograd.Function wrapper
  attention_bonds.py        ERIC: entropy-regulated importance constraint,
                            confidence gating, position-based causal weighting
  attention_analysis.py     Attention entropy, collapse detection, fragmentation
  gradient_coherence.py     Temporal/spatial cosine, LoRA weight norm, turbulence
  lora_dropout.py           Bernoulli dropout on LoRA A during generation
  lora_pro.py               LoRA-Pro gradient adjustment (Sylvester equation solver)
  lora_verify.py            Weight hash verification after sync
  weight_bus.py             MERGE/DIRECT_COPY strategy dispatcher
  weight_export.py          PEFT-aware weight extraction + Params4bit patch
  weight_load.py            vLLM weight injection with SyncLifecycle state machine
  sync_state.py             Unified state machine for weight sync, dropout, cache
  logging.py                MLflow metrics + JSONL completion logs
  schema.py                 Declarative schema validation for checkpoint fields
  hints.py                  Hint extraction and injection for EGRS Q4 tokens
  nemo_extracted/           Apache-2.0 code from NeMo RL v0.5.0:
    kl.py                     KL divergence (k1/k2/k3) + masked_mean
    llds.py                   Log-Likelihood Divergence Smoothing
    logits.py                 selective_log_softmax, logprobs_from_logits
    loss_functions.py         ClippedPGLossFn + eligibility traces

examples/
  hamiltonian/              Hamiltonian mechanics: config, reward fn, data generator,
                            system prompts, 121 sympy-verified problems
  math/                     Math reasoning example
  hypergraph/               Hypergraph reasoning example

tests/                      504 tests (492 CPU + 12 GPU)
```

---

## Research Features

| Feature | Module | Description |
|---|---|---|
| SPO (Single-stream Policy Optimization) | `advantages.py` | n=1 persistent EMA baseline per prompt per quality per step |
| VPRM (Verifiable Process Reward Mapping) | `advantages.py`, `critic.py`, `spans.py` | Segmented per-token per-quality advantage with learned critic |
| Phase-gated curriculum with skill tree | `types.py`, `data.py`, `config.py` | DAG-based prerequisite mastery with regression detection |
| ERIC (Entropy-Regulated Importance Constraint) | `attention_bonds.py`, `advantages.py` | 4-quadrant advantage modification via chunked entropy proxy |
| AlignedLossFrame | `types.py`, `trainer.py` | Centralized tensor alignment with shape validation |
| Triton fused logprobs | `triton_logprobs.py` | Forward via Triton kernel, backward via PyTorch, autograd wrapper |
| MERGE weight sync | `weight_bus.py`, `weight_export.py`, `weight_load.py` | Share GPU memory between training and vLLM |
| LoRA dropout | `lora_dropout.py` | Bernoulli masks on LoRA A for structured exploration |
| Dr.GRPO | `trainer.py`, `config.py` | Unbiased gradients: no length or std normalization |
| Region-specific KL | `trainer.py` | THINK/FORMAT/STEP regions with different KL weights |
| LLDS (collapse prevention) | `nemo_extracted/llds.py` | Log-Likelihood Divergence Smoothing from NeMo RL |
| GRPO-lambda eligibility traces | `nemo_extracted/loss_functions.py` | Lambda-return credit assignment |
| Aspiration gap | `advantages.py` | Gradient preservation through converged baselines |
| Variance-aware baseline | `advantages.py` | Slow baseline LR when reward variance drops |
| Frontier amplification | `advantages.py` | Amplified gradient on curriculum-blocking steps |
| LoRA-Pro gradient adjustment | `lora_pro.py` | Sylvester equation solver for better low-rank approximation |
| Gradient coherence monitoring | `gradient_coherence.py` | Temporal cosine, LoRA weight norm, turbulence detection |
| math-verify reward verification | `examples/hamiltonian/reward_fn_v2.py` | Battle-tested expression parsing + symbolic equivalence |
| Substitution fallback | `examples/hamiltonian/reward_fn_v2.py` | Recognize correct generic formulas with free variables |
| Hint injection (EGRS Q4) | `hints.py`, `generation.py` | Domain-specific hints for uncertain-wrong tokens |
| Fused logprobs (PyTorch) | `fused_logprobs.py` | Chunked lm_head avoids full [seq, vocab] allocation |
| Checkpoint schema migration | `checkpoint.py`, `schema.py` | Forward-compatible state serialization with validation |
| SyncState machine | `sync_state.py` | Thread-safe unified state for weight sync lifecycle |

---

## Known Constraints

- **Single GPU, single process.** By design. Multi-GPU support is a non-goal.
- **On-policy.** Each step generates then trains. Generation-time logprobs available via vLLM (LLDS active).
- **MERGE requires Params4bit patch.** PEFT's `merge_adapter()` does not natively support 4-bit quantized weights. `weight_export.py` patches `Params4bit.data` to return the dequantized tensor.
- **16 GB VRAM ceiling with MERGE.** 11 GB actual for Qwen3-1.7B rank 32. Larger models require larger GPUs or lower LoRA rank.
- **Unsloth dependency.** Fast inference, vLLM colocation, and 4-bit LoRA integration rely on Unsloth. The `GenerationBackend` protocol abstracts this, but no alternative backend is implemented.
- **Unix-only sympy timeout.** The Hamiltonian reward function uses `SIGALRM` for sympy timeout. Windows requires an alternative timeout mechanism.
- **vLLM VRAM stability** depends on `torch.cuda.empty_cache()` after merge/unmerge cycles. Without it, fragmentation accumulates.

---

## Linting and Type Checking

```bash
# Ruff (linter + formatter)
ruff check qgre/ tests/
ruff format qgre/ tests/

# Pyright (static type checking)
pyright qgre/

# Bandit (security)
bandit -c pyproject.toml -r qgre/
```

The project enforces strict linting via Ruff with 50+ rule sets enabled, Pyright for static type checking, and Bandit for security analysis. Configuration lives in `pyproject.toml`.

---

## References

- **GRPO**: Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (2024)
- **Dr.GRPO**: Liu et al., "Understanding R1-Zero-Like Training: A Critical Analysis of GRPO and Related Methods" (arXiv:2503.20783)
- **LoRA-Pro**: Wang et al., "LoRA-Pro: Are Low-Rank Adapters Properly Optimized?" (ICLR 2025, arXiv:2407.18242)
- **NoisyGRPO**: "Noise Injection Reveals Hidden Capabilities of Language Models" (NeurIPS 2025)
- **ERPO**: Entropy-Regulated Policy Optimization for language model alignment
- **math-verify**: HuggingFace Math-Verify library for expression parsing and equivalence verification
- **NeMo RL**: NVIDIA NeMo RL v0.5.0 (Apache-2.0) — ClippedPGLossFn, KL divergence, LLDS, selective_log_softmax
- **Gradient regularization**: arXiv:2602.18037 — format scoring as reward hacking vector
- **GRPO-lambda**: GRPO with eligibility traces (ICLR 2026)

---

## License

Apache-2.0. See [LICENSE](LICENSE).

`qgre/nemo_extracted/` contains code from NVIDIA NeMo RL (Apache-2.0), modified for single-GPU single-process operation.
