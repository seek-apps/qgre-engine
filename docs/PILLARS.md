# QGRE Engine — Pillars & Components

Six pillars. Each pillar is independently testable. Build order follows
dependency graph from pressure test (prerequisites → assembly).

---

## Pillar 1: DATA PIPELINE
**Status:** Must build | **Effort:** 2 hours | **Prerequisite:** 0e

### Components
| Component | What it does | Lines | Source |
|-----------|-------------|-------|--------|
| ParquetLoader | Load parquet, extract prompt/ground_truth/extra_info | ~20 | HF datasets |
| PromptTokenizer | apply_chat_template → token IDs, left-pad | ~30 | transformers |
| BatchAssembler | Group into train_batch_size, shuffle per epoch | ~20 | torch DataLoader |
| PromptExpander | Repeat each prompt × n for rollout | ~10 | simple loop |
| OverlongFilter | Drop prompts > max_prompt_length after tokenization | ~5 | filter |
| EpochTracker | Track current epoch, step within epoch, total steps | ~15 | counter |

### Key questions for research
- How does simple_GRPO handle data loading and batching?
- How does TRL's GRPOTrainer prepare data for vLLM generation?
- What's the minimal DataLoader for RL that handles prompt expansion?

---

## Pillar 2: GENERATION ENGINE
**Status:** Already working | **Effort:** 1 hour (wiring only)

### Components
| Component | What it does | Lines | Source |
|-----------|-------------|-------|--------|
| UnslothModelLoader | FastLanguageModel.from_pretrained + get_peft_model | ~30 | Already in fork |
| VLLMGenerator | model.fast_generate + SamplingParams | ~40 | Already in fork |
| LoRASyncer | save_lora / load_lora between training and vLLM | ~10 | Already in fork |
| ResponseBuilder | vLLM RequestOutput → padded response tensors | ~30 | Already in fork |
| ChatTemplateApplier | Tokenize prompts with nothink template | ~10 | Pillar 1 overlap |
| StopTokenConfig | Set stop_token_ids=[151643, 151645] | ~5 | SamplingParams |

### Key questions for research
- How does TRL colocated vLLM handle weight sync after each training step?
- Does vLLM's external_launcher offer anything beyond Unsloth's fast_generate?
- Best practices for vLLM SamplingParams in GRPO (temperature, top_p for exploration)?

---

## Pillar 3: REWARD & CURRICULUM
**Status:** Mostly exists | **Effort:** 4 hours (adapter + tensor construction)

### Components
| Component | What it does | Lines | Source |
|-----------|-------------|-------|--------|
| RewardFunction | compute_score → per-component dict | ~900 | reward_fn.py (exists) |
| GameState | Phase tracking, Elo, mastery, curriculum tier | ~200 | reward_fn.py (exists) |
| RewardPlacement | VPRM: segment_completion + compute_step_region_rewards — per-token reward by step region (see SPECIAL-TOKENS-SUPERPOWER.md) | ~80 | New |
| BatchRewardTensors | list[dict] → dict[str, Tensor] per component | ~20 | New (prerequisite 0c) |
| QGREGDPOAdapter | Filter gated rewards, pass active-only to GDPO | ~40 | New (prerequisite 0d) |
| GameStateSerializer | to_dict / from_dict (deque→list, defaultdict→dict) | ~30 | New (prerequisite 0a) |
| CurriculumTierManager | Read/write tier from MLflow, auto-advance | ~20 | Exists in reward_fn |
| TokenDecoder | Decode response for reward_fn (skip_special_tokens=False) | ~5 | Exists |

### Key questions for research
- How does GDPO paper handle variable number of reward components?
- How does NeMo RL's GDPOAdvantageEstimator handle component weighting?
- Best practices for serializing stateful RL components (Elo, windows)?

---

## Pillar 4: ALGORITHM LAYER
**Status:** Must extract + build | **Effort:** 5 hours

### Components
| Component | What it does | Lines | Source |
|-----------|-------------|-------|--------|
| GDPOAdvantage | Per-component normalization + advantage | ~222 | NeMo RL (extract) |
| ClippedPGLoss | GRPO/PPO clip math | ~200 | NeMo RL (extract) |
| KLCovLoss | Selective KL on high-covariance tokens | ~80 | verl core_algos (extract) |
| LLDSLoss | Lazy Likelihood Displacement Stabilization | ~40 | verl core_algos (extract) |
| LogProbComputer | Forward pass → log_softmax → gather (fp32 cast) | ~30 | New (from dp_actor pattern) |
| ResponseMask | Scan for EOS, mask padding tokens | ~10 | verl torch_functional (extract) |
| FilterGroups | Drop zero-variance reward groups | ~20 | verl ray_trainer (extract) |
| BaselineUtils | calculate_baseline_and_std_per_prompt | ~100 | NeMo RL (extract) |
| KLCalculation | calculate_kl for penalty/monitoring | ~50 | NeMo RL (extract) |

### Key questions for research
- How does NeMo RL's advantage_estimator.py handle the GDPO math specifically?
- What's the exact kl_cov implementation — is it in the GDPO paper or separate?
- How does simple_GRPO compute advantages without a separate estimator class?
- How does the clipped loss handle seq-mean-token-sum-norm aggregation?

---

## Pillar 5: TRAINING LOOP
**Status:** Must build | **Effort:** 4 hours

### Components
| Component | What it does | Lines | Source |
|-----------|-------------|-------|--------|
| QGRETrainer | Main class — orchestrates all pillars | ~100 | New |
| GradientAccumulator | Split mini-batch → micro-batches, accumulate grads | ~15 | Standard PyTorch |
| OptimizerStep | AdamW8bit step + zero_grad | ~5 | bitsandbytes |
| GradientClipper | clip_grad_norm_ with nan detection | ~10 | From dp_actor pattern |
| LRScheduler | Constant/cosine/warmup scheduler | ~10 | torch.optim.lr_scheduler |
| OnPolicyMode | old_log_probs = log_probs.detach() (skip recompute) | ~5 | From dp_actor pattern |
| StepMetrics | Compute and collect per-step metrics | ~20 | New |
| TrainingDiagnostics | Loss, grad_norm, GPU memory, nan detection | ~15 | From our DIAG prints |

### Key questions for research
- How does simple_GRPO structure its training loop?
- How does policy-gradients (zafstojano) handle gradient accumulation?
- How does TRL's GRPOTrainer handle on-policy vs off-policy log probs?
- What's the minimal GRPO loop that handles all edge cases (nan, zero-variance)?

---

## Pillar 6: PERSISTENCE
**Status:** Must build | **Effort:** 3 hours

### Components
| Component | What it does | Lines | Source |
|-----------|-------------|-------|--------|
| CheckpointSaver | torch.save full state every N steps | ~20 | Standard PyTorch |
| CheckpointResumer | Find latest checkpoint, restore all state | ~30 | New (prerequisite 0f) |
| CheckpointDiscovery | Scan dir for global_step_N, return latest | ~15 | New |
| MLflowLogger | Direct mlflow.log_metrics/log_params/set_tag | ~30 | mlflow SDK |
| CompletionDumper | JSONL with input/output/score/reward components | ~30 | Exists (adapted) |
| MLflowExperimentSetup | set_experiment, start_run, log config | ~15 | From launch_training.py |

### Key questions for research
- How does simple_GRPO handle checkpointing?
- How does NeMo RL handle checkpoint save/resume in their GRPO loop?
- Best practices for MLflow metric logging frequency in RL training?
- How to serialize custom RL state (Elo ratings, sliding windows) reliably?

---

## Dependency Graph

```
Pillar 1 (Data) ──────────────────────┐
Pillar 2 (Generation) ────────────────┤
Pillar 3 (Reward) ─── needs 0a,0c,0d ─┤──→ Pillar 5 (Training Loop) ──→ DONE
Pillar 4 (Algorithm) ─ needs 0b ──────┤
Pillar 6 (Persistence) ─ needs 0a,0f ─┘
```

All pillars feed into Pillar 5 (Training Loop), which assembles them.
Pillars 1, 2, and parts of 3/4/6 can be built in parallel.

---

## Research Findings Per Pillar

### Pillar 1 (Data Pipeline) — Key findings
- **Hands-on GRPO from scratch** (Baicen Xiao, medium): Shows exact data prep pattern —
  tokenize prompts, left-pad, expand ×n, batch. Uses vLLM on GPU:0, train on GPU:1.
- **TRL GRPOTrainer**: Uses HF `Dataset` directly. No custom DataLoader needed for simple
  cases. `train_dataset` is just a HF dataset with a `prompt` column.
- **simple_GRPO**: Uses `deepspeed` DataLoader. ~20 lines of data code total.
- **NeMo RL**: Uses `torchdata.stateful_dataloader.StatefulDataLoader` for epoch tracking.
  This is the most robust approach for checkpoint resume.
- **DECISION**: Use HF `Dataset` + standard `torch.utils.data.DataLoader` with a custom
  collate function that applies chat_template + left-pads. StatefulDataLoader for epoch state.

### Pillar 2 (Generation Engine) — Key findings
- **TRL PR #3189**: "Parameter Synchronization Issue Between GRPO Training and vLLM Generation"
  — the model in vLLM stays at initial version unless explicitly synced. Our `save_lora`/`load_lora`
  handles this. TRL PR #2730 (tgaddair) solves it with dynamic LoRA loading.
- **TRL PR #2730 WARNING**: "vLLM leaks host memory when dynamically loading LoRA adapters
  over and over." Fix: "periodically recreate the LLM instance every 50-100 steps." Monitor
  for this in our engine.
- **vLLM colocate mode** (`vllm_mode="colocate"`): Uses `external_launcher` for in-process
  vLLM. This is what TRL does. Our Unsloth fast_generate is the same concept but through
  Unsloth's wrapper.
- **DECISION**: Keep Unsloth fast_generate (already working). Monitor for memory leaks.
  Add periodic vLLM engine recreation if needed.

### Pillar 3 (Reward & Curriculum) — Key findings
- **GDPO official repo** (NVlabs/GDPO, 412 stars): Has implementations for HF-TRL, verl,
  AND NeMo RL. Three reference implementations to cross-check.
- **GDPO handles variable reward counts**: Normalizes per-component, then batch-wise.
  Our QGRE adapter just needs to pass only active-phase components.
- **NeMo RL ProRLv2**: Combines GRPO + DAPO dynamic sampling + decoupled clipping +
  importance sampling + Reinforce++. All in one config. We can steal the DAPO dynamic
  sampling (our filter_groups is a simpler version).
- **DECISION**: Extract GDPO from NVlabs/GDPO (already has verl implementation we can
  reference). Build adapter as thin layer between reward_fn dict and GDPO tensor format.

### Pillar 4 (Algorithm Layer) — Key findings
- **kl_cov (Entropy Mechanism paper)**: verl recipe `recipe/dapo/7b_kl_cov.sh`. The paper
  shows R = -a·exp(H) + b — performance and entropy are linked. KL-Cov selectively
  applies KL penalty on high-covariance tokens. This is already in our verl fork's
  core_algos.py. Extract it.
- **EntroPIC (Tencent, 2026)**: Adaptive entropy control via PID controller. More
  sophisticated than kl_cov but same goal. Worth watching for Phase 2.
- **SPO (Single-stream Policy Optimization, ICLR 2026)**: Eliminates group-based
  normalization entirely with a persistent KL-adaptive value tracker. Could replace
  GRPO/GDPO entirely. "SPO replaces per-group baselines with a persistent, KL-adaptive
  value tracker and normalizes advantages globally across the batch."
  Worth watching but too experimental for Phase 1.
- **NeMo RL source code** (nemo_rl/algorithms/grpo.py, 3197 lines): The full GRPO
  implementation including StatefulDataLoader integration, loss computation, and
  advantage estimation. This is our primary extraction source.
- **DECISION**: Extract kl_cov from verl, GDPO from NeMo RL, LLDS from our fork.
  Watch SPO for future replacement of the entire advantage estimator.

### Pillar 5 (Training Loop) — Key findings
- **policy-gradients (zafstojano)**: Cleanest reference. `train.py` + `loss.py` +
  `buffer.py` + `config.py` + `utils.py`. Total ~500 lines. Trains Qwen3-1.7B on
  24GB A10G. This IS our reference implementation.
- **Gradient accumulation**: Standard PyTorch pattern. `loss /= accumulation_steps`
  before backward, `optimizer.step()` after N micro-batches.
- **TRL VRAM optimization** (PR #2669): Mini-batch approach for logit calculations.
  Instead of `model(full_batch).logits`, do `for chunk in chunks: model(chunk).logits`.
  This is our micro-batch pattern from dp_actor.
- **DECISION**: Use policy-gradients as structural reference. Port micro-batch pattern
  from our dp_actor. Add QGRE-specific wiring (curriculum, GDPO, LLDS).

### Pillar 6 (Persistence) — Key findings
- **verl issue #4534**: "How to save checkpoints and resume training?" — `trainer.resume_mode=auto`
  in verl. We need to replicate this: scan for latest checkpoint, restore all state.
- **Unsloth issue #2168**: "Resume Training from Checkpoint for GRPO Results in OOM" — common
  problem. The vLLM engine holds memory from previous session. Fix: recreate vLLM on resume.
- **TRL issue #3247**: "Resume from checkpoint failed in GRPO" — also common. RNG state
  must be saved/restored for reproducibility.
- **Key pattern**: Save `{'model': ..., 'optimizer': ..., 'scheduler': ..., 'step': ...,
  'rng_states': ..., 'custom_state': ...}`. On resume, detect latest checkpoint, load
  all state, verify step number matches, continue.
- **GameState serialization**: Use `dataclasses.asdict()` after converting deque→list and
  defaultdict→dict. On load, reconstruct with `deque(list_data, maxlen=N)` and
  `defaultdict(factory, dict_data)`.
- **DECISION**: Standard torch.save/load with custom GameState to_dict/from_dict.
  Recreate vLLM engine on resume to avoid memory leaks.

---

## Build Schedule (24 hours focused)

| Session | Duration | What | Pillars |
|---------|----------|------|---------|
| 1 | 4 hours | Prerequisites 0a-0f | 3,4,6 (parts) |
| 2 | 4 hours | Extract + test algorithm layer | 4 |
| 3 | 3 hours | Data pipeline + persistence | 1, 6 |
| 4 | 4 hours | QGRETrainer assembly | 5 (combines all) |
| 5 | 3 hours | Wiring + generation integration | 2, 5 |
| 6 | 4 hours | Equivalence test + GDPO benchmark | All |
| 7 | 2 hours | Polish, docs, edge cases | All |

## Future Watch (not for Phase 1)

- **SPO** (ICLR 2026): Single-stream Policy Optimization — eliminates group normalization.
  Could replace GRPO/GDPO entirely. Monitor for stability reports.
- **EntroPIC** (Tencent 2026): PID-controlled entropy. More precise than kl_cov.
- **ProRLv2** (NeMo RL): GRPO + all stability tricks in one config. Good reference for
  combining multiple loss terms.
- **CalibRL** (2026): Hybrid-policy RLVR with distribution-aware advantage weighting.
  Addresses exploration collapse differently than entropy mechanisms.
