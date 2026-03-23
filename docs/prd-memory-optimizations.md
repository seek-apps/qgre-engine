# PRD: QGRE Memory Optimizations for 14B on 16GB

## Objective

Enable training of 14B parameter models (Qwen3-14B, 4-bit quantized via Unsloth + bnb) on a single 16GB GPU (RTX 5080) within the existing QGRE training loop.

## Background & Motivation

Torad Labs builds the QGRE engine: a "vibe training" platform for non-ML-engineers who want to fine-tune specialist models using reward functions they define, not gradient math they understand. The engine handles generation, segmented credit assignment, SPO/GRPO advantage estimation, and LoRA training in a single-GPU loop.

Today the engine runs reliably on Qwen3-1.7B. That model is useful for development and testing, but a 1.7B specialist is a toy. The production quality threshold is 14B. At 14B, models produce structured outputs (HIF JSON, Hamiltonian derivations, multi-step XML) that are actually usable by downstream systems. Below 14B, the reasoning and format compliance degrade to the point where the reward signal becomes noisy and training stalls.

The target hardware is a single RTX 5080 (16GB VRAM). This is the GPU that Torad Labs' users will have access to. If 14B training requires an A100 or multi-GPU setup, the product loses its core value proposition: any engineer with a consumer GPU can build a production-quality specialist.

The memory challenge: Qwen3-14B in 4-bit quantization consumes approximately 8.5GB for weights alone. With LoRA adapters, optimizer states (AdamW8bit), activations, and the logit tensor during forward pass, peak VRAM exceeds 16GB. The specific bottleneck is the logit tensor: at seq_len=4096, vocab_size=151936, a single full forward produces a 4096 x 151936 x 4B = 2.34GB tensor per sequence. This single allocation pushes the total over the 16GB budget.

These optimizations are not performance improvements. They are ship-or-don't-ship for the 14B use case.

## Success Criteria

- Qwen3-14B trains for 100+ steps without OOM on 16GB VRAM with seq_len=4096, micro_batch_size=1
- All 52 reward tests + 25 trainer tests pass
- Training dynamics (loss curve, reward progression) are equivalent to baseline within 5% over 100 steps on Qwen3-1.7B
- Wall-clock regression < 10% per training step

## Architecture Diagram

### Current Forward Path (trainer.py lines 367-409)

```
                        CURRENT PATH — Full Forward
                        ===========================

    input_ids [1, 4096]
         |
         v
    +---------------------+
    | self.model(mb_ids)  |   Full CausalLM forward pass
    | (Unsloth-wrapped)   |   Activations: ~1.4GB (14B, grad checkpointing on)
    +---------------------+
         |
         v
    mb_output.logits [1, 4096, 151936]    <-- 2.34GB (float32)
         |                                     THIS IS THE KILLER
         v
    logprobs_from_logits(logits, labels)
         |  log_softmax over vocab dim
         |  gather at label positions
         v
    mb_lp [1, 4095]                        <-- 16KB (just scalars)
         |
    del mb_logits                          (2.34GB freed, but peak already hit)
         |
         v
    loss_fn(...) -> backward

    PEAK VRAM: weights(8.5) + LoRA(0.1) + optim(0.2) + activations(1.4)
             + logits(2.34) + overhead(0.5) = ~13.0GB on 1.7B
             For 14B: weights(8.5) + LoRA(0.3) + optim(0.6) + activations(2.8)
             + logits(2.34) + overhead(0.5) = ~15.0GB  <-- too close to 16GB
```

### Fused Forward Path (OPT-1: chunked_logprobs_from_hidden)

**CRITICAL: Deep analysis (2026-03-23) found that naive chunking saves ZERO net memory
because autograd stores all chunk logits for backward. The fix: torch.checkpoint per chunk.**

```
                        FUSED PATH — Chunked lm_head + Checkpoint
                        ==========================================

    input_ids [1, 4096]
         |
         v
    +---------------------+
    | body(mb_ids,        |   Body-only forward (transformer blocks, no lm_head)
    |   attention_mask)   |   Activations: same ~2.8GB (grad checkpointing on)
    +---------------------+
         |
         v
    hidden_states [1, 4096, 5120]          <-- 40MB (bf16, 14B hidden_dim=5120)
         |                                      requires_grad=True (CRITICAL)
         |
         v  FOR chunk IN range(0, 4096, 256):
    +-------------------------------+
    | torch.checkpoint:             |      <-- CRITICAL: prevents autograd from
    |   chunk_hidden [1, 256, 5120] |          storing chunk_logits for backward.
    |         |                     |          Without checkpoint, autograd saves
    |    lm_head(chunk_hidden)      |          ALL 16 chunks = 2.37GB (same as
    |     .float()                  |          full logits). Checkpoint recomputes
    |         |                     |          each chunk during backward instead.
    |    selective_log_softmax      |
    |         |                     |
    |    chunk_logprobs [1, 256]    |      <-- 1KB (gathered scalars only)
    +-------------------------------+
         |
         v
    mb_lp = torch.cat(chunks)              <-- CRITICAL: NOT in-place assignment.
         |                                      torch.zeros + slice assignment breaks
         |                                      autograd graph. torch.cat preserves it.
         v
    loss_fn(...) -> backward
         |
         v  (backward recomputes each chunk's lm_head on the fly — 2× lm_head compute)

    PEAK VRAM: weights(8.5) + LoRA(0.3) + optim(0.6) + activations(2.8)
             + hidden(0.04) + 1_chunk_recompute(0.15) + overhead(0.5) = ~12.9GB
             Headroom: ~3.1GB  <-- SAFE on 16GB
             Savings vs current: ~2.1GB (logits tensor eliminated from BOTH fwd and bwd)
```

**Why naive chunking fails (without checkpoint):**
`del chunk_logits` only deletes the Python reference. Autograd's saved_tensors still
holds it because `torch.logsumexp` backward requires the original logits and
`torch.gather` backward requires logits.shape. All 16 chunks × 148MB = 2.37GB stored
in the autograd graph — the same memory as the full logit tensor. Chunking moves the
mountain from forward to backward; it doesn't remove it.

**Why torch.cat instead of in-place assignment:**
`result = torch.zeros(...)` creates a leaf tensor with no grad_fn. `result[:, s:e] = val`
copies values but severs the autograd graph. `loss.backward()` stops at `result` — zero
gradients reach the model. Using `torch.cat(chunks, dim=1)` creates a proper graph node
that connects all chunk results back to hidden_states → body → model parameters.

### Memory Budget Breakdown (14B, 4-bit, seq_len=4096)

**REQUIRES: Unsloth Standby ON (`UNSLOTH_VLLM_STANDBY=1`) + torch.checkpoint per chunk**

```
    Component                 Current Path    Fused+Ckpt    Delta
    ---------------------------------------------------------------
    Base weights (4-bit)      8,500 MB        8,500 MB        0
    LoRA adapters (rank 8)      300 MB          300 MB        0
    Optimizer (AdamW8bit)       600 MB          600 MB        0
    Activations (grad ckpt)   2,800 MB        2,800 MB        0
    Logit tensor (fwd)        2,340 MB            0 MB    -2,340
    Autograd saved logits(bwd)    0 MB*           0 MB**      0
    Hidden states buffer          0 MB           40 MB       +40
    Chunk recompute (bwd peak)    0 MB          148 MB      +148
    vLLM (WITH Standby)           0 MB***         0 MB***     0
    CUDA overhead + frags       500 MB          500 MB        0
    ---------------------------------------------------------------
    PEAK TOTAL               15,040 MB       12,888 MB    -2,152
    16GB headroom                960 MB        3,112 MB

    *  Current path: logits freed after logprobs computed (before backward)
    ** Fused+Ckpt: torch.checkpoint recomputes per chunk during backward
    *** Standby offloads vLLM weights during training, reloads for generation

    WITHOUT STANDBY (vLLM stays loaded):
    vLLM reservation (0.35×16GB) = 5,600 MB
    Fused+Ckpt total WITH vLLM  = 18,488 MB  <-- DOES NOT FIT
    This is why Standby is a HARD P0 DEPENDENCY, not an alternative.
```

## Deliverables

### P0: Ship

**OPT-0: Unsloth Standby (HARD DEPENDENCY)**
- Set `UNSLOTH_VLLM_STANDBY=1` before Unsloth import
- Set `gpu_memory_utilization=0.95` (Standby handles the sharing)
- Without Standby, vLLM reserves 5.6GB (0.35×16GB), making 14B impossible regardless of other opts
- Acceptance: `nvidia-smi` shows vLLM memory freed during training step, reclaimed during generation

**OPT-1: Fused Logprobs (Chunked lm_head + Checkpoint)**
- Wire `chunked_logprobs_from_hidden` into `trainer.step()` as the training forward path
- Split body forward from lm_head projection, process lm_head in 256-token chunks
- **CRITICAL: Wrap each chunk in `torch.utils.checkpoint.checkpoint()`** — without this, autograd stores all 16 chunk logits (2.37GB) for backward, negating the savings entirely
- **CRITICAL: Use `torch.cat(chunks)` NOT in-place assignment to `torch.zeros`** — in-place to a leaf tensor severs the autograd graph, causing silent gradient death
- **CRITICAL: Cast chunk_logits to float32 before `selective_log_softmax`** — bf16 path uses Python for-loop per row (10-50× slower)
- Must pass attention_mask through body forward
- Must preserve autograd graph (verify .grad is not None AND .grad_fn is not None on mb_lp)
- Must work through Unsloth PeftModel wrapper chain (test on actual Unsloth-loaded model)
- Config: `algorithm.use_fused_logprobs: true` (default true)
- Fallback: if body/lm_head split fails, fall back to full `self.model(mb_ids)` forward
- Acceptance: logprob allclose within 1e-3 of full forward; loss decreases after optimizer step; peak VRAM reduction >= 800MB on BOTH forward AND backward peaks on Qwen3-1.7B

**OPT-3: Gate KL Region Weights**
- Skip `kl_region_weights` construction when `kl_cov_ratio == 0.0` or `loss_mode != "kl_cov"`
- Pass `kl_region_weights=None` to loss function (already handled)
- Acceptance: existing tests pass; no Python loop over tokens when KL disabled

**OPT-5: Tensor Cleanup**
- `del batch_regions` after KL region weights construction
- `del` pre-filter tensors after SPO reindexing
- Acceptance: no test regressions; peak memory <= baseline

### P1: Ship with Measurement

**OPT-2: CUDA empty_cache Between Micro-batches**
- Add `torch.cuda.empty_cache()` after `del mb_lp` at end of each micro-batch iteration
- DO NOT add empty_cache inside `chunked_logprobs_from_hidden` -- the sync cost is too high
- Config: `training.empty_cache_between_microbatches: true` (default true)
- Acceptance: peak VRAM reduction >= 100MB AND wall-clock regression < 5%
- Kill criterion: if wall-clock regression > 10%, disable by default

### P2: Do Not Ship

**OPT-4: Fused Optimizer into Backward Pass**
- Rejected. Incompatible with bitsandbytes AdamW8bit. Changes grad clipping semantics. Risk/reward ratio is wrong for an engine in active training.
- Revisit only if: (a) bnb publishes per-parameter step support, or (b) engine migrates away from bnb 8-bit optimizer.

## Implementation Order

0. OPT-0 (Standby) -- 5 minutes, HARD BLOCKER for 14B, must be first
1. OPT-3 (gate KL) -- 15 minutes, zero risk, unblocks measurement of other opts
2. OPT-5 (tensor cleanup) -- 15 minutes, hygiene
3. OPT-1 (fused logprobs + checkpoint + torch.cat) -- 2-4 hours, highest impact, highest risk
4. OPT-2 (empty_cache) -- 30 minutes, measure after OPT-1 is in

## Risk Mitigation

### OPT-1: Fused Logprobs

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Unsloth wrapper chain breaks `get_hidden_states_and_lm_head` | High | Blocking | Test wrapper resolution on Unsloth-loaded Qwen3-1.7B FIRST, before writing trainer integration. The `body` attribute may not be `inner.model` -- Unsloth 2025.3+ wraps differently. Write a standalone test: load model, call `get_hidden_states_and_lm_head`, verify hidden_states.shape and lm_head.weight.shape. |
| `body(input_ids)` ignores attention_mask | High | Silent corruption | The current `get_hidden_states_and_lm_head` on line 100 of `fused_logprobs.py` calls `body(input_ids)` WITHOUT attention_mask. This MUST be fixed -- padded sequences will attend to padding tokens, producing wrong hidden states. Fix: pass `attention_mask` as a parameter to `get_hidden_states_and_lm_head` and forward it to `body()`. |
| Autograd graph detached by Unsloth | Medium | Silent failure (no gradients) | After the body forward, assert `hidden_states.requires_grad is True`. If False, Unsloth's gradient checkpointing is detaching the output. Mitigation: try `hidden_states = hidden_states.requires_grad_(True)` as a workaround, but verify gradient flow end-to-end by checking that `model.parameters()` have non-zero `.grad` after backward. |
| Chunk boundary artifacts in logprobs | Low | Numeric drift | log_softmax is computed per-chunk. Since each chunk sees the full vocab dim, there are no cross-chunk dependencies in the softmax computation. The result is mathematically identical. Verify with `torch.allclose(fused, full, atol=1e-3)`. |
| Performance regression from 16 sequential matmuls | Medium | Slowdown | 16 chunks of [1, 256, 5120] @ [5120, 151936] is the same total FLOP count as one [1, 4096, 5120] @ [5120, 151936]. The overhead is kernel launch latency (~16 x 10us = 0.16ms). Acceptable. If measured regression > 10%, increase chunk_size to 512 (doubles peak chunk memory to 296MB, still well within budget). |

### OPT-2: CUDA empty_cache

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Wall-clock regression from CUDA sync | Medium | Slowdown | `empty_cache()` triggers a CUDA synchronization. Measure on 50 steps, compare. If > 10% regression, disable by default and only enable for 14B models via config. |
| No measurable VRAM benefit | Medium | Wasted effort | If allocator is already reusing freed blocks effectively, empty_cache adds nothing. Measure with `torch.cuda.max_memory_allocated()`. If < 100MB improvement, revert. |

### OPT-3: Gate KL Region Weights

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Gate condition wrong -- skips KL when it should be active | Low | Training regression | The gate checks `kl_cov_ratio == 0.0`. Verify this is the right field. If the loss function uses a different field to enable KL, the gate will silently disable it. Write a test: config with `kl_cov_ratio=0.5`, assert `kl_region_weights is not None`. |

### OPT-5: Tensor Cleanup

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Deleting tensor still referenced downstream | Low | Runtime crash | Grep for every use of `batch_regions` after the KL weights construction. Currently it's only used in lines 288-294 of trainer.py. If a future change references it later, the `del` will cause a NameError. Safe today. |

## Rollback Plan

Each optimization is independently toggleable. Rollback is a config change, not a code revert.

### OPT-1: Fused Logprobs
- **Config toggle:** `algorithm.use_fused_logprobs: false`
- **What happens:** Trainer falls back to full `self.model(mb_ids)` forward path (current code, lines 374-409 of trainer.py)
- **How to verify rollback works:** Run 10 training steps on Qwen3-1.7B with `use_fused_logprobs: false`. Compare loss and grad norms to a known-good baseline. Must match within 1%.
- **Code rollback:** If the config toggle path is itself broken, `git revert` the commit that wired fused logprobs into `trainer.step()`. The `fused_logprobs.py` module is standalone and doesn't affect anything unless called.

### OPT-2: CUDA empty_cache
- **Config toggle:** `training.empty_cache_between_microbatches: false`
- **What happens:** No `torch.cuda.empty_cache()` call between micro-batches. Returns to current behavior exactly.
- **Code rollback:** Delete the single `if` block around the `empty_cache()` call.

### OPT-3: Gate KL Region Weights
- **Config toggle:** None needed. The gate is transparent -- when `kl_cov_ratio > 0`, the weights are computed as before. When `kl_cov_ratio == 0`, they were already being computed and then ignored by the loss function.
- **Code rollback:** Remove the `if` guard. The only cost is a Python loop over tokens that takes ~2ms.

### OPT-5: Tensor Cleanup
- **Code rollback:** Remove the `del` statements. No config needed. The only cost is memory held slightly longer than necessary.

### Full Rollback (Nuclear Option)
If the entire optimization effort produces unexpected interactions:
1. `git revert` all optimization commits (they should be separate, one per OPT)
2. Set all config toggles to false
3. Run full test suite to confirm clean state
4. Re-baseline memory measurements

## Dependencies & Blockers

### OPT-1: Fused Logprobs

| Dependency | Status | Blocker? | Resolution |
|---|---|---|---|
| `get_hidden_states_and_lm_head()` resolves correctly on Unsloth-loaded model | UNTESTED | YES | Must run a standalone script that loads Qwen3-1.7B via Unsloth, calls the function, verifies shapes. Do this BEFORE writing trainer integration. |
| `body()` accepts `attention_mask` kwarg | UNTESTED | YES | Unsloth's patched transformer body may or may not forward `attention_mask`. Test with a padded input batch. If it doesn't accept the kwarg, investigate Unsloth's internal body wrapper. |
| `hidden_states.requires_grad` is True after body forward | UNTESTED | YES | If Unsloth's gradient checkpointing detaches the output, the entire fused path produces zero gradients silently. Must verify before proceeding. |
| `selective_log_softmax` handles bf16 input | VERIFIED | No | The function has a bf16 fallback path (per-row log_softmax). |
| Config field `algorithm.use_fused_logprobs` exists | NOT YET | No | Must be added to `AlgorithmConfig` in `config.py`. Trivial. |
| Unsloth `for_training()` called before body forward | VERIFIED | No | Already done in trainer.py lines 356-365. The fused path runs after this call. |

### OPT-2: CUDA empty_cache

| Dependency | Status | Blocker? | Resolution |
|---|---|---|---|
| OPT-1 landed and measured | NOT YET | YES | empty_cache measurement only makes sense after the dominant memory optimization is in place. Measuring before OPT-1 would give misleading results. |
| Config field `training.empty_cache_between_microbatches` exists | NOT YET | No | Add to `TrainingConfig` in `config.py`. |

### OPT-3: Gate KL Region Weights

| Dependency | Status | Blocker? | Resolution |
|---|---|---|---|
| None | -- | No | This is a pure refactor of existing code with no external dependencies. |

### OPT-5: Tensor Cleanup

| Dependency | Status | Blocker? | Resolution |
|---|---|---|---|
| Confirm `batch_regions` is not used after line 294 in trainer.py | VERIFIED | No | Grepped. Only used in the KL weights loop. |

## Metrics & Monitoring

### During Implementation (per-optimization)

Run these after each OPT is implemented, BEFORE moving to the next:

```python
# Memory measurement snippet — add to a test or standalone script
import torch

torch.cuda.reset_peak_memory_stats()
# ... run 10 training steps ...
peak_mb = torch.cuda.max_memory_allocated() / 1024**2
current_mb = torch.cuda.memory_allocated() / 1024**2
reserved_mb = torch.cuda.memory_reserved() / 1024**2

print(f"Peak: {peak_mb:.0f}MB  Current: {current_mb:.0f}MB  Reserved: {reserved_mb:.0f}MB")
```

| Metric | How to Measure | Expected (1.7B) | Expected (14B) | Alert Threshold |
|---|---|---|---|---|
| Peak VRAM | `torch.cuda.max_memory_allocated()` | Baseline - 800MB after OPT-1 | Under 14GB after all opts | > 15GB on 14B |
| Step wall-clock | `time.perf_counter()` around `trainer.step()` | < 10% regression | < 15% regression | > 20% regression |
| Loss at step 100 | 100-step fixed-seed run on 1.7B | Within 5% of baseline | N/A (no 14B baseline yet) | > 10% divergence |
| Grad norm | `torch.nn.utils.clip_grad_norm_()` return value | Non-zero, stable | Non-zero, stable | Zero (dead gradients) or > 100 (exploding) |
| `neg_logprob_mean` | Already logged in trainer.py line 434 | Stable, no sudden jumps | Stable | > 2x jump between consecutive steps |

### After Deployment (ongoing)

These are the signals that tell you the optimizations are working correctly over long training runs:

| Metric | What It Means | Watch For |
|---|---|---|
| `reward/mean` trend | Model is learning | Flat or decreasing after 200+ steps = possible gradient issue |
| `completion_length/mean` | Model verbosity | Sudden increase = possible logprob corruption affecting length penalty |
| `kl_penalty` (when enabled) | Policy drift from reference | Sudden spike after enabling fused path = logprob mismatch |
| `neg_logprob_mean` | Policy entropy | Collapse to near-zero = the model is memorizing, not generalizing |
| CUDA OOM frequency | Memory headroom | Any OOM on 14B after opts = insufficient savings, revisit budget |

### Measurement Baseline Protocol

Before implementing ANY optimization, record the baseline on Qwen3-1.7B:

1. Fixed seed (42), fixed dataset (first 100 prompts from Hamiltonian example)
2. Record: peak VRAM, step wall-clock (mean of 100 steps), loss at step 100, grad norm at step 100
3. Save these numbers. Every optimization is measured against THIS baseline.
4. After ALL optimizations are in, run the same 100 steps again and compare.

## Key Files

- `qgre/trainer.py` -- training loop (lines 236-545)
- `qgre/fused_logprobs.py` -- existing chunked implementation
- `qgre/triton_logprobs.py` -- Triton kernel (inference-only, no backward)
- `qgre/config.py` -- config dataclasses (needs new fields)
- `qgre/nemo_extracted/loss_functions.py` -- ClippedPGLossFn (already handles kl_region_weights=None)
- `qgre/nemo_extracted/logits.py` -- selective_log_softmax used by chunked path

## Deep Analysis Findings (2026-03-23)

The following critical findings were discovered by deep analysis AFTER the initial pressure test:

### Finding 1: Naive chunking saves ZERO net memory (CRITICAL)
`del chunk_logits` only deletes the Python reference. Autograd's saved_tensors holds the chunk logits because `torch.logsumexp` backward requires the original logits and `torch.gather` backward requires logits.shape. All 16 chunks × 148MB = 2.37GB stored in the autograd graph — identical to the full logit tensor. **Fix:** `torch.utils.checkpoint.checkpoint()` per chunk.

### Finding 2: vLLM reserves 5.6GB not accounted in budget (CRITICAL)
`gpu_memory_utilization=0.35` × 16GB = 5.6GB reserved by vLLM. The original memory budget assumed training has 16GB available. Real available: 10.4GB. All optimizations combined save 2.1GB but the deficit is 5.6GB. **Fix:** Unsloth Standby (`UNSLOTH_VLLM_STANDBY=1`) offloads vLLM during training.

### Finding 3: In-place assignment to torch.zeros breaks autograd (CRITICAL)
`result = torch.zeros(...)` creates a leaf tensor with no grad_fn. `result[:, s:e] = val` copies values but severs the graph. `loss.backward()` produces zero gradients for all model parameters. **Fix:** Use `torch.cat(chunks, dim=1)` instead.

### Finding 4: bf16 path in selective_log_softmax is 10-50× slower (IMPORTANT)
Unsloth runs in bf16. The bf16 path in `selective_log_softmax` uses a Python for-loop per row. **Fix:** Cast `chunk_logits = chunk_logits.float()` before calling selective_log_softmax.

### Finding 5: On-policy ratio masks gradient death (IMPORTANT)
With `force_on_policy_ratio=True`, `log_ratios = curr - curr.detach() = 0`, `ratio = 1`. Loss computes as `-advantages × 1`. Loss looks normal and decreases from momentum even with zero gradients. **Detection:** Check `mb_lp.grad_fn is not None` not just loss values.

### Finding 6: vLLM recreation at step 50 causes memory spike (IMPORTANT)
`recreate_engine()` runs every 50 steps. During recreation, old and new engine overlap. Canary protocol tests 10 and 100 steps — misses the spike. **Fix:** Add step 55 canary.

## Pressure Test Date

2026-03-23
