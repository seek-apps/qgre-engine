# Rejection Framework: Memory Optimizations

Post-implementation kill criteria. If any signal fires, revert the optimization immediately.

## Pre-flight Checks

Before writing ANY optimization code, verify these conditions. If any check fails, resolve it first.

### Environment Pre-flight

| Check | Command / Method | Expected Result | If It Fails |
|---|---|---|---|
| Unsloth loads Qwen3-1.7B | `FastLanguageModel.from_pretrained("unsloth/Qwen3-1.7B", load_in_4bit=True)` | Model loads, no errors | Unsloth version mismatch. Pin to known-good version in requirements. |
| **Unsloth Standby works** | `os.environ["UNSLOTH_VLLM_STANDBY"] = "1"` before import, `gpu_memory_utilization=0.95` | vLLM memory freed during training (check nvidia-smi) | Standby is broken or not supported on this Unsloth version. **HARD BLOCKER** — without Standby, vLLM reserves 5.6GB making 14B impossible. |
| `PYTORCH_CUDA_ALLOC_CONF` set | `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | No fragmentation-related false OOMs | Driver doesn't support expandable_segments. Remove the flag and monitor fragmentation manually. |
| VRAM baseline is stable | Run 10 training steps, record peak VRAM 3 times | Variance < 50MB between runs | CUDA allocator non-determinism. Re-measure with expandable_segments. |
| Full test suite passes | `python -m pytest tests/ -q` | All 77 tests pass (52 reward + 25 trainer) | Do not start optimizations on a broken baseline. Fix tests first. |
| `get_hidden_states_and_lm_head()` resolves | Load model via Unsloth, call function, check return types | `(Tensor[1, seq, hidden], nn.Linear)` -- neither is None | Wrapper chain broken. Debug in `fused_logprobs.py` lines 78-96. Print `type(inner)` at each step to find where the chain breaks. |
| Body accepts attention_mask | `body(input_ids, attention_mask=mask)` | No TypeError | Body wrapper doesn't forward kwargs. Investigate Unsloth's model wrapping. May need to pass via `position_ids` or use the full model's forward with `output_hidden_states=True` instead. |
| Hidden states have grad | After body forward, check `hidden_states.requires_grad` | True | Unsloth detaches output. Try: (a) `hidden_states.requires_grad_(True)`, (b) disable Unsloth gradient checkpointing temporarily to test, (c) use `model(input_ids, output_hidden_states=True)` and extract last hidden from full forward. |
| **torch.cat preserves grad_fn** | `chunks = [f(h[:, s:e]) for s,e in ...]; result = torch.cat(chunks, dim=1); assert result.grad_fn is not None` | result.grad_fn is CatBackward | If None, the chunk computation detached somewhere. Check each chunk's grad_fn individually. |
| **torch.checkpoint works with lm_head** | `out = torch.utils.checkpoint.checkpoint(lm_head, chunk, use_reentrant=False)` | No error, output matches non-checkpointed | Unsloth's lm_head may not be compatible with checkpoint. If fails, fall back to non-checkpointed chunking (accepts 2.37GB autograd storage as a known cost). |

### Code Pre-flight

| Check | What to Verify | How |
|---|---|---|
| `batch_regions` usage scope | Confirm it's only used in KL weights construction (trainer.py lines 284-294) | `grep -n batch_regions qgre/trainer.py` -- should only appear in that block |
| `kl_cov_ratio` is the right gate field | Confirm this is the field that enables/disables KL covariance in the loss | Read `ClippedPGLossFn` -- verify it checks this field to decide whether to use `kl_region_weights` |
| Config dataclass has room for new fields | Check `AlgorithmConfig` and `TrainingConfig` in `config.py` | Read the dataclasses. Add `use_fused_logprobs: bool = True` and `empty_cache_between_microbatches: bool = True` |
| selective_log_softmax handles bf16 | Verify the function doesn't crash on bf16 input | Read `qgre/nemo_extracted/logits.py` -- check for dtype handling |

## Per-Optimization Kill Signals

### OPT-1: Fused Logprobs

| Kill Signal | How to Measure | Revert Threshold |
|---|---|---|
| Silent gradient death | `sum(abs(p.grad) for p in model.parameters())` after backward | == 0 on any step |
| **Autograd graph severed** | **`mb_lp.grad_fn is not None` after chunked_logprobs** | **grad_fn is None = graph broken** |
| Numeric divergence | `torch.allclose(fused_logprobs, full_logprobs, atol=1e-3)` | False on any sequence |
| Loss curve divergence | 100-step loss comparison vs baseline on Qwen3-1.7B | > 10% divergence at step 100 |
| Wrapper resolution failure | `get_hidden_states_and_lm_head()` returns `(None, None)` | On any Unsloth-loaded model |
| Attention mask mismatch | Compare hidden states with/without padding in batch | > 1e-3 divergence on padded sequences |
| **Backward peak not reduced** | **`torch.cuda.max_memory_allocated()` during backward** | **Backward peak >= forward peak of baseline (autograd stored all chunks)** |
| **bf16 slow path hit** | **Step wall-clock with vs without `.float()` cast** | **> 50% slower = bf16 path active** |

**Known landmines (updated from deep analysis):**
1. `body(input_ids)` must receive attention_mask -- without it, padding tokens get attended to
2. Unsloth PeftModel wrapper depth changed in 2025.3 -- verify wrapper-walking resolves correctly
3. Hidden states must have `requires_grad=True` after body forward -- Unsloth may detach
4. **`torch.zeros` + in-place slice assignment = gradient death.** Use `torch.cat(chunks)` instead. Root cause: leaf tensor has no grad_fn, in-place write copies values but not graph.
5. **Without `torch.checkpoint` per chunk, autograd saves all 16 chunk logits (2.37GB).** This negates the forward savings entirely. The fix costs 2× lm_head compute but saves 2.2GB.
6. **`selective_log_softmax` bf16 path is 10-50× slower.** Cast to float32 before calling.
7. **`force_on_policy_ratio=True` masks gradient death.** Loss looks normal (ratio=1, loss=-advantages) even with zero gradients. Check `mb_lp.grad_fn`, not just loss values.

### OPT-2: CUDA empty_cache

| Kill Signal | How to Measure | Revert Threshold |
|---|---|---|
| Wall-clock regression | Mean step time over 50 steps | > 10% slower than without |
| No VRAM benefit | `torch.cuda.max_memory_allocated()` comparison | < 100MB reduction |

**Known landmines:**
1. DO NOT put empty_cache inside `chunked_logprobs_from_hidden` -- 16 sync points per sequence kills perf
2. Only between micro-batches (after `del mb_lp`)

### OPT-3: Gate KL Region Weights

| Kill Signal | How to Measure | Revert Threshold |
|---|---|---|
| KL loss missing when expected | Metrics `kl_penalty` with `kl_cov_ratio > 0` | == 0 when config enables KL |
| Test regression | Full test suite | Any failure |

### OPT-4: Fused Optimizer -- REJECTED

Do not implement. Reasons:
- bitsandbytes AdamW8bit maintains quantized state tensors that expect full gradient accumulation before step()
- Per-parameter stepping during backward produces corrupt optimizer states
- Changes global grad norm clipping to per-parameter clipping (different algorithm)
- 268MB ceiling savings (LoRA rank 8 on 14B) doesn't justify the risk

**Revisit conditions:**
- bitsandbytes publishes per-parameter step support
- Engine migrates to standard AdamW (losing 4x optimizer memory savings)
- gradient_accumulation_steps permanently fixed to 1 with no possibility of change

### OPT-5: Tensor Cleanup

| Kill Signal | How to Measure | Revert Threshold |
|---|---|---|
| Test regression | Full test suite | Any failure |
| Accessing deleted variable | Runtime NameError | Any occurrence |

## Canary Testing Protocol

Every optimization is tested on Qwen3-1.7B BEFORE attempting Qwen3-14B. The 1.7B model is the canary -- if it dies, the 14B model would too, and you've wasted less time debugging.

### Step-by-step procedure for each optimization:

**Phase 1: Unit Validation (no GPU needed for OPT-3, OPT-5)**

1. Run `python -m pytest tests/ -q` -- full suite must pass before starting.
2. Implement the optimization.
3. Run `python -m pytest tests/ -q` again -- full suite must still pass.
4. If any test fails, fix it before proceeding. Do not rationalize the failure.

**Phase 2: 1.7B Canary (GPU required, for OPT-1 and OPT-2)**

1. Load Qwen3-1.7B via Unsloth with 4-bit quantization.
2. Run 10 training steps on the Hamiltonian example dataset.
3. Record:
   - Peak VRAM (`torch.cuda.max_memory_allocated()`)
   - Step wall-clock (mean of 10 steps)
   - Loss at step 10
   - Grad norm at step 10 (`clip_grad_norm_` return value)
4. Compare against the 1.7B baseline (recorded in pre-flight).
5. **Kill criteria at this stage:**
   - Peak VRAM INCREASED (optimization made things worse)
   - Grad norm is zero (gradient death)
   - Loss is NaN or Inf
   - Wall-clock > 20% slower (something is very wrong)
6. If canary passes, run the 100-step comparison:
   - Fixed seed 42, first 100 Hamiltonian prompts
   - Loss at step 100 within 5% of baseline
   - `neg_logprob_mean` within 10% of baseline

**Phase 3: Numeric Equivalence (OPT-1 only)**

1. Run a SINGLE forward pass on a known input through BOTH paths:
   - Current: `logprobs_from_logits(model(input_ids).logits[:, :-1], input_ids[:, 1:])`
   - Fused: `chunked_logprobs_from_hidden(hidden_states[:, :-1], lm_head, input_ids[:, 1:])`
2. Assert `torch.allclose(current_lp, fused_lp, atol=1e-3)`.
3. If this fails, the fused path has a bug. Do not proceed to training.
4. Test with PADDED input (two sequences, different lengths, padded to max). The padded positions must also match.

**Phase 3.5: Backward Peak Measurement (OPT-1 only)**

1. Instrument backward pass: record `torch.cuda.max_memory_allocated()` BEFORE and AFTER `.backward()`.
2. The backward peak MUST be less than the full-forward baseline's forward peak.
3. If backward peak >= baseline forward peak, `torch.checkpoint` is not working — autograd is storing all chunk logits.
4. Also check: `mb_lp.grad_fn is not None`. If None, the `torch.cat` or in-place assignment broke the graph.

**Phase 3.75: vLLM Recreation Spike (Step 55 canary)**

1. Run 55 training steps (past the step-50 `recreate_engine()` boundary).
2. If OOM occurs at step 50-51, the vLLM recreation spike exceeds headroom.
3. Record peak VRAM at step 49 (before recreation) and step 51 (after recreation).
4. If the spike > 1GB above steady-state, consider disabling periodic recreation for 14B.

**Phase 4: 14B Attempt (only after 1.7B canary passes for ALL opts)**

1. **Verify Standby is active:** `UNSLOTH_VLLM_STANDBY=1` must be set. Without it, 14B CANNOT fit.
2. Load Qwen3-14B via Unsloth with 4-bit quantization and `gpu_memory_utilization=0.95`.
3. Record VRAM after model load (before training): should be ~8.5GB.
4. Run 1 training step. If OOM, record the peak VRAM and stop. Calculate how much more savings are needed.
5. If step 1 succeeds, run 10 steps. Record peak VRAM and wall-clock.
6. If step 10 succeeds, run 55 steps (test through vLLM recreation).
7. If step 55 succeeds, run 100 steps. Record loss curve.
8. Compare 14B loss curve shape (not absolute values) to 1.7B -- should show same convergence pattern.

## Failure Taxonomy

Every failure mode for every optimization falls into one of three categories. The category determines the response.

### Category A: Silent Failures

Training proceeds. No crash. No error. But the model learns wrong things or learns nothing.

**These are the most dangerous failures because they waste GPU hours.**

| Optimization | Silent Failure Mode | How It Manifests | Detection |
|---|---|---|---|
| OPT-1 | Gradient death from detached hidden states | Loss decreases initially (from momentum), then plateaus. Model outputs don't improve. | Check `sum(abs(p.grad))` after backward -- if zero, gradients are dead. Also compare `neg_logprob_mean` trend: if it never changes, policy isn't updating. |
| OPT-1 | Wrong logprobs from missing attention_mask | Model attends to padding tokens, producing corrupted hidden states. Logprobs are numerically different from full forward. Training converges to a different (worse) policy. | Numeric equivalence test (Phase 3 of canary protocol). If `allclose` fails, this is likely the cause. |
| OPT-1 | Chunk boundary misalignment | Off-by-one in chunk slicing causes labels to misalign with hidden states. Every logprob is computed for the wrong token. Loss is finite but meaningless. | Compare first and last chunk's logprobs against full forward individually. Check that `labels[:, start:end]` corresponds to `hidden_states[:, start:end]`. |
| OPT-1 | **Autograd stores all chunk logits (deep analysis finding)** | **`del chunk_logits` only deletes Python ref. Autograd holds 16×148MB=2.37GB in saved_tensors. Backward peak equals full logit tensor. Memory "savings" are illusory.** | **Measure backward peak with `torch.cuda.max_memory_allocated()` around `.backward()`. If backward peak >= baseline forward peak, torch.checkpoint is not active.** |
| OPT-1 | **torch.zeros in-place assignment kills grad (deep analysis finding)** | **`result = torch.zeros(...)` is a leaf with no grad_fn. Slice assignment copies values but not graph. `loss.backward()` produces zero grads. Loss still decreases from advantages×ratio=1 masking the death.** | **Check `mb_lp.grad_fn is not None` BEFORE passing to loss_fn. If None, the graph is severed.** |
| OPT-3 | KL gate fires when KL should be active | `kl_cov_ratio` is non-zero but the gate skips construction. KL penalty disappears. Model drifts from reference unchecked. | Monitor `kl_penalty` metric. If it's always 0.0 when `kl_cov_ratio > 0` in config, the gate is wrong. |

**Response to Category A failures:** STOP training immediately. Revert the optimization. Do not attempt to fix while a training run is in progress -- you'll waste GPU hours on corrupted weights.

### Category B: Loud Failures

Training crashes with an error. OOM, RuntimeError, TypeError, NameError.

**These are annoying but safe -- no GPU hours wasted on corrupt training.**

| Optimization | Loud Failure Mode | Error Message | Fix |
|---|---|---|---|
| OPT-1 | Unsloth wrapper chain doesn't resolve | `get_hidden_states_and_lm_head()` returns `(None, None)` -> fallback to full forward -> OOM on 14B | Debug wrapper chain. Print `type(inner)` at each level. Unsloth wraps as: `PeftModelForCausalLM -> PeftModel -> base_model -> model -> Qwen3ForCausalLM`. The lm_head is on the Qwen3ForCausalLM level. |
| OPT-1 | body() doesn't accept attention_mask | `TypeError: forward() got an unexpected keyword argument 'attention_mask'` | The body may use a different kwarg name, or the Unsloth-patched body may strip kwargs. Check Unsloth source for the patched forward signature. |
| OPT-1 | OOM even with fused path | CUDA OOM during body forward (not lm_head) | The savings from fused logprobs are insufficient. Activations dominate. Enable Unsloth's `use_gradient_checkpointing = "unsloth"` if not already on. If already on, reduce seq_len to 2048 for 14B. |
| OPT-2 | empty_cache causes deadlock | Hangs at `torch.cuda.empty_cache()` (extremely rare, usually driver bug) | Remove the call. This is a CUDA driver issue, not a code bug. |
| OPT-5 | NameError on deleted tensor | `NameError: name 'batch_regions' is not defined` | Some code path references `batch_regions` after the `del`. Find the reference and move the `del` after it. |

**Response to Category B failures:** Read the error. Fix the specific issue. Re-run canary from Phase 1.

### Category C: Slow Failures (Performance Regression)

Training runs correctly but is slower than acceptable.

| Optimization | Slow Failure Mode | Detection | Threshold |
|---|---|---|---|
| OPT-1 | Chunk loop overhead | Step wall-clock measurement | > 10% slower than full forward |
| OPT-1 | Cache thrashing from repeated small allocations | GPU utilization drops (nvidia-smi shows low GPU%) | GPU util < 80% when baseline is > 90% |
| OPT-2 | CUDA sync from empty_cache | Step wall-clock measurement | > 10% slower than without |
| OPT-2 | Allocator fragmentation after cache clear | Subsequent allocations are slower because cached blocks were freed | Step times increase over the run (first 10 steps faster than last 10) |

**Response to Category C failures:** Disable the optimization via config toggle. Do not revert the code -- it may be useful on different hardware or model sizes. Set the config default to `false` and document the performance characteristics.

## Decision Tree

For each optimization, follow this flowchart during implementation. Do not skip steps.

### OPT-1: Fused Logprobs Decision Tree

```
START
  |
  v
[Pre-flight: Does get_hidden_states_and_lm_head() resolve on Unsloth model?]
  |                                    |
  YES                                  NO
  |                                    |
  v                                    v
[Does body() accept attention_mask?]   STOP. Debug wrapper chain.
  |                    |               Do not proceed until resolved.
  YES                  NO
  |                    |
  v                    v
  |                    [Does model(ids, output_hidden_states=True) work?]
  |                      |                    |
  |                      YES                  NO
  |                      |                    |
  |                      v                    v
  |                    Use alternative:       STOP. Cannot split
  |                    full forward with      body from lm_head.
  |                    output_hidden_states,  File Unsloth issue.
  |                    extract last hidden,   Fall back to current path.
  |                    then chunk lm_head.
  |                      |
  +<---------------------+
  |
  v
[Does hidden_states.requires_grad == True?]
  |                    |
  YES                  NO
  |                    |
  v                    v
  |                  [Try hidden_states.requires_grad_(True)]
  |                    |
  |                    v
  |                  [Run backward. Are param grads non-zero?]
  |                    |                    |
  |                    YES                  NO
  |                    |                    |
  |                    v                    v
  +<-------------------+                  STOP. Autograd graph
  |                                       is broken. The body
  |                                       forward doesn't connect
  |                                       to model parameters.
  |                                       Fall back to current path.
  v
[Implement fused path in trainer.step()]
  |
  v
[Run numeric equivalence test (Phase 3)]
  |                    |
  PASS                 FAIL
  |                    |
  v                    v
  |                  Debug: compare chunk-by-chunk.
  |                  Check label alignment.
  |                  Check attention_mask forwarding.
  |                  Fix and re-test.
  |
  v
[Run 1.7B canary (Phase 2): 10 steps]
  |                    |
  PASS                 FAIL
  |                    |
  v                    v
  |                  [Is it OOM?]
  |                    |         |
  |                    YES       NO (gradient/loss issue)
  |                    |         |
  |                    v         v
  |                  Bug in      Debug gradient flow.
  |                  fused       Check kill signals table.
  |                  path --     Revert if not fixable
  |                  should      in 30 minutes.
  |                  use LESS
  |                  memory.
  |                  Debug.
  |
  v
[Run 100-step comparison on 1.7B]
  |                    |
  PASS (< 5% div)     FAIL (> 5% divergence)
  |                    |
  v                    v
  |                  [Is divergence > 10%?]
  |                    |              |
  |                    YES            NO (5-10%)
  |                    |              |
  |                    v              v
  |                  Silent          Acceptable.
  |                  corruption.     Log the divergence.
  |                  REVERT.         Monitor over longer
  |                  Debug offline.  runs. Ship with
  |                                  measurement.
  v
[Attempt 14B: 1 step]
  |                    |
  PASS                 OOM
  |                    |
  v                    v
[Run 100 steps]       Record peak VRAM.
  |                   Calculate gap to 16GB.
  v                   Evaluate: reduce seq_len?
SHIP IT.              Enable gradient checkpointing?
                      CPU offload optimizer?
```

### OPT-2: empty_cache Decision Tree

```
START (only after OPT-1 is shipped)
  |
  v
[Add empty_cache after del mb_lp, guarded by config]
  |
  v
[Measure 50-step wall-clock: enabled vs disabled]
  |
  v
[Regression > 10%?]
  |           |
  YES         NO
  |           |
  v           v
Set default   [Measure peak VRAM: enabled vs disabled]
to false.     |
Done.         v
              [Reduction >= 100MB?]
                |           |
                YES         NO
                |           |
                v           v
              SHIP IT.    Set default to false.
              Default     The allocator is already
              true.       reusing blocks effectively.
```

### OPT-3: Gate KL Decision Tree

```
START
  |
  v
[Add if-guard around KL region weights construction]
  |
  v
[Run full test suite]
  |           |
  PASS        FAIL
  |           |
  v           v
SHIP IT.    [Does failing test use kl_cov_ratio > 0?]
              |                    |
              YES                  NO
              |                    |
              v                    v
            Gate condition       Unrelated regression.
            is wrong. Fix       Fix the test, not
            the condition.      the gate.
```

### OPT-5: Tensor Cleanup Decision Tree

```
START
  |
  v
[Grep for all uses of batch_regions and pre-filter tensors]
  |
  v
[Any use AFTER the proposed del point?]
  |           |
  YES         NO
  |           |
  v           v
Move del    [Add del statements]
after last  |
use.        v
            [Run full test suite]
              |           |
              PASS        FAIL
              |           |
              v           v
            SHIP IT.    NameError? Move del.
                        Other error? Unrelated, fix it.
```

## Post-Mortem Template

If an optimization is reverted, fill out this template and save it to `docs/post-mortems/OPT-N-revert-YYYY-MM-DD.md`.

```markdown
# Post-Mortem: OPT-N [Name] Revert

## Date
YYYY-MM-DD

## What Was the Optimization?
One sentence: what it changed and what it was supposed to achieve.

## What Went Wrong?
Failure category: [Silent / Loud / Slow]
Specific failure mode from the taxonomy above.

## How Was It Detected?
Which kill signal fired? At which stage of the canary protocol?

## Timeline
- HH:MM — Implemented optimization
- HH:MM — Ran canary phase N
- HH:MM — Detected failure
- HH:MM — Reverted

## Root Cause
What specifically caused the failure? Not "it didn't work" but
the precise technical reason (e.g., "Unsloth 2025.3.12 wraps the
body forward with a detach() call for memory efficiency, breaking
the autograd graph through the hidden states").

## What Would Need to Change for This to Work?
Specific conditions under which this optimization becomes viable again.
(e.g., "Unsloth removes the detach(), or we use output_hidden_states
instead of body-only forward")

## Impact
- GPU hours wasted: N
- Training runs corrupted: N (should be 0 if canary protocol was followed)
- Time to detect and revert: N minutes

## Prevention
What pre-flight check should be added to prevent this class of failure in the future?
```

## Meta-Rejection Criterion

**Gate 0:** If Unsloth Standby does not work on the target hardware/version, STOP. 14B on 16GB is impossible without it — vLLM reserves 5.6GB leaving only 10.4GB for training.

**Gate 1:** If total VRAM savings from ALL shipped optimizations combined is **less than 500MB** measured on Qwen3-1.7B, the entire effort is insufficient for 14B on 16GB.

**Gate 2:** If backward peak is NOT reduced by torch.checkpoint (autograd still stores all chunk logits), the fused path saves zero net memory. Fall back to reducing seq_len.

**If any gate fails — alternative approaches to evaluate:**
1. Reducing max sequence length from 4096 to 2048 for 14B models (immediate, no engineering)
2. CPU offloading of optimizer states (bnb already does partial — evaluate full offload)
3. Activation checkpointing at transformer block level (Unsloth's `use_gradient_checkpointing = "unsloth"` — already enabled)
4. Using GGUF quantized weights instead of bnb 4-bit (different memory profile)
5. Using `torch.autograd.Function` with explicit save_for_backward (Liger Kernel pattern — heavy but proven)

## Measurement Protocol

Before declaring any optimization "shipped":

1. Run `torch.cuda.reset_peak_memory_stats()` before training step
2. Run 10 training steps
3. Record `torch.cuda.max_memory_allocated()` -- this is peak VRAM
4. Compare against baseline (same 10 steps without the optimization)
5. Record wall-clock time for the 10 steps
6. Run full test suite: `python -m pytest tests/ -q`
7. Run 100-step loss comparison on Qwen3-1.7B with fixed seed

## Pressure Test Date

2026-03-23
