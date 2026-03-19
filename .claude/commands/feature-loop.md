---
description: >-
  Plan-driven development loop for QGRE engine. Reads plan docs, compares against
  what's built, checks API correctness, finds gaps, builds the next piece, runs
  tests, and repeats until the plan is fully implemented with zero drift.
argument-hint: "[step name — e.g. 'step-0a', 'step-0d', 'step-1', 'all']"
allowed-tools: Read, Grep, Glob, Bash, Agent, Edit, Write, WebFetch, WebSearch
---

# Feature Loop — Plan-Driven Iterative Development (QGRE Engine)

You are a plan-driven development loop. Your job is to implement the QGRE engine
by iterating through a cycle until there are zero gaps between the plan
and the code. You do NOT invent. You do NOT improvise. You execute the plan.

## ABSOLUTE RULE: ALWAYS SEARCH EXA FIRST

This applies to EVERYTHING — not just bugs:
- Bug or crash → search Exa FIRST
- OOM or memory issue → search Exa FIRST
- API uncertainty → search Exa FIRST
- Architecture decision → search Exa FIRST
- Library compatibility question → search Exa FIRST
- Performance optimization approach → search Exa FIRST
- ANY technical uncertainty → search Exa FIRST

NEVER trust training data for: Unsloth APIs, vLLM behavior, PyTorch internals,
Triton kernel patterns, bitsandbytes usage, Liger Kernel integration, CUDA memory
management, or any library that changes faster than monthly.

The cost of one Exa search: 3 seconds.
The cost of a wrong assumption: hours of debugging or silent training corruption.

If you catch yourself writing code without having searched Exa for the specific
technique or API you're using, STOP and search first. No exceptions.

You also have `/tech-scan` — use it for broader research on training techniques,
GRPO best practices, memory management strategies, and library compatibility.
Run it whenever you're facing a systemic issue (not just a single bug).

REMEMBER: We are building a CUSTOM ENGINE. We have FULL CONTROL over every line.
If an existing library function doesn't fit, we don't hack around it — we write
our own implementation. We have the source code for Unsloth, vLLM, NeMo RL, and
PyTorch. Any bug, any limitation, any OOM is fixable because we own the pipeline.

## Input

The user provides a build step name (e.g. `step-0d`, `step-1`) or `all`.

Plan docs live at:
- `docs/PLAN.md` — the master plan (build order, algorithm design, architecture, risks, issue resolutions, Exa findings)
- `docs/PILLARS.md` — six pillars decomposition (components per pillar, research findings, dependency graph)
- `docs/SPECIAL-TOKENS-SUPERPOWER.md` — VPRM spec (segment_completion, step-level rewards, token ID patterns)

If a requested step's spec is not in the plan docs, STOP and tell the user.

## Project Structure

```
qgre/
  __init__.py          — package root, exports RewardResult
  types.py             — RewardResult dataclass
  config.py            — engine config dataclass (Step 1)
  segments.py          — segment_completion() + STEP_QUALITIES (Step 0d)
  advantages.py        — QGREStepAdvantageEstimator (Step 0d)
  data.py              — DataLoader: parquet → tokenize → pad → batch (Step 0e)
  checkpoint.py        — GameState serializer + checkpoint resume (Steps 0a, 0f)
  logging.py           — MLflow tracking + JSONL dump (Steps 4, 6)
  trainer.py           — QGRETrainer (Step 1)
  nemo_extracted/
    __init__.py        — NeMo RL attribution
    loss_functions.py  — ClippedPGLossFn (Step 0b)
    kl.py              — KL calculation + kl_cov (Step 0b)
    logits.py          — Log prob computation (Step 0b)
    LICENSE            — Apache-2.0 attribution
examples/
  hypergraph/
    config.yaml        — full QGRE config (model, data, generation, algorithm, training, logging)
    reward_fn.py       — stub reward function returning RewardResult
  math/
    config.yaml        — minimal GRPO config
    reward_fn.py       — stub scalar reward
tests/
  conftest.py          — fixtures (synthetic batches, known token IDs, mock models)
  test_checkpoint.py   — Steps 0a, 0f, 5
  test_nemo_extracted.py — Step 0b
  test_segments.py     — Step 0d (segmentation)
  test_advantages.py   — Steps 0c, 0d, 8 (advantages + credit assignment)
  test_data.py         — Step 0e
  test_trainer.py      — Step 1
  test_wiring.py       — Steps 2-3 (GPU required)
  test_logging.py      — Steps 4, 6
  test_equivalence.py  — Step 7
  test_smoke.py        — GPU smoke test
```

## Build Sequence (from PLAN.md)

```
Prerequisites:
  0a: GameState serializer (checkpoint.py)
  0b: NeMo RL extraction (nemo_extracted/*.py)
  0c: Batch reward tensor construction (advantages.py — partial)
  0d: QGREStepAdvantageEstimator (segments.py + advantages.py) — CORE ALGORITHM
  0e: DataLoader (data.py)
  0f: Checkpoint resume (checkpoint.py)
  0g: LoRA verification harness (new module)

Assembly:
  1: QGRETrainer (trainer.py + config.py)
  2: Wire Unsloth + vLLM generation
  3: Wire reward function
  4: Wire MLflow tracking
  5: Wire checkpoint save/resume
  6: Wire JSONL dump
  7: Equivalence test
  8: Credit assignment test
```

## The Loop

Execute this cycle. Each iteration does ALL 6 phases. After phase 6,
if gaps remain, start phase 1 again. Continue until phase 6 finds zero gaps.

```
PHASE 1: READ THE PLAN
    ↓
PHASE 2: READ WHAT'S BUILT
    ↓
PHASE 3: CHECK API CORRECTNESS
    ↓
PHASE 4: FIND GAPS (plan vs built)
    ↓
PHASE 5: BUILD THE NEXT GAP
    ↓
PHASE 6: VERIFY + TEST + LOOP OR DONE
    ↓
 gaps > 0? → PHASE 1
 gaps = 0? → COMPLETE
```

---

### PHASE 1: Read the Plan

Read the relevant plan sections for the requested step. Extract:

1. **Build sequence** — ordered steps and their dependencies
2. **File manifest** — every file that should exist when this step is done
3. **Architecture** — what imports what, what depends on what
4. **Data types** — RewardResult, config dataclasses, STEP_QUALITIES mapping
5. **Algorithm spec** — pseudocode from PLAN.md for this step's component
6. **Test spec** — tests listed in "Verifiable Tests Per Deliverable" section of PLAN.md
7. **Risks** — relevant risks from the risk table and Exa findings

Hold this as your source of truth. The plan is the contract.

---

### PHASE 2: Read What's Built

Scan the actual codebase to see what exists RIGHT NOW:

1. `glob` for all `.py` files in `qgre/`, `tests/`, `examples/`
2. `git diff --name-only` to see what's changed recently
3. Read key files that should exist per the step's file manifest
4. Check if files are stubs (1-3 line comments) or have real implementation
5. `python -c "import qgre"` — does the package import?

Produce a status checklist:
```
[x] qgre/types.py — exists, RewardResult defined
[x] qgre/__init__.py — exists, exports RewardResult
[ ] qgre/segments.py — STUB (comment only, no implementation)
[ ] qgre/advantages.py — STUB (comment only, no implementation)
[ ] tests/test_segments.py — STUB (comment only, no tests)
...
```

---

### PHASE 3: Check API Correctness

For every file that has REAL implementation (not stubs), verify correctness:

**PyTorch / Training:**
- `torch.Tensor` operations use correct dtypes (float32 for advantages)
- No in-place operations on tensors that require grad
- `torch.no_grad()` where appropriate
- Loss reduction matches plan (token-level vs sequence-level)
- Gradient accumulation: `loss /= accumulation_steps` before backward

**QGRE-Specific:**
- `RewardResult` is imported from `qgre.types`, not redefined
- `STEP_QUALITIES` mapping matches SPECIAL-TOKENS-SUPERPOWER.md exactly
- Token IDs match Qwen3 verified values (THINK_START=151667, THINK_END=151668, etc.)
- `segment_completion()` uses token ID pattern matching, NOT decoded-text regex
- SPO value tracker: EMA update `V = V + lr * (r - V)`, NOT replacement
- GDPO normalization: per-step across batch, NOT per-sequence
- Phase gating: only active qualities contribute to step rewards

**NeMo RL Extracted:**
- Apache-2.0 headers preserved on all extracted files
- No imports from `nemo_rl.*` (all deps stripped)
- No imports from `ray`, `megatron`, `nemo_rl.distributed`
- `masked_mean` handles zero-mask edge case (no division by zero)

**Config:**
- `generation` section includes temperature, top_p, stop_token_ids
- `algorithm.mode` is either "spo" or "grpo"
- SPO config has `lr` and `n=1`; GRPO config has `n` and `filter_groups`

**Architecture boundaries:**
- `qgre/` modules import only from `qgre/` and standard libs (torch, numpy)
- `examples/` import from `qgre` package
- `tests/` import from `qgre` package
- No circular imports within `qgre/`

If you find violations, fix them immediately before proceeding to phase 4.

---

### PHASE 4: Find Gaps

Compare the plan's spec for this step against what exists. Produce a gap list:

```
GAP LIST (iteration N):
1. [MISSING] qgre/segments.py — segment_completion() not implemented
2. [INCOMPLETE] qgre/advantages.py — SPO warm-start logic missing
3. [DRIFT] examples/hypergraph/config.yaml — clip_ratio_high is 0.28, plan says 0.28 ✓
4. [API] qgre/nemo_extracted/loss_functions.py — still imports nemo_rl.algorithms.interfaces
5. [TEST] tests/test_segments.py — stub only, no test functions
```

Gap types:
- **MISSING** — file doesn't exist or is a stub
- **INCOMPLETE** — implementation exists but doesn't match plan spec
- **DRIFT** — values, structure, or behavior diverged from plan
- **API** — using wrong API, wrong imports, wrong patterns
- **TEST** — test from "Verifiable Tests Per Deliverable" not implemented
- **RISK** — known risk from Exa findings not mitigated

Rank gaps by dependency order (from PLAN.md build sequence).
The next gap to build is the FIRST one that unblocks other gaps.

---

### PHASE 5: Build the Next Gap

Pick the highest-priority gap. Build it:

1. **Read the plan section** for this specific component
2. **Read adjacent files** that this component depends on or is depended on by
3. **Write or edit the file** — match the plan exactly
4. **Run the step's tests:** `pytest tests/test_{module}.py -v`
5. **Run import check:** `python -c "from qgre.{module} import {class}"`

Rules:
- Build ONE gap per iteration, not all of them
- Match the plan's types, names, and algorithm exactly
- Use pseudocode from PLAN.md and SPECIAL-TOKENS-SUPERPOWER.md as implementation spec
- Token IDs must match the verified Qwen3 values in the plan
- Do NOT add features, refactors, or improvements not in the plan
- Do NOT add comments unless the logic is non-obvious
- If the plan specifies test cases, implement them in the corresponding test file
- Smoke test model is Qwen3-1.7B (NOT 8B). Config defaults: unsloth/Qwen3-1.7B-unsloth-bnb-4bit
- Do NOT copy patterns from training-dojo v1. This engine is a clean rewrite.

**MANDATORY: Exa search before ANY bug fix.**
When a test fails, an error occurs, or unexpected behavior is observed:
1. STOP — do not guess the fix
2. Search Exa for the specific error message, library version, or API in question
3. Read the search results to understand the root cause
4. ONLY THEN apply a fix based on evidence, not training data assumptions
5. Cite the source (GitHub issue #, docs URL, or paper) in a comment if non-obvious

This rule exists because training data is stale. Unsloth, vLLM, and PyTorch
APIs change faster than any model's knowledge cutoff. The cost of one Exa search
is seconds. The cost of a wrong fix is hours of debugging.

If building this gap reveals a plan deficiency, NOTE IT but don't fix the plan.
Report it in phase 6 so the user can decide.

---

### PHASE 6: Verify + Test + Loop or Done

After building the gap:

1. **Test:** `pytest tests/test_{module}.py -v` — must pass
2. **Import check:** `python -c "import qgre"` — must succeed
3. **Re-read the file** — verify it matches the plan
4. **Check imports** — no boundary violations introduced
5. **Re-run phase 4** mentally — how many gaps remain?

**If ANYTHING goes wrong — test failure, OOM, crash, unexpected behavior, or
even uncertainty about how an API works — search Exa IMMEDIATELY.**
Do NOT attempt a fix, workaround, or code change until you have searched.
Do NOT rely on training data knowledge of Unsloth, vLLM, PyTorch, Triton, or
any fast-moving library. Search Exa. Read the results. THEN act.

If gaps remain:
```
ITERATION N COMPLETE
Built: [what was built]
Tests: PASS/FAIL (N passed, M failed)
Remaining gaps: N
Next gap: [description]
Continuing...
```
→ Go back to PHASE 1.

If zero gaps remain for this step:
```
STEP COMPLETE: [step name]

Files created/modified:
  [list every file touched]

Plan compliance:
  Algorithm: matches PLAN.md pseudocode
  Token IDs: verified against SPECIAL-TOKENS-SUPERPOWER.md
  Architecture boundaries: CLEAN
  Tests: ALL PASSING

Remaining concerns:
  [any plan deficiencies noted during build]

Ready for: next step / integration test / PR
```

---

## Autonomous Continuation

When a step completes, DO NOT STOP AND WAIT for user input. Immediately
proceed to the next gap. The loop is designed to run unattended. The only
reasons to pause:

1. A test fails and Exa search doesn't resolve it — ask the user
2. A plan deficiency is found that requires a design decision — ask the user
3. ALL gaps are closed — report completion

For GPU tests: run them with `pytest --gpu` if the GPU is available
(`nvidia-smi` shows free memory). If GPU is busy, skip GPU tests and
continue with CPU-testable work.

## Current Work Queue (auto-updated, 2026-03-19)

FEATURE LOOP COMPLETE. 103 CPU tests + 9 GPU tests (3 smoke + 3 wiring + 3 Triton) = 112 total.
All plan items implemented. Phase 4 Triton kernel built. Dr.GRPO + DAPO modes added.
8×4096 GPU stress test passes on RTX 5080 16GB (8.2s, peak 8666 MB).

### Engine Status

```
=== COMPLETED (all committed to main) ===

Phase 1 — Core Engine:
  [x] 0a: GameState serializer → checkpoint.py (11 tests)
  [x] 0b: NeMo RL extraction → loss_functions.py, kl.py, logits.py (10 tests)
  [x] 0c+0d: Advantage estimator → segments.py + advantages.py (22 tests)
  [x] 0e: DataLoader → data.py (9 tests)
  [x] 0f: Checkpoint resume → checkpoint.py
  [x] 0g: LoRA verifier → lora_verify.py (7 tests)
  [x] 1+4+6: Trainer + config + logging → trainer.py, config.py, logging.py (12 tests)
  [x] 2: Generation backend → generation.py (GPU tests)
  [x] M1: LLDS loss (arXiv:2512.04220)
  [x] M2: AdamW8bit optimizer (bitsandbytes)
  [x] M3: Low-advantage filter for SPO
  [x] M4: seq-mean-token-sum-norm loss aggregation
  [x] M5: Region-specific KL (THINK=0.1, FORMAT=2.0, STEP=1.0) — fully wired
  [x] GameState engine-managed phase advancement
  [x] Configurable step_qualities, pluggable segmenters
  [x] Full train() loop with generate → score → step → checkpoint → log
  [x] PLAN.md Phase 4 reassessment (committed 71cf277)

Phase 1 — Reviews (3 rounds):
  [x] 13 bugs found and fixed in round 1
  [x] 2 critical alignment bugs fixed in round 2
  [x] 5 regression tests added
  [x] Stress test: 8×4096 tokens on RTX 5080 16GB passes

=== COMPLETED THIS SESSION (2026-03-19, uncommitted) ===

Phase 2 — Memory + Triton:
  [x] selective_log_softmax (TRL PR #2799) — 37,000× less memory per chunk
  [x] Triton fused lm_head→logprobs kernel — zero vocab-tensor allocation (BLOCK_V=128)
  [x] Dr.GRPO unbiased mode — no length/std bias (arXiv:2503.20783)
  [x] DAPO Dynamic Sampling — filter zero-variance groups
  [x] for_training() per micro-batch — activates Unsloth GC + gradient offloading
  [x] Adaptive micro_batch_size — 1 for seq≥2048

Phase 3 — torch.compile:
  [HOLD] Unsloth + torch.compile still fragile (issues #4181, #1790, #2702)

Plan gap fixes (found by reading every line of all 3 plan docs):
  [x] scheduler_state_dict saved/restored in checkpoint (PLAN line 474)
  [x] cuda_rng_state restored on resume (PLAN line 475)
  [x] GLOBAL_QUALITIES defined (SPECIAL-TOKENS line 104-106)
  [x] MLflow set_experiment/start_run in train() (PILLARS line 128)
  [x] Per-step reward metrics logged to MLflow (PLAN line 517-518)
  [x] Periodic vLLM recreation every 50 steps in train() (PLAN line 719)
  [x] LoRA verify_sync called after weight sync (PLAN line 484-487)
  [x] verify_active() implemented in lora_verify.py (PLAN line 485)
  [x] 3 missing plan-specified tests added (total: 101 CPU tests)

=== TECH SCAN FINDINGS (2026-03-19) ===

CRITICAL facts for VRAM math (quantized model):
  - lm_head is NOT quantized — stays bf16: 1536 × 151936 × 2 = 446MB
  - Embeddings also bf16: ~446MB
  - Model body (4-bit): ~850MB
  - Total model VRAM: ~1.7GB (not 850MB)
  - vLLM at 0.35: 5.6GB for KV cache
  - Remaining for activations: ~8.7GB

CRITICAL Qwen3 vocab constraint:
  - 151936 / 256 = 593.5 (NOT divisible)
  - 151936 / 128 = 1187 (divisible)
  - Any Triton kernel tiling vocab dimension MUST use block_size ≤ 128

DAPO improvements over GRPO (from Chinese forums):
  - Clip-Higher: asymmetric clipping (we already do this: 0.2/0.28)
  - Dynamic Sampling: filter all-0 or all-1 reward batches (consider implementing)
  - Token-Level Policy Loss: per-token not per-sequence (we already do this)
  - Overlong Reward Shaping: length-aware reward correction (not implemented)

Unsloth colocate:
  - Unsloth forces vllm_mode="colocate" since June 2025 — validates our approach
  - Standby mode (gpu_memory_utilization=0.95) available but fragile on consumer GPUs
  - Our fixed 0.35 split is safer for RTX 5080 16GB

Liger Kernel GRPO loss:
  - PR #672: complete fused GRPO loss in Triton — 46GB savings
  - Includes fused_selective_log_softmax (old_logp + ref_logp without vocab tensor)
  - BUT: verl #2656 showed logprobs can be POSITIVE (wrong) with fused kernels
  - MUST validate numerically against our current implementation
```

### Known constraints:
- GPU smoke tests must run individually — 16GB RTX 5080 can't load 3 models in sequence
- Primary GPU test: `pytest tests/test_smoke.py::test_three_steps_no_crash --gpu -v`
- Qwen3-1.7B for smoke tests, NOT 8B
- Never copy patterns from training-dojo v1
- Always search Exa before any bug fix
- `force_on_policy_ratio=True` disables ratio clipping (by design for on-policy, documented)
- Qwen3 vocab 151936 NOT divisible by 256 — Triton kernels must use block_size ≤ 128

## Important: What This Command Does NOT Do

- Does NOT modify plan docs. The plan is the contract. If you find issues,
  report them to the user — don't silently "fix" the plan.
- Does NOT commit to git. The user decides when to commit.
- Does NOT make architectural decisions. If the plan doesn't specify
  something, ask the user rather than inventing.
- Does NOT skip phases. Every iteration runs all 6 phases. Phase 3 (API check)
  catches drift that accumulates silently.
- Does NOT touch files outside the current step's scope. One step at a time.

## Completion Verification — MANDATORY every time

**You MUST execute ALL of these commands and checks before reporting done.**
**Do NOT skip any. Do NOT summarize from memory. RUN the commands. READ the output.**
**If you find yourself saying "already verified" — run it again anyway.**

### CHECK 1: Run all tests (EXECUTE, don't remember)
```bash
python -m pytest tests/ -q --tb=short
```
Report the EXACT output line showing passed/failed/skipped counts.

### CHECK 2: Zero stubs (EXECUTE)
```bash
grep -rn "TODO\|FIXME\|STUB\|placeholder\|HACK" qgre/*.py qgre/nemo_extracted/*.py
```
Must return zero results.

### CHECK 3: Import check (EXECUTE)
```bash
python -c "import qgre; print('OK')"
```

### CHECK 4: Spawn an Explore agent to audit plan docs vs code
Use the Agent tool with subagent_type=Explore to:
- Read docs/PLAN.md IN FULL
- Read docs/SPECIAL-TOKENS-SUPERPOWER.md IN FULL
- Read docs/PILLARS.md IN FULL
- For EACH feature/algorithm/config/test mentioned, grep the codebase to verify it exists
- Report ONLY items that are MISSING or WRONG — not items that are correct
- If it finds zero issues, that's the answer

This agent MUST actually read the files and search the code. It cannot rely on
previous context or memory. Fresh eyes every time.

### CHECK 5: Verify feature-loop.md Engine Status matches reality
- Run: `wc -l qgre/*.py qgre/nemo_extracted/*.py` — compare file list to Engine Status
- Run: `python -m pytest tests/ --collect-only -q | tail -5` — compare test count to claim
- Verify every "[x]" item in Engine Status has corresponding code (grep for key function names)

### CHECK 6: Verify the save/restore cycle is complete
```bash
# Every state the plan says to save must be both SAVED and RESTORED
grep -n "save_checkpoint\|scheduler_state\|cuda_rng\|advantage_estimator_state\|game_state\|rng_state" qgre/trainer.py qgre/checkpoint.py
```
For each state field: verify it appears in BOTH save() AND resume().

### CHECK 7: Report findings
After ALL 6 checks complete, report:
```
VERIFICATION RESULTS:
  Tests: [exact count] passed, [exact count] skipped
  Stubs: [count]
  Import: OK/FAIL
  Plan audit: [number of issues found, or "zero issues"]
  Engine Status: [accurate/inaccurate — list discrepancies]
  Save/restore: [complete/incomplete — list gaps]
```

If ANY check fails or finds issues → FIX THEM before reporting complete.
If ALL checks pass with zero issues → report:

```
FEATURE LOOP COMPLETE — zero gaps remaining (verified)
```

**The key difference: you must SHOW the output of each check, not just claim you ran it.**

$ARGUMENTS
