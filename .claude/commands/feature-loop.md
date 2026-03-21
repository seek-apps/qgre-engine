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
- `docs/PLAN-engine-improvements.md` — ACTIVE improvements plan (Items 0-10, CPA findings, research references)
- `docs/TECH-SCAN-2026-03-19.md` — live research findings per technology
- `docs/PLAN.md` — original engine build plan (COMPLETE — reference only)
- `docs/PILLARS.md` — six pillars decomposition (reference only)
- `docs/SPECIAL-TOKENS-SUPERPOWER.md` — VPRM spec (reference only)

The ACTIVE plan is `docs/PLAN-engine-improvements.md`. Read it FIRST every loop iteration.
If an item's spec is not in the improvements plan, STOP and tell the user.

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
  generation.py        — UnslothBackend: vLLM colocated generation (Step 2)
  lora_verify.py       — LoRA weight sync verification (Step 0g)
  fused_logprobs.py    — Chunked logprobs without full logits materialization
  triton_logprobs.py   — Triton fused lm_head→logprobs kernel (BLOCK_V=128)
  nemo_extracted/
    __init__.py        — NeMo RL attribution
    loss_functions.py  — ClippedPGLossFn (Step 0b)
    kl.py              — KL calculation + kl_cov (Step 0b)
    logits.py          — Log prob computation + selective_log_softmax (Step 0b)
    llds.py            — LLDS loss (arXiv:2512.04220)
    LICENSE            — Apache-2.0 attribution
examples/
  hypergraph/
    config.yaml        — full QGRE config (model, data, generation, algorithm, training, logging)
    reward_fn.py       — reward function returning RewardResult with per-quality scores
  math/
    config.yaml        — minimal SPO config
    reward_fn.py       — math reward function
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
  test_triton_logprobs.py — Triton kernel tests
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

**Active plan:** docs/PLAN-engine-improvements.md (engine-notes scan + tech scan + CPA pressure test)
**Plan doc for original build:** docs/PLAN.md (original engine build — COMPLETE)

Engine core is COMPLETE and VERIFIED (121 tests, 20 source files, 2,872 lines).
Now implementing pre-training-run improvements from PLAN-engine-improvements.md.

### Engine Improvements — ACTIVE WORK

```
SHIP NOW (clean spots — no dependencies, ship immediately):
  [x] Item 0:  vLLM logprob passthrough — BLOCKED by Unsloth
               unsloth_zoo/vllm_utils.py:1759 hardcodes max_logprobs=0
               Path A dead. Path B (forward pass in trainer) is the way.
               Finding: DO NOT touch vLLM init — 5 hours of tuning.
  [x] Item 2:  Configurable reference_policy_kl_type — DONE
               config.py: added reference_policy_kl_type field (default "k3")
               trainer.py:104: replaced hardcoded "k3" with alg.reference_policy_kl_type
               Verified: config load, YAML override, default value. 123 tests pass.
  [x] Item 8:  GDPO NaN guard — DONE
               advantages.py: nanmean for step reward extraction + nan_to_num before GDPO norm
               Warning emitted when NaN detected. 123 tests pass.
  [x] Item 10: Monitor output lengths — DONE
               trainer.py: completion_length/mean,max,min added to step metrics
               123 tests pass.
  [x] Item 4:  CompletionLogger context manager — DONE
               logging.py: added __del__, __enter__, __exit__. 123 tests pass.
  [x] Item 5:  Remove HIF_V2_STEP_QUALITIES + MATH_STEP_QUALITIES — DONE
               segments.py: removed both. KEPT: HYPERGRAPH_V1, STEP_QUALITIES alias,
               segment_completion alias. 123 tests pass.

DONE (this session):
  [x] Item 3:  HIF JSON region segmenter — DONE
               segments.py: hif_json_segmenter (decode-and-regex), make_hif_json_segmenter(tokenizer)
               trainer.py: "hif_json" registered in segmenter resolver with tokenizer binding
               Tested: mock tokenizer, think blocks, all 5 JSON sections, empty input. 122 tests pass.
  [x] Item 6:  Stagnation detection — DONE
               types.py: StagnationStatus enum, check_stagnation() method, 4 new fields on GameState
               trainer.py: stagnation metric logged (0=normal, 1=stagnating, 2=stuck)
               config.py: stagnation_timeout, plateau_window, plateau_threshold
               checkpoint.py: new fields serialized/restored
               check_phase_advance() now resets steps_at_phase_start. 122 tests pass.
  [x] Item 9:  Dead code cleanup — DONE
               trainer.py: removed compute_loss() (never called, step() uses loss_fn directly)
               tests/test_trainer.py: removed test_on_policy_mode (tested removed method)
               122 tests pass (was 123 — removed 1 dead test).
  [x] ENV:    Pin mlflow<3.0 in pyproject.toml — DONE
  [ ] ENV:    Add PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to launch scripts

DEFERRED (no evidence of need):
  [DEFER] Item 1:  Store generation logprobs — DEFERRED
               Drift ~10-15%, dropped completions handle it. No collapse.
               Revisit if collapse observed or drift exceeds drop threshold.
  [DEFER] Item 7:  Scaffold fading hook — no evidence of need yet
```

### What's Done (original engine build — all committed to main)

```
Core Engine (Steps 0a-0g, 1-8): ALL COMPLETE
  [x] GameState, NeMo RL extraction, advantage estimator, dataloader, checkpoint,
      LoRA verifier, trainer, config, logging, generation backend
  [x] 121 tests (112 CPU pass + 9 GPU skip)
  [x] 19 bugs fixed across 11 adversarial audit rounds
  [x] Real e2e validation: 10 steps on Qwen3-1.7B, VRAM 6.44GB stable, no OOM/NaN
  [x] All research-backed optimizations: LLDS, AdamW8bit, SPO filter, selective_log_softmax,
      Triton fused logprobs, Dr.GRPO mode, DAPO dynamic sampling, region-specific KL,
      KL-adaptive SPO lr, prioritized sampling, λ-return traces, length control
  [x] CLI: python -m qgre train --config --reward --segmenter
```

### Critical implementation notes (from code review 2026-03-19)

```
KL activation requires THREE gates (all must be true):
  1. loss_mode: "kl_cov"     (config.py:67, default "pg" = KL off)
  2. kl_cov_ratio: > 0       (config.py:68, default 0.0)
  3. reference_logprobs != curr_logprobs  (trainer.py:359/373)
  Item 1 fixes gate 3. Gates 1+2 must be set in training run YAML.

LLDS activation requires TWO gates:
  1. llds_coef: > 0          (config.py:69, default 0.05 = ON)
  2. old_log_prob != log_prob (trainer.py:404)
  Item 1 fixes gate 2. Gate 1 already satisfied. LLDS comes alive for free.

trainer.py:359 mb_old_lp = mb_lp.detach() — used in FOUR places:
  Line 370: prev_logprobs (ratio computation)
  Line 373: reference_logprobs (KL penalty)
  Line 404: old_log_prob (LLDS gate)
  All four must use stored generation logprobs.

generation.py:111-117 SamplingParams has NO logprobs= param.
  vLLM output.outputs[0].logprobs is None because we never request it.
  Adding logprobs=1 is one line at line 116.

Segmenter type: Callable[[list[int]], list[str]] — no tokenizer arg.
  HIF segmenter needs tokenizer for decode. Use functools.partial at registration.

CompletionLogger has close(), __del__, __enter__, __exit__ (logging.py:90-103). DONE.

KEEP these aliases (used by 15+ test files):
  segment_completion, STEP_QUALITIES, HYPERGRAPH_V1_STEP_QUALITIES
```

### Key Facts (from tech scan)

```
VRAM math (quantized Qwen3-1.7B):
  - lm_head NOT quantized — stays bf16: 446MB
  - Embeddings also bf16: ~446MB
  - Model body (4-bit): ~850MB
  - Total model VRAM: ~1.7GB
  - vLLM at 0.35: 5.6GB for KV cache
  - Remaining for activations: ~8.7GB

Qwen3 tokenizer:
  - PAD=151669 (<|PAD_TOKEN|>) — correct, NOT <|endoftext|>
  - EOS=151645 (<|im_end|>)
  - Stop tokens: [151643, 151645]
  - Vocab: 151936 (divisible by 128, NOT 256)

SPO is the PRIMARY training algorithm (NOT GRPO):
  - n=1, persistent EMA value tracker V(x) += lr * (r - V(x))
  - GRPO mode exists but SPO is default
  - All tests must use mode="spo" with partial rewards
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

---

## After Every Work Session — MANDATORY

This section runs at the END of every session, not just when the loop "completes."
Even if you only built one gap, even if you got interrupted, DO THIS before stopping.

### 1. Compare what you built against the plan

For every file you touched this session:
- Re-read the relevant section of PLAN.md, SPECIAL-TOKENS-SUPERPOWER.md, or PILLARS.md
- Verify your implementation matches the plan spec (not just "it works" — does it match?)
- If you find drift, fix it now or flag it explicitly

### 2. Update the REMAINING list in this file

After each work item completes:
- Move it from REMAINING to COMPLETED with a one-line summary of what was verified
- Add any NEW items discovered during the work (bugs found, follow-ups needed, etc.)
- The REMAINING list must always reflect the TRUE current state — never stale

### 3. Update the feature-loop Engine Status if you changed code

If you created, modified, or deleted any file in `qgre/`, `tests/`, or `examples/`:
- Update the "What's Done" section to reflect the change
- Update test counts if tests were added/removed
- Update file manifest if files were added/removed

### 4. Never report "done" without running verification

Even for small changes — run `pytest tests/ -q`, check import, grep for stubs.
"I already verified" is not acceptable. Run it again. Show the output.

### 5. This file is the SINGLE SOURCE OF TRUTH

- Do NOT put active work items in memory files — they go here
- Do NOT put next-session tasks in memory — they go in REMAINING above
- Memory files should only point here: "see feature-loop for active work"
- Every new session starts by reading THIS file — make sure it's accurate

---

## HARD RULE: Never trust your own status

**Every time this loop fires — even if you reported "complete" last time — you MUST:**

1. **Read the ACTIVE plan** (docs/PLAN-engine-improvements.md) fresh. Do not rely on memory.
2. **Read the actual code** for every item marked [x]. Verify the implementation matches the plan spec.
   - Read the file. Check the logic. Confirm it does what the plan says.
   - "I already verified" is NOT acceptable. Verify AGAIN.
3. **Run tests.** Show the output. Every time.
4. **If you find drift between code and plan** — fix it or flag it. Do not skip.
5. **Never cancel the cron loop** without explicit user permission.
6. **Never report "FEATURE LOOP COMPLETE"** without having read every modified file
   against its plan section in THIS iteration. Not a previous one. THIS one.

The purpose of the loop is to catch drift that accumulates silently. If you skip
verification because "nothing changed since last time," you defeat the purpose.

$ARGUMENTS
