# QGRE Engine Improvements Plan

Source: engine-notes.md scan from hypergraph-scan-v2 run planning.
Tech scan: TECH-SCAN-2026-03-19.md (live Exa research, English + Chinese sources)
Pressure test: CPA 7-block analysis (demand/constraint field dynamics)
Exa verification: vLLM logprobs API, RL-Struct, structured output RL papers, verbosity monitoring
Date: 2026-03-19

---

## Implementation Order

| # | Item | Effort | Impact | Status |
|---|------|--------|--------|--------|
| 0 | Verify vLLM logprob passthrough in fast_generate | Tiny | Blocker | DONE (blocked) |
| 1 | Store generation logprobs (THR/LLDS unlock) | Small–Medium | Low | DEFERRED |
| 2 | Configurable `reference_policy_kl_type` | Small | High | DONE |
| 3 | HIF JSON region segmenter (decode-and-regex) | Small–Medium | High | DONE |
| 4 | Fix CompletionLogger file handle leak | Small | Medium | DONE |
| 5 | Decouple HIF_V2_STEP_QUALITIES from engine | Small | Medium | DONE |
| 6 | Stagnation detection in GameState | Medium | Medium | DONE |
| 7 | Scaffold fading hook (informed by Scaf-GRPO) | Medium | Medium | DEFERRED |
| 8 | GDPO NaN guard | Small | High | DONE |
| 9 | Dead code cleanup | Small | Low | DONE |
| 10 | Monitor output lengths (MLflow metric) | Small | High | DONE |

**Engine identity:** This is an SPO-first engine. SPO (persistent EMA baseline, n=1) is the
default algorithm. GRPO mode exists as a fallback but may not be used. All plan items,
research references, and design decisions should be evaluated through SPO n=1 lens, not
through the n=8 group-comparison paradigm that dominates the RL literature.

**Implementation cross-check (2026-03-19):**
- `loss_mode: "pg"` is the default (config.py:67) — KL is deliberately OFF for on-policy.
  The comment on lines 64-66 says this was intentional. Item 1 changes this assumption.
- **Three-layer KL gate:** KL requires ALL of: (1) `loss_mode: "kl_cov"`, (2) `kl_cov_ratio > 0`,
  (3) `reference_logprobs != curr_logprobs`. Item 1 fixes gate 3. Gates 1+2 must be set in
  training run YAML. Neither Item 1 nor Item 2 enables KL by itself.
- **LLDS is different:** `llds_coef: 0.05` (default ON). Only blocked by gate 3 (`old_log_prob == log_prob`).
  Item 1 alone resurrects LLDS with no config change needed.
- trainer.py:359 `mb_old_lp = mb_lp.detach()` is used in FOUR places: line 370 (prev_logprobs),
  line 373 (reference_logprobs), line 404 (LLDS old_log_prob). All four must use stored logprobs.
- `CompletionLogger.close()` already exists (logging.py:90-93) but `__del__`/`__enter__`/`__exit__` are missing. Item 4 is still needed.
- `HYPERGRAPH_V1_STEP_QUALITIES` is used by tests and scripts — do NOT remove. Item 5 targets only `HIF_V2_STEP_QUALITIES` and `MATH_STEP_QUALITIES` (unused outside segments.py).
- `segment_completion` alias (segments.py:91) is used in 15+ test files — keep it.
- `STEP_QUALITIES` alias (segments.py:122) is used in tests — keep it as backward compat.
- `loss_type: "grpo"` / `"dr_grpo"` in config refers to loss aggregation mode (normalization), not the SPO/GRPO algorithm selection. This naming is confusing but not wrong.

**CPA finding:** Item 1 was the plan's #1 priority AND the most constrained feature.
The vLLM logprob MVP (Item 0) resolves this — if fast_generate passes through logprobs,
Item 1 drops from Medium-large to Small effort with zero VRAM cost. Item 0 is a
5-line test that determines the entire implementation path.

**CPA finding:** Item 7 (scaffold fading) genuinely doesn't survive current constraints
(I=0.45 > demand even with latent correction). Marked DEFERRED. Revisit after first
training run provides evidence of stagnation that scaffold would address.

**CPA finding:** Item 10 (monitor output lengths) surfaced as a real spot — verbosity
drift from loss normalization choices is documented in the literature and must be observable.

---

## 0. Verify vLLM Logprob Passthrough in fast_generate — DONE (blocked)

**Result:** `unsloth_zoo/vllm_utils.py:1759` hardcodes `max_logprobs=0`. vLLM rejects `logprobs=1` with `VLLMValidationError: Requested sample logprobs of 1, which is greater than max allowed: 0`. Path A (free vLLM logprobs) is dead without modifying Unsloth's vLLM init — which we will NOT do (5 hours of tuning to get vLLM configured right).

### Verification test (5 lines)

```python
# In generation.py or a test script:
sampling_params = SamplingParams(
    temperature=1.0, max_tokens=100,
    logprobs=1,  # Request logprob of sampled token
)
outputs = model.fast_generate(prompts, sampling_params=sampling_params)
# Check: does output.outputs[0].logprobs exist and contain per-token data?
print(type(outputs[0].outputs[0].logprobs))  # Should be list[dict[int, Logprob]]
print(len(outputs[0].outputs[0].logprobs))   # Should == number of generated tokens
```

### Outcomes

**If logprobs are returned:**
- Item 1 becomes SMALL effort — extract logprobs from vLLM output, convert to tensor, store on GenerationOutput
- Zero VRAM cost, zero extra forward pass, zero mode transition risk
- The CPA constraint field on F1 drops from v=0.220 to v≈0.030 → Net flips to +0.053 (clean survival)

**If logprobs are NOT returned (Unsloth strips them):**
- Fall back to Option B: forward pass at start of `step()` before training mode
- Must validate VRAM budget: one inference forward pass on completion tokens (~100-200MB for 4096 tokens)
- Still feasible on 16GB but adds ~2-3s per step (from 20s to 22-23s)

### Known vLLM logprob nuances (from Exa search)

- `logprobs=1` returns top-1 logprob. With temperature=1.0, the sampled token may not be the top-1 token, meaning its logprob could be missing (vLLM issues #2613, #3779).
- **Fix:** Use `logprobs=-1` (returns ALL logprobs — but O(vocab) memory per token, ~1.2MB/token). Too expensive.
- **Better fix:** vLLM PR #22387 (merged) adds `logprobs_mode="processed_logprobs"` returning logprobs after temperature/sampling. PR #35961 adds `score_mode` for GPU-side per-token logprob extraction.
- **For our case (SPO n=1):** We need the logprob of THE sampled token at each position. Even `logprobs=1` should include it because vLLM returns logprobs for the sampled token + top-k others. Verify empirically.
- **Performance:** vLLM issue #14300 shows logprobs retrieval scales O(n) with k. At k=1, overhead is negligible.

---

## 1. Store Generation Logprobs — DEFERRED

**Status:** DEFERRED. Model learns fast without it. Drift is ~10-15% but dropped completions handle it. KL is OFF by design (`loss_mode: "pg"`). LLDS is a collapse safety net for a problem we don't have. Unsloth hardcodes `max_logprobs=0` (Path A dead), Path B adds ~4s/step for zero immediate benefit.

**Revisit when:** (1) training collapse observed, (2) drift exceeds the drop threshold and hurts learning, (3) multi-epoch training needed.

**Files:** generation.py, trainer.py, types.py (GenerationOutput)
**Original rationale:** Currently `reference_logprobs` in the loss function receives `mb_lp.detach()` (trainer.py:359,373). Since this is the same forward pass, `reference_logprobs == curr_logprobs` and all KL = 0. Region KL multipliers (THINK=0.1, FORMAT=2.0, STEP=1.0) scale a zero. LLDS has the same problem — `old_logprobs == curr_logprobs` means the trajectory gate never fires.

**Unlocks:**
- Real KL regularization (prevents policy drift from base model)
- LLDS auxiliary loss (prevents log-prob decay on correct completions — the "LLD Death Spiral" from arXiv:2512.04220, ICLR 2026 Workshop)
- Region-specific KL actually doing something (validated by Archer, ICLR 2026)
- Foundation for multi-epoch training if needed later

### Implementation Path A: vLLM Logprobs (preferred — depends on Item 0)

**a) Modify SamplingParams in `backend.generate()`:**
```python
sampling_params = SamplingParams(
    temperature=self.generation_config.temperature,
    top_p=self.generation_config.top_p,
    max_tokens=self.generation_config.max_tokens,
    stop_token_ids=self.generation_config.stop_token_ids,
    logprobs=1,  # NEW: request per-token logprobs from vLLM
)
```

**b) Extract logprobs from vLLM output:**
```python
# In generation.py, after vLLM returns:
batch_logprobs = []
for output in outputs:
    token_lps = []
    for lp_dict in output.outputs[0].logprobs:
        # lp_dict maps token_id → Logprob. Get the sampled token's logprob.
        sampled_id = output.outputs[0].token_ids[len(token_lps)]
        if sampled_id in lp_dict:
            token_lps.append(lp_dict[sampled_id].logprob)
        else:
            # Fallback: token not in top-k. Use the rank-1 logprob as approximation.
            token_lps.append(next(iter(lp_dict.values())).logprob)
    batch_logprobs.append(token_lps)
```

**c) Store on GenerationOutput:**
```python
@dataclass
class GenerationOutput:
    token_ids: list[list[int]]
    texts: list[str]
    logprobs: list[list[float]] | None = None  # NEW: per-token logprobs from generation
```

**d) Pass into trainer.py step():**
- Replace `mb_old_lp = mb_lp.detach()` (line 359) with stored generation logprobs
- Pad and align to match micro-batch slicing
- Four touch points that currently use `mb_old_lp`:
  - Line 370: `prev_logprobs=mb_old_lp` → use stored logprobs (ratio computation)
  - Line 373: `reference_logprobs=mb_old_lp` → use stored logprobs (KL penalty)
  - Line 404: `old_log_prob=mb_old_lp` → use stored logprobs (LLDS gate)
  - All three must use the SAME stored tensor (generation-time policy)

**e) What this unlocks vs what needs config changes:**
- **LLDS:** Alive immediately. `llds_coef: 0.05` is already the default (config.py:69). No config change needed.
- **KL regularization:** Requires ADDITIONAL config changes in training YAML:
  - `loss_mode: "kl_cov"` (default is `"pg"` = no KL)
  - `kl_cov_ratio: 0.01` (or appropriate value — default is 0.0)
  - Without these, KL penalty remains zero even with stored logprobs.
- **Region-specific KL multipliers:** Already configured (THINK=0.1, FORMAT=2.0, STEP=1.0).
  They activate automatically once KL is non-zero (gates 1+2 in training YAML).

### Implementation Path B: Forward Pass Fallback (if Item 0 fails)

If Unsloth's `fast_generate` doesn't expose vLLM logprobs:

**a) Run inference forward pass after generation, before mode switch:**
- Build full input_ids (prompt + completion) for each sequence
- Run `model(input_ids)` in inference mode (no gradients)
- Extract logprobs via `logprobs_from_logits()` (already memory-efficient)
- Store on GenerationOutput

**b) VRAM budget for Path B:**
- One forward pass on [batch, max_seq] — uses vLLM's already-allocated KV cache area
- Peak: ~200MB for batch=8, seq=4096 at bf16 (logits are computed per-chunk via selective_log_softmax)
- At gpu_memory_utilization=0.35, we have ~5.6GB for vLLM + ~10GB for training. Should fit.
- **Must test empirically (Item 0 covers this)**

**c) Speed impact for Path B:**
- Extra forward pass adds ~2-3s per step (inference only, no backward)
- 20s → 22-23s per step. Acceptable — SPO n=1 is already ~8x faster per step than n=8 frameworks

### Memory budget (both paths)

- Stored logprobs: one float32 tensor per batch: [batch_size, max_seq_len] ≈ 8 × 4096 × 4B = 128KB
- Negligible compared to model activations

### Research context

- Standard pattern across 16 RL libraries surveyed by HuggingFace (March 2026)
- vLLM `SamplingParams(logprobs=1)` returns per-token logprobs natively (vLLM docs, PRs #22387, #35961, #21792)
- "A Comedy of Estimators" (Shah, Bengio et al., arXiv:2512.21852): once logprobs are stored, k3 bias activates. Item 2 must ship alongside.
- "On the Design of KL-Regularized PG Algorithms" (Zhang et al., ICLR 2026): RPG view — on-policy mismatch is zero only if logprobs captured at generation time.
- Re-tokenization bug (TRL #5224, March 2026): BPE decode→re-tokenize produces different IDs. Our engine works on raw token IDs — safe.

### Forward-KL vs Reverse-KL Per Region (DEFERRED)

Research finding (Chen et al., NYU/EPFL, Oct 2025): KL direction matters less than coefficient strength. The per-region multipliers (THINK=0.1, FORMAT=2.0) already achieve "loose on thinking, tight on structure." Skip directional KL — simpler, same practical outcome. Revisit only if we observe mode collapse in thinking regions.

### Per-Region KL Type (DEFERRED)

Once generation logprobs land and KL is non-zero, evaluate whether different regions need different estimators. Low priority — region multipliers handle most differentiation.

---

## 2. Configurable `reference_policy_kl_type`

**Files:** config.py, trainer.py
**Why:** Currently hardcoded as `"k3"` at trainer.py:104. Three ICLR 2026 papers prove k3 has biased gradients when `pi_theta << pi_ref`. This bias activates the moment Item 1 lands and KL becomes non-zero. Must ship alongside or before Item 1.

### Research basis

- **"A Comedy of Estimators" (Shah, Bengio et al.):** k3 assigns unbounded gradient weights when `pi_theta << pi_ref`.
- **"Rethinking KL Regularization" (Liu et al.):** k1-in-reward is the principled loss for reverse KL. Under on-policy, k2-as-loss is gradient-equivalent.
- **"On the Design of KL-Regularized PG Algorithms" (Zhang et al.):** k3 = unnormalized KL. Unified framework identifies IS mismatch in policy gradient methods.
- **Chinese confirmation (CSDN):** Simulation shows k3 lowest variance but biased; k1 unbiased but high variance.

### Implementation

**a) Add to AlgorithmConfig (config.py):**
```python
reference_policy_kl_type: str = "k3"  # "k1" (linear/unbiased), "k2" (squared), "k3" (exponential)
```

**b) Wire through trainer.py `__init__`:**
```python
"reference_policy_kl_type": alg.reference_policy_kl_type,
```

Replace the hardcoded `"k3"` at line 104 with `alg.reference_policy_kl_type`.

**c) No changes to kl.py** — k1/k2/k3 all already implemented.

### Notes

- For on-policy training (current mode), the difference between estimators is minimal since `pi_theta ≈ pi_ref`
- Once Item 1 lands: default to `"k1"` for new training runs, keep `"k3"` as default for backward compat
- If we ever move to multi-epoch: use the DeepSeek-V3.2 corrected k3 (multiply by IS ratio `pi_theta/pi_old`) — would need a new `"k3_corrected"` option in kl.py

---

## 3. HIF JSON Region Segmenter (Decode-and-Regex)

**Files:** qgre/segments.py, trainer.py (registration)
**Why:** The upcoming training run produces HIF JSON output, not XML steps. Need a segmenter that maps token positions to structural regions for VPRM credit assignment and THR region-specific KL.

**CPA finding:** This item sat at near-equilibrium (Net = -0.030) with the original token-level pattern matching approach (I=0.5). Simplifying to decode-and-regex drops I to ~0.3, flipping Net to +0.012 (survival). The RL-Struct paper (arXiv:2512.00319) and GRAPH-GRPO-LEX (Nov 2025) both use programmatic segmentation on structured output — none use token-level matching for JSON.

### Region Map

| Region | JSON path | Purpose | KL multiplier |
|--------|-----------|---------|---------------|
| THINK | `<think>...</think>` | Reasoning block (before JSON) | 0.1x (explore) |
| HEADER | `network-type`, `metadata` | Document-level fields | 2.0x (lock) |
| NODES | `nodes` array | Entity declarations | 1.0x |
| EDGES | `edges` array | Hyperedge definitions | 1.0x |
| INCIDENCES | `incidences` array | Node-edge membership | 1.0x |
| SCAN | `scan-results`, `hamiltonian-score` | Analysis output | 1.0x |
| CONTENT | fallback | Unrecognized tokens | 1.0x |

KL multiplier design validated by Archer (ICLR 2026): weaker KL on reasoning tokens, stronger on structural tokens.

### Implementation Approach: Decode-and-Regex

**NOT token-level pattern matching.** JSON keys are stable strings — regex on decoded text is standard practice and dramatically simpler.

```python
def hif_json_segmenter(token_ids: list[int], tokenizer=None) -> list[str]:
    """Segment HIF JSON completions via decoded text + regex.

    1. Decode token IDs to text
    2. Use regex to find JSON section boundaries on the text
    3. Map character offsets back to token positions via tokenizer offset mapping
    4. Unrecognized tokens → CONTENT fallback
    """
```

**Step 1: Decode**
- Decode full token_ids to text (skip_special_tokens=False to preserve `<think>` tags)

**Step 2: Regex section boundaries**
- `<think>` ... `</think>` → THINK
- `"network-type"` or `"metadata"` → HEADER (until next top-level key)
- `"nodes"\s*:` → NODES section
- `"edges"\s*:` → EDGES section
- `"incidences"\s*:` → INCIDENCES section
- `"scan-results"` or `"hamiltonian-score"` → SCAN section
- Everything else → CONTENT

**Step 3: Char-to-token mapping**
- Use tokenizer's `encode()` with `return_offsets_mapping=True` to get (start, end) char offsets per token
- For each token, check which regex-identified section its char range falls in
- Assign region label accordingly

**Step 4: Fallback**
- Partial/malformed JSON: everything after last recognized key → CONTENT
- If decode fails entirely: fall back to uniform_segmenter behavior (all STEP_1)

### Phantom prerequisite: HIF output format spec

The segmenter parses specific JSON keys. Those keys must match what the model actually produces. The key names (`nodes`, `edges`, `incidences`, `scan-results`, `network-type`, `metadata`, `hamiltonian-score`) come from the HIF schema defined in the training run config — NOT baked into the engine.

**Resolution:** The segmenter accepts a `keys` config parameter mapping section names to regex patterns. Default matches HIF v2 schema. Training runs can override.

```python
DEFAULT_HIF_KEYS = {
    "HEADER": r'"(?:network-type|metadata)"\s*:',
    "NODES": r'"nodes"\s*:',
    "EDGES": r'"edges"\s*:',
    "INCIDENCES": r'"incidences"\s*:',
    "SCAN": r'"(?:scan-results|hamiltonian-score)"\s*:',
}
```

### Registration

Add `"hif_json"` to the segmenter resolver in trainer.py:84-91:
```python
elif alg.segmenter == "hif_json":
    from qgre.segments import hif_json_segmenter
    segmenter = hif_json_segmenter
```

### Note on tokenizer dependency

The decode-and-regex approach requires the tokenizer for decoding and offset mapping. The current `Segmenter` type signature is `Callable[[list[int]], list[str]]` — no tokenizer argument. Two options:
- **Option A:** Change signature to `Callable[[list[int], Any], list[str]]` (breaking change for existing segmenters)
- **Option B:** Use `functools.partial` to bind the tokenizer at registration time
- **Recommendation: Option B.** No signature change. Bind tokenizer when resolving the segmenter in trainer.py.

---

## 4. Fix CompletionLogger File Handle Leak

**Files:** qgre/logging.py
**Why:** `CompletionLogger` opens file handles per-step (logging.py:76) and has a `close()` method (logging.py:90-93), but no `__del__`, `__enter__`, or `__exit__`. If training crashes, file handles leak. The trainer calls `close()` at the end of `train()` (trainer.py:683) but not on exception paths.

**Already exists:** `close()` method (logging.py:90-93). The trainer calls it at end of training.
**Missing:** `__del__` for crash cleanup, context manager protocol for safe usage.

### Implementation

Add to CompletionLogger:
```python
def __del__(self):
    self.close()

def __enter__(self):
    return self

def __exit__(self, *exc):
    self.close()
    return False
```

Update QGRETrainer `train()` to wrap the training loop in try/finally calling `self.completion_logger.close()`.

---

## 5. Decouple Domain-Specific Step Qualities from Engine

**Files:** qgre/segments.py
**Why:** `HIF_V2_STEP_QUALITIES` and `MATH_STEP_QUALITIES` are domain-specific quality mappings baked into the engine. Step qualities belong in the training run's config YAML, not the engine source.

### Implementation

- Delete `HIF_V2_STEP_QUALITIES` from segments.py (unused outside segments.py itself)
- Delete `MATH_STEP_QUALITIES` from segments.py (unused outside segments.py itself)
- **KEEP** `HYPERGRAPH_V1_STEP_QUALITIES` — used by 8+ test files, scripts/run_e2e_test.py, scripts/run_long_training.py, and examples/. This is the V1 domain preset used by test infrastructure.
- **KEEP** `STEP_QUALITIES` alias (segments.py:122) — used by test files as shorthand.
- **KEEP** `segment_completion` alias (segments.py:91) — used by 15+ test references.
- Quality mappings are already configurable via `step_qualities` in YAML config — that's the canonical location for real training runs.

---

## 6. Stagnation Detection in GameState

**Files:** qgre/types.py, trainer.py, config.py
**Why:** `check_phase_advance()` only checks mastery >= threshold. No timeout, no plateau detection. A training run can silently stall on a phase forever.

### Research basis

- **Scaf-GRPO (arXiv:2510.19807):** Introduces "guidance exemption period" to distinguish true-hard vs pseudo-hard problems before intervening. Our timeout should be long enough to serve as this exemption period.
- **ACTOR-CURATOR (ICLR 2026 Workshop):** Uses policy improvement as the optimization signal. Our |advantage| as a proxy aligns with this.
- **SEC (Self-Evolving Curriculum):** Multi-armed bandit treats each difficulty level as an arm. Uses absolute advantage as learning gain proxy. Close to our mastery-based approach.

### Detection Signals

1. **Timeout:** `steps_in_current_phase > N` (configurable, default 200)
2. **Plateau:** Mastery improvement < 0.02 over last 50 steps (slope of mastery window ≈ 0)
3. **Entropy collapse** (future): Policy distribution entropy drops below threshold — requires logits tracking, defer to after Item 1

### Implementation

**a) GameState additions (types.py):**
```python
class StagnationStatus(Enum):
    NORMAL = "normal"
    STAGNATING = "stagnating"    # plateau detected
    STUCK = "stuck"              # timeout exceeded

# New fields on GameState
steps_at_phase_start: int = 0
stagnation_timeout: int = 200
plateau_window: int = 50
plateau_threshold: float = 0.02

def check_stagnation(self) -> StagnationStatus:
    steps_in_phase = self.step_count - self.steps_at_phase_start
    if steps_in_phase >= self.stagnation_timeout:
        return StagnationStatus.STUCK

    mastery_values = list(self.step_mastery.get(self.phase, deque()))
    if len(mastery_values) >= self.plateau_window:
        recent = mastery_values[-self.plateau_window:]
        slope = (sum(recent[len(recent)//2:]) - sum(recent[:len(recent)//2])) / (len(recent) // 2)
        if abs(slope) < self.plateau_threshold:
            return StagnationStatus.STAGNATING

    return StagnationStatus.NORMAL
```

**b) Trainer response (trainer.py):**
- After `check_phase_advance()`, call `check_stagnation()`
- Log status to MLflow: `stagnation_status` metric
- On STAGNATING: log warning, optionally increase entropy bonus
- On STUCK: log error, emit MLflow alert tag

**c) Intervention logic lives in the trainer, not GameState.** GameState only detects — it doesn't act. This keeps the type clean and lets different training runs respond differently.

**d) Reset `steps_at_phase_start` on phase advance.** When `check_phase_advance()` returns True, set `steps_at_phase_start = step_count`.

### Config additions (config.py TrainingConfig):
```python
stagnation_timeout: int = 200
plateau_window: int = 50
plateau_threshold: float = 0.02
```

---

## 7. Scaffold Fading Hook — DEFERRED

**Status:** DEFERRED until first training run provides evidence of stagnation that scaffold would address.

**CPA finding:** This item genuinely doesn't survive current constraints (I=0.45, Net=-0.056 even with latent correction). Three reasons: (1) touches config, data, AND trainer (wide blast radius), (2) requires scaffold prompt templates that don't exist yet (phantom prerequisite), (3) no user has experienced the need — it's a latent demand with no behavioral evidence. Scaf-GRPO paper validates the concept but doesn't reduce our implementation cost.

**When to revisit:** After first training run, if stagnation detection (Item 6) fires and the team determines scaffold modification would have helped. At that point, demand evidence exists and the item re-enters the field with u(x) based on real training data.

**Files:** config.py, data.py, trainer.py
**Why (for future reference):** The engine needs a way to modify system prompts based on training phase. Early phases use verbose scaffolding (full instructions, examples). Later phases strip scaffolding so the model internalizes the behavior.

### Research basis

**Scaf-GRPO (arXiv:2510.19807, Feb 2026)** is our scaffold fading concept published as a paper. Key design insights to adopt:

1. **Guidance exemption period:** Don't inject hints immediately. First let the model attempt problems unassisted for N steps. This distinguishes "true-hard" problems (model genuinely can't solve) from "pseudo-hard" (model can solve with more training). Our stagnation detection (Item 6) serves this role.
2. **Hierarchical hints, not binary on/off:** Scaf-GRPO uses tiered hints — abstract concepts first, then concrete steps. Our scaffold_phases should support at least 3 tiers: full scaffold → partial hints → no scaffold.
3. **In-prompt scaffolding, not prefix forcing:** Hints go in the system prompt, not forced as output prefixes. This maintains policy consistency — the model still generates freely.

### Design: Config-Driven with Tiered Hints

```yaml
scaffold_phases:
  1: "prompts/full_scaffold.txt"       # verbose instructions + examples
  3: "prompts/partial_scaffold.txt"    # key reminders only
  5: "prompts/minimal_scaffold.txt"    # format hint only
  7: null                              # no scaffold — model internalizes
```

### Implementation

**a) Config (config.py):**
```python
scaffold_phases: dict[int, str | None] | None = None  # phase → prompt template path (null = no scaffold)
```

**b) Trainer:**
- On phase change, check if `scaffold_phases` has an entry for the new phase
- If yes and value is a path, load the template and update the dataloader's system prompt
- If yes and value is null, clear the system prompt scaffold
- Dataloader applies the active template during batch construction

**c) Dataloader (data.py):**
- Add `set_system_prompt(template: str | None)` method
- Chat template uses this as the system message when set

---

## 8. GDPO NaN Guard

**Files:** qgre/advantages.py
**Why:** Tech scan finding (ms-swift #8123): GDPO normalization propagates NaN if any reward function returns NaN. Our `build_batch_reward_tensors()` zero-fills missing keys, but if a reward_fn returns NaN (not None), it flows through `.mean()` and `.std()` in GDPO normalization, causing silent training failure (loss=0.0, grad_norm=NaN).

### Implementation

Add NaN guard in `compute_advantages()` before GDPO normalization (advantages.py, between step reward extraction and GDPO normalization):

```python
# NaN guard: replace NaN step rewards with 0.0 before normalization
for step_num in self._step_nums:
    if step_advs[step_num].isnan().any():
        import warnings
        nan_count = step_advs[step_num].isnan().sum().item()
        warnings.warn(
            f"Step {step_num}: {nan_count}/{len(step_advs[step_num])} advantages are NaN. "
            f"Check reward_fn for NaN returns. Replacing with 0.0."
        )
        step_advs[step_num] = torch.nan_to_num(step_advs[step_num], nan=0.0)
```

Also guard step reward extraction:
```python
# In the per-step reward computation loop
step_rews[step_num] = float(np.nanmean([...]))  # nanmean instead of mean
```

---

## 9. Dead Code Cleanup

**Files:** trainer.py, segments.py

- **`compute_loss()` (trainer.py:176-215):** Never called — `step()` calls `self.loss_fn()` directly via micro-batched loop. Remove.
- **`HIF_V2_STEP_QUALITIES` + `MATH_STEP_QUALITIES`:** Covered by Item 5.
- **KEEP** `STEP_QUALITIES` alias, `segment_completion` alias, `HYPERGRAPH_V1_STEP_QUALITIES` — all used by test infrastructure.

**CPA note:** This is a shortcut survivor (convention-driven, not user-driven). Ship last or skip for V2.

---

## 10. Monitor Output Lengths (MLflow Metric)

**Files:** trainer.py
**Why:** CPA surfaced this as a genuine spot (Net = +0.030). Research documents that removing length normalization from the loss (as our `dr_grpo` loss_type does) can introduce verbosity bias — output lengths grow without quality improvement. RL reward curves follow exponential saturation — if reward saturates but length keeps growing, that's verbosity drift.

Our GDPO per-step normalization partially mitigates this (each step normalized independently), but without tracking the metric, drift is invisible until it wastes tokens and slows generation. At ~20s/step with ~10-12s on generation, length growth directly impacts training speed.

### Implementation

Add to trainer.py `step()` metrics, after reward computation:

```python
# Track completion lengths for verbosity drift detection
comp_lengths = [len(c) for c in completions]
metrics["completion_length/mean"] = float(np.mean(comp_lengths))
metrics["completion_length/max"] = float(np.max(comp_lengths))
metrics["completion_length/min"] = float(np.min(comp_lengths))
```

Log to MLflow via existing `log_step_metrics()`.

### What to watch for

- **Monotonic length increase with flat reward:** Verbosity drift. Loss normalization bias or reward function not penalizing length.
- **Length collapse (decreasing):** Model learning to produce minimal output. May indicate reward function over-penalizing length or phase gate masking too aggressively.
- **Length variance collapse:** All completions converging to same length. May indicate loss of diversity.
- **Length growth correlating with generation time growth:** Direct training speed impact. At ~20s/step, if mean length doubles, generation time could double too.

---

## Environment Hardening (from tech scan)

Quick wins that don't need their own items:

1. **Pin MLflow < 3.0** in pyproject.toml until we test with MLflow 3.x. Breaking changes to `log_model()` API and artifact storage paths. We don't use `log_model()` yet, but avoids surprises on `pip install --upgrade`.

2. **Add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** to launch scripts / README. Reduces CUDA memory fragmentation — multiple community reports confirm it helps with Unsloth RL training VRAM growth.

3. **Monitor output lengths** — now promoted to Item 10 (not just a quick win — it's a real spot per CPA).

---

## Future Considerations (post-launch)

Items identified by tech scan but not blocking V2:

1. **GTPO (entropy-weighted token credit).** Composes orthogonally with VPRMs. ~5% slower, ~50-100MB extra. Implement after core items are stable. arXiv:2508.04349.

2. **F-GRPO (focal-loss difficulty scaling).** Down-weights updates on high-success prompts. Complementary to our SPO prioritized sampling. Lightweight to add. arXiv:2602.06717.

3. **CoRPO (correctness-biased baseline).** Clips minimum baseline to correctness threshold. Potentially applicable to SPO baseline if we see advantage assigned to incorrect solutions. arXiv:2511.04439.

4. **Liger-Kernel fused PG loss.** 10x faster, 46GB less peak memory. Requires adapting to our custom loss. github.com/linkedin/Liger-Kernel/pull/672.

5. **torchforge (Meta).** PyTorch-native RL library with policy gradient losses and vLLM integration. Future alternative if we need multi-GPU. pytorch.org/blog/introducing-torchforge.

6. **λ-return eligibility traces.** Already implemented (lambda_return in loss_functions.py:126-130, default 0.0). Enable and test once training is stable. Validated by ICLR 2026 paper.

---

## Research References

Full details in [TECH-SCAN-2026-03-19.md](TECH-SCAN-2026-03-19.md).

**KL estimators:**
- "A Comedy of Estimators" — Shah, Bengio et al., arXiv:2512.21852
- "Rethinking KL Regularization" — Liu et al., ICLR 2026, openreview.net/forum?id=keCnsHtION
- "On the Design of KL-Regularized PG Algorithms" — Zhang et al., ICLR 2026, arXiv:2505.17508

**LLDS:**
- "On Policy Optimization Collapse: The Lazy Likelihood-Displacement" — Deng et al., arXiv:2512.04220, ICLR 2026 Workshop

**Region-specific KL:**
- "Archer: Dual-Token Constraints for RLVR" — ICLR 2026 under review
- "Ignore the KL Penalty" — Vassoyan et al., NAACL 2025

**Scaffold fading:**
- "Scaf-GRPO: Scaffolded Group Relative Policy Optimization" — arXiv:2510.19807

**GTPO:**
- "GTPO and GRPO-S: Token and Sequence-Level Reward Shaping" — Tan, Pan et al., arXiv:2508.04349

**Loss normalization / advantage variants:**
- "MAD: Robust reward scaling" — huggingface.co/blog/telcom/mad-grpo
- "F-PO: Focal difficulty scaling" — arXiv:2602.06717
- "CoRPO: Correctness-biased baseline" — arXiv:2511.04439
- "GSPO: Sequence-level IS" — Qwen team, arXiv:2507.18071

**Structured output RL:**
- "RL-Struct: Lightweight RL for Structured Output" — arXiv:2512.00319 (emergent curriculum in JSON tasks)
- "GRAPH-GRPO-LEX: Contract Graph Modeling" — SSRN:5566538 (structured segmentation + RL)

**Training speed / scaling:**
- "Predictive Scaling Laws for RL Post-Training" — arXiv:2507.18014 (exponential saturation, 80% epoch sufficiency)
- "CPPO: Accelerating RL via Completion Pruning" — arXiv:2503.22342 (7.98x speedup)
- "FastGRPO: Speculative Decoding for RL Generation" — arXiv:2509.21792 (2.35-2.72x speedup)

**vLLM logprobs:**
- vLLM PR #22387: `logprobs_mode` with `processed_logprobs` (final logprobs after sampling)
- vLLM PR #35961: `score_mode` for GPU-side per-token logprob extraction
- vLLM PR #21792: `logprobs=-1` for all token logprobs

---

## CPA Pressure Test Summary

Full 7-block Competitive Pressure Analysis was run on this plan. Key structural findings:

### Novel Finding

The plan's #1 item (store generation logprobs) was simultaneously the highest-demand AND highest-constraint feature. Standard prioritization put it first because it unlocks the most value. The field confirmed the value but revealed it carried the highest constraint load — making it the hardest to ship, not the easiest to start. The vLLM logprob MVP (Item 0) resolves this by eliminating the constraint entirely: zero VRAM cost, zero extra forward pass.

### Phantom Checklist (gaps not in any plan)

| Phantom | Type | Needed by | Resolution |
|---------|------|-----------|------------|
| vLLM logprob passthrough test | ENABLER | Item 1 | Now Item 0 — 5-line test |
| HIF output format spec (stable JSON keys) | PREREQUISITE | Item 3 | Keys come from training run config, not engine. Segmenter accepts configurable key patterns. |
| Scaffold prompt templates | PREREQUISITE | Item 7 | Domain work, not engine work. Item 7 deferred until templates exist. |
| MLflow alerting on metrics | ENABLER | Item 10 | Without alerts, metrics are passive logs. Add threshold alerts post-launch. |

### CPA Parameters

```
Du = 0.7 (universal demand — single user, single training run)
Dv = 1.5 (high complexity: 16GB GPU, Unsloth quirks, micro-batch OOM, vLLM mode transitions)
f  = 0.08 (near-blocking: three dead subsystems without these fixes)
k  = 0.06 (moderate: engine can train without fixes but quality degrades)
Stability: 0.08 < 0.100 → PASSES
```

### Decision Map

```
SHIP NOW:  Items 0, 2, 8, 10, 4, 5 (all clean spots, ship immediately)
SHIP NEXT: Items 1, 3 (prerequisites with MVPs that resolve constraints)
HOLD:      Item 6 (stagnation detection — scope to detection-only to cross)
DEFERRED:  Item 7 (scaffold fading — constraints exceed demand)
LAST:      Item 9 (dead code — convention-driven, not user-driven)
```
