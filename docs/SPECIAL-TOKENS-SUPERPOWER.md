# QGRE Special Token Superpower — Step-Level Process Rewards

## The Insight

QGRE's structured XML output creates something no other RL training setup has:
**4 verifiable checkpoints within each completion, detectable via special tokens,
scored by programmatic verifiers, with region-specific exploration/exploitation control.**

Math problems have one answer at the end. Code has pass/fail. QGRE has:
```
<think>...</think>                    ← exploration zone (try reasoning paths)
<step1_extraction>...</step1>         ← checkpoint 1: did extraction succeed?
<step2_shared_context>...</step2>     ← checkpoint 2: does it ground in step 1?
<step3_nondecomposable>...</step3>    ← checkpoint 3: does it reference step 2?
<step4_output>{JSON}</step4>          ← checkpoint 4: correct verdict?
```

Each checkpoint is verifiable without a neural judge — our reward_fn already
scores each quality independently. We just need to assign the rewards at the
right token positions instead of dumping everything at the end.

---

## Technique 1: Step-Level Process Rewards (VPRMs)

### What it is
Instead of one scalar reward at the last token, assign rewards at each
step boundary. The model gets credit for getting step 1 right even if
step 4 is wrong.

### Source
- **VPRMs** (IBM Research, arxiv.org/abs/2601.17223, Jan 2026)
- **OpenAI "Let's Verify Step by Step"** (PRM800K, foundational paper)
- **ThinkPRM** (arxiv.org/abs/2504.16828, ICLR 2026 under review)

### How it applies to QGRE

Current reward placement:
```
tokens:  [think tokens...] [step1 tokens...] [step2 tokens...] [step3 tokens...] [step4 tokens...]
reward:  [0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0.65  0]
                                                                              last valid token ↑
```

Step-level reward placement:
```
tokens:  [think tokens...] [step1 tokens...] [step2 tokens...] [step3 tokens...] [step4 tokens...]
reward:  [0  0  0  0  0  0  0  1.0  0  0  0  0  0.8  0  0  0  0  0.7  0  0  0  0  0  0.5  0  0]
                     </step1> ↑        </step2> ↑        </step3> ↑              </step4> ↑
```

### Implementation

**CRITICAL: Step tags are multi-token sequences, NOT single tokens.**
Verified on Qwen3-1.7B nothink tokenizer (2026-03-18):
```
</step1_extraction>       -> 5 tokens: [522, 9520, 16, 94842, 29]  = ['</', 'step', '1', '_extraction', '>']
</step2_shared_context>   -> 6 tokens: [522, 9520, 17, 20405, 8467, 29]
</step3_nondecomposable>  -> 8 tokens: [522, 9520, 18, 1089, 17231, 874, 17421, 29]
</step4_output>           -> 5 tokens: [522, 9520, 19, 7645, 29]
<think> / </think>        -> 1 token each: [151667] / [151668]  ← THESE are true special tokens
```

The naive approach of `tokenizer.encode("</step1_extraction>")[-1]` gives token 29 (`>`),
which fires everywhere.

### Design Decision: SEGMENT propagation, not boundary-only

**The reward goes to ALL tokens in a step's region, not just the boundary token.**
Boundary-only placement gives 99% of tokens zero reward → zero advantage → no gradient.
Segment propagation gives each step's tokens the SAME reward → different steps get DIFFERENT
advantages → credit assignment works.

```
WRONG (boundary-only):
tokens:  [think...] [step1 content...] [step2 content...] [step3 content...] [step4 content...]
reward:  [0 0 0 0 0] [0 0 0 0 0 0 1.0] [0 0 0 0 0 0 0.8] [0 0 0 0 0 0 0.7] [0 0 0 0 0 0 0.5]
                              only here ↑          only here ↑         only here ↑      only here ↑
→ 99% of tokens get advantage ≈ 0. No learning signal for step content.

CORRECT (segment propagation):
tokens:  [think...] [step1 content...] [step2 content...] [step3 content...] [step4 content...]
reward:  [0 0 0 0 0] [1.0 1.0 1.0 1.0] [0.8 0.8 0.8 0.8] [0.7 0.7 0.7 0.7] [0.5 0.5 0.5 0.5]
          think=0 ←→  entire region ←→   entire region ←→   entire region ←→   entire region ←→
→ All tokens in step 1 get step 1's reward. Different steps get different advantages. ✓
```

This is how PRPO (arxiv 2601.07182) and StepGRPO segment reasoning by semantic structure.

### Quality→Step mapping (explicit)

Each step has multiple qualities. The step reward is the WEIGHTED MEAN of ACTIVE
qualities for that step (phase-gated by GameState).

```python
# Which qualities belong to which step region
STEP_QUALITIES = {
    1: ["q_format_tags", "q_tag_content", "q_node_in_prompt", "q_node_format", "q_node_length"],
    2: ["q_chain_s2_refs_s1"],
    3: ["q_chain_s3_refs_s2", "q_self_consistency"],
    4: ["q_step4_valid_json", "q_step4_has_keys", "q_existence_correct",
        "q_archetype_correct", "q_node_f1"],
}
# Global qualities (not step-specific) — add to overall sequence reward
GLOBAL_QUALITIES = ["q_eos_correct"]
```

### Token region segmentation (token ID pattern matching)

Uses structural token patterns — no decoded-text regex needed for segmentation.

```python
# Qwen3 token IDs for structural detection
THINK_START = 151667   # <think> — single special token
THINK_END   = 151668   # </think> — single special token
STEP_TOKEN  = 9520     # 'step' — appears in both <stepN_...> and </stepN_...>
OPEN_ANGLE  = 27       # '<'
CLOSE_SLASH = 522      # '</'
STEP_NUM_TOKENS = {16: 1, 17: 2, 18: 3, 19: 4}  # '1','2','3','4'

def segment_completion(token_ids):
    """Assign each token to a region: THINK, STEP_1..4, FORMAT, OTHER.

    Uses token ID patterns, not decoded text. Fast, exact.
    Returns list[str] of region labels, same length as token_ids.
    """
    regions = ["OTHER"] * len(token_ids)
    current = "OTHER"

    i = 0
    while i < len(token_ids):
        tid = token_ids[i]

        # Think block boundaries (single special tokens — reliable)
        if tid == THINK_START:
            current = "THINK"
        elif tid == THINK_END:
            regions[i] = "THINK"  # the </think> token itself is still think
            current = "OTHER"
            i += 1
            continue

        # Step opening: < step N _ ... >
        # Pattern: OPEN_ANGLE(27) STEP_TOKEN(9520) NUM(16-19) ...
        if tid == OPEN_ANGLE and i + 2 < len(token_ids):
            if token_ids[i+1] == STEP_TOKEN and token_ids[i+2] in STEP_NUM_TOKENS:
                step = STEP_NUM_TOKENS[token_ids[i+2]]
                current = f"STEP_{step}"
                # Mark the tag tokens themselves as FORMAT (not CONTENT)
                # Find the closing > of this opening tag
                j = i
                while j < len(token_ids) and token_ids[j] != 29:  # 29 = '>'
                    regions[j] = "FORMAT"
                    j += 1
                if j < len(token_ids):
                    regions[j] = "FORMAT"  # the > itself
                i = j + 1
                continue

        # Step closing: </ step N _ ... >
        if tid == CLOSE_SLASH and i + 2 < len(token_ids):
            if token_ids[i+1] == STEP_TOKEN and token_ids[i+2] in STEP_NUM_TOKENS:
                # Mark closing tag tokens as FORMAT
                j = i
                while j < len(token_ids) and token_ids[j] != 29:
                    regions[j] = "FORMAT"
                    j += 1
                if j < len(token_ids):
                    regions[j] = "FORMAT"
                current = "OTHER"
                i = j + 1
                continue

        regions[i] = current
        i += 1

    return regions
```

### Full VPRM implementation

```python
def compute_step_region_rewards(token_ids, reward_result, active_qualities):
    """Per-token rewards via SEGMENT PROPAGATION.

    All tokens in step N's region get step N's reward.
    Step reward = mean of ACTIVE qualities for that step.
    Think tokens and format tokens get 0 (no reward for structure/exploration).

    Args:
        token_ids: list[int] — completion token IDs
        reward_result: RewardResult from _hypergraph_reward_internal
        active_qualities: list[str] — phase-active quality keys from GameState

    Returns:
        token_rewards: Tensor[seq_len] — per-token reward values
    """
    regions = segment_completion(token_ids)

    # Compute per-step reward: mean of ACTIVE qualities for that step
    step_rewards = {}
    for step_num, quality_keys in STEP_QUALITIES.items():
        active = [k for k in quality_keys if k in active_qualities]
        if active:
            step_rewards[step_num] = float(np.mean([
                reward_result.scores.get(k, 0.0) for k in active
            ]))
        else:
            step_rewards[step_num] = 0.0

    # Assign rewards by region
    token_rewards = torch.zeros(len(token_ids))
    for i, region in enumerate(regions):
        if region.startswith("STEP_"):
            step_num = int(region.split("_")[1])
            token_rewards[i] = step_rewards.get(step_num, 0.0)
        # THINK, FORMAT, OTHER → 0.0 (no reward for structure/exploration)

    return token_rewards
```

### How per-token rewards flow into advantages

**With GRPO (n>1):** Per-token rewards are used instead of per-sequence scalar.
Group advantage per token: A_t = (r_t - mean(r_t across group)) / (std + eps).
Tokens in correct step 1 across all group members get positive advantage even
when step 4 fails in some members.

**With SPO (n=1):** Per-token rewards compared against persistent value tracker.
Need per-STEP value trackers: V[prompt][step_num] = EMA of that step's reward.
Advantage: A_t = r_t - V(prompt, step_of_t).

**Composition with GDPO:** Per-component per-step advantages are possible but
adds complexity. Start with per-step aggregated rewards (weighted mean of
active qualities). Add per-component tracking in phase 2 if needed.

### Why this matters for GRPO/SPO

Standard GRPO: advantage is the same for ALL tokens in the completion.
A correct step 1 followed by wrong step 4 → the entire completion gets
low advantage → step 1's correct behavior gets punished.

Step-level rewards: step 1's correct behavior gets rewarded even when
step 4 fails. The model learns "how to do step 1 right" independently
of step 4 performance. This is **process supervision without a neural judge**.

### Interaction with QGRE phases

Phase 1 (format only): rewards placed at each `</stepN>` tag for tag presence
Phase 2 (+ grounding): step 1-2 get grounding reward, steps 3-4 get format only
Phase 3 (+ chain): steps 1-3 get chain coherence reward
Phase 4+ (+ accuracy): step 4 gets accuracy reward

The phase gating COMPOSES with step-level rewards — each step only receives
reward for the qualities active in the current phase.

---

## Technique 2: Entropy-Weighted Credit Assignment (GTPO)

### What it is
Use the model's own per-token entropy to weight which tokens get more
gradient signal. High-entropy tokens = decision points where the model
is uncertain = where learning signal matters most.

### Source
- **GTPO** (ByteDance, arxiv.org/abs/2508.04349, ICML accepted)
- **Token Hidden Reward** (arxiv.org/abs/2510.03669)
- **λ-GRPO** (arxiv.org/abs/2510.06870, learnable token preferences)

### How it applies to QGRE

The model's entropy profile during generation looks roughly like:

```
entropy: [HIGH HIGH HIGH ...think... HIGH] [LOW LOW ... step1 content ... MED] [MED ... step2 ...]
          ← thinking (exploring)          → ← format tokens (confident)       →

Spikes at:
  - Start of <think> block (what reasoning strategy?)
  - Transition from </think> to <step1> (commit to structure)
  - Inside step content (choosing node names, shared context description)
  - step4 JSON (exists: true vs false — the key decision)
```

### Implementation

```python
def entropy_weighted_advantages(log_probs, advantages, response_mask):
    """Weight advantages by per-token entropy.

    High-entropy tokens get MORE gradient signal (they're the decision points).
    Low-entropy tokens (format tags, common words) get LESS (model is already confident).
    """
    # Per-token entropy from log_probs (already computed during forward pass)
    # H(t) = -sum(p * log(p)) across vocab — but we approximate from log_softmax
    # Actually, we already compute entropy in the forward pass via entropy_from_logits
    # Just need to NOT throw it away

    # Normalize entropy across sequence to get weights
    entropy_weights = token_entropy / (token_entropy.mean() + 1e-8)

    # Clamp to prevent extreme weights
    entropy_weights = entropy_weights.clamp(0.5, 2.0)

    # Apply to advantages
    weighted_advantages = advantages * entropy_weights * response_mask

    return weighted_advantages
```

### Why this matters for QGRE

Without entropy weighting: the model gets equal gradient for typing
`<step1_extraction>` (easy, low entropy) and for choosing the correct
node names (hard, high entropy). Wastes gradient on what's already learned.

With entropy weighting: the gradient concentrates on the DECISIONS —
which nodes to extract, whether the hyperedge exists, what archetype
to assign. The format tokens get minimal gradient because the model
is already confident about them.

### The think → XML transition

The highest-entropy moment in a QGRE completion is the transition from
`</think>` to `<step1_extraction>`. This is where the model commits
from open-ended reasoning to structured output. Entropy weighting
naturally amplifies gradient at this exact point — teaching the model
to make this transition effectively.

---

## Technique 3: Region-Specific Exploration/Exploitation (THR)

### What it is
Different regions of the completion should have different learning dynamics:
- Think blocks → EXPLORATION (try different reasoning paths)
- XML tags → EXPLOITATION (nail the format and content)

### Source
- **Token Hidden Reward** (UBC, arxiv.org/abs/2510.03669)
- **"Ignore the KL Penalty"** (arxiv.org/abs/2502.06533, critical tokens)

### How it applies to QGRE

We can detect regions using special tokens + decoded text:

**Note:** `<think>` (151667) and `</think>` (151668) ARE true single-token special tokens
in Qwen3. Step tags are NOT — they are 5-8 tokens each. Region classification must
use a hybrid approach.

```python
def classify_token_regions(token_ids, tokenizer):
    """Classify each token as THINK, FORMAT, or CONTENT.

    Think boundaries: single special tokens (reliable, exact).
    Step boundaries: decoded text regex → token position mapping (see VPRM impl above).
    """
    THINK_START = 151667  # <think> — verified Qwen3 special token
    THINK_END = 151668    # </think> — verified Qwen3 special token

    # Phase 1: classify THINK vs non-THINK (reliable — single tokens)
    regions = ["OTHER"] * len(token_ids)
    in_think = False
    for i, tid in enumerate(token_ids):
        if tid == THINK_START:
            in_think = True
        elif tid == THINK_END:
            in_think = False
            regions[i] = "OTHER"
            continue
        if in_think:
            regions[i] = "THINK"

    # Phase 2: classify FORMAT vs CONTENT within non-THINK regions
    # Uses decoded text + regex to find step tag token spans
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    # Build char→token map, find step tag spans, mark FORMAT/CONTENT
    # (see compute_step_rewards for the mapping pattern)
    # Tokens inside step tags but not the tags themselves = CONTENT
    # The tag tokens themselves = FORMAT

    return regions
```

### Learning dynamics per region

| Region | KL penalty | Advantage weight | Effect |
|--------|-----------|-----------------|--------|
| THINK | LOW (0.1x) | Normal | Explore reasoning paths freely |
| FORMAT | HIGH (2.0x) | Low (already learned) | Lock in structure |
| CONTENT | MEDIUM (1.0x) | High (entropy-weighted) | Focus on quality |
| OTHER | Normal | Normal | Default behavior |

### Why this matters

Current training applies the same KL penalty everywhere. This means:
- Think blocks are penalized for exploring novel reasoning → kills creativity
- Format tokens are under-penalized → allows format regression
- Content quality gets same signal as format → model can't distinguish
  "get the structure right" from "get the answer right"

Region-specific control lets the model THINK freely while ANSWERING precisely.

---

## Technique 4: F-GRPO — Difficulty-Aware Advantage Scaling

### What it is
Down-weight gradient updates on prompts the model already handles well.
Focus learning budget on prompts where the model struggles.

### Source
- **F-GRPO** (arxiv.org/abs/2602.06717, inspired by Focal Loss)
- **SPO adaptive curriculum** (built-in via persistent value tracker)

### How it applies to QGRE

QGRE already has tier gating (active vs inactive prompts) and phase
gating (which qualities are scored). F-GRPO adds WITHIN-TIER difficulty
scaling: among active tier-1 prompts, some are easy (shared_mutable_state)
and some are hard (shared_consensus). F-GRPO scales advantages so:

```
advantage_scaled = advantage * (1 - success_rate)^γ
```

Where `success_rate` comes from GameState's per-archetype quality windows.
Easy archetypes (success_rate ≈ 0.8) get small advantages (model already
knows this). Hard archetypes (success_rate ≈ 0.2) get large advantages
(model needs to focus here).

This is ALREADY partially implemented via GameState's Elo system and
combo multiplier — but F-GRPO makes it mathematically principled.

With SPO's persistent value tracker, this happens naturally: prompts
with high V(prompt) produce small advantages for good completions.

---

## Combined Architecture: QGRE Process Rewards

All four techniques compose into a single system:

```
1. Generate completion with <think> + 4 XML steps
2. Detect step boundaries via special tokens
3. Score each step independently (VPRM — our reward_fn already does this)
4. Place rewards at step boundaries, not just at the end
5. Compute per-token entropy from the forward pass
6. Classify tokens into THINK/FORMAT/CONTENT regions
7. Weight advantages by entropy (GTPO) × region multiplier (THR)
8. Scale by difficulty (F-GRPO via GameState success rate or SPO value tracker)
9. Apply GDPO per-component normalization on the step-level rewards
```

This gives per-token, per-step, per-component, difficulty-scaled,
region-aware advantages — all from programmatic verifiers and the
model's own entropy. No neural judge. No reward model. No human labels.

### Paper contribution

"QGRE with Verifiable Process Rewards: Step-level supervision for
novel-domain structured reasoning via programmatic verification and
entropy-weighted credit assignment."

This is a publishable result because:
1. Process rewards without neural judges (VPRMs use our programmatic reward_fn)
2. Entropy-weighted credit assignment on structured output (novel application of GTPO)
3. Region-specific KL control for think-then-structure generation (novel)
4. Composition with phase-gated curriculum (QGRE + VPRM + GTPO + SPO — nobody has this)

---

## Implementation Priority (UPDATED after CPA + Research Validation)

| Technique | Impact | Effort | When | Status |
|-----------|--------|--------|------|--------|
| Step-level rewards (VPRM) | HIGH — fixes credit assignment | 4 hours* | Phase 1 | **SHIP** |
| Entropy weighting (GTPO) | MEDIUM — focuses gradient on decisions | 3 hours | Phase 1.5 | **HOLD** (H100) |
| Region KL control (THR) | MEDIUM — think freely, answer precisely | 4 hours | Phase 2 | **HOLD** (engine) |
| Difficulty scaling (F-GRPO) | REDUNDANT — SPO handles this | — | Never | **KILL** |

*VPRM effort revised from 2→4 hours: multi-token tag handling adds complexity (see impl above).

Step-level rewards are the highest priority — they fix the core credit assignment
problem. **Effort is higher than originally estimated** because step tags are multi-token
sequences requiring decoded-text regex (not single-token scanning).

---

## CPA Pressure Test Results (2026-03-18)

Full 7-block Competitive Pressure Analysis run against this document.
Parameters: Du=0.5, Dv=0.9, f=0.06, k=0.05.

### Key findings

1. **Only the VPRM bundle survives under pressure.** Step boundary detection +
   per-step scoring (already exists) + reward placement at boundaries. Ship together.

2. **GTPO is HELD behind hardware.** Entropy computation blocked on 16GB GPU
   (entropy_coeff already disabled). Resolves on H100 80GB. Not a design problem.

3. **THR is HELD behind engine.** Region-specific KL needs the custom QGRE engine
   (keystone constraint C1, cascade score=6). Think block detection is easy
   (`<think>`/`</think>` are single tokens) but step region detection shares
   the multi-token complexity with VPRMs.

4. **F-GRPO is KILLED.** SPO's persistent value tracker handles difficulty scaling
   naturally. Building F-GRPO is paper-motivated redundancy.

5. **Combined 4-technique architecture is KILLED.** Genuine void — composition of 4
   multiplicative advantage modifications is untested and structurally unsound.
   Ship techniques individually, not as monolith.

6. **C1 (custom engine) is KEYSTONE.** Removing it unlocks 6 held features.
   Minimum viable version: per-token reward loop, not full 6-pillar engine.

### Eli parallax analysis (2026-03-18)

> The CPA got the most important thing right: VPRMs are the structural core.
> They extend the QGRE philosophy (gate rewards by quality phase) to a finer
> grain (gate rewards by position). Same pedagogy, finer resolution.
>
> The CPA double-counted C1 — the engine constraint was priced into both
> constraint nodes AND individual feature I(x) values. After the engine exists,
> THR (#13, #14, #15) come back to life with strong recovery (Net=+0.014).
>
> VPRMs are the ONLY technique that extends the actual QGRE thesis (meeting the
> learner where it is). GTPO and THR are optimization techniques — they make
> training faster, not better. Different category of intervention entirely.

### Paper framing recommendation

**"Verifiable Process Rewards for Novel-Domain Structured Reasoning"**
— not "4 techniques composed." VPRMs with programmatic verifiers in a domain
where no neural PRM exists is novel enough alone. Add GTPO as "preliminary
results + future work" once running on H100.

---

## Research Validation (2026-03-18, Exa live search)

### GRPO is Secretly a Process Reward Model (Sullivan, ICLR 2026 under review)
arxiv.org/abs/2509.21154

**Critical finding:** GRPO already induces an implicit PRM through within-group
token overlap. When completions share the same step 1 prefix but diverge at
step 4, GRPO naturally gives step 1 tokens a smoothed advantage and polarizes
step 4 tokens. This IS process supervision — hidden in the group structure.

**Implication for QGRE + SPO:**
- SPO uses n=1 → NO group → NO within-group overlap → implicit PRM disappears
- **With SPO, explicit VPRMs become ESSENTIAL** (not just nice-to-have)
- With Dr.GRPO fallback (n=4), implicit PRM helps, but explicit VPRMs still
  add value because QGRE step boundaries don't align with natural token overlap

### GTPO Validated (ByteDance, ICML accepted)
arxiv.org/abs/2508.04349 — v6 as of Feb 2026

- SOTA on AIME and MATH 500, outperforming DAPO baseline
- GitHub implementation: winstonsmith1897/GTPO (38 stars, MIT license)
- Uses entropy ratio H_{i,t} / sum(H_{k,t}) to weight per-token rewards
- Confirmed: entropy weighting works in practice, not just theory
- **Also under review at ICLR 2026** (OpenReview CFF6zXErgS)

### PRPO: Process Relative Policy Optimization (Shanghai University, Feb 2026)
arxiv.org/abs/2601.07182

- Combines outcome rewards with process-level guidance in critic-free framework
- Segments reasoning by semantic clues, normalizes PRM scores to token-level advantages
- Uses "location-parameter shift" to align process advantages with outcome advantages
- MATH500: Qwen2.5-Math-1.5B from 61.2% to higher (exact figure in paper)
- **Relevant pattern** for how we'd integrate VPRMs with existing reward

### StepGRPO (emergentmind topic, multiple papers)
Per-step group normalization — computes and normalizes cumulative returns at
each timestep. Gains in multimodal, visual generation, interactive control.

### VPRMs Confirmed (IBM Research, Jan 2026)
arxiv.org/abs/2601.17223

- Applied to medical evidence synthesis (risk-of-bias assessment)
- Rule-based verifiers (no neural judge) — exactly our approach
- **20% higher F1** than SOTA, **6.5% higher** than verifiable outcome rewards
- Domain: guideline-defined criteria with rule-based decision paths
- **Direct analogue to QGRE:** structured reasoning with programmatic verification

### λ-GRPO: Learnable Token Preferences (Wang et al.)
arxiv.org/abs/2510.06870 — ICLR 2026, WITHDRAWN

- Different paper from Sullivan's "secretly a PRM" — same name, different concept
- Introduces learnable λ for adaptive token-level weighting
- +1.9% average accuracy on Qwen2.5-1.5B vs vanilla GRPO
- Withdrawn from ICLR 2026 but concept is validated

---

## Multi-Token Tag Finding (2026-03-18)

**Verified empirically on Qwen3-1.7B nothink tokenizer.**
This invalidates naive single-token boundary detection throughout the document.

| Tag | Token count | Token IDs |
|-----|-------------|-----------|
| `</step1_extraction>` | 5 | [522, 9520, 16, 94842, 29] |
| `</step2_shared_context>` | 6 | [522, 9520, 17, 20405, 8467, 29] |
| `</step3_nondecomposable>` | 8 | [522, 9520, 18, 1089, 17231, 874, 17421, 29] |
| `</step4_output>` | 5 | [522, 9520, 19, 7645, 29] |
| `<think>` | 1 | [151667] — true special token |
| `</think>` | 1 | [151668] — true special token |

**Impact:** All implementation pseudocode in this document has been updated to use
decoded-text regex + char→token position mapping instead of single-token scanning.
Think/think boundary detection remains trivial (single special tokens).

---

## References

- **VPRMs**: arxiv.org/abs/2601.17223 (IBM Research, Jan 2026) — process rewards with rule-based verifiers
- **OpenAI PRM800K**: "Let's Verify Step by Step" — foundational process reward paper
- **ThinkPRM**: arxiv.org/abs/2504.16828 — generative process reward models
- **GTPO**: arxiv.org/abs/2508.04349 (ByteDance, ICML) — entropy-weighted token rewards
- **THR**: arxiv.org/abs/2510.03669 (UBC) — token-level exploration/exploitation
- **λ-GRPO (learnable)**: arxiv.org/abs/2510.06870 — learnable token preferences (ICLR 2026, withdrawn)
- **λ-GRPO (PRM)**: arxiv.org/abs/2509.21154 — GRPO is secretly a PRM (ICLR 2026 under review)
- **PRPO**: arxiv.org/abs/2601.07182 — process + outcome reward alignment
- **F-GRPO**: arxiv.org/abs/2602.06717 — difficulty-aware advantage scaling (KILLED — SPO handles this)
- **Critical Tokens**: arxiv.org/abs/2502.06533 — ignore KL on pivotal tokens
- **StepGRPO**: emergentmind.com/topics/step-wise-group-relative-policy-optimization-stepgrpo — per-step normalization
- **S-GRPO/T-SPMO**: arxiv.org/abs/2504.20834 — token-efficient RL for reasoning
- **TP-GRPO**: arxiv.org/abs/2602.06422 — step-wise rewards for flow-based GRPO (diffusion, but relevant concept)
- **Awesome PRMs**: github.com/RyanLiu112/Awesome-Process-Reward-Models — comprehensive list
