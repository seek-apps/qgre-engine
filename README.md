# QGRE Engine

Single-GPU GRPO training engine for novel-domain structured reasoning.

No Ray. No multi-process serialization. Just: generate → score → compute advantages → backward → update.

## What is QGRE?

**Quality-Gated Reward Escalation** is a phase-gated curriculum for training LLMs on novel domains where correct reasoning methodology is unknown. Instead of rewarding all qualities from step 1, QGRE unlocks reward components progressively: format first, then grounding, then chain coherence, then accuracy.

The engine combines four innovations in one unified system:

- **VPRM** — Step-level process rewards via programmatic verifiers. Each step of the structured output gets its own reward, enabling credit assignment at the step level.
- **SPO** — Persistent per-prompt value tracking replaces noisy group-mean baselines.
- **GDPO** — Per-step normalization prevents reward crowding across steps.
- **QGRE** — Phase-gated curriculum controls which qualities are active per step.

## Architecture

```
Python loop → vLLM generate → reward_fn → QGREStepAdvantageEstimator → loss → backward → update
```

One process. One GPU. Direct function calls. ~500 lines of code.

## Bring Your Own Domain

```python
from dataclasses import dataclass

@dataclass
class RewardResult:
    reward: float
    scores: dict          # {"quality_a": 0.8, "quality_b": 1.0, ...}
    phase: int            # current curriculum phase

# Map your qualities to your output structure's steps
STEP_QUALITIES = {
    1: ["quality_a", "quality_b"],
    2: ["quality_c"],
    3: ["quality_d", "quality_e"],
}
```

Implement your reward function, define step-quality mapping, run the engine.

## Install

```bash
pip install -e ".[unsloth]"
```

## Status

Under active development. See [docs/PLAN.md](docs/PLAN.md) for the full build plan.

## References

- QGRE paper: (forthcoming)
- [VPRMs](https://arxiv.org/abs/2601.17223) (IBM Research, Jan 2026)
- [SPO](https://arxiv.org/abs/2509.13232) (Tencent, ICLR 2026)
- [GDPO](https://arxiv.org/abs/2601.05242) (NVIDIA, Jan 2026)
- [GTPO](https://arxiv.org/abs/2508.04349) (ByteDance, ICML)

## License

Apache-2.0. NeMo RL extracted components retain their original Apache-2.0 headers.
