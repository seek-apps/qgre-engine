# Minimal math reward function — demonstrates QGRE engine on a simple domain
# Uses MATH dataset format: prompt -> chain of thought -> \boxed{answer}
#
# This is the simplest possible reward function: correct answer = 1.0, wrong = 0.0
# No step-level scoring — demonstrates the engine works even with scalar rewards
# (falls back to uniform per-token advantages via uniform_segmenter)

import re

from qgre.types import RewardResult


def math_reward_fn(prompt: str, completion: str, metadata: dict | None = None) -> RewardResult:
    """Score a math completion by extracting \\boxed{answer} and comparing to ground truth."""
    ground_truth = (metadata or {}).get("answer", "")

    # Extract answer from \boxed{...}
    match = re.search(r"\\boxed\{([^}]+)\}", completion)
    predicted = match.group(1).strip() if match else ""

    correct = 1.0 if predicted == str(ground_truth).strip() else 0.0

    return RewardResult(
        reward=correct,
        scores={"q_correct_answer": correct},
        phase=1,
    )
