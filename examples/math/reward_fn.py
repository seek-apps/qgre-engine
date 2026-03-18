# Minimal math reward function — demonstrates QGRE engine on a simple domain
# Uses MATH dataset format: prompt → chain of thought → \\boxed{answer}
#
# This is the simplest possible reward function: correct answer = 1.0, wrong = 0.0
# No step-level scoring — demonstrates the engine works even with scalar rewards
# (falls back to uniform per-token advantages, equivalent to standard GRPO)
