"""Tests for Hamiltonian reward function — catch scoring bugs before training."""

import pytest
from examples.hamiltonian.reward_fn import hamiltonian_reward


SPRING_PROMPT = (
    "A block of mass 3 kg is attached to a spring with spring constant "
    "k = 6 N/m on a frictionless surface. Let x be the displacement from "
    "equilibrium.\n\nDerive the Hamiltonian H(x, p) from first principles "
    "and find Hamilton's equations of motion."
)

SPRING_META = {
    "ground_truth": "H = p**2/6 + 3*x**2; dx/dt = p/3; dp/dt = -6*x",
    "H_expr": "p**2/6 + 3*x**2",
    "T_expr": "p**2/6",
    "V_expr": "3*x**2",
    "dqdt": "p/3",
    "dpdt": "-6*x",
    "coordinates": "x",
    "difficulty": "tier1",
    "system": "spring",
}

PERFECT_STRUCTURED = """
COORDINATES: q = x
MOMENTUM: p = m*dx/dt = 3*dx/dt
KINETIC: T = p²/(2*3) = p²/6
POTENTIAL: V = (6/2)*x² = 3x²
HAMILTONIAN: H = p²/6 + 3x²
EQUATIONS:
  dq/dt = p/3
  dp/dt = -6x
"""

VELOCITY_FORM = """
COORDINATES: q = x
MOMENTUM: p = 3*dx/dt
KINETIC: T = (1/2)*3*(dx/dt)² = (3/2)*(dx/dt)²
POTENTIAL: V = 3x²
HAMILTONIAN: H = (3/2)*(dx/dt)² + 3x²
EQUATIONS:
  dq/dt = dx/dt
  dp/dt = -6x
"""

WRONG_COEFFICIENT = """
COORDINATES: q = x
MOMENTUM: p = 3*dx/dt
KINETIC: T = p²/6
POTENTIAL: V = 3x²
HAMILTONIAN: H = p²/6 + 3x²
EQUATIONS:
  dq/dt = p/3
  dp/dt = -3x
"""

NO_LABELS = """
The kinetic energy is T = p²/6 and the potential energy is V = 3x².
So H = p²/6 + 3x²
Hamilton's equations give us:
  dx/dt = p/3
  dp/dt = -6x
"""


class TestStructuredFormat:
    """Test that labeled sections are correctly extracted and scored."""

    def test_perfect_structured_scores_high(self):
        result = hamiltonian_reward(SPRING_PROMPT, PERFECT_STRUCTURED, SPRING_META)
        assert result.scores["q_format"] >= 0.9
        assert result.scores["q_momentum_defined"] >= 0.7
        assert result.scores["q_T_uses_p"] >= 0.7
        assert result.scores["q_V_correct"] >= 0.9
        assert result.scores["q_correct_dqdt"] >= 0.9
        assert result.scores["q_correct_dpdt"] >= 0.9
        assert result.scores["q_correct_H"] >= 0.9
        assert result.reward >= 0.8

    def test_velocity_form_T_scores_low(self):
        """T in velocity form (dx/dt)² instead of momentum form (p²) → q_T_uses_p should be low."""
        result = hamiltonian_reward(SPRING_PROMPT, VELOCITY_FORM, SPRING_META)
        assert result.scores["q_T_uses_p"] <= 0.5, \
            f"Velocity form T should score <= 0.5, got {result.scores['q_T_uses_p']}"

    def test_wrong_coefficient_gradient(self):
        """dp/dt = -3x instead of -6x → should score lower than correct."""
        correct = hamiltonian_reward(SPRING_PROMPT, PERFECT_STRUCTURED, SPRING_META)
        wrong = hamiltonian_reward(SPRING_PROMPT, WRONG_COEFFICIENT, SPRING_META)
        gap = correct.scores["q_correct_dpdt"] - wrong.scores["q_correct_dpdt"]
        assert gap >= 0.15, f"Gap should be >= 0.15, got {gap}"

    def test_no_labels_still_works(self):
        """Without structured labels, fallback extraction should still work."""
        result = hamiltonian_reward(SPRING_PROMPT, NO_LABELS, SPRING_META)
        assert result.scores["q_correct_dqdt"] >= 0.5
        assert result.scores["q_correct_H"] >= 0.5

    def test_format_rewards_labels(self):
        """q_format should reward structured labels over free-form."""
        structured = hamiltonian_reward(SPRING_PROMPT, PERFECT_STRUCTURED, SPRING_META)
        freeform = hamiltonian_reward(SPRING_PROMPT, NO_LABELS, SPRING_META)
        assert structured.scores["q_format"] > freeform.scores["q_format"]


class TestMomentumForm:
    """Test that momentum-form qualities distinguish p from q̇."""

    def test_momentum_with_numbers(self):
        text = "MOMENTUM: p = 3*dx/dt"
        result = hamiltonian_reward(SPRING_PROMPT, f"COORDINATES: q = x\n{text}\nKINETIC: T = p²/6\nPOTENTIAL: V = 3x²\nHAMILTONIAN: H = p²/6 + 3x²\nEQUATIONS:\n  dq/dt = p/3\n  dp/dt = -6x", SPRING_META)
        assert result.scores["q_momentum_defined"] >= 0.7

    def test_no_momentum_section(self):
        text = "H = p²/6 + 3x²"
        result = hamiltonian_reward(SPRING_PROMPT, text, SPRING_META)
        assert result.scores["q_momentum_defined"] <= 0.5


class TestNumericalEquivalence:
    """Test that symbolically equivalent expressions score 1.0."""

    def test_equivalent_H_forms(self):
        for H_form in ["p**2/6 + 3*x**2", "(1/6)*p**2 + 3*x**2", "3*x**2 + p**2/6"]:
            text = f"COORDINATES: q = x\nMOMENTUM: p = 3v\nKINETIC: T = p²/6\nPOTENTIAL: V = 3x²\nHAMILTONIAN: H = {H_form}\nEQUATIONS:\n  dq/dt = p/3\n  dp/dt = -6*x"
            result = hamiltonian_reward(SPRING_PROMPT, text, SPRING_META)
            assert result.scores["q_correct_H"] >= 0.9, \
                f"H form '{H_form}' should score >= 0.9, got {result.scores['q_correct_H']}"


class TestScoreOrdering:
    """Test that better answers score higher — this IS the gradient signal."""

    def test_overall_ordering(self):
        scores = {}
        completions = {
            "perfect": PERFECT_STRUCTURED,
            "wrong_coeff": WRONG_COEFFICIENT,
            "velocity_form": VELOCITY_FORM,
            "garbage": "The Hamiltonian is a thing in physics.",
        }
        for name, comp in completions.items():
            result = hamiltonian_reward(SPRING_PROMPT, comp, SPRING_META)
            scores[name] = result.reward

        assert scores["perfect"] > scores["wrong_coeff"], \
            f"perfect ({scores['perfect']:.2f}) > wrong_coeff ({scores['wrong_coeff']:.2f})"
        assert scores["wrong_coeff"] > scores["velocity_form"], \
            f"wrong_coeff ({scores['wrong_coeff']:.2f}) > velocity_form ({scores['velocity_form']:.2f})"
        assert scores["velocity_form"] > scores["garbage"], \
            f"velocity_form ({scores['velocity_form']:.2f}) > garbage ({scores['garbage']:.2f})"

    def test_empty_scores_zero(self):
        result = hamiltonian_reward(SPRING_PROMPT, "", SPRING_META)
        assert result.reward < 0.1

    def test_garbage_scores_low(self):
        result = hamiltonian_reward(SPRING_PROMPT, "I don't know.", SPRING_META)
        assert result.reward < 0.2
