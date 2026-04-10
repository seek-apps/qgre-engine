"""Correctness-only reward function for Hamiltonian mechanics training.

Scores mathematical correctness via sympy equivalence. No format scoring.
The model can write in any format — LaTeX, ASCII, unicode — the reward
function extracts all expressions and checks each against ground truth.

Design principles (from April 2026 analysis):
- RL teaches WHAT to think (correctness). SFT teaches HOW to present (format).
- Format scoring in RL is the #1 reward hacking vector (confirmed by
  gradient regularization paper arxiv:2602.18037).
- Score the first correct occurrence, ignore duplicates.
- Use sympy equivalence, not string matching.

Qualities:
  q_coordinates: correct generalized coordinate choice
  q_momentum:    correct conjugate momentum expression
  q_kinetic:     correct kinetic energy in momentum form
  q_potential:   correct potential energy expression
  q_hamiltonian: correct Hamiltonian (T + V)
  q_dqdt:        correct dq/dt Hamilton equation
  q_dpdt:        correct dp/dt Hamilton equation
  q_consistency: equations are consistent with H (derivative check)
"""

from __future__ import annotations

import logging
import re
import signal
from contextlib import contextmanager

import sympy as sp

from qgre.types import RewardResult


logger = logging.getLogger("qgre.hamiltonian_reward")


# ─── Timeout for sympy operations ──────────────────────────────────────────────


class SympyTimeoutError(Exception):
    pass


@contextmanager
def sympy_timeout(seconds: int = 2):
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):
        raise SympyTimeoutError(f"Sympy timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ─── Expression extraction ─────────────────────────────────────────────────────

# Symbols the model might use
_SYMBOLS = sp.symbols("x y r theta q p t m k g", real=True)
_SYMBOL_MAP = {str(s): s for s in _SYMBOLS}
# Add common derivative notations
_SYMBOL_MAP.update({"p_theta": sp.Symbol("p_theta"), "p_r": sp.Symbol("p_r")})


def _clean_for_sympy(expr_str: str) -> str:
    """Minimal cleanup to make an expression parseable by sympy.sympify.

    Handles LaTeX artifacts, unicode, and common notation without
    trying to be a full LaTeX parser.
    """
    s = expr_str.strip()
    # Strip LaTeX delimiters
    s = re.sub(r"\$+", "", s)
    # Strip markdown bold/italic at string boundaries only — preserve interior **
    # (which is the power operator in math expressions like p**2)
    s = re.sub(r"^\*{1,3}", "", s)
    s = re.sub(r"\*{1,3}$", "", s).strip()
    # Strip trailing backslashes
    s = re.sub(r"\\+$", "", s).strip()
    # LaTeX fractions: \frac{a}{b} → (a)/(b)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
    # LaTeX \cdot → *
    s = s.replace("\\cdot", "*")
    # LaTeX \left, \right, display commands
    s = re.sub(r"\\(left|right|displaystyle|,|;|quad|qquad|text\{[^}]*\})", "", s)
    # LaTeX sqrt
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    # LaTeX trig
    s = re.sub(r"\\(sin|cos|tan|exp|log|ln)\b", r"\1", s)
    # Strip remaining LaTeX commands
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    # Strip braces
    s = s.replace("{", "").replace("}", "")
    # Unicode superscripts
    s = s.replace("²", "**2").replace("³", "**3")
    # Unicode operators
    s = s.replace("·", "*").replace("×", "*")
    # Implicit multiplication: 2x → 2*x, )x → )*x
    s = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", s)
    s = re.sub(r"\)([a-zA-Z(])", r")*\1", s)
    return s.strip()


def _try_parse(expr_str: str) -> sp.Basic | None:
    """Try to parse an expression string into a sympy expression."""
    cleaned = _clean_for_sympy(expr_str)
    if not cleaned or len(cleaned) < 1:
        return None
    try:
        with sympy_timeout(2):
            return sp.sympify(cleaned, locals=_SYMBOL_MAP)
    except (SympyTimeoutError, sp.SympifyError, ValueError, TypeError, SyntaxError):
        return None


def _expressions_equivalent(student: sp.Basic, teacher: sp.Basic) -> bool:
    """Check if two sympy expressions are mathematically equivalent."""
    try:
        with sympy_timeout(3):
            diff = sp.simplify(student - teacher)
            return diff == 0 or sp.simplify(diff) == 0
    except (SympyTimeoutError, sp.SympifyError, ValueError, TypeError):
        return False


def _extract_all_expressions(text: str) -> list[tuple[str, str, int, int]]:
    """Extract all 'LHS = RHS' patterns from text.

    Returns list of (lhs, rhs, char_start, char_end) tuples.
    char_start/char_end are character offsets into the original text
    covering the entire 'LHS = RHS' expression.

    Finds:
    - Standard assignments: T = p²/(2m)
    - Derivative expressions: dq/dt = p/m, \\frac{dq}{dt} = p/m
    - Labeled assignments: KINETIC: T = p²/(2m)
    - LaTeX-wrapped: $ T = p^2/(2m) $
    """
    results = []
    # Match any line containing '='
    for match in re.finditer(r"[^\n]*=[^\n]*", text):
        line = match.group()
        line_start = match.start()

        # Find the '=' position
        eq_pos = line.find("=")
        if eq_pos < 0:
            continue

        lhs_raw = line[:eq_pos].strip()
        rhs_raw = line[eq_pos + 1 :].strip()

        # Clean LHS: strip labels, markdown, LaTeX delimiters
        lhs = re.sub(r"^[A-Z][A-Z\s]*:", "", lhs_raw).strip()  # Strip "KINETIC:" prefix
        lhs = re.sub(r"^\*+|^\#+", "", lhs).strip()  # Strip markdown
        lhs = re.sub(r"\$", "", lhs).strip()  # Strip LaTeX delimiters
        lhs = lhs.split()[-1] if lhs.split() else ""  # Take last token as var name

        # Clean RHS
        rhs = rhs_raw.strip()
        # Strip trailing markdown/LaTeX
        rhs = re.sub(r"[\*$\\]+$", "", rhs).strip()
        # Strip trailing comma or period
        rhs = re.sub(r"[,.]$", "", rhs).strip()

        if not lhs or not rhs:
            continue

        # Compute span: from the LHS start to the RHS end in the original text
        # Find where the actual expression starts (skip leading whitespace/labels)
        expr_text = f"{lhs} = {rhs}"
        # Use the line's position in the original text
        char_start = line_start
        char_end = line_start + len(line)

        results.append((lhs, rhs, char_start, char_end))

    return results


# ─── Per-quality scoring ───────────────────────────────────────────────────────


def _find_correct_expression(
    expressions: list[tuple[str, str, int, int]],
    ground_truth_str: str,
    lhs_patterns: list[str] | None = None,
) -> tuple[float, list[tuple[int, int]]]:
    """Search all expressions for one that matches ground truth.

    Args:
        expressions: (lhs, rhs, char_start, char_end) from _extract_all_expressions
        ground_truth_str: sympy-parseable string from training data metadata
        lhs_patterns: optional list of LHS patterns to filter by (e.g., ["T", "t"])
            If None, checks all expressions.

    Returns:
        (score, spans) where score is 1.0 if correct match found, 0.0 otherwise,
        and spans is [(char_start, char_end)] of the first correct match.
    """
    teacher = _try_parse(ground_truth_str)
    if teacher is None:
        logger.warning(f"Cannot parse ground truth: {ground_truth_str}")
        return 0.0, []

    for lhs, rhs, char_start, char_end in expressions:
        # Filter by LHS pattern if specified
        if lhs_patterns is not None:
            lhs_lower = lhs.lower().strip("*$\\")
            if not any(lhs_lower == p.lower() for p in lhs_patterns):
                continue

        student = _try_parse(rhs)
        if student is None:
            continue

        if _expressions_equivalent(student, teacher):
            return 1.0, [(char_start, char_end)]

    return 0.0, []


def _find_correct_derivative(
    expressions: list[tuple[str, str, int, int]],
    ground_truth_str: str,
    var: str,  # "q" or "p"
) -> tuple[float, list[tuple[int, int]]]:
    """Search for a correct Hamilton equation: dVAR/dt = ground_truth.

    Matches LHS patterns: dq/dt, dp/dt, \\frac{dq}{dt}, dx/dt, etc.
    """
    teacher = _try_parse(ground_truth_str)
    if teacher is None:
        logger.warning(f"Cannot parse derivative ground truth: {ground_truth_str}")
        return 0.0, []

    # Patterns that indicate a derivative of the right variable
    derivative_patterns = [
        f"d{var}/dt",
        f"d{var}",
        f"\\frac{{d{var}}}{{dt}}",
        f"frac{{d{var}}}{{dt}}",
    ]
    # Also match the coordinate name if var is "q" → might use "x", "y", etc.
    coord_aliases = {"q": ["x", "y", "r", "theta", "s"], "p": ["p"]}
    for alias in coord_aliases.get(var, []):
        derivative_patterns.extend(
            [
                f"d{alias}/dt",
                f"d{alias}",
                f"\\frac{{d{alias}}}{{dt}}",
            ]
        )

    for lhs, rhs, char_start, char_end in expressions:
        lhs_clean = _clean_for_sympy(lhs).lower()
        # Check if LHS looks like a derivative of the right variable
        if not any(p in lhs_clean for p in derivative_patterns):
            continue

        student = _try_parse(rhs)
        if student is None:
            continue

        if _expressions_equivalent(student, teacher):
            return 1.0, [(char_start, char_end)]

    return 0.0, []


def _score_consistency(
    expressions: list[tuple[str, str, int, int]],
    meta: dict,
) -> float:
    """Check if the model's dq/dt and dp/dt are consistent with its H.

    Specifically: dq/dt should equal dH/dp, and dp/dt should equal -dH/dq.
    This tests whether the model understands Hamilton's equations, not just
    whether it copied the right answer.
    """
    # Find the model's H expression
    h_expr = None
    for lhs, rhs, _, _ in expressions:
        lhs_clean = lhs.lower().strip("*$\\")
        if lhs_clean in ("h", "hamiltonian"):
            h_expr = _try_parse(rhs)
            if h_expr is not None:
                break

    if h_expr is None:
        return 0.0

    # Find the model's dq/dt and dp/dt
    dqdt_expr = None
    dpdt_expr = None
    coord = meta.get("coordinates", "x")
    # Use the SAME symbol objects that _try_parse uses (from _SYMBOL_MAP)
    # so that sp.diff finds the variable in the parsed expression.
    p = _SYMBOL_MAP.get("p", sp.Symbol("p"))
    q = _SYMBOL_MAP.get(coord, sp.Symbol(coord))

    for lhs, rhs, _, _ in expressions:
        lhs_clean = _clean_for_sympy(lhs).lower()
        if any(pat in lhs_clean for pat in [f"d{coord}/dt", "dq/dt", f"d{coord}"]):
            if "/dt" in lhs_clean or "frac" in lhs_clean:
                dqdt_expr = _try_parse(rhs)
        if any(pat in lhs_clean for pat in ["dp/dt", "dp"]):
            if "/dt" in lhs_clean or "frac" in lhs_clean:
                dpdt_expr = _try_parse(rhs)

    if dqdt_expr is None or dpdt_expr is None:
        return 0.0

    # Check: dq/dt == dH/dp and dp/dt == -dH/dq
    try:
        with sympy_timeout(3):
            expected_dqdt = sp.diff(h_expr, p)
            expected_dpdt = -sp.diff(h_expr, q)

            dqdt_ok = _expressions_equivalent(dqdt_expr, expected_dqdt)
            dpdt_ok = _expressions_equivalent(dpdt_expr, expected_dpdt)

            if dqdt_ok and dpdt_ok:
                return 1.0
            if dqdt_ok or dpdt_ok:
                return 0.5
    except (SympyTimeoutError, sp.SympifyError, ValueError, TypeError):
        pass
    return 0.0


# ─── Main reward function ──────────────────────────────────────────────────────


def hamiltonian_reward(
    prompt: str,
    completion: str,
    metadata: dict | None = None,
) -> RewardResult:
    """Score a Hamiltonian derivation by mathematical correctness only.

    No format scoring. The model can write in any format — LaTeX, ASCII,
    unicode, labeled or unlabeled. The reward function extracts ALL
    expressions and checks each against ground truth via sympy equivalence.

    Qualities:
      q_kinetic:     T expression matches ground truth
      q_potential:   V expression matches ground truth
      q_hamiltonian: H expression matches ground truth
      q_dqdt:        dq/dt equation matches ground truth
      q_dpdt:        dp/dt equation matches ground truth
      q_consistency: model's equations are consistent with its H
    """
    meta = metadata or {}
    text = completion
    scores: dict[str, float] = {}
    scored_spans: dict[str, list[tuple[int, int]]] = {}

    # Extract ALL expressions from the completion
    expressions = _extract_all_expressions(text)

    # Score each quality against ground truth
    # T (kinetic energy)
    if meta.get("T_expr"):
        score, spans = _find_correct_expression(
            expressions, meta["T_expr"], lhs_patterns=["T", "t", "kinetic"]
        )
        scores["q_kinetic"] = score
        if spans:
            scored_spans["q_kinetic"] = spans

    # V (potential energy)
    if meta.get("V_expr"):
        score, spans = _find_correct_expression(
            expressions, meta["V_expr"], lhs_patterns=["V", "v", "potential"]
        )
        scores["q_potential"] = score
        if spans:
            scored_spans["q_potential"] = spans

    # H (Hamiltonian)
    if meta.get("H_expr"):
        score, spans = _find_correct_expression(
            expressions, meta["H_expr"], lhs_patterns=["H", "h", "hamiltonian"]
        )
        scores["q_hamiltonian"] = score
        if spans:
            scored_spans["q_hamiltonian"] = spans

    # dq/dt (Hamilton's first equation)
    if meta.get("dqdt"):
        coord = meta.get("coordinates", "x")
        score, spans = _find_correct_derivative(expressions, meta["dqdt"], var=coord)
        # Also try "q" if coordinate is different
        if score == 0.0 and coord != "q":
            score, spans = _find_correct_derivative(expressions, meta["dqdt"], var="q")
        scores["q_dqdt"] = score
        if spans:
            scored_spans["q_dqdt"] = spans

    # dp/dt (Hamilton's second equation)
    if meta.get("dpdt"):
        score, spans = _find_correct_derivative(expressions, meta["dpdt"], var="p")
        scores["q_dpdt"] = score
        if spans:
            scored_spans["q_dpdt"] = spans

    # Consistency: model's own equations match its own H
    scores["q_consistency"] = _score_consistency(expressions, meta)

    # Aggregate: mean of all scored qualities (floor at 0.01 for Dr.GRPO)
    total = max(sum(scores.values()) / max(len(scores), 1), 0.01)

    return RewardResult(
        reward=total,
        scores=scores,
        scored_spans=scored_spans,
    )
