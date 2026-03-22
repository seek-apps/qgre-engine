"""4-tier progressive reward function for Hamiltonian mechanics training.

Phase 1: q_format, q_has_math — substantial response with equations
Phase 2: q_identifies_T, q_identifies_V, q_grounding — physics identification + parameter grounding
Phase 3: q_correct_dqdt, q_correct_dpdt — Hamilton's equations match sympy ground truth
Phase 4: q_correct_H, q_consistency — full Hamiltonian + internal consistency

Sympy-based scoring for phases 3-4. String matching fallback when sympy parsing fails.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import sympy as sp

from qgre.types import RewardResult

logger = logging.getLogger("qgre.hamiltonian_reward")

# Diagnostic log file — one JSON line per scored completion
_DIAG_PATH = Path("output/hamiltonian/diagnostics.jsonl")


# ─── Expression extraction and normalization ─────────────────────────────────

def _normalize_text(s: str) -> str:
    """Normalize text for fuzzy matching."""
    s = s.lower()
    s = s.replace("**", "^").replace("\\frac", "").replace("\\cdot", "*")
    s = s.replace("·", "*").replace("×", "*").replace(" ", "")
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"(\d)\*([a-z])", r"\1\2", s)
    return s


def _normalize_for_sympy(expr_str: str) -> str:
    """Clean up expression string for sympy parsing."""
    s = expr_str.strip()
    # Common substitutions
    s = s.replace("^", "**")
    s = s.replace("²", "**2").replace("³", "**3").replace("⁴", "**4")
    s = s.replace("θ", "theta").replace("ω", "omega")
    s = s.replace("p_θ", "p_theta").replace("p_r", "p_r").replace("p_s", "p_s")
    # Remove LaTeX fractions
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
    # Remove other LaTeX commands
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    # Implicit multiplication: 2x → 2*x, but not inside function names
    s = re.sub(r"(\d)([a-zA-Z_])", r"\1*\2", s)
    return s


_EQUATION_PATTERNS = [
    # dq/dt = ... or dx/dt = ... or dtheta/dt = ...
    r"d[a-z_]+/dt\s*=\s*([^\n;,]+)",
    # ∂H/∂p = ... or ∂H/∂p_theta = ...
    r"[∂d]H/[∂d][a-z_]+\s*=\s*([^\n;,]+)",
    # q̇ = ... or ẋ = ... or θ̇ = ...
    r"[a-zθ]̇\s*=\s*([^\n;,]+)",
]

_H_PATTERNS = [
    r"H\s*[=(]\s*([^\n;]+)",
    r"[Hh]amiltonian[:\s]*=?\s*([^\n;]+)",
]

_T_PATTERNS = [
    r"T\s*=\s*([^\n;]+)",
    r"[Kk]inetic\s*(?:energy)?\s*[=:]\s*([^\n;]+)",
]

_V_PATTERNS = [
    r"V\s*=\s*([^\n;]+)",
    r"[Pp]otential\s*(?:energy)?\s*[=:]\s*([^\n;]+)",
]

# ─── Structured section extractors ───────────────────────────────────────────
# These match the labeled format from the system prompt:
#   HAMILTONIAN: H = ...
#   KINETIC ENERGY: T = ...
#   EQUATIONS:\n  dq/dt = ...\n  dp/dt = ...

_SECTION_RE = {
    "H": re.compile(r"HAMILTONIAN:\s*H\s*=\s*([^\n]+)", re.IGNORECASE),
    "T": re.compile(r"KINETIC ENERGY:\s*T\s*=\s*([^\n]+)", re.IGNORECASE),
    "V": re.compile(r"POTENTIAL ENERGY:\s*V\s*=\s*([^\n]+)", re.IGNORECASE),
}

_EQUATIONS_BLOCK_RE = re.compile(
    r"EQUATIONS:\s*\n((?:\s+d[^\n]+\n?)+)", re.IGNORECASE
)


def _extract_section(text: str, key: str) -> str | None:
    """Extract from structured section label first, fall back to regex patterns."""
    # Try structured label
    m = _SECTION_RE.get(key)
    if m:
        hit = m.search(text)
        if hit:
            expr = hit.group(1).strip()
            # If there's a trailing "= simplified", take the last part after final "="
            parts = expr.rsplit("=", 1)
            return parts[-1].strip() if len(parts) > 1 else expr
    return None


def _extract_first(text: str, patterns: list[str], section_key: str | None = None) -> str | None:
    """Extract expression: try structured section first, then regex patterns."""
    if section_key:
        result = _extract_section(text, section_key)
        if result:
            return result
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            expr = m.group(1).strip()
            expr = re.split(r"[,\.](?:\s|$)|where|since|because|note|which", expr, maxsplit=1)[0]
            return expr.strip()
    return None


def _extract_all_equations(text: str) -> list[str]:
    """Extract equation RHS values: try EQUATIONS: block first, then regex."""
    results = []
    # Try structured EQUATIONS: block
    block_m = _EQUATIONS_BLOCK_RE.search(text)
    if block_m:
        block = block_m.group(1)
        for line in block.strip().split("\n"):
            eq_m = re.search(r"=\s*([^\n;,]+)", line)
            if eq_m:
                results.append(eq_m.group(1).strip())
        if results:
            return results
    # Fall back to scattered regex patterns
    for pat in _EQUATION_PATTERNS:
        for m in re.finditer(pat, text):
            results.append(m.group(1).strip())
    return results


# ─── Sympy-based scoring ────────────────────────────────────────────────────

# Common symbols for sympy parsing
_SYMPY_LOCALS = {
    "x": sp.Symbol("x"), "y": sp.Symbol("y"), "r": sp.Symbol("r", positive=True),
    "s": sp.Symbol("s"), "q": sp.Symbol("q"),
    "p": sp.Symbol("p"), "p_x": sp.Symbol("p_x"), "p_y": sp.Symbol("p_y"),
    "p_r": sp.Symbol("p_r", positive=True), "p_s": sp.Symbol("p_s"),
    "p_theta": sp.Symbol("p_theta"), "p1": sp.Symbol("p1"), "p2": sp.Symbol("p2"),
    "theta": sp.Symbol("theta"), "theta1": sp.Symbol("theta1"), "theta2": sp.Symbol("theta2"),
    "x1": sp.Symbol("x1"), "x2": sp.Symbol("x2"),
    "m": sp.Symbol("m", positive=True), "g": sp.Rational(98, 10),
    "pi": sp.pi,
}


def _try_sympify(expr_str: str) -> sp.Expr | None:
    """Try to parse a string as a sympy expression. Return None on failure."""
    try:
        cleaned = _normalize_for_sympy(expr_str)
        return sp.sympify(cleaned, locals=_SYMPY_LOCALS)
    except Exception:
        return None


def _score_expression(completion_expr_str: str | None, ground_truth_str: str,
                      variables: list[str]) -> float:
    """Score a mathematical expression against ground truth using sympy.

    Returns 0.0-1.0:
    - 1.0: exact symbolic match
    - 0.5: partial credit (key structural elements present)
    - 0.2: attempted but unparseable
    - 0.0: not found
    """
    if completion_expr_str is None:
        return 0.0

    student = _try_sympify(completion_expr_str)
    teacher = _try_sympify(ground_truth_str)

    if teacher is None:
        # Ground truth unparseable — fall back to string matching
        return _string_similarity(completion_expr_str, ground_truth_str)

    if student is None:
        # Student wrote something but it's unparseable
        return 0.2

    # Guard against sympify returning bool/non-Expr types
    if not isinstance(student, sp.Basic) or isinstance(student, sp.logic.boolalg.BooleanAtom):
        return 0.2
    if not isinstance(teacher, sp.Basic) or isinstance(teacher, sp.logic.boolalg.BooleanAtom):
        return _string_similarity(completion_expr_str, ground_truth_str)

    # Exact symbolic match — try multiple simplification strategies
    for simplifier in [sp.simplify, sp.trigsimp, sp.ratsimp, sp.nsimplify]:
        try:
            diff = simplifier(student - teacher)
            if diff == 0:
                return 1.0
        except Exception:
            continue

    # Try expanding both
    try:
        diff = sp.simplify(sp.expand(student) - sp.expand(teacher))
        if diff == 0:
            return 1.0
    except Exception:
        pass

    # Try expanding with trig
    try:
        diff = sp.trigsimp(sp.expand_trig(student - teacher))
        if diff == 0:
            return 1.0
    except Exception:
        pass

    # Numerical equivalence check — evaluate at random points
    try:
        free = (student.free_symbols | teacher.free_symbols) - {sp.Symbol('pi')}
        if free:
            test_point = {s: sp.Rational(3, 7) for s in free}
            s_val = float(student.subs(test_point))
            t_val = float(teacher.subs(test_point))
            if abs(s_val - t_val) < 1e-6 * max(abs(t_val), 1):
                # Verify at a second point
                test_point2 = {s: sp.Rational(5, 11) for s in free}
                s_val2 = float(student.subs(test_point2))
                t_val2 = float(teacher.subs(test_point2))
                if abs(s_val2 - t_val2) < 1e-6 * max(abs(t_val2), 1):
                    return 1.0
    except Exception:
        pass

    # Partial credit: use numerical closeness, not just symbol overlap
    # Wrong coefficients (-3x vs -6x) must score much lower than correct answers
    try:
        # Numerical ratio check: how close is the student's value to teacher's?
        free = (student.free_symbols | teacher.free_symbols) - {sp.Symbol('pi')}
        numerical_score = 0.0
        if free:
            test_point = {s: sp.Rational(3, 7) for s in free}
            try:
                s_val = float(student.subs(test_point))
                t_val = float(teacher.subs(test_point))
                if abs(t_val) > 1e-10:
                    ratio = s_val / t_val
                    # Perfect = 1.0, half-off = 0.5, wrong sign = 0.0
                    numerical_score = max(0.0, 1.0 - abs(1.0 - ratio))
                elif abs(s_val) < 1e-10:
                    numerical_score = 1.0  # Both near zero
            except Exception:
                pass

        # Symbol overlap (secondary signal)
        student_syms = student.free_symbols
        teacher_syms = teacher.free_symbols
        if teacher_syms:
            sym_overlap = len(student_syms & teacher_syms) / len(teacher_syms)
        else:
            sym_overlap = 1.0

        # Weight numerical closeness heavily — it catches coefficient errors
        partial = 0.2 + 0.5 * numerical_score + 0.2 * sym_overlap
        return min(partial, 0.85)
    except Exception:
        return 0.2


def _string_similarity(a: str, b: str) -> float:
    """Simple string similarity for fallback."""
    na = _normalize_text(a)
    nb = _normalize_text(b)
    if na == nb:
        return 1.0
    if nb in na or na in nb:
        return 0.7

    # Token overlap
    tokens_a = set(re.findall(r"[a-z_]+|\d+", na))
    tokens_b = set(re.findall(r"[a-z_]+|\d+", nb))
    if not tokens_b:
        return 0.0
    overlap = len(tokens_a & tokens_b) / len(tokens_b)
    return 0.3 * overlap


# ─── Grounding check ────────────────────────────────────────────────────────

def _extract_numbers_from_prompt(prompt: str) -> set[str]:
    """Extract numerical values from the prompt (mass, spring constant, etc.)."""
    # Find numbers that are parameter values (preceded by = or common physics words)
    nums = set()
    # Pattern: "m = 2", "k = 4", "L = 1", etc.
    for m in re.finditer(r"[=]\s*(\d+(?:\.\d+)?)", prompt):
        nums.add(m.group(1))
    # Also grab standalone numbers after common physics words
    for m in re.finditer(r"(?:mass|constant|length|radius|velocity|charge|field)\s+.*?(\d+(?:\.\d+)?)", prompt):
        nums.add(m.group(1))
    return nums


def _check_grounding(completion: str, prompt: str) -> float:
    """Check that the completion uses actual numerical values from the prompt.

    Returns 0.0-1.0. The QGRE insight: model must ground in the input,
    not hallucinate plausible physics.
    """
    prompt_nums = _extract_numbers_from_prompt(prompt)
    if not prompt_nums:
        return 1.0  # No numbers to check

    # Filter out very common numbers that aren't parameter-specific
    specific_nums = {n for n in prompt_nums if n not in ("0", "1", "2")}
    if not specific_nums:
        return 1.0  # Only common numbers — can't meaningfully check

    found = sum(1 for n in specific_nums if n in completion)
    return found / len(specific_nums)


# ─── Quality scorers ────────────────────────────────────────────────────────

def _has_any(text: str, patterns: list[str]) -> bool:
    t = text.lower()
    return any(p.lower() in t for p in patterns)


def _count_matches(text: str, patterns: list[str]) -> int:
    t = text.lower()
    return sum(1 for p in patterns if p.lower() in t)


def _score_format(text: str) -> float:
    """q_format: substantial physics response."""
    has_length = len(text.strip()) > 100
    has_math = _has_any(text, ["=", "hamiltonian", "H ", "H=", "kinetic", "potential"])
    has_structure = _has_any(text, [
        "dq/dt", "dp/dt", "dH/dp", "dH/dq", "∂H/∂p", "∂H/∂q",
        "hamilton", "equation of motion", "dx/dt", "dtheta/dt",
    ])
    if has_length and has_math and has_structure:
        return 1.0
    if has_length and has_math:
        return 0.7
    if has_length:
        return 0.3
    return 0.0


def _score_has_math(text: str) -> float:
    """q_has_math: has mathematical content with equations."""
    indicators = [
        "=", "**2", "^2", "²", "/2", "p^2", "p²", "p**2",
        "cos(", "sin(", "exp(", "sqrt(",
        "H(", "H =", "T =", "V =",
    ]
    count = _count_matches(text, indicators)
    return min(1.0, count / 3)


def _score_identifies_T(text: str) -> float:
    """q_identifies_T: identifies kinetic energy term."""
    patterns = [
        "kinetic energy", "T =", "T=",
        "p^2/", "p²/", "p**2/", "p_r^2", "p_theta^2",
        "p₁", "p₂", "½mv²", "(1/2)mv", "p_s^2", "p_x^2", "p_y^2",
    ]
    return min(1.0, _count_matches(text, patterns) / 2)


def _score_identifies_V(text: str) -> float:
    """q_identifies_V: identifies potential energy term."""
    patterns = [
        "potential energy", "V =", "V=", "V(",
        "mgh", "mgL", "mgl", "-mgL",
        "kx²", "kx^2", "(1/2)k",
        "cos(θ", "cos(theta", "-cos(",
        "/r^", "/r²", "-α/r", "-G",
    ]
    return min(1.0, _count_matches(text, patterns) / 2)


def _score_dqdt(text: str, meta: dict) -> float:
    """q_correct_dqdt: Hamilton's first equation matches ground truth."""
    expected = meta.get("dqdt", "")
    if not expected or expected == "none":
        return 0.0

    # For multi-DOF, split by semicolon and score each
    expected_parts = [e.strip() for e in expected.split(";")]
    extracted = _extract_all_equations(text)

    if not extracted:
        # Check if at least mentions the equation form
        if _has_any(text, ["dq/dt", "∂H/∂p", "dx/dt", "dtheta/dt", "dr/dt", "ds/dt"]):
            return 0.2
        return 0.0

    # Try to match each expected equation against extracted ones
    scores = []
    for exp_part in expected_parts:
        best = 0.0
        for ext in extracted:
            score = _score_expression(ext, exp_part, [])
            best = max(best, score)
        # Also try direct string matching against full text
        str_score = _string_similarity(text, exp_part)
        best = max(best, str_score)
        scores.append(best)

    return sum(scores) / len(scores) if scores else 0.0


def _score_dpdt(text: str, meta: dict) -> float:
    """q_correct_dpdt: Hamilton's second equation matches ground truth."""
    expected = meta.get("dpdt", "")
    if not expected or expected == "none":
        return 0.0

    expected_parts = [e.strip() for e in expected.split(";")]
    extracted = _extract_all_equations(text)

    if not extracted:
        if _has_any(text, ["dp/dt", "-∂H/∂q", "-dH/dq", "dp_r/dt", "dp_theta/dt"]):
            return 0.2
        return 0.0

    scores = []
    for exp_part in expected_parts:
        best = 0.0
        for ext in extracted:
            score = _score_expression(ext, exp_part, [])
            best = max(best, score)
        str_score = _string_similarity(text, exp_part)
        best = max(best, str_score)
        scores.append(best)

    return sum(scores) / len(scores) if scores else 0.0


def _score_correct_H(text: str, meta: dict) -> float:
    """q_correct_H: full Hamiltonian matches ground truth."""
    expected_H = meta.get("H_expr", "")
    if not expected_H or expected_H == "none":
        return 0.0

    extracted_H = _extract_first(text, _H_PATTERNS, section_key="H")
    return _score_expression(extracted_H, expected_H, [])


def _score_consistency(text: str, meta: dict) -> float:
    """q_consistency: internal consistency — stated H gives stated equations.

    Extracts H from completion, computes dH/dp and -dH/dq, checks they
    match the equations the model wrote.
    """
    extracted_H = _extract_first(text, _H_PATTERNS, section_key="H")
    if extracted_H is None:
        return 0.0

    H_sym = _try_sympify(extracted_H)
    if H_sym is None or not isinstance(H_sym, sp.Expr):
        return 0.2  # Wrote an H but it's unparseable

    # Extract coordinate info from metadata
    coords = meta.get("coordinates", "x")
    coord_list = [c.strip() for c in coords.split(",")]

    # Try to check at least one equation for consistency
    extracted_eqs = _extract_all_equations(text)
    if not extracted_eqs:
        return 0.3  # Has H but no equations

    # For each coordinate, compute dH/dp and check against extracted equations
    consistency_scores = []
    for coord in coord_list:
        p_name = f"p_{coord}" if coord not in ("x", "y", "s", "q") else "p"
        if coord in ("x1", "x2"):
            p_name = coord.replace("x", "p")

        p_sym = _SYMPY_LOCALS.get(p_name) or sp.Symbol(p_name)
        q_sym = _SYMPY_LOCALS.get(coord) or sp.Symbol(coord)

        try:
            expected_dqdt = sp.diff(H_sym, p_sym)
            expected_dpdt = -sp.diff(H_sym, q_sym)

            # Check if any extracted equation matches
            for eq_str in extracted_eqs:
                eq_sym = _try_sympify(eq_str)
                if eq_sym is not None:
                    try:
                        if sp.simplify(eq_sym - expected_dqdt) == 0:
                            consistency_scores.append(1.0)
                            break
                        if sp.simplify(eq_sym - expected_dpdt) == 0:
                            consistency_scores.append(1.0)
                            break
                    except Exception:
                        pass
            else:
                consistency_scores.append(0.3)
        except Exception:
            consistency_scores.append(0.2)

    return sum(consistency_scores) / max(len(consistency_scores), 1)


# ─── Main reward function ───────────────────────────────────────────────────

def hamiltonian_reward(
    prompt: str,
    completion: str,
    metadata: dict | None = None,
) -> RewardResult:
    """Score a Hamiltonian mechanics derivation with 4-tier progressive qualities.

    Phase 1: q_format, q_has_math
    Phase 2: q_identifies_T, q_identifies_V, q_grounding
    Phase 3: q_correct_dqdt, q_correct_dpdt
    Phase 4: q_correct_H, q_consistency
    """
    meta = metadata or {}
    text = completion
    scores: dict[str, float] = {}

    # ── Phase 1: Format ──
    scores["q_format"] = _score_format(text)
    scores["q_has_math"] = _score_has_math(text)

    # ── Phase 2: Physics identification + grounding ──
    scores["q_identifies_T"] = _score_identifies_T(text)
    scores["q_identifies_V"] = _score_identifies_V(text)
    scores["q_grounding"] = _check_grounding(text, prompt)

    # ── Phase 3: Equation correctness (sympy) ──
    scores["q_correct_dqdt"] = _score_dqdt(text, meta)
    scores["q_correct_dpdt"] = _score_dpdt(text, meta)

    # ── Phase 4: Full Hamiltonian + consistency ──
    scores["q_correct_H"] = _score_correct_H(text, meta)
    scores["q_consistency"] = _score_consistency(text, meta)

    total = sum(scores.values()) / max(len(scores), 1)

    # ── Diagnostic logging ──
    try:
        _DIAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        diag = {
            "system": meta.get("system", "unknown"),
            "difficulty": meta.get("difficulty", "unknown"),
            "scores": {k: round(v, 3) for k, v in scores.items()},
            "total": round(total, 3),
            "completion_len": len(text),
        }
        with open(_DIAG_PATH, "a") as f:
            f.write(json.dumps(diag) + "\n")
    except Exception:
        pass  # Never let diagnostics crash training

    return RewardResult(reward=total, scores=scores)
