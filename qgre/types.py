"""Core types for QGRE Engine."""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class RewardResult:
    """Output of a reward function evaluation.

    The reward_fn scores completions and returns per-quality scores.
    The engine uses .scores to compute per-step advantages and manage phase gating.
    Phase is engine-managed via GameState. reward_fn should NOT set phase.
    """

    reward: float
    scores: dict = field(default_factory=dict)  # {"quality_name": float, ...}
    phase: int = 1  # Engine-managed — set by GameState, not reward_fn
    scored_spans: dict = field(default_factory=dict)
    # scored_spans: {"quality_name": [(char_start, char_end), ...], ...}
    # Character offsets into the completion text. When populated, the engine
    # uses these for per-token advantage assignment instead of the segmenter.
    # Reward functions that don't populate this field get the legacy segmenter path.


QUALITY_WINDOW_SIZE = 20


class StagnationStatus(Enum):
    NORMAL = "normal"
    STAGNATING = "stagnating"
    STUCK = "stuck"


@dataclass
class GameState:
    """QGRE 2D mastery matrix — tracks quality × difficulty independently.

    Two axes:
    - Quality phases (1→N): format → identification → equations → full derivation
    - Difficulty tiers: tier1 → tier2 → tier3 → ... (domain-specific)

    Each cell mastery[tier][phase] has its own rolling window. Quality phases
    advance per-tier. Tier N+1 unlocks when tier N reaches a configurable
    quality phase threshold.

    When no tiers are configured, all prompts map to a single "default" tier,
    making this equivalent to the original 1D phase system.
    """

    step_count: int = 0
    mastery_threshold: float = 0.8
    stagnation_timeout: int = 200
    plateau_window: int = 50
    plateau_threshold: float = 0.02

    # 2D mastery matrix
    tier_mastery: dict = field(default_factory=dict)
    # tier_mastery[tier][step_num] = deque of scores
    tier_phases: dict = field(default_factory=lambda: {"default": 1})
    # Per-tier quality phase
    active_tiers: list = field(default_factory=lambda: ["default"])
    # Currently unlocked tiers
    tier_steps_at_phase_start: dict = field(default_factory=dict)
    # tier_steps_at_phase_start[tier] = step_count when tier's current phase started
    phase_history: list = field(default_factory=list)
    # [(step_count, tier, old_phase, new_phase), ...]

    # ── 1D backward compat properties ──

    @property
    def phase(self) -> int:
        """Global phase = min phase across active tiers. For 1D compat."""
        if not self.tier_phases:
            return 1
        active = [self.tier_phases[t] for t in self.active_tiers if t in self.tier_phases]
        return min(active) if active else 1

    @property
    def step_mastery(self) -> dict:
        """Legacy: return default tier's mastery windows."""
        return self.tier_mastery.get("default", {})

    # ── 2D mastery tracking ──

    def record_tier_step_score(self, tier: str, step_num: int, score: float):
        """Record a quality score for a tier+step cell."""
        if tier not in self.tier_mastery:
            self.tier_mastery[tier] = {}
        if step_num not in self.tier_mastery[tier]:
            self.tier_mastery[tier][step_num] = deque(maxlen=QUALITY_WINDOW_SIZE)
        self.tier_mastery[tier][step_num].append(score)

    def get_tier_step_mastery(self, tier: str, step_num: int) -> float:
        """Get the mean quality score for a tier+step cell."""
        windows = self.tier_mastery.get(tier, {})
        window = windows.get(step_num)
        if not window:
            return 0.0
        return sum(window) / len(window)

    def check_tier_phase_advance(self, tier: str, max_phase: int) -> bool:
        """Check if a tier's current quality phase is mastered. Advance if so."""
        current_phase = self.tier_phases.get(tier, 1)
        if current_phase >= max_phase:
            return False

        mastery = self.get_tier_step_mastery(tier, current_phase)
        if mastery >= self.mastery_threshold:
            old_phase = current_phase
            self.tier_phases[tier] = current_phase + 1
            self.phase_history.append((self.step_count, tier, old_phase, current_phase + 1))
            self.tier_steps_at_phase_start[tier] = self.step_count
            return True
        return False

    def check_tier_unlock(
        self, tier_order: list[str], unlock_phase: int, unlock_threshold: float,
    ) -> str | None:
        """Check if the next tier should be unlocked.

        Finds the first inactive tier in tier_order. Checks if ALL active tiers
        before it have reached unlock_phase with mastery >= unlock_threshold.
        Returns the newly unlocked tier name, or None.
        """
        active_set = set(self.active_tiers)

        # Find first inactive tier in tier_order
        next_tier = None
        next_idx = -1
        for i, t in enumerate(tier_order):
            if t not in active_set:
                next_tier = t
                next_idx = i
                break

        if next_tier is None:
            return None  # All tiers already unlocked

        # All active tiers before next_tier must have reached unlock_phase with threshold
        for t in tier_order[:next_idx]:
            if t not in active_set:
                continue  # Skip tiers not in active set (shouldn't happen in ordered unlock)
            phase = self.tier_phases.get(t, 1)
            if phase < unlock_phase:
                return None  # This tier hasn't reached the required quality phase
            mastery = self.get_tier_step_mastery(t, unlock_phase)
            if mastery < unlock_threshold:
                return None  # This tier hasn't mastered the required phase

        # All active tiers before next_tier are ready — unlock it
        self.active_tiers.append(next_tier)
        self.tier_phases[next_tier] = 1
        self.tier_steps_at_phase_start[next_tier] = self.step_count
        return next_tier

    def check_tier_stagnation(self, tier: str) -> StagnationStatus:
        """Check if a specific tier is stagnating in its current phase."""
        start = self.tier_steps_at_phase_start.get(tier, 0)
        steps_in_phase = self.step_count - start
        if steps_in_phase >= self.stagnation_timeout:
            return StagnationStatus.STUCK

        current_phase = self.tier_phases.get(tier, 1)
        windows = self.tier_mastery.get(tier, {})
        window = windows.get(current_phase)
        if window and len(window) >= self.plateau_window:
            recent = list(window)[-self.plateau_window:]
            half = len(recent) // 2
            first_half_mean = sum(recent[:half]) / half
            second_half_mean = sum(recent[half:]) / (len(recent) - half)
            if abs(second_half_mean - first_half_mean) < self.plateau_threshold:
                return StagnationStatus.STAGNATING

        return StagnationStatus.NORMAL

    # ── 1D compat methods (delegate to "default" tier) ──

    def record_step_score(self, step_num: int, score: float):
        """Legacy 1D: record to default tier."""
        self.record_tier_step_score("default", step_num, score)

    def get_step_mastery(self, step_num: int) -> float:
        """Legacy 1D: read from default tier."""
        return self.get_tier_step_mastery("default", step_num)

    def check_phase_advance(self, max_phase: int) -> bool:
        """Legacy 1D: advance default tier's phase."""
        return self.check_tier_phase_advance("default", max_phase)

    def check_stagnation(self) -> StagnationStatus:
        """Legacy 1D: check default tier stagnation."""
        return self.check_tier_stagnation("default")
