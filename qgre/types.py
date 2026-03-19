"""Core types for QGRE Engine."""

from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class RewardResult:
    """Output of a reward function evaluation.

    Every reward_fn must return this type. The engine consumes .scores
    for per-step advantage computation and .phase for curriculum gating.
    """

    reward: float
    scores: dict = field(default_factory=dict)  # {"quality_name": float, ...}
    phase: int = 1


# Default window size for quality tracking
QUALITY_WINDOW_SIZE = 20


@dataclass
class GameState:
    """QGRE curriculum state — tracks mastery, phases, and quality windows.

    Contains types that don't serialize naively (deque, defaultdict).
    Use to_dict() / from_dict() for checkpoint persistence.
    """

    phase: int = 1
    max_active_tier: int = 1
    step_count: int = 0
    quality_windows: dict = field(default_factory=dict)
    # {archetype: {quality_name: deque([float, ...], maxlen=QUALITY_WINDOW_SIZE)}}
    elo_ratings: defaultdict = field(default_factory=lambda: defaultdict(lambda: 1500.0))
    mastery_counts: defaultdict = field(default_factory=lambda: defaultdict(int))
    tier_history: list = field(default_factory=list)

    def add_quality_score(self, archetype: str, quality: str, score: float):
        if archetype not in self.quality_windows:
            self.quality_windows[archetype] = {}
        if quality not in self.quality_windows[archetype]:
            self.quality_windows[archetype][quality] = deque(maxlen=QUALITY_WINDOW_SIZE)
        self.quality_windows[archetype][quality].append(score)

    def get_quality_mean(self, archetype: str, quality: str) -> float:
        window = self.quality_windows.get(archetype, {}).get(quality)
        if not window:
            return 0.0
        return sum(window) / len(window)
