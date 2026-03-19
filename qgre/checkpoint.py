from __future__ import annotations

import re
from collections import defaultdict, deque
from pathlib import Path

from qgre.types import QUALITY_WINDOW_SIZE, GameState


def gamestate_to_dict(gs: GameState) -> dict:
    """Serialize GameState to a plain dict safe for json.dumps and torch.save.

    Converts: deque → list, defaultdict → dict, preserves maxlen metadata.
    """
    # quality_windows: {archetype: {quality: deque}} → {archetype: {quality: list}}
    qw = {}
    for arch, qualities in gs.quality_windows.items():
        qw[arch] = {}
        for q_name, dq in qualities.items():
            qw[arch][q_name] = {
                "values": list(dq),
                "maxlen": dq.maxlen,
            }

    return {
        "phase": gs.phase,
        "max_active_tier": gs.max_active_tier,
        "step_count": gs.step_count,
        "quality_windows": qw,
        "elo_ratings": dict(gs.elo_ratings),
        "mastery_counts": dict(gs.mastery_counts),
        "tier_history": list(gs.tier_history),
    }


def gamestate_from_dict(d: dict) -> GameState:
    """Reconstruct GameState from a plain dict.

    Restores: list → deque (with maxlen), dict → defaultdict.
    """
    gs = GameState()
    gs.phase = d.get("phase", 1)
    gs.max_active_tier = d.get("max_active_tier", 1)
    gs.step_count = d.get("step_count", 0)
    gs.tier_history = list(d.get("tier_history", []))

    # Restore quality_windows with deque maxlen
    qw = d.get("quality_windows", {})
    gs.quality_windows = {}
    for arch, qualities in qw.items():
        gs.quality_windows[arch] = {}
        for q_name, window_data in qualities.items():
            maxlen = window_data.get("maxlen", QUALITY_WINDOW_SIZE)
            values = window_data.get("values", [])
            gs.quality_windows[arch][q_name] = deque(values, maxlen=maxlen)

    # Restore defaultdicts with correct factories
    elo = d.get("elo_ratings", {})
    gs.elo_ratings = defaultdict(lambda: 1500.0, {k: float(v) for k, v in elo.items()})

    mastery = d.get("mastery_counts", {})
    gs.mastery_counts = defaultdict(int, {k: int(v) for k, v in mastery.items()})

    return gs


def save_checkpoint(
    path: str | Path,
    global_step: int,
    model_state_dict: dict | None = None,
    optimizer_state_dict: dict | None = None,
    scheduler_state_dict: dict | None = None,
    game_state: GameState | None = None,
    advantage_estimator_state: dict | None = None,
    rng_state=None,
    cuda_rng_state=None,
):
    """Save full training state to a checkpoint file."""
    import torch

    checkpoint = {
        "global_step": global_step,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "game_state": gamestate_to_dict(game_state) if game_state else None,
        "advantage_estimator_state": advantage_estimator_state,
        "rng_state": rng_state if rng_state is not None else torch.get_rng_state(),
        "cuda_rng_state": cuda_rng_state,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path) -> dict:
    """Load checkpoint from file. Returns raw dict — caller restores state."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("game_state") is not None:
        checkpoint["game_state"] = gamestate_from_dict(checkpoint["game_state"])
    return checkpoint


def discover_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Scan directory for global_step_N checkpoints, return path to latest."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    pattern = re.compile(r"global_step_(\d+)")
    candidates = []
    for entry in checkpoint_dir.iterdir():
        match = pattern.search(entry.name)
        if match:
            candidates.append((int(match.group(1)), entry))

    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]
