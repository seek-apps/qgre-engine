"""Tests for GameState serialization (Step 0a) and checkpoint resume (Step 0f)."""

import json
import tempfile
from collections import defaultdict, deque
from pathlib import Path

import pytest

from qgre.checkpoint import (
    discover_latest_checkpoint,
    gamestate_from_dict,
    gamestate_to_dict,
    load_checkpoint,
    save_checkpoint,
)
from qgre.types import QUALITY_WINDOW_SIZE, GameState


# --- Step 0a: GameState Serializer ---


def test_gamestate_roundtrip(mock_game_state):
    """to_dict() → from_dict() → all fields equal."""
    d = gamestate_to_dict(mock_game_state)
    restored = gamestate_from_dict(d)

    assert restored.phase == mock_game_state.phase
    assert restored.max_active_tier == mock_game_state.max_active_tier
    assert restored.step_count == mock_game_state.step_count
    assert restored.tier_history == mock_game_state.tier_history

    # Elo ratings
    assert dict(restored.elo_ratings) == dict(mock_game_state.elo_ratings)

    # Mastery counts
    assert dict(restored.mastery_counts) == dict(mock_game_state.mastery_counts)

    # Quality windows — check values match
    for arch in mock_game_state.quality_windows:
        assert arch in restored.quality_windows
        for q_name in mock_game_state.quality_windows[arch]:
            assert q_name in restored.quality_windows[arch]
            orig = list(mock_game_state.quality_windows[arch][q_name])
            rest = list(restored.quality_windows[arch][q_name])
            assert orig == rest


def test_gamestate_json_serializable(mock_game_state):
    """to_dict() output passes json.dumps without error."""
    d = gamestate_to_dict(mock_game_state)
    result = json.dumps(d)
    assert isinstance(result, str)
    assert len(result) > 10

    # Verify round-trip through JSON
    parsed = json.loads(result)
    restored = gamestate_from_dict(parsed)
    assert restored.phase == mock_game_state.phase


def test_gamestate_preserves_deque_maxlen(mock_game_state):
    """After round-trip, deque.maxlen matches original."""
    d = gamestate_to_dict(mock_game_state)
    restored = gamestate_from_dict(d)

    for arch in restored.quality_windows:
        for q_name, dq in restored.quality_windows[arch].items():
            assert isinstance(dq, deque)
            assert dq.maxlen == QUALITY_WINDOW_SIZE


def test_gamestate_preserves_defaultdict_behavior():
    """After from_dict(), accessing missing key returns default, not KeyError."""
    gs = GameState()
    gs.elo_ratings["existing"] = 1600.0
    d = gamestate_to_dict(gs)
    restored = gamestate_from_dict(d)

    # Existing key preserved
    assert restored.elo_ratings["existing"] == 1600.0

    # Missing key returns default (1500.0 for elo, 0 for mastery)
    assert restored.elo_ratings["never_seen"] == 1500.0
    assert restored.mastery_counts["never_seen"] == 0


def test_gamestate_empty_roundtrip():
    """Empty GameState survives round-trip."""
    gs = GameState()
    d = gamestate_to_dict(gs)
    restored = gamestate_from_dict(d)

    assert restored.phase == 1
    assert restored.max_active_tier == 1
    assert restored.step_count == 0
    assert restored.quality_windows == {}
    assert restored.tier_history == []


# --- Step 0f: Checkpoint Resume ---


def test_checkpoint_save_load_roundtrip(mock_game_state):
    """Save full state dict → load → all fields match."""
    import torch

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "global_step_50.pt"

        rng_state = torch.get_rng_state()
        save_checkpoint(
            path=path,
            global_step=50,
            model_state_dict={"weight": torch.randn(4, 4)},
            optimizer_state_dict={"lr": 5e-6},
            game_state=mock_game_state,
            advantage_estimator_state={"V": {1: {1: 0.5}}},
            rng_state=rng_state,
        )

        loaded = load_checkpoint(path)
        assert loaded["global_step"] == 50
        assert isinstance(loaded["game_state"], GameState)
        assert loaded["game_state"].phase == mock_game_state.phase
        assert loaded["advantage_estimator_state"] == {"V": {1: {1: 0.5}}}
        assert torch.equal(loaded["rng_state"], rng_state)


def test_checkpoint_discovery_finds_latest():
    """Create dir with global_step_10, _50, _30 → returns _50."""
    import torch

    with tempfile.TemporaryDirectory() as tmpdir:
        for step in [10, 50, 30]:
            path = Path(tmpdir) / f"global_step_{step}.pt"
            torch.save({"global_step": step}, path)

        latest = discover_latest_checkpoint(tmpdir)
        assert latest is not None
        assert "global_step_50" in latest.name


def test_checkpoint_discovery_empty_dir():
    """Empty dir → returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        assert discover_latest_checkpoint(tmpdir) is None


def test_checkpoint_discovery_nonexistent_dir():
    """Nonexistent dir → returns None."""
    assert discover_latest_checkpoint("/tmp/nonexistent_qgre_test_dir") is None


def test_checkpoint_rng_state_restored():
    """Save RNG → generate random → restore RNG → same random sequence."""
    import torch

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "global_step_1.pt"
        rng_state = torch.get_rng_state()
        save_checkpoint(path=path, global_step=1, rng_state=rng_state)

        # Generate some random numbers
        seq1 = torch.randn(5)

        # Restore RNG state
        loaded = load_checkpoint(path)
        torch.set_rng_state(loaded["rng_state"])

        # Generate again — should match
        seq2 = torch.randn(5)
        assert torch.equal(seq1, seq2)


def test_checkpoint_includes_advantage_estimator_state(mock_game_state):
    """V tracker and _step_seen persist through save/load."""
    import torch

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "global_step_100.pt"

        adv_state = {
            "V": {42: {1: 0.85, 2: 0.6, 3: 0.0, 4: 0.3}},
            "step_seen": {42: [1, 2, 4]},
        }

        save_checkpoint(
            path=path,
            global_step=100,
            game_state=mock_game_state,
            advantage_estimator_state=adv_state,
        )

        loaded = load_checkpoint(path)
        assert loaded["advantage_estimator_state"]["V"][42][1] == 0.85
        assert loaded["advantage_estimator_state"]["step_seen"][42] == [1, 2, 4]


# --- Step 5: Trainer checkpoint wiring ---


def test_trainer_save_load_step_counter(mock_game_state):
    """Train 3 steps → save → new trainer → load → step counter == 3."""
    import torch.nn as nn
    from unittest.mock import MagicMock
    from qgre.config import QGREConfig
    from qgre.trainer import QGRETrainer

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
        def forward(self, x):
            return MagicMock(logits=self.linear(torch.randn(1, 4).expand(x.shape[0], -1).unsqueeze(1).expand(-1, x.shape[1], -1)))

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = QGREConfig()
        cfg.logging.checkpoint_dir = tmpdir
        cfg.logging.completion_dir = str(Path(tmpdir) / "completions")

        model = TinyModel()
        trainer = QGRETrainer(
            model=model, tokenizer=None,
            reward_fn=lambda *a: None, config=cfg,
            game_state=mock_game_state,
        )
        trainer.setup_optimizer()
        trainer.global_step = 3
        trainer.save()

        # New trainer, load checkpoint
        model2 = TinyModel()
        trainer2 = QGRETrainer(
            model=model2, tokenizer=None,
            reward_fn=lambda *a: None, config=cfg,
        )
        trainer2.setup_optimizer()
        resumed = trainer2.resume(tmpdir)

        assert resumed is True
        assert trainer2.global_step == 3
        assert trainer2.game_state.phase == mock_game_state.phase


def test_v_tracker_persists_across_checkpoint():
    """SPO V tracker values present after save/load cycle."""
    import torch.nn as nn
    from unittest.mock import MagicMock
    from qgre.config import QGREConfig
    from qgre.trainer import QGRETrainer

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
        def forward(self, x):
            return MagicMock(logits=self.linear(torch.randn(1, 4)))

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = QGREConfig()
        cfg.logging.checkpoint_dir = tmpdir
        cfg.logging.completion_dir = str(Path(tmpdir) / "completions")

        model = TinyModel()
        trainer = QGRETrainer(
            model=model, tokenizer=None,
            reward_fn=lambda *a: None, config=cfg,
        )
        trainer.setup_optimizer()

        # Set some V tracker state
        trainer.advantage_estimator.V[42][1] = 0.85
        trainer.advantage_estimator.V[42][2] = 0.6
        trainer.advantage_estimator._step_seen[42] = {1, 2}
        trainer.global_step = 10
        trainer.save()

        # Restore
        model2 = TinyModel()
        trainer2 = QGRETrainer(
            model=model2, tokenizer=None,
            reward_fn=lambda *a: None, config=cfg,
        )
        trainer2.setup_optimizer()
        trainer2.resume(tmpdir)

        assert trainer2.advantage_estimator.V[42][1] == 0.85
        assert trainer2.advantage_estimator.V[42][2] == 0.6
        assert 1 in trainer2.advantage_estimator._step_seen[42]
