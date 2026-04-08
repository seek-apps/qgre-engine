"""Tests for unified SyncState."""

import pytest


class TestSyncStateTransitions:
    """Test state transition logic."""

    def test_enter_dropout_sets_flag(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        assert not state.dropout_active
        state.enter_dropout()
        assert state.dropout_active

    def test_exit_dropout_clears_flag(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.enter_dropout()
        state.exit_dropout(success=True)
        assert not state.dropout_active
        assert not state.restore_failed

    def test_exit_dropout_failure_sets_restore_failed(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.enter_dropout()
        state.exit_dropout(success=False)
        assert not state.dropout_active
        assert state.restore_failed

    def test_enter_dropout_after_failure_raises(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.restore_failed = True
        with pytest.raises(RuntimeError, match="Weights are corrupted"):
            state.enter_dropout()

    def test_successful_restore_resets_restore_failed(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.restore_failed = True
        # Manually clear to simulate recovery
        state.restore_failed = False
        state.enter_dropout()
        state.exit_dropout(success=True)
        assert not state.restore_failed

    def test_double_dropout_warns(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.enter_dropout()
        # FIX 5: Double-entry now raises instead of warns
        with pytest.raises(RuntimeError, match="already active"):
            state.enter_dropout()


class TestSyncStateCanSync:
    """Test sync precondition checks."""

    def test_can_sync_when_clean(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        assert state.can_sync()

    def test_cannot_sync_when_dropout_active(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.enter_dropout()
        assert not state.can_sync()

    def test_cannot_sync_when_cache_stale(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.mark_cache_stale()
        assert not state.can_sync()

    def test_check_sync_allowed_raises_on_stale(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.mark_cache_stale()
        with pytest.raises(RuntimeError, match="KV cache is potentially stale"):
            state.check_sync_allowed()


class TestSyncStateLifecycle:
    """Test lifecycle transitions."""

    def test_begin_sync_transitions_to_loading(self):
        from qgre.sync_state import SyncLifecycle, SyncState

        state = SyncState()
        state.begin_sync()
        assert state.lifecycle == SyncLifecycle.LOADING

    def test_complete_sync_transitions_to_ready(self):
        from qgre.sync_state import SyncLifecycle, SyncState

        state = SyncState()
        state.begin_sync()
        state.complete_sync()
        assert state.lifecycle == SyncLifecycle.READY

    def test_complete_sync_first_call_sets_initialized(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        assert not state.initialized
        state.begin_sync()
        state.complete_sync(first_call=True)
        assert state.initialized

    def test_fail_sync_transitions_to_error(self):
        from qgre.sync_state import SyncLifecycle, SyncState

        state = SyncState()
        state.begin_sync()
        state.fail_sync()
        assert state.lifecycle == SyncLifecycle.ERROR

    def test_fail_sync_clears_dropout_flag(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.enter_dropout()
        state.fail_sync()
        assert not state.dropout_active


class TestSyncStateReset:
    """Test engine recreation reset."""

    def test_reset_clears_transient_state(self):
        from qgre.sync_state import SyncLifecycle, SyncState

        state = SyncState()
        state.enter_dropout()
        state.mark_cache_stale()
        state.begin_sync()
        state.complete_sync(first_call=True)

        state.reset_for_engine_recreate()

        assert state.lifecycle == SyncLifecycle.UNINITIALIZED
        assert not state.initialized
        assert not state.cache_stale
        assert not state.dropout_active

    def test_reset_preserves_restore_failed(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.restore_failed = True
        state.reset_for_engine_recreate()
        # restore_failed survives reset - engine recreation doesn't fix corrupted weights
        assert state.restore_failed


class TestSyncStateSerialization:
    """Test checkpoint serialization."""

    def test_state_dict_captures_persistent_state(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.begin_sync()
        state.complete_sync(first_call=True)
        state.restore_failed = True

        sd = state.state_dict()

        assert sd["initialized"] is True
        assert sd["restore_failed"] is True
        assert sd["lifecycle"] == "READY"

    def test_state_dict_excludes_transient_state(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.enter_dropout()
        state.mark_cache_stale()

        sd = state.state_dict()

        assert "dropout_active" not in sd
        assert "cache_stale" not in sd

    def test_load_state_dict_restores_persistent_state(self):
        from qgre.sync_state import SyncLifecycle, SyncState

        state = SyncState()
        state.load_state_dict(
            {
                "initialized": True,
                "restore_failed": True,
                "lifecycle": "READY",
            }
        )

        assert state.initialized
        assert state.restore_failed
        assert state.lifecycle == SyncLifecycle.READY
        assert not state.dropout_active
        assert not state.cache_stale

    def test_roundtrip_serialization(self):
        from qgre.sync_state import SyncState

        state1 = SyncState()
        state1.begin_sync()
        state1.complete_sync(first_call=True)

        sd = state1.state_dict()

        state2 = SyncState()
        state2.load_state_dict(sd)

        assert state2.initialized == state1.initialized
        assert state2.lifecycle == state1.lifecycle


class TestWeightLoaderStateBridge:
    """Test the SyncState ↔ WeightLoaderState bridge that trainer.save/resume uses.

    The bridge has to handle two seam issues:
      1. Lifecycle case: SyncLifecycle.name is uppercase ("READY"),
         WeightLoaderLifecycle value is lowercase ("ready").
      2. restore_failed must round-trip — it's the only safety flag whose
         persistence prevents silent weight corruption across restart.
    """

    def test_clean_state_roundtrip(self):
        """Fresh SyncState → WeightLoaderState → SyncState produces a clean state."""
        from qgre.sync_state import SyncLifecycle, SyncState
        from qgre.types import WeightLoaderState

        before = SyncState()
        sd = before.state_dict()
        wls = WeightLoaderState(
            initialized=sd["initialized"],
            restore_failed=sd["restore_failed"],
            lifecycle=sd["lifecycle"].lower(),
        )

        after = SyncState()
        if wls.initialized:
            after.lifecycle = SyncLifecycle.READY
            after.initialized = True
        elif wls.load_lora_called:
            after.lifecycle = SyncLifecycle.LOADING
        after.restore_failed = wls.restore_failed

        assert after.initialized is False
        assert after.restore_failed is False
        assert after.lifecycle == SyncLifecycle.UNINITIALIZED

    def test_initialized_ready_roundtrip(self):
        """A trained run (initialized=True, lifecycle=READY) survives the round-trip."""
        from qgre.sync_state import SyncLifecycle, SyncState
        from qgre.types import WeightLoaderState

        before = SyncState()
        before.begin_sync()
        before.complete_sync(first_call=True)

        sd = before.state_dict()
        wls = WeightLoaderState(
            initialized=sd["initialized"],
            restore_failed=sd["restore_failed"],
            lifecycle=sd["lifecycle"].lower(),
        )
        # WeightLoaderState normalizes the case at the seam (uppercase enum
        # name → lowercase enum value).
        assert wls.lifecycle == "ready"
        assert wls.initialized is True

        after = SyncState()
        if wls.initialized:
            after.lifecycle = SyncLifecycle.READY
            after.initialized = True
        after.restore_failed = wls.restore_failed

        assert after.initialized is True
        assert after.lifecycle == SyncLifecycle.READY

    def test_restore_failed_persists(self):
        """A run that crashed mid-restore must remain blocked across the round-trip.

        This is the load-bearing test: restore_failed is the safety flag that
        prevents the model from re-entering LoRA dropout after a corruption.
        If it doesn't round-trip, a crashed run will silently re-corrupt
        weights on resume.
        """
        from qgre.sync_state import SyncState
        from qgre.types import WeightLoaderState

        before = SyncState()
        before.enter_dropout()
        before.exit_dropout(success=False)
        assert before.restore_failed is True

        sd = before.state_dict()
        assert sd["restore_failed"] is True

        wls = WeightLoaderState(
            initialized=sd["initialized"],
            restore_failed=sd["restore_failed"],
            lifecycle=sd["lifecycle"].lower(),
        )
        assert wls.restore_failed is True

        after = SyncState()
        after.restore_failed = wls.restore_failed
        assert after.restore_failed is True

        # And the safety check fires on the resumed state, blocking dropout.
        with pytest.raises(RuntimeError, match="Weights are corrupted"):
            after.enter_dropout()

    def test_back_compat_old_checkpoint_without_restore_failed(self):
        """Old checkpoints predate restore_failed; from_dict should default it to False."""
        from qgre.types import WeightLoaderState

        # Simulate an old checkpoint dict without restore_failed
        old = {
            "load_lora_called": True,
            "initialized": True,
            "cleaned_up": False,
            "lifecycle": "ready",
        }
        wls = WeightLoaderState.from_dict(old)
        assert wls.restore_failed is False  # safe default
        assert wls.initialized is True
        assert wls.lifecycle == "ready"
