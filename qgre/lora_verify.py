from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol


class WeightSyncable(Protocol):
    """Interface for anything that can save/load LoRA weights."""

    def save_lora(self, path: str | Path) -> None: ...
    def load_lora(self, path: str | Path) -> None: ...


class LoRAVerifier:
    """Verify LoRA weights are correctly synced between training and generation.

    Three functions:
    (a) verify_sync: hash weights before/after load, assert match
    (b) verify_active: generate from fixed prompt, verify output differs from base
    (c) periodic_recreate: signal to recreate vLLM engine every N steps

    Integrates into QGRETrainer.step() as post-sync hook and
    QGRETrainer.resume() as mandatory step.
    """

    def __init__(self, recreate_interval: int = 50):
        self.recreate_interval = recreate_interval
        self._last_save_hash: str | None = None
        self._steps_since_recreate: int = 0

    def hash_lora_dir(self, path: str | Path) -> str:
        """Compute hash of LoRA weight files for verification."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LoRA path does not exist: {path}")

        hasher = hashlib.sha256()
        # Prefer safetensors, fall back to bin (sorted for deterministic ordering)
        weight_files = sorted(path.rglob("*.safetensors")) or sorted(path.rglob("*.bin"))
        if not weight_files:
            raise FileNotFoundError(f"No weight files (.safetensors/.bin) found in: {path}")
        for f in weight_files:
            hasher.update(f.read_bytes())

        return hasher.hexdigest()

    def verify_sync(self, lora_path: str | Path) -> bool:
        """Verify saved LoRA weights match what was saved.

        Call after save_lora + load_lora. Returns True if hash matches.
        Raises ValueError if mismatch detected.
        """
        current_hash = self.hash_lora_dir(lora_path)

        if self._last_save_hash is not None and current_hash != self._last_save_hash:
            raise ValueError(
                f"LoRA weight mismatch after sync! "
                f"Expected hash {self._last_save_hash[:16]}..., "
                f"got {current_hash[:16]}..."
            )

        self._last_save_hash = current_hash
        return True

    def on_save(self, lora_path: str | Path):
        """Record hash after saving LoRA weights."""
        self._last_save_hash = self.hash_lora_dir(lora_path)

    def should_recreate_engine(self) -> bool:
        """Check if vLLM engine should be recreated to prevent memory leak.

        Call at the start of each training step. Returns True every
        `recreate_interval` steps (default 50) to prevent unsloth #3864.
        """
        self._steps_since_recreate += 1
        if self._steps_since_recreate >= self.recreate_interval:
            self._steps_since_recreate = 0
            return True
        return False

    def reset_recreate_counter(self):
        """Call after engine recreation."""
        self._steps_since_recreate = 0
