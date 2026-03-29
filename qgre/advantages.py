from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Callable

import numpy as np
import torch

from qgre.segments import Segmenter, segmenter_region_count, uniform_segmenter
from qgre.types import PromptContext, RewardResult


def apply_frontier_amplification(
    step_advs: dict,
    step_nums: list[int],
    frontier_steps: set[int] | None,
    amplification: float,
) -> None:
    """Multiply advantages for frontier steps (blocking phase advancement).

    Modifies step_advs in place.
    """
    if frontier_steps and amplification > 0:
        for sn in step_nums:
            # ADV-R1-7: Check if step exists in step_advs before amplifying
            if sn in frontier_steps and sn in step_advs:
                step_advs[sn] = step_advs[sn] * (1.0 + amplification)


def broadcast_step_advantages_to_tokens(
    step_advs: dict[int, float | torch.Tensor],
    regions: list[str],
    region_extra_steps: dict[int, list[int]],
    sample_idx: int | None = None,
) -> torch.Tensor:
    """Broadcast per-step advantages to per-token by region label.

    Shared by SPO/GRPO path and VPRM path. Each token gets the sum of its
    region's primary step advantage + any virtual steps mapped to that region.

    Args:
        step_advs: step_num → advantage (float for VPRM, Tensor[batch] for SPO/GRPO)
        regions: per-token region labels from segmenter
        region_extra_steps: region_step → [virtual steps mapped to it]
        sample_idx: when step_advs values are batch tensors, index into them
    """
    # Pre-build label → advantage value map from unique regions (O(n_labels) string ops)
    # then do O(seq_len) dict lookups instead of per-token string parsing
    label_to_adv: dict[str, float | torch.Tensor] = {}
    for region in set(regions):
        if region.startswith("STEP_"):
            sn = int(region.split("_")[1])
            if sn in step_advs:
                val = step_advs[sn]
                primary = val[sample_idx] if sample_idx is not None else val
                contribs = [primary]
                for vs in region_extra_steps.get(sn, []):
                    if vs in step_advs:
                        v = step_advs[vs]
                        contribs.append(v[sample_idx] if sample_idx is not None else v)
                # Convert all to tensors before torch.stack
                if isinstance(contribs[0], torch.Tensor):
                    contribs_tensors = [c if isinstance(c, torch.Tensor) else torch.tensor(c, device=contribs[0].device) for c in contribs]
                    label_to_adv[region] = torch.stack(contribs_tensors).sum()
                else:
                    label_to_adv[region] = sum(contribs)
        elif region == "THINK" and 0 in step_advs:
            val = step_advs[0]
            label_to_adv[region] = val[sample_idx] if sample_idx is not None else val

    seq_len = len(regions)
    # Infer device from step_advs values (may be Tensor or float)
    device = None
    for val in step_advs.values():
        if isinstance(val, torch.Tensor):
            device = val.device
            break
    token_advs = torch.zeros(seq_len, device=device)
    for t, region in enumerate(regions):
        if region in label_to_adv:
            token_advs[t] = label_to_adv[region]
    return token_advs


def build_batch_reward_tensors(
    reward_results: list[RewardResult],
) -> dict[str, torch.Tensor]:
    """Convert list[RewardResult] → dict[str, Tensor] per quality component.

    Each tensor has shape [batch_size]. Missing keys are zero-filled.
    """
    if not reward_results:
        return {}

    all_keys: set[str] = set()
    for rr in reward_results:
        all_keys.update(rr.scores.keys())

    tensors: dict[str, torch.Tensor] = {}
    for key in sorted(all_keys):
        values = [rr.scores.get(key, 0.0) for rr in reward_results]
        tensors[key] = torch.tensor(values, dtype=torch.float32)

    return tensors


def build_phase_qualities(
    step_qualities: dict[int, list[str]],
    cumulative: bool = True,
) -> dict[int, list[str]]:
    """Build phase→qualities mapping from step_qualities config.

    If cumulative=True (default QGRE behavior): phase N includes all qualities from steps 1..N.
    If cumulative=False: phase N includes only step N's qualities.
    """
    steps = sorted(step_qualities.keys())
    if cumulative:
        return {
            phase: [q for s in steps if s <= phase for q in step_qualities[s]]
            for phase in steps
        }
    else:
        return dict(step_qualities)


class QGREStepAdvantageEstimator:
    """Unified: SPO + GDPO + VPRM + QGRE phase gating.

    Configurable for any domain. Accepts:
    - step_qualities: mapping of step_num → quality names (from your reward_fn)
    - segmenter: function that splits token IDs into step regions
    - mode: "spo" (persistent value tracker) or "grpo" (group-mean baseline)
    """

    def __init__(
        self,
        lr: float = 0.1,
        mode: str = "spo",
        step_qualities: dict[int, list[str]] | None = None,
        segmenter: Segmenter | None = None,
        normalize_advantages: bool = True,
        filter_groups: bool = True,
        step_region_map: dict[int, int] | None = None,
        frontier_amplification: float = 2.0,
        var_aware: bool = True,
        var_threshold: float = 0.01,
        var_lr: float = 0.05,
        min_var_ratio: float = 0.01,
        staleness_window: int = 50,
        baseline_prior: float = 0.5,
    ):
        self.lr = lr
        self.mode = mode
        self.normalize_advantages = normalize_advantages
        self.filter_groups = filter_groups
        self.frontier_amplification = frontier_amplification
        self._reward_key_checked = False
        if step_qualities is None:
            raise ValueError(
                "step_qualities is required. Pass a dict mapping step numbers to quality names, e.g.:\n"
                "  {1: ['q_format'], 2: ['q_grounding'], 3: ['q_accuracy']}\n"
                "See examples/ for domain-specific configs."
            )
        self.step_qualities = step_qualities
        self.segmenter = segmenter or uniform_segmenter
        self._step_nums = sorted(self.step_qualities.keys())
        # Per-quality SPO baselines (keyed by quality name, not step number)
        self.V: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.V_last_seen: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._quality_seen: dict[int, set[str]] = defaultdict(set)
        # Legacy step-seen for backward compat (deprecated)
        self._step_seen: dict[int, set[int]] = defaultdict(set)
        # Target-aware aspiration gap
        self._aspiration_beta = 0.0  # Set from config via trainer
        self._aspiration_target = 0.0

        # Variance-aware baseline: track per-(prompt, quality) reward variance
        self._var_aware = var_aware
        self._var_threshold = var_threshold
        self._var_lr = var_lr
        self._min_var_ratio = min_var_ratio
        self._reward_var: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(lambda: self._var_threshold))
        self._reward_mean: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # Staleness decay for sparse qualities
        self._staleness_window = staleness_window
        self._baseline_prior = baseline_prior
        self._current_step = 0  # Updated externally by trainer
        # step_region_map: virtual steps (no segmenter region) → region step whose tokens carry their advantage
        # e.g., {7: 2} means step 7's advantage is added to STEP_2 tokens
        self.step_region_map = step_region_map or {}
        # Validate: virtual steps must exist in step_qualities, region targets must too
        if self.step_region_map:
            for vs, rs in self.step_region_map.items():
                if vs not in self.step_qualities:
                    warnings.warn(
                        f"step_region_map key {vs} not in step_qualities — "
                        f"mapped advantage will never be computed."
                    )
                if rs not in self.step_qualities:
                    warnings.warn(
                        f"step_region_map value {rs} (target for step {vs}) not in step_qualities — "
                        f"no tokens will carry step {vs}'s advantage. Check segmenter regions."
                    )
        # Build reverse map: region_step → [virtual steps that map to it]
        self._region_extra_steps: dict[int, list[int]] = defaultdict(list)
        for virtual_step, region_step in self.step_region_map.items():
            self._region_extra_steps[region_step].append(virtual_step)

    def get_baseline(self, prompt_id: int, quality_name: str) -> float:
        """Get baseline with staleness decay for sparse qualities.

        Qualities that haven't been seen in staleness_window steps decay
        toward baseline_prior to prevent stale baselines from anchoring advantage.
        """
        V = self.V[prompt_id][quality_name]
        last_seen = self.V_last_seen[prompt_id][quality_name]

        if last_seen == 0:
            return self._baseline_prior  # Never seen → use prior

        # Guard: if current_step not yet set or negative staleness, use prior
        if self._current_step == 0:
            return self._baseline_prior
        steps_since = self._current_step - last_seen
        if steps_since < 0:
            # Checkpoint restored with future last_seen — use prior
            warnings.warn(
                f"Negative staleness {steps_since} for prompt {prompt_id} quality {quality_name}. "
                f"current_step={self._current_step}, last_seen={last_seen}. Using prior."
            )
            return self._baseline_prior

        if steps_since > self._staleness_window:
            # Continuous exponential decay (no discrete jumps)
            decay = 0.9 ** (steps_since / self._staleness_window)
            return V * decay + self._baseline_prior * (1 - decay)

        return V

    def update_baseline(
        self, prompt_id: int, quality_name: str, reward: float, effective_lr: float
    ) -> None:
        """Update baseline and track last-seen step."""
        V = self.V[prompt_id][quality_name]
        self.V[prompt_id][quality_name] = V + effective_lr * (reward - V)
        self.V_last_seen[prompt_id][quality_name] = self._current_step

    def set_current_step(self, step: int) -> None:
        """Called by trainer at start of each step."""
        self._current_step = step

    def compute_advantages(
        self,
        batch_prompt_ids: list[int],
        batch_token_ids: list[list[int]],
        batch_reward_results: list[RewardResult],
        batch_active_qualities: list[list[str]],
        group_size: int | None = None,
        frontier_steps: set[int] | None = None,
        batch_contexts: list[PromptContext] | None = None,
    ) -> tuple[list[torch.Tensor], list[list[str]]]:
        """Compute per-token advantages via segment → step rewards → SPO/GRPO → GDPO → broadcast.

        Args:
            frontier_steps: Steps that block phase advancement (below mastery threshold).
                When set, these steps receive amplified advantages (frontier_amplification).
                Steps NOT in the frontier get base weight 1.0; frontier steps get
                1.0 + frontier_amplification (default: 3x total gradient pressure).

        Returns:
            (batch_advantages, batch_regions) — per-token advantages and region labels
        """
        batch_size = len(batch_token_ids)

        # First-batch invariant: check reward keys overlap with step_qualities
        if not self._reward_key_checked and batch_reward_results:
            all_quality_keys = set()
            for qs in self.step_qualities.values():
                all_quality_keys.update(qs)
            all_reward_keys = set()
            for rr in batch_reward_results:
                all_reward_keys.update(rr.scores.keys())
            overlap = all_quality_keys & all_reward_keys
            if not overlap:
                warnings.warn(
                    f"Reward key mismatch: reward_fn returns {sorted(all_reward_keys)} "
                    f"but step_qualities expects {sorted(all_quality_keys)}. "
                    f"All step rewards will be 0.0 — training has no signal."
                )
            self._reward_key_checked = True

        # Phase 1: Segment tokens + compute per-step rewards
        all_regions: list[list[str]] = []
        all_step_rewards: list[dict[int, float]] = []

        for i in range(batch_size):
            regions = self.segmenter(batch_token_ids[i])
            step_rews: dict[int, float] = {}
            for step_num, quality_keys in self.step_qualities.items():
                active = [k for k in quality_keys if k in batch_active_qualities[i]]
                if active:
                    vals = [batch_reward_results[i].scores.get(k, 0.0) for k in active]
                    step_rews[step_num] = sum(vals) / len(vals)
                else:
                    step_rews[step_num] = 0.0
            all_step_rewards.append(step_rews)
            all_regions.append(regions)

        # Phase 2: Per-step advantages (SPO or GRPO baseline)
        step_advs: dict[int, torch.Tensor] = {
            s: torch.zeros(batch_size) for s in self._step_nums
        }

        if self.mode == "spo":
            self._compute_spo_advantages(batch_prompt_ids, all_step_rewards, step_advs, batch_size,
                                         batch_contexts=batch_contexts)
        else:
            self._compute_grpo_advantages(
                batch_prompt_ids, all_step_rewards, step_advs, batch_size,
                group_size=group_size or batch_size,
            )

        # Advantage normalization — mode-dependent:
        # SPO raw: no normalization (EMA baseline is the centering).
        # GRPO+normalize: per-step mean+std (GDPO-style).
        # GRPO+dr_grpo: per-step mean-only (no std division).
        self._normalize_step_advantages(step_advs)

        # Phase-aware frontier amplification: focus gradient on bottleneck steps
        apply_frontier_amplification(step_advs, self._step_nums, frontier_steps, self.frontier_amplification)

        # Phase 3: Broadcast per-step advantages to per-token by region
        batch_advantages: list[torch.Tensor] = []
        for i in range(batch_size):
            token_advs = broadcast_step_advantages_to_tokens(
                step_advs, all_regions[i], self._region_extra_steps, sample_idx=i,
            )
            batch_advantages.append(token_advs)

        return batch_advantages, all_regions

    def _normalize_step_advantages(self, step_advs: dict[int, torch.Tensor]):
        """NaN guard + mode-dependent normalization. Shared by region and span paths."""
        for step_num in self._step_nums:
            # NaN guard (ms-swift #8123): reward_fn can return NaN on malformed completions
            if step_advs[step_num].isnan().any():
                nan_count = step_advs[step_num].isnan().sum().item()
                warnings.warn(
                    f"Step {step_num}: {nan_count}/{len(step_advs[step_num])} advantages are NaN. "
                    f"Check reward_fn for NaN returns. Replacing with 0.0."
                )
                step_advs[step_num] = torch.nan_to_num(step_advs[step_num], nan=0.0)
            if self.mode == "spo":
                # SPO raw: no normalization. The per-prompt EMA baseline IS the centering
                # mechanism. Batch normalization would double-center and erase the importance
                # hierarchy between steps (bottleneck steps should produce larger gradients).
                pass
            elif self.normalize_advantages:
                mean = step_advs[step_num].mean()
                std = step_advs[step_num].std(correction=0)
                if std > 1e-8:
                    step_advs[step_num] = (step_advs[step_num] - mean) / (std + 1e-8)
                else:
                    step_advs[step_num] = step_advs[step_num] - mean
            else:
                mean = step_advs[step_num].mean()
                step_advs[step_num] = step_advs[step_num] - mean

    def _compute_spo_advantages(
        self,
        batch_prompt_ids: list[int],
        all_step_rewards: list[dict[int, float]],
        step_advs: dict[int, torch.Tensor],
        batch_size: int,
        batch_contexts: list[PromptContext] | None = None,
    ):
        # ADV-R1-2: Validate batch_contexts length if provided
        if batch_contexts is not None and len(batch_contexts) < batch_size:
            raise ValueError(
                f"batch_contexts length ({len(batch_contexts)}) < batch_size ({batch_size}). "
                "Must provide context for all samples."
            )
        # Pre-compute batch mean per step for warm-start (PLAN.md spec: use batch mean, not sample)
        batch_means: dict[int, float] = {}
        for step_num in self._step_nums:
            rewards = [all_step_rewards[i].get(step_num, 0.0) for i in range(batch_size)]
            batch_means[step_num] = float(np.mean(rewards)) if rewards else 0.0

        for step_num in self._step_nums:
            for i in range(batch_size):
                # ADV-R2-4: Check for None before accessing attributes
                ctx = batch_contexts[i] if batch_contexts and i < len(batch_contexts) else None
                pid = ctx.prompt_id if ctx is not None else batch_prompt_ids[i]
                r = all_step_rewards[i].get(step_num, 0.0)
                v = self.V[pid][step_num]

                # Warm-start: first observation → set baseline to BATCH MEAN (not sample)
                if step_num not in self._step_seen[pid]:
                    v = batch_means[step_num]
                    self._step_seen[pid].add(step_num)

                # Perfect score (1.0) = zero advantage. Nothing to learn — don't waste
                # gradient reinforcing specific tokens. Imperfect = push toward 1.0.
                if r >= 1.0:
                    step_advs[step_num][i] = 0.0
                else:
                    step_advs[step_num][i] = r - v
                    # Aspiration gap: push toward perfection (1.0).
                    # ADV-R2-1: Safe attribute access with getattr
                    if self._aspiration_beta > 0:
                        ctx = batch_contexts[i] if batch_contexts and i < len(batch_contexts) and batch_contexts[i] is not None else None
                        warmup = getattr(ctx, 'aspiration_warmup', 1.0) if ctx else 1.0
                        step_advs[step_num][i] += self._aspiration_beta * warmup * (r - 1.0)

                # Variance-aware baseline: slow lr when reward is constant
                effective_lr = self.lr
                if self._var_aware:
                    # Track reward variance via running mean + EMA of squared deviation
                    r_mean = self._reward_mean[pid][step_num]
                    new_mean = r_mean + self._var_lr * (r - r_mean)
                    self._reward_mean[pid][step_num] = new_mean
                    old_var = self._reward_var[pid][step_num]
                    # ADV-R2-2: Use new_mean for variance (not stale r_mean)
                    new_var = old_var + self._var_lr * ((r - new_mean) ** 2 - old_var)
                    if new_var < 0:
                        # Negative variance indicates numerical instability in EMA — log and clamp
                        warnings.warn(
                            f"Negative variance {new_var:.6f} for prompt {pid} step {step_num}. "
                            f"old_var={old_var:.6f}, r={r:.4f}, r_mean={r_mean:.4f}. Clamping to 0."
                        )
                    self._reward_var[pid][step_num] = max(0.0, new_var)
                    if new_var < self._var_threshold:
                        effective_lr = self.lr * max(new_var / self._var_threshold, self._min_var_ratio)

                self.V[pid][step_num] = v + effective_lr * (r - v)

    def _compute_grpo_advantages(
        self,
        batch_prompt_ids: list[int],
        all_step_rewards: list[dict[int, float]],
        step_advs: dict[int, torch.Tensor],
        batch_size: int,
        group_size: int,
    ):
        if batch_size % group_size != 0:
            raise ValueError(
                f"GRPO requires batch_size divisible by group_size, "
                f"got {batch_size} % {group_size} != 0."
            )
        num_groups = batch_size // group_size
        for step_num in self._step_nums:
            for g in range(num_groups):
                start = g * group_size
                end = start + group_size
                group_rewards = [all_step_rewards[i].get(step_num, 0.0) for i in range(start, end)]
                mean = float(np.mean(group_rewards))
                std = float(np.std(group_rewards))
                if std < 1e-8 and self.filter_groups:
                    # DAPO Dynamic Sampling: all-identical rewards → zero advantage (no signal)
                    for i in range(start, end):
                        step_advs[step_num][i] = 0.0
                else:
                    # Mean-only subtraction: outer GDPO loop handles normalization per-step.
                    # No std division here — GDPO replaces group-level std normalization.
                    for i in range(start, end):
                        step_advs[step_num][i] = all_step_rewards[i].get(step_num, 0.0) - mean

    def adapt_lr(
        self,
        kl: float,
        kl_threshold: float = 0.1,
        kl_factor: float = 2.0,
        lr_factor: float = 1.5,
        min_lr: float = 0.01,
        max_lr: float = 0.5,
    ):
        """KL-adaptive SPO learning rate (SPO paper Algorithm 1).

        Decrease lr when KL high (model drifting), increase when KL low (stagnating).
        """
        if kl > kl_factor * kl_threshold:
            self.lr = max(self.lr / lr_factor, min_lr)
        elif kl < kl_threshold / kl_factor:
            self.lr = min(self.lr * lr_factor, max_lr)

    def get_prompt_priorities(self) -> dict[int, float]:
        """Return |mean advantage| per prompt for prioritized sampling (SPO paper Section 3.2).

        Prompts with large |advantage| are sampled more often — adaptive curriculum.
        Returns dict mapping prompt_id → priority weight (higher = sample more).
        """
        priorities: dict[int, float] = {}
        for pid, steps in self.V.items():
            if not steps:
                continue
            # Use mean |V| across steps as priority proxy — prompts where the model
            # is far from baseline have high learning signal
            mean_abs_v = float(np.mean([abs(v) for v in steps.values()])) if steps else 0.0
            priorities[pid] = mean_abs_v
        return priorities

    def on_tier_advance(self, new_tier: int, prompt_tier_map: dict[int, int]):
        """Reset SPO baseline for the NEW step only — preserve learned baselines for mastered steps."""
        for pid, tier in prompt_tier_map.items():
            if tier == new_tier:
                self.V[pid][new_tier] = 0.0
                self._step_seen[pid].discard(new_tier)

    def state_dict(self) -> dict:
        return {
            # Per-quality baselines (keyed by quality name string)
            "V": {pid: dict(qualities) for pid, qualities in self.V.items()},
            "V_last_seen": {pid: dict(qualities) for pid, qualities in self.V_last_seen.items()},
            "quality_seen": {pid: list(qualities) for pid, qualities in self._quality_seen.items()},
            # Legacy step_seen for backward compat
            "step_seen": {pid: list(steps) for pid, steps in self._step_seen.items()},
            # Per-quality variance tracking
            "reward_var": {pid: dict(qualities) for pid, qualities in self._reward_var.items()},
            "reward_mean": {pid: dict(qualities) for pid, qualities in self._reward_mean.items()},
            "lr": self.lr,
            "mode": self.mode,
            "current_step": self._current_step,
        }

    def compute_advantages_with_spans(
        self,
        batch_prompt_ids: list[int],
        batch_token_ids: list[list[int]],
        batch_reward_results: list["RewardResult"],
        batch_active_qualities: list[list[str]],
        batch_token_masks: list[dict[str, torch.Tensor]],
        group_size: int | None = None,
        frontier_steps: set[int] | None = None,
        batch_contexts: list[PromptContext] | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, dict[str, float]]]:
        """Compute per-token advantages using PER-QUALITY span-based token masks.

        Unlike step-level averaging, this computes independent advantages per quality
        and broadcasts each quality's advantage to only its own span tokens.

        Returns:
            (batch_advantages, batch_quality_metrics):
            - batch_advantages: per-token advantage tensors
            - batch_quality_metrics: per-sample dict of quality_name → {reward, baseline, advantage}
        """
        batch_size = len(batch_token_ids)

        # Per-quality advantages for each sample
        all_quality_advs: list[dict[str, float]] = []
        batch_quality_metrics: dict[str, dict[str, float]] = {}

        # Phase 1+2: Compute per-quality advantages directly (no step averaging)
        for i in range(batch_size):
            ctx = batch_contexts[i] if batch_contexts and i < len(batch_contexts) else None
            pid = ctx.prompt_id if ctx is not None else batch_prompt_ids[i]
            warmup = getattr(ctx, 'aspiration_warmup', 1.0) if ctx else 1.0

            quality_advs: dict[str, float] = {}

            for quality_name in batch_active_qualities[i]:
                r = batch_reward_results[i].scores.get(quality_name, 0.0)
                v = self.get_baseline(pid, quality_name)

                # Warm-start: first observation → use prior
                if quality_name not in self._quality_seen[pid]:
                    v = self._baseline_prior
                    self._quality_seen[pid].add(quality_name)

                # Perfect score (1.0) = zero advantage — nothing to learn
                if r >= 1.0:
                    quality_advs[quality_name] = 0.0
                else:
                    adv = r - v
                    # Aspiration gap: push toward target (usually mastery_threshold)
                    if self._aspiration_beta > 0:
                        adv += self._aspiration_beta * warmup * (r - self._aspiration_target)
                    quality_advs[quality_name] = adv

                # Variance-aware baseline learning rate
                effective_lr = self.lr
                if self._var_aware:
                    r_mean = self._reward_mean[pid][quality_name]
                    new_mean = r_mean + self._var_lr * (r - r_mean)
                    self._reward_mean[pid][quality_name] = new_mean
                    old_var = self._reward_var[pid][quality_name]
                    new_var = old_var + self._var_lr * ((r - new_mean) ** 2 - old_var)
                    if new_var < 0:
                        warnings.warn(
                            f"Negative variance {new_var:.6f} for prompt {pid} quality {quality_name}. "
                            f"old_var={old_var:.6f}, r={r:.4f}, r_mean={r_mean:.4f}. Clamping to 0."
                        )
                    self._reward_var[pid][quality_name] = max(0.0, new_var)
                    if new_var < self._var_threshold:
                        effective_lr = self.lr * max(new_var / self._var_threshold, self._min_var_ratio)

                # Update baseline using new per-quality method
                self.update_baseline(pid, quality_name, r, effective_lr)

                # Collect metrics for logging
                batch_quality_metrics[f"sample_{i}/{quality_name}"] = {
                    "reward": r,
                    "baseline": v,
                    "advantage": quality_advs[quality_name],
                }

            all_quality_advs.append(quality_advs)

        # Phase 3: Broadcast per-quality advantages to tokens (additive + normalized)
        batch_advantages: list[torch.Tensor] = []
        for i in range(batch_size):
            seq_len = len(batch_token_ids[i])
            # Get device from batch_token_masks
            device = None
            if batch_token_masks[i]:
                first_mask = next(iter(batch_token_masks[i].values()), None)
                if first_mask is not None and isinstance(first_mask, torch.Tensor):
                    device = first_mask.device

            token_advs = torch.zeros(seq_len, device=device)
            token_coverage = torch.zeros(seq_len, device=device)
            masks = batch_token_masks[i]

            for quality_name in batch_active_qualities[i]:
                if quality_name not in masks:
                    continue
                q_adv = all_quality_advs[i].get(quality_name, 0.0)
                if abs(q_adv) < 1e-10:
                    continue
                q_mask = masks[quality_name]
                # Graceful handling instead of assert — don't crash on data-dependent mismatch
                if q_mask.shape[0] != seq_len:
                    warnings.warn(
                        f"Mask shape mismatch for quality '{quality_name}': "
                        f"mask has {q_mask.shape[0]} tokens but sequence has {seq_len}. "
                        f"Skipping — check reward_fn scored_spans and tokenizer consistency."
                    )
                    continue
                token_advs += q_adv * q_mask
                token_coverage += q_mask

            # Normalize: tokens covered by N qualities get advantage / N
            # Preserves direction, bounds magnitude
            token_coverage = torch.clamp(token_coverage, min=1.0)
            token_advs = token_advs / token_coverage

            batch_advantages.append(token_advs)

        return batch_advantages, batch_quality_metrics

    def load_state_dict(self, state: dict):
        self.lr = state.get("lr", self.lr)
        self.mode = state.get("mode", self.mode)
        self._current_step = state.get("current_step", 0)

        # Per-quality baselines (string keys)
        self.V = defaultdict(lambda: defaultdict(float))
        for pid, qualities in state.get("V", {}).items():
            for quality_name, val in qualities.items():
                # Handle both old (int step keys) and new (string quality keys)
                self.V[int(pid)][str(quality_name)] = float(val)

        self.V_last_seen = defaultdict(lambda: defaultdict(int))
        for pid, qualities in state.get("V_last_seen", {}).items():
            for quality_name, val in qualities.items():
                self.V_last_seen[int(pid)][str(quality_name)] = int(val)

        self._quality_seen = defaultdict(set)
        for pid, qualities in state.get("quality_seen", {}).items():
            self._quality_seen[int(pid)] = set(str(q) for q in qualities)

        # Legacy step_seen for backward compat
        self._step_seen = defaultdict(set)
        for pid, steps in state.get("step_seen", {}).items():
            self._step_seen[int(pid)] = set(int(s) for s in steps)

        # Per-quality variance tracking (string keys)
        for pid, qualities in state.get("reward_var", {}).items():
            for quality_name, val in qualities.items():
                self._reward_var[int(pid)][str(quality_name)] = float(val)

        self._reward_mean = defaultdict(lambda: defaultdict(float))
        for pid, qualities in state.get("reward_mean", {}).items():
            for quality_name, val in qualities.items():
                self._reward_mean[int(pid)][str(quality_name)] = float(val)


def compute_advantages_vprm(
    critic,  # VPRMCritic
    hidden_states: torch.Tensor,
    regions: list[str],
    reward_result: "RewardResult",
    step_qualities: dict[int, list[str]],
    active_qualities: list[str],
    step_region_map: dict[int, int] | None = None,
    frontier_steps: set[int] | None = None,
    frontier_amplification: float = 2.0,
    min_regions: int = 2,
    aspiration_beta: float = 0.0,
    aspiration_target: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Compute per-token advantages using VPRM critic for a single sample.

    Returns:
        (token_advantages, critic_loss, used_critic):
        - token_advantages: [seq_len] per-token advantages
        - critic_loss: scalar MSE loss for critic training
        - used_critic: True if critic was used, False if SPO fallback
    """
    seq_len = hidden_states.shape[0]
    device = hidden_states.device

    # Check if enough regions for critic — else SPO fallback
    n_regions = segmenter_region_count(regions)
    if n_regions < min_regions:
        return (
            torch.zeros(seq_len, device=device),
            torch.tensor(0.0, device=device),
            False,
        )

    # Get actual rewards per quality
    actual_rewards = {k: reward_result.scores.get(k, 0.0) for k in active_qualities}

    # Compute advantages via critic (hidden states must be DETACHED)
    advs_dict, critic_losses = critic.compute_advantages(
        hidden_states.detach(), regions, actual_rewards,
    )

    # Build reverse map: region_step → [virtual steps that map to it]
    region_extra_steps: dict[int, list[int]] = defaultdict(list)
    if step_region_map:
        for vs, rs in step_region_map.items():
            region_extra_steps[rs].append(vs)

    # Broadcast per-quality advantages to per-token by region
    step_nums = sorted(step_qualities.keys())
    # Build per-step advantages from quality advantages
    step_advs: dict[int, float] = {}
    for step_num in step_nums:
        qualities = [q for q in step_qualities[step_num] if q in active_qualities]
        if qualities:
            vals = [advs_dict.get(q, 0.0) for q in qualities]
            step_advs[step_num] = sum(vals) / len(vals)
        else:
            step_advs[step_num] = 0.0

    # ADV-R1-8: Initialize virtual steps with 0.0 advantage
    if step_region_map:
        for vs in step_region_map.keys():
            if vs not in step_advs:
                step_advs[vs] = 0.0

    # Perfect score = zero advantage. Imperfect = push toward 1.0.
    for step_num in step_nums:
        qualities = [q for q in step_qualities[step_num] if q in active_qualities]
        if qualities:
            step_reward = sum(reward_result.scores.get(q, 0.0) for q in qualities) / len(qualities)
            if step_reward >= 1.0:
                step_advs[step_num] = 0.0
            elif aspiration_beta > 0:
                step_advs[step_num] += aspiration_beta * (step_reward - 1.0)

    apply_frontier_amplification(step_advs, step_nums, frontier_steps, frontier_amplification)

    # Broadcast to tokens
    token_advantages = broadcast_step_advantages_to_tokens(
        step_advs, regions, region_extra_steps,
    ).to(device)

    # Total critic loss
    if critic_losses:
        losses = [loss for loss in critic_losses.values() if not torch.isnan(loss)]
        if len(losses) < len(critic_losses):
            import warnings
            warnings.warn(f"Filtered {len(critic_losses) - len(losses)} NaN critic losses before aggregation")
        total_critic_loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
    else:
        total_critic_loss = torch.tensor(0.0, device=device)

    return token_advantages, total_critic_loss, True
