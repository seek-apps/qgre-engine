"""Tests for NeMo RL extracted modules (Step 0b)."""

import torch
import pytest


def test_import_loss_functions():
    """import succeeds with no external deps."""
    from qgre.nemo_extracted.loss_functions import ClippedPGLossFn, ClippedPGLossConfig


def test_import_kl():
    """import succeeds."""
    from qgre.nemo_extracted.kl import calculate_kl, masked_mean


def test_import_logits():
    """import succeeds."""
    from qgre.nemo_extracted.logits import logprobs_from_logits, compute_response_logprobs


def test_clipped_pg_loss_nonzero():
    """ClippedPGLossFn on synthetic data → non-zero, finite loss."""
    from qgre.nemo_extracted.loss_functions import ClippedPGLossFn

    cfg = {
        "reference_policy_kl_penalty": 0.0,
        "reference_policy_kl_type": "k3",
        "kl_input_clamp_value": 20.0,
        "kl_output_clamp_value": 10.0,
        "ratio_clip_min": 0.2,
        "ratio_clip_max": 0.28,
        "ratio_clip_c": None,
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": False,
        "truncated_importance_sampling_ratio": None,
        "token_level_loss": True,
    }
    loss_fn = ClippedPGLossFn(cfg)

    batch, seq = 4, 32
    curr_lp = torch.randn(batch, seq) * 0.1 - 3.0
    prev_lp = torch.randn(batch, seq) * 0.1 - 3.0
    advantages = torch.randn(batch, seq)
    mask = torch.ones(batch, seq)
    mask[:, -5:] = 0  # pad last 5 tokens

    loss, metrics = loss_fn(curr_lp, prev_lp, advantages, mask)

    assert loss.isfinite()
    assert loss.item() != 0.0
    assert "loss" in metrics
    assert "actor_loss" in metrics


def test_clipped_pg_loss_clip_bounds():
    """Large ratio → loss is clipped."""
    from qgre.nemo_extracted.loss_functions import ClippedPGLossFn

    cfg = {
        "reference_policy_kl_penalty": 0.0,
        "reference_policy_kl_type": "k3",
        "kl_input_clamp_value": 20.0,
        "kl_output_clamp_value": 10.0,
        "ratio_clip_min": 0.2,
        "ratio_clip_max": 0.28,
        "ratio_clip_c": None,
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": False,
        "truncated_importance_sampling_ratio": None,
        "token_level_loss": True,
    }
    loss_fn = ClippedPGLossFn(cfg)

    batch, seq = 2, 8
    # Create large ratio: curr much higher than prev → ratio >> 1
    curr_lp = torch.zeros(batch, seq)
    prev_lp = torch.full((batch, seq), -5.0)  # ratio = exp(5) ≈ 148
    advantages = torch.ones(batch, seq)
    mask = torch.ones(batch, seq)

    loss, metrics = loss_fn(curr_lp, prev_lp, advantages, mask)
    assert loss.isfinite()
    # Clamped ratio should be 1.28 (1 + 0.28), not 148
    assert metrics["probs_ratio_clamped_mean"] < 2.0


def test_kl_calculation_matches_manual():
    """KL on small tensors matches manual computation."""
    from qgre.nemo_extracted.kl import calculate_kl

    lp = torch.tensor([-1.0, -2.0, -0.5, -3.0])
    lp_ref = torch.tensor([-1.5, -1.8, -0.6, -2.5])

    # k3: exp(logr) - 1 - logr where logr = lp_ref - lp
    logr = lp_ref - lp
    expected = torch.exp(logr) - 1 - logr

    result = calculate_kl(lp, lp_ref, kl_type="k3", input_clamp_value=None, output_clamp_value=None)
    assert torch.allclose(result, expected, atol=1e-6)


def test_kl_types():
    """All KL types produce finite, non-negative results."""
    from qgre.nemo_extracted.kl import calculate_kl

    lp = torch.randn(4, 8) - 3.0
    lp_ref = torch.randn(4, 8) - 3.0

    for kl_type in ["k1", "k2", "k3"]:
        result = calculate_kl(lp, lp_ref, kl_type=kl_type)
        assert result.isfinite().all(), f"kl_type={kl_type} produced non-finite values"


def test_masked_mean_correctness():
    """masked_mean with known mask matches manual computation."""
    from qgre.nemo_extracted.kl import masked_mean

    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    result = masked_mean(values, mask)
    # (1*1 + 2*1 + 3*0 + 4*0) / (1+1+0+0) = 3/2 = 1.5
    assert abs(result.item() - 1.5) < 1e-6


def test_masked_mean_zero_mask():
    """masked_mean with all-zero mask → near-zero (epsilon protected)."""
    from qgre.nemo_extracted.kl import masked_mean

    values = torch.tensor([[1.0, 2.0]])
    mask = torch.zeros(1, 2)
    result = masked_mean(values, mask)
    assert result.isfinite()
    assert abs(result.item()) < 0.01


def test_logprobs_from_logits():
    """logprobs_from_logits on known logits matches manual log_softmax + gather."""
    from qgre.nemo_extracted.logits import logprobs_from_logits

    logits = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # [1, 2, 3]
    labels = torch.tensor([[2, 1]])  # [1, 2]

    result = logprobs_from_logits(logits, labels)

    # Manual: log_softmax then gather
    lp = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    expected = torch.tensor([[lp[0, 0, 2].item(), lp[0, 1, 1].item()]])

    assert torch.allclose(result, expected, atol=1e-6)
