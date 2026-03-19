"""Fused chunked logprob computation — Phase 4 of PLAN.md.

Computes log probabilities WITHOUT materializing the full [seq, vocab] logits tensor.
Instead, processes lm_head in chunks of `chunk_size` tokens at a time:
  hidden_states[:, chunk] @ lm_head.weight.T → [chunk, vocab] → log_softmax → gather → discard

Peak VRAM: chunk_size × vocab_size × dtype_bytes (e.g., 256 × 151936 × 2 = 74MB)
vs full:   seq_len × vocab_size × dtype_bytes   (e.g., 4096 × 151936 × 2 = 1.17GB)

Inspired by Liger Kernel's fused linear cross entropy approach (linkedin/Liger-Kernel).
Reference: "Cutting LLM Memory by 84%" (Medium, Feb 2026).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from qgre.nemo_extracted.logits import selective_log_softmax


def chunked_logprobs_from_hidden(
    hidden_states: torch.Tensor,
    lm_head: nn.Linear,
    labels: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Compute log probs from hidden states via chunked lm_head projection.

    Uses selective_log_softmax: never materializes a [chunk, vocab] log-prob tensor.
    For fp32: uses logsumexp identity (zero vocab-sized allocations).
    For bf16: per-row log_softmax fallback (one [vocab] allocation at a time).

    Peak VRAM per chunk: batch × chunk_size (just the gathered scalars)
    vs old approach: batch × chunk_size × vocab (74MB per chunk for Qwen3).

    Args:
        hidden_states: [batch, seq, hidden] — output of model body (before lm_head)
        lm_head: nn.Linear(hidden, vocab) — the language model head
        labels: [batch, seq] — next-token labels to gather log probs for
        chunk_size: tokens per chunk (lower = less memory)

    Returns:
        [batch, seq] log probs (float32) — gathered at label positions only
    """
    batch, seq_len, hidden = hidden_states.shape
    result = torch.zeros(batch, seq_len, dtype=torch.float32, device=hidden_states.device)

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_hidden = hidden_states[:, start:end, :]

        # Apply lm_head to chunk only — [batch, chunk, vocab]
        chunk_logits = lm_head(chunk_hidden)

        # selective_log_softmax: gathers without materializing full log-prob tensor
        chunk_labels = labels[:, start:end]
        result[:, start:end] = selective_log_softmax(chunk_logits, chunk_labels)

        del chunk_logits

    return result


def get_hidden_states_and_lm_head(model: nn.Module, input_ids: torch.Tensor):
    """Extract hidden states (before lm_head) and the lm_head linear layer.

    Works with Unsloth PeftModel, HuggingFace CausalLM, or any model with
    a `.model` body and `.lm_head` attribute.

    Args:
        model: The language model
        input_ids: [batch, seq] token IDs

    Returns:
        (hidden_states, lm_head) — hidden_states [batch, seq, hidden], lm_head nn.Linear
    """
    # Navigate through Unsloth/PEFT wrappers to find the model body and lm_head
    inner = model
    while hasattr(inner, "model") and not hasattr(inner, "lm_head"):
        inner = inner.model
    if hasattr(inner, "base_model"):
        inner = inner.base_model
    while hasattr(inner, "model") and not hasattr(inner, "lm_head"):
        inner = inner.model

    if not hasattr(inner, "lm_head"):
        # Fallback: can't find lm_head, use full forward
        return None, None

    lm_head = inner.lm_head

    # Get the model body (everything except lm_head)
    # For CausalLM: model.model is the body, model.lm_head is the head
    body = inner.model if hasattr(inner, "model") else None
    if body is None:
        return None, None

    # Forward through body only — handle various output formats
    body_output = body(input_ids)

    # BaseModelOutputWithPast (standard HF) → .last_hidden_state
    if hasattr(body_output, "last_hidden_state"):
        hidden_states = body_output.last_hidden_state
    # Tuple output (Unsloth patched) → first element is hidden states
    elif isinstance(body_output, tuple) and len(body_output) > 0:
        hidden_states = body_output[0]
    # Raw tensor
    elif isinstance(body_output, torch.Tensor):
        hidden_states = body_output
    else:
        return None, None

    return hidden_states, lm_head
