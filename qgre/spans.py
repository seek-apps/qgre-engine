"""Span-based advantage assignment — maps reward function character spans to token indices.

The reward function identifies WHERE in the completion text each quality was scored
(character offsets). This module converts those character spans to per-token boolean
masks so the advantage estimator can target the exact tokens that express each quality.

This replaces the section-based segmenter for reward functions that return scored_spans.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch


def build_char_to_token_map(
    token_ids: list[int],
    tokenizer: Any,
    completion_text: str | None = None,
) -> list[int] | None:
    """Build a character-offset → token-index mapping from ORIGINAL token_ids.

    Decodes the full sequence once, then uses convert_ids_to_tokens + offset tracking
    to map each character position to its source token. No re-encoding — uses the
    actual tokens that were generated.

    Returns a list where char_to_token[char_idx] = token_idx.
    Returns None if mapping cannot be built reliably.
    """
    if not token_ids:
        return []

    # Pre-validation: tokenizer must have decode method
    if not hasattr(tokenizer, "decode"):
        warnings.warn("build_char_to_token_map: tokenizer lacks decode method — returning None")
        return None

    # Get the full decoded text (this is what the reward function scored)
    try:
        full_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception as e:
        warnings.warn(f"build_char_to_token_map: full decode failed: {e}")
        return None

    full_len = len(full_text)
    if full_len == 0:
        return []

    # Build offset map by decoding each token and tracking cumulative position
    # This uses the ORIGINAL token_ids — no re-encoding
    char_to_token: list[int] = [-1] * full_len
    char_pos = 0

    for tok_idx, tid in enumerate(token_ids):
        try:
            # Decode this single token
            tok_text = tokenizer.decode([tid], skip_special_tokens=False)
            tok_len = len(tok_text)

            # Find where this token's text appears in full_text starting from char_pos
            # Usually it's exactly at char_pos, but BPE merges can shift things
            if tok_len > 0:
                # Try exact match first
                if full_text[char_pos:char_pos + tok_len] == tok_text:
                    for c in range(char_pos, min(char_pos + tok_len, full_len)):
                        char_to_token[c] = tok_idx
                    char_pos += tok_len
                else:
                    # BPE merge caused text difference — search nearby
                    found = False
                    for offset in range(min(5, full_len - char_pos)):
                        if full_text[char_pos + offset:char_pos + offset + tok_len] == tok_text:
                            for c in range(char_pos + offset, min(char_pos + offset + tok_len, full_len)):
                                char_to_token[c] = tok_idx
                            char_pos = char_pos + offset + tok_len
                            found = True
                            break
                    if not found:
                        # Can't find exact match — assign remaining chars proportionally
                        # This handles cases where individual decode differs from joint decode
                        remaining_tokens = len(token_ids) - tok_idx
                        remaining_chars = full_len - char_pos
                        if remaining_tokens > 0 and remaining_chars > 0:
                            chars_for_this = max(1, remaining_chars // remaining_tokens)
                            for c in range(char_pos, min(char_pos + chars_for_this, full_len)):
                                char_to_token[c] = tok_idx
                            char_pos += chars_for_this
        except Exception:
            # Decode failed for this token — skip
            pass

    # Fill any remaining gaps with nearest valid token
    last_valid = 0
    for c in range(full_len):
        if char_to_token[c] >= 0:
            last_valid = char_to_token[c]
        else:
            char_to_token[c] = last_valid

    return char_to_token


def scored_spans_to_token_masks(
    scored_spans: dict[str, list[tuple[int, int]]],
    char_to_token: list[int],
    seq_len: int,
) -> dict[str, torch.Tensor]:
    """Convert character-based scored_spans to per-token boolean masks.

    Args:
        scored_spans: quality_name → [(char_start, char_end), ...]
        char_to_token: char_idx → token_idx mapping (from build_char_to_token_map)
        seq_len: number of tokens in the completion

    Returns:
        dict mapping quality_name → torch.Tensor of shape [seq_len] with 1.0
        at token positions covered by that quality's spans, 0.0 elsewhere.
    """
    masks: dict[str, torch.Tensor] = {}
    max_char = len(char_to_token)

    for quality_name, spans in scored_spans.items():
        mask = torch.zeros(seq_len)
        for char_start, char_end in spans:
            # Clamp to valid range
            cs = max(0, min(char_start, max_char - 1))
            ce = max(0, min(char_end, max_char))
            if cs != char_start or ce != char_end:
                import warnings
                warnings.warn(
                    f"Span offset clamped for quality '{quality_name}': "
                    f"original ({char_start}, {char_end}) → clamped ({cs}, {ce}). "
                    f"max_char={max_char}. Final tokens may lose advantage signal."
                )
            # Map char range → token indices and set mask
            for c in range(cs, ce):
                tok_idx = char_to_token[c]
                if tok_idx < seq_len:
                    mask[tok_idx] = 1.0
        masks[quality_name] = mask

    return masks
