from __future__ import annotations

# Qwen3 verified token IDs (from SPECIAL-TOKENS-SUPERPOWER.md, 2026-03-18)
THINK_START = 151667  # <think> — single special token
THINK_END = 151668    # </think> — single special token
STEP_TOKEN = 9520     # 'step'
OPEN_ANGLE = 27       # '<'
CLOSE_SLASH = 522     # '</'
CLOSE_ANGLE = 29      # '>'
STEP_NUM_TOKENS = {16: 1, 17: 2, 18: 3, 19: 4}  # token_id → step_number

# Which qualities belong to which step region
STEP_QUALITIES: dict[int, list[str]] = {
    1: ["q_format_tags", "q_tag_content", "q_node_in_prompt", "q_node_format", "q_node_length"],
    2: ["q_chain_s2_refs_s1"],
    3: ["q_chain_s3_refs_s2", "q_self_consistency"],
    4: ["q_step4_valid_json", "q_step4_has_keys", "q_existence_correct",
        "q_archetype_correct", "q_node_f1"],
}

GLOBAL_QUALITIES = ["q_eos_correct"]


def segment_completion(token_ids: list[int]) -> list[str]:
    """Assign each token to a region: THINK, STEP_1..4, FORMAT, OTHER.

    Uses token ID patterns, not decoded text. Fast, exact.
    Returns list[str] of region labels, same length as token_ids.
    """
    regions = ["OTHER"] * len(token_ids)
    current = "OTHER"
    n = len(token_ids)

    i = 0
    while i < n:
        tid = token_ids[i]

        # Think block boundaries (single special tokens — reliable)
        if tid == THINK_START:
            current = "THINK"
            regions[i] = "THINK"
            i += 1
            continue

        if tid == THINK_END:
            regions[i] = "THINK"
            current = "OTHER"
            i += 1
            continue

        # Step opening tag: < step N ...>
        # Pattern: OPEN_ANGLE(27) STEP_TOKEN(9520) NUM(16-19) ...content... CLOSE_ANGLE(29)
        if tid == OPEN_ANGLE and i + 2 < n:
            if token_ids[i + 1] == STEP_TOKEN and token_ids[i + 2] in STEP_NUM_TOKENS:
                step = STEP_NUM_TOKENS[token_ids[i + 2]]
                current = f"STEP_{step}"
                # Mark the opening tag tokens as FORMAT
                j = i
                while j < n and token_ids[j] != CLOSE_ANGLE:
                    regions[j] = "FORMAT"
                    j += 1
                if j < n:
                    regions[j] = "FORMAT"  # the > itself
                i = j + 1
                continue

        # Step closing tag: </ step N ...>
        if tid == CLOSE_SLASH and i + 2 < n:
            if token_ids[i + 1] == STEP_TOKEN and token_ids[i + 2] in STEP_NUM_TOKENS:
                # Mark closing tag tokens as FORMAT
                j = i
                while j < n and token_ids[j] != CLOSE_ANGLE:
                    regions[j] = "FORMAT"
                    j += 1
                if j < n:
                    regions[j] = "FORMAT"
                current = "OTHER"
                i = j + 1
                continue

        regions[i] = current
        i += 1

    return regions
