"""Tests for token segmentation (Step 0d — segmentation)."""

from qgre.segments import (
    CLOSE_ANGLE, CLOSE_SLASH, OPEN_ANGLE, STEP_NUM_TOKENS, STEP_TOKEN,
    THINK_END, THINK_START, segment_completion,
)


def test_segment_known_sequence(known_token_ids):
    """Hand-crafted tokens → correct region labels."""
    regions = segment_completion(known_token_ids)

    # <think> block: tokens 0-4
    assert regions[0] == "THINK"   # THINK_START
    assert regions[1] == "THINK"
    assert regions[2] == "THINK"
    assert regions[3] == "THINK"
    assert regions[4] == "THINK"   # THINK_END

    # <step1_extraction> opening tag: tokens 5-9 → FORMAT
    assert regions[5] == "FORMAT"  # <
    assert regions[6] == "FORMAT"  # step
    assert regions[7] == "FORMAT"  # 1
    assert regions[8] == "FORMAT"  # _extraction
    assert regions[9] == "FORMAT"  # >

    # step 1 content: tokens 10-12 → STEP_1
    assert regions[10] == "STEP_1"
    assert regions[11] == "STEP_1"
    assert regions[12] == "STEP_1"

    # </step1_extraction> closing tag: tokens 13-17 → FORMAT
    assert regions[13] == "FORMAT"
    assert regions[14] == "FORMAT"
    assert regions[15] == "FORMAT"
    assert regions[16] == "FORMAT"
    assert regions[17] == "FORMAT"

    # <step2_shared_context> opening tag: tokens 18-23 → FORMAT
    assert regions[18] == "FORMAT"

    # step 2 content: tokens 24-25 → STEP_2
    assert regions[24] == "STEP_2"
    assert regions[25] == "STEP_2"


def test_segment_no_think_block():
    """nothink template: no THINK tokens → no THINK regions."""
    tokens = [
        OPEN_ANGLE, STEP_TOKEN, 16, 94842, CLOSE_ANGLE,  # <step1_extraction>
        100, 101,  # content
        CLOSE_SLASH, STEP_TOKEN, 16, 94842, CLOSE_ANGLE,  # </step1_extraction>
    ]
    regions = segment_completion(tokens)

    assert "THINK" not in regions
    assert "STEP_1" in regions
    assert regions[5] == "STEP_1"
    assert regions[6] == "STEP_1"


def test_segment_malformed_tags():
    """Missing closing tag → region extends to end as current type."""
    tokens = [
        OPEN_ANGLE, STEP_TOKEN, 16, 94842, CLOSE_ANGLE,  # <step1_extraction>
        100, 101, 102, 103,  # content with no closing tag
    ]
    regions = segment_completion(tokens)

    # Content tokens should all be STEP_1 since no closing tag resets to OTHER
    assert regions[5] == "STEP_1"
    assert regions[6] == "STEP_1"
    assert regions[7] == "STEP_1"
    assert regions[8] == "STEP_1"


def test_segment_all_four_steps():
    """Full completion with steps 1-4 → 4 STEP regions."""
    tokens = []
    for step_num_tok, step_num in STEP_NUM_TOKENS.items():
        # Opening tag
        tokens.extend([OPEN_ANGLE, STEP_TOKEN, step_num_tok, 9999, CLOSE_ANGLE])
        # Content
        tokens.extend([100 + step_num, 200 + step_num])
        # Closing tag
        tokens.extend([CLOSE_SLASH, STEP_TOKEN, step_num_tok, 9999, CLOSE_ANGLE])

    regions = segment_completion(tokens)

    # Each step has 12 tokens: 5 opening + 2 content + 5 closing
    for step_num in range(1, 5):
        assert f"STEP_{step_num}" in regions, f"STEP_{step_num} not found"

    # Format tokens exist (opening/closing tags)
    assert regions.count("FORMAT") == 4 * 10  # 4 steps × (5 open + 5 close)


def test_segment_empty_input():
    """Empty token list → empty regions."""
    assert segment_completion([]) == []


def test_segment_no_step_tags():
    """Tokens with no step structure → all OTHER."""
    regions = segment_completion([100, 101, 102, 103])
    assert all(r == "OTHER" for r in regions)
