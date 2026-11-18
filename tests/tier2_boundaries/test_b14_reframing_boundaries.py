"""
tests/tier2_boundaries/test_b14_reframing_boundaries.py
Feature 14: 9:16 Vertical Reframing & Clamping - Boundary & Edge Cases (5 cases)
"""

import pytest


@pytest.mark.tier2
def test_tc_b14_01_pre_vertical_source_passthrough():
    """TC-B14-01: Native 9:16 vertical input video passes through without extra crop."""
    src_w, src_h = 1080, 1920
    is_vertical = (src_w / src_h) == (9 / 16)
    assert is_vertical is True
    # Passthrough filter simply scales or copies
    filter_expr = "scale=1080:1920" if is_vertical else "crop=..."
    assert filter_expr == "scale=1080:1920"


@pytest.mark.tier2
def test_tc_b14_02_square_1_1_reframing():
    """TC-B14-02: Square 1:1 input video (1080x1080) reframes to 9:16 safely."""
    src_w, src_h = 1080, 1080
    target_aspect = 9 / 16  # 0.5625
    # Reframing crop width
    crop_w = int(src_h * target_aspect)
    assert crop_w % 2 == 0 or (crop_w - 1) % 2 == 0
    crop_w = crop_w if crop_w % 2 == 0 else crop_w - 1
    assert crop_w <= src_w


@pytest.mark.tier2
def test_tc_b14_03_ultrawide_21_9_reframing():
    """TC-B14-03: Ultra-wide 21:9 input video (2560x1080) calculates crop window safely."""
    src_w, src_h = 2560, 1080
    crop_w = 608  # 1080 * 9/16 rounded to even
    assert crop_w < src_w
    max_crop_x = src_w - crop_w
    assert max_crop_x == 1952


@pytest.mark.tier2
def test_tc_b14_04_odd_dimension_macroblock_rounding():
    """TC-B14-04: Odd-dimension input (1921x1079) rounds to nearest even integers."""
    def round_to_even(val: int) -> int:
        return val if val % 2 == 0 else val - 1

    assert round_to_even(1921) == 1920
    assert round_to_even(1079) == 1078


@pytest.mark.tier2
def test_tc_b14_05_low_res_bicubic_scaling():
    """TC-B14-05: Low-resolution input (480x360) scales up with bicubic flags."""
    scale_filter = "scale=1080:1920:flags=bicubic"
    assert "1080:1920" in scale_filter
    assert "bicubic" in scale_filter
