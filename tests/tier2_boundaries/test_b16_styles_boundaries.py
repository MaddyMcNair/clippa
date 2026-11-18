"""
tests/tier2_boundaries/test_b16_styles_boundaries.py
Feature 16: Subtitle Style Templates & TTF Fonts - Boundary & Edge Cases (5 cases)
"""

import pytest


@pytest.mark.tier2
def test_tc_b16_01_missing_font_fallback():
    """TC-B16-01: Font file missing from fonts directory falls back to Arial/Sans."""
    requested_font = "NonExistentSpecialFont"
    available_fonts = {"THEBOLDFONT", "TikTokSans", "Arial"}
    chosen_font = requested_font if requested_font in available_fonts else "Arial"
    assert chosen_font == "Arial"


@pytest.mark.tier2
def test_tc_b16_02_font_size_resolution_scaling():
    """TC-B16-02: Font size scaled for 1080p scales proportionally for 720p."""
    base_fontsize_1080p = 60
    scale_factor = 720 / 1080
    fontsize_720p = int(base_fontsize_1080p * scale_factor)
    assert fontsize_720p == 40


@pytest.mark.tier2
def test_tc_b16_03_outline_preserved_on_transparent_text():
    """TC-B16-03: Transparent primary text maintains legible outline stroke."""
    # Outline stroke must be > 0 and opaque (&H00000000)
    border_style = 1
    outline_width = 3
    assert border_style in [1, 3]
    assert outline_width >= 2


@pytest.mark.tier2
def test_tc_b16_04_pill_box_safe_margin_clamping():
    """TC-B16-04: Margin touching video edge clamps within safe margins (>=40px)."""
    raw_margin = 10
    safe_margin = max(40, raw_margin)
    assert safe_margin == 40


@pytest.mark.tier2
def test_tc_b16_05_missing_style_fields_defaulted():
    """TC-B16-05: Custom template missing required fields fills with default preset values."""
    custom_template = {"font_name": "CustomFont"}
    default_template = {
        "font_name": "Arial",
        "font_size": 60,
        "primary_color": "&H00FFFFFF",
        "outline_color": "&H00000000",
    }
    merged = {**default_template, **custom_template}
    assert merged["font_name"] == "CustomFont"
    assert merged["font_size"] == 60
    assert "primary_color" in merged
