"""
tests/tier3_combinations/test_pair04_virality_hooktitle.py
Pairwise Test 04: F06 (Virality Detection) + F09 (Hook Title Generation)
Interaction Seam: Segment analysis produces 3–9 word hook title within character budget.
"""

from unittest.mock import patch
import pytest

from tests.fixtures.synthetic import make_sample_transcript, make_sample_viral_segment

try:
    import clippa.analysis as analysis
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


@pytest.mark.tier3
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.analysis not implemented yet")
def test_pair04_virality_generates_valid_hook_title():
    """TC-PAIR-04: Segment analysis produces valid hook title adhering to constraints."""
    transcript = make_sample_transcript(60000)
    mock_segment = make_sample_viral_segment(0, 30000, hook_title="The Greatest Hidden Truth Exposed")

    with patch("clippa.analysis.extract_viral_segments", return_value=[mock_segment]):
        segments = analysis.extract_viral_segments(transcript, max_clips=1)
        seg = segments[0]
        words = seg.hook_title.split()
        assert 3 <= len(words) <= 9
        assert len(seg.hook_title) <= 64
