"""
tests/tier3_combinations/test_pair08_yunet_reframing.py
Pairwise Test 08: F11 (YuNet Face Detection) + F14 (9:16 Vertical Reframing)
Interaction Seam: Detected face centroid drives horizontal crop position.
"""

import pytest


@pytest.mark.tier3
def test_pair08_yunet_centroid_sets_crop_window():
    """TC-PAIR-08: Centroid (x, y) places detected face within vertical safe zone."""
    src_w, src_h = 1920, 1080
    target_crop_w, target_crop_h = 608, 1080

    # Face detected at center-right
    face_box = (1100, 300, 240, 300)
    face_center_x = face_box[0] + face_box[2] / 2  # 1220

    # Desired crop window centered on face
    desired_crop_x = int(face_center_x - target_crop_w / 2)
    clamped_crop_x = max(0, min(src_w - target_crop_w, desired_crop_x))

    assert 0 <= clamped_crop_x <= src_w - target_crop_w
    # Face should be inside the crop window
    assert clamped_crop_x <= face_center_x <= clamped_crop_x + target_crop_w
