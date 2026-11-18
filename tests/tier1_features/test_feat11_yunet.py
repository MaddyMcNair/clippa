"""
tests/tier1_features/test_feat11_yunet.py
Feature 11: YuNet ONNX Face Detection
Category-Partition Tests (5 cases)
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

try:
    import clippa.reframing as reframing
    HAS_CLIPPA = True
except ImportError:
    HAS_CLIPPA = False


def _create_mock_cv2_detector(detected_faces=None):
    mock_detector = MagicMock()
    if detected_faces is None:
        detected_faces = np.array([[800, 300, 320, 400, 850, 420, 1050, 420, 950, 500, 870, 600, 1030, 600, 0.95]], dtype=np.float32)
    mock_detector.detect.return_value = (1, detected_faces)
    return mock_detector


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.reframing not implemented yet")
def test_tc_f11_01_yunet_model_loading():
    """TC-F11-01: Loads face detector model without external download."""
    load_fn = getattr(reframing, "load_face_detector", None)
    if load_fn:
        with patch("cv2.FaceDetectorYN.create", return_value=MagicMock()):
            detector = load_fn()
            assert detector is not None
    else:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with patch("cv2.FaceDetectorYN.create", return_value=_create_mock_cv2_detector()):
            faces = reframing.detect_faces(frame)
            assert isinstance(faces, list)


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.reframing not implemented yet")
def test_tc_f11_02_detects_frontal_face():
    """TC-F11-02: Detects frontal human face in frame and returns (x, y, w, h, score)."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    mock_faces = np.array([[800, 300, 320, 400, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95]], dtype=np.float32)
    mock_detector = _create_mock_cv2_detector(mock_faces)

    with patch("cv2.FaceDetectorYN.create", return_value=mock_detector):
        faces = reframing.detect_faces(frame)
        assert len(faces) == 1
        face = faces[0]
        # Check structure: either tuple or object with x, y, w, h, score
        x = getattr(face, "x", face[0])
        y = getattr(face, "y", face[1])
        w = getattr(face, "w", face[2])
        h = getattr(face, "h", face[3])
        score = getattr(face, "confidence", getattr(face, "score", face[4]))
        assert x >= 0 and y >= 0 and w > 0 and h > 0
        assert 0.0 <= score <= 1.0


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.reframing not implemented yet")
def test_tc_f11_03_downscales_wide_frames():
    """TC-F11-03: Downscales high-resolution frames wider than 960px for fast inference."""
    frame_4k = np.zeros((2160, 3840, 3), dtype=np.uint8)
    mock_faces = np.array([[1600, 600, 640, 800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.92]], dtype=np.float32)
    mock_detector = _create_mock_cv2_detector(mock_faces)

    with patch("cv2.FaceDetectorYN.create", return_value=mock_detector):
        faces = reframing.detect_faces(frame_4k)
        assert len(faces) == 1
        face = faces[0]
        x = getattr(face, "x", face[0])
        w = getattr(face, "w", face[2])
        assert x + w <= 3840


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.reframing not implemented yet")
def test_tc_f11_04_detects_multiple_faces():
    """TC-F11-04: Detects multiple faces in frame and returns candidate list."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    mock_faces = np.array([
        [400, 300, 250, 320, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.94],
        [1200, 300, 260, 330, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.91],
    ], dtype=np.float32)
    mock_detector = _create_mock_cv2_detector(mock_faces)

    with patch("cv2.FaceDetectorYN.create", return_value=mock_detector):
        faces = reframing.detect_faces(frame)
        assert len(faces) == 2


@pytest.mark.tier1
@pytest.mark.skipif(not HAS_CLIPPA, reason="clippa.reframing not implemented yet")
def test_tc_f11_05_blank_frame_returns_empty():
    """TC-F11-05: Returns empty list [] on blank black frame without throwing exceptions."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    mock_detector = MagicMock()
    mock_detector.detect.return_value = (0, None)

    with patch("cv2.FaceDetectorYN.create", return_value=mock_detector):
        faces = reframing.detect_faces(frame)
        assert isinstance(faces, list)
        assert len(faces) == 0
