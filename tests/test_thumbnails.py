"""
tests/test_thumbnails.py — Tests for automatic thumbnail generation.
"""

from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from shorts_maker.thumbnails import (
    THUMBNAIL_WIDTH,
    THUMBNAIL_HEIGHT,
    _evaluate_frame_quality,
    _wrap_text,
    extract_best_frame,
    compose_thumbnail,
    generate_thumbnail,
)


def _make_dummy_video(path: Path, width: int = 1080, height: int = 1920, duration_sec: float = 3.0, fps: int = 30):
    """Create a synthetic MP4 using OpenCV VideoWriter."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    total_frames = int(duration_sec * fps)

    for i in range(total_frames):
        # Create patterned frames with changing intensity
        frame = np.full((height, width, 3), (i * 2) % 200 + 30, dtype=np.uint8)
        # Draw some high-contrast shapes in the center to simulate screen content
        cv2.putText(
            frame,
            f"Frame {i}",
            (100, 500),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (255, 255, 255),
            3,
        )
        cv2.rectangle(frame, (200, 600), (800, 1200), (0, 200, 255), -1)
        writer.write(frame)

    writer.release()


class TestThumbnailQualityHeuristics:
    def test_evaluate_frame_quality_rejects_empty_or_solid(self):
        assert _evaluate_frame_quality(None) == -1.0
        assert _evaluate_frame_quality(np.array([])) == -1.0

        # Solid black
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        assert _evaluate_frame_quality(black) == 0.0

        # Solid white
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        assert _evaluate_frame_quality(white) == 0.0

    def test_evaluate_frame_quality_prefers_contrast(self):
        # Low contrast vs high contrast text
        low_contrast = np.full((300, 300, 3), 120, dtype=np.uint8)
        high_contrast = low_contrast.copy()
        cv2.putText(high_contrast, "HIGH CONTRAST", (30, 150), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 3)

        score_low = _evaluate_frame_quality(low_contrast)
        score_high = _evaluate_frame_quality(high_contrast)
        assert score_high > score_low


class TestTextWrapping:
    def test_wrap_short_text(self):
        lines = _wrap_text("Short Title", cv2.FONT_HERSHEY_DUPLEX, 1.0, 2, max_width=800)
        assert len(lines) == 1
        assert lines[0] == "Short Title"

    def test_wrap_long_text_into_multiple_lines(self):
        long_title = "This is a very long technical title that should wrap onto multiple distinct lines nicely"
        lines = _wrap_text(long_title, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3, max_width=400, max_lines=4)
        assert len(lines) > 1
        assert len(lines) <= 4

    def test_wrap_truncates_with_ellipsis_if_excessive(self):
        huge_title = "Word " * 50
        lines = _wrap_text(huge_title, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3, max_width=300, max_lines=3)
        assert len(lines) == 3
        assert lines[-1].endswith("...")


class TestCompositionAndResolution:
    def test_compose_thumbnail_exact_dimensions(self):
        # 16:9 input frame scaled to 9:16
        bg_16_9 = np.full((720, 1280, 3), 100, dtype=np.uint8)
        composed = compose_thumbnail(bg_16_9, title="Testing Title Composition")
        assert composed.shape == (THUMBNAIL_HEIGHT, THUMBNAIL_WIDTH, 3)

    def test_compose_thumbnail_with_empty_title(self):
        bg = np.full((1920, 1080, 3), 80, dtype=np.uint8)
        composed = compose_thumbnail(bg, title="")
        assert composed.shape == (THUMBNAIL_HEIGHT, THUMBNAIL_WIDTH, 3)

    def test_compose_handles_unicode_safely(self):
        bg = np.full((1920, 1080, 3), 80, dtype=np.uint8)
        unicode_title = "“Python Tips” & Tricks — What’s New in 3.14? 🚀"
        composed = compose_thumbnail(bg, title=unicode_title)
        assert composed.shape == (THUMBNAIL_HEIGHT, THUMBNAIL_WIDTH, 3)


class TestThumbnailGenerationPipeline:
    def test_generate_thumbnail_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test_short.mp4"
            out_thumb = Path(tmpdir) / "test_short-thumbnail.png"
            _make_dummy_video(video_path, width=1080, height=1920, duration_sec=2.0)

            metadata = {
                "content": {
                    "title": "Build Fast Python Apps",
                    "hook": "Learn Python today"
                }
            }

            res = generate_thumbnail(video_path, out_thumb, metadata=metadata)
            assert res is not None
            assert out_thumb.exists()

            # Verify saved PNG properties
            img = cv2.imread(str(out_thumb))
            assert img is not None
            assert img.shape == (THUMBNAIL_HEIGHT, THUMBNAIL_WIDTH, 3)

    def test_generate_thumbnail_fallback_to_hook(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test_short.mp4"
            out_thumb = Path(tmpdir) / "test_short-thumbnail.png"
            _make_dummy_video(video_path, width=1080, height=1920, duration_sec=1.5)

            metadata = {
                "content": {
                    "hook": "Fallback Hook Title"
                }
            }

            res = generate_thumbnail(video_path, out_thumb, metadata=metadata)
            assert res is not None
            assert out_thumb.exists()

    def test_generate_thumbnail_missing_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test_short.mp4"
            out_thumb = Path(tmpdir) / "test_short-thumbnail.png"
            _make_dummy_video(video_path, width=1080, height=1920, duration_sec=1.0)

            # Metadata is None
            res = generate_thumbnail(video_path, out_thumb, metadata=None)
            assert res is not None
            assert out_thumb.exists()
            img = cv2.imread(str(out_thumb))
            assert img.shape == (THUMBNAIL_HEIGHT, THUMBNAIL_WIDTH, 3)

    def test_generate_thumbnail_very_short_video(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "tiny.mp4"
            out_thumb = Path(tmpdir) / "tiny-thumbnail.png"
            _make_dummy_video(video_path, duration_sec=0.2, fps=10)

            res = generate_thumbnail(video_path, out_thumb)
            assert res is not None
            assert out_thumb.exists()

    def test_generate_thumbnail_missing_video_returns_none(self):
        missing_video = Path("/nonexistent/path/to/video.mp4")
        out_thumb = Path("/tmp/out-thumbnail.png")
        res = generate_thumbnail(missing_video, out_thumb)
        assert res is None

    def test_generate_thumbnail_rerun_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "test_short.mp4"
            out_thumb = Path(tmpdir) / "test_short-thumbnail.png"
            _make_dummy_video(video_path, duration_sec=1.0)

            # Pre-create the thumbnail with a dummy file
            out_thumb.write_text("existing")
            mtime_before = out_thumb.stat().st_mtime

            # Call with overwrite=False
            res = generate_thumbnail(video_path, out_thumb, overwrite=False)
            assert res == out_thumb
            assert out_thumb.read_text() == "existing"
