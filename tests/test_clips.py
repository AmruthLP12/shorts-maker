"""
tests/test_clips.py — Unit tests for the content-aware crop implementation.

All tests use synthetically generated frames or tiny in-memory videos so no
real video file needs to be downloaded.  Where FFmpeg is needed (render tests),
the test generates a small synthetic MP4 using FFmpeg's lavfi source.
"""

import math
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from shorts_maker.clips import (
    STABILITY_THRESHOLD,
    FrameSample,
    SourceInfo,
    _best_x_from_frame,
    _build_crop_x_expr,
    _content_column_scores,
    _probe_source,
    _smooth_positions,
    create_clip,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_info(w: int, h: int, fps: float = 30.0) -> SourceInfo:
    """Construct a SourceInfo without opening a real file."""
    crop_w = int(h * 9 / 16)
    max_x = max(0, w - crop_w)
    return SourceInfo(width=w, height=h, fps=fps, crop_width=crop_w, max_x=max_x)


def _solid_frame(w: int, h: int, value: int = 128) -> np.ndarray:
    """Return a uniform grey BGR frame."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def _text_frame(
    w: int = 1920,
    h: int = 1080,
    text_x_start: int = 0,
    text_x_end: int | None = None,
) -> np.ndarray:
    """
    Dark background with bright vertical stripe simulating a code region.
    text_x_start / text_x_end define the stripe in source px.
    """
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    if text_x_end is None:
        text_x_end = text_x_start + w // 4
    frame[:, text_x_start:text_x_end] = 200  # bright "text" region
    return frame


def _make_synthetic_mp4(
    path: Path,
    w: int = 1920,
    h: int = 1080,
    duration: float = 6.0,
    fps: int = 30,
) -> Path:
    """
    Create a tiny synthetic MP4 using FFmpeg's lavfi testsrc.
    Requires FFmpeg to be on PATH.
    """
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"testsrc=size={w}x{h}:rate={fps}:duration={duration}",
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={duration}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "30",
            "-c:a", "aac",
            "-b:a", "64k",
            "-t", str(duration),
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _make_portrait_mp4(
    path: Path,
    w: int = 1080,
    h: int = 1920,
    duration: float = 4.0,
) -> Path:
    """Synthetic portrait-format MP4."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=blue:s={w}x{h}:r=30:d={duration}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "30",
            "-t", str(duration),
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


# ─── _content_column_scores ───────────────────────────────────────────────────

class TestContentColumnScores:

    def test_returns_correct_length(self):
        frame = _solid_frame(1920, 1080)
        scores = _content_column_scores(frame, 480)
        assert len(scores) == 480

    def test_dtype_is_float32(self):
        frame = _solid_frame(640, 480)
        scores = _content_column_scores(frame, 320)
        assert scores.dtype == np.float32

    def test_uniform_frame_low_score(self):
        """A solid grey frame has almost no content anywhere."""
        frame = _solid_frame(1920, 1080, value=128)
        scores = _content_column_scores(frame, 480)
        # Mean should be low (< 0.15); there's no text or detail.
        assert scores.mean() < 0.15

    def test_text_region_higher_score(self):
        """Bright stripe on dark background should score higher than dark areas."""
        frame = _text_frame(1920, 1080, text_x_start=800, text_x_end=1200)
        scores = _content_column_scores(frame, 480)
        # Map stripe to analysis coords
        start_a = int(800 / 1920 * 480)
        end_a = int(1200 / 1920 * 480)
        stripe_mean = scores[start_a:end_a].mean()
        outside_mean = np.concatenate([scores[:start_a], scores[end_a:]]).mean()
        assert stripe_mean > outside_mean


# ─── _best_x_from_frame ───────────────────────────────────────────────────────

class TestBestXFromFrame:

    def test_returns_within_bounds_16_9_source(self):
        info = _make_info(1920, 1080)
        frame = _solid_frame(1920, 1080)
        x, score = _best_x_from_frame(frame, info)
        assert 0 <= x <= info.max_x

    def test_content_left_of_center_favours_left(self):
        """Text block on the left should pull crop X toward 0."""
        info = _make_info(1920, 1080)
        # Text occupies columns 0–400 (left fifth of the frame).
        frame = _text_frame(1920, 1080, text_x_start=0, text_x_end=400)
        x, _ = _best_x_from_frame(frame, info)
        # Best X should be in the left half of the allowed range.
        assert x <= info.max_x // 2

    def test_content_right_of_center_favours_right(self):
        """Text block on the right should pull crop X toward max_x."""
        info = _make_info(1920, 1080)
        # Text occupies columns 1500–1920 (right ~22 %).
        frame = _text_frame(1920, 1080, text_x_start=1500, text_x_end=1920)
        x, _ = _best_x_from_frame(frame, info)
        assert x >= info.max_x // 2

    def test_returns_zero_for_portrait_source(self):
        """Portrait source has max_x=0; should return 0."""
        info = _make_info(1080, 1920)
        frame = _solid_frame(1080, 1920)
        x, _ = _best_x_from_frame(frame, info)
        assert x == 0

    def test_handles_unusual_aspect_ratio(self):
        """2.35:1 wide source — crop still within bounds."""
        info = _make_info(2560, 1080)
        frame = _solid_frame(2560, 1080)
        x, _ = _best_x_from_frame(frame, info)
        assert 0 <= x <= info.max_x


# ─── _smooth_positions ────────────────────────────────────────────────────────

class TestSmoothPositions:

    def _samples(self, xs: list[int], dt: float = 5.0) -> list[FrameSample]:
        return [
            FrameSample(timestamp=i * dt, best_x=x, score=1.0)
            for i, x in enumerate(xs)
        ]

    def test_single_sample_passthrough(self):
        s = self._samples([300])
        kf = _smooth_positions(s)
        assert len(kf) == 1
        assert kf[0][1] == 300

    def test_identical_samples_stable(self):
        s = self._samples([400, 400, 400, 400])
        kf = _smooth_positions(s)
        xs = [x for _, x in kf]
        assert all(x == xs[0] for x in xs)

    def test_small_jitter_snapped(self):
        """Changes smaller than STABILITY_THRESHOLD should be snapped."""
        delta = STABILITY_THRESHOLD - 1
        s = self._samples([500, 500 + delta, 500, 500 + delta])
        kf = _smooth_positions(s)
        xs = [x for _, x in kf]
        # After snapping, consecutive values should not vary by more than
        # STABILITY_THRESHOLD.
        for a, b in zip(xs, xs[1:]):
            assert abs(a - b) <= STABILITY_THRESHOLD

    def test_large_change_preserved(self):
        """A genuine large shift should survive smoothing."""
        s = self._samples([0, 0, 0, 1200, 1200, 1200])
        kf = _smooth_positions(s)
        xs = [x for _, x in kf]
        # After smoothing, early xs should be smaller than late xs.
        assert xs[0] < xs[-1]


# ─── _build_crop_x_expr ───────────────────────────────────────────────────────

class TestBuildCropXExpr:

    def _info(self) -> SourceInfo:
        return _make_info(1920, 1080)

    def test_single_keyframe_returns_constant(self):
        info = self._info()
        expr = _build_crop_x_expr([(0.0, 300)], clip_start=0.0, info=info)
        # Should just be the constant, not a complex expression.
        assert "300" in expr
        assert "if(" not in expr or "clip(" in expr  # may be wrapped in clip()

    def test_constant_keyframes_returns_simple(self):
        info = self._info()
        kf = [(0.0, 400), (5.0, 400), (10.0, 400)]
        expr = _build_crop_x_expr(kf, clip_start=0.0, info=info)
        assert "400" in expr

    def test_expression_clips_to_max_x(self):
        info = self._info()
        kf = [(0.0, 0), (10.0, info.max_x)]
        expr = _build_crop_x_expr(kf, clip_start=0.0, info=info)
        assert str(info.max_x) in expr

    def test_empty_keyframes_returns_fallback(self):
        info = self._info()
        expr = _build_crop_x_expr([], clip_start=0.0, info=info)
        # Should not crash; returns some valid integer string.
        int(expr)  # must be parseable as int

    def test_expression_is_string(self):
        info = self._info()
        kf = [(0.0, 100), (5.0, 800)]
        expr = _build_crop_x_expr(kf, clip_start=0.0, info=info)
        assert isinstance(expr, str)
        assert len(expr) > 0


# ─── _probe_source ────────────────────────────────────────────────────────────

class TestProbeSource:

    def test_nonexistent_file_returns_none(self):
        result = _probe_source(Path("/does/not/exist.mp4"))
        assert result is None

    def test_real_synthetic_video(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_synthetic_mp4(Path(td) / "test.mp4", w=1280, h=720)
            info = _probe_source(path)
            assert info is not None
            assert info.width == 1280
            assert info.height == 720
            assert info.fps > 0
            assert info.crop_width == int(720 * 9 / 16)
            assert info.max_x == 1280 - info.crop_width

    def test_portrait_video_max_x_is_zero(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_portrait_mp4(Path(td) / "portrait.mp4")
            info = _probe_source(path)
            assert info is not None
            assert info.max_x == 0


# ─── create_clip (integration) ────────────────────────────────────────────────

class TestCreateClip:

    def test_output_resolution_is_1080x1920(self):
        """The rendered Short must be exactly 1080×1920."""
        with tempfile.TemporaryDirectory() as td:
            src = _make_synthetic_mp4(Path(td) / "src.mp4", w=1920, h=1080, duration=8)
            out = Path(td) / "short.mp4"
            create_clip(src, out, start=1.0, end=6.0, sample_interval=2.0)
            assert out.exists()

            cap = cv2.VideoCapture(str(out))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            assert w == 1080
            assert h == 1920

    def test_portrait_source_output_resolution(self):
        """Portrait source should still yield a 1080×1920 output."""
        with tempfile.TemporaryDirectory() as td:
            src = _make_portrait_mp4(Path(td) / "portrait.mp4", duration=5)
            out = Path(td) / "short.mp4"
            create_clip(src, out, start=0.5, end=3.5, sample_interval=1.5)
            assert out.exists()

            cap = cv2.VideoCapture(str(out))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            assert w == 1080
            assert h == 1920

    def test_audio_stream_present(self):
        """Audio must survive the render."""
        with tempfile.TemporaryDirectory() as td:
            src = _make_synthetic_mp4(Path(td) / "src.mp4", duration=6)
            out = Path(td) / "short.mp4"
            create_clip(src, out, start=0.5, end=5.0, sample_interval=2.0)

            result = subprocess.run(
                ["ffprobe", "-v", "error",
                 "-select_streams", "a",
                 "-show_entries", "stream=codec_type",
                 "-of", "csv=p=0",
                 str(out)],
                capture_output=True, text=True,
            )
            assert "audio" in result.stdout

    def test_nonexistent_source_raises(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "short.mp4"
            with pytest.raises(RuntimeError, match="Cannot read source"):
                create_clip(Path("/no/such/file.mp4"), out, 0, 5)

    def test_1280x720_source(self):
        """Common screen-recording resolution."""
        with tempfile.TemporaryDirectory() as td:
            src = _make_synthetic_mp4(Path(td) / "src.mp4", w=1280, h=720, duration=7)
            out = Path(td) / "short.mp4"
            create_clip(src, out, start=1.0, end=6.0, sample_interval=2.0)

            cap = cv2.VideoCapture(str(out))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            assert w == 1080
            assert h == 1920

    def test_unusual_aspect_ratio(self):
        """2560×1080 ultrawide source — still renders correctly."""
        with tempfile.TemporaryDirectory() as td:
            src = _make_synthetic_mp4(Path(td) / "src.mp4", w=2560, h=1080, duration=6)
            out = Path(td) / "short.mp4"
            create_clip(src, out, start=0.5, end=4.5, sample_interval=2.0)

            cap = cv2.VideoCapture(str(out))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            assert w == 1080
            assert h == 1920

    def test_content_changes_during_clip(self):
        """
        Create a two-segment synthetic clip where the content region shifts
        from left to right midway through.  The output must still be rendered
        at 1080×1920 without errors.
        """
        with tempfile.TemporaryDirectory() as td:
            # Left-content half: green bar on the left.
            # Right-content half: red bar on the right.
            # Build with two lavfi color sources concatenated via concat filter.
            src = Path(td) / "shifting.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi",
                    "-i", "color=c=0x00ff00:s=1920x1080:r=30:d=5",
                    "-f", "lavfi",
                    "-i", "color=c=0xff0000:s=1920x1080:r=30:d=5",
                    "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]",
                    "-map", "[v]",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                    str(src),
                ],
                check=True,
                capture_output=True,
            )

            out = Path(td) / "short.mp4"
            create_clip(src, out, start=0.0, end=9.0, sample_interval=2.5)

            cap = cv2.VideoCapture(str(out))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            assert w == 1080
            assert h == 1920


# ─── Boundary / crop-position tests ───────────────────────────────────────────

class TestCropBoundaries:

    def test_best_x_never_exceeds_max_x(self):
        info = _make_info(1920, 1080)
        for _ in range(10):
            frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            x, _ = _best_x_from_frame(frame, info)
            assert 0 <= x <= info.max_x, f"x={x} out of [0, {info.max_x}]"

    def test_smoothed_keyframes_within_bounds(self):
        info = _make_info(1920, 1080)
        # Intentionally extreme raw values.
        samples = [
            FrameSample(i * 5.0, min(info.max_x, max(0, x)), 1.0)
            for i, x in enumerate([0, 5000, -100, info.max_x + 200, info.max_x])
        ]
        # Clamp inputs first (as the real code does).
        clamped = [
            FrameSample(s.timestamp, max(0, min(s.best_x, info.max_x)), s.score)
            for s in samples
        ]
        kf = _smooth_positions(clamped)
        for _, x in kf:
            assert 0 <= x <= info.max_x
