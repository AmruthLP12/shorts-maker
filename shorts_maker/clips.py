"""
clips.py — Content-aware 9:16 reframing for screen recordings.

Strategy
--------
Instead of picking a single crop X for the entire clip, we:

  1. Sample representative frames from the clip at a configurable interval.
  2. For each frame compute a per-column "content score" combining:
       - bright-on-dark text density  (weighted most heavily — the dominant
         signal for coding screens / terminals)
       - local variance / detail
       - edge density (weighted least — syntax highlighting creates noise)
  3. Score every candidate crop window by summing the column scores inside it,
     then penalise windows whose left or right edge cuts through a detected
     high-density content column (to avoid slicing code mid-line).
  4. Collect the best X per sampled frame into a time-series.
  5. Smooth the time-series with a Gaussian kernel to remove jitter.
  6. When neighbouring keyframes differ by less than STABILITY_THRESHOLD px,
     snap them to the same value (prefer stability over micro-movements).
  7. Build an FFmpeg crop=w:h:x:0 filter where x is expressed as a piecewise-
     linear keyframed expression using FFmpeg's `if()` / `between()` syntax.
  8. One FFmpeg render pass per clip.

Fallback
--------
If fewer than 2 frames are readable, or the source is already portrait,
a documented centre-crop is used and a warning is printed.
"""

import subprocess
import time
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

# ─── Output ────────────────────────────────────────────────────────────────────
OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920

# ─── Frame-sampling ────────────────────────────────────────────────────────────
# How often to sample the clip for crop-position analysis (seconds).
DEFAULT_SAMPLE_INTERVAL = 5.0

# ─── Scoring weights ───────────────────────────────────────────────────────────
# These three must sum to 1.0.
W_TEXT = 0.55    # bright-text-on-dark (code, terminal output)
W_DETAIL = 0.30  # local variance / detail density
W_EDGE = 0.15    # Canny edge density (kept small to reduce UI-border noise)

# ─── Temporal smoothing ────────────────────────────────────────────────────────
# Gaussian sigma (in frames) for smoothing the per-frame best-X series.
SMOOTH_SIGMA = 1.5

# If two adjacent keyframes differ by ≤ this many px, snap to the same value.
STABILITY_THRESHOLD = 30  # px in source-video coordinates

# ─── Analysis resolution ───────────────────────────────────────────────────────
# Frames are shrunk to this width for scoring (faster analysis, same signal).
ANALYSIS_WIDTH = 480

# ─── Boundary penalty ──────────────────────────────────────────────────────────
# Fraction of the crop-window width that is inspected near each edge.
# If the average content score in this border zone exceeds the threshold,
# the crop candidate is penalised.
BOUNDARY_ZONE = 0.08   # 8 % of crop width on each side
BOUNDARY_PENALTY = 0.5  # multiply score by this when cutting through content


# ─── Data types ────────────────────────────────────────────────────────────────

class FrameSample(NamedTuple):
    timestamp: float   # seconds in source video
    best_x: int        # best crop-left in source-video pixel coordinates
    score: float       # composite score of the chosen window


class SourceInfo(NamedTuple):
    width: int
    height: int
    fps: float
    crop_width: int   # width of a 9:16 crop extracted from this source
    max_x: int        # maximum allowed left edge (width - crop_width)


# ─── Source inspection ─────────────────────────────────────────────────────────

def _probe_source(source: Path) -> SourceInfo | None:
    """
    Open the video and read its dimensions and frame-rate.
    Returns None if the file cannot be opened.
    """
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"[crop] Cannot open: {source}")
        cap.release()
        return None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if w <= 0 or h <= 0:
        print(f"[crop] Invalid source dimensions ({w}x{h})")
        return None

    if not fps or fps <= 0:
        fps = 30.0

    crop_w = int(h * 9 / 16)
    max_x = w - crop_w

    return SourceInfo(
        width=w,
        height=h,
        fps=fps,
        crop_width=crop_w,
        max_x=max_x,
    )


# ─── Per-frame content scoring ─────────────────────────────────────────────────

def _content_column_scores(frame_bgr: np.ndarray, target_width: int) -> np.ndarray:
    """
    Return a 1-D array (length = target_width) of normalised content scores,
    one value per column at analysis resolution.

    Three signals are combined:

    1. Text score — bright pixels on a locally dark background.
       Code editors and terminals are typically light text on a dark
       background (or vice-versa in light mode).  We detect this by
       comparing each pixel's brightness to its neighbourhood mean:
       high contrast = likely text.

    2. Detail score — local variance of a small patch around each pixel.
       High variance = detailed content (code, diagrams).

    3. Edge score — fraction of Canny-detected edge pixels per column.
       Kept low-weight because UI elements (borders, separators) also
       have high edge density and would otherwise dominate.
    """
    # Resize to analysis width, preserving aspect ratio.
    h, w = frame_bgr.shape[:2]
    scale = target_width / w
    new_h = max(1, int(h * scale))
    small = cv2.resize(
        frame_bgr,
        (target_width, new_h),
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # --- Text score ---------------------------------------------------------
    # Compute local mean with a box filter; compare each pixel to its mean.
    kernel = 9
    local_mean = cv2.blur(gray, (kernel, kernel))
    contrast = np.abs(gray - local_mean)
    # A pixel is "text-like" when it is bright AND strongly contrasting with
    # its background.  We multiply normalised brightness × normalised contrast.
    norm_bright = gray / 255.0
    norm_contrast = np.clip(contrast / 80.0, 0, 1)
    text_map = norm_bright * norm_contrast
    text_cols = text_map.mean(axis=0)  # shape: (target_width,)

    # --- Detail score -------------------------------------------------------
    # Local variance via a square integral image trick.
    blurred = cv2.blur(gray, (5, 5))
    diff = gray - blurred
    variance_map = np.abs(diff)
    detail_cols = variance_map.mean(axis=0) / 128.0
    detail_cols = np.clip(detail_cols, 0, 1)

    # --- Edge score ---------------------------------------------------------
    gray_u8 = gray.astype(np.uint8)
    edges = cv2.Canny(gray_u8, 60, 120)
    edge_cols = (edges > 0).mean(axis=0).astype(np.float32)

    # --- Composite ----------------------------------------------------------
    composite = (
        W_TEXT   * text_cols
        + W_DETAIL * detail_cols
        + W_EDGE   * edge_cols
    )

    return composite.astype(np.float32)


def _score_candidates(
    col_scores: np.ndarray,
    candidate_xs_analysis: list[int],
    crop_w_analysis: int,
    analysis_width: int,
) -> dict[int, float]:
    """
    Given per-column scores at analysis resolution, compute the integral
    (mean) score for every candidate crop window and apply boundary penalties.

    Returns a mapping {candidate_x_analysis: score}.
    """
    # Prefix sums for O(1) window integrals.
    prefix = np.concatenate([[0.0], np.cumsum(col_scores)])

    def window_mean(x: int) -> float:
        x0 = max(0, x)
        x1 = min(len(prefix) - 1, x + crop_w_analysis)
        if x1 <= x0:
            return 0.0
        return float(prefix[x1] - prefix[x0]) / (x1 - x0)

    def boundary_mean(x: int) -> float:
        zone = max(1, int(crop_w_analysis * BOUNDARY_ZONE))
        # Left edge zone
        left_x0 = max(0, x)
        left_x1 = min(left_x0 + zone, len(prefix) - 1)
        # Right edge zone
        right_x1 = min(x + crop_w_analysis, len(prefix) - 1)
        right_x0 = max(0, right_x1 - zone)

        scores = []
        if left_x1 > left_x0:
            scores.append(float(prefix[left_x1] - prefix[left_x0]) / (left_x1 - left_x0))
        if right_x1 > right_x0:
            scores.append(float(prefix[right_x1] - prefix[right_x0]) / (right_x1 - right_x0))
        return float(np.mean(scores)) if scores else 0.0

    results = {}
    for cx in candidate_xs_analysis:
        base = window_mean(cx)
        boundary = boundary_mean(cx)
        # Penalise crops that cut through dense content at their edges.
        if boundary > 0.15:
            base *= BOUNDARY_PENALTY
        results[cx] = base

    return results


def _best_x_from_frame(
    frame_bgr: np.ndarray,
    info: SourceInfo,
    analysis_width: int = ANALYSIS_WIDTH,
) -> tuple[int, float]:
    """
    Return (best_x_source_px, score) for one frame.
    best_x_source_px is clamped to [0, info.max_x].
    """
    scale = analysis_width / info.width
    crop_w_a = max(1, int(info.crop_width * scale))
    max_x_a = max(0, analysis_width - crop_w_a)

    # Candidate positions at analysis resolution (~every 2 % of crop width).
    step_a = max(2, crop_w_a // 25)
    cxs = list(range(0, max_x_a + 1, step_a))
    if max_x_a not in cxs:
        cxs.append(max_x_a)

    col_scores = _content_column_scores(frame_bgr, analysis_width)
    scored = _score_candidates(col_scores, cxs, crop_w_a, analysis_width)

    if not scored:
        return info.max_x // 2, 0.0

    best_x_a = max(scored, key=scored.get)
    best_score = scored[best_x_a]

    # Map back to source coordinates.
    best_x_src = int(best_x_a / scale)
    best_x_src = max(0, min(best_x_src, info.max_x))

    return best_x_src, best_score


# ─── Temporal analysis ─────────────────────────────────────────────────────────

def _sample_frames(
    source: Path,
    info: SourceInfo,
    start: float,
    end: float,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
) -> list[FrameSample]:
    """
    Sample frames from [start, end] at `sample_interval` second intervals.
    Always includes a frame near the start and one near the end.

    Returns a list of FrameSample, sorted by timestamp.
    """
    if end <= start:
        end = start + 0.1

    duration = end - start
    n_samples = max(2, int(duration / sample_interval) + 1)
    timestamps = np.linspace(start, end - 0.05, n_samples).tolist()

    # Always include start + end anchors.
    if timestamps[0] > start + 0.5:
        timestamps.insert(0, start)
    if timestamps[-1] < end - 0.5:
        timestamps.append(max(start, end - 0.1))

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print("[crop] Cannot open source for frame sampling.")
        return []

    samples = []
    for ts in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        best_x, score = _best_x_from_frame(frame, info)
        samples.append(FrameSample(timestamp=ts, best_x=best_x, score=score))

    cap.release()
    return samples


def _smooth_positions(
    samples: list[FrameSample],
    sigma: float = SMOOTH_SIGMA,
    stability_threshold: int = STABILITY_THRESHOLD,
) -> list[tuple[float, int]]:
    """
    Apply Gaussian smoothing to the best-X series and snap near-identical
    neighbours to the same value.

    Returns a list of (timestamp_seconds, x_source_px) keyframes.
    """
    if not samples:
        return []

    if len(samples) == 1:
        return [(samples[0].timestamp, samples[0].best_x)]

    xs = np.array([s.best_x for s in samples], dtype=float)

    # Gaussian kernel over the sample indices.
    n = len(xs)
    if n >= 3 and sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter1d  # type: ignore[import]
            xs = gaussian_filter1d(xs, sigma=sigma)
        except (ImportError, ModuleNotFoundError):
            # scipy not available — use a simple box filter instead.
            kernel = max(1, int(sigma * 2))
            pad = np.pad(xs, kernel, mode="edge")
            xs = np.convolve(
                pad,
                np.ones(kernel * 2 + 1) / (kernel * 2 + 1),
                mode="valid",
            )
            xs = xs[:n]


    xs = np.round(xs).astype(int)

    # Stability: snap near-identical adjacent keyframes.
    for i in range(1, len(xs)):
        if abs(int(xs[i]) - int(xs[i - 1])) <= stability_threshold:
            xs[i] = xs[i - 1]

    return [(samples[i].timestamp, int(xs[i])) for i in range(n)]


# ─── FFmpeg expression builder ────────────────────────────────────────────────

def _build_crop_x_expr(
    keyframes: list[tuple[float, int]],
    clip_start: float,
    info: SourceInfo,
) -> str:
    """
    Build an FFmpeg crop-x expression that linearly interpolates between
    keyframe values.

    For a single keyframe (or all identical): return the constant.

    For multiple keyframes: use if(between(t, t0, t1), lerp(x0, x1, ...), ...)
    structure.

    Times in the expression are relative to the start of the rendered clip
    (FFmpeg `t` starts at 0 in the output clip).
    """
    if not keyframes:
        return str(info.max_x // 2)

    # Convert absolute timestamps → clip-relative.
    kf = [(ts - clip_start, max(0, min(x, info.max_x))) for ts, x in keyframes]

    # Deduplicate consecutive identical values to keep the expression small.
    deduped: list[tuple[float, int]] = [kf[0]]
    for ts, x in kf[1:]:
        if x != deduped[-1][1]:
            deduped.append((ts, x))

    if len(deduped) == 1:
        return str(deduped[0][1])

    # Build a piecewise-linear if() chain.
    # FFmpeg: t is time in seconds of the current frame in the output.
    # We clamp at both ends.
    parts = []
    for i in range(len(deduped) - 1):
        t0, x0 = deduped[i]
        t1, x1 = deduped[i + 1]

        if t1 <= t0:
            continue

        if x0 == x1:
            segment = f"if(between(t,{t0:.3f},{t1:.3f}),{x0}"
        else:
            # Linear interpolation: x0 + (x1-x0)*(t-t0)/(t1-t0)
            dx = x1 - x0
            dt = t1 - t0
            segment = (
                f"if(between(t,{t0:.3f},{t1:.3f}),"
                f"round({x0}+({dx})*((t-{t0:.3f})/{dt:.3f}))"
            )

        parts.append(segment)

    if not parts:
        return str(deduped[0][1])

    # Before first keyframe: hold first value.
    first_t, first_x = deduped[0]
    before = f"if(lt(t,{first_t:.3f}),{first_x}"

    # After last keyframe: hold last value.
    last_t, last_x = deduped[-1]
    after = str(last_x)

    # Nest: before, *segments, after
    # Structure: if(cond, val, if(cond, val, ... after...))
    expr = after
    for seg in reversed(parts):
        expr = seg + "," + expr + ")"
    expr = before + "," + expr + ")"

    # Clamp to [0, max_x].
    expr = f"clip({expr},0,{info.max_x})"

    return expr


# ─── Public API ────────────────────────────────────────────────────────────────

def create_clip(
    source: Path,
    output: Path,
    start: float,
    end: float,
    srt_path: Path | None = None,
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
) -> None:
    """
    Render a 1080×1920 Short from `source[start:end]`.

    Parameters
    ----------
    source          : Path to the source video file.
    output          : Destination MP4 path.
    start           : Clip start time in source seconds.
    end             : Clip end time in source seconds.
    srt_path        : Optional subtitle file; embedded as a soft subtitle track.
    sample_interval : How often to sample frames for crop analysis (seconds).
    """
    t0 = time.monotonic()

    output.parent.mkdir(parents=True, exist_ok=True)

    duration = end - start
    if duration <= 0:
        raise ValueError(f"[crop] Invalid clip: start={start} end={end}")

    # ── 1. Probe source ──────────────────────────────────────────────────────
    info = _probe_source(source)
    if info is None:
        raise RuntimeError(f"[crop] Cannot read source: {source}")

    print(
        f"[crop] Source: {info.width}x{info.height} @ {info.fps:.1f} fps  "
        f"| clip: {start:.1f}s→{end:.1f}s ({duration:.1f}s)"
    )

    # ── 2. Portrait / square source: passthrough crop ────────────────────────
    if info.max_x <= 0:
        print("[crop] Source is already portrait or square — using full-width crop.")
        crop_x_expr = "0"
        _render(source, output, start, duration, info, "0", srt_path)
        return

    # ── 3. Sample frames ────────────────────────────────────────────────────
    print(f"[crop] Analysing frames (interval={sample_interval:.1f}s)…")
    samples = _sample_frames(source, info, start, end, sample_interval)
    print(f"[crop] Sampled {len(samples)} frames")

    # ── 4. Temporal smoothing ────────────────────────────────────────────────
    if len(samples) < 2:
        # Fallback: documented centre-crop.
        print("[crop] Too few readable frames — using centre crop (fallback).")
        cx = info.max_x // 2
        keyframes = [(start, cx)]
    else:
        keyframes = _smooth_positions(samples)

    # Print summary of detected positions.
    unique_xs = sorted({x for _, x in keyframes})
    print(
        f"[crop] Crop width: {info.crop_width}px  |  "
        f"X range: {min(unique_xs)}–{max(unique_xs)}px"
    )
    for ts, x in keyframes:
        rel = ts - start
        print(f"       t={rel:5.1f}s  x={x:5d}px")

    # ── 5. Build FFmpeg crop expression ─────────────────────────────────────
    crop_x_expr = _build_crop_x_expr(keyframes, start, info)

    # ── 6. Render ────────────────────────────────────────────────────────────
    _render(source, output, start, duration, info, crop_x_expr, srt_path)

    elapsed = time.monotonic() - t0
    print(
        f"[crop] Done → {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}  "
        f"(analysis+render: {elapsed:.1f}s)"
    )


def _render(
    source: Path,
    output: Path,
    start: float,
    duration: float,
    info: SourceInfo,
    crop_x_expr: str,
    srt_path: Path | None,
) -> None:
    """One FFmpeg render pass."""
    # crop=w:h:x:y  then scale to 1080×1920 with Lanczos.
    video_filter = (
        f"crop={info.crop_width}:{info.height}:{crop_x_expr}:0,"
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos"
    )

    use_subs = srt_path is not None and srt_path.exists()

    base = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(source),
    ]

    if use_subs:
        base += ["-i", str(srt_path)]

    base += ["-t", str(duration)]

    if use_subs:
        base += [
            "-map", "0:v",
            "-map", "0:a?",
            "-map", "1:s",
        ]

    base += [
        "-vf", video_filter,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "128k",
    ]

    if use_subs:
        base += [
            "-c:s", "mov_text",
            "-metadata:s:s:0", "language=eng",
        ]

    base += ["-movflags", "+faststart", str(output)]

    print(f"[crop] Rendering {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}…")
    subprocess.run(base, check=True)
