"""
thumbnails.py — Automatic thumbnail generator for rendered Shorts.

Generates a clean 1080x1920 (9:16) PNG thumbnail for each Short:
  1. Selects the most informative, non-blank frame from the rendered Short
     (using variance and edge/contrast heuristics across candidate timestamps).
  2. Composes a modern technical/educational cover:
     - Background video frame with a subtle dark gradient/overlay for legibility.
     - Formatted title with auto-wrapping, drop-shadow / high-contrast backing,
       and optional accent pill.
     - Fallbacks gracefully to hook or clean frame-only if metadata is missing.
  3. Uses standard OpenCV + NumPy (no extra heavy dependencies needed).
"""

from pathlib import Path
from typing import NamedTuple
import cv2
import numpy as np

# Thumbnail canvas specifications (matches 9:16 Short resolution)
THUMBNAIL_WIDTH = 1080
THUMBNAIL_HEIGHT = 1920

# Safe margins for mobile 9:16 screens (avoiding YouTube / IG UI overlays)
MARGIN_X = 80
MARGIN_TOP = 260
MAX_TEXT_WIDTH = THUMBNAIL_WIDTH - (2 * MARGIN_X)

# Visual styling defaults
FONT_FACE = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 1.65
FONT_THICKNESS = 3
LINE_SPACING = 28
MAX_TITLE_LINES = 4


class FrameCandidate(NamedTuple):
    timestamp: float
    frame: np.ndarray
    score: float


def _evaluate_frame_quality(frame: np.ndarray) -> float:
    """
    Evaluate visual quality of a candidate frame.
    Higher score indicates rich, well-lit, sharp content (code/editor/diagram).
    Rejects completely black, pure white, or excessively blurry frames.
    """
    if frame is None or frame.size == 0:
        return -1.0

    # Downscale for fast heuristic evaluation
    h, w = frame.shape[:2]
    scale = 320 / w
    small = cv2.resize(frame, (320, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    mean_val = float(np.mean(gray))
    # Reject extreme black or blown-out white frames
    if mean_val < 15.0 or mean_val > 240.0:
        return 0.0

    # Variance of Laplacian measures sharpness / edge frequency
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Standard deviation of intensities measures contrast
    std_val = float(np.std(gray))

    # Composite score
    score = (laplacian_var * 0.6) + (std_val * 1.5)
    return score


def extract_best_frame(video_path: Path, num_candidates: int = 7) -> np.ndarray | None:
    """
    Extract the best visual frame from a rendered video clip.
    Avoids transitional artifacts at the very beginning (0-5%) or end (95-100%).
    """
    if not video_path.exists():
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not fps or fps <= 0 or frame_count <= 0:
        duration = 5.0
    else:
        duration = frame_count / fps

    # Target interior candidate timestamps (between 10% and 85% of duration)
    start_t = max(0.2, duration * 0.10)
    end_t = max(start_t + 0.5, duration * 0.85)
    timestamps = np.linspace(start_t, end_t, max(2, num_candidates))

    best_cand: FrameCandidate | None = None

    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        score = _evaluate_frame_quality(frame)
        if best_cand is None or score > best_cand.score:
            best_cand = FrameCandidate(timestamp=t, frame=frame.copy(), score=score)

    # Fallback to the first available readable frame if heuristics didn't pick one
    if best_cand is None or best_cand.frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret and frame is not None:
            best_cand = FrameCandidate(timestamp=0.0, frame=frame.copy(), score=0.0)

    cap.release()
    return best_cand.frame if best_cand else None


def _wrap_text(
    text: str,
    font_face: int,
    font_scale: float,
    thickness: int,
    max_width: int,
    max_lines: int = MAX_TITLE_LINES,
) -> list[str]:
    """
    Word-wrap text to fit within max_width. Truncates with ellipsis if
    text exceeds max_lines.
    """
    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current_line = []

    for word in words:
        test_line = " ".join(current_line + [word])
        (w, _), _ = cv2.getTextSize(test_line, font_face, font_scale, thickness)
        if w <= max_width or not current_line:
            current_line.append(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            if len(lines) == max_lines - 1:
                break

    if current_line and len(lines) < max_lines:
        lines.append(" ".join(current_line))
    elif current_line:
        # We truncated
        pass

    # If there are remaining words that did not fit, add ellipsis to the last line
    consumed_count = sum(len(line.split()) for line in lines)
    if consumed_count < len(words) and lines:
        last = lines[-1]
        while last:
            test = last + "..."
            (w, _), _ = cv2.getTextSize(test, font_face, font_scale, thickness)
            if w <= max_width:
                lines[-1] = test
                break
            # Drop the last word
            parts = last.rsplit(" ", 1)
            if len(parts) == 1:
                lines[-1] = "..."
                break
            last = parts[0]

    return lines


def compose_thumbnail(
    background_frame: np.ndarray,
    title: str = "",
    accent_text: str = "CODE SHORT",
    overlay_strength: float = 0.55,
) -> np.ndarray:
    """
    Compose the final 1080x1920 thumbnail image with background and typography.
    """
    # 1. Resize or crop background frame to exactly 1080x1920
    h, w = background_frame.shape[:2]
    if (w, h) != (THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT):
        # Scale preserving aspect ratio and center crop
        scale = max(THUMBNAIL_WIDTH / w, THUMBNAIL_HEIGHT / h)
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        resized = cv2.resize(background_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)

        # Center crop to 1080x1920
        start_x = (scaled_w - THUMBNAIL_WIDTH) // 2
        start_y = (scaled_h - THUMBNAIL_HEIGHT) // 2
        canvas = resized[start_y : start_y + THUMBNAIL_HEIGHT, start_x : start_x + THUMBNAIL_WIDTH].copy()
    else:
        canvas = background_frame.copy()

    # 2. Add top and middle gradient darkening for clear readability
    # Define vertical darkness ramp (darkest at top, subtle at bottom)
    y_indices = np.linspace(0, 1, THUMBNAIL_HEIGHT, dtype=np.float32)[:, None, None]
    # Strong darkness from top to y=800, tapering down
    mask = np.clip(1.0 - (y_indices * 1.3), 0.0, 1.0) * overlay_strength
    canvas = (canvas.astype(np.float32) * (1.0 - mask)).astype(np.uint8)


    clean_title = (title or "").strip()
    if not clean_title:
        # Return clean frame with subtle styling
        return canvas

    # Sanitize title for ASCII OpenCV text rendering (replacing curly quotes, etc.)
    ascii_title = (
        clean_title.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("—", "-")
        .replace("–", "-")
    )
    # Strip characters outside printable ASCII range to prevent OpenCV artifacting
    safe_title = "".join(c if (32 <= ord(c) <= 126) else " " for c in ascii_title)
    safe_title = " ".join(safe_title.split())

    # Wrap title
    lines = _wrap_text(
        safe_title,
        font_face=FONT_FACE,
        font_scale=FONT_SCALE,
        thickness=FONT_THICKNESS,
        max_width=MAX_TEXT_WIDTH,
        max_lines=MAX_TITLE_LINES,
    )

    if not lines:
        return canvas

    # Measure total text block height
    line_metrics = [
        cv2.getTextSize(line, FONT_FACE, FONT_SCALE, FONT_THICKNESS) for line in lines
    ]
    line_height = line_metrics[0][0][1]
    total_text_h = sum(m[0][1] for m in line_metrics) + (len(lines) - 1) * LINE_SPACING

    # Draw an accent category pill above title if provided
    current_y = MARGIN_TOP
    if accent_text:
        pill_text = accent_text.upper()
        pill_scale = 0.8
        pill_thick = 2
        (pw, ph), pb = cv2.getTextSize(pill_text, cv2.FONT_HERSHEY_DUPLEX, pill_scale, pill_thick)
        pill_pad_x, pill_pad_y = 20, 12
        p_x1 = MARGIN_X
        p_y1 = current_y - ph - pill_pad_y
        p_x2 = p_x1 + pw + (2 * pill_pad_x)
        p_y2 = current_y + pill_pad_y

        # Cyan/Blue accent pill background (BGR: 220, 140, 20)
        cv2.rectangle(canvas, (p_x1, p_y1), (p_x2, p_y2), (220, 140, 20), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            pill_text,
            (p_x1 + pill_pad_x, current_y + (ph // 4)),
            cv2.FONT_HERSHEY_DUPLEX,
            pill_scale,
            (255, 255, 255),
            pill_thick,
            cv2.LINE_AA,
        )
        current_y += ph + 45

    # Semi-transparent text backing card for high contrast
    card_x1 = max(0, MARGIN_X - 30)
    card_y1 = max(0, current_y - 25)
    card_x2 = min(THUMBNAIL_WIDTH, MARGIN_X + MAX_TEXT_WIDTH + 30)
    card_y2 = min(THUMBNAIL_HEIGHT, current_y + total_text_h + 35)

    sub_rect = canvas[card_y1:card_y2, card_x1:card_x2]
    dark_card = np.zeros_like(sub_rect)
    # Blend 60% black card
    cv2.addWeighted(sub_rect, 0.40, dark_card, 0.60, 0, sub_rect)

    # Draw left border accent stripe (height of text card)
    cv2.rectangle(canvas, (card_x1, card_y1), (card_x1 + 10, card_y2), (250, 180, 40), -1)

    # Draw text lines with drop shadow and clean white foreground
    text_baseline_y = current_y + line_height
    for line, ((lw, lh), base) in zip(lines, line_metrics):
        # Drop shadow (black)
        cv2.putText(
            canvas,
            line,
            (MARGIN_X + 3, text_baseline_y + 3),
            FONT_FACE,
            FONT_SCALE,
            (0, 0, 0),
            FONT_THICKNESS + 2,
            cv2.LINE_AA,
        )
        # Main text (pure white)
        cv2.putText(
            canvas,
            line,
            (MARGIN_X, text_baseline_y),
            FONT_FACE,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS,
            cv2.LINE_AA,
        )
        text_baseline_y += lh + LINE_SPACING

    return canvas


def generate_thumbnail(
    video_path: Path,
    output_path: Path,
    metadata: dict | None = None,
    num_candidates: int = 7,
    overwrite: bool = True,
) -> Path | None:
    """
    Generate a 1080x1920 PNG thumbnail for a given Short video.

    Parameters
    ----------
    video_path     : Path to the generated short MP4.
    output_path    : Destination path for the thumbnail PNG.
    metadata       : Optional clip metadata dictionary containing title/hook.
    num_candidates : Number of candidate frames to sample from the video.
    overwrite      : Overwrite if the thumbnail file already exists.

    Returns
    -------
    Path to the generated thumbnail PNG, or None if generation failed.
    """
    if not video_path.exists():
        print(f"[thumbnail] Video not found: {video_path}")
        return None

    if output_path.exists() and not overwrite:
        print(f"[thumbnail] Using existing thumbnail: {output_path}")
        return output_path

    # Extract best candidate frame
    frame = extract_best_frame(video_path, num_candidates=num_candidates)
    if frame is None:
        print(f"[thumbnail] Could not extract frame from: {video_path}")
        return None

    # Determine title from metadata (title -> hook -> empty)
    title = ""
    if metadata and isinstance(metadata, dict):
        content = metadata.get("content", {})
        title = content.get("title") or metadata.get("title") or content.get("hook") or ""

    # Compose final thumbnail
    thumbnail = compose_thumbnail(frame, title=title)

    # Save as PNG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), thumbnail, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not success:
        print(f"[thumbnail] Failed to write image: {output_path}")
        return None

    return output_path
