import subprocess
from pathlib import Path

import cv2
import numpy as np

OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920


def find_best_crop_x(
    source: Path,
    start: float,
    end: float,
) -> int:
    """
    Find the most useful horizontal crop position for a
    16:9 screen recording being converted to 9:16.

    Uses edge/text density across several frames. This works
    reasonably well for coding screens, terminals, editors,
    browser windows, etc.
    """

    cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        print("[crop] Could not open video. Using center crop.")
        return 0

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30

    # Width required for a 9:16 crop from the source.
    crop_width = int(video_height * 9 / 16)

    if crop_width >= video_width:
        cap.release()
        return 0

    max_x = video_width - crop_width

    # Sample roughly 8 frames throughout the selected clip.
    sample_count = 8

    if end <= start:
        times = [start]
    else:
        times = np.linspace(
            start,
            max(start, end - 0.1),
            sample_count,
        )

    # Candidate crop positions.
    #
    # More positions = more accurate but slightly more CPU work.
    step = max(20, crop_width // 20)

    candidate_positions = list(range(0, max_x + 1, step))

    if max_x not in candidate_positions:
        candidate_positions.append(max_x)

    scores = {x: 0.0 for x in candidate_positions}

    for timestamp in times:
        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            float(timestamp) * 1000,
        )

        success, frame = cap.read()

        if not success:
            continue

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        # Resize for faster analysis.
        scale = 640 / gray.shape[1]

        small = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

        # Edge detection is useful for:
        # - text
        # - code
        # - terminal
        # - UI boundaries
        edges = cv2.Canny(
            small,
            80,
            160,
        )

        small_width = small.shape[1]

        scale_x = small_width / video_width

        for x in candidate_positions:
            sx = int(x * scale_x)
            sw = max(1, int(crop_width * scale_x))

            region = edges[
                :,
                sx : min(sx + sw, edges.shape[1]),
            ]

            if region.size == 0:
                continue

            # Percentage of pixels containing edges.
            edge_density = np.mean(region > 0)

            scores[x] += float(edge_density)

    cap.release()

    if not scores:
        return max_x // 2

    best_x = max(
        scores,
        key=scores.get,
    )

    # Smooth the result slightly toward the center.
    #
    # This prevents an occasional tiny UI element from causing
    # an extreme crop.
    center_x = max_x / 2

    best_x = int(best_x * 0.75 + center_x * 0.25)

    best_x = max(
        0,
        min(best_x, max_x),
    )

    print(f"[crop] Source: " f"{video_width}x{video_height}")

    print(f"[crop] 9:16 crop width: " f"{crop_width}px")

    print(f"[crop] Selected X position: " f"{best_x}px")

    return best_x


def create_clip(
    source: Path,
    output: Path,
    start: float,
    end: float,
):
    duration = end - start

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("[crop] Analyzing screen content...")

    crop_x = find_best_crop_x(
        source,
        start,
        end,
    )

    cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    crop_width = int(video_height * 9 / 16)

    # Make sure crop stays inside the source.
    crop_x = max(
        0,
        min(
            crop_x,
            video_width - crop_width,
        ),
    )

    print(
        f"[crop] Using region: "
        f"x={crop_x}, "
        f"width={crop_width}, "
        f"height={video_height}"
    )

    # Crop the original video first.
    #
    # Then scale the cropped region to exactly
    # 1080x1920.
    video_filter = (
        f"crop={crop_width}:{video_height}:{crop_x}:0,"
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:"
        f"flags=lanczos"
    )

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        str(source),
        "-t",
        str(duration),
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(output),
    ]

    print("[crop] Rendering 1080x1920...")

    subprocess.run(
        command,
        check=True,
    )
