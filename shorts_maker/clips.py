import subprocess
from pathlib import Path


def create_clip(
    source: Path,
    output: Path,
    start: float,
    end: float,
):
    duration = end - start

    output.parent.mkdir(parents=True, exist_ok=True)

    filter_complex = (
        "[0:v]split=2[bgsrc][fgsrc];"
        "[bgsrc]"
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,"
        "boxblur=20[bg];"
        "[fgsrc]"
        "scale=1080:-2:force_original_aspect_ratio=decrease"
        "[fg];"
        "[bg][fg]"
        "overlay=(W-w)/2:(H-h)/2[v]"
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
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "0:a?",
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

    subprocess.run(command, check=True)
