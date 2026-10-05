import json
from pathlib import Path

from faster_whisper import WhisperModel


def transcribe(video: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_file = output_dir / "transcript.json"

    if json_file.exists():
        print("[transcribe] using cached transcript")
        return json.loads(json_file.read_text())

    print("[transcribe] loading Whisper...")

    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
    )

    segments, info = model.transcribe(
        str(video),
        vad_filter=True,
        beam_size=5,
    )

    result = []

    for segment in segments:
        result.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
        })

    json_file.write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )

    # Also create an SRT
    write_srt(result, output_dir / "transcript.srt")

    return result


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)

    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def write_srt(segments, output):
    with output.open("w", encoding="utf-8") as f:
        for index, segment in enumerate(segments, 1):
            f.write(f"{index}\n")
            f.write(
                f"{format_time(segment['start'])} --> "
                f"{format_time(segment['end'])}\n"
            )
            f.write(segment["text"] + "\n\n")


def write_srt_for_clip(
    segments: list,
    clip_start: float,
    clip_end: float,
    output: Path,
) -> None:
    """
    Extract transcript segments that overlap [clip_start, clip_end],
    shift all timestamps so the clip starts at 0:00:00,000, and write
    a valid SRT file.

    Parameters
    ----------
    segments   : list of {"start", "end", "text"} dicts from transcribe()
    clip_start : start of the clip in source-video seconds
    clip_end   : end of the clip in source-video seconds
    output     : Path to the output .srt file
    """

    clip_segments = []

    for seg in segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])

        # Skip segments entirely outside the clip window.
        if seg_end <= clip_start or seg_start >= clip_end:
            continue

        # Clamp to clip boundaries, then shift to 0-based.
        shifted_start = max(seg_start, clip_start) - clip_start
        shifted_end = min(seg_end, clip_end) - clip_start

        if shifted_end <= shifted_start:
            continue

        clip_segments.append(
            {
                "start": shifted_start,
                "end": shifted_end,
                "text": seg["text"].strip(),
            }
        )

    write_srt(clip_segments, output)

