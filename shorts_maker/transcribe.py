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
