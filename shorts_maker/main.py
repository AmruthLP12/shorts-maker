import argparse
import hashlib
import json
from pathlib import Path

from .download import download_video
from .transcribe import transcribe
from .candidates import create_candidates
from .ollama import find_highlights
from .clips import create_clip


def main():
    parser = argparse.ArgumentParser(
        description="Create YouTube Shorts using Whisper + Ollama + FFmpeg"
    )

    parser.add_argument(
        "url",
        help="YouTube video URL",
    )

    parser.add_argument(
        "--clips",
        type=int,
        default=5,
        help="Number of Shorts to create",
    )

    args = parser.parse_args()

    video_id = hashlib.sha1(args.url.encode()).hexdigest()[:10]

    data_dir = Path("data") / video_id

    output_dir = Path("output") / video_id

    # -----------------------------------------------------
    # 1. Download
    # -----------------------------------------------------

    print("\n[1/4] Downloading video...")

    source = download_video(
        args.url,
        data_dir,
    )

    # -----------------------------------------------------
    # 2. Transcribe
    # -----------------------------------------------------

    print("\n[2/4] Transcribing...")

    transcript = transcribe(
        source,
        data_dir,
    )

    print(f"[transcribe] " f"{len(transcript)} transcript segments")

    # -----------------------------------------------------
    # 3. Find Shorts
    # -----------------------------------------------------

    print("\n[3/4] Finding Shorts...")

    candidates = create_candidates(
        transcript,
        window=45,
        step=30,
    )

    print(f"[candidates] Created " f"{len(candidates)} candidate windows")

    for candidate in candidates:
        print(
            f"  {candidate['id']:02d}. "
            f"{candidate['start']:.0f}s → "
            f"{candidate['end']:.0f}s"
        )

    highlights = find_highlights(
        candidates,
        num_clips=args.clips,
    )

    # -----------------------------------------------------
    # Save highlights
    # -----------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    highlights_file = output_dir / "highlights.json"

    highlights_file.write_text(
        json.dumps(
            highlights,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # No clips
    # -----------------------------------------------------

    if not highlights:
        print()
        print("[error] No valid clips selected.")

        print(f"[error] Check:" f" {highlights_file}")

        raise SystemExit(1)

    # -----------------------------------------------------
    # Display selected clips
    # -----------------------------------------------------

    print("\nSelected clips:")

    for index, clip in enumerate(
        highlights,
        1,
    ):
        duration = clip["end"] - clip["start"]

        print(
            f"{index}. "
            f"{clip['start']:.1f}s → "
            f"{clip['end']:.1f}s "
            f"({duration:.1f}s) "
            f"[{clip['score']}/10] "
            f"{clip['title']}"
        )

    # -----------------------------------------------------
    # 4. Create Shorts
    # -----------------------------------------------------

    print("\n[4/4] Creating Shorts...")

    for index, clip in enumerate(
        highlights,
        1,
    ):
        output = output_dir / f"short-{index:02}.mp4"

        print(f"\n[render {index}/{len(highlights)}]")

        create_clip(
            source,
            output,
            clip["start"],
            clip["end"],
        )

        print(f"Created: {output}")

    print("\nDone.")

    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
