import argparse
import hashlib
import json
from pathlib import Path

from .download import download_video
from .transcribe import transcribe
from .candidates import create_candidates
from .ollama import find_highlights
from .clips import create_clip


def save_json(path: Path, data):
    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description=("Create YouTube Shorts using " "Whisper + Ollama + FFmpeg")
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

    print("\n[1/4] Downloading video...")

    source = download_video(
        args.url,
        data_dir,
    )

    print("\n[2/4] Transcribing...")

    transcript = transcribe(
        source,
        data_dir,
    )

    print(f"[transcribe] " f"{len(transcript)} transcript segments")

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

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save complete metadata for all Shorts.
    highlights_file = output_dir / "highlights.json"

    save_json(
        highlights_file,
        highlights,
    )

    if not highlights:
        print()
        print("[error] No valid clips selected.")
        print(f"[error] Check: {highlights_file}")
        raise SystemExit(1)

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
            f"[{clip['score']}/10]"
        )

        print(f"   Title: {clip['title']}")

    print("\n[4/4] Creating Shorts...")

    for index, clip in enumerate(
        highlights,
        1,
    ):
        output = output_dir / f"short-{index:02}.mp4"

        print(f"\n[render " f"{index}/{len(highlights)}]")

        create_clip(
            source,
            output,
            clip["start"],
            clip["end"],
        )

        # Save metadata specifically for this Short.
        short_metadata = {
            "short_number": index,
            "video": output.name,
            "source": {
                "url": args.url,
                "video_id": video_id,
            },
            "clip": {
                "candidate_id": clip["candidate_id"],
                "start": clip["start"],
                "end": clip["end"],
                "duration": (clip["end"] - clip["start"]),
                "score": clip["score"],
            },
            "content": {
                "title": clip["title"],
                "hook": clip["hook"],
                "reason": clip["reason"],
                "description": clip["description"],
                "tags": clip["tags"],
                "hashtags": clip["hashtags"],
            },
            "youtube": clip["youtube"],
            "instagram": clip["instagram"],
        }

        metadata_file = output_dir / f"short-{index:02}.json"

        save_json(
            metadata_file,
            short_metadata,
        )

        print(f"Created: {output}")

        print(f"Metadata: {metadata_file}")

    print("\nDone.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
