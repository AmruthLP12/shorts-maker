import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .download import download_video
from .transcribe import transcribe, write_srt_for_clip
from .candidates import create_candidates
from .ollama import find_highlights_from_transcript, find_highlights
from .clips import create_clip
from .report import write_markdown_report



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
        description=(
            "Create YouTube Shorts using Whisper + Ollama + FFmpeg.\n"
            "Pass a YouTube URL or a local video file path."
        )
    )

    parser.add_argument(
        "input",
        help="YouTube video URL  OR  path to a local video file",
    )

    parser.add_argument(
        "--clips",
        type=int,
        default=5,
        help="Number of Shorts to create (default: 5)",
    )

    parser.add_argument(
        "--srt-only",
        action="store_true",
        help=(
            "Only transcribe the video and write an SRT file. "
            "Skips clip selection and rendering."
        ),
    )

    args = parser.parse_args()

    # ----------------------------------------------------------------
    # Detect: local file  vs  YouTube URL
    # ----------------------------------------------------------------

    local_path = Path(args.input)
    is_local = local_path.exists() and local_path.is_file()

    # Derive a stable, short ID from the canonical input string.
    id_source = str(local_path.resolve()) if is_local else args.input
    video_id = hashlib.sha1(id_source.encode()).hexdigest()[:10]

    data_dir = Path("data") / video_id

    # ----------------------------------------------------------------
    # Step 1 — Acquire video
    # ----------------------------------------------------------------

    if is_local:
        print(f"\n[1/4] Using local video: {local_path.resolve()}")
        source = local_path.resolve()
    else:
        print("\n[1/4] Downloading video...")
        source = download_video(args.input, data_dir)

    # ----------------------------------------------------------------
    # Step 2 — Transcribe
    # ----------------------------------------------------------------

    print("\n[2/4] Transcribing...")

    transcript = transcribe(source, data_dir)

    print(f"[transcribe] {len(transcript)} transcript segments")

    # ----------------------------------------------------------------
    # --srt-only mode: save full SRT and exit
    # ----------------------------------------------------------------

    if args.srt_only:
        # The full-video SRT is already written to data_dir by transcribe().
        full_srt = data_dir / "transcript.srt"

        if is_local:
            # Also copy it right next to the source video.
            dest = local_path.with_suffix(".srt")
            shutil.copy2(full_srt, dest)
            print(f"\n[srt-only] Saved: {dest}")
        else:
            print(f"\n[srt-only] Saved: {full_srt}")

        return

    # ----------------------------------------------------------------
    # Step 3 — Find highlights with Ollama
    # ----------------------------------------------------------------

    print("\n[3/4] Finding Shorts...")

    # Primary: send the full transcript to the LLM and let it choose
    # exact start/end timestamps. This avoids the fixed-window problem
    # where good content that spans window boundaries gets missed.
    highlights = find_highlights_from_transcript(
        transcript,
        num_clips=args.clips,
    )

    if not highlights:
        # Fallback: old sliding-window approach.
        # The LLM picks from pre-defined 45-second candidates instead.
        print(
            "[warn] Full-transcript pass returned no clips. "
            "Falling back to sliding-window approach..."
        )

        candidates = create_candidates(transcript, window=45, step=30)

        print(f"[candidates] {len(candidates)} candidate windows")

        highlights = find_highlights(candidates, num_clips=args.clips)

    output_dir = Path("output") / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full highlights metadata.
    highlights_file = output_dir / "highlights.json"
    save_json(highlights_file, highlights)

    if not highlights:
        print()
        print("[error] No valid clips selected.")
        print(f"[error] Check: {highlights_file}")
        raise SystemExit(1)

    print("\nSelected clips:")

    for index, clip in enumerate(highlights, 1):
        duration = clip["end"] - clip["start"]

        print(
            f"{index}. "
            f"{clip['start']:.1f}s → "
            f"{clip['end']:.1f}s "
            f"({duration:.1f}s) "
            f"[{clip['score']}/10]"
        )

        print(f"   Title: {clip['title']}")

    # ----------------------------------------------------------------
    # Step 4 — Render Shorts
    # ----------------------------------------------------------------

    print("\n[4/4] Creating Shorts...")

    # Label shown in metadata / reports.
    source_label = str(source) if is_local else args.input

    for index, clip in enumerate(highlights, 1):
        output = output_dir / f"short-{index:02d}.mp4"
        srt_file = output_dir / f"short-{index:02d}.srt"
        metadata_file = output_dir / f"short-{index:02d}.json"
        md_file = output_dir / f"short-{index:02d}.md"

        print(f"\n[render {index}/{len(highlights)}]")

        # 1. Write clip-relative SRT (timestamps start at 0:00:00,000).
        write_srt_for_clip(
            transcript,
            clip["start"],
            clip["end"],
            srt_file,
        )

        print(f"Subtitles (SRT): {srt_file}")

        # 2. Render the Short with the SRT embedded as a soft subtitle track.
        create_clip(
            source,
            output,
            clip["start"],
            clip["end"],
            srt_path=srt_file,
        )

        # 3. Build and save JSON metadata.
        short_metadata = {
            "short_number": index,
            "video": output.name,
            "source": {
                "input": source_label,
                "video_id": video_id,
            },
            "clip": {
                "candidate_id": clip.get("candidate_id"),
                "start": clip["start"],
                "end": clip["end"],
                "duration": clip["end"] - clip["start"],
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

        save_json(metadata_file, short_metadata)

        # 4. Write human-readable Markdown report.
        write_markdown_report(
            md_file,
            index,
            source_label,
            short_metadata,
        )

        print(f"Created:          {output}")
        print(f"Subtitles (SRT):  {srt_file}")
        print(f"Metadata (JSON):  {metadata_file}")
        print(f"Metadata (MD):    {md_file}")

    print("\nDone.")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
