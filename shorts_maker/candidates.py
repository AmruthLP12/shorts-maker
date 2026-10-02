def create_candidates(
    segments,
    window=45,
    step=30,
):
    """
    Create fixed-length candidate windows from the transcript.

    Example:

        0-45
        30-75
        60-105
        ...

    Ollama evaluates these windows instead of choosing timestamps itself.
    """

    if not segments:
        return []

    video_end = float(segments[-1]["end"])

    candidates = []

    start = 0.0

    while start + window <= video_end:
        end = start + window

        text_parts = []

        for segment in segments:
            segment_start = float(segment["start"])
            segment_end = float(segment["end"])

            if segment_end <= start:
                continue

            if segment_start >= end:
                break

            text_parts.append(segment["text"].strip())

        text = " ".join(text_parts).strip()

        if text:
            candidates.append(
                {
                    "id": len(candidates) + 1,
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "text": text,
                }
            )

        start += step

    return candidates
