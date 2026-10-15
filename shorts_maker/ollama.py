import json
import itertools
import sys
import threading
import time

import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3.2:latest"

# Transcript is sent as-is when under this size.
# Larger transcripts are split into CHUNK_DURATION_SECS windows.
MAX_TRANSCRIPT_CHARS = 80_000

# Chunking settings for long videos
CHUNK_DURATION_SECS = 600   # 10-minute windows
CHUNK_OVERLAP_SECS = 120    # 2-minute overlap so clips near boundaries aren't missed


# ---------------------------------------------------------
# Spinner
# ---------------------------------------------------------


def spinner(stop_event):
    frames = [
        "⠋",
        "⠙",
        "⠹",
        "⠸",
        "⠼",
        "⠴",
        "⠦",
        "⠧",
        "⠇",
        "⠏",
    ]

    start = time.time()

    for frame in itertools.cycle(frames):
        if stop_event.is_set():
            break

        elapsed = int(time.time() - start)

        sys.stdout.write(f"\r[ollama] {frame} Generating highlights... {elapsed}s")
        sys.stdout.flush()

        time.sleep(0.1)

    # Completely clear the spinner line.
    sys.stdout.write("\r" + (" " * 80) + "\r")
    sys.stdout.flush()


# ---------------------------------------------------------
# Ollama request
# ---------------------------------------------------------


def ask_ollama(prompt):
    print(f"[ollama] Model: {MODEL}")

    print("[ollama] Sending request...")

    stop_spinner = threading.Event()

    spinner_thread = threading.Thread(
        target=spinner,
        args=(stop_spinner,),
        daemon=True,
    )

    spinner_thread.start()

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                },
            },
            timeout=600,
        )

    except requests.RequestException as exc:
        print()
        print(f"[ollama] Request failed: {exc}")
        return None

    finally:
        stop_spinner.set()
        spinner_thread.join()

    print("[ollama] Generation finished.")

    if response.status_code != 200:
        print(f"[ollama] HTTP {response.status_code}")

        print(response.text[:2000])

        return None

    try:
        data = response.json()

    except ValueError:
        print("[ollama] Server returned invalid JSON.")

        print(response.text[:2000])

        return None

    raw = data.get(
        "response",
        "",
    )

    if raw is None:
        raw = ""

    raw = str(raw).strip()

    print(f"[ollama] Response characters: {len(raw)}")

    if not raw:
        print("[ollama] Empty model response.")

        print("[ollama] Full server response:")

        print(
            json.dumps(
                data,
                indent=2,
            )[:5000]
        )

        return None

    print("[ollama] Raw model response:")

    print(raw)

    return raw


# ---------------------------------------------------------
# Transcript formatting helpers
# ---------------------------------------------------------


def _ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS for LLM readability."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _parse_time(value) -> float:
    """
    Parse a time value that the LLM may return as:
      - a float/int (seconds)
      - a string like "83.5", "HH:MM:SS", or "MM:SS"
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        v = value.strip()
        if ":" in v:
            parts = v.split(":")
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s)
            if len(parts) == 2:
                m, s = parts
                return int(m) * 60 + float(s)
        return float(v)
    raise ValueError(f"Cannot parse time: {value!r}")


def _format_transcript(segments: list) -> str:
    """
    Format transcript segments as [HH:MM:SS] lines.
    This is much more informative for the LLM than pre-sliced text blocks.
    """
    lines = []
    for seg in segments:
        ts = _ts(float(seg["start"]))
        lines.append(f"[{ts}] {seg['text'].strip()}")
    return "\n".join(lines)


def _snap_timestamp(t: float, segments: list, tolerance: float = 15.0) -> float:
    """
    Snap a timestamp to the nearest segment start/end boundary within
    `tolerance` seconds. Returns the original value if nothing is close.

    This corrects slight LLM timestamp drift — the LLM may say "83.0" when
    the nearest segment boundary is 82.4.
    """
    best = t
    best_dist = float("inf")

    for seg in segments:
        for boundary in (float(seg["start"]), float(seg["end"])):
            dist = abs(boundary - t)
            if dist < best_dist:
                best_dist = dist
                best = boundary

    return best if best_dist <= tolerance else t


# ---------------------------------------------------------
# Response parser (shared by both single-pass and chunked)
# ---------------------------------------------------------


def _parse_response(raw: str, segments: list, num_clips: int) -> list:
    """
    Parse the LLM JSON response and return validated highlight dicts.

    Each returned dict has the same shape as find_highlights() output so
    main.py needs no changes.
    """
    # Try to decode JSON; attempt markdown-fence cleanup if it fails.
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[ollama] Invalid JSON: {exc}")
        cleaned = raw
        if "```json" in cleaned:
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        elif "```" in cleaned:
            cleaned = cleaned.replace("```", "").strip()
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            print("[ollama] Could not recover JSON.")
            return []

    selected = result.get("selected", [])
    if not isinstance(selected, list):
        print("[ollama] 'selected' is not a list.")
        return []

    video_start = float(segments[0]["start"]) if segments else 0.0
    video_end = float(segments[-1]["end"]) if segments else 0.0

    highlights = []

    for item in selected:
        if not isinstance(item, dict):
            continue

        # Parse start / end timestamps.
        try:
            start = _parse_time(item.get("start"))
            end = _parse_time(item.get("end"))
        except (TypeError, ValueError):
            print(f"[ollama] Bad timestamps in item: {item.get('start')!r} → {item.get('end')!r}")
            continue

        # Clamp to video bounds.
        start = max(video_start, start)
        end = min(video_end, end)

        # Duration gate: 15–90 seconds.
        duration = end - start
        if duration < 15:
            print(f"[ollama] Clip too short ({duration:.1f}s), skipping.")
            continue
        if duration > 90:
            # Trim to 60 s from the chosen start.
            end = start + 60.0

        # Snap to actual segment boundaries to avoid mid-word cuts.
        start = _snap_timestamp(start, segments)
        end = _snap_timestamp(end, segments)

        if end - start < 10:
            print(f"[ollama] Clip too short after snapping, skipping.")
            continue

        try:
            score = int(item.get("score", 0))
        except (TypeError, ValueError):
            score = 0

        if score < 1:
            continue

        tags = item.get("tags", [])
        hashtags = item.get("hashtags", [])
        youtube = item.get("youtube", {})
        instagram = item.get("instagram", {})

        if not isinstance(tags, list):
            tags = []
        if not isinstance(hashtags, list):
            hashtags = []
        if not isinstance(youtube, dict):
            youtube = {}
        if not isinstance(instagram, dict):
            instagram = {}

        highlights.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                # candidate_id is None for full-transcript selections;
                # the fallback path sets a real integer.
                "candidate_id": None,
                "score": score,
                "title": item.get("title", f"Clip at {_ts(start)}"),
                "reason": item.get("reason", ""),
                "hook": item.get("hook", ""),
                "description": item.get("description", ""),
                "tags": tags,
                "hashtags": hashtags,
                "youtube": {
                    "title": youtube.get("title", item.get("title", "")),
                    "description": youtube.get("description", item.get("description", "")),
                    "tags": youtube.get("tags", tags),
                    "hashtags": youtube.get("hashtags", hashtags),
                },
                "instagram": {
                    "caption": instagram.get("caption", item.get("description", "")),
                    "hashtags": instagram.get("hashtags", hashtags),
                },
            }
        )

    return highlights


def _dedup_and_limit(highlights: list, num_clips: int) -> list:
    """
    Sort by score (desc), remove overlapping clips, cap at num_clips,
    then restore chronological order.
    """
    highlights.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    for h in highlights:
        overlaps = any(
            h["start"] < e["end"] and h["end"] > e["start"]
            for e in selected
        )
        if overlaps:
            print(f"[ollama] Removing overlapping clip at {_ts(h['start'])}")
            continue
        selected.append(h)
        if len(selected) >= num_clips:
            break

    selected.sort(key=lambda x: x["start"])
    return selected


# ---------------------------------------------------------
# LLM prompt builder
# ---------------------------------------------------------

_SHARED_RULES = """\
Rules:
- start and end MUST be valid times taken directly from the transcript
- Each clip must be 30–60 seconds long
- Return at most {num_clips} clips
- Do not return markdown
- Do not return explanations outside the JSON object
- Do not invent facts not present in the transcript
- Do not use excessive emojis
- Do not use generic spam hashtags like #viral or #fyp
"""

_SHARED_FIELDS = """\
For each clip provide:
  start        — start time in seconds (float), taken from the transcript
  end          — end time in seconds (float), taken from the transcript
  score        — integer 1–10
  title        — concise title, 40–70 chars, no clickbait
  reason       — one sentence: why this works as a standalone Short
  hook         — compelling opening line / caption
  description  — YouTube description, 1–3 short paragraphs
  tags         — 5–12 YouTube search tags (no # symbol)
  hashtags     — 4–8 hashtags (include # symbol)
  youtube      — object with: title, description, tags, hashtags
  instagram    — object with: caption, hashtags
"""

_JSON_EXAMPLE = """\
{{
  "selected": [
    {{
      "start": 83.5,
      "end": 127.2,
      "score": 9,
      "title": "What Are Python Decorators?",
      "reason": "Standalone explanation of decorators with a clear takeaway.",
      "hook": "Python decorators explained in under a minute.",
      "description": "Learn what decorators are and how to use them in Python.",
      "tags": ["python", "decorators", "python tutorial"],
      "hashtags": ["#Python", "#Programming", "#LearnToCode"],
      "youtube": {{
        "title": "What Are Python Decorators?",
        "description": "Learn what decorators are and how to use them in Python.",
        "tags": ["python", "decorators", "python tutorial"],
        "hashtags": ["#Python", "#Programming"]
      }},
      "instagram": {{
        "caption": "Python decorators in 45 seconds — no fluff.",
        "hashtags": ["#Python", "#Programming", "#100DaysOfCode"]
      }}
    }}
  ]
}}
"""


def _build_prompt(transcript_text: str, num_clips: int, video_duration_str: str) -> str:
    return f"""You are an expert YouTube Shorts and Instagram Reels editor
specialising in programming and technical content.

You are given a full transcript (with timestamps) from a video
(total duration: {video_duration_str}).

YOUR TASK:
  Identify the best {num_clips} moments to turn into YouTube Shorts / Instagram Reels.
  You choose the EXACT start and end times — do NOT pick from pre-defined windows.
  Each clip should be 30–60 seconds long.

A good Short:
- Explains one complete, self-contained idea
- Is useful to a programmer or learner
- Makes sense without watching the rest of the video
- Has a clear takeaway
- Covers a different topic from the other selected clips

Avoid:
- Greetings, "welcome back", "in this video we'll..."
- Intros / outros / sponsor segments
- Incomplete explanations that cut off mid-thought
- Moments where the speaker is just transitioning between topics
- Repetitive content

TRANSCRIPT:
{transcript_text}

{_SHARED_FIELDS}

{_SHARED_RULES.format(num_clips=num_clips)}

Return ONLY this JSON (no other text):

{_JSON_EXAMPLE}
"""


# ---------------------------------------------------------
# Primary: full-transcript analysis  (single-pass or chunked)
# ---------------------------------------------------------


def find_highlights_from_transcript(segments: list, num_clips: int = 5) -> list:
    """
    Send the full transcript to the LLM and let it choose exact start/end
    timestamps for the best clips.

    For transcripts longer than MAX_TRANSCRIPT_CHARS the video is split into
    overlapping CHUNK_DURATION_SECS windows, each processed separately, then
    results are merged and the top num_clips are returned.
    """
    if not segments:
        return []

    transcript_text = _format_transcript(segments)
    char_count = len(transcript_text)

    if char_count <= MAX_TRANSCRIPT_CHARS:
        print(f"[ollama] Transcript: {char_count:,} chars — single-pass mode")
        return _single_pass(transcript_text, segments, num_clips)

    print(
        f"[ollama] Transcript: {char_count:,} chars — "
        f"chunking into {CHUNK_DURATION_SECS // 60}-min sections "
        f"with {CHUNK_OVERLAP_SECS // 60}-min overlap"
    )
    return _chunked(segments, num_clips)


def _single_pass(transcript_text: str, segments: list, num_clips: int) -> list:
    """Run one LLM call over the full transcript text."""
    video_end = float(segments[-1]["end"]) if segments else 0.0
    prompt = _build_prompt(transcript_text, num_clips, _ts(video_end))

    raw = ask_ollama(prompt)
    if not raw:
        return []

    results = _parse_response(raw, segments, num_clips)
    results = _dedup_and_limit(results, num_clips)

    print(f"[ollama] Selected {len(results)} valid clips.")
    return results


def _chunked(segments: list, num_clips: int) -> list:
    """
    Split the transcript into CHUNK_DURATION_SECS windows with
    CHUNK_OVERLAP_SECS overlap, run the LLM on each, then merge.
    """
    video_end = float(segments[-1]["end"])
    chunk_start = float(segments[0]["start"])
    all_highlights: list = []
    chunk_num = 0

    while chunk_start < video_end:
        chunk_end = min(chunk_start + CHUNK_DURATION_SECS, video_end)

        chunk_segs = [
            s for s in segments
            if float(s["end"]) > chunk_start and float(s["start"]) < chunk_end
        ]

        if chunk_segs:
            chunk_num += 1
            chunk_text = _format_transcript(chunk_segs)

            print(
                f"[ollama] Chunk {chunk_num}: "
                f"{_ts(chunk_start)} → {_ts(chunk_end)} "
                f"({len(chunk_text):,} chars)"
            )

            chunk_results = _single_pass(chunk_text, chunk_segs, num_clips)
            all_highlights.extend(chunk_results)

        if chunk_end >= video_end:
            break

        chunk_start += CHUNK_DURATION_SECS - CHUNK_OVERLAP_SECS

    merged = _dedup_and_limit(all_highlights, num_clips)
    print(f"[ollama] Merged {len(all_highlights)} chunk candidates → {len(merged)} final clips.")
    return merged


# ---------------------------------------------------------
# Fallback: sliding-window candidate approach (legacy)
# ---------------------------------------------------------


def find_highlights(candidates, num_clips=5):
    """
    Legacy approach: receive pre-sliced candidate windows and ask the LLM
    to pick the best IDs.

    Used as a fallback when find_highlights_from_transcript() returns nothing.
    """
    if not candidates:
        return []

    candidate_text = []

    for candidate in candidates:
        candidate_text.append(f"""
CANDIDATE {candidate['id']}

START: {candidate['start']:.2f}
END: {candidate['end']:.2f}

TRANSCRIPT:
{candidate['text']}
""")

    candidates_prompt = "\n".join(candidate_text)

    prompt = f"""
You are an expert YouTube Shorts and Instagram Reels editor
specializing in programming and technical content.

You are given FIXED candidate sections from a longer technical
programming video.

Select the best {num_clips} candidates and create complete
publishing metadata for each selected Short.

IMPORTANT:

You are NOT choosing timestamps.

You must ONLY select candidate IDs.

Each candidate is already approximately 45 seconds long.

A good Short:

- explains one complete idea
- is useful to a programmer
- makes sense without the full video
- has a clear takeaway
- is educational or interesting
- does not require too much previous context

Avoid:

- greetings
- introductions
- "welcome back"
- "in this video"
- sponsor sections
- outro sections
- incomplete explanations
- repetitive sections
- sections with no useful takeaway

Prefer candidates covering different ideas.

Candidates:

{candidates_prompt}

For every selected candidate, create:

1. score
   Integer from 1 to 10.

2. title
   A concise, natural title.
   Do not use clickbait.
   Ideally 40-70 characters.

3. reason
   Short explanation of why this section works as a Short.

4. hook
   A short opening hook suitable for the first line/caption.
   Make it interesting but technically accurate.

5. description
   A useful YouTube description of 1-3 short paragraphs.
   Explain what the viewer will learn.
   Do not invent information that is not present in the transcript.

6. tags
   5-12 relevant YouTube search tags.
   Do not include # symbols.

7. hashtags
   4-8 relevant hashtags.
   Include the # symbol.

8. youtube
   YouTube-specific title, description, tags and hashtags.

9. instagram
   Instagram Reel caption and hashtags.

Instagram caption should be concise and natural.
Do not simply copy the YouTube description.

Rules:

- id MUST be one of the provided candidate IDs.
- score MUST be an integer from 1 to 10.
- Return at most {num_clips} candidates.
- Do not return timestamps.
- Do not return markdown.
- Do not return explanations outside JSON.
- Do not invent technical facts.
- Do not use excessive emojis.
- Do not use generic spam hashtags such as #viral or #fyp unless genuinely relevant.

Return ONLY this JSON structure:

{{
  "selected": [
    {{
      "id": 3,
      "score": 9,
      "title": "What Are Django Signals?",
      "reason": "Clearly explains Django signals with a practical programming takeaway.",
      "hook": "What if Django could automatically react when something changes?",
      "description": "Learn how Django signals can trigger actions automatically when events occur in your application.",
      "tags": [
        "django",
        "python",
        "django signals",
        "python tutorial",
        "django tutorial"
      ],
      "hashtags": [
        "#Django",
        "#Python",
        "#DjangoTutorial",
        "#WebDevelopment",
        "#Programming"
      ],
      "youtube": {{
        "title": "What Are Django Signals?",
        "description": "Learn how Django signals can trigger actions automatically when events occur in your application.",
        "tags": [
          "django",
          "python",
          "django signals",
          "django tutorial",
          "python tutorial"
        ],
        "hashtags": [
          "#Django",
          "#Python",
          "#DjangoTutorial",
          "#Programming"
        ]
      }},
      "instagram": {{
        "caption": "What if Django could automatically react when something changes? Django signals make this possible.\\n\\nSave this if you're learning Django.",
        "hashtags": [
          "#Django",
          "#Python",
          "#DjangoDeveloper",
          "#WebDevelopment",
          "#Programming"
        ]
      }}
    }}
  ]
}}
"""

    raw = ask_ollama(prompt)

    if not raw:
        return []

    try:
        result = json.loads(raw)

    except json.JSONDecodeError as exc:
        print(f"[ollama] Invalid JSON: {exc}")

        cleaned = raw

        if "```json" in cleaned:
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        elif "```" in cleaned:
            cleaned = cleaned.replace("```", "").strip()

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            print("[ollama] Could not recover JSON.")
            return []

    selected = result.get("selected", [])

    if not isinstance(selected, list):
        print("[ollama] 'selected' is not a list.")
        return []

    candidate_map = {candidate["id"]: candidate for candidate in candidates}

    highlights = []

    for item in selected:
        if not isinstance(item, dict):
            continue

        try:
            candidate_id = int(item["id"])
        except (KeyError, TypeError, ValueError):
            continue

        candidate = candidate_map.get(candidate_id)

        if candidate is None:
            print(f"[ollama] Unknown candidate: {candidate_id}")
            continue

        try:
            score = int(item.get("score", 0))
        except (TypeError, ValueError):
            score = 0

        if score < 1:
            continue

        youtube = item.get("youtube", {})
        instagram = item.get("instagram", {})

        if not isinstance(youtube, dict):
            youtube = {}

        if not isinstance(instagram, dict):
            instagram = {}

        tags = item.get("tags", [])
        hashtags = item.get("hashtags", [])

        if not isinstance(tags, list):
            tags = []

        if not isinstance(hashtags, list):
            hashtags = []

        highlights.append(
            {
                "start": candidate["start"],
                "end": candidate["end"],
                "candidate_id": candidate_id,
                "score": score,
                "title": item.get(
                    "title",
                    f"Candidate {candidate_id}",
                ),
                "reason": item.get(
                    "reason",
                    "",
                ),
                "hook": item.get(
                    "hook",
                    "",
                ),
                "description": item.get(
                    "description",
                    "",
                ),
                "tags": tags,
                "hashtags": hashtags,
                "youtube": {
                    "title": youtube.get(
                        "title",
                        item.get("title", ""),
                    ),
                    "description": youtube.get(
                        "description",
                        item.get("description", ""),
                    ),
                    "tags": youtube.get(
                        "tags",
                        tags,
                    ),
                    "hashtags": youtube.get(
                        "hashtags",
                        hashtags,
                    ),
                },
                "instagram": {
                    "caption": instagram.get(
                        "caption",
                        item.get("description", ""),
                    ),
                    "hashtags": instagram.get(
                        "hashtags",
                        hashtags,
                    ),
                },
            }
        )

    # Highest scoring candidates first while selecting.
    highlights.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    selected_highlights = []

    for highlight in highlights:
        overlaps = False

        for existing in selected_highlights:
            if (
                highlight["start"] < existing["end"]
                and highlight["end"] > existing["start"]
            ):
                overlaps = True
                break

        if overlaps:
            print(
                "[ollama] Removing overlapping "
                f"candidate {highlight['candidate_id']}"
            )
            continue

        selected_highlights.append(highlight)

        if len(selected_highlights) >= num_clips:
            break

    # Restore chronological order.
    selected_highlights.sort(key=lambda x: x["start"])

    print(f"[ollama] Selected {len(selected_highlights)} valid candidates.")

    return selected_highlights
