import json
import itertools
import sys
import threading
import time

import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3.2:latest"


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
# Find highlights
# ---------------------------------------------------------


def find_highlights(candidates, num_clips=5):
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
        except KeyError, TypeError, ValueError:
            continue

        candidate = candidate_map.get(candidate_id)

        if candidate is None:
            print(f"[ollama] Unknown candidate: {candidate_id}")
            continue

        try:
            score = int(item.get("score", 0))
        except TypeError, ValueError:
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

    print(f"[ollama] Selected " f"{len(selected_highlights)} " f"valid candidates.")

    return selected_highlights
