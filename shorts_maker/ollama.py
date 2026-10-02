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


def find_highlights(
    candidates,
    num_clips=5,
):
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
You are an expert editor selecting YouTube Shorts
from a technical programming video.

You are given FIXED candidate sections.

Select the best {num_clips} candidates.

IMPORTANT:

You are NOT choosing timestamps.

You must ONLY return candidate IDs.

Each candidate is already approximately 45 seconds.

A good Short:

- explains one complete idea
- is useful to a programmer
- makes sense without the full video
- has a clear explanation
- has a useful takeaway
- is interesting or educational

Avoid:

- greetings
- introductions
- "welcome back"
- "in this video"
- sponsor sections
- outro sections
- incomplete explanations
- repetitive explanations
- sections requiring too much previous context

Prefer candidates covering different ideas.

Candidates:

{candidates_prompt}

Return ONLY JSON.

Required format:

{{
  "selected": [
    {{
      "id": 3,
      "score": 9,
      "title": "What is a Django signal?",
      "reason": "Clearly explains the concept."
    }}
  ]
}}

Rules:

- id MUST be one of the provided candidate IDs.
- score MUST be an integer from 1 to 10.
- Return at most {num_clips} candidates.
- Do not return timestamps.
- Do not return markdown.
- Do not return explanations outside the JSON.
"""

    raw = ask_ollama(prompt)

    if not raw:
        return []

    # -----------------------------------------------------
    # Parse JSON
    # -----------------------------------------------------

    try:
        result = json.loads(raw)

    except json.JSONDecodeError as exc:
        print(f"[ollama] Invalid JSON: {exc}")

        # Sometimes small models wrap JSON in markdown.
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

    selected = result.get(
        "selected",
        [],
    )

    if not isinstance(
        selected,
        list,
    ):
        print("[ollama] 'selected' is not a list.")

        return []

    # -----------------------------------------------------
    # Candidate lookup
    # -----------------------------------------------------

    candidate_map = {candidate["id"]: candidate for candidate in candidates}

    highlights = []

    for item in selected:

        if not isinstance(
            item,
            dict,
        ):
            continue

        try:
            candidate_id = int(item["id"])

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        candidate = candidate_map.get(candidate_id)

        if candidate is None:
            print(f"[ollama] Unknown candidate: " f"{candidate_id}")
            continue

        try:
            score = int(
                item.get(
                    "score",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            score = 0

        if score < 1:
            continue

        highlights.append(
            {
                "start": candidate["start"],
                "end": candidate["end"],
                "title": item.get(
                    "title",
                    f"Candidate {candidate_id}",
                ),
                "reason": item.get(
                    "reason",
                    "",
                ),
                "score": score,
                "candidate_id": candidate_id,
            }
        )

    # -----------------------------------------------------
    # Sort by score
    # -----------------------------------------------------

    highlights.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    # -----------------------------------------------------
    # Remove overlapping candidates
    # -----------------------------------------------------

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

    # Chronological order
    selected_highlights.sort(key=lambda x: x["start"])

    print(f"[ollama] Selected " f"{len(selected_highlights)} " f"valid candidates.")

    return selected_highlights
