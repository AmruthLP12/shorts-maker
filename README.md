# shorts-maker

Automatically turn any YouTube video (or a local video file) into **YouTube Shorts / Instagram Reels** using local AI — no cloud APIs required.

It downloads (or reads) the video, transcribes it with **Whisper**, picks the best moments with **Ollama (Llama 3.2)**, smart-crops the footage to 9:16, and renders ready-to-upload `.mp4` files with **embedded soft subtitles** — along with per-short `.srt` files and a Markdown publishing report for YouTube and Instagram.

---

## How it works

```
YouTube URL  OR  local video file
    │
    ▼
[1] yt-dlp / local file   → source video (up to 1080p)
    │
    ▼
[2] faster-whisper         → timestamped transcript (JSON + SRT)
    │
    ▼
[3] candidates             → 45-second sliding windows every 30 s
    │
    ▼
[4] Ollama (Llama 3.2)    → picks best windows, generates titles /
    │                        hooks / descriptions / tags / hashtags
    ▼
[5] OpenCV + FFmpeg        → smart 16:9→9:16 crop, renders 1080×1920
                             MP4 with embedded soft subtitle track
```

Everything runs **locally** — Whisper on CPU, Ollama on your machine.

---

## Requirements

| Tool | Why |
|------|-----|
| **Python ≥ 3.14** | Runtime |
| **[uv](https://docs.astral.sh/uv/)** | Dependency & venv manager |
| **[FFmpeg](https://ffmpeg.org/)** | Video rendering & subtitle embedding |
| **[Ollama](https://ollama.com/)** | Local LLM inference |

### Install system dependencies

```bash
# FFmpeg (Ubuntu / Debian)
sudo apt install ffmpeg

# FFmpeg (macOS)
brew install ffmpeg

# Ollama — download from https://ollama.com/download, then pull the model:
ollama pull llama3.2
```

---

## Installation

```bash
git clone https://github.com/yourname/shorts-maker.git
cd shorts-maker
uv sync
```

> `uv sync` reads `pyproject.toml`, creates `.venv`, and installs all packages automatically.

---

## Usage

Start Ollama first (keep it running in a separate terminal):

```bash
ollama serve
```

### From a YouTube URL

```bash
uv run python -m shorts_maker "https://www.youtube.com/watch?v=..."
```

### From a local video file

```bash
uv run python -m shorts_maker /path/to/your/video.mp4
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--clips N` | `5` | Number of Shorts to generate |
| `--srt-only` | off | Just transcribe and write an SRT; skip clip selection and rendering |

### Examples

```bash
# 5 Shorts from a YouTube video (default)
uv run python -m shorts_maker "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# 3 Shorts from a local file
uv run python -m shorts_maker /home/amruth/videos/talk.mp4 --clips 3

# Just get an SRT for a local file (saved next to the video)
uv run python -m shorts_maker /home/amruth/videos/talk.mp4 --srt-only

# Just get an SRT for a YouTube video (saved to data/<id>/transcript.srt)
uv run python -m shorts_maker "https://www.youtube.com/watch?v=..." --srt-only
```

---

## Output

### Shorts output — `output/<video-id>/`

```
output/
└── a3f9c12b4e/
    ├── highlights.json       ← all selected clips & raw metadata
    │
    ├── short-01.mp4          ← 1080×1920 vertical video
    │                           (soft subtitle track embedded — toggle in player)
    ├── short-01.srt          ← subtitle file, timestamps start at 0:00:00
    ├── short-01.json         ← full publishing metadata (JSON)
    ├── short-01.md           ← human-readable publishing report (Markdown)
    │
    ├── short-02.mp4
    ├── short-02.srt
    ├── short-02.json
    ├── short-02.md
    └── ...
```

### --srt-only output

| Input | SRT saved at |
|-------|-------------|
| Local file `/videos/talk.mp4` | `/videos/talk.srt` (next to the video) |
| YouTube URL | `data/<id>/transcript.srt` |

---

## Subtitles

Each Short's `.mp4` contains the subtitles as a **soft (embedded) subtitle track** — they are not burned into the pixels. This means:

- Turn them on/off in VLC, QuickTime, IINA, or any modern player
- No quality loss from text rendering
- The `.srt` file is also saved alongside the `.mp4` for uploading to YouTube's caption system or editing in any subtitle editor

---

## Markdown report format

Each `short-NN.md` file contains:

```markdown
# Short 01 — What Are Django Signals?

> *What if Django could automatically react when something changes?*

| Field    | Value |
|----------|-------|
| Score    | `████████░░ 8/10` |
| Clip     | `2m 00s → 2m 45s` (45s) |
| Source   | `https://...` |
| Files    | `short-01.mp4` · `short-01.srt` · `short-01.json` |

---

## Why this works as a Short
...

## Description
...

## Tags & Hashtags
...

## YouTube
...

## Instagram
...
```

---

## Caching

Downloaded videos and transcripts are cached in `data/<video-id>/` so re-running on the same URL or file skips slow steps:

```
data/
└── a3f9c12b4e/
    ├── source.mp4         (YouTube downloads only)
    ├── transcript.json
    └── transcript.srt     (full-video SRT)
```

Delete `data/<id>/` to force a fresh download and transcription.

---

## Project structure

```
shorts-maker/
├── shorts_maker/
│   ├── main.py         ← CLI entry point & pipeline orchestration
│   ├── download.py     ← yt-dlp video download
│   ├── transcribe.py   ← faster-whisper speech-to-text, SRT export,
│   │                      clip-relative SRT generation
│   ├── candidates.py   ← sliding-window transcript segmentation
│   ├── ollama.py       ← Ollama LLM prompt, scoring & metadata generation
│   ├── clips.py        ← smart 9:16 crop (OpenCV) + FFmpeg render
│   │                      (with optional soft subtitle embedding)
│   └── report.py       ← Markdown publishing report generator
├── data/               ← cached downloads & transcripts (auto-created)
├── output/             ← rendered Shorts + metadata (auto-created)
├── pyproject.toml
└── uv.lock
```

---

## Configuration

**Ollama model** — top of [`shorts_maker/ollama.py`](shorts_maker/ollama.py):

```python
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL      = "llama3.2:latest"
```

Change `MODEL` to any locally pulled model, e.g. `mistral`, `gemma3`, `phi4`.

**Whisper model** — [`shorts_maker/transcribe.py`](shorts_maker/transcribe.py):

```python
model = WhisperModel("base", device="cpu", compute_type="int8")
```

Replace `"base"` with `"small"`, `"medium"`, or `"large-v3"` for better accuracy.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `yt-dlp` | Download YouTube videos |
| `faster-whisper` | Fast CPU Whisper transcription |
| `opencv-python-headless` | Frame analysis for smart crop |
| `av` | Video container support |
| `requests` | HTTP calls to Ollama |
| `rich` | Terminal output formatting |

---

## Tips

- Works best on **technical / educational** content (coding tutorials, talks, lectures). The LLM prompt is tuned to prefer self-contained, informative moments and skip intros, outros, and sponsor sections.
- The smart crop uses **edge density analysis** (8 sampled frames) to find the content-rich region of the screen — great for screen recordings, terminals, and editors.
- First run on a new video is slow (download + transcription). Re-runs on the same input reuse the cache instantly.
- Ollama inference time depends on your hardware and the number of candidate windows (~1–5 min on CPU).
- For uploading to YouTube, use the `.srt` file in the Subtitles section of YouTube Studio — no conversion needed.
