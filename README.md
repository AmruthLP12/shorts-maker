# shorts-maker

Automatically turn any YouTube video (or a local video file) into **YouTube Shorts / Instagram Reels** using local AI — no cloud APIs required.

It downloads (or reads) the video, transcribes it with **Whisper**, selects the best moments via **Ollama (Llama 3.2)**, intelligently reframes screen recordings to 9:16 using content-aware tracking, and renders ready-to-upload `.mp4` files with **embedded soft subtitles** — along with per-short `.srt` files and Markdown publishing reports for YouTube and Instagram.

---

## How it works

```
YouTube URL  OR  local video file
    │
    ▼
[1] yt-dlp / local file   → source video (any landscape / portrait resolution)
    │
    ▼
[2] faster-whisper         → timestamped transcript (JSON + SRT)
    │
    ▼
[3] Ollama (Llama 3.2)    → analyzes full transcript with timestamps, selects exact
    │                        start/end times, titles, hooks, tags & descriptions
    │                        (falls back to candidate windowing if needed)
    ▼
[4] Content-Aware Crop     → multi-signal text/detail analysis + temporal tracking:
    │                        samples frames, detects code/terminal regions,
    │                        penalizes boundary cuts, smooths X positions over time
    ▼
[5] OpenCV + FFmpeg        → dynamic piecewise-linear 9:16 crop, renders 1080×1920
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

## Intelligent 9:16 Auto-Reframing

Converting a 16:9 screen recording into a 9:16 vertical Short poses unique challenges: editors, terminals, browsers, and sidebars occupy different screen areas across the clip.

Instead of a static center-crop or crude edge-density calculation, `shorts-maker` uses a multi-stage reframing pipeline:

1. **Multi-Signal Column Scoring**:
   - **Bright-on-dark text detection (55%)**: Primary signal detecting code, editor text, and terminal output.
   - **Local detail variance (30%)**: Distinguishes rich content areas from flat UI backgrounds.
   - **Edge density (15%)**: Kept low to prevent syntax highlighting and UI borders from skewing the crop.
2. **Boundary Cut Penalty**: Candidate crop windows whose left or right edges slice through dense content regions are heavily penalized.
3. **Temporal Tracking**:
   - Frames are sampled across the clip duration at regular intervals (default: 5s).
   - Best X positions are tracked per frame.
   - Gaussian / box-filter temporal smoothing removes jitter.
   - Small movements (`≤ 30px`) are snapped to preserve rock-solid camera stability.
4. **Dynamic FFmpeg Rendering**:
   - Keyframe positions are compiled into a piecewise-linear FFmpeg `crop=w:h:x:0` filter expression.
   - Transitions between screen regions pan gradually rather than jumping abruptly.
   - All rendering is completed in a single pass with high-quality Lanczos scaling to exactly `1080×1920`.
5. **Fallback & Aspect Ratio Safety**:
   - Sources already portrait or square bypass cropping without distortion.
   - Handled gracefully for arbitrary aspect ratios (e.g., 21:9 ultrawide, 16:10, 4:3).
   - Unreadable sources or scenes with zero detected features fall back safely to a documented center crop.

---

## Output

### Shorts output — `output/<video-id>/`

```
output/
└── a3f9c12b4e/
    ├── highlights.json       ← all selected clips & raw metadata
    │
    ├── short-01.mp4          ← 1080×1920 vertical video
    │                           (content-aware dynamic crop + embedded soft subtitle track)
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

Each Short's `.mp4` contains the subtitles as a **soft (embedded) subtitle track** — they are not burned into the pixels:

- Turn them on/off in VLC, QuickTime, IINA, or any modern player
- No quality loss or blurriness from text rasterization
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
│   ├── candidates.py   ← sliding-window transcript segmentation (fallback)
│   ├── ollama.py       ← Ollama LLM prompt, scoring & metadata generation
│   ├── clips.py        ← content-aware 9:16 reframing + temporal tracking + FFmpeg render
│   └── report.py       ← Markdown publishing report generator
├── tests/              ← pytest test suite (unit + synthetic video tests)
├── data/               ← cached downloads & transcripts (auto-created)
├── output/             ← rendered Shorts + metadata (auto-created)
├── pyproject.toml
└── uv.lock
```

---

## Running Tests

The test suite uses synthetic in-memory frames and FFmpeg test patterns without downloading any external videos:

```bash
uv run python -m pytest tests/test_clips.py -v
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
| `opencv-python-headless` | Content scoring, frame analysis, and resolution probing |
| `av` | Video container support |
| `requests` | HTTP calls to Ollama |
| `rich` | Terminal output formatting |
| `pytest` | Test suite execution (dev dependency) |

---

## Tips & Limitations

- Works best on **technical / educational** screen recordings (coding tutorials, terminal screencasts, IDE walkthroughs).
- **Narrow 9:16 Aspect Ratio**: Cropping a 16:9 screen into 9:16 necessarily isolates ~33% of the horizontal screen area at any given moment. For screens where the editor is on the left and the terminal is on the far right simultaneously, the reframer focuses on the most active text/code region and pans smoothly if focus shifts.
- First run on a new video is slow (download + transcription). Re-runs on the same input reuse the cache instantly.
- Ollama inference time depends on your hardware and video length.
