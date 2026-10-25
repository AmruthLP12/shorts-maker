# Makefile for shorts-maker
# Provides convenient shortcuts for installation, execution, verification, and tests.

.DEFAULT_GOAL := help

# Executable variables
UV ?= uv
PYTHON ?= $(UV) run python

# Processing parameters
CLIPS ?= 5

.PHONY: help install run srt test test-crop test-thumbnails check clean-output clean-cache

help: ## Show this help message and examples
	@echo "========================================================================"
	@echo "                      shorts-maker — Command Menu                       "
	@echo "========================================================================"
	@echo "Usage: make <target> [VARIABLE=value]"
	@echo ""
	@echo "Common Targets:"
	@echo "  make install             Install dependencies using 'uv sync'"
	@echo "  make run URL=\"...\"       Process a YouTube URL (default CLIPS=5)"
	@echo "  make run INPUT=\"...\"     Process a local video file (default CLIPS=5)"
	@echo "  make srt INPUT=\"...\"     Generate SRT subtitle file only (skips rendering)"
	@echo "  make srt URL=\"...\"       Generate SRT subtitle file for YouTube URL"
	@echo "  make check               Check system tools (Python, uv, FFmpeg, Ollama)"
	@echo ""
	@echo "Testing Targets:"
	@echo "  make test                Run the full test suite"
	@echo "  make test-crop           Run tests for content-aware reframing & clips"
	@echo "  make test-thumbnails     Run tests for automatic thumbnail generation"
	@echo ""
	@echo "Maintenance Targets:"
	@echo "  make clean-output        Remove generated output directory (output/)"
	@echo "  make clean-cache         Remove cached downloads and transcripts (data/)"
	@echo ""
	@echo "Examples:"
	@echo "  make run URL=\"https://www.youtube.com/watch?v=dQw4w9WgXcQ\" CLIPS=3"
	@echo "  make run INPUT=\"/home/user/videos/talk.mp4\" CLIPS=3"
	@echo "  make srt INPUT=\"/home/user/videos/talk.mp4\""
	@echo "  make test"
	@echo "========================================================================"

install: ## Install dependencies using uv sync
	@echo "--> Installing dependencies via uv sync..."
	@$(UV) sync

run: ## Process a YouTube URL (URL="...") or local video (INPUT="...")
	@if [ -n "$(URL)" ] && [ -n "$(INPUT)" ]; then \
		echo "[error] Both URL and INPUT were provided. Please specify only one."; \
		exit 1; \
	fi
	@if [ -z "$(URL)" ] && [ -z "$(INPUT)" ]; then \
		echo "[error] Missing video source. Provide either URL=\"...\" or INPUT=\"...\"."; \
		echo "Example: make run URL=\"https://www.youtube.com/watch?v=...\" CLIPS=$(CLIPS)"; \
		echo "Example: make run INPUT=\"/path/to/video.mp4\" CLIPS=$(CLIPS)"; \
		exit 1; \
	fi
	@if [ -n "$(URL)" ]; then \
		echo "--> Processing YouTube URL: $(URL) (CLIPS=$(CLIPS))"; \
		$(PYTHON) -m shorts_maker "$(URL)" --clips $(CLIPS); \
	else \
		echo "--> Processing local video: $(INPUT) (CLIPS=$(CLIPS))"; \
		$(PYTHON) -m shorts_maker "$(INPUT)" --clips $(CLIPS); \
	fi

srt: ## Generate SRT only without rendering clips (INPUT="..." or URL="...")
	@if [ -n "$(URL)" ] && [ -n "$(INPUT)" ]; then \
		echo "[error] Both URL and INPUT were provided. Please specify only one."; \
		exit 1; \
	fi
	@if [ -z "$(URL)" ] && [ -z "$(INPUT)" ]; then \
		echo "[error] Missing video source. Provide either INPUT=\"...\" or URL=\"...\"."; \
		echo "Example: make srt INPUT=\"/path/to/video.mp4\""; \
		exit 1; \
	fi
	@if [ -n "$(URL)" ]; then \
		echo "--> Generating subtitles (SRT only) for: $(URL)"; \
		$(PYTHON) -m shorts_maker "$(URL)" --srt-only; \
	else \
		echo "--> Generating subtitles (SRT only) for: $(INPUT)"; \
		$(PYTHON) -m shorts_maker "$(INPUT)" --srt-only; \
	fi

test: ## Run the full test suite
	@echo "--> Running complete test suite..."
	@$(PYTHON) -m pytest tests/ -v

test-crop: ## Run crop & reframing tests
	@echo "--> Running crop and reframing tests..."
	@$(PYTHON) -m pytest tests/test_clips.py -v

test-thumbnails: ## Run thumbnail generation tests
	@echo "--> Running thumbnail generation tests..."
	@$(PYTHON) -m pytest tests/test_thumbnails.py -v

check: ## Verify availability of required local tools (Python, uv, FFmpeg, Ollama)
	@echo "--> Checking local environment..."
	@echo -n "Python:  "
	@which python3 >/dev/null 2>&1 && python3 --version || echo "NOT FOUND"
	@echo -n "uv:      "
	@which $(UV) >/dev/null 2>&1 && $(UV) --version || echo "NOT FOUND"
	@echo -n "ffmpeg:  "
	@which ffmpeg >/dev/null 2>&1 && ffmpeg -version 2>&1 | head -n 1 || echo "NOT FOUND"
	@echo -n "ffprobe: "
	@which ffprobe >/dev/null 2>&1 && ffprobe -version 2>&1 | head -n 1 || echo "NOT FOUND"
	@echo -n "Ollama CLI: "
	@which ollama >/dev/null 2>&1 && ollama --version || echo "NOT FOUND"
	@echo -n "Ollama Server: "
	@if curl -s -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then \
		echo "ONLINE (http://127.0.0.1:11434)"; \
	else \
		echo "OFFLINE or UNREACHABLE (start with: ollama serve)"; \
	fi

clean-output: ## Remove generated output directory
	@echo "[warning] This will delete all generated videos, thumbnails, and metadata in output/"
	@read -p "Are you sure you want to proceed? [y/N] " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "--> Removing output/..."; \
		rm -rf output; \
		echo "Output directory cleaned."; \
	else \
		echo "Cancelled."; \
	fi

clean-cache: ## Remove cached downloads and transcripts
	@echo "[warning] This will delete all cached videos and transcripts in data/"
	@read -p "Are you sure you want to proceed? [y/N] " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "--> Removing data/..."; \
		rm -rf data; \
		echo "Cache directory cleaned."; \
	else \
		echo "Cancelled."; \
	fi
