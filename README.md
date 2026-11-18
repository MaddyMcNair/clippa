# CLIPPA: 100% Local-First Video Clipping Pipeline

CLIPPA is a fully self-hosted, automated pipeline that ingests long-form videos (YouTube URLs or local MP4 files), transcribes audio locally via Faster-Whisper, extracts viral highlight moments with local LLMs (Ollama) or heuristics, reframes landscape video to 9:16 vertical format with dynamic shot-aware face tracking (YuNet ONNX), and burns animated karaoke subtitles using FFmpeg.

**Zero paid external APIs. 100% privacy, local execution.**

---

## Key Features

1. **Dynamic Shot-Aware 9:16 Face Framing**:
   - Uses OpenCV YuNet ONNX model (`face_detection_yunet_2023mar.onnx`) for face detection.
   - Detects shot cuts and camera angle switches (`cut_threshold = 0.16 * src_w`).
   - Generates piecewise time-varying FFmpeg crop expressions (`crop=...:x='if(lt(t, T1), X0, ...)'`) to keep active speakers centered without placing the camera in empty space between multiple people.

2. **Accurate Context-Aware Clip Titles & File Naming**:
   - Synthesizes punchy, 3–9 word hook titles using the source video title and the clip's transcript dialogue.
   - Saves rendered clips named after their sanitized title (e.g., `you_say_you_want_to_battle_Serena_Williams.mp4`) instead of generic numbered filenames.

3. **YouTube "Most Replayed" Heatmap Integration**:
   - Automatically detects host JavaScript runtimes (Node.js/Deno) to extract YouTube's 100-point seek heatmap.
   - Identifies viewer replay peaks and centers clips on the most popular parts of the video.

4. **Narrative Sentence Boundary Alignment**:
   - Uses sentence-level boundary search so clips never start or end mid-thought.
   - Strictly bounded to $\le 120$ seconds (2 minutes).

5. **Subtitle Karaoke Styling & No Emojis**:
   - Word-by-word highlighted karaoke subtitles (`\k` ASS format) with a 3-second top hook overlay.
   - Built-in templates: `default`, `hormozi`, `mrbeast`, `tiktok`.
   - Emojis disabled by default for clean, professional captions.

---

## Quick Start

### 1. One-Click Launchers (Windows)
- Double-click **`run_web.bat`**: Automatically starts the server and opens the dashboard in your default browser at `http://127.0.0.1:8000`.
- Double-click **`run_cli.bat`**: Prompts for a YouTube URL or video path and runs the pipeline interactively.

### 2. Manual CLI Activation
```powershell
.\.venv\Scripts\activate
```

### 3. Launch Local Web Dashboard
```powershell
python -m clippa --web
# Open in browser: http://127.0.0.1:8000
```
- In-browser HTML5 vertical video player.
- Instant 1-click downloads (`.mp4`).
- Direct clip deletion (`🗑 Delete`) and folder explorer access (`📂 Open Folder`).
- Live real-time title search and filtering.
- Interactive NVENC hardware acceleration toggle.


### 4. Run from CLI on a YouTube URL
```powershell
# Extract top 3 clips (default style)
python -m clippa --url "https://www.youtube.com/watch?v=jJ-zWCw85qA" --max-clips 3

# Use the MrBeast subtitle template
python -m clippa --url "https://www.youtube.com/watch?v=_c2e1NcI3Sc" --template mrbeast --max-clips 2

# Use Hormozi bold style with heuristic fallback (no Ollama required)
python -m clippa --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --template hormozi --llm-model heuristic
```

### 5. Run on a Local Video File
```powershell
python -m clippa --url "temp/video.mp4" --output-dir "outputs" --template tiktok
```

---

## CLI Options Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--web` | `False` | Launch the local web dashboard and in-browser video previewer at `http://127.0.0.1:8000`. |
| `--port` | `8000` | Port for the local web dashboard server. |
| `--url` | *Optional* | YouTube video URL or path to a local `.mp4` file (required when not using `--web`). |
| `--max-clips` | `3` | Maximum number of viral clips to extract and render. |
| `--max-duration` | `120` | Maximum duration per clip in seconds (upper limit: 2 minutes). |
| `--template` | `default` | Subtitle styling template (`default`, `hormozi`, `mrbeast`, `tiktok`). |
| `--whisper-model` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`). |
| `--llm-model` | `ollama:llama3.2:3b` | Virality analysis engine (`ollama:<model>` or `heuristic`). |
| `--resolution` | `1080p` | Output vertical resolution (`1080p` = 1080x1920, `720p` = 720x1280). |
| `--device` | `cuda` | Hardware accelerator for Whisper (`cuda` or `cpu`). |
| `--output-dir` | `outputs` | Directory to save rendered vertical MP4 clips. |
| `--temp-dir` | `temp` | Working directory for cached audio, video, transcripts, and ASS files. |
| `--enable-emojis` | `False` | Enable automatic emoji insertion based on keywords. |
| `--no-zoom` | `False` | Disable subtle Ken Burns talking-head zoom effect. |
| `--no-cache` | `False` | Force re-download and re-transcription bypassing local cache sidecars. |
| `--nvenc` | `False` | Enable NVIDIA NVENC hardware acceleration (`h264_nvenc`) for 5x–6x faster GPU video encoding. |

---

## Architecture & Modules

- **`clippa/ingestion.py`**: Video download via `yt-dlp`, Node.js JS runtime integration, YouTube heatmap extraction, metadata caching.
- **`clippa/transcription.py`**: Audio extraction, Faster-Whisper with automatic CUDA DLL registration and CPU fallback, schema v3 transcript caching.
- **`clippa/analysis.py`**: Virality scoring (Ollama LLM or heuristic), YouTube heatmap peak alignment, complete narrative boundary detection ($\le 120$s), context-aware hook title synthesis.
- **`clippa/reframing.py`**: OpenCV YuNet ONNX face detection, shot-cut segmentation, multi-speaker tracking, time-varying FFmpeg crop expressions.
- **`clippa/subtitles.py`**: ASS word-level karaoke subtitle generator, style templates (`hormozi`, `mrbeast`, etc.), top hook banner overlay.
- **`clippa/rendering.py`**: Single-pass FFmpeg complex filtergraph burning 9:16 crop and ASS subtitles with faststart MP4 output.
- **`clippa/cli.py`**: CLI pipeline orchestration and title-based output file naming.

---

## Running Tests

All 206 unit, feature, and adversarial coverage tests run fully offline:

```powershell
# Run all 110 feature tests
.\.venv\Scripts\python -m pytest tests/tier1_features/

# Run all 96 adversarial and edge-case tests
.\.venv\Scripts\python -m pytest tests/test_adversarial_m4_coverage.py
```
