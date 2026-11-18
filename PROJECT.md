# Project: CLIPPA — Local Video Clipping Pipeline

## Architecture
CLIPPA is a local-first video clipping pipeline based on `FujiwaraChoki/supoclip` that ingests YouTube videos, transcribes speech with word-level precision, analyzes content for viral moments using local models, reframes landscape footage into 9:16 vertical video using face detection, and burns animated subtitles with hook titles into export-ready MP4 clips.

### System Topology & Data Flow
```
[YouTube URL / Video Path]
          │
          ▼
[Ingestion: yt-dlp] ───► [source.mp4]
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
   [FFmpeg Audio Extract]            [Probe Geometry & FPS]
             │                                 │
     [audio.mp3 (16kHz)]                       │
             │                                 │
             ▼                                 │
[Local Whisper (CUDA/CPU)]                     │
             │                                 │
[Word Timestamps & Transcript]                 │
             │                                 │
             ▼                                 │
[Local AI Analysis (Ollama / Heuristic)]       │
             │                                 │
[Viral Segments (Hook, Scores, Hook Title)]    │
             │                                 │
             ▼                                 ▼
   [Face Detection (OpenCV YuNet ONNX)] ◄──────┘
             │
   [Smoothed 9:16 Trajectory / Crop Window]
             │
             ▼
   [ASS Subtitle Generation (Templates, Word Karaoke, Hook Overlay)]
             │
             ▼
   [FFmpeg Video Render (Crop 9:16 + libass subtitle burn + AAC)]
             │
             ▼
   [Final Vertical 9:16 MP4 Clips in Output Folder]
```

---

## Feature Inventory
Every feature from the Phase 0 Survey is inventoried and assigned to a milestone:

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | YouTube Video Download | Ingests YouTube URLs and downloads best video+audio up to 1080p MP4 via yt-dlp | M3 | survey |
| 2 | Video Metadata Preflight | Extracts title, duration, author via yt-dlp preflight without full download | M3 | survey |
| 3 | Audio Track Extraction | Extracts mono 16kHz audio from video via FFmpeg for transcription | M2 | survey |
| 4 | Local Whisper Transcription | Generates millisecond word-level timestamps using local Whisper on CUDA/CPU | M2 | survey |
| 5 | Transcript Cache Sidecar | Stores words, utterances, text, and language in `.transcript_cache.json` | M2 | survey |
| 6 | Viral Highlight & Moment Detection | Scores 15–60s segments for hook, engagement, value, shareability using Ollama / local LLM | M2 | survey |
| 7 | Rule-Based Heuristic Scorer Fallback | Fallback acoustic & keyword density virality scorer when Ollama is offline | M2 | survey |
| 8 | Windowed Long-Video Analysis | Splits >15m videos into overlapping windows and reranks top candidates | M2 | survey |
| 9 | Hook Title Generation | AI / heuristic headline generator creating 3–9 word overlay title | M2 | survey |
| 10 | Cold-Open Teaser Detection | Identifies best 2–5s punchline to play as opener before main clip | M2 | survey |
| 11 | YuNet ONNX Face Detection | 100% local deep-learning face detection using bundled OpenCV YuNet model | M2 | survey |
| 12 | Face Centroid Tracking & Smoothing | Tracks speaker face trajectory with exponential smoothing across frames | M2 | survey |
| 13 | Ken Burns Talking Head Zoom | Subtle slow punch-in zoom for talking head dynamism | M2 | survey |
| 14 | 9:16 Vertical Reframing & Clamping | Crops and scales landscape frame to exact 1080x1920 or 720x1280 vertical video | M3 | survey |
| 15 | Animated ASS Subtitle Builder | Word-by-word karaoke pop animation with active word highlighting | M2 | survey |
| 16 | Subtitle Style Templates & TTF Fonts | Multiple style presets (Default, Hormozi, MrBeast, TikTok, etc.) with local TTF fonts | M2 | survey |
| 17 | Contextual Emoji Injection | Places relevant emojis next to recognized keyword triggers in captions | M2 | survey |
| 18 | Burned Hook Title Overlay | Overlays AI headline at top safe area for first ~4 seconds of clip | M2 | survey |
| 19 | Subtitle Burning via FFmpeg | Uses FFmpeg `libass` filter to burn subtitles directly into video frames | M3 | survey |
| 20 | Self-Hosted Zero-Paid-API Mode | Sets `SELF_HOST=true` and bypasses all cloud API keys (AssemblyAI, OpenAI, Stripe, etc.) | M1 | survey |
| 21 | Standalone CLI Pipeline Runner | Unified CLI interface (`python -m clippa` / `clippa_pipeline.py`) taking URL and outputting 9:16 clips | M3 | survey |
| 22 | End-to-End Acceptance Verification | Complete end-to-end execution test producing valid playable 9:16 clips with audio & subtitles | M4 | survey |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Environment & Repository Setup | Populate supoclip codebase in workspace, configure Python 3.11 virtual environment, install FFmpeg on PATH, install dependencies, verify CUDA and Ollama model | none | COMPLETE |
| M2 | Local-First Processing Adaptation | Configure and verify local Whisper transcription, local Ollama / heuristic virality scoring, YuNet face detection & reframing, and ASS animated subtitles | M1 | COMPLETE |
| M3 | End-to-End Pipeline & CLI Integration | Integrate full YouTube ingestion -> transcription -> highlight detection -> 9:16 crop -> ASS subtitle burning pipeline; build unified CLI runner | M2 | COMPLETE |
| M4 | Final Milestone: E2E Verification & Test Suite Pass | Verify 100% pass of E2E test suite (Tiers 1-4) with real YouTube video URL, producing valid playable 9:16 clips with synchronized subtitles and audio | M3 | COMPLETE |

---

## Interface Contracts

### 1. Ingestion Contract (`clippa.ingestion` ↔ `clippa.transcription`)
- Function: `download_video(url: str, output_dir: Path) -> Path`
- Inputs: YouTube URL or local video path
- Returns: Absolute path to verified `.mp4` file
- Error Handling: Raises `VideoDownloadError` with descriptive reason if download fails.

### 2. Transcription Contract (`clippa.transcription` ↔ `clippa.analysis`)
- Function: `transcribe_audio(video_path: Path, model_size: str = "base", device: str = "cuda") -> TranscriptData`
- Output Schema:
  ```json
  {
    "text": "Full transcript...",
    "language": "en",
    "words": [
      {
        "text": "Hello",
        "start": 120,   // integer milliseconds
        "end": 450,     // integer milliseconds
        "confidence": 0.95
      }
    ]
  }
  ```
- Error Handling: Fallback from CUDA to CPU if VRAM unavailable; raises `TranscriptionError` if zero speech detected.

### 3. Highlight Analysis Contract (`clippa.analysis` ↔ `clippa.reframing`)
- Function: `extract_viral_segments(transcript: TranscriptData, max_clips: int = 3, llm_model: str = "ollama:llama3.2:3b") -> list[ViralSegment]`
- Output Schema:
  ```json
  [
    {
      "start_time_ms": 15000,
      "end_time_ms": 48000,
      "hook_title": "The Secret Nobody Told You",
      "hook_score": 22,
      "engagement_score": 23,
      "value_score": 21,
      "shareability_score": 24,
      "total_score": 90,
      "reasoning": "Strong opening question with high emotional resonance."
    }
  ]
  ```
- Error Handling: Automatic fallback to rule-based heuristic virality scorer if Ollama is unreachable.

### 4. Reframing & Subtitle Contract (`clippa.reframing` ↔ `clippa.rendering`)
- Function: `build_crop_filter(video_path: Path, start_ms: int, end_ms: int, target_w: int = 1080, target_h: int = 1920) -> str`
- Returns: FFmpeg filter string (e.g. `crop=1080:1920:x:y,scale=1080:1920`)
- Function: `generate_ass_subtitles(words: list[Word], hook_title: str, template: str, target_w: int = 1080, target_h: int = 1920) -> Path`
- Returns: Path to generated `.ass` file.

### 5. Final Rendering Contract (`clippa.rendering` ↔ `clippa.cli`)
- Function: `render_clip(source_video: Path, start_ms: int, end_ms: int, crop_filter: str, ass_path: Path, output_file: Path) -> Path`
- Executes FFmpeg to output valid 9:16 vertical MP4 video with H.264 video, AAC audio, and burned subtitles.
- Returns: Path to verified rendered MP4 file.

---

## Code Layout
```
c:\Users\madha\OneDrive\Documents\PERSONAL PROJECTS\CLIPPA\
├── .venv\                     # Python 3.11 virtual environment
├── backend\                   # FujiwaraChoki/supoclip backend source
│   ├── fonts\                 # Bundled TTF fonts for subtitles
│   ├── src\
│   │   ├── media\             # Video processing, FFmpeg, YuNet ONNX, ASS captions
│   │   ├── services\          # VideoService, TaskService
│   │   ├── ai.py              # Pydantic AI virality & Ollama integration
│   │   ├── config.py          # Configuration & environment variables
│   │   └── youtube_utils.py   # yt-dlp wrapper
│   └── pyproject.toml
├── clippa\                    # Unified CLIPPA local-first pipeline package
│   ├── __init__.py
│   ├── __main__.py            # CLI entry point (`python -m clippa`)
│   ├── cli.py                 # Argument parser & workflow coordinator
│   ├── ingestion.py           # yt-dlp downloader & preflight validator
│   ├── transcription.py       # Whisper local transcription (CUDA/CPU)
│   ├── analysis.py            # Ollama & heuristic virality segment selector
│   ├── reframing.py           # YuNet face tracking & 9:16 crop calculator
│   ├── subtitles.py           # ASS word karaoke & hook title builder
│   └── rendering.py           # FFmpeg clip renderer & verification
├── tests\                     # E2E & integration test suite
│   ├── runner.py              # Test suite runner
│   ├── tier1_features\        # Tier 1 feature coverage tests
│   ├── tier2_boundaries\      # Tier 2 boundary & edge case tests
│   ├── tier3_combinations\    # Tier 3 pairwise integration tests
│   └── tier4_real_world\      # Tier 4 end-to-end YouTube ingestion & clipping
├── outputs\                   # Rendered 9:16 vertical clips
├── .env                       # Local-first environment configuration
├── PROJECT.md                 # Living architecture & milestone tracker
├── TEST_INFRA.md              # E2E testing framework index
├── TEST_READY.md              # E2E test suite ready signal
└── ORIGINAL_REQUEST.md        # Verbatim original user request
```
