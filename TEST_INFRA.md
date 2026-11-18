# TEST_INFRA.md — CLIPPA Opaque-Box E2E Testing Framework

## 1. Test Philosophy

CLIPPA is a local-first, privacy-respecting video clipping pipeline designed to convert landscape YouTube videos into viral 9:16 vertical shorts with zero external paid API subscriptions.

The testing architecture adheres to strict **opaque-box** testing principles:
- **Contract-Driven & Implementation-Agnostic**: Tests exercise public CLI commands (`python -m clippa`) and public module interfaces (`clippa.*`). Private classes, helper functions, and internal state are never asserted upon.
- **Observable Artifact Verification**: Assertions validate generated filesystem artifacts: output MP4 video containers, stream parameters via `ffprobe` (1080x1920 / 720x1280 resolution, H.264 video codec, AAC audio codec, duration), ASS subtitle files, and JSON transcript caches.
- **Deterministic & Fast Synthetic Testing (Tiers 1–3)**: Tiers 1 to 3 execute offline using synthetic media fixtures generated on-the-fly via FFmpeg and procedural data. No external network requests or heavy model downloads are required for standard test passes.
- **Zero-Paid-API Guarantee**: All tests assert that `SELF_HOST=true` executes without calling paid cloud endpoints (AssemblyAI, OpenAI, Anthropic, Apify, Stripe).
- **Comprehensive 4-Tier Hierarchy**:
  1. **Tier 1 — Feature Coverage**: Category-Partition testing for each of the 22 features (≥5 cases per feature = 110 cases).
  2. **Tier 2 — Boundary Value Analysis & Edge Cases**: Extreme values, malformed inputs, boundary limits, and error handling (≥5 cases per feature = 110 cases).
  3. **Tier 3 — Pairwise Cross-Feature Combinations**: Seam and integration tests between pipeline stages (15 combinatorial pairs).
  4. **Tier 4 — Real-World Application Scenarios**: Full pipeline end-to-end runs with real video profiles (5 realistic workloads).

---

## 2. Feature Inventory with Test Counts

| # | Feature | Public Module / CLI | Tier 1 (Partition) | Tier 2 (Boundaries) | Tier 3 (Pairwise) | Tier 4 (Workload) | Total Cases |
|---|---------|---------------------|--------------------|---------------------|-------------------|-------------------|-------------|
| 1 | YouTube Video Download | `clippa.ingestion` / CLI | 5 | 5 | 1 | 1 | 12 |
| 2 | Video Metadata Preflight | `clippa.ingestion` | 5 | 5 | 0 | 0 | 10 |
| 3 | Audio Track Extraction | `clippa.transcription` | 5 | 5 | 1 | 0 | 11 |
| 4 | Local Whisper Transcription | `clippa.transcription` | 5 | 5 | 3 | 1 | 14 |
| 5 | Transcript Cache Sidecar | `clippa.transcription` | 5 | 5 | 1 | 0 | 11 |
| 6 | Viral Highlight & Moment Detection | `clippa.analysis` | 5 | 5 | 3 | 1 | 14 |
| 7 | Rule-Based Heuristic Scorer Fallback | `clippa.analysis` | 5 | 5 | 1 | 1 | 12 |
| 8 | Windowed Long-Video Analysis | `clippa.analysis` | 5 | 5 | 1 | 1 | 12 |
| 9 | Hook Title Generation | `clippa.analysis` | 5 | 5 | 2 | 1 | 13 |
| 10 | Cold-Open Teaser Detection | `clippa.analysis` | 5 | 5 | 1 | 1 | 12 |
| 11 | YuNet ONNX Face Detection | `clippa.reframing` | 5 | 5 | 2 | 1 | 13 |
| 12 | Face Centroid Tracking & Smoothing | `clippa.reframing` | 5 | 5 | 2 | 1 | 13 |
| 13 | Ken Burns Talking Head Zoom | `clippa.reframing` | 5 | 5 | 1 | 1 | 12 |
| 14 | 9:16 Vertical Reframing & Clamping | `clippa.reframing` | 5 | 5 | 2 | 1 | 13 |
| 15 | Animated ASS Subtitle Builder | `clippa.subtitles` | 5 | 5 | 2 | 1 | 13 |
| 16 | Subtitle Style Templates & TTF Fonts | `clippa.subtitles` | 5 | 5 | 1 | 1 | 12 |
| 17 | Contextual Emoji Injection | `clippa.subtitles` | 5 | 5 | 1 | 1 | 12 |
| 18 | Burned Hook Title Overlay | `clippa.subtitles` | 5 | 5 | 1 | 1 | 12 |
| 19 | Subtitle Burning via FFmpeg | `clippa.rendering` | 5 | 5 | 1 | 1 | 12 |
| 20 | Self-Hosted Zero-Paid-API Mode | `clippa.config` / CLI | 5 | 5 | 1 | 1 | 12 |
| 21 | Standalone CLI Pipeline Runner | `clippa.cli` (`__main__`) | 5 | 5 | 1 | 1 | 12 |
| 22 | End-to-End Acceptance Verification | Full Pipeline / CLI | 5 | 5 | 1 | 1 | 12 |
| **TOTALS** | **22 Features** | | **110** | **110** | **15\*** | **5\*** | **240** |

*\*Note: Tier 3 tests span multi-feature pairs (15 pairwise test modules covering cross-feature interaction seams). Tier 4 tests span full real-world pipeline scenarios (5 comprehensive workloads).*

---

## 3. Test Architecture

### 3.1 Directory Layout
```
tests/
├── runner.py                  # Master test suite runner & CLI orchestrator
├── conftest.py                # Pytest fixtures, environment validation, tmp dirs
├── utils.py                   # Media probing, codec verification, ffprobe assertions
├── fixtures/                  # Procedural generators and synthetic data
│   └── synthetic.py           # Procedural FFmpeg video/audio generator & mock data
├── tier1_features/            # Tier 1 Category-Partition tests (110 cases across 22 modules)
│   ├── test_feat01_download.py
│   ├── test_feat02_preflight.py
│   ├── test_feat03_audio_extract.py
│   ├── test_feat04_whisper.py
│   ├── test_feat05_cache.py
│   ├── test_feat06_virality.py
│   ├── test_feat07_heuristic.py
│   ├── test_feat08_windowing.py
│   ├── test_feat09_hook_title.py
│   ├── test_feat10_teaser.py
│   ├── test_feat11_yunet.py
│   ├── test_feat12_tracking.py
│   ├── test_feat13_ken_burns.py
│   ├── test_feat14_reframing.py
│   ├── test_feat15_ass_builder.py
│   ├── test_feat16_styles.py
│   ├── test_feat17_emojis.py
│   ├── test_feat18_hook_overlay.py
│   ├── test_feat19_burning.py
│   ├── test_feat20_self_host.py
│   ├── test_feat21_cli.py
│   └── test_feat22_acceptance.py
├── tier2_boundaries/          # Tier 2 Boundary & Corner Case tests (110 cases across 22 modules)
│   ├── test_b01_download_boundaries.py
│   ├── test_b02_preflight_boundaries.py
│   ├── test_b03_audio_boundaries.py
│   ├── test_b04_whisper_boundaries.py
│   ├── test_b05_cache_boundaries.py
│   ├── test_b06_virality_boundaries.py
│   ├── test_b07_heuristic_boundaries.py
│   ├── test_b08_windowing_boundaries.py
│   ├── test_b09_hook_title_boundaries.py
│   ├── test_b10_teaser_boundaries.py
│   ├── test_b11_yunet_boundaries.py
│   ├── test_b12_tracking_boundaries.py
│   ├── test_b13_ken_burns_boundaries.py
│   ├── test_b14_reframing_boundaries.py
│   ├── test_b15_ass_boundaries.py
│   ├── test_b16_styles_boundaries.py
│   ├── test_b17_emojis_boundaries.py
│   ├── test_b18_hook_overlay_boundaries.py
│   ├── test_b19_burning_boundaries.py
│   ├── test_b20_self_host_boundaries.py
│   ├── test_b21_cli_boundaries.py
│   └── test_b22_acceptance_boundaries.py
├── tier3_combinations/        # Tier 3 Pairwise Cross-Feature tests (15 modules)
│   ├── test_pair01_ingest_whisper.py
│   ├── test_pair02_whisper_ollama.py
│   ├── test_pair03_whisper_heuristic.py
│   ├── test_pair04_virality_hooktitle.py
│   ├── test_pair05_virality_teaser.py
│   ├── test_pair06_yunet_tracking.py
│   ├── test_pair07_tracking_kenburns.py
│   ├── test_pair08_yunet_reframing.py
│   ├── test_pair09_whisper_ass_subtitles.py
│   ├── test_pair10_styles_emojis.py
│   ├── test_pair11_hooktitle_overlay.py
│   ├── test_pair12_reframing_burning.py
│   ├── test_pair13_selfhost_ollama.py
│   ├── test_pair14_cli_e2e_pipeline.py
│   └── test_pair15_windowing_cache.py
└── tier4_real_world/          # Tier 4 Real-World Application Scenarios (5 modules)
    ├── test_rw01_lecture_pipeline.py
    ├── test_rw02_podcast_interview.py
    ├── test_rw03_gaming_commentary.py
    ├── test_rw04_airgapped_offline.py
    └── test_rw05_long_video_batch.py
```

### 3.2 Test Runner Specification (`tests/runner.py`)
The unified runner executes tests without requiring manual pytest flag memorization:
- **Invocation**: `python tests/runner.py [OPTIONS]`
- **Supported Options**:
  - `-t, --tier {1,2,3,4,all}`: Filter test execution by tier (default: `1,2,3`). Comma-separated lists accepted (e.g., `1,2`).
  - `--all`: Execute all test tiers (equivalent to `--tier all`).
  - `--tier4-live`: Enable live YouTube network ingestion in Tier 4 (default: False, uses cached fixture).
  - `-v, --verbose`: Detailed test output with assertion details.
  - `-q, --quiet`: Minimal test output.
  - `-x, --fail-fast`: Halt execution immediately on first failure.
  - `-k, --keyword <expression>`: Run only tests matching keyword expression.
  - `-m, --marker <expression>`: Run only tests matching marker expression.
  - `--dry-run`: List discovered test cases without executing.
  - `--report-json <path>`: Write JSON test execution report (default: `test_report.json`).
  - `--report-md <path>`: Write Markdown test execution summary (default: `test_report.md`).
  - `--keep-artifacts`: Retain generated media fixtures for manual inspection.
- **Exit Codes**:
  - `0`: All selected tests passed (100% success).
  - `1`: One or more tests failed.
  - `2`: Test execution interrupted by user (SIGINT).
  - `3`: Missing critical environment dependency or configuration error.
  - `5`: No tests collected by selection filters.

### 3.3 Synthetic Media Fixture Strategy
To guarantee fast (<60 seconds), reproducible, and network-independent execution of Tiers 1–3:
1. **Synthetic Video Generator (`tests/fixtures/synthetic.py`)**:
   Uses FFmpeg `testsrc2`, `sine`, and `drawbox` filters to generate valid MP4 clips:
   - 16:9 Landscape (`1920x1080`, 30 fps, H.264, 16kHz mono / AAC stereo).
   - Pre-Vertical 9:16 (`1080x1920`).
   - Square 1:1 (`1080x1080`).
   - Moving Face Simulation (`drawbox` moving horizontally across frame).
   - Silent Audio stream generator (`anullsrc`).
   - Pure Audio MP3/WAV generator.
   - Micro-clips and corrupted media generators.
2. **Deterministic Mock Data**:
   Standardized `TranscriptData`, `Word`, and `ViralSegment` objects matching schema version 3 with millisecond timestamps and word arrays for testing virality, hook titles, and subtitles.
3. **Automatic Cleanup**:
   All test clips, temporary audio extracts, and ASS files are generated inside `pytest` `tmp_path` fixtures, guaranteeing zero workspace pollution unless `--keep-artifacts` is passed.

### 3.4 Network Interception & Paid API Guard (`tests/conftest.py`)
To guarantee that CLIPPA operates in genuine self-hosted mode without unverified external network calls:
- `paid_api_interceptor`: Socket-level (`socket.getaddrinfo`, `socket.socket.connect`) and HTTP-client (`http.client.HTTPConnection.connect`) interceptor that intercepts and blocks any outbound network requests to paid API domains (`api.openai.com`, `api.anthropic.com`, `api.assemblyai.com`, `api.apify.com`), enforcing zero paid calls across all tiers.
- `zero_paid_api_guard`: Autouse pytest fixture active across all test cases that monitors socket and HTTP connections and raises `PaidAPIBlockedError` if any unauthorized external paid API request is initiated.

---

## 4. Comprehensive Test Matrix

### 4.1 Tier 1: Feature Coverage (Category-Partition) — 110 Cases

#### Feature 1: YouTube Video Download (`clippa.ingestion`)
- **TC-F01-01**: Ingest standard YouTube watch URL (`https://www.youtube.com/watch?v=...`) and download best MP4 stream capped at 1080p.
- **TC-F01-02**: Ingest shortened YouTube URL (`https://youtu.be/...`) and download valid MP4.
- **TC-F01-03**: Ingest YouTube embed URL (`https://www.youtube.com/embed/...`) and download valid MP4.
- **TC-F01-04**: When 4K/8K stream is present, verify selector caps resolution to 1080p to preserve local compute resources.
- **TC-F01-05**: Passing local file path instead of YouTube URL bypasses download step and returns verified absolute path.

#### Feature 2: Video Metadata Preflight (`clippa.ingestion`)
- **TC-F02-01**: Extract title, duration, author, and id without downloading full video payload.
- **TC-F02-02**: Handle video titles containing special characters, Unicode, and punctuation cleanly without crash.
- **TC-F02-03**: Detect livestream URL and return duration 0 or raise `StreamDurationUnavailableError`.
- **TC-F02-04**: Reject video exceeding maximum configured duration (`max_video_duration = 5400s`) during preflight.
- **TC-F02-05**: Verify preflight metadata cache prevents duplicate network preflight queries for same URL.

#### Feature 3: Audio Track Extraction (`clippa.transcription`)
- **TC-F03-01**: Extract mono 16kHz audio from standard 44.1kHz stereo AAC video source.
- **TC-F03-02**: Extract mono 16kHz audio from Opus audio stream in video.
- **TC-F03-03**: Downmix multi-channel (5.1 surround) audio source cleanly into single-channel mono without clipping.
- **TC-F03-04**: Extracted audio duration matches source video duration within ±50ms.
- **TC-F03-05**: Video with missing audio stream raises `AudioExtractionError` without hanging FFmpeg process.

#### Feature 4: Local Whisper Transcription (`clippa.transcription`)
- **TC-F04-01**: Local Whisper on CUDA transcribes audio and produces word-level millisecond timestamps.
- **TC-F04-02**: Automatic fallback to CPU transcription when CUDA is unavailable.
- **TC-F04-03**: Model size selection (`tiny`, `base`, `small`) instantiates corresponding local Whisper checkpoint.
- **TC-F04-04**: Output timestamps are strictly monotonic (`start < end` and `end <= next_start`).
- **TC-F04-05**: Confidence scores for all transcribed words are within valid float range `[0.0, 1.0]`.

#### Feature 5: Transcript Cache Sidecar (`clippa.transcription`)
- **TC-F05-01**: Transcription generates `<video_path>.transcript_cache.json` matching version 3 schema.
- **TC-F05-02**: Subsequent transcription request on same file loads from sidecar cache, skipping Whisper execution.
- **TC-F05-03**: Cache file validates fields: `words`, `utterances`, `text`, `language_code`, `version`.
- **TC-F05-04**: Corrupted/truncated cache file is detected and safely regenerated from source audio.
- **TC-F05-05**: `--no-cache` / `force_retranscribe=True` flag bypasses existing cache and overwrites sidecar file.

#### Feature 6: Viral Highlight & Moment Detection (`clippa.analysis`)
- **TC-F06-01**: Ollama analysis evaluates segments across 4 dimensions: hook, engagement, value, shareability (0–25 each).
- **TC-F06-02**: Total virality score equals sum of 4 dimensions (0–100 scale).
- **TC-F06-03**: Selected viral segments strictly adhere to duration bounds: 15s ≤ duration ≤ 60s.
- **TC-F06-04**: Multiple extracted segments are sorted in descending order of total virality score.
- **TC-F06-05**: Each segment includes non-empty text justification (`reasoning`) and valid start/end timestamps.

#### Feature 7: Rule-Based Heuristic Scorer Fallback (`clippa.analysis`)
- **TC-F07-01**: When Ollama daemon is offline/unreachable, pipeline automatically falls back to heuristic scorer without error.
- **TC-F07-02**: Heuristic scorer evaluates power words ("secret", "mistake", "why", "never") and question marks to compute hook score.
- **TC-F07-03**: Heuristic output conforms to exact `ViralSegment` schema as LLM output.
- **TC-F07-04**: Heuristic segments respect 15–60s duration constraints.
- **TC-F07-05**: Explicit flag (`--llm-model heuristic`) executes heuristic engine directly without probing Ollama.

#### Feature 8: Windowed Long-Video Analysis (`clippa.analysis`)
- **TC-F08-01**: Videos ≤ 15 minutes are analyzed in a single unified window.
- **TC-F08-02**: Videos > 15 minutes are partitioned into 12-minute windows with 2-minute overlap.
- **TC-F08-03**: Candidate segments overlapping across window boundaries by >50% are deduplicated, keeping highest-scoring segment.
- **TC-F08-04**: Deduplication selects top candidate segments without temporal collisions.
- **TC-F08-05**: Clip quota scales proportionally with footage duration (`max_clips` clamped cleanly).

#### Feature 9: Hook Title Generation (`clippa.analysis`)
- **TC-F09-01**: AI generates punchy hook title between 3 and 9 words matching segment premise.
- **TC-F09-02**: Hook title character length does not exceed 64 characters.
- **TC-F09-03**: Title formatting excludes social hashtags (`#`) and surrounding quotation marks.
- **TC-F09-04**: Heuristic fallback generates hook title from segment's opening clause when LLM title is unavailable.
- **TC-F09-05**: Generated hook title is preserved in segment metadata for downstream subtitle burning.

#### Feature 10: Cold-Open Teaser Detection (`clippa.analysis`)
- **TC-F10-01**: Detects 2–5s punchline segment within clip as teaser candidate (`hook_start_time_ms`, `hook_end_time_ms`).
- **TC-F10-02**: Clips opening with an immediate high-scoring statement omit teaser (`hook_start_time_ms = None`).
- **TC-F10-03**: Teaser duration strictly conforms to 2.0s ≤ teaser ≤ 5.0s.
- **TC-F10-04**: Teaser timestamps align with exact Whisper word boundaries.
- **TC-F10-05**: Teaser bounds sit strictly within segment start and end boundaries.

#### Feature 11: YuNet ONNX Face Detection (`clippa.reframing`)
- **TC-F11-01**: Loads bundled `face_detection_yunet_2023mar.onnx` via OpenCV without external download.
- **TC-F11-02**: Detects frontal human faces in frame and returns bounding boxes `(x, y, w, h)` with confidence scores.
- **TC-F11-03**: Downscales high-resolution frames wider than 960px to maintain ≥30 fps detection throughput.
- **TC-F11-04**: Detects multiple faces in frame and returns candidate list.
- **TC-F11-05**: Returns empty list `[]` on blank black frame without throwing exceptions.

#### Feature 12: Face Centroid Tracking & Smoothing (`clippa.reframing`)
- **TC-F12-01**: Computes weighted face centroid prioritizing primary speaking face in talking head video.
- **TC-F12-02**: Temporal exponential smoothing eliminates high-frequency jitter across consecutive frames.
- **TC-F12-03**: Face absent defaults trajectory to frame center `(width / 2, height / 2)`.
- **TC-F12-04**: Horizontal crop trajectory clamps strictly inside source video boundaries.
- **TC-F12-05**: Static speaker produces stable, non-oscillating crop window coordinates.

#### Feature 13: Ken Burns Talking Head Zoom (`clippa.reframing`)
- **TC-F13-01**: Applies subtle Ken Burns zoom (1.0 to 1.05 scale) on static single-speaker segments.
- **TC-F13-02**: Zoom effect automatically suppresses when horizontal face tracking velocity exceeds pan threshold.
- **TC-F13-03**: Filtergraph compiles valid FFmpeg `zoompan` or crop-scale expressions without frame stutter.
- **TC-F13-04**: Video duration and audio synchronization remain perfectly preserved after Ken Burns effect.
- **TC-F13-05**: Configurable flag (`--no-zoom` / `enable_ken_burns=False`) disables zoom effect completely.

#### Feature 14: 9:16 Vertical Reframing & Clamping (`clippa.reframing`)
- **TC-F14-01**: Crops standard 1920x1080 landscape footage to exact 1080x1920 vertical video filter string.
- **TC-F14-02**: Crops 1280x720 video to exact 720x1280 vertical video when 720p output is configured.
- **TC-F14-03**: Non-standard source aspect ratio (4:3 or 21:9) crops to 9:16 with even width and height coordinates.
- **TC-F14-04**: Crop coordinates `x` and `y` are strictly clamped: `0 <= x <= src_w - crop_w` and `0 <= y <= src_h - crop_h`.
- **TC-F14-05**: Generated crop filter string parses cleanly in FFmpeg `-vf crop=...` without syntax error.

#### Feature 15: Animated ASS Subtitle Builder (`clippa.subtitles`)
- **TC-F15-01**: Generates valid ASS subtitle file containing standard headers (`[Script Info]`, `[V4+ Styles]`, `[Events]`).
- **TC-F15-02**: Emits word-by-word karaoke timing tags (`\k` / `\K`) matching Whisper millisecond intervals.
- **TC-F15-03**: Active spoken word animates with pop scale effect and highlight color transition.
- **TC-F15-04**: Subtitle lines are capped at 3 to 5 words to optimize mobile readability in vertical format.
- **TC-F15-05**: Subtitle event start and end timestamps match audio speech timestamps within 20ms.

#### Feature 16: Subtitle Style Templates & TTF Fonts (`clippa.subtitles`)
- **TC-F16-01**: Theme selection `hormozi` produces bold uppercase typography, green highlight, and pill-box styling.
- **TC-F16-02**: Theme selection `mrbeast` produces yellow text with bold black outline and red highlight.
- **TC-F16-03**: Theme selection `tiktok` loads `TikTokSans-Bold.ttf` with fast tempo transitions.
- **TC-F16-04**: Referencing local font files verifies existence in `backend/fonts/` directory.
- **TC-F16-05**: Specifying an invalid template name falls back gracefully to `default` theme without crashing.

#### Feature 17: Contextual Emoji Injection (`clippa.subtitles`)
- **TC-F17-01**: Spoken trigger words ("money", "dollar", "cash") trigger contextual emoji insertion (💰 / 💵).
- **TC-F17-02**: Emotion keywords ("fire", "danger", "warning") insert corresponding visual emojis (🔥 / ⚠️).
- **TC-F17-03**: Emoji placement does not corrupt word start/end millisecond timing tags in ASS file.
- **TC-F17-04**: Segments with no matching keyword triggers generate clean subtitles without rogue emojis.
- **TC-F17-05**: Configurable flag (`--no-emojis` / `enable_emojis=False`) disables emoji injection.

#### Feature 18: Burned Hook Title Overlay (`clippa.subtitles`)
- **TC-F18-01**: Hook title is placed at top safe area (10-15% from top) preventing overlap with center subtitles.
- **TC-F18-02**: Hook title appears at timestamp 00:00:00.00 and fades out after 3.5 to 4.0 seconds.
- **TC-F18-03**: Hook title style uses high-contrast background banner or stroke for legibility across diverse video backgrounds.
- **TC-F18-04**: If hook title is empty or omitted, top overlay dialogue event is cleanly omitted from ASS file.
- **TC-F18-05**: Hook title with long text auto-wraps into two lines without truncating.

#### Feature 19: Subtitle Burning via FFmpeg (`clippa.rendering`)
- **TC-F19-01**: FFmpeg command executes with `-vf "ass='...'"` (or combined crop + ass filter), burning subtitles into pixel data.
- **TC-F19-02**: Windows absolute file paths with backslashes and spaces are properly escaped for FFmpeg filter syntax.
- **TC-F19-03**: Output video stream encodes as H.264 (avc1) playable in standard web/mobile players.
- **TC-F19-04**: Output audio stream encodes as AAC with 44.1kHz or 48kHz.
- **TC-F19-05**: Burning fails gracefully with descriptive error if ASS subtitle file has syntax errors.

#### Feature 20: Self-Hosted Zero-Paid-API Mode (`clippa.config` / CLI)
- **TC-F20-01**: Running with empty/dummy API keys and `SELF_HOST=true` executes pipeline without authentication errors.
- **TC-F20-02**: Video duration quotas and minute restrictions are disabled in self-hosted mode.
- **TC-F20-03**: Full 1080p export resolution is accessible without subscription paywall.
- **TC-F20-04**: All transcription requests route to local Whisper; zero network calls to AssemblyAI.
- **TC-F20-05**: All virality scoring requests route to local Ollama or heuristic engine; zero network calls to OpenAI/Claude.

#### Feature 21: Standalone CLI Pipeline Runner (`clippa.cli` / `__main__`)
- **TC-F21-01**: CLI executes with minimal arguments: `python -m clippa --url <URL>` and outputs clips to default `./outputs/`.
- **TC-F21-02**: CLI accepts custom options `--output-dir`, `--template hormozi`, `--max-clips 2`, `--whisper-model tiny`.
- **TC-F21-03**: CLI displays formatted `--help` documentation explaining all options and arguments.
- **TC-F21-04**: Missing required arguments (`--url`) returns exit code 2 with usage instructions.
- **TC-F21-05**: CLI emits structured progress logs to stdout (Ingesting -> Transcribing -> Analyzing -> Reframing -> Rendering).

#### Feature 22: End-to-End Acceptance Verification (Acceptance Gate)
- **TC-F22-01**: Complete pipeline run produces at least one rendered 9:16 vertical MP4 clip.
- **TC-F22-02**: Rendered clip has exact aspect ratio 9:16 (1080x1920 or 720x1280) verified via ffprobe.
- **TC-F22-03**: Rendered clip contains valid video stream (`codec_name = h264`) and audio stream (`codec_name = aac`).
- **TC-F22-04**: Rendered clip duration matches selected viral segment duration (+/- 1.0s).
- **TC-F22-05**: Entire run completes locally without invoking any paid third-party API service.

---

### 4.2 Tier 2: Boundary Value Analysis & Corner Cases — 110 Cases

#### Feature 1: YouTube Video Download (Boundaries)
- **TC-B01-01**: Empty string or whitespace-only URL raises `InvalidURLError` or `VideoDownloadError` immediately before network attempt.
- **TC-B01-02**: Malformed URL without video ID raises `InvalidURLError` or `VideoDownloadError`.
- **TC-B01-03**: Private or deleted YouTube video URL raises `VideoUnavailableError` or `VideoDownloadError` with graceful exit.
- **TC-B01-04**: Age-restricted / sign-in required video triggers clean authentication error message rather than unhandled trace.
- **TC-B01-05**: Live stream in progress (duration 0 or negative) raises `StreamDurationUnavailableError` and exits cleanly.

#### Feature 2: Video Metadata Preflight (Boundaries)
- **TC-B02-01**: Zero-duration video (0 seconds) in preflight metadata is rejected before download.
- **TC-B02-02**: Video duration exceeding cap (e.g. 5401s vs 5400s max) triggers duration limit rejection.
- **TC-B02-03**: Video title with 500+ characters or complex Unicode code points (Cyrillic, Arabic, emojis) handled without truncation crash.
- **TC-B02-04**: Video with missing author in metadata substitutes default fallback string.
- **TC-B02-05**: Network socket timeout (1s on hanging connection) fails fast with `PreflightTimeoutError` or network error.

#### Feature 3: Audio Track Extraction (Boundaries)
- **TC-B03-01**: Zero-duration or 0-byte video file raises descriptive error on audio extraction.
- **TC-B03-02**: Pure silence audio track extracts valid silent audio file without FFmpeg encoder crash.
- **TC-B03-03**: Source video with zero audio streams raises `AudioExtractionError` without hanging.
- **TC-B03-04**: Extreme audio sample rate (96kHz or 8kHz) successfully resamples to required 16kHz mono.
- **TC-B03-05**: Corrupted audio frames at end of file truncate cleanly without failing entire extraction.

#### Feature 4: Local Whisper Transcription (Boundaries)
- **TC-B04-01**: Pure silence or ambient noise audio (zero speech detected) returns empty word list without throwing unhandled exception.
- **TC-B04-02**: Ultra-fast speech rate (>250 WPM) maintains monotonic word timestamps and does not drop subsequent segments.
- **TC-B04-03**: Ultra-slow speech (<40 WPM, long pauses) correctly creates separate utterances rather than one infinite segment.
- **TC-B04-04**: Low signal-to-noise ratio audio (heavy background music) assigns lower confidence scores but completes without crash.
- **TC-B04-05**: Audio clip exactly 1 word in length produces valid 1-item word list with accurate start/end timestamps.

#### Feature 5: Transcript Cache Sidecar (Boundaries)
- **TC-B05-01**: Empty (0-byte) cache file is detected as corrupt and triggers automated regeneration.
- **TC-B05-02**: Malformed JSON cache file (missing closing bracket) triggers regeneration without crashing pipeline.
- **TC-B05-03**: Cache file with mismatched schema version triggers version upgrade or re-transcription.
- **TC-B05-04**: Read-only directory where cache cannot be written logs warning and continues pipeline in-memory.
- **TC-B05-05**: Cache containing zero words logs warning and prevents downstream AI analysis crashes.

#### Feature 6: Viral Highlight & Moment Detection (Boundaries)
- **TC-B06-01**: Total video transcript duration under 15 seconds clamps clip to full video duration.
- **TC-B06-02**: Ollama returns malformed JSON or markdown codeblocks (e.g. ```json ... ```) — robust regex extraction parses payload.
- **TC-B06-03**: Ollama returns segment start/end timestamps outside transcript duration — boundary snapping clamps to `[0, total_duration]`.
- **TC-B06-04**: Segment candidate with end_time <= start_time is dropped during validation.
- **TC-B06-05**: Ollama returns scores outside 0-25 or 0-100 range — validator clamps scores to valid numeric ranges.

#### Feature 7: Rule-Based Heuristic Scorer Fallback (Boundaries)
- **TC-B07-01**: Transcript with zero punctuation segments sentences using word gaps > 500ms.
- **TC-B07-02**: Transcript containing no recognized power keywords still scores using speech rate and volume variance.
- **TC-B07-03**: Transcript with repetitive single word ("test test test") does not divide by zero in density calculation.
- **TC-B07-04**: Transcript exactly 15 seconds produces single valid segment with valid scores.
- **TC-B07-05**: Extremely long transcript (>10,000 words) computes heuristic scores in <1 second without memory spike.

#### Feature 8: Windowed Long-Video Analysis (Boundaries)
- **TC-B08-01**: Video duration exactly at boundary (15 minutes / 900 seconds) handles transition between single and windowed mode cleanly.
- **TC-B08-02**: Video duration just over boundary (15 minutes + 1 second) triggers 2 windows without off-by-one errors.
- **TC-B08-03**: Last window duration is shorter than standard window size (<12 minutes) without index out-of-range error.
- **TC-B08-04**: Two candidate clips with identical scores in overlapping window region resolve stably via deterministic tie-breaker.
- **TC-B08-05**: Video with 2+ hours duration limits total returned clips to configured maximum (`max_clips`).

#### Feature 9: Hook Title Generation (Boundaries)
- **TC-B09-01**: AI generates 1-word title (below minimum 3 words) — validator appends context or uses fallback.
- **TC-B09-02**: AI generates excessively long title (>15 words or >100 characters) — validator truncates cleanly to 9 words / 64 characters.
- **TC-B09-03**: AI generates title containing only punctuation or emojis — validator strips emojis or substitutes segment opening line.
- **TC-B09-04**: Empty string returned from LLM defaults to sanitized video title or snippet.
- **TC-B09-05**: Title containing double quotes, backslashes, or ASS escape characters (`{`, `}`, `\`) escapes properly to prevent ASS injection.

#### Feature 10: Cold-Open Teaser Detection (Boundaries)
- **TC-B10-01**: Proposed teaser duration shorter than 2.0s is rejected or padded to minimum 2.0s.
- **TC-B10-02**: Proposed teaser duration longer than 5.0s is truncated to maximum 5.0s.
- **TC-B10-03**: Teaser timestamp located after clip end timestamp is discarded.
- **TC-B10-04**: Teaser timestamp identical to clip start timestamp (teaser is already the clip start) disables cold-open prepend.
- **TC-B10-05**: Clip total length < 20s automatically suppresses cold-open teaser to prevent repetitive short loops.

#### Feature 11: YuNet ONNX Face Detection (Boundaries)
- **TC-B11-01**: Blank solid black or solid white video frame detects 0 faces without crashing.
- **TC-B11-02**: Micro-resolution video frame (e.g. 160x120) runs face detector without dimension assertion failure.
- **TC-B11-03**: Ultra-high resolution 4K frame (3840x2160) downscales correctly to 960px width without memory error.
- **TC-B11-04**: Crowded scene with 20+ detected faces selects primary speaking face based on size and centrality without slowing pipeline.
- **TC-B11-05**: Extremely low confidence detections (score < 0.3) are filtered out, preventing false positives on background objects.

#### Feature 12: Face Centroid Tracking & Smoothing (Boundaries)
- **TC-B12-01**: Face position hugging the extreme left edge (`x = 0`) clamps crop window so it never samples negative coordinates (`crop_x >= 0`).
- **TC-B12-02**: Face position hugging the extreme right edge (`x = src_w`) clamps crop window so `crop_x + crop_w <= src_w`.
- **TC-B12-03**: Face intermittently disappearing and reappearing maintains smooth interpolation without sudden snaps.
- **TC-B12-04**: Instantaneous full-frame scene change instantly resets centroid without 30-frame drift.
- **TC-B12-05**: Continuous zig-zag motion in source footage is damped by exponential moving average filter.

#### Feature 13: Ken Burns Talking Head Zoom (Boundaries)
- **TC-B13-01**: Video clip shorter than 5 seconds applies scaled zoom curve without abrupt jumps or division by zero.
- **TC-B13-02**: Max zoom boundary cap (e.g. 1.08x) is never exceeded regardless of clip duration.
- **TC-B13-03**: Fast panning speaker footage completely zeroes out zoompan factor to prevent disorienting double motion.
- **TC-B13-04**: Video with odd height/width dimensions retains even coordinates after zoom scaling.
- **TC-B13-05**: Zero duration or 1-frame clip avoids divide-by-zero in zoom step increment calculation.

#### Feature 14: 9:16 Vertical Reframing & Clamping (Boundaries)
- **TC-B14-01**: Native 9:16 vertical input video (1080x1920 source) passes through without unnecessary cropping or distortion.
- **TC-B14-02**: Square 1:1 input video (1080x1080) reframes to 9:16 by adding top/bottom padding or vertical crop.
- **TC-B14-03**: Ultra-wide 21:9 input video (2560x1080) calculates wide-range horizontal crop window safely.
- **TC-B14-04**: Odd-dimension input video (1921x1079) rounds to nearest even integers (`round_to_even`) to satisfy H.264 macroblock constraints.
- **TC-B14-05**: Low-resolution input (480x360) scales up cleanly to 720x1280 or 1080x1920 using bicubic scaling.

#### Feature 15: Animated ASS Subtitle Builder (Boundaries)
- **TC-B15-01**: Words with duration under 50ms (ultra-fast speech) get minimum visible karaoke duration without rendering glitches.
- **TC-B15-02**: Single sentence with 50+ words properly wraps across multiple sequential subtitle events rather than overflowing screen.
- **TC-B15-03**: Zero word gaps between consecutive words transition karaoke highlights smoothly without overlapping tags.
- **TC-B15-04**: Multi-second silence pause between words closes current subtitle event and starts a new event at the next word start.
- **TC-B15-05**: ASS special control characters (`\N`, `\h`, `{}`) inside word text are stripped or escaped to prevent format injection.

#### Feature 16: Subtitle Style Templates & TTF Fonts (Boundaries)
- **TC-B16-01**: Font file missing from `backend/fonts/` falls back to system Arial/Sans font without FFmpeg render crash.
- **TC-B16-02**: Font size setting scaled for 1080x1920 scales proportionally when outputting 720x1280 resolution.
- **TC-B16-03**: Extreme text colors (full transparency `&HFF000000`, white on white) maintain legible outline stroke.
- **TC-B16-04**: Pill-box margin with text touching video edge clamps within safe margins (safe margin >= 40px).
- **TC-B16-05**: Custom template missing required style fields fills with default preset values.

#### Feature 17: Contextual Emoji Injection (Boundaries)
- **TC-B17-01**: Consecutive trigger words in same phrase ("fire money cash") limits emoji insertion to max 1 emoji per line.
- **TC-B17-02**: Trigger word occurring as part of larger word (e.g. "fisherman" containing "fish") uses word boundary regex to avoid false positives.
- **TC-B17-03**: Unsupported Unicode emoji codepoints gracefully fall back or drop without rendering black box / question mark glyphs.
- **TC-B17-04**: Emoji at very end of line does not cause line to wrap unexpectedly onto next line.
- **TC-B17-05**: 100% emoji-free mode (`--no-emojis`) removes all emoji mappings from caption generation.

#### Feature 18: Burned Hook Title Overlay (Boundaries)
- **TC-B18-01**: Hook title on clip shorter than 4 seconds terminates exactly at clip end time rather than exceeding clip duration.
- **TC-B18-02**: Hook title with maximum 64 characters fits within safe title area without overlapping burned subtitle area.
- **TC-B18-03**: Hook title with zero characters generates zero dialogue lines in ASS file without blank placeholder boxes.
- **TC-B18-04**: Hook title containing quotes and apostrophes escapes cleanly in ASS script format.
- **TC-B18-05**: Hook title fade-out duration (500ms) clamps if total clip duration is ultra-short.

#### Feature 19: Subtitle Burning via FFmpeg (Boundaries)
- **TC-B19-01**: Subtitle path containing spaces and special characters (`C:\My Videos\clip [1].ass`) escapes properly in `-vf "ass='...'"` filter.
- **TC-B19-02**: FFmpeg execution in environment with read-only source directory outputs to configured writable target directory.
- **TC-B19-03**: Video file with corrupted PTS/DTS timestamps normalizes timestamps (`-vsync cfr` / `-vf setpts=PTS-STARTPTS`).
- **TC-B19-04**: Audio/video duration mismatch in source (audio longer than video) truncates cleanly to shortest stream (`-shortest`).
- **TC-B19-05**: Process interruption (SIGINT / Ctrl+C) cleans up partial temporary files in output directory.

#### Feature 20: Self-Hosted Zero-Paid-API Mode (Boundaries)
- **TC-B20-01**: Total network disconnection (airplane mode / no internet) runs entire pipeline on local media file without error.
- **TC-B20-02**: Presence of invalid/expired API keys in `.env` (e.g. `OPENAI_API_KEY=sk-invalid`) is completely ignored when `SELF_HOST=true`.
- **TC-B20-03**: Attempting to call non-local endpoints when `SELF_HOST=true` raises an explicit configuration assertion error.
- **TC-B20-04**: Local Ollama timeout (model loading delay > 30s) triggers fallback to heuristic scorer without aborting.
- **TC-B20-05**: Memory allocation cap reached on local GPU triggers graceful CPU offload rather than cloud fallback.

#### Feature 21: Standalone CLI Pipeline Runner (Boundaries)
- **TC-B21-01**: Running CLI with no arguments prints usage and exits with error code 2.
- **TC-B21-02**: Specifying non-existent local file path prints `FileNotFoundError` or error diagnostic and exits cleanly.
- **TC-B21-03**: Passing negative value to `--max-clips` (e.g. `--max-clips -1`) is rejected with argument validation error.
- **TC-B21-04**: Specifying non-existent output directory automatically creates the directory tree.
- **TC-B21-05**: Concurrent execution of two CLI instances with same output directory uses unique file naming to prevent collisions.

#### Feature 22: End-to-End Acceptance Verification (Boundaries)
- **TC-B22-01**: Video containing single speaker with continuous speech renders clip with 100% subtitle timestamp coverage.
- **TC-B22-02**: Video containing zero human faces (nature scenery / screen recording) renders centered 9:16 vertical clip.
- **TC-B22-03**: Video containing fast action / sport scene renders valid 9:16 clip without dropped audio frames.
- **TC-B22-04**: Video with non-English speech transcribes with language detection and renders synchronized subtitles.
- **TC-B22-05**: Output MP4 file has moov atom at beginning of file (`-movflags +faststart`) for instant streaming playback.

---

### 4.3 Tier 3: Pairwise Cross-Feature Interactions — 15 Cases

| Test Case ID | Feature Pair | Interaction Seam | Verification Assertion |
|---|---|---|---|
| **TC-PAIR-01** | F01 (Download) + F04 (Whisper) | Audio stream extracted from yt-dlp MP4 feeds Whisper transcription | Transcribed words match spoken dialogue in downloaded video. |
| **TC-PAIR-02** | F04 (Whisper) + F06 (Ollama) | Whisper millisecond timestamps feed Ollama virality selector | Selected segment timestamps correspond directly to valid Whisper word boundaries. |
| **TC-PAIR-03** | F04 (Whisper) + F07 (Heuristic) | Whisper words feed heuristic scorer when Ollama is stopped | Pipeline seamlessly completes scoring using keyword density without unhandled exceptions. |
| **TC-PAIR-04** | F06 (Ollama) + F09 (Hook Title) | Ollama segment analysis produces 3–9 word hook title | Output segment contains valid non-empty `hook_title` within 64 character budget. |
| **TC-PAIR-05** | F06 (Ollama) + F10 (Teaser) | Ollama virality detection flags 2–5s punchline teaser | If teaser detected, teaser timestamps are strictly contained within parent clip interval. |
| **TC-PAIR-06** | F11 (YuNet) + F12 (Tracking) | YuNet raw face bounding boxes feed exponential smoothing tracker | Output crop centroid trajectory has zero sudden single-frame coordinate spikes. |
| **TC-PAIR-07** | F12 (Tracking) + F13 (Ken Burns) | Pan velocity dynamically controls talking-head zoompan factor | Zoompan is applied when velocity is low; suppressed when velocity exceeds pan threshold. |
| **TC-PAIR-08** | F11 (YuNet) + F14 (9:16 Crop) | Face centroid sets horizontal crop window in 1080x1920 crop filter | Centroid `(x, y)` places detected face within vertical safe zone `[0.2 * H, 0.6 * H]`. |
| **TC-PAIR-09** | F04 (Whisper) + F15 (ASS Builder) | Whisper millisecond start/end intervals drive ASS `\k` karaoke tags | Cumulative karaoke tags on each line match line duration within ±20ms. |
| **TC-PAIR-10** | F16 (Templates) + F17 (Emojis) | Preset theme styling formats lines containing injected emojis | Emojis render beside trigger words without breaking font glyphs or ASS style margins. |
| **TC-PAIR-11** | F09 (Hook Title) + F18 (Overlay) | Generated hook title string is formatted into top-safe-area ASS dialog | Hook title event starts at 0.0s, ends at 3.5s, positioned at `MarginV=120` without overlapping captions. |
| **TC-PAIR-12** | F14 (9:16 Crop) + F19 (Burning) | FFmpeg filtergraph combines `crop=...` and `ass=...` in single pass | Filtergraph renders valid 1080x1920 MP4 with burned subtitles in a single re-encode pass. |
| **TC-PAIR-13** | F20 (Zero-Paid API) + F06 (Ollama) | Pipeline executes virality scoring in `SELF_HOST=true` mode | Zero outbound HTTP connections to `api.openai.com` or `api.anthropic.com`. |
| **TC-PAIR-14** | F21 (CLI Runner) + F22 (E2E Pass) | Single CLI invocation runs full ingestion -> transcribe -> crop -> render | CLI outputs valid playable 9:16 MP4 with audio and subtitles to `--output-dir`. |
| **TC-PAIR-15** | F05 (Cache Sidecar) + F08 (Windowing)| Windowed long video analysis reads from `.transcript_cache.json` | 30-minute transcript cache partitions across 12-minute windows without re-transcribing. |

---

### 4.4 Tier 4: Real-World Application Scenarios — 5 Workloads

| Scenario ID | Workload Profile | Source Video Characteristics | Processing Pipeline Profile | Expected Outputs & Verification Assertions |
|---|---|---|---|---|
| **RW-01** | Technical / Educational Lecture | 10–15 min single-speaker talking head, 1080p landscape, clear audio | Local Whisper `base`, Ollama virality scoring, YuNet face centering, Hormozi subtitle preset | - Output: 1080x1920 vertical MP4 (`rendered_clip.exists()`)<br>- Real ffprobe stream verification (`assert_vertical_9_16`, `assert_valid_h264_aac`)<br>- Subtitle burning verified via frame pixel variance (`assert_burned_subtitles_present`)<br>- Media stream integrity verified (`assert_media_integrity`)<br>- Zero paid API calls verified via `paid_api_interceptor.assert_zero_calls()` |
| **RW-02** | 2-Person Podcast / Interview | 20–30 min two-speaker conversation, alternating camera angles | Windowed analysis (12m/2m), heuristic scorer fallback, YuNet multi-face detection, TikTok subtitle preset | - Output: 2 to 3 distinct vertical MP4 clips<br>- Real ffprobe stream verification on each clip (`assert_vertical_9_16`, `assert_valid_h264_aac`)<br>- Audio/video duration sync within 0.5s (`assert_audio_duration_sync`)<br>- Heuristic virality fallback executed when Ollama is offline<br>- Zero paid API calls verified via `paid_api_interceptor.assert_zero_calls()` |
| **RW-03** | Fast High-Energy Commentary | 5–8 min high speech rate (>200 WPM), background music | Local Whisper on GPU, Ollama scoring, Ken Burns zoom enabled, MrBeast subtitle preset | - Output: 1080x1920 vertical MP4<br>- Real ffprobe stream verification (`assert_vertical_9_16`, `assert_valid_h264_aac`)<br>- Non-silent audio track verification (`assert_non_silent_audio`)<br>- Burned pixel presence confirmed (`assert_burned_subtitles_present`)<br>- Zero paid API calls verified via `paid_api_interceptor.assert_zero_calls()` |
| **RW-04** | Air-Gapped / Offline Local Media | Pre-existing local `.mp4` file in workspace (zero network access) | CLI run: `clippa --url sample.mp4 --llm-model heuristic`, Default subtitle preset | - Air-gapped / offline local media processing (`SELF_HOST=true`)<br>- Rendered 9:16 MP4 verified via `probe_media`<br>- Zero network queries to YouTube or external APIs (`paid_api_interceptor.assert_zero_calls()`)<br>- Valid container streams (`assert_valid_h264_aac`) |
| **RW-05** | Batch Highlight Extraction | 25 min multi-topic video (keynote presentation) | yt-dlp download, 3-window analysis, deduplication, cold-open teaser detection, batch render | - Batch highlight extraction produces multiple distinct 9:16 MP4 clips<br>- Clip durations between 15s and 60s verified via `probe_media`<br>- Codec and geometry verification on all generated batch clips<br>- Media integrity verification on all batch outputs (`assert_media_integrity`) |

---

## 5. Coverage Thresholds & Quality Gates

To achieve production certification and publish `TEST_READY.md`:

| Gate Dimension | Minimum Requirement | Enforcement Mechanism |
|---|---|---|
| **Feature Coverage (Tier 1)** | 100% of 22 features covered (≥5 tests/feature = 110 tests) | `tests/runner.py --tier 1` automated check |
| **Boundary Coverage (Tier 2)**| 100% of 22 features covered (≥5 tests/feature = 110 tests) | `tests/runner.py --tier 2` automated check |
| **Pairwise Combinations (Tier 3)**| 15 core cross-feature interaction seams verified | `tests/runner.py --tier 3` automated check |
| **Real-World Workloads (Tier 4)** | 5 realistic application workloads pass acceptance | `tests/runner.py --tier 4` execution |
| **Pass Rate Requirement** | **100% pass rate** (0 failures, 0 errors, 0 unhandled exceptions)| Exit code 0 required from `tests/runner.py` |
| **Execution Performance** | Tiers 1–3 must execute in under 60 seconds combined | Pytest duration profiling with synthetic fixtures |
| **Zero Paid APIs** | 0 outbound network requests to paid API domains | Enforced via socket & HTTP interceptor fixture (`paid_api_interceptor`) in `conftest.py` |
