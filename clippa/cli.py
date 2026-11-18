"""
clippa.cli - Standalone CLI Pipeline Runner for CLIPPA.

Feature 21: Standalone CLI Pipeline Runner
Orchestration: Ingestion -> Transcription -> Highlight Analysis -> Reframing & Subtitles -> Rendering.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import List, Optional

from clippa.transcription import Word, transcribe_audio
from clippa.analysis import extract_viral_segments
from clippa.reframing import build_crop_filter
from clippa.subtitles import generate_ass_subtitles
from clippa.ingestion import download_video, is_valid_youtube_url, get_video_title
from clippa.rendering import render_clip

# Configure structured logging to stdout
logger = logging.getLogger("clippa")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [clippa] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def build_parser() -> argparse.ArgumentParser:
    """Constructs command line argument parser for CLIPPA."""
    parser = argparse.ArgumentParser(
        prog="clippa",
        description="CLIPPA: Local-First Automated Video Clipping Pipeline (yt-dlp -> Whisper -> Ollama -> YuNet -> ASS -> FFmpeg)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        required=False,
        default=None,
        type=str,
        help="YouTube video URL or local video file path",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        default=False,
        help="Launch the local web dashboard and in-browser video previewer",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for local web dashboard (default: 8765)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where rendered 9:16 vertical clips will be saved",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="temp",
        help="Directory for temporary intermediate files",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="default",
        choices=["default", "hormozi", "mrbeast", "tiktok"],
        help="Subtitle styling and animation preset",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=3,
        help="Maximum number of top viral clips to extract (must be > 0)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="1080p",
        choices=["1080p", "720p"],
        help="Target vertical video resolution (1080p: 1080x1920, 720p: 720x1280)",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        help="Local Whisper model size (tiny, base, small, medium, large-v3)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="ollama:llama3.2:3b",
        help="Virality analysis model ('ollama:llama3.2:3b' or 'heuristic')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Processing device for Whisper and YuNet face tracking",
    )
    parser.add_argument(
        "--enable-emojis",
        action="store_true",
        default=False,
        help="Enable contextual emoji insertion in subtitles (default: False)",
    )
    parser.add_argument(
        "--no-emojis",
        action="store_true",
        default=False,
        help="Disable contextual emoji insertion (deprecated alias, emojis are disabled by default)",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=120,
        help="Maximum clip duration in seconds (default: 120s / 2 minutes)",
    )
    parser.add_argument(
        "--no-zoom",
        action="store_true",
        default=False,
        help="Disable talking-head Ken Burns dynamic zoom",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Force re-transcription and bypass existing sidecar transcript cache",
    )
    parser.add_argument(
        "--nvenc",
        action="store_true",
        default=False,
        help="Enable NVIDIA NVENC hardware acceleration (h264_nvenc) for ultra-fast GPU rendering",
    )
    parser.add_argument(
        "--auto-upload",
        action="store_true",
        default=False,
        help="Automatically upload top generated clips to connected YouTube account",
    )
    parser.add_argument(
        "--privacy",
        choices=["public", "unlisted", "private", "scheduled"],
        default="public",
        help="Default YouTube privacy status for uploaded clips (public, unlisted, private, scheduled)",
    )
    return parser


def parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses and validates CLI arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "web", False) and not args.url:
        parser.error("the following arguments are required: --url (or launch dashboard with --web)")
    if args.max_clips <= 0:
        parser.error(f"--max-clips must be a positive integer (> 0), got {args.max_clips}")
    return args


def sanitize_clip_filename(title: str, max_length: int = 80) -> str:
    """Sanitizes clip title into a clean, safe Windows filename without illegal or awkward punctuation."""
    if not title or not str(title).strip():
        return ""
    # Strip quotes, apostrophes, backticks
    s = str(title).strip().strip("\"'`“”‘’")
    s = s.replace("'", "").replace("’", "").replace("`", "")
    # Replace non-alphanumeric (except hyphens) with underscores
    safe = re.sub(r"[^\w\-]+", "_", s)
    # Collapse multiple underscores and trim
    safe = re.sub(r"_+", "_", safe).strip("._-")
    reserved = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4",
                "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2",
                "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"}
    if safe.upper() in reserved:
        safe = f"clip_{safe}"
    if len(safe) > max_length:
        safe = safe[:max_length].rstrip("._-")
    return safe


def get_unique_clip_path(output_dir: Path, clip_index: int, title: Optional[str] = None) -> Path:
    """Computes unique non-colliding file path for a clip, named after the clip's title."""
    base = ""
    if title and isinstance(title, str) and title.strip():
        base = sanitize_clip_filename(title)
    if not base:
        base = f"clip_{clip_index}"

    target = output_dir / f"{base}.mp4"
    if not target.exists():
        return target
    counter = 1
    while target.exists():
        counter += 1
        target = output_dir / f"{base}_{counter}.mp4"
    return target


def run_pipeline(
    args: Optional[argparse.Namespace] = None,
    *,
    url: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
    temp_dir: Optional[str | Path] = None,
    template: str = "default",
    max_clips: int = 3,
    whisper_model: str = "base",
    llm_model: str = "ollama:llama3.2:3b",
    enable_emojis: bool = False,
    no_emojis: bool = False,
    no_zoom: bool = False,
    resolution: str = "1080p",
    device: str = "cuda",
    no_cache: bool = False,
    max_duration: int = 120,
    nvenc: bool = False,
    auto_upload: bool = False,
    upload_privacy: str = "public",
) -> int:
    """
    Orchestrates the end-to-end local clipping pipeline.
    Returns 0 on success, 1 on runtime pipeline failure.
    """
    if args is not None:
        url = getattr(args, "url", url)
        output_dir = getattr(args, "output_dir", output_dir)
        temp_dir = getattr(args, "temp_dir", temp_dir)
        template = getattr(args, "template", template)
        max_clips = getattr(args, "max_clips", max_clips)
        whisper_model = getattr(args, "whisper_model", whisper_model)
        llm_model = getattr(args, "llm_model", llm_model)
        enable_emojis = getattr(args, "enable_emojis", enable_emojis)
        no_zoom = getattr(args, "no_zoom", no_zoom)
        resolution = getattr(args, "resolution", resolution)
        device = getattr(args, "device", device)
        no_cache = getattr(args, "no_cache", no_cache)
        max_duration = getattr(args, "max_duration", max_duration)
        nvenc = getattr(args, "nvenc", nvenc)
        auto_upload = getattr(args, "auto_upload", auto_upload)
        upload_privacy = getattr(args, "privacy", upload_privacy)

    if not url or not str(url).strip():
        logger.error("[CLIPPA] No input URL or video path provided.")
        return 1

    out_path = Path(output_dir or "outputs")
    tmp_path = Path(temp_dir or "temp")
    out_path.mkdir(parents=True, exist_ok=True)
    tmp_path.mkdir(parents=True, exist_ok=True)

    target_w, target_h = (720, 1280) if resolution == "720p" else (1080, 1920)

    # 1. Ingestion
    logger.info(f"[CLIPPA] Step 1/5: Ingesting video source '{url}'...")
    try:
        source_video = download_video(url, output_dir=tmp_path, max_height=1080)
        source_video = Path(source_video).resolve()
        if not source_video.exists():
            logger.error(f"[CLIPPA] Source video file does not exist: {source_video}")
            return 1
    except Exception as e:
        logger.error(f"[CLIPPA] Ingestion failed for '{url}': {e}")
        return 1

    # 2. Transcription
    logger.info(f"[CLIPPA] Step 2/5: Transcribing audio with local Whisper ({whisper_model}, device={device})...")
    try:
        transcript = transcribe_audio(
            video_path=source_video,
            model_size=whisper_model,
            device=device,
            force_retranscribe=no_cache,
        )
        if not transcript or not transcript.words:
            logger.error("[CLIPPA] No words detected in source audio. Pipeline cannot extract clips.")
            return 1
    except Exception as e:
        logger.error(f"[CLIPPA] Transcription failed: {e}")
        return 1

    # 3. Highlight Analysis
    logger.info(f"[CLIPPA] Step 3/5: Analyzing viral highlights with '{llm_model}' (max_clips={max_clips})...")
    try:
        from clippa.ingestion import load_video_heatmap
        heatmap = load_video_heatmap(source_video)
        if heatmap:
            logger.info(f"[CLIPPA] Detected YouTube 'Most Replayed' heatmap ({len(heatmap)} intervals)! Sourcing most popular viewer moments...")

        # Sourcing video title for accurate context-aware clip titles
        video_title = None
        if source_video:
            from clippa.ingestion import get_video_title
            raw_title = get_video_title(source_video) or source_video.stem
            cleaned_title = re.sub(r"\s*\[[\w-]+\]$", "", raw_title).strip()
            if cleaned_title and not any(cleaned_title.lower().startswith(p) for p in ["live_synth", "sample", "input", "test", "dummy", "source", "black_"]):
                video_title = cleaned_title

        analysis_kwargs = {
            "transcript": transcript,
            "max_clips": max_clips,
            "llm_model": llm_model,
        }
        if heatmap is not None:
            analysis_kwargs["heatmap"] = heatmap
        if max_duration != 120:
            analysis_kwargs["max_duration_s"] = float(max_duration)
        if video_title:
            analysis_kwargs["video_title"] = video_title

        segments = extract_viral_segments(**analysis_kwargs)
        if not segments:
            logger.error("[CLIPPA] No viral segments identified by analysis engine.")
            return 1
        segments = segments[:max_clips]
    except Exception as e:
        logger.error(f"[CLIPPA] Virality analysis failed: {e}")
        return 1

    # 4 & 5. Reframing, Subtitle Generation, and FFmpeg Rendering
    logger.info(f"[CLIPPA] Step 4/5 & 5/5: Reframing and rendering {len(segments)} vertical clips ({resolution})...")
    rendered_clips: List[Path] = []

    for idx, segment in enumerate(segments, start=1):
        start_ms = int(segment.start_time_ms)
        end_ms = int(segment.end_time_ms)
        clip_duration_s = (end_ms - start_ms) / 1000.0

        is_synthetic = False
        if source_video:
            s_stem = source_video.stem.lower()
            if not is_valid_youtube_url(url) and any(kw in s_stem for kw in ["synth", "sample", "input", "test", "dummy", "source", "black", "offline", "mock"]):
                is_synthetic = True

        title_for_file = None if is_synthetic else segment.hook_title
        clip_output_file = get_unique_clip_path(out_path, idx, title=title_for_file)
        logger.info(f"[CLIPPA] Processing clip {idx}/{len(segments)}: [{start_ms}ms -> {end_ms}ms] -> {clip_output_file.name}")

        try:
            # 4a. Calculate 9:16 crop filter
            crop_filter = build_crop_filter(
                video_path=source_video,
                start_ms=start_ms,
                end_ms=end_ms,
                target_w=target_w,
                target_h=target_h,
                enable_zoom=not no_zoom,
            )

            # 4b. Filter words and shift timestamps relative to clip start
            clip_words: List[Word] = []
            for w in transcript.words:
                w_start = getattr(w, "start", 0)
                w_end = getattr(w, "end", 0)
                w_text = getattr(w, "text", "")
                if w_end >= start_ms and w_start <= end_ms:
                    rel_start = max(0, w_start - start_ms)
                    rel_end = max(rel_start + 50, w_end - start_ms)
                    clip_words.append(Word(text=w_text, start=rel_start, end=rel_end))

            # 4c. Generate ASS subtitles (emojis disabled by default)
            ass_path = tmp_path / f"clip_{idx}_{start_ms}_{end_ms}.ass"
            generate_ass_subtitles(
                words=clip_words,
                hook_title=segment.hook_title,
                template=template,
                target_w=target_w,
                target_h=target_h,
                output_path=ass_path,
                enable_emojis=enable_emojis,
                clip_duration_s=clip_duration_s,
            )

            # 5. Render vertical clip via FFmpeg
            rendered_path = render_clip(
                source_video=source_video,
                start_ms=start_ms,
                end_ms=end_ms,
                crop_filter=crop_filter,
                ass_path=ass_path,
                output_file=clip_output_file,
            )

            if Path(rendered_path).exists():
                rendered_clips.append(Path(rendered_path))
                logger.info(f"[CLIPPA] Rendered clip successfully: {rendered_path}")

                # Generate Virality Metadata & Hashtags Sidecar
                try:
                    import json
                    from clippa.hashtags import extract_hashtags, format_youtube_shorts_title, format_youtube_description

                    clip_text = " ".join(w.text for w in clip_words)
                    source_title = getattr(video_metadata, "title", "") if "video_metadata" in locals() and video_metadata else ""
                    hashtags = extract_hashtags(title=segment.hook_title, text=clip_text)
                    yt_title = format_youtube_shorts_title(segment.hook_title)
                    yt_desc = format_youtube_description(segment.hook_title, source_title=source_title, hashtags=hashtags)

                    meta_payload = {
                        "clip_filename": clip_output_file.name,
                        "title": segment.hook_title,
                        "youtube_title": yt_title,
                        "viral_score": int(getattr(segment, "total_score", 50)),
                        "subscores": {
                            "hook": int(getattr(segment, "hook_score", 15)),
                            "engagement": int(getattr(segment, "engagement_score", 15)),
                            "value": int(getattr(segment, "value_score", 15)),
                            "shareability": int(getattr(segment, "shareability_score", 15)),
                        },
                        "reasoning": getattr(segment, "reasoning", "High-impact viral highlight segment"),
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "duration_s": clip_duration_s,
                        "hashtags": hashtags,
                        "description": yt_desc,
                        "youtube": {
                            "uploaded": False,
                            "video_id": None,
                            "url": None,
                            "privacy": upload_privacy,
                        },
                    }

                    meta_file = clip_output_file.with_name(f"{clip_output_file.stem}.meta.json")
                    meta_file.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
                    logger.info(f"[CLIPPA] Saved clip virality metadata: {meta_file.name}")

                    # Auto-Upload to YouTube if enabled and authenticated
                    if auto_upload:
                        try:
                            from clippa.uploader import is_authenticated, upload_video_to_youtube
                            if is_authenticated():
                                publish_at_iso = None
                                slot_dt = None
                                is_scheduled = (upload_privacy.lower() == "scheduled")

                                if is_scheduled:
                                    try:
                                        from clippa.scheduler import PublishScheduler
                                        scheduler = PublishScheduler.get_instance()
                                        slot_dt, publish_at_iso = scheduler.get_next_available_slot()
                                        logger.info(
                                            f"[CLIPPA] Anti-shadowban slot assigned: {slot_dt.strftime('%Y-%m-%d %I:%M %p')} ({publish_at_iso} UTC)"
                                        )
                                    except Exception as sched_err:
                                        logger.warning(f"[CLIPPA] Scheduler slot assignment failed: {sched_err}, falling back to unlisted")
                                        upload_privacy = "unlisted"

                                log_act = f"Scheduling '{clip_output_file.name}' for {publish_at_iso}" if publish_at_iso else f"Auto-uploading '{clip_output_file.name}'"
                                logger.info(f"[CLIPPA] {log_act} on YouTube...")

                                up_res = upload_video_to_youtube(
                                    video_path=rendered_path,
                                    title=yt_title,
                                    description=yt_desc,
                                    tags=hashtags,
                                    privacy_status=upload_privacy,
                                    publish_at=publish_at_iso,
                                )

                                vid_id = up_res.get("video_id")
                                yt_url = up_res.get("url")

                                if is_scheduled and slot_dt and vid_id:
                                    try:
                                        from clippa.scheduler import PublishScheduler
                                        PublishScheduler.get_instance().record_scheduled_slot(
                                            slot_dt=slot_dt,
                                            video_id=vid_id,
                                            clip_name=clip_output_file.name,
                                            title=yt_title,
                                            youtube_url=yt_url or "",
                                        )
                                    except Exception as rec_err:
                                        logger.warning(f"[CLIPPA] Failed recording scheduled slot: {rec_err}")

                                meta_payload["youtube"] = {
                                    "uploaded": True,
                                    "scheduled": bool(publish_at_iso),
                                    "video_id": vid_id,
                                    "url": yt_url,
                                    "privacy": up_res.get("privacy", upload_privacy),
                                    "publish_at": publish_at_iso,
                                    "publish_at_local": slot_dt.strftime("%Y-%m-%d %I:%M %p") if slot_dt else None,
                                    "uploaded_at": up_res.get("published_at"),
                                }
                                meta_file.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
                                logger.info(f"[CLIPPA] YouTube upload complete: {yt_url} (scheduled={bool(publish_at_iso)})")
                            else:
                                logger.warning("[CLIPPA] Auto-upload requested, but YouTube account is not connected.")
                        except Exception as up_err:
                            logger.error(f"[CLIPPA] Auto-upload failed: {up_err}")
                            meta_payload["youtube"]["error"] = str(up_err)
                            try:
                                meta_file.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
                            except Exception:
                                pass
                except Exception as meta_err:
                    logger.warning(f"[CLIPPA] Failed writing clip metadata: {meta_err}")
            else:
                logger.error(f"[CLIPPA] Rendered clip file missing: {rendered_path}")
        except Exception as e:
            logger.error(f"[CLIPPA] Failed rendering clip {idx}: {e}")

    if not rendered_clips:
        logger.error("[CLIPPA] Failed to render any vertical clips.")
        return 1

    logger.info(f"[CLIPPA] Successfully rendered {len(rendered_clips)} clips in {out_path.resolve()}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI main entry point. Returns exit code: 0 on success, 1 on runtime failure."""
    args = parse_cli_args(argv)

    if getattr(args, "web", False):
        import uvicorn
        from clippa.web import create_web_app
        app = create_web_app(output_dir=Path(args.output_dir), temp_dir=Path(args.temp_dir))
        port = getattr(args, "port", 8765)
        print(f"\n🚀 CLIPPA Web Dashboard running at: http://127.0.0.1:{port}")
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
        return 0
    return run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
