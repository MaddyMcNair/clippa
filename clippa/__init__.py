"""
clippa - Local Video Clipping Pipeline

Local-first processing adaptation based on FujiwaraChoki/supoclip.
"""

# Milestone 2: Transcription
from clippa.transcription import (
    Word,
    Utterance,
    TranscriptData,
    AudioExtractionError,
    TranscriptionError,
    extract_audio,
    transcribe_audio,
    get_cache_path,
    load_cached_transcript,
    save_cached_transcript,
)

# Milestone 2: Virality Analysis
from clippa.analysis import (
    ViralSegment,
    TeaserCandidate,
    extract_viral_segments,
    heuristic_score_segments,
    create_analysis_windows,
    deduplicate_segments,
    analyze_long_video_windowed,
    sanitize_hook_title,
    generate_hook_title,
    detect_cold_open_teaser,
    extract_heatmap_highlights,
    find_narrative_boundaries,
)

# Milestone 2: Face Tracking & Reframing
from clippa.reframing import (
    FaceBox,
    YuNetFaceDetector,
    load_face_detector,
    detect_faces,
    compute_face_centroid,
    smooth_face_trajectory,
    calculate_crop_coordinates,
    build_crop_filter,
    compute_ken_burns_crops,
    calculate_zoom_factor,
    build_zoompan_filter,
    round_to_even,
)

# Milestone 2: ASS Subtitles
from clippa.subtitles import (
    generate_ass_subtitles,
    get_available_templates,
    inject_emojis,
    ms_to_ass_time,
    sanitize_ass_text,
    wrap_hook_title,
    resolve_font_name,
)

# Milestone 3: Ingestion
from clippa.ingestion import (
    VideoMetadata,
    VideoDownloadError,
    InvalidURLError,
    VideoUnavailableError,
    VideoAgeRestrictedError,
    StreamDurationUnavailableError,
    VideoDurationExceededError,
    PreflightTimeoutError,
    download_video,
    get_video_metadata,
    extract_video_id,
    is_valid_youtube_url,
    canonical_youtube_url,
    clear_metadata_cache,
    validate_video_duration,
    is_local_video,
    resolve_local_video_path,
    load_video_heatmap,
    get_video_title,
)

# Milestone 3: Rendering
from clippa.rendering import (
    RenderingError,
    FFmpegExecutionError,
    SubtitleBurnError,
    MediaNotFoundError,
    find_ffmpeg,
    escape_filter_path,
    escape_path_for_ffmpeg,
    build_ffmpeg_render_args,
    render_clip,
)

# Milestone 3: Config
from clippa.config import (
    Config,
    get_config,
    load_config,
    set_config_override,
    is_self_hosted,
)

# Milestone 3: CLI
from clippa.cli import (
    build_parser,
    parse_cli_args,
    get_unique_clip_path,
    run_pipeline,
    main,
)

__all__ = [
    # Transcription
    "Word",
    "Utterance",
    "TranscriptData",
    "AudioExtractionError",
    "TranscriptionError",
    "extract_audio",
    "transcribe_audio",
    "get_cache_path",
    "load_cached_transcript",
    "save_cached_transcript",
    # Analysis
    "ViralSegment",
    "TeaserCandidate",
    "extract_viral_segments",
    "heuristic_score_segments",
    "create_analysis_windows",
    "deduplicate_segments",
    "analyze_long_video_windowed",
    "sanitize_hook_title",
    "generate_hook_title",
    "detect_cold_open_teaser",
    "extract_heatmap_highlights",
    "find_narrative_boundaries",
    # Reframing
    "FaceBox",
    "YuNetFaceDetector",
    "load_face_detector",
    "detect_faces",
    "compute_face_centroid",
    "smooth_face_trajectory",
    "calculate_crop_coordinates",
    "build_crop_filter",
    "compute_ken_burns_crops",
    "calculate_zoom_factor",
    "build_zoompan_filter",
    "round_to_even",
    # Subtitles
    "generate_ass_subtitles",
    "get_available_templates",
    "inject_emojis",
    "ms_to_ass_time",
    "sanitize_ass_text",
    "wrap_hook_title",
    "resolve_font_name",
    # Ingestion
    "VideoMetadata",
    "VideoDownloadError",
    "InvalidURLError",
    "VideoUnavailableError",
    "VideoAgeRestrictedError",
    "StreamDurationUnavailableError",
    "VideoDurationExceededError",
    "PreflightTimeoutError",
    "download_video",
    "get_video_metadata",
    "extract_video_id",
    "is_valid_youtube_url",
    "canonical_youtube_url",
    "clear_metadata_cache",
    "validate_video_duration",
    "is_local_video",
    "resolve_local_video_path",
    "load_video_heatmap",
    # Rendering
    "RenderingError",
    "FFmpegExecutionError",
    "SubtitleBurnError",
    "MediaNotFoundError",
    "find_ffmpeg",
    "escape_filter_path",
    "escape_path_for_ffmpeg",
    "build_ffmpeg_render_args",
    "render_clip",
    # Config
    "Config",
    "get_config",
    "load_config",
    "set_config_override",
    "is_self_hosted",
    # CLI
    "build_parser",
    "parse_cli_args",
    "get_unique_clip_path",
    "run_pipeline",
    "main",
]
