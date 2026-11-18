"""
clippa.config - Self-Hosted Zero-Paid-API Configuration for CLIPPA.

Feature 20: Self-Hosted Zero-Paid-API Mode
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load workspace .env by default
load_dotenv()

_config_override: Optional[Config] = None


class Config:
    """CLIPPA configuration manager enforcing self-hosted, zero-paid-API operation."""

    def __init__(self) -> None:
        raw_self_host = os.getenv("SELF_HOST", "true").strip().lower()
        self.self_hosted: bool = raw_self_host in {"1", "true", "yes", "on"}
        self.self_host: bool = self.self_hosted

        # Zero-paid-API enforcement: cloud keys are empty/None in self-hosted mode
        if self.self_hosted:
            self.openai_api_key: Optional[str] = None
            self.assembly_ai_api_key: Optional[str] = None
            self.anthropic_api_key: Optional[str] = None
            self.google_api_key: Optional[str] = None
            self.openrouter_api_key: Optional[str] = None
            self.apify_api_token: Optional[str] = None
        else:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            self.assembly_ai_api_key = os.getenv("ASSEMBLY_AI_API_KEY")
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            self.google_api_key = os.getenv("GOOGLE_API_KEY")
            self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            self.apify_api_token = os.getenv("APIFY_API_TOKEN")

        # Quotas and duration limits
        self.quotas_enabled: bool = False if self.self_hosted else (os.getenv("QUOTAS_ENABLED", "false").lower() == "true")
        self.max_duration_seconds: int = 86400  # Unrestricted 24h limit in self-hosted mode

        # Resolutions
        self.allowed_resolutions: List[str] = ["1080p", "720p"]
        self.default_resolution: str = "1080p"

        # Directory defaults
        self.output_dir: Path = Path(os.getenv("OUTPUT_DIR", "outputs"))
        self.temp_dir: Path = Path(os.getenv("TEMP_DIR", "temp"))

        # Pipeline defaults
        self.transcription_provider: str = os.getenv("TRANSCRIPTION_PROVIDER", "whisper")
        self.whisper_model: str = os.getenv("WHISPER_MODEL", "base")
        self.llm_model: str = os.getenv("LLM", "ollama:llama3.2:3b")
        self.ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.template: str = os.getenv("TEMPLATE", "default")
        self.max_clips: int = int(os.getenv("MAX_CLIPS", "3"))
        self.device: str = os.getenv("DEVICE", "cuda")
        self.youtube_max_height: int = int(os.getenv("YOUTUBE_MAX_HEIGHT", "1080"))

    def is_quota_enforced(self) -> bool:
        """Return False in self-hosted mode; quotas are completely bypassed."""
        return False if self.self_hosted else self.quotas_enabled

    def get_allowed_resolutions(self) -> List[str]:
        """Return allowed export resolutions (1080p unlocked)."""
        return list(self.allowed_resolutions)


def get_config() -> Config:
    """Retrieve global Config singleton or override."""
    global _config_override
    if _config_override is not None:
        return _config_override
    return Config()


def load_config(env_path: Optional[str | Path] = None) -> Config:
    """Reload configuration from environment file."""
    if env_path is not None:
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=True)
    cfg = Config()
    return cfg


def set_config_override(config: Optional[Config]) -> None:
    """Set or reset configuration override for testing."""
    global _config_override
    _config_override = config


def is_self_hosted() -> bool:
    """Check if system is operating in self-hosted zero-paid-API mode."""
    return get_config().self_hosted
