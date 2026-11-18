"""
clippa/uploader.py - YouTube Data API v3 OAuth & Video Upload Module

Features:
- Local OAuth 2.0 authentication for Google accounts
- Secure token storage at ~/.clippa/youtube_credentials.json
- Resumable chunked video upload for YouTube Shorts
- Channel profile querying and disconnection
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger("clippa.uploader")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Config directory for local credentials
CONFIG_DIR = Path.home() / ".clippa"
TOKEN_FILE = CONFIG_DIR / "youtube_credentials.json"
CLIENT_SECRETS_FILE = CONFIG_DIR / "client_secrets.json"

YOUTUBE_UPLOAD_SCOPE = "https://www.googleapis.com/auth/youtube.upload"
YOUTUBE_READ_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"
SCOPES = [YOUTUBE_UPLOAD_SCOPE, YOUTUBE_READ_SCOPE]


class YouTubeAuthError(Exception):
    """Raised when YouTube authentication or token retrieval fails."""
    pass


class YouTubeUploadError(Exception):
    """Raised when video upload to YouTube Data API fails."""
    pass


def get_stored_credentials() -> Any:
    """
    Loads saved OAuth 2.0 user credentials from ~/.clippa/youtube_credentials.json.
    Automatically refreshes expired access tokens if a valid refresh token exists.
    """
    if not TOKEN_FILE.exists() or TOKEN_FILE.stat().st_size == 0:
        return None

    try:
        from google.oauth2.credentials import Credentials  # type: ignore
        from google.auth.transport.requests import Request  # type: ignore

        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
        return creds
    except Exception as exc:
        logger.warning("Failed to load or refresh stored credentials: %s", exc)
        return None


def is_authenticated() -> bool:
    """Returns True if a valid or refreshable YouTube OAuth token is available."""
    creds = get_stored_credentials()
    return creds is not None and (creds.valid or creds.refresh_token is not None)


def save_client_secrets(secrets_data: Union[str, dict, Path]) -> Path:
    """Saves user's client_secrets.json content into ~/.clippa/client_secrets.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if isinstance(secrets_data, Path) or (isinstance(secrets_data, str) and Path(secrets_data).is_file()):
        content = Path(secrets_data).read_text(encoding="utf-8")
    elif isinstance(secrets_data, dict):
        content = json.dumps(secrets_data, indent=2)
    else:
        content = str(secrets_data)

    # Validate JSON structure
    parsed = json.loads(content)
    if "installed" not in parsed and "web" not in parsed:
        raise ValueError("Invalid client_secrets.json format. Expected 'installed' or 'web' root key.")

    CLIENT_SECRETS_FILE.write_text(content, encoding="utf-8")
    return CLIENT_SECRETS_FILE


def authenticate(
    client_secrets_path: Optional[Union[str, Path]] = None,
    open_browser: bool = True,
) -> Dict[str, Any]:
    """
    Initiates Google OAuth 2.0 Desktop App Flow.
    Opens user's default browser for consent and saves tokens locally.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore

    secrets_path = Path(client_secrets_path) if client_secrets_path else CLIENT_SECRETS_FILE
    if not secrets_path.exists():
        raise FileNotFoundError(
            f"Google client_secrets.json not found at {secrets_path}. "
            "Please upload or configure your Google Cloud OAuth Client ID."
        )

    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), SCOPES)
    creds = flow.run_local_server(
        port=0,
        authorization_prompt_message="Please visit this URL to authorize CLIPPA: {url}",
        success_message="CLIPPA successfully authorized! You may now close this window.",
        open_browser=open_browser,
    )

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    logger.info("Saved YouTube credentials to %s", TOKEN_FILE)

    return get_channel_info(creds)


def disconnect_account() -> bool:
    """Revokes and deletes saved local YouTube credentials."""
    if TOKEN_FILE.exists():
        try:
            TOKEN_FILE.unlink()
            logger.info("Removed stored YouTube credentials at %s", TOKEN_FILE)
            return True
        except Exception as e:
            logger.error("Failed to delete credentials file: %s", e)
            return False
    return True


def get_channel_info(creds: Any = None) -> Dict[str, Any]:
    """Fetches channel metadata for the connected YouTube account."""
    if creds is None:
        creds = get_stored_credentials()

    if creds is None:
        return {"connected": False, "message": "No account connected"}

    try:
        from googleapiclient.discovery import build  # type: ignore

        youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
        resp = youtube.channels().list(mine=True, part="snippet,statistics").execute()

        items = resp.get("items", [])
        if not items:
            return {"connected": True, "title": "Connected Account", "subscriber_count": 0}

        channel = items[0]
        snippet = channel.get("snippet", {})
        stats = channel.get("statistics", {})

        return {
            "connected": True,
            "channel_id": channel.get("id"),
            "title": snippet.get("title", "My YouTube Channel"),
            "description": snippet.get("description", ""),
            "custom_url": snippet.get("customUrl", ""),
            "thumbnail": snippet.get("thumbnails", {}).get("default", {}).get("url", ""),
            "subscriber_count": stats.get("subscriberCount", 0),
            "video_count": stats.get("videoCount", 0),
        }
    except Exception as exc:
        logger.warning("Failed to fetch YouTube channel info: %s", exc)
        return {"connected": True, "title": "Authorized Account", "error": str(exc)}


def upload_video_to_youtube(
    video_path: Union[str, Path],
    title: str,
    description: str,
    tags: Optional[List[str]] = None,
    privacy_status: str = "public",
    publish_at: Optional[str] = None,
    category_id: str = "22",  # 22 = People & Blogs
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """
    Uploads a vertical MP4 video to YouTube as a Short using resumable upload.
    If publish_at is provided (ISO 8601 string, e.g. '2026-11-15T08:30:00Z'),
    privacy_status is set to 'private' and YouTube schedules the video release.
    """
    video_p = Path(video_path)
    if not video_p.is_file():
        raise FileNotFoundError(f"Video file to upload does not exist: {video_p}")

    creds = get_stored_credentials()
    if creds is None:
        raise YouTubeAuthError("Not authenticated with YouTube. Please connect your account first.")

    from googleapiclient.discovery import build  # type: ignore
    from googleapiclient.http import MediaFileUpload  # type: ignore

    clean_tags = [t.strip().lstrip("#") for t in (tags or ["Shorts", "Viral"])]
    if "Shorts" not in clean_tags and "shorts" not in clean_tags:
        clean_tags.insert(0, "Shorts")

    # If scheduling is requested, privacyStatus MUST be private per YouTube API specs
    final_privacy = "private" if publish_at else privacy_status.lower()

    status_dict: Dict[str, Any] = {
        "privacyStatus": final_privacy,
        "selfDeclaredMadeForKids": False,
    }
    if publish_at:
        status_dict["publishAt"] = publish_at

    body = {
        "snippet": {
            "title": title[:100],
            "description": description,
            "tags": clean_tags[:20],
            "categoryId": str(category_id),
        },
        "status": status_dict,
    }

    youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
    media = MediaFileUpload(
        str(video_p),
        mimetype="video/mp4",
        chunksize=1024 * 1024 * 4,  # 4MB chunks
        resumable=True,
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    log_schedule = f" (scheduled for {publish_at})" if publish_at else f" (privacy={final_privacy})"
    logger.info("Starting YouTube upload for '%s'%s...", video_p.name, log_schedule)
    response = None
    try:
        while response is None:
            status, response = request.next_chunk()
            if status and progress_callback:
                progress_pct = float(status.progress()) * 100.0
                progress_callback(progress_pct)
    except Exception as exc:
        err_msg = str(exc)
        if "uploadLimitExceeded" in err_msg:
            raise YouTubeUploadError(
                "YouTube Daily Upload Limit Exceeded: YouTube has capped daily uploads on this channel (~10/day). "
                "Verify your channel with phone verification at https://www.youtube.com/verify or wait 24 hours."
            ) from exc
        raise YouTubeUploadError(f"YouTube upload failed: {exc}") from exc

    video_id = response.get("id")
    if not video_id:
        raise YouTubeUploadError(f"YouTube upload succeeded but no video ID was returned: {response}")

    shorts_url = f"https://youtube.com/shorts/{video_id}"
    logger.info("Successfully uploaded video to YouTube: %s", shorts_url)

    return {
        "video_id": video_id,
        "url": shorts_url,
        "title": title,
        "privacy": final_privacy,
        "published_at": response.get("snippet", {}).get("publishedAt"),
        "scheduled_for": publish_at,
    }
