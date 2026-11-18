"""
clippa/web.py - Local Web & Desktop UI Dashboard for CLIPPA

Features:
- Windows Desktop UI with dark-mode styling
- YouTuber Video Browser: View any YouTuber's latest videos directly in the app
- 1-Click "Clip This Video" & Prioritized Sequential Queue ("clip this first, then this, then this")
- Re-orderable Queue (Move Up/Down) with automatic background execution
- Automatic YouTube Shorts Publishing for all queued and clipped videos
- 100% Hands-Free 24/7 Channel Autopilot
- Virality Confidence scoring & metric breakdown display
- Intelligent hashtag & SEO generator preview
- Local video player for 9:16 vertical clips
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from clippa.analysis import sanitize_hook_title
from clippa.cli import run_pipeline
from clippa.queue_manager import ClippingQueueManager, QueueItem
from clippa.uploader import (
    authenticate,
    disconnect_account,
    get_channel_info,
    is_authenticated,
    save_client_secrets,
    upload_video_to_youtube,
)
from clippa.watcher import (
    AutopilotDaemon,
    add_channel_to_watchlist,
    fetch_recent_channel_videos,
    get_autopilot_settings,
    get_history,
    get_watchlist,
    remove_channel_from_watchlist,
    save_autopilot_settings,
)

logger = logging.getLogger("clippa.web")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# In-memory jobs store for direct jobs
JOBS: Dict[str, Dict[str, Any]] = {}


class GenerateRequest(BaseModel):
    url: str
    template: str = "default"
    max_clips: int = 3
    max_duration: int = 120
    resolution: str = "1080p"
    llm_model: str = "ollama:llama3.2:3b"
    enable_emojis: bool = False
    nvenc: bool = True
    auto_upload: bool = True
    upload_privacy: str = "scheduled"


class YouTubeUploadRequest(BaseModel):
    filename: str
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    privacy: str = "scheduled"


class ScheduleSettingsRequest(BaseModel):
    max_daily_posts: Optional[int] = None
    slot_times: Optional[List[str]] = None


class SecretsConnectRequest(BaseModel):
    secrets_content: Optional[str] = None
    secrets_path: Optional[str] = None


class AddChannelRequest(BaseModel):
    channel: str


class AutopilotToggleRequest(BaseModel):
    enabled: bool
    poll_interval_seconds: Optional[int] = None
    auto_upload: Optional[bool] = None
    upload_privacy: Optional[str] = None


class AddQueueRequest(BaseModel):
    video_id: str
    title: str
    url: str
    channel: str = ""
    thumbnail: str = ""
    duration: int = 0
    template: str = "default"
    max_clips: int = 3
    max_duration: int = 120
    auto_upload: bool = True
    upload_privacy: str = "scheduled"
    nvenc: bool = True
    prioritize_first: bool = False


class ReorderQueueRequest(BaseModel):
    item_id: str
    direction: str  # "up" or "down"


def create_web_app(output_dir: Path = Path("outputs"), temp_dir: Path = Path("temp")) -> FastAPI:
    """Builds and returns the FastAPI application."""
    output_dir = Path(output_dir).resolve()
    temp_dir = Path(temp_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Initialize and start background Autopilot Daemon & Queue Manager
    daemon = AutopilotDaemon.get_instance(output_dir=output_dir, temp_dir=temp_dir)
    daemon.start()

    queue_manager = ClippingQueueManager.get_instance(output_dir=output_dir, temp_dir=temp_dir)
    queue_manager.start_worker()

    app = FastAPI(title="CLIPPA Desktop Dashboard", version="1.3.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount outputs for HTML5 video playback
    app.mount("/videos", StaticFiles(directory=str(output_dir)), name="videos")

    @app.get("/api/clips")
    async def list_clips():
        """Lists all rendered clips with their virality metadata and YouTube status."""
        clips = []
        for p in sorted(output_dir.glob("*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True):
            stat = p.stat()
            clean_title = p.stem.replace("_", " ")

            meta_file = p.with_name(f"{p.stem}.meta.json")
            meta_data: Dict[str, Any] = {}
            if meta_file.exists():
                try:
                    meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            viral_score = meta_data.get("viral_score", 75)
            subscores = meta_data.get("subscores", {"hook": 18, "engagement": 19, "value": 18, "shareability": 20})
            reasoning = meta_data.get("reasoning", "High audience retention and engagement potential.")
            hashtags = meta_data.get("hashtags", ["#Shorts", "#Viral", "#Trending", "#FYP"])
            youtube_meta = meta_data.get("youtube", {"uploaded": False, "video_id": None, "url": None, "privacy": "public"})

            raw_title = meta_data.get("title") or clean_title
            display_title = sanitize_hook_title(raw_title)
            yt_title = meta_data.get("youtube_title") or f"{display_title} #Shorts"

            clips.append({
                "filename": p.name,
                "title": display_title,
                "youtube_title": yt_title,
                "size_mb": round(stat.st_size / (1024 * 1024), 1),
                "modified": stat.st_mtime,
                "url": f"/videos/{p.name}",
                "viral_score": viral_score,
                "subscores": subscores,
                "reasoning": reasoning,
                "hashtags": hashtags,
                "description": meta_data.get("description", ""),
                "youtube": youtube_meta,
            })
        return JSONResponse({"clips": clips})

    @app.delete("/api/clips/{filename}")
    async def delete_clip(filename: str):
        """Deletes a rendered clip and its companion metadata from outputs."""
        clean_name = Path(filename).name
        target = (output_dir / clean_name).resolve()
        if not str(target).startswith(str(output_dir)):
            raise HTTPException(status_code=400, detail="Invalid filename path.")
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Clip file not found.")
        try:
            target.unlink()
            meta_file = target.with_name(f"{target.stem}.meta.json")
            if meta_file.exists():
                meta_file.unlink()
            return JSONResponse({"status": "deleted", "filename": clean_name})
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to delete clip: {exc}")

    @app.post("/api/open-outputs")
    async def open_outputs_folder():
        """Opens the outputs directory in Windows File Explorer."""
        try:
            if os.name == "nt":
                os.startfile(str(output_dir))
            else:
                subprocess.Popen(["xdg-open", str(output_dir)])
            return JSONResponse({"status": "opened"})
        except Exception as exc:
            return JSONResponse({"status": "error", "message": str(exc)}, status_code=500)

    # -------------------------------------------------------------------------
    # YouTube Integration Endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/youtube/status")
    async def youtube_status():
        """Returns the connected YouTube channel status."""
        info = get_channel_info()
        return JSONResponse(info)

    @app.post("/api/youtube/connect")
    async def youtube_connect(req: SecretsConnectRequest):
        """Saves Google OAuth client_secrets.json and starts authorization."""
        try:
            if req.secrets_content:
                save_client_secrets(req.secrets_content)
            elif req.secrets_path:
                save_client_secrets(Path(req.secrets_path))

            info = authenticate(open_browser=True)
            return JSONResponse({"status": "connected", "channel": info})
        except Exception as exc:
            logger.exception("YouTube authentication failed: %s", exc)
            return JSONResponse({"status": "error", "message": str(exc)}, status_code=400)

    @app.post("/api/youtube/disconnect")
    async def youtube_disconnect():
        """Revokes and clears connected YouTube account."""
        ok = disconnect_account()
        return JSONResponse({"status": "disconnected" if ok else "failed"})

    @app.post("/api/youtube/upload")
    async def youtube_upload(req: YouTubeUploadRequest):
        """Uploads a specific clip to the connected YouTube channel."""
        if not is_authenticated():
            raise HTTPException(status_code=401, detail="YouTube account is not connected. Please connect your account first.")

        clean_name = Path(req.filename).name
        target = (output_dir / clean_name).resolve()
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Clip file not found.")

        meta_file = target.with_name(f"{target.stem}.meta.json")
        meta_data: Dict[str, Any] = {}
        if meta_file.exists():
            try:
                meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        title = req.title or meta_data.get("youtube_title") or f"{target.stem.replace('_', ' ')} #Shorts"
        description = req.description or meta_data.get("description") or f"{title}\n\n#Shorts #Viral #Trending"
        tags = req.tags or meta_data.get("hashtags") or ["#Shorts", "#Viral"]

        is_sched = (req.privacy.lower() == "scheduled")
        publish_at_iso = None
        slot_dt = None

        if is_sched:
            try:
                from clippa.scheduler import PublishScheduler
                slot_dt, publish_at_iso = PublishScheduler.get_instance().get_next_available_slot()
            except Exception as sched_err:
                logger.warning("Scheduler error: %s, falling back to unlisted", sched_err)

        try:
            result = upload_video_to_youtube(
                video_path=target,
                title=title,
                description=description,
                tags=tags,
                privacy_status="private" if publish_at_iso else req.privacy,
                publish_at=publish_at_iso,
            )

            vid_id = result.get("video_id")
            if is_sched and slot_dt and vid_id:
                try:
                    from clippa.scheduler import PublishScheduler
                    PublishScheduler.get_instance().record_scheduled_slot(
                        slot_dt=slot_dt,
                        video_id=vid_id,
                        clip_name=target.name,
                        title=title,
                        youtube_url=result.get("url", ""),
                    )
                except Exception as rec_err:
                    logger.warning("Failed recording scheduled slot: %s", rec_err)

            meta_data["youtube"] = {
                "uploaded": True,
                "scheduled": bool(publish_at_iso),
                "video_id": vid_id,
                "url": result.get("url"),
                "privacy": result.get("privacy", req.privacy),
                "publish_at": publish_at_iso,
                "publish_at_local": slot_dt.strftime("%Y-%m-%d %I:%M %p") if slot_dt else None,
                "uploaded_at": result.get("published_at"),
            }
            meta_file.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")
            return JSONResponse({"status": "uploaded", "youtube": result, "scheduled": bool(publish_at_iso)})
        except Exception as exc:
            logger.exception("YouTube upload failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"YouTube upload failed: {exc}")

    @app.post("/api/youtube/upload-all")
    async def youtube_upload_all(privacy: str = "scheduled"):
        """Uploads all unuploaded clips in outputs to YouTube using scheduled anti-shadowban spacing."""
        if not is_authenticated():
            raise HTTPException(status_code=401, detail="YouTube account is not connected. Please connect your account first.")

        uploaded_results = []
        failed_results = []

        for p in sorted(output_dir.glob("*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True):
            meta_file = p.with_name(f"{p.stem}.meta.json")
            meta_data: Dict[str, Any] = {}
            if meta_file.exists():
                try:
                    meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            yt_meta = meta_data.get("youtube", {})
            if yt_meta.get("uploaded"):
                continue

            raw_title = meta_data.get("title") or p.stem.replace("_", " ")
            title = meta_data.get("youtube_title") or f"{sanitize_hook_title(raw_title)} #Shorts"
            description = meta_data.get("description") or f"{title}\n\n#Shorts #Viral #Trending"
            tags = meta_data.get("hashtags") or ["#Shorts", "#Viral"]

            is_sched = (privacy.lower() == "scheduled")
            slot_dt = None
            publish_at_iso = None

            if is_sched:
                try:
                    from clippa.scheduler import PublishScheduler
                    slot_dt, publish_at_iso = PublishScheduler.get_instance().get_next_available_slot()
                except Exception as sched_err:
                    logger.warning("Scheduler error: %s", sched_err)

            try:
                result = upload_video_to_youtube(
                    video_path=p,
                    title=title,
                    description=description,
                    tags=tags,
                    privacy_status="private" if publish_at_iso else privacy,
                    publish_at=publish_at_iso,
                )

                vid_id = result.get("video_id")
                if is_sched and slot_dt and vid_id:
                    try:
                        from clippa.scheduler import PublishScheduler
                        PublishScheduler.get_instance().record_scheduled_slot(
                            slot_dt=slot_dt,
                            video_id=vid_id,
                            clip_name=p.name,
                            title=title,
                            youtube_url=result.get("url", ""),
                        )
                    except Exception:
                        pass

                meta_data["youtube"] = {
                    "uploaded": True,
                    "scheduled": bool(publish_at_iso),
                    "video_id": vid_id,
                    "url": result.get("url"),
                    "privacy": result.get("privacy", privacy),
                    "publish_at": publish_at_iso,
                    "publish_at_local": slot_dt.strftime("%Y-%m-%d %I:%M %p") if slot_dt else None,
                    "uploaded_at": result.get("published_at"),
                }
                meta_file.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")
                uploaded_results.append({"filename": p.name, "title": title, "result": result, "scheduled": bool(publish_at_iso)})
            except Exception as exc:
                logger.exception("Failed uploading clip %s in upload-all: %s", p.name, exc)
                failed_results.append({"filename": p.name, "title": title, "error": str(exc)})

        return JSONResponse({
            "status": "completed",
            "total_processed": len(uploaded_results) + len(failed_results),
            "uploaded": uploaded_results,
            "failed": failed_results,
        })

    # -------------------------------------------------------------------------
    # Channel Observer & Video Browsing Endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/channels")
    async def list_watched_channels():
        """Lists all YouTube channels currently in the watchlist."""
        return JSONResponse({"channels": get_watchlist()})

    @app.post("/api/channels")
    async def add_watched_channel(req: AddChannelRequest):
        """Resolves and adds a YouTube channel to the watchlist."""
        raw = req.channel.strip()
        if not raw:
            raise HTTPException(status_code=400, detail="Channel handle or URL is required.")
        try:
            info = add_channel_to_watchlist(raw)
            return JSONResponse({"status": "added", "channel": info})
        except Exception as exc:
            logger.exception("Failed adding channel %s: %s", raw, exc)
            raise HTTPException(status_code=400, detail=f"Failed to resolve channel: {exc}")

    @app.delete("/api/channels/{channel_id:path}")
    async def delete_watched_channel(channel_id: str):
        """Removes a channel from the watchlist by ID or handle."""
        removed = remove_channel_from_watchlist(channel_id)
        if not removed:
            raise HTTPException(status_code=404, detail="Channel not found in watchlist.")
        return JSONResponse({"status": "removed", "channel_id": channel_id})

    @app.get("/api/channels/{channel_id_or_handle:path}/videos")
    async def get_channel_videos(channel_id_or_handle: str, limit: int = 12):
        """Fetches recent videos from a YouTuber's channel for browsing & 1-click queuing."""
        try:
            vids = fetch_recent_channel_videos(channel_id_or_handle, limit=limit)
            return JSONResponse({"videos": vids})
        except Exception as exc:
            logger.exception("Failed fetching videos for channel '%s': %s", channel_id_or_handle, exc)
            return JSONResponse({"videos": [], "error": str(exc)}, status_code=400)

    # -------------------------------------------------------------------------
    # Queue Management Endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/queue")
    async def get_queue():
        """Returns the current prioritized video clipping & auto-upload queue state."""
        qm = ClippingQueueManager.get_instance(output_dir=output_dir, temp_dir=temp_dir)
        return JSONResponse(qm.get_state())

    @app.post("/api/queue")
    async def add_queue_item(req: AddQueueRequest):
        """Adds a video to the clipping queue."""
        qm = ClippingQueueManager.get_instance(output_dir=output_dir, temp_dir=temp_dir)
        item = qm.add_to_queue(
            video_id=req.video_id,
            title=req.title,
            url=req.url,
            channel=req.channel,
            thumbnail=req.thumbnail,
            duration=req.duration,
            template=req.template,
            max_clips=req.max_clips,
            max_duration=req.max_duration,
            auto_upload=req.auto_upload,
            upload_privacy=req.upload_privacy,
            nvenc=req.nvenc,
            prioritize_first=req.prioritize_first,
        )
        return JSONResponse({"status": "queued", "item": item.to_dict()})

    @app.post("/api/queue/reorder")
    async def reorder_queue(req: ReorderQueueRequest):
        """Moves an item up or down in the queue."""
        qm = ClippingQueueManager.get_instance(output_dir=output_dir, temp_dir=temp_dir)
        ok = qm.move_item(req.item_id, req.direction)
        return JSONResponse({"status": "reordered" if ok else "failed", "state": qm.get_state()})

    @app.delete("/api/queue/{item_id}")
    async def delete_queue_item(item_id: str):
        """Removes a video from the pending queue."""
        qm = ClippingQueueManager.get_instance(output_dir=output_dir, temp_dir=temp_dir)
        ok = qm.remove_from_queue(item_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Queue item not found or currently active.")
        return JSONResponse({"status": "removed", "item_id": item_id})

    @app.post("/api/queue/retry/{item_id}")
    async def retry_queue_item(item_id: str):
        """Moves a failed item from history back to the front of the queue."""
        qm = ClippingQueueManager.get_instance(output_dir=output_dir, temp_dir=temp_dir)
        item = qm.retry_item(item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Queue item not found in history.")
        return JSONResponse({"status": "retrying", "item": item.to_dict()})

    @app.post("/api/queue/clear")
    async def clear_queue():
        """Clears all pending items from queue."""
        qm = ClippingQueueManager.get_instance(output_dir=output_dir, temp_dir=temp_dir)
        count = qm.clear_pending()
        return JSONResponse({"status": "cleared", "cleared_count": count})

    # -------------------------------------------------------------------------
    # Anti-Shadowban YouTube Scheduling Endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/schedule")
    async def get_schedule():
        """Returns the upcoming calendar of scheduled YouTube releases."""
        from clippa.scheduler import PublishScheduler
        return JSONResponse(PublishScheduler.get_instance().get_schedule_overview())

    @app.post("/api/schedule/settings")
    async def update_schedule_settings(req: ScheduleSettingsRequest):
        """Updates daily limits and release slot times."""
        from clippa.scheduler import PublishScheduler
        updated = PublishScheduler.get_instance().update_settings(
            max_daily_posts=req.max_daily_posts,
            slot_times=req.slot_times,
        )
        return JSONResponse({"status": "updated", "settings": updated})

    @app.post("/api/schedule/clear")
    async def clear_schedule():
        """Clears all tracked schedule slots."""
        from clippa.scheduler import PublishScheduler
        count = PublishScheduler.get_instance().clear_schedule()
        return JSONResponse({"status": "cleared", "cleared_count": count})

    # -------------------------------------------------------------------------
    # Autopilot Control Endpoints
    # -------------------------------------------------------------------------

    @app.get("/api/autopilot/status")
    async def autopilot_status():
        """Returns daemon state, watched channels, and real-time activity log."""
        d = AutopilotDaemon.get_instance(output_dir=output_dir, temp_dir=temp_dir)
        status = d.get_status()
        status["activity"] = d.get_activity_feed()
        return JSONResponse(status)

    @app.post("/api/autopilot/toggle")
    async def autopilot_toggle(req: AutopilotToggleRequest):
        """Toggles autopilot ON/OFF or updates polling parameters."""
        updates: Dict[str, Any] = {"enabled": req.enabled}
        if req.poll_interval_seconds is not None:
            updates["poll_interval_seconds"] = req.poll_interval_seconds
        if req.auto_upload is not None:
            updates["auto_upload"] = req.auto_upload
        if req.upload_privacy is not None:
            updates["upload_privacy"] = req.upload_privacy

        settings = save_autopilot_settings(updates)
        d = AutopilotDaemon.get_instance(output_dir=output_dir, temp_dir=temp_dir)
        state_str = "ON (Observing Channels & Auto-Uploading)" if req.enabled else "OFF"
        d.log_activity(f"⚡ Autopilot toggled to {state_str}")
        return JSONResponse({"status": "updated", "settings": settings})

    @app.post("/api/autopilot/check-now")
    async def autopilot_check_now(background_tasks: BackgroundTasks):
        """Triggers an immediate polling check across all watched channels."""
        d = AutopilotDaemon.get_instance(output_dir=output_dir, temp_dir=temp_dir)
        background_tasks.add_task(d.poll_once)
        return JSONResponse({"status": "started", "message": "Channel check started in background."})

    @app.get("/api/autopilot/history")
    async def autopilot_history():
        """Returns the history ledger of processed and uploaded videos."""
        return JSONResponse(get_history())

    # -------------------------------------------------------------------------
    # Pipeline Execution Endpoints
    # -------------------------------------------------------------------------

    def _execute_pipeline_job(job_id: str, req: GenerateRequest):
        """Runs the pipeline in background thread."""
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["message"] = "Processing video with CLIPPA pipeline..."
        try:
            exit_code = run_pipeline(
                url=req.url,
                output_dir=output_dir,
                temp_dir=temp_dir,
                template=req.template,
                max_clips=req.max_clips,
                max_duration=req.max_duration,
                resolution=req.resolution,
                llm_model=req.llm_model,
                enable_emojis=req.enable_emojis,
                nvenc=req.nvenc,
                auto_upload=req.auto_upload,
                upload_privacy=req.upload_privacy,
            )
            if exit_code == 0:
                JOBS[job_id]["status"] = "completed"
                # Check actual upload results from output directory meta files
                recent_meta_files = sorted(output_dir.glob("*.meta.json"), key=lambda f: f.stat().st_mtime, reverse=True)[:req.max_clips]
                uploaded_count = 0
                scheduled_count = 0
                upload_error = None
                for mf in recent_meta_files:
                    try:
                        md = json.loads(mf.read_text(encoding="utf-8"))
                        yt = md.get("youtube", {})
                        if yt.get("uploaded"):
                            if yt.get("scheduled"):
                                scheduled_count += 1
                            else:
                                uploaded_count += 1
                        elif yt.get("error"):
                            upload_error = yt.get("error")
                    except Exception:
                        pass

                msg = "Successfully generated clips!"
                if req.auto_upload:
                    if scheduled_count > 0:
                        msg += f" {scheduled_count} Shorts scheduled on YouTube!"
                    elif uploaded_count > 0:
                        msg += f" {uploaded_count} Shorts uploaded to YouTube!"
                    elif upload_error:
                        if "uploadLimitExceeded" in upload_error:
                            msg += " ⚠️ YouTube upload limit reached (~10/24h). Verify your channel with phone at youtube.com/verify or wait 24h."
                        else:
                            msg += f" ⚠️ Upload issue: {upload_error[:60]}"
                    else:
                        msg += " Clips ready in gallery below (click 🚀 Upload when ready)."
                JOBS[job_id]["message"] = msg
            else:
                JOBS[job_id]["status"] = "failed"
                JOBS[job_id]["message"] = "Pipeline exited with an error. Check server logs."
        except Exception as exc:
            logger.exception("Job %s failed: %s", job_id, exc)
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["message"] = str(exc)

    @app.post("/api/generate")
    async def generate_clips(req: GenerateRequest, background_tasks: BackgroundTasks):
        """Submits a new clipping job."""
        if not req.url or not req.url.strip():
            raise HTTPException(status_code=400, detail="URL or video path is required.")

        job_id = str(uuid.uuid4())[:8]
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "message": "Job queued for processing...",
            "url": req.url,
            "template": req.template,
            "auto_upload": req.auto_upload,
        }
        thread = threading.Thread(target=_execute_pipeline_job, args=(job_id, req), daemon=True)
        thread.start()

        return JSONResponse({"job_id": job_id, "status": "queued"})

    @app.get("/api/jobs/{job_id}")
    async def get_job_status(job_id: str):
        """Checks the status of a clipping job."""
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="Job not found.")
        return JSONResponse(JOBS[job_id])

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Renders modern dark-themed web dashboard."""
        return get_dashboard_html()

    return app


DASHBOARD_FILE = Path(__file__).parent / "dashboard.html"


def get_dashboard_html() -> str:
    """Reads dashboard HTML directly from clippa/dashboard.html."""
    if DASHBOARD_FILE.exists():
        return DASHBOARD_FILE.read_text(encoding="utf-8")
    return HTML_DASHBOARD


HTML_DASHBOARD = DASHBOARD_FILE.read_text(encoding="utf-8") if DASHBOARD_FILE.exists() else "<html><body>CLIPPA Dashboard</body></html>"


def main():
    """Runs the FastAPI server locally."""
    import argparse
    parser = argparse.ArgumentParser(description="CLIPPA Local Web Dashboard")
    parser.add_argument("--port", type=int, default=8765, help="Web server port (default: 8765)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory path")
    args = parser.parse_args()

    app = create_web_app(output_dir=Path(args.output_dir))
    print(f"\n🚀 CLIPPA Web Dashboard running at: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
