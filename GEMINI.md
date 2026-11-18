# CLIPPA Project Guidelines & Environment Rules

## 1. Python Environment Invariant
- **Always use virtualenv Python**: Execute all python commands, tests, and scripts using `.venv\Scripts\python.exe` (or `.venv\Scripts\pythonw.exe` for silent background workers). Never invoke bare `python` or system Python, as project dependencies (`cv2`, `torch`, `faster_whisper`, `fastapi`) reside exclusively in `.venv`.

## 2. Windows & OneDrive File-Locking Guardrail
- The repository is located inside a Microsoft OneDrive synchronized folder (`...\OneDrive\Documents\...`).
- **File Lock Sensitivity**: Windows Search Indexer and OneDrive background synchronization immediately acquire read handles on any newly written or modified `.mp4`, `.json`, or `.tmp` files.
- **Rules for File I/O**:
  1. **Never overwrite in-place unconditionally**: Before initiating video downloads with `yt-dlp`, check if a valid file matching `*[{video_id}].*` already exists in `temp/` or `outputs/` (>100KB). Reuse existing files to avoid download locks and save bandwidth.
  2. **Atomic File Replace with Retry**: Whenever replacing temporary files with final outputs (`tmp.replace(dest)`), always wrap in a retry loop (3–5 attempts, 200–500ms backoff) and provide a fallback copy (`shutil.copy2`) if the lock persists.
  3. **Sequential Pipeline Serialization**: Never spawn concurrent pipeline processes on the same files. All clipping and upload tasks must be dispatched through `ClippingQueueManager` so they process sequentially.

## 3. Local Web Server Invariant
- **Localhost Execution**: CLIPPA runs as a local web dashboard and automation engine hosted on `http://127.0.0.1:8765`. Launch the server using `.venv\Scripts\python.exe run_server.py` or `.venv\Scripts\python.exe -m clippa.web --port 8765`. No desktop window wrapper (`pywebview` or browser subprocess) is needed; the user accesses it via their local browser.

## 4. UI Dashboard & Video Display Standards
- **Standalone Dashboard**: Maintain web UI markup and logic inside dedicated files (`clippa/dashboard.html`), not embedded inside Python multiline string literals where character escapes can cause JavaScript syntax corruption.
- **JavaScript Syntax Validation**: Always run `node --check` on any modified script in `dashboard.html` to guarantee zero syntax errors before testing.
- **Full Video Containment**: Video players for vertical 9:16 Shorts must use `object-fit: contain; background: #000;` and contain height constraints (e.g., `max-height: 380px` for cards, `max-height: 90vh` for theater modals) to ensure subtitles, speaker faces, and letterboxes are never cropped, and cards fit on standard laptop viewports.
- **Safe Click Handlers**: Never concatenate unescaped dynamic strings into inline HTML `onclick="..."` attributes. Always cache objects in a lookup table (e.g., `window.CURRENT_CHANNEL_VIDEOS[id]`) and pass alphanumeric IDs.
