"""
run_server.py - Launches CLIPPA Local Web Dashboard & Automation Engine

Starts the local FastAPI server with:
- Web dashboard accessible at http://127.0.0.1:8765
- Sequential auto-upload queue worker
- 24/7 channel autopilot observer
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Guarantee project root in sys.path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

if __name__ == "__main__":
    import uvicorn
    from clippa.web import create_web_app

    output_dir = PROJECT_ROOT / "outputs"
    temp_dir = PROJECT_ROOT / "temp"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    app = create_web_app(output_dir=output_dir, temp_dir=temp_dir)
    print("\n=======================================================")
    print("  ⚡ CLIPPA Local Web Dashboard & YouTube Automation")
    print("  🌐 http://127.0.0.1:8765")
    print("=======================================================\n")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
