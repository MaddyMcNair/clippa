@echo off
title CLIPPA Local Server
cd /d "%~dp0"
echo =======================================================
echo   CLIPPA Local Web Dashboard & YouTube Automation
echo   Opening http://127.0.0.1:8765 in your browser...
echo =======================================================
start "" "http://127.0.0.1:8765"
.venv\Scripts\python.exe run_server.py
pause
