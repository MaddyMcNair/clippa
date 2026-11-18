@echo off
cd /d "%~dp0"
echo ========================================================
echo   Starting CLIPPA Local Web Dashboard ^& Video Previewer
echo ========================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .venv.
    echo Please run setup first.
    pause
    exit /b 1
)

echo Opening browser at http://127.0.0.1:8000...
start "" http://127.0.0.1:8000
.\.venv\Scripts\python.exe -m clippa --web --port 8000
pause
