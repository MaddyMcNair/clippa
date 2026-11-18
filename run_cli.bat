@echo off
cd /d "%~dp0"
echo ========================================================
echo   CLIPPA - Local-First Automated Video Clipping Pipeline
echo ========================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .venv.
    pause
    exit /b 1
)

set /p VIDEO_URL="Enter YouTube URL or local video file path: "
if "%VIDEO_URL%"=="" (
    echo [ERROR] No URL provided.
    pause
    exit /b 1
)

echo.
echo Select Subtitle Style Template:
echo   1. Default (White with Yellow highlight)
echo   2. MrBeast (Bold Uppercase with Yellow/Red pop)
echo   3. Hormozi (Bold Uppercase with Neon Green pop)
echo   4. TikTok  (TikTokSans typography)
set /p STYLE_CHOICE="Enter choice [1-4, default=1]: "

set TEMPLATE=default
if "%STYLE_CHOICE%"=="2" set TEMPLATE=mrbeast
if "%STYLE_CHOICE%"=="3" set TEMPLATE=hormozi
if "%STYLE_CHOICE%"=="4" set TEMPLATE=tiktok

echo.
echo Running CLIPPA pipeline with template '%TEMPLATE%'...
echo.

.\.venv\Scripts\python.exe -m clippa --url "%VIDEO_URL%" --template %TEMPLATE% --max-clips 3

echo.
echo ========================================================
echo   Processing complete! Clips saved in 'outputs' folder.
echo ========================================================
pause
