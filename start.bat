@echo off
echo.
echo  ██████╗ ██████╗ ██╗███████╗██╗  ██╗████████╗██╗     █████╗ ██╗
echo  ██╔══██╗██╔══██╗██║██╔════╝██║  ██║╚══██╔══╝██║    ██╔══██╗██║
echo  ██║  ██║██████╔╝██║███████╗███████║   ██║   ██║    ███████║██║
echo  ██║  ██║██╔══██╗██║╚════██║██╔══██║   ██║   ██║    ██╔══██║██║
echo  ██████╔╝██║  ██║██║███████║██║  ██║   ██║   ██║    ██║  ██║██║
echo  ╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝   ╚═╝   ╚═╝    ╚═╝  ╚═╝╚═╝
echo.
echo  AI-Assisted Diabetic Retinopathy Screening System
echo  ================================================================
echo.

cd /d "%~dp0"

echo [1/3] Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
  echo ERROR: Dependency installation failed.
  pause
  exit /b 1
)

echo [2/3] Dependencies ready.
echo [3/3] Starting DRishti AI server...
echo.
echo  Open your browser at:  http://localhost:8000
echo  Press Ctrl+C to stop.
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
