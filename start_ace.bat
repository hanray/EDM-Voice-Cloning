@echo off
setlocal
rem ACE-Step 1.5 sidecar API on http://127.0.0.1:8001
rem First run downloads model checkpoints from Hugging Face (several GB).
cd /d "%~dp0ACE-Step-1.5" || (
    echo [ERROR] ACE-Step-1.5 folder not found next to this script.
    exit /b 1
)
py -3.11 -m uv run acestep-api --port 8001
endlocal
