@echo off
setlocal

rem Resolve paths; ROOT relative to script, UI_DIR fixed per user request
set "ROOT=%~dp0seed-vc"
set "UI_DIR=E:\E\Creative Work\Backend Dev\seed-vc-project\seed-vc\ui-client"

pushd "%ROOT%" || (
	echo [ERROR] Cannot cd into %ROOT%
	exit /b 1
)

rem Prefer local venv python; otherwise use system python
set "PY_EXE=python"
if exist "..\.venv310\Scripts\python.exe" set "PY_EXE=..\.venv310\Scripts\python.exe"

rem Fixed API port (override by setting API_PORT before running)
if "%API_PORT%"=="" set "API_PORT=7860"
set "VITE_API_TARGET=http://127.0.0.1:%API_PORT%"

rem Start UI in a new window (only if UI folder exists)
if exist "%UI_DIR%\package.json" (
	echo [INFO] UI_DIR resolved to: %UI_DIR%
	start "Seed-VC UI (Vite)" /D "%UI_DIR%" cmd /k "npm run dev"
) else (
	echo [WARN] ui-client folder not found at %UI_DIR%
)

echo Starting API server on %API_PORT% ...
"%PY_EXE%" api_server.py --host 0.0.0.0 --port %API_PORT%

popd
endlocal
