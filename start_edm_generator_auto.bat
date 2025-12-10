@echo off
echo.
echo ================================
echo   EDM Neural Voice Generator
echo      AUTO-START VERSION
echo ================================
echo.

:: Change to the project directory
cd /d "e:\Creative Work\Backend Dev\EDM-Vocal-Generator"

:: Check if virtual environment exists
if not exist "env\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Looking for alternative Python environments...
    
    :: Try env311 folder
    if exist "env311\Scripts\activate.bat" (
        echo Found env311 environment, using that instead...
        set ENV_PATH=env311
    ) else (
        echo No virtual environment found. Please create one first.
        pause
        exit /b 1
    )
) else (
    set ENV_PATH=env
)

:: Activate virtual environment
echo Activating virtual environment (%ENV_PATH%)...
call %ENV_PATH%\Scripts\activate.bat

:: Check if required packages are installed
echo Checking dependencies...
python -c "import fastapi, gradio_client, soundfile" 2>nul
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
)

:: Start the server in background and open browser
echo.
echo Starting EDM Neural Voice Generator...
echo Server starting at: http://localhost:8005
echo.

:: Start server in a new window
start "EDM Voice Generator Server" cmd /k "python -m src.api.simple_server --port 8005"

:: Wait a moment for server to start
echo Waiting for server to initialize...
timeout /t 3 /nobreak > nul

:: Open browser
echo Opening browser...
start http://localhost:8005

echo.
echo ================================
echo   🎵 EDM Voice Generator Ready!
echo ================================
echo.
echo The server is running in a separate window.
echo Your browser should open automatically.
echo.
echo To stop the server:
echo 1. Close the server window, or
echo 2. Press Ctrl+C in the server window
echo.
echo This window can be closed safely.
echo.
pause