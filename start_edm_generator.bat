@echo off
echo.
echo ================================
echo   EDM Neural Voice Generator
echo ================================
echo.

:: Change to the project directory
cd /d "e:\E\Creative Work\Backend Dev\EDM-Voice-Cloning-master\EDM-Voice-Cloning-master"

:: Check if virtual environment exists
if not exist "env\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please make sure you have created a virtual environment in the 'env' folder.
    pause
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call env\Scripts\activate.bat

:: Check if required packages are installed
echo Checking dependencies...
python -c "import fastapi, gradio_client, soundfile" 2>nul
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
)

:: Start the server
echo.
echo Starting EDM Neural Voice Generator...
echo Server will be available at: http://localhost:8005
echo.
echo Press Ctrl+C to stop the server
echo.

:: Start the FastAPI server
python -m src.api.simple_server --port 8005

:: Keep the window open if there's an error
if errorlevel 1 (
    echo.
    echo ERROR: Failed to start the server!
    echo Check the error messages above.
    pause
)