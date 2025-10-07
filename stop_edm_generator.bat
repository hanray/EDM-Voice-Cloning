@echo off
echo.
echo ================================
echo   Stopping EDM Voice Generator
echo ================================
echo.

:: Kill any running Python processes on port 8005
echo Stopping server processes...

:: Find and kill processes using port 8005
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8005') do (
    echo Stopping process %%a...
    taskkill /f /pid %%a 2>nul
)

:: Also kill any Python processes with our server name
taskkill /f /im python.exe /fi "WINDOWTITLE eq EDM Voice Generator Server*" 2>nul

echo.
echo Server stopped!
echo.
pause