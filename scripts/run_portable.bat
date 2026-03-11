@echo off
REM Bespoke Decision Tree Utility - Portable Launcher
REM This script is for OPTION A users (Standard Corporate User)
REM Uses portable Python environment - no system Python installation required

echo ========================================
echo  Bespoke Decision Tree Utility
echo  Portable Launcher  
echo ========================================

REM Change to the parent directory (where main.py is located)
cd /d "%~dp0\.."

REM Check for portable Python environment
if exist "python-portable\python.exe" (
    echo [OK] Found portable Python environment
    set PYTHON_CMD=python-portable\python.exe
    goto :run_app
)

REM Check for WinPython portable installation  
if exist "WinPython\python-*\python.exe" (
    echo [OK] Found WinPython portable installation
    for /d %%d in (WinPython\python-*) do set PYTHON_CMD=%%d\python.exe
    goto :run_app
)

REM Check if system Python is available (fallback)
python --version >nul 2>&1
if not errorlevel 1 (
    echo [OK] Using system Python (fallback)
    set PYTHON_CMD=python
    goto :run_app
)

REM No Python found
echo [ERROR] No portable Python environment found!
echo.
echo Expected setup:
echo   python-portable\
echo   ├── python.exe
echo   ├── Lib\ 
echo   └── Scripts\
echo.
echo SOLUTIONS:
echo 1. Download DecisionTreeUtility-Python.zip from your IT department
echo 2. Extract to this folder (should create python-portable\ folder)
echo 3. Or install WinPython portable to WinPython\ folder
echo 4. For system Python setup: See SETUP_WINDOWS.md Option B
echo.
pause
exit /b 1

:run_app
REM Set environment variables for better Qt performance
set QT_AUTO_SCREEN_SCALE_FACTOR=1
set QT_ENABLE_HIGHDPI_SCALING=1

REM Launch the application  
echo Starting Decision Tree Utility...
echo Using: %PYTHON_CMD%
echo.

%PYTHON_CMD% main.py

REM Check if application exited with an error
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with an error.
    echo Check the logs\ directory for error details.
    echo.
    pause
) else (
    echo.
    echo [OK] Application closed successfully.
)

echo.
pause