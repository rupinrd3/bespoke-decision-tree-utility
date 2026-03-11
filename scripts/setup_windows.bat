@echo off
REM Bespoke Decision Tree Utility - Windows Setup Script
REM Pure batch file setup - no PowerShell required

echo ========================================
echo  Bespoke Decision Tree Utility
echo  Windows Setup Script
echo ========================================
echo.
echo This script will set up the Decision Tree Utility on your Windows system.
echo It will create a virtual environment and install dependencies.
echo.
echo Requirements:
echo - Python 3.12+ installed
echo - Internet connection for package downloads
echo.

REM Change to parent directory
cd /d "%~dp0\.."

REM Check if Python is available
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
    echo.
    echo SOLUTIONS:
    echo 1. Install Python 3.12+ from https://www.python.org/downloads/
    echo 2. Make sure Python is added to your PATH during installation
    echo 3. Or use portable version: See SETUP_WINDOWS.md Option A
    echo.
    pause
    exit /b 1
)

REM Check Python version
echo [OK] Python found. Checking version...
python -c "import sys; exit(0 if sys.version_info >= (3,12) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.12+ required
    echo Your Python version:
    python --version
    echo Please upgrade Python to 3.12 or higher
    echo.
    pause
    exit /b 1
)

echo [OK] Python version check passed.

REM Create virtual environment
echo.
echo [2/6] Creating virtual environment...
if exist "venv" (
    echo [INFO] Virtual environment already exists, skipping creation
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        echo Make sure Python has venv module installed
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

REM Activate virtual environment
echo.
echo [3/6] Activating virtual environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
) else (
    echo [ERROR] Virtual environment activation script not found
    pause
    exit /b 1
)

REM Upgrade pip
echo.
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARNING] Pip upgrade failed, continuing anyway...
)

REM Install requirements
echo.
echo [5/6] Installing dependencies...
if exist "requirements.txt" (
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        echo Please check your internet connection and try again
        echo.
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed successfully
) else (
    echo [ERROR] requirements.txt not found
    pause
    exit /b 1
)

REM Test installation
echo.
echo [6/6] Testing installation...
python -c "import PyQt5; import pandas; import numpy; print('[OK] Core dependencies verified')" 2>nul
if errorlevel 1 (
    echo [WARNING] Some dependencies may not be properly installed
    echo Try running the application to see if it works
) else (
    echo [OK] Installation test passed
)

REM Create desktop shortcut (simple approach)
echo.
echo Creating desktop shortcut...
set DESKTOP=%USERPROFILE%\Desktop
set CURRENT_DIR=%CD%
echo @echo off > "%DESKTOP%\Decision Tree Utility.bat"
echo cd /d "%CURRENT_DIR%" >> "%DESKTOP%\Decision Tree Utility.bat"
echo scripts\run_windows.bat >> "%DESKTOP%\Decision Tree Utility.bat"
echo [OK] Desktop shortcut created: Decision Tree Utility.bat

echo.
echo ========================================
echo  [SUCCESS] Setup Complete!
echo ========================================
echo.
echo To run the application:
echo 1. Run: scripts\run_portable.bat
echo 2. Or manually: activate venv and run python main.py
echo.
echo The virtual environment has been created and all dependencies installed.
echo.

pause