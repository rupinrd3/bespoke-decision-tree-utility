@echo off
REM Bespoke Decision Tree Utility - Portable Environment Builder
REM For IT administrators to create portable Python environment for distribution

echo ========================================
echo  Bespoke Decision Tree Utility
echo  Portable Environment Builder
echo ========================================
echo.
echo This script creates a portable Python environment for
echo standard corporate users with no command line access.
echo.

REM Change to the parent directory (where main.py is located)
cd /d "%~dp0\.."

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo This build script requires Python 3.12+ with pip access
    echo Please install Python or use Anaconda Prompt
    echo.
    pause
    exit /b 1
)

REM Check Python version
echo Checking Python version...
python -c "import sys; exit(0 if sys.version_info >= (3,12) else 1)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.12+ required
    echo Your Python version:
    python --version
    echo.
    pause
    exit /b 1
)

echo [OK] Python version check passed
echo.

REM Clean up previous builds
if exist "python-portable" (
    echo Cleaning previous portable environment...
    rmdir /s /q "python-portable"
)

REM Create portable Python environment
echo Creating portable Python environment...
python -m venv python-portable
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo [OK] Virtual environment created
echo.

REM Activate and install dependencies
echo Installing dependencies...
call python-portable\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo Check your internet connection and requirements.txt
    pause
    exit /b 1
)

echo [OK] Dependencies installed
echo.

REM Test the portable environment
echo Testing portable environment...
python-portable\Scripts\python.exe -c "import PyQt5; import pandas; import numpy; print('[OK] Core dependencies verified')"

if errorlevel 1 (
    echo ERROR: Dependency test failed
    pause
    exit /b 1
)

echo [OK] Portable environment test passed
echo.

REM Create distribution packages
echo Creating distribution packages...

REM Create application ZIP (exclude portable Python)
PowerShell -Command "Compress-Archive -Path '.' -DestinationPath 'DecisionTreeUtility-App.zip' -Exclude 'python-portable','*.git*','__pycache__','*.zip' -Force"

REM Create portable Python ZIP  
PowerShell -Command "Compress-Archive -Path 'python-portable' -DestinationPath 'DecisionTreeUtility-Python.zip' -Force"

REM Calculate sizes
for %%i in ("DecisionTreeUtility-App.zip") do set APP_SIZE=%%~zi
for %%i in ("DecisionTreeUtility-Python.zip") do set PYTHON_SIZE=%%~zi

set /a APP_MB=APP_SIZE/1048576
set /a PYTHON_MB=PYTHON_SIZE/1048576

echo.
echo ========================================
echo  [SUCCESS] BUILD COMPLETED SUCCESSFULLY!
echo ========================================
echo.
echo Distribution files created:
echo - DecisionTreeUtility-App.zip     (%APP_MB% MB) - Application source
echo - DecisionTreeUtility-Python.zip  (%PYTHON_MB% MB) - Portable Python environment
echo - python-portable\                - Local portable environment (for testing)
echo.
echo Instructions for deployment:
echo 1. Provide both ZIP files to end users
echo 2. Users extract both to same folder
echo 3. Users run scripts\run_portable.bat
echo 4. Or deploy to network drive for shared access
echo.
echo For detailed instructions: See SETUP_WINDOWS.md
echo.

pause