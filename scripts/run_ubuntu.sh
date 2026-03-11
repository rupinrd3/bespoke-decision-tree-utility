#!/bin/bash
# Bespoke Decision Tree Utility - Ubuntu/Linux Runner Script
# This script launches the application on Ubuntu/Linux systems

set -e  # Exit on any error

echo "========================================"
echo " Bespoke Decision Tree Utility"
echo " Ubuntu/Linux Launcher"
echo "========================================"

# Change to the parent directory (where main.py is located)
cd "$(dirname "$0")/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found"
    print_error "Please install Python 3.8+ using:"
    print_error "  sudo apt update"
    print_error "  sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

print_status "Python 3 found: $(python3 --version)"

# Check Python version
python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null
if [ $? -ne 0 ]; then
    print_error "Python 3.8+ required"
    print_error "Your Python version: $(python3 --version)"
    print_error "Please upgrade Python to 3.8 or higher"
    exit 1
fi

print_status "Python version check passed"

# Check for virtual environment and activate if exists
if [ -f "venv/bin/activate" ]; then
    print_status "Activating virtual environment..."
    source venv/bin/activate
else
    print_warning "No virtual environment found. Using system Python."
    print_warning "For better dependency management, consider creating a virtual environment:"
    print_warning "  python3 -m venv venv"
    print_warning "  source venv/bin/activate"
    print_warning "  pip install -r requirements.txt"
fi

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_status "Using virtual environment: $VIRTUAL_ENV"
fi

# Check if required packages are installed
print_status "Checking dependencies..."
python3 -c "import PyQt5" 2>/dev/null
if [ $? -ne 0 ]; then
    print_status "Installing required dependencies..."
    
    # Try installing with pip first
    if command -v pip &> /dev/null; then
        pip install -r requirements.txt
    elif command -v pip3 &> /dev/null; then
        pip3 install -r requirements.txt
    else
        print_error "pip not found. Installing with apt..."
        sudo apt update
        sudo apt install -y python3-pip
        pip3 install -r requirements.txt
    fi
    
    if [ $? -ne 0 ]; then
        print_error "Failed to install dependencies"
        print_error "Please try installing manually:"
        print_error "  pip3 install -r requirements.txt"
        exit 1
    fi
fi

print_status "Dependencies check passed"

# Check for Qt platform plugins
if [ -n "$WAYLAND_DISPLAY" ]; then
    print_status "Wayland session detected"
    export QT_QPA_PLATFORM=wayland
    print_status "Set QT_QPA_PLATFORM=wayland"
elif [ -n "$DISPLAY" ]; then
    print_status "X11 session detected"
    export QT_QPA_PLATFORM=xcb
    print_status "Set QT_QPA_PLATFORM=xcb"
else
    print_warning "No display server detected"
    print_warning "If running over SSH, ensure X11 forwarding is enabled: ssh -X"
fi

# Additional environment variables for better Qt performance
export QT_AUTO_SCREEN_SCALE_FACTOR=1
export QT_ENABLE_HIGHDPI_SCALING=1

# For some graphics cards that have issues with shared memory
export QT_X11_NO_MITSHM=1

# Launch the application
print_status "Launching Bespoke Decision Tree Utility..."
echo ""

# Try to launch with error handling
if python3 main.py; then
    print_status "Application closed successfully"
else
    exit_code=$?
    echo ""
    print_error "Application exited with error code: $exit_code"
    print_error "Check the logs directory for error details"
    
    # Provide troubleshooting hints
    echo ""
    print_warning "Troubleshooting tips:"
    print_warning "1. Check if all system dependencies are installed:"
    print_warning "   sudo apt install python3-pyqt5 libxcb-xinerama0"
    print_warning "2. If using Wayland, try forcing X11:"
    print_warning "   export QT_QPA_PLATFORM=xcb"
    print_warning "3. For detailed logs, run: python3 main.py --debug"
    
    exit $exit_code
fi

# Deactivate virtual environment if it was activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    deactivate 2>/dev/null || true
fi

echo ""