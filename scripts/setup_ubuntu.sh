#!/bin/bash
# Bespoke Decision Tree Utility - Ubuntu Setup Script
# This script sets up the development environment on Ubuntu/Linux systems

set -e  # Exit on any error

# Script options
SKIP_SYSTEM_PACKAGES=false
DEVELOPMENT_MODE=false
PYTHON_VERSION="python3"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-system)
            SKIP_SYSTEM_PACKAGES=true
            shift
            ;;
        --dev)
            DEVELOPMENT_MODE=true
            shift
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --skip-system    Skip system package installation"
            echo "  --dev           Install development dependencies"
            echo "  --python <ver>   Use specific Python version (default: python3)"
            echo "  --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN} Bespoke Decision Tree Utility${NC}"
    echo -e "${CYAN} Ubuntu/Linux Setup Script${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_section() {
    echo ""
    echo -e "${BLUE}--- $1 ---${NC}"
}

# Change to the parent directory
cd "$(dirname "$0")/.."

print_header

print_status "Starting setup for Bespoke Decision Tree Utility..."

# Check if running with sudo for system packages
if [[ $EUID -eq 0 ]]; then
    print_warning "Running as root. This is not recommended for the entire setup."
    print_warning "Please run without sudo. System packages will be installed with sudo when needed."
    exit 1
fi

# Detect Ubuntu version
if [[ -f /etc/os-release ]]; then
    source /etc/os-release
    print_status "Detected: $PRETTY_NAME"
else
    print_warning "Could not detect Ubuntu version. Proceeding with generic setup."
fi

print_section "System Packages"

if [[ "$SKIP_SYSTEM_PACKAGES" == false ]]; then
    print_status "Updating package lists..."
    sudo apt update

    print_status "Installing system dependencies..."
    
    # Core development tools
    sudo apt install -y \
        build-essential \
        git \
        curl \
        wget \
        software-properties-common \
        ca-certificates \
        gnupg \
        lsb-release
    
    # Python and development packages
    sudo apt install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        python3-distutils
    
    # Qt5 and GUI dependencies
    sudo apt install -y \
        python3-pyqt5 \
        python3-pyqt5.qtsvg \
        pyqt5-dev-tools \
        qttools5-dev-tools \
        libqt5gui5 \
        libqt5core5a \
        libqt5widgets5 \
        qt5-gtk-platformtheme \
        libxcb-xinerama0 \
        libxcb-cursor0
    
    # Scientific computing libraries
    sudo apt install -y \
        libopenblas-base \
        libopenblas-dev \
        libblas3 \
        libblas-dev \
        liblapack3 \
        liblapack-dev \
        libatlas-base-dev \
        gfortran \
        pkg-config
    
    # Graphics and media libraries
    sudo apt install -y \
        libfreetype6-dev \
        libpng-dev \
        libjpeg-dev \
        libfontconfig1-dev
    
    print_success "System packages installed"
else
    print_warning "Skipping system package installation"
fi

print_section "Python Environment"

# Check Python version
if ! command -v $PYTHON_VERSION &> /dev/null; then
    print_error "$PYTHON_VERSION not found"
    print_error "Please install Python 3.8+ or specify correct version with --python"
    exit 1
fi

PYTHON_VER_OUTPUT=$($PYTHON_VERSION --version)
print_status "Found: $PYTHON_VER_OUTPUT"

# Check Python version is 3.8+
$PYTHON_VERSION -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null
if [ $? -ne 0 ]; then
    print_error "Python 3.8+ required"
    print_error "Current version: $PYTHON_VER_OUTPUT"
    exit 1
fi

print_success "Python version check passed"

# Create virtual environment
print_status "Creating virtual environment..."

if [[ -d "venv" ]]; then
    print_warning "Virtual environment already exists"
    print_status "To recreate, delete 'venv' directory and run setup again"
else
    $PYTHON_VERSION -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Verify we're in the virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_error "Failed to activate virtual environment"
    exit 1
fi

print_success "Virtual environment activated: $VIRTUAL_ENV"

print_section "Python Dependencies"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_status "Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    print_error "Failed to install dependencies"
    print_error "Check your internet connection and requirements.txt file"
    exit 1
fi

print_success "Python dependencies installed"

# Install development dependencies if requested
if [[ "$DEVELOPMENT_MODE" == true ]]; then
    print_section "Development Dependencies"
    
    print_status "Installing development tools..."
    pip install pytest black flake8 mypy pre-commit jupyter
    
    # Set up pre-commit hooks if in git repository
    if [[ -d ".git" ]]; then
        print_status "Setting up pre-commit hooks..."
        pre-commit install
        print_success "Pre-commit hooks installed"
    fi
    
    print_success "Development dependencies installed"
fi

print_section "Installation Test"

# Test the installation
print_status "Testing core imports..."
python -c "
try:
    import PyQt5
    import pandas
    import numpy
    import sklearn
    import matplotlib
    print('✓ All core dependencies imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    print_success "Installation test passed"
else
    print_error "Installation test failed"
    print_error "Some dependencies may not be properly installed"
    exit 1
fi

print_section "Desktop Integration"

# Create desktop entry
DESKTOP_FILE="$HOME/.local/share/applications/bespoke-decision-tree.desktop"
ICON_PATH="$(pwd)/images/icon.png"
EXEC_PATH="$(pwd)/scripts/run_ubuntu.sh"

print_status "Creating desktop entry..."

mkdir -p "$HOME/.local/share/applications"

cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Bespoke Decision Tree Utility
Comment=Interactive Decision Tree Builder for Credit Risk Assessment
Exec=$EXEC_PATH
Icon=$ICON_PATH
Terminal=false
StartupWMClass=bespoke-decision-tree
Categories=Development;Science;Education;
Keywords=machine-learning;decision-tree;data-science;analytics;
EOF

chmod +x "$DESKTOP_FILE"

# Make run script executable
chmod +x "$EXEC_PATH"

print_success "Desktop integration completed"

print_section "Final Configuration"

# Create .env file if it doesn't exist
if [[ ! -f ".env" && -f ".env.example" ]]; then
    print_status "Creating .env file from template..."
    cp .env.example .env
    print_success "Created .env file (you may customize it if needed)"
fi

# Set up log directory
mkdir -p logs
print_success "Log directory created"

# Deactivate virtual environment
deactivate

print_section "Setup Complete"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} 🎉 Setup Complete! 🎉${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${CYAN}To run the application:${NC}"
echo "1. ./scripts/run_ubuntu.sh"
echo "2. Or from the applications menu: 'Bespoke Decision Tree Utility'"
echo "3. Or manually: source venv/bin/activate && python main.py"
echo ""
echo -e "${CYAN}Documentation:${NC}"
echo "• README.md - Main documentation"
echo "• SETUP_UBUNTU.md - Detailed Ubuntu setup guide"  
echo "• Workflow_System_User_Manual.md - User manual"
echo "• CONTRIBUTING.md - Contribution guidelines"
echo ""
echo -e "${CYAN}Troubleshooting:${NC}"
echo "• Check logs/ directory for error logs"
echo "• Run with --debug flag for verbose output"
echo "• See setup guides for platform-specific issues"
echo ""

# Test run prompt
echo -e "${YELLOW}Would you like to test run the application now? (y/n)${NC}"
read -r response

if [[ "$response" == "y" || "$response" == "Y" ]]; then
    echo ""
    print_status "Launching application..."
    ./scripts/run_ubuntu.sh
fi

echo ""
print_status "Setup script completed successfully!"
echo ""