# Ubuntu Setup Guide

Quick setup for Ubuntu/Linux systems.

## System Requirements
- Ubuntu 18.04+ (or any recent Linux distribution)
- Python 3.12+ 
- 4GB RAM (8GB recommended for large datasets)
- 2GB free disk space

## Quick Installation

### Step 1: Install System Dependencies
```bash
# Update system
sudo apt update

# Install Python 3.12+ and essential packages
sudo apt install python3.12 python3.12-venv python3-pip git -y

# Install Qt5 for GUI (required)
sudo apt install python3-pyqt5 libxcb-xinerama0 -y
```

### Step 2: Download and Setup
```bash
# Download the project
git clone https://github.com/rupinrd3/bespoke-decision-tree-utility.git
cd bespoke-decision-tree-utility

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
# Make sure environment is activated
source venv/bin/activate

# Launch the application
python main.py
```

## Alternative: System-Wide Installation
```bash
# Download project
git clone https://github.com/rupinrd3/bespoke-decision-tree-utility.git
cd bespoke-decision-tree-utility

# Install dependencies system-wide
pip3 install --user -r requirements.txt

# Run application
python3 main.py
```

## Display Issues (Remote/SSH)

**For SSH with X11 forwarding:**
```bash
ssh -X username@hostname
export DISPLAY=:0
python3 main.py
```

**For Wayland users:**
```bash
export QT_QPA_PLATFORM=xcb
python3 main.py
```

## Quick Test
1. Launch application → you'll see splash screen
2. Help → About to verify version
3. File → Import CSV → load a sample dataset
4. Create Model → build your first decision tree

## Troubleshooting

**Python 3.12 not available?**
```bash
# Add deadsnakes PPA for newer Python versions
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.12 python3.12-venv -y
```

**Qt/GUI errors?**
```bash
sudo apt install python3-pyqt5 libxcb-xinerama0 libxcb-cursor0 -y
```

**Permission errors?**
```bash
# Use virtual environment or --user flag
pip3 install --user -r requirements.txt
```

**Need help?**
- Check logs in the `logs/` folder  
- Report issues: [GitHub Issues](https://github.com/rupinrd3/bespoke-decision-tree-utility/issues)

---

**That's it!** The application should now be running. See the [User Manual](USER_MANUAL.md) for how to use the application.