# Windows Setup Guide

Two setup approaches depending on your Windows 11 environment and permissions.

## System Requirements
- Windows 10/11 (64-bit)
- 4GB RAM (8GB recommended for large datasets)
- 2GB free disk space

---

# Option A: Standard User 

**For users with NO command line access or software installation privileges**

## Method 1: Portable Application Package
1. **Download** two ZIP files from your IT department:
   - `DecisionTreeUtility-App.zip` (the application)
   - `Python.zip` (portable Python environment)

2. **Extract both files** to the same folder (e.g., `C:\Tools\DecisionTree\`)
   ```
   C:\Tools\DecisionTree\
   ├── main.py
   ├── scripts/
   ├── ui/
   ├── data/
   └── python-portable/  (from Python ZIP)
       ├── python.exe
       ├── Lib/
       └── Scripts/
   ```

3. **Double-click** `scripts\run_portable.bat` to launch the application

4. **No Python installation** or internet required - everything is included

---

# Option B: Developer/Power User Setup

**For users with command line access and software installation privileges**

## Method 1: Using Anaconda (Recommended)

### Step 1: Open Anaconda Prompt
- Search for "Anaconda Prompt" in Start Menu
- Run as regular user (no admin rights needed)

### Step 2: Download and Setup
```bash
# Download the project
git clone https://github.com/rupinrd3/bespoke-decision-tree-utility.git
cd bespoke-decision-tree-utility

# Create isolated environment
conda create -n bespoke_tree python=3.12 -y
conda activate bespoke_tree

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
# Make sure environment is activated
conda activate bespoke_tree

# Launch the application
python main.py
```

## Method 2: Standard Python Installation

If you don't have Anaconda:

1. **Install Python 3.12+** from [python.org](https://python.org) (check "Add to PATH")
2. **Download project:**
   ```cmd
   git clone https://github.com/rupinrd3/bespoke-decision-tree-utility.git
   cd bespoke-decision-tree-utility
   ```
3. **Install and run:**
   ```cmd
   pip install -r requirements.txt
   python main.py
   ```

---

# Quick Test (Both Options)
1. Launch application → you'll see a splash screen
2. Go to Help → About to verify version
3. File → Import Excel → load a sample dataset
4. Create Model → build your first decision tree

# Troubleshooting

## Standard User Issues

**run_portable.bat shows "No portable Python environment found"?**
- Verify `python-portable\` folder exists in the same directory as main.py
- Download `DecisionTreeUtility-Python.zip` from your IT department
- Extract the ZIP - should create `python-portable\` folder with python.exe

**Application won't start?**
- Try running as administrator (right-click `run_portable.bat` → "Run as administrator")
- Check if antivirus is blocking python.exe in the portable folder
- Verify all files were extracted properly (no empty folders)

**"Access Denied" or permission errors?**
- Extract to a folder where you have write permissions (Documents, Desktop)
- Avoid extracting to Program Files or C:\ root
- Check folder permissions for your data files

**Missing DLL errors?**
- Ensure you downloaded the complete portable Python environment
- Try installing Visual C++ Redistributables (usually pre-installed on Windows 11)

## Developer/Power User Issues

**Can't find Anaconda Prompt?**
- Search for "cmd" or "PowerShell" instead
- Try the Standard Python method above

**Permission errors?**
- Use Anaconda method (doesn't need admin rights)
- Or install Python with "Install for all users" unchecked

**PyQt5 errors?**
```bash
conda install pyqt=5.15.9 -y
```

**Git not found?**
- Download project as ZIP from GitHub instead
- Or install Git from [git-scm.com](https://git-scm.com/)

## General Issues

**Application crashes?**
- Check logs in the `logs/` folder
- Verify you have at least 4GB available RAM
- Close other memory-intensive applications

**Performance issues with large files?**
- Files >100MB may take time to load
- Consider using data sampling for initial analysis
- Close unused applications to free memory

**Need help?** 
- Check logs in the `logs/` folder
- Report issues: [GitHub Issues](https://github.com/rupinrd3/bespoke-decision-tree-utility/issues)

---

# Deployment Notes for IT Administrators

## Creating Portable Python Environment for Standard Users

### Step 1: Create Application ZIP
```cmd
REM Download/clone the application source
git clone https://github.com/your-repo/decision-tree-utility.git
cd decision-tree-utility

REM Create application ZIP (exclude Python environment)
PowerShell Compress-Archive -Path "." -DestinationPath "DecisionTreeUtility-App.zip" -Exclude "python-portable","*.git*","__pycache__"
```

### Step 2: Create Portable Python Environment
```cmd
REM Install Python 3.12+ on a development machine
REM Create virtual environment with all dependencies
python -m venv python-portable
python-portable\Scripts\activate

REM Install all required packages
pip install -r requirements.txt

REM Create portable Python ZIP
PowerShell Compress-Archive -Path "python-portable" -DestinationPath "DecisionTreeUtility-Python.zip"
```

### Step 3: Alternative - Use WinPython Portable
1. Download **WinPython** from https://winpython.github.io/
2. Choose "3.12.x Zero" version (~150MB) 
3. Extract and install required packages:
   ```cmd
   cd WinPython\python-3.12.x\Scripts
   pip install -r path\to\requirements.txt
   ```
4. ZIP the entire WinPython folder

### Step 4: Distribution Package Structure
Create final distribution with both ZIPs or combine into one:
```
DecisionTreeUtility-Complete.zip
├── main.py
├── scripts\
├── ui\
├── data\
├── models\
├── config.json
├── requirements.txt
└── python-portable\          (from Step 2)
    ├── python.exe
    ├── Lib\
    └── Scripts\
```

### Step 5: Network Deployment
- Deploy complete package to shared network drive accessible to all users
- Ensure read/execute permissions for target user groups
- Create desktop shortcuts pointing to `scripts\run_portable.bat`
- Test with restricted user account before rollout

## Package Size Estimates
- Application only: ~5-10MB
- Portable Python environment: ~200-400MB  
- Complete package: ~400-500MB
- Network deployment: Store once, access by many users

---

**That's it!** The application should now be running. See the [User Manual](USER_MANUAL.md) for how to use the application.