# Bespoke Decision Tree Utility

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Ubuntu-lightgrey)](https://github.com/rupinrd3/bespoke-decision-tree-utility)

A powerful, interactive desktop application for building, visualizing, and analyzing decision tree models with a focus on credit risk assessment and binary classification tasks.


## 🌟 Features

- **🎯 Interactive Tree Building**: Visual drag-and-drop interface for creating decision trees
- **📊 Advanced Analytics**: Variable importance, performance metrics, and detailed node statistics  
- **🚀 High Performance**: Efficiently handles datasets up to 800MB/400,000 records
- **🔄 Workflow Canvas**: Visual pipeline builder for multi-step data processing
- **📤 Multiple Export Formats**: Python code, PMML, JSON, and proprietary formats
- **💾 Smart Memory Management**: Optimized for large datasets with minimal memory footprint
- **🎨 Modern UI**: Clean, intuitive interface built with PyQt5
- **🔧 Data Processing**: Built-in data transformation, filtering, and feature engineering tools

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Support](#support)

## 🚀 Installation

### Prerequisites

- **Windows**: Windows 10/11 (64-bit) with Anaconda (typical in corporate environments)
- **Ubuntu**: Ubuntu 18.04+ or any recent Linux distribution  
- **Python**: 3.12+ (required for optimal performance)
- **RAM**: 4GB minimum (8GB recommended for large datasets)
- **Storage**: 2GB free space

### Platform-Specific Setup

- **Windows Users**: See [Windows Setup Guide](SETUP_WINDOWS.md) - Optimized for corporate Anaconda environments
- **Ubuntu Users**: See [Ubuntu Setup Guide](SETUP_UBUNTU.md) - Quick setup with virtual environments

### Quick Install (Ubuntu)

```bash
# Clone the repository
git clone https://github.com/rupinrd3/bespoke-decision-tree-utility.git
cd bespoke-decision-tree-utility

# Create virtual environment  
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Quick Install (Windows - Anaconda)

```bash
# Open Anaconda Prompt
git clone https://github.com/rupinrd3/bespoke-decision-tree-utility.git
cd bespoke-decision-tree-utility

# Create conda environment
conda create -n bespoke_tree python=3.12 -y
conda activate bespoke_tree

# Install dependencies
pip install -r requirements.txt

# Run the application  
python main.py
```

## 🎯 Quick Start

1. **Launch the Application**
   ```bash
   python main.py  # Ubuntu/Linux
   # or activate conda environment on Windows: conda activate bespoke_tree && python main.py
   ```
   - You'll see a splash screen, then the main workflow canvas

2. **Import Your Data**
   - **Method 1**: Use toolbar button **📊 Import CSV** or **📈 Import Excel**  
   - **Method 2**: Use **File → Import Data** menu
   - **Method 3**: Use **🧙‍♂️ Data Import Wizard** for guided import
   - Supported formats: CSV, Excel (.xlsx, .xls), TSV, pipe-delimited

3. **Create a Workflow**
   - A blue **Dataset** node appears after data import
   - Click **🌳 Create Model** to add a decision tree node
   - The system automatically connects Dataset → Decision Tree

4. **Execute the Workflow First**
   - Click **▶️ Execute Workflow** button (or press F5)  
   - This prepares the data flow and makes nodes ready for configuration
   - Wait for workflow processing to complete

5. **Configure Your Model**
   - **Right-click** the Decision Tree node (don't double-click yet!)
   - Select **"Configure"** from the context menu
   - Set **Title** and select your **target variable** from dropdown
   - Click **"Configure Model Parameters"** for advanced settings (splitting criteria, pruning, etc.)
   - Click **"Start Manual Tree Building"** to enable interactive control

6. **Build Your Tree Interactively**
   - **Double-click** the Decision Tree node to open the tree visualization window
   - **Right-click** on tree nodes → **"Find Optimal Split"** → Apply splits
   - Continue building your tree by selecting optimal splits

7. **Analyze Performance**
   - Add **📊 Evaluation** node and connect it to your model
   - Execute workflow again to see performance metrics
   - Double-click Evaluation node to view accuracy, precision, recall, F1 score, and more

8. **Export Your Model**
   - **File → Export Model → Export to Python** - Generate Python code
   - **File → Export Model → Export to PMML** - Industry-standard format
   - Models can be deployed in production systems

## 🏗️ Project Structure

```
bespoke-decision-tree-utility/
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LICENSE                     # MIT License
├── SETUP_WINDOWS.md           # Windows setup instructions
├── SETUP_UBUNTU.md            # Ubuntu setup instructions
├── CHANGELOG.md               # Version history
├── CONTRIBUTING.md            # Contribution guidelines
├── analytics/                 # Statistical analysis modules
├── business/                  # Business logic and calculations  
├── data/                      # Data import and processing
├── export/                    # Model export functionality
├── models/                    # Core decision tree algorithms
├── ui/                        # User interface components
├── utils/                     # Utility functions and helpers
├── workflow/                  # Workflow execution engine
├── scripts/                   # Platform-specific run scripts
└── old/                       # Archive of development artifacts
```

## 📈 Supported Data Formats

- **Input**: CSV, Excel (.xlsx, .xls), TSV, pipe-delimited
- **Output**: Python scripts, PMML, JSON, proprietary format
- **Data Size**: Up to 800MB files / 400,000 records
- **Variable Types**: Numerical, categorical, binary

## 🔧 Key Capabilities

### Data Processing
- Advanced filtering and data transformation
- Missing value handling strategies
- Feature engineering with formula editor
- Data quality assessment tools

### Model Building
- Interactive split point selection
- Automatic optimal split finding
- Tree pruning algorithms
- Cross-validation support

### Analysis & Reporting
- Variable importance calculations
- Performance curves (ROC, Lift, Gains)
- Node-level statistics and reports
- Model comparison tools

### Workflow Management
- Visual pipeline builder
- Reusable workflow templates
- Batch processing capabilities
- Automated model validation

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10 (64-bit) or Ubuntu 18.04+
- **RAM**: 4GB
- **CPU**: Dual-core processor
- **Storage**: 1GB free space
- **Display**: 1024x768 resolution

### Recommended Requirements  
- **RAM**: 8GB or higher
- **CPU**: Quad-core processor
- **Storage**: 2GB free space
- **Display**: 1920x1080 resolution

## 📚 Documentation

- **[📖 User Manual](USER_MANUAL.md)** - Complete step-by-step guide with screenshots
- **[🪟 Windows Setup](SETUP_WINDOWS.md)** - Corporate Anaconda environment setup
- **[🐧 Ubuntu Setup](SETUP_UBUNTU.md)** - Linux installation and configuration  
- **[🤝 Contributing Guide](CONTRIBUTING.md)** - How to contribute to the project

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Code of conduct and community guidelines
- Development environment setup
- Submitting issues and pull requests  
- Code style and testing guidelines

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **🐛 Report Issues**: [GitHub Issues](https://github.com/rupinrd3/bespoke-decision-tree-utility/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/rupinrd3/bespoke-decision-tree-utility/discussions) 
- **📧 Email**: For complex issues or commercial support

## 🏷️ Version History

See [CHANGELOG.md](CHANGELOG.md) for a complete list of changes and version history.

## 🙏 Acknowledgments

- Built with [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) for the user interface
- Powered by [scikit-learn](https://scikit-learn.org/) and [NumPy](https://numpy.org/) for machine learning
- Visualization using [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)

---

**Made with ❤️ for the data science community**