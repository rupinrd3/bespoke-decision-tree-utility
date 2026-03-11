# User Manual: Bespoke Decision Tree Utility

Complete step-by-step guide for building decision tree models.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Understanding Menus and Toolbar](#understanding-menus-and-toolbar)
3. [Importing Data](#importing-data)
4. [Creating Your First Model](#creating-your-first-model)
5. [Working with the Workflow Canvas](#working-with-the-workflow-canvas)
6. [Building Decision Trees](#building-decision-trees)
7. [Analyzing Model Performance](#analyzing-model-performance)
8. [Advanced Features](#advanced-features)
9. [Tips and Best Practices](#tips-and-best-practices)

---

## Getting Started

### First Launch
**Main window opens** - The interface has three main areas:
   - **Menu Bar** at the top (File, Operations, Data, Model, Analysis, View, Help)
   - **Toolbar** with quick action buttons  
   - **Workflow Canvas** in the center (large gray area)
   - **Status bar** at the bottom showing memory usage and context

### Understanding the Interface

The application uses a **workflow-based approach** where you connect different types of nodes to build your analysis pipeline.

**Main Toolbar Buttons:**
- 📁 **New Project** - Start a new analysis project (Ctrl+N)
- 📂 **Open Project** - Load an existing project (Ctrl+O)
- 💾 **Save** - Save your current work (Ctrl+S)
- 📊 **Import CSV** - Load data from CSV files
- 📈 **Import Excel** - Load data from Excel files  
- 🧙‍♂️ **Data Import Wizard** - Guided data import process
- 🌳 **Create Model** - Build a new decision tree
- 📊 **Evaluation** - Add model evaluation node
- 🎨 **Visualization** - Add visualization node
- ▶️ **Execute Workflow** - Run your analysis pipeline (F5)
- 🔍 **Filter** - Add data filtering node
- 🔄 **Transform** - Add data transformation node
- 🔍+ **Zoom In** - Zoom into workflow canvas (Ctrl++)
- 🔍- **Zoom Out** - Zoom out of workflow canvas (Ctrl+-)

---

## Understanding Menus and Toolbar

### File Menu
- **New Project** (Ctrl+N) - Create new project
- **Open Project** (Ctrl+O) - Open existing project
- **Save Project** (Ctrl+S) - Save current project
- **Recent Projects** - Access recently opened projects
- **Import Data** submenu:
  - **Import Wizard** - Comprehensive data import
  - **Import CSV** - Load CSV files
  - **Import Excel** - Load Excel files
- **Export Model** submenu:
  - **Export to Python** - Generate Python code
  - **Export to PMML** - Export in PMML format
- **Exit** (Ctrl+Q) - Close application

### Operations Menu (Main Workflow Actions)
- **📊 Dataset** - Add dataset to workflow
- **🔍 Filter** - Add filter node to workflow
- **🔄 Transform** - Add transform node to workflow
- **🌳 Model** - Add model node to workflow
- **🎨 Visualization** - Add visualization node
- **▶️ Execute Workflow** (F5) - Run complete workflow

### Data Menu
- **Filter Data** - Apply filters to current dataset
- **Transform Data** - Apply data transformations
- **Create Variable** - Create new variables with formulas

### Model Menu
- **Create Model** - Add new decision tree model
- **Configure Model** - Set up tree parameters

### Analysis Menu
- **Variable Importance** - Analyze feature importance
- **Performance Evaluation** - Evaluate model performance

### View Menu
- **Zoom In** (Ctrl++) - Zoom into canvas
- **Zoom Out** (Ctrl+-) - Zoom out of canvas
- **Fit to View** (Ctrl+0) - Fit workflow to screen

### Help Menu
- **Help Contents** (F1) - Show help documentation
- **About** - Application information

---

## Importing Data

### Step 1: Choose Your Data Source

**For Excel files:**
1. Click **📈 Import Excel** button
2. Browse and select your `.xlsx` or `.xls` file
3. Choose the worksheet if multiple sheets exist
4. Click **OK**

**For CSV files:**
1. Click **📊 Import CSV** button  
2. Browse and select your `.csv` file
3. Set delimiter (comma, semicolon, tab) if needed
4. Click **OK**

### Step 2: Verify Data Import
- A **Dataset** node (blue rectangle) appears on the canvas
- The node shows your dataset name (e.g., "application_test")
- Double-click the node to preview your data

### Data Requirements
✅ **What works well:**
- Binary target variables (0/1, Yes/No, Default/No Default)
- Mix of numerical and categorical variables
- Clean data with minimal missing values

⚠️ **What to check:**
- Target variable should be clearly defined
- Remove ID columns or irrelevant fields
- Handle missing values appropriately

---

## Creating Your First Model

### Quick Workflow Overview
The complete workflow for building a decision tree is:
1. **Set up workflow** - Add dataset and decision tree model nodes
2. **Execute workflow** - Click ▶️ Execute Workflow button to prepare data flow
3. **Configure model** - Right-click decision tree node → Configure → Set title and target variable
4. **Set parameters** - Configure Model Parameters → Set splitting criteria, pruning, etc.
5. **Start building** - Click "Start Manual Tree Building" for interactive control
6. **Open tree window** - Double-click decision tree node to view tree structure
7. **Build interactively** - Right-click nodes → Find Optimal Split → Apply splits

---

### Step 1: Set Up Your Workflow
1. With your dataset ready, click **🌳 Create Model**
2. A **Decision_Tree_1** node (green rectangle) appears
3. The system automatically connects your dataset to the tree node
4. **Add evaluation and visualization nodes** (optional but recommended):
   - Click **📊 Evaluation** to add model performance analysis
   - Click **🎨 Visualization** to add tree visualization
   - **Connect the nodes:** Dataset → Decision Tree → Evaluation → Visualization

### Step 2: Execute the Workflow
1. **Click the ▶️ Execute Workflow button** (or press F5)
2. This prepares the data flow and makes the nodes ready for configuration
3. You'll see the workflow processing - wait for it to complete

### Step 3: Configure the Decision Tree Model
1. **Right-click** on the Decision Tree node (don't double-click yet!)
2. Select **"Configure"** from the context menu
3. The **"Configure Decision_Tree_1"** dialog opens
4. **Essential settings:**
   - **Title:** Give your model a descriptive name (e.g., "Credit Risk Model")
   - **Target Variable:** Select your outcome variable from dropdown (e.g., "default_flag")

### Step 4: Set Model Parameters
1. Click **"Configure Model Parameters..."** button
2. The **Decision Tree Configuration** dialog opens with advanced settings:

**General Tree Structure:**
- **Splitting Criterion:** `Entropy` (recommended) or `Gini`
- **Max Depth:** `Unlimited` (or set a number like 5-10)
- **Growth Mode:** `Manual` (for interactive building) or `Automatic`

**Node Splitting Conditions:**
- **Min Samples to Split:** `2` (minimum records to make a split)
- **Min Samples per Leaf:** `1` (minimum records in final nodes)
- **Min Impurity Decrease:** `0.00000` (quality threshold for splits)

**Pruning (Recommended):**
- ✅ **Enable Pruning** (prevents overfitting)
- **Pruning Method:** `Cost Complexity`
- **Complexity Parameter:** `0.01000`

3. Click **"Apply Configuration"** to save settings

### Step 5: Start Manual Tree Building
1. Back in the main configuration dialog, click **"Start Manual Tree Building..."**
2. This switches to interactive mode where you control each split decision

---

## Working with the Workflow Canvas

### Understanding Node Types

**📊 Dataset Nodes (Blue)**
- Represent your data sources
- Show dataset name and record count
- Connect to other nodes via blue connection points

**🌳 Decision Tree Nodes (Green)** 
- Your machine learning models
- Show model status ("Model: Not assigned" initially)
- Connect to datasets and evaluation nodes

**🔍 Filter Nodes (Orange)**
- Remove or select specific records
- Useful for data cleaning
- Connect between datasets and models

**🔄 Transform Nodes (Purple)**
- Modify your data (new variables, data types)
- Create calculated fields
- Apply data transformations

**📊 Evaluation Nodes (Pink/Red)**
- Assess model performance  
- Generate accuracy metrics
- Create performance reports

**📈 Visualization Nodes (Yellow)**
- Display charts and graphs
- Show model results visually
- Generate presentation-ready outputs

### Connecting Nodes
1. **Hover over** a node's edge - you'll see small colored dots
2. **Click and drag** from one dot to another node
3. **Lines connect** your workflow - blue lines show data flow

---

## Building Decision Trees

### Step 6: Open the Tree Building Window
1. After completing the configuration steps above, **double-click** the Decision Tree node
2. This opens the **Tree Building Window** showing your decision tree structure
3. Initially, you'll see just the **root node** containing all your data

### Step 7: Build Your Tree Interactively
1. **Right-click** on the root node (or any node you want to split)
2. From the context menu, select **🔍 Find Optimal Split**
3. The **"Find Optimal Split"** dialog opens for that node

**Understanding the Find Optimal Split Dialog:**

**Split Candidates Table (Left side):**
- **Variable column:** Shows all available variables ranked by information gain
- **Type column:** Numerical or Categorical  
- **Stat Value:** Information gain score (higher = better split quality)
- Variables are automatically ranked - best splits appear at the top

**Split Preview Panel (Right side):**
- Shows the exact split condition (e.g., "EXT_SOURCE_2 ≤ 0.415")
- **Left branch:** Records meeting the condition with sample count and metric
- **Right branch:** Records not meeting the condition
- **Preview the business logic** before applying

**Variable Distribution Panel (Bottom right):**
- Statistical summary for the selected variable
- **Min, Max, Mean, Standard deviation**
- **Quartile information:** Q1, Median (Q2), Q3
- Helps understand the data distribution

### Step 8: Apply Splits and Continue Building
1. **Select a variable** from the ranked list (top variables usually work best)
2. **Review the split preview** - does it make business sense?
3. Check the **sample counts** and **target distribution** in each branch
4. Click **"Apply Split"** to create the split
5. **The tree updates** showing your new branches
6. **Repeat the process:** Right-click on new nodes → Find Optimal Split
7. **Continue until satisfied** with the tree depth and accuracy

### Tree Visualization and Navigation
Your decision tree displays as a hierarchical structure:

**Root Node (Top):**
- Contains all your data initially  
- Shows total sample count and target distribution
- Example: "0" (282,686) 91.93%, "1" (24,825) 8.07%

**Split Nodes (Middle levels):**
- Show the split condition (e.g., "EXT_SOURCE_2 ≤ 0.415")
- Display **Gini impurity score** (lower = better separation)
- Show **sample counts** flowing to each branch

**Leaf Nodes (Bottom):**
- Final prediction nodes (no more splits)
- Show the **predicted class** and confidence
- Display sample counts and percentages

### Advanced Tree Operations (Right-click on any node)
- **📊 View Node Statistics (Ctrl+I)** - Detailed statistical analysis  
- **🔍 Find Optimal Split (Ctrl+F)** - Find best variable to split this node
- **✂️ Manual Split (Ctrl+M)** - Create custom split conditions
- **📋 Copy Node Info (Ctrl+C)** - Copy node details to clipboard
- **📁 Paste Node Structure (Ctrl+V)** - Duplicate tree sections

### Alternative: Automatic Tree Building
If you prefer automatic building:
1. In the initial configuration, set **Growth Mode** to `Automatic`
2. Click **OK** instead of "Start Manual Tree Building"
3. The system builds the complete tree automatically using optimal splits
4. **Double-click** the Decision Tree node to view the results

---

## Analyzing Model Performance

### Step 1: Ensure Evaluation Node is Connected to Decision Tree Node
1. If you haven't already, add Evaluation and Visualization nodes
2. **Connect** your Decision Tree node to the Evaluation node
3. **Connect** your Evaluation node to the Visualization node
4. Click on **Execute Workflow** button to run the evaluation

### Step 2: View Performance Metrics  
1. **Double-click** the Visualization node to open the Detail Window
2. The **Model Overview** tab displays comprehensive performance metrics:

**Decision Tree Model Summary:**
- **Tree Depth:** How many levels deep
- **Total Nodes:** Total decision points
- **Leaf Nodes:** Final prediction nodes
- **Decision Rules:** Total number of rules

**Core Performance Metrics:**
- **Accuracy:** Overall correctness (e.g., 0.9193 = 91.93%)
- **Precision:** How many selected items are relevant (1.0000 = 100%)
- **Recall:** How many relevant items are selected (0.9193)
- **F1 Score:** Balance of precision and recall (0.9579)
- **AUC-ROC:** Area under curve (0.6062)
- **KS Statistic:** Separation measure (0.2221)

**Confusion Matrix:**
- Shows actual vs predicted results
- **True Positives/Negatives:** Correct predictions
- **False Positives/Negatives:** Incorrect predictions

**Model Complexity vs Performance:**
- **Complexity Score:** Model simplicity (8.32)
- **Overfitting Risk:** Low/Medium/High
- **Interpretability:** How easy to understand

### Step 3: Performance Analysis Tabs

**📊 Tree Analysis Tab:**
- Detailed tree structure
- Node-by-node breakdown
- Split quality assessment

**📈 Performance Curves Tab:**
- ROC Curve (Receiver Operating Characteristic)  
- Lift Chart (Model improvement over random)
- Gains Chart (Cumulative gains)

**📋 Node Report Tab:**
- Detailed statistics for every node
- Export options (Excel, CSV)
- Business rule generation

---

## Advanced Features

### Data Filtering
1. Click **🔍 Filter** button
2. Connect Dataset → Filter → Decision Tree
3. Set filter conditions:
   - Variable selection
   - Comparison operators (>, <, =, ≠)
   - Value thresholds
   - Multiple condition logic (AND/OR)

### Data Transformation
1. Click **🔄 Transform** button  
2. Connect Dataset → Transform → Decision Tree
3. Available transformations:
   - **New Variables:** Calculate fields (e.g., Income/Age ratio)
   - **Binning:** Group numerical values into categories
   - **Encoding:** Convert text to numbers
   - **Scaling:** Normalize value ranges

### Model Comparison
1. Create multiple Decision Tree nodes
2. Connect same dataset to different trees
3. Use different configurations
4. Compare evaluation metrics
5. Choose the best performing model

### Export Options
**Export Model Code:**
- Python script for production use
- PMML format for other tools
- JSON for system integration

**Export Results:**
- Performance reports (Excel, PDF)
- Tree diagrams (PNG, SVG)
- Prediction rules (text format)

---

## Tips and Best Practices

### 🎯 Data Preparation
- **Clean your data first** - Remove duplicates, fix missing values
- **Understand your target** - What are you trying to predict?
- **Feature selection** - Remove irrelevant columns (IDs, timestamps)
- **Balance check** - Ensure reasonable distribution of target classes

### 🌳 Model Building
- **Start simple** - Begin with default settings
- **Enable pruning** - Prevents overfitting to your data
- **Set reasonable depth** - Trees deeper than 10 levels are often too complex
- **Check business logic** - Do the splits make sense in your domain?

### 📊 Performance Evaluation
- **Don't just look at accuracy** - Consider precision and recall
- **Test on new data** - High accuracy on training data isn't enough
- **Interpret results** - Can you explain why the model makes its decisions?
- **Document assumptions** - Record what data and settings you used

### ⚠️ Common Mistakes to Avoid
- **Overfitting** - Model works great on training data but fails on new data
- **Data leakage** - Including future information in predictions
- **Ignoring business context** - Technically good but practically useless splits
- **Not validating** - Always test your model on holdout data

### 💡 Workflow Tips
- **Save frequently** - Use Save button to preserve your work
- **Name meaningfully** - Give descriptive names to nodes and models
- **Document changes** - Keep notes on what settings you tried
- **Version control** - Save different versions as you experiment

---

## Getting Help

### Built-in Help
- **Help → About** - Check version information
- **Right-click nodes** - Context menus with relevant options
- **Tooltips** - Hover over buttons for quick descriptions

### Troubleshooting
- **Check logs** - Look in the `logs/` folder for error details
- **Verify data** - Ensure your data loaded correctly
- **Memory issues** - Close other applications for large datasets
- **Display problems** - See setup guides for graphics driver issues

### Support Resources
- **GitHub Issues:** [Report bugs and request features](https://github.com/rupinrd3/bespoke-decision-tree-utility/issues)
- **User Manual:** This document
- **Setup Guides:** Platform-specific installation help

---

**🎉 You're ready to build decision trees!** Start with a simple dataset, follow the steps above, and gradually explore more advanced features as you become comfortable with the interface.