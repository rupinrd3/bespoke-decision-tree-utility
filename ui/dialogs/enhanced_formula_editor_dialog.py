#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Formula Editor Dialog for Bespoke Utility
Advanced interface for creating new variables with syntax checking and preview
"""

import logging
import re
import ast
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt5.QtGui import QFont, QSyntaxHighlighter, QTextCharFormat, QColor, QTextCursor
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QTabWidget, QWidget, QListWidget, QListWidgetItem,
    QSplitter, QProgressBar, QMessageBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QFrame, QSizePolicy, QPlainTextEdit, QTreeWidget, QTreeWidgetItem
)

logger = logging.getLogger(__name__)


class FormulaSyntaxHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for formula expressions"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.highlighting_rules = []
        
        keyword_format = QTextCharFormat()
        keyword_format.setColor(QColor(0, 0, 255))
        keyword_format.setFontWeight(QFont.Bold)
        
        keywords = ['if', 'else', 'elif', 'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None']
        for keyword in keywords:
            pattern = f'\\b{keyword}\\b'
            self.highlighting_rules.append((re.compile(pattern), keyword_format))
            
        function_format = QTextCharFormat()
        function_format.setColor(QColor(128, 0, 128))
        function_format.setFontWeight(QFont.Bold)
        
        functions = ['abs', 'max', 'min', 'sum', 'mean', 'std', 'var', 'len', 'round', 
                    'sqrt', 'log', 'exp', 'sin', 'cos', 'tan', 'upper', 'lower', 'strip']
        for func in functions:
            pattern = f'\\b{func}\\s*\\('
            self.highlighting_rules.append((re.compile(pattern), function_format))
            
        number_format = QTextCharFormat()
        number_format.setColor(QColor(0, 128, 0))
        pattern = r'\\b\\d+\\.?\\d*\\b'
        self.highlighting_rules.append((re.compile(pattern), number_format))
        
        string_format = QTextCharFormat()
        string_format.setColor(QColor(255, 0, 0))
        patterns = [r'"[^"]*"', r"'[^']*'"]
        for pattern in patterns:
            self.highlighting_rules.append((re.compile(pattern), string_format))
            
        operator_format = QTextCharFormat()
        operator_format.setColor(QColor(128, 128, 0))
        operators = ['+', '-', '*', '/', '//', '%', '**', '==', '!=', '<', '>', '<=', '>=']
        for op in operators:
            pattern = f'\\{op}' if op in ['+', '*', '/', '(', ')', '[', ']'] else op
            self.highlighting_rules.append((re.compile(pattern), operator_format))
            
        comment_format = QTextCharFormat()
        comment_format.setColor(QColor(128, 128, 128))
        comment_format.setFontItalic(True)
        pattern = r'#.*'
        self.highlighting_rules.append((re.compile(pattern), comment_format))
        
    def highlightBlock(self, text):
        """Apply syntax highlighting to a block of text"""
        for pattern, format_obj in self.highlighting_rules:
            for match in pattern.finditer(text):
                start, end = match.span()
                self.setFormat(start, end - start, format_obj)


class FormulaValidator:
    """Validates formula expressions"""
    
    def __init__(self, columns: List[str]):
        self.columns = columns
        self.allowed_functions = {
            'abs', 'max', 'min', 'sum', 'mean', 'std', 'var', 'len', 'round',
            'sqrt', 'log', 'exp', 'sin', 'cos', 'tan', 'upper', 'lower', 'strip',
            'int', 'float', 'str', 'bool'
        }
        
    def validate(self, formula: str) -> Tuple[bool, str]:
        """
        Validate a formula expression
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not formula.strip():
            return False, "Formula cannot be empty"
            
        try:
            tree = ast.parse(formula, mode='eval')
            
            danger_check = self._check_dangerous_operations(tree)
            if not danger_check[0]:
                return danger_check
                
            column_check = self._check_column_references(formula)
            if not column_check[0]:
                return column_check
                
            function_check = self._check_functions(tree)
            if not function_check[0]:
                return function_check
                
            return True, "Formula is valid"
            
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
            
    def _check_dangerous_operations(self, tree) -> Tuple[bool, str]:
        """Check for potentially dangerous operations"""
        dangerous_nodes = [
            ast.Import, ast.ImportFrom, ast.Exec, ast.Eval,
            ast.Global, ast.Nonlocal, ast.Delete
        ]
        
        for node in ast.walk(tree):
            if any(isinstance(node, danger) for danger in dangerous_nodes):
                return False, "Dangerous operations not allowed"
                
        return True, ""
        
    def _check_column_references(self, formula: str) -> Tuple[bool, str]:
        """Check if column references are valid"""
        pattern = r'\\b([a-zA-Z_][a-zA-Z0-9_]*)\\b'
        
        formula_no_strings = re.sub(r'"[^"]*"|\'[^\']*\'', '', formula)
        
        matches = re.findall(pattern, formula_no_strings)
        
        for match in matches:
            if (match not in self.allowed_functions and 
                match not in ['True', 'False', 'None', 'if', 'else', 'elif', 'and', 'or', 'not', 'in', 'is'] and
                match not in self.columns):
                return False, f"Unknown column or function: '{match}'"
                
        return True, ""
        
    def _check_functions(self, tree) -> Tuple[bool, str]:
        """Check if functions are allowed"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name not in self.allowed_functions:
                        return False, f"Function '{func_name}' is not allowed"
                        
        return True, ""


class FormulaEvaluator(QThread):
    """Thread for evaluating formulas safely"""
    
    resultReady = pyqtSignal(object, str)  # result, error_message
    
    def __init__(self, formula: str, dataframe: pd.DataFrame, sample_size: int = 100):
        super().__init__()
        self.formula = formula
        self.dataframe = dataframe
        self.sample_size = sample_size
        
    def run(self):
        """Evaluate the formula"""
        try:
            if len(self.dataframe) > self.sample_size:
                sample_df = self.dataframe.sample(n=self.sample_size, random_state=42)
            else:
                sample_df = self.dataframe.copy()
                
            context = self._create_context(sample_df)
            
            result = eval(self.formula, {"__builtins__": {}}, context)
            
            if not isinstance(result, pd.Series):
                if np.isscalar(result):
                    result = pd.Series([result] * len(sample_df), index=sample_df.index)
                else:
                    result = pd.Series(result, index=sample_df.index)
                    
            self.resultReady.emit(result, "")
            
        except Exception as e:
            self.resultReady.emit(None, str(e))
            
    def _create_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create evaluation context with columns and safe functions"""
        context = {}
        
        for col in df.columns:
            context[col] = df[col]
            
        safe_functions = {
            'abs': abs,
            'max': max,
            'min': min,
            'sum': sum,
            'len': len,
            'round': round,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
            'sqrt': np.sqrt,
            'log': np.log,
            'exp': np.exp,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
        }
        
        safe_functions.update({
            'mean': lambda x: x.mean() if hasattr(x, 'mean') else np.mean(x),
            'std': lambda x: x.std() if hasattr(x, 'std') else np.std(x),
            'var': lambda x: x.var() if hasattr(x, 'var') else np.var(x),
            'upper': lambda x: x.str.upper() if hasattr(x, 'str') else str(x).upper(),
            'lower': lambda x: x.str.lower() if hasattr(x, 'str') else str(x).lower(),
            'strip': lambda x: x.str.strip() if hasattr(x, 'str') else str(x).strip(),
        })
        
        context.update(safe_functions)
        
        return context


class EnhancedFormulaEditorDialog(QDialog):
    """Enhanced dialog for creating and editing variable formulas"""
    
    def __init__(self, dataframe: pd.DataFrame, variable_name: str = "", 
                 initial_formula: str = "", parent=None):
        super().__init__(parent)
        
        self.dataframe = dataframe
        self.variable_name = variable_name
        self.initial_formula = initial_formula
        
        self.final_formula = ""
        self.final_variable_name = ""
        self.preview_result = None
        
        self.validator = FormulaValidator(list(dataframe.columns))
        self.live_preview_timer = QTimer()
        self.live_preview_timer.setSingleShot(True)
        self.live_preview_timer.timeout.connect(self.updateLivePreview)
        self.live_preview_enabled = True
        
        self.setWindowTitle("Enhanced Formula Editor")
        self.setModal(True)
        self.resize(1000, 700)
        
        self.setupUI()
        
        if variable_name:
            self.variable_name_edit.setText(variable_name)
        if initial_formula:
            self.formula_editor.setPlainText(initial_formula)
            
    def setupUI(self):
        """Setup the main UI"""
        layout = QVBoxLayout()
        
        header_label = QLabel("Enhanced Formula Editor")
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header_label)
        
        config_group = QGroupBox("Variable Configuration")
        config_layout = QFormLayout()
        
        self.variable_name_edit = QLineEdit()
        self.variable_name_edit.setPlaceholderText("Enter variable name (e.g., Income_Ratio)")
        config_layout.addRow("Variable Name:", self.variable_name_edit)
        
        self.variable_type_combo = QComboBox()
        self.variable_type_combo.addItems(['Auto-detect', 'Numeric', 'Text', 'Boolean', 'Date'])
        config_layout.addRow("Data Type:", self.variable_type_combo)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        splitter = QSplitter(Qt.Horizontal)
        
        left_panel = self.createFormulaPanel()
        splitter.addWidget(left_panel)
        
        right_panel = self.createPreviewPanel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([600, 400])
        
        layout.addWidget(splitter)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; }")
        layout.addWidget(self.status_label)
        
        button_layout = QHBoxLayout()
        
        self.validate_button = QPushButton("Validate Formula")
        self.validate_button.clicked.connect(self.validateFormula)
        
        self.preview_button = QPushButton("Preview Result")
        self.preview_button.clicked.connect(self.previewFormula)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clearFormula)
        
        button_layout.addWidget(self.validate_button)
        button_layout.addWidget(self.preview_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.acceptFormula)
        self.ok_button.setEnabled(False)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def createFormulaPanel(self) -> QWidget:
        """Create the formula editing panel"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        editor_group = QGroupBox("Formula Editor")
        editor_layout = QVBoxLayout()
        
        toolbar_layout = QHBoxLayout()
        
        self.syntax_checkbox = QCheckBox("Syntax Highlighting")
        self.syntax_checkbox.setChecked(True)
        self.syntax_checkbox.toggled.connect(self.toggleSyntaxHighlighting)
        toolbar_layout.addWidget(self.syntax_checkbox)
        
        toolbar_layout.addStretch()
        
        font_size_label = QLabel("Font Size:")
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 24)
        self.font_size_spin.setValue(12)
        self.font_size_spin.valueChanged.connect(self.updateEditorFont)
        
        toolbar_layout.addWidget(font_size_label)
        toolbar_layout.addWidget(self.font_size_spin)
        
        editor_layout.addLayout(toolbar_layout)
        
        self.formula_editor = QPlainTextEdit()
        self.formula_editor.setPlaceholderText("Enter your formula here...\\nExample: Income / Age\\nExample: if(Credit_Score > 700, 'Good', 'Bad')")
        self.formula_editor.setFont(QFont("Courier", 12))
        
        self.highlighter = FormulaSyntaxHighlighter(self.formula_editor.document())
        
        self.validation_timer = QTimer()
        self.validation_timer.setSingleShot(True)
        self.validation_timer.timeout.connect(self.autoValidate)
        
        self.formula_editor.textChanged.connect(self.onFormulaChanged)
        
        editor_layout.addWidget(self.formula_editor)
        
        editor_group.setLayout(editor_layout)
        layout.addWidget(editor_group)
        
        helper_tabs = QTabWidget()
        
        columns_tab = self.createColumnsTab()
        helper_tabs.addTab(columns_tab, "Columns")
        
        functions_tab = self.createFunctionsTab()
        helper_tabs.addTab(functions_tab, "Functions")
        
        examples_tab = self.createExamplesTab()
        helper_tabs.addTab(examples_tab, "Examples")
        
        layout.addWidget(helper_tabs)
        
        widget.setLayout(layout)
        return widget
        
    def createPreviewPanel(self) -> QWidget:
        """Create the preview and validation panel"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        controls_group = QGroupBox("Live Preview Controls")
        controls_layout = QHBoxLayout()
        
        self.live_preview_checkbox = QCheckBox("Enable Live Preview")
        self.live_preview_checkbox.setChecked(True)
        self.live_preview_checkbox.toggled.connect(self.toggleLivePreview)
        controls_layout.addWidget(self.live_preview_checkbox)
        
        self.preview_rows_spin = QSpinBox()
        self.preview_rows_spin.setRange(5, 100)
        self.preview_rows_spin.setValue(20)
        self.preview_rows_spin.valueChanged.connect(self.updatePreviewRows)
        controls_layout.addWidget(QLabel("Preview Rows:"))
        controls_layout.addWidget(self.preview_rows_spin)
        
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        validation_group = QGroupBox("Validation & Live Preview")
        validation_layout = QVBoxLayout()
        
        self.validation_result = QLabel("Enter a formula to validate")
        self.validation_result.setWordWrap(True)
        self.validation_result.setStyleSheet("QLabel { padding: 10px; background-color: #f8f8f8; }")
        validation_layout.addWidget(self.validation_result)
        
        self.live_status_label = QLabel("Live preview: Waiting for input...")
        self.live_status_label.setStyleSheet("QLabel { padding: 5px; background-color: #e8f4f8; color: #2c5aa0; }")
        validation_layout.addWidget(self.live_status_label)
        
        validation_group.setLayout(validation_layout)
        layout.addWidget(validation_group)
        
        preview_group = QGroupBox("Preview (First 20 rows)")
        preview_layout = QVBoxLayout()
        
        self.preview_table = QTableWidget()
        self.preview_table.setMaximumHeight(200)
        self.preview_table.setAlternatingRowColors(True)
        preview_layout.addWidget(self.preview_table)
        
        self.preview_info = QLabel("")
        preview_layout.addWidget(self.preview_info)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout()
        
        self.stats_type_label = QLabel("-")
        self.stats_unique_label = QLabel("-")
        self.stats_null_label = QLabel("-")
        self.stats_min_label = QLabel("-")
        self.stats_max_label = QLabel("-")
        self.stats_mean_label = QLabel("-")
        
        stats_layout.addRow("Data Type:", self.stats_type_label)
        stats_layout.addRow("Unique Values:", self.stats_unique_label)
        stats_layout.addRow("Null Values:", self.stats_null_label)
        stats_layout.addRow("Min Value:", self.stats_min_label)
        stats_layout.addRow("Max Value:", self.stats_max_label)
        stats_layout.addRow("Mean Value:", self.stats_mean_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        widget.setLayout(layout)
        return widget
        
    def createColumnsTab(self) -> QWidget:
        """Create the columns helper tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        self.column_search = QLineEdit()
        self.column_search.setPlaceholderText("Filter columns...")
        self.column_search.textChanged.connect(self.filterColumns)
        
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.column_search)
        layout.addLayout(search_layout)
        
        self.columns_list = QTreeWidget()
        self.columns_list.setHeaderLabels(['Column', 'Type', 'Sample'])
        self.columns_list.itemDoubleClicked.connect(self.insertColumn)
        
        for col in self.dataframe.columns:
            item = QTreeWidgetItem()
            item.setText(0, col)
            item.setText(1, str(self.dataframe[col].dtype))
            
            sample_values = self.dataframe[col].dropna().head(3).tolist()
            sample_text = ', '.join([str(v)[:20] for v in sample_values])
            item.setText(2, sample_text)
            
            self.columns_list.addTopLevelItem(item)
            
        self.columns_list.resizeColumnToContents(0)
        layout.addWidget(self.columns_list)
        
        instructions = QLabel("Double-click a column to insert it into the formula")
        instructions.setStyleSheet("QLabel { font-style: italic; color: gray; }")
        layout.addWidget(instructions)
        
        widget.setLayout(layout)
        return widget
        
    def createFunctionsTab(self) -> QWidget:
        """Create the functions helper tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        functions_data = {
            'Mathematical': [
                ('abs(x)', 'Absolute value'),
                ('sqrt(x)', 'Square root'),
                ('log(x)', 'Natural logarithm'),
                ('exp(x)', 'Exponential function'),
                ('round(x, n)', 'Round to n decimal places'),
                ('max(a, b, ...)', 'Maximum value'),
                ('min(a, b, ...)', 'Minimum value'),
            ],
            'Statistical': [
                ('mean(series)', 'Average value'),
                ('std(series)', 'Standard deviation'),
                ('var(series)', 'Variance'),
                ('sum(series)', 'Sum of values'),
                ('len(series)', 'Count of values'),
            ],
            'String': [
                ('upper(text)', 'Convert to uppercase'),
                ('lower(text)', 'Convert to lowercase'),
                ('strip(text)', 'Remove whitespace'),
                ('str(value)', 'Convert to string'),
            ],
            'Logical': [
                ('if(condition, true_val, false_val)', 'Conditional expression'),
                ('and', 'Logical AND'),
                ('or', 'Logical OR'),
                ('not', 'Logical NOT'),
            ],
            'Type Conversion': [
                ('int(x)', 'Convert to integer'),
                ('float(x)', 'Convert to float'),
                ('bool(x)', 'Convert to boolean'),
            ]
        }
        
        self.functions_tree = QTreeWidget()
        self.functions_tree.setHeaderLabels(['Function', 'Description'])
        self.functions_tree.itemDoubleClicked.connect(self.insertFunction)
        
        for category, functions in functions_data.items():
            category_item = QTreeWidgetItem()
            category_item.setText(0, category)
            category_item.setFont(0, QFont("Arial", 10, QFont.Bold))
            
            for func, desc in functions:
                func_item = QTreeWidgetItem()
                func_item.setText(0, func)
                func_item.setText(1, desc)
                category_item.addChild(func_item)
                
            self.functions_tree.addTopLevelItem(category_item)
            
        self.functions_tree.expandAll()
        self.functions_tree.resizeColumnToContents(0)
        layout.addWidget(self.functions_tree)
        
        instructions = QLabel("Double-click a function to insert it into the formula")
        instructions.setStyleSheet("QLabel { font-style: italic; color: gray; }")
        layout.addWidget(instructions)
        
        widget.setLayout(layout)
        return widget
        
    def createExamplesTab(self) -> QWidget:
        """Create the examples helper tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        examples_data = [
            ('Simple Ratio', 'Income / Age', 'Create a ratio of two numeric columns'),
            ('Conditional Category', "if(Credit_Score > 700, 'Good', 'Bad')", 'Create categories based on conditions'),
            ('Multiple Conditions', "if(Age < 30, 'Young', if(Age < 60, 'Middle', 'Senior'))", 'Nested conditions'),
            ('Log Transformation', 'log(Income + 1)', 'Apply logarithmic transformation'),
            ('Standardization', '(Income - mean(Income)) / std(Income)', 'Standardize a variable'),
            ('String Concatenation', "upper(First_Name) + ' ' + upper(Last_Name)", 'Combine text columns'),
            ('Date Calculation', 'Current_Date - Birth_Date', 'Calculate differences between dates'),
            ('Boolean Logic', '(Age > 25) and (Income > 50000)', 'Create boolean variables'),
            ('Binning', "if(Income < 30000, 'Low', if(Income < 70000, 'Medium', 'High'))", 'Create income bins'),
            ('Missing Value Handling', "if(Income.isnull(), mean(Income), Income)", 'Handle missing values'),
        ]
        
        self.examples_list = QTreeWidget()
        self.examples_list.setHeaderLabels(['Name', 'Formula', 'Description'])
        self.examples_list.itemDoubleClicked.connect(self.insertExample)
        
        for name, formula, desc in examples_data:
            item = QTreeWidgetItem()
            item.setText(0, name)
            item.setText(1, formula)
            item.setText(2, desc)
            self.examples_list.addTopLevelItem(item)
            
        self.examples_list.resizeColumnToContents(0)
        self.examples_list.resizeColumnToContents(1)
        layout.addWidget(self.examples_list)
        
        instructions = QLabel("Double-click an example to insert it into the formula")
        instructions.setStyleSheet("QLabel { font-style: italic; color: gray; }")
        layout.addWidget(instructions)
        
        widget.setLayout(layout)
        return widget
        
    def filterColumns(self):
        """Filter columns based on search text"""
        search_text = self.column_search.text().lower()
        
        for i in range(self.columns_list.topLevelItemCount()):
            item = self.columns_list.topLevelItem(i)
            column_name = item.text(0).lower()
            item.setHidden(search_text not in column_name)
            
    def insertColumn(self, item):
        """Insert column name into formula"""
        if item.text(0):  # Make sure it's a column item, not a category
            column_name = item.text(0)
            self.insertTextAtCursor(column_name)
            
    def insertFunction(self, item):
        """Insert function into formula"""
        if item.parent() is not None:  # Make sure it's a function item, not a category
            function_text = item.text(0)
            self.insertTextAtCursor(function_text)
            
    def insertExample(self, item):
        """Insert example formula"""
        formula = item.text(1)
        self.formula_editor.setPlainText(formula)
        
    def insertTextAtCursor(self, text: str):
        """Insert text at current cursor position"""
        cursor = self.formula_editor.textCursor()
        cursor.insertText(text)
        self.formula_editor.setFocus()
        
    def toggleSyntaxHighlighting(self, enabled: bool):
        """Toggle syntax highlighting"""
        if enabled:
            self.highlighter.setDocument(self.formula_editor.document())
        else:
            self.highlighter.setDocument(None)
            
    def updateEditorFont(self):
        """Update editor font size"""
        font = QFont("Courier", self.font_size_spin.value())
        self.formula_editor.setFont(font)
        
    def autoValidate(self):
        """Automatically validate formula after typing"""
        self.validateFormula(show_success=False)
        
    def validateFormula(self, show_success: bool = True):
        """Validate the current formula"""
        formula = self.formula_editor.toPlainText().strip()
        
        if not formula:
            self.validation_result.setText("Enter a formula to validate")
            self.validation_result.setStyleSheet("QLabel { padding: 10px; background-color: #f8f8f8; }")
            self.ok_button.setEnabled(False)
            return
            
        is_valid, message = self.validator.validate(formula)
        
        if is_valid:
            if show_success:
                self.validation_result.setText("✓ Formula is valid")
                self.validation_result.setStyleSheet("QLabel { padding: 10px; background-color: #d4edda; color: #155724; }")
            self.ok_button.setEnabled(True)
            self.status_label.setText("Formula validated successfully")
        else:
            self.validation_result.setText(f"✗ {message}")
            self.validation_result.setStyleSheet("QLabel { padding: 10px; background-color: #f8d7da; color: #721c24; }")
            self.ok_button.setEnabled(False)
            self.status_label.setText(f"Validation error: {message}")
            
    def previewFormula(self):
        """Preview the formula result"""
        formula = self.formula_editor.toPlainText().strip()
        
        if not formula:
            QMessageBox.warning(self, "No Formula", "Please enter a formula first.")
            return
            
        is_valid, message = self.validator.validate(formula)
        if not is_valid:
            QMessageBox.warning(self, "Invalid Formula", f"Formula validation failed: {message}")
            return
            
        self.status_label.setText("Evaluating formula...")
        
        self.evaluator = FormulaEvaluator(formula, self.dataframe, sample_size=100)
        self.evaluator.resultReady.connect(self.onPreviewReady)
        self.evaluator.start()
        
    def onPreviewReady(self, result, error_message: str):
        """Handle preview result"""
        if error_message:
            self.status_label.setText(f"Preview error: {error_message}")
            self.preview_info.setText(f"Error: {error_message}")
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
            return
            
        if result is None:
            return
            
        self.preview_result = result
        self.status_label.setText("Preview completed successfully")
        
        self.updatePreviewTable(result)
        
        self.updateStatistics(result)
        
    def updatePreviewTable(self, result: pd.Series):
        """Update the preview table"""
        preview_data = result.head(20)
        
        self.preview_table.setRowCount(len(preview_data))
        self.preview_table.setColumnCount(2)
        self.preview_table.setHorizontalHeaderLabels(['Index', 'Value'])
        
        for i, (idx, value) in enumerate(preview_data.items()):
            index_item = QTableWidgetItem(str(idx))
            self.preview_table.setItem(i, 0, index_item)
            
            if pd.isna(value):
                value_text = "<NA>"
            else:
                value_text = str(value)
            value_item = QTableWidgetItem(value_text)
            if pd.isna(value):
                value_item.setBackground(Qt.lightGray)
            self.preview_table.setItem(i, 1, value_item)
            
        self.preview_table.resizeColumnsToContents()
        
        total_rows = len(result)
        preview_rows = len(preview_data)
        self.preview_info.setText(f"Showing {preview_rows} of {total_rows} rows")
        
    def updateStatistics(self, result: pd.Series):
        """Update statistics panel"""
        try:
            dtype = str(result.dtype)
            self.stats_type_label.setText(dtype)
            
            unique_count = result.nunique()
            self.stats_unique_label.setText(str(unique_count))
            
            null_count = result.isnull().sum()
            null_pct = (null_count / len(result)) * 100
            self.stats_null_label.setText(f"{null_count} ({null_pct:.1f}%)")
            
            if pd.api.types.is_numeric_dtype(result):
                self.stats_min_label.setText(f"{result.min():.4f}")
                self.stats_max_label.setText(f"{result.max():.4f}")
                self.stats_mean_label.setText(f"{result.mean():.4f}")
            else:
                self.stats_min_label.setText("-")
                self.stats_max_label.setText("-")
                self.stats_mean_label.setText("-")
                
        except Exception as e:
            logger.warning(f"Error updating statistics: {e}")
            
    def clearFormula(self):
        """Clear the formula editor"""
        self.formula_editor.clear()
        self.validation_result.setText("Enter a formula to validate")
        self.validation_result.setStyleSheet("QLabel { padding: 10px; background-color: #f8f8f8; }")
        self.preview_table.setRowCount(0)
        self.preview_table.setColumnCount(0)
        self.preview_info.setText("")
        self.ok_button.setEnabled(False)
        
    def acceptFormula(self):
        """Accept the formula and close dialog"""
        variable_name = self.variable_name_edit.text().strip()
        if not variable_name:
            QMessageBox.warning(self, "No Variable Name", "Please enter a variable name.")
            return
            
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', variable_name):
            QMessageBox.warning(self, "Invalid Variable Name", 
                              "Variable name must start with a letter or underscore and contain only letters, numbers, and underscores.")
            return
            
        if variable_name in self.dataframe.columns:
            reply = QMessageBox.question(self, "Variable Exists", 
                                       f"Variable '{variable_name}' already exists. Do you want to replace it?",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
                
        formula = self.formula_editor.toPlainText().strip()
        is_valid, message = self.validator.validate(formula)
        
        if not is_valid:
            QMessageBox.warning(self, "Invalid Formula", f"Formula validation failed: {message}")
            return
            
        self.final_formula = formula
        self.final_variable_name = variable_name
        
        self.accept()
        
    def getResults(self) -> Tuple[str, str]:
        """Get the final variable name and formula"""
        return self.final_variable_name, self.final_formula
        
    def onFormulaChanged(self):
        """Handle formula text changes"""
        self.validation_timer.start(1000)
        
        if self.live_preview_enabled:
            self.live_preview_timer.start(1500)  # Slight delay for live preview
            
    def toggleLivePreview(self, enabled: bool):
        """Toggle live preview on/off"""
        self.live_preview_enabled = enabled
        if enabled:
            self.updateLivePreview()
        else:
            self.live_status_label.setText("Live preview: Disabled")
            self.clearPreviewResults()
            
    def updatePreviewRows(self, rows: int):
        """Update number of preview rows"""
        if self.live_preview_enabled:
            self.updateLivePreview()
            
    def updateLivePreview(self):
        """Update live preview of formula results"""
        if not self.live_preview_enabled:
            return
            
        formula = self.formula_editor.toPlainText().strip()
        if not formula:
            self.live_status_label.setText("Live preview: Waiting for input...")
            self.clearPreviewResults()
            return
            
        self.live_status_label.setText("Live preview: Evaluating...")
        
        is_valid, message = self.validator.validate(formula)
        if not is_valid:
            self.live_status_label.setText(f"Live preview: Error - {message}")
            self.clearPreviewResults()
            return
            
        try:
            sample_size = min(self.preview_rows_spin.value(), len(self.dataframe))
            sample_df = self.dataframe.head(sample_size).copy()
            
            safe_dict = {
                'abs': abs, 'max': max, 'min': min, 'sum': sum, 'round': round,
                'len': len, 'int': int, 'float': float, 'str': str, 'bool': bool,
                'sqrt': np.sqrt, 'log': np.log, 'exp': np.exp,
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'upper': str.upper, 'lower': str.lower, 'strip': str.strip,
                'np': np, 'pd': pd
            }
            
            for col in sample_df.columns:
                safe_dict[col] = sample_df[col]
                
            result = eval(formula, {"__builtins__": {}}, safe_dict)
            
            if not isinstance(result, pd.Series):
                if hasattr(result, '__len__') and len(result) == len(sample_df):
                    result = pd.Series(result, index=sample_df.index)
                else:
                    result = pd.Series([result] * len(sample_df), index=sample_df.index)
                    
            self.updatePreviewTable(result, sample_df)
            self.updatePreviewStats(result)
            self.live_status_label.setText(f"Live preview: Success ({len(result)} values)")
            
        except Exception as e:
            self.live_status_label.setText(f"Live preview: Error - {str(e)}")
            self.clearPreviewResults()
            
    def updatePreviewTable(self, result: pd.Series, original_df: pd.DataFrame):
        """Update the preview table with results"""
        display_cols = ['Row'] + list(original_df.columns)[:3] + ['New_Variable']  # Show first 3 original columns
        
        self.preview_table.setColumnCount(len(display_cols))
        self.preview_table.setHorizontalHeaderLabels(display_cols)
        self.preview_table.setRowCount(len(result))
        
        for i in range(len(result)):
            self.preview_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            
            for j, col in enumerate(list(original_df.columns)[:3]):
                value = str(original_df.iloc[i, original_df.columns.get_loc(col)])
                self.preview_table.setItem(i, j + 1, QTableWidgetItem(value))
                
            result_value = str(result.iloc[i])
            item = QTableWidgetItem(result_value)
            item.setBackground(QColor(230, 255, 230))  # Light green background
            self.preview_table.setItem(i, len(display_cols) - 1, item)
            
        self.preview_table.resizeColumnsToContents()
        
    def updatePreviewStats(self, result: pd.Series):
        """Update preview statistics"""
        try:
            dtype_str = str(result.dtype)
            self.stats_type_label.setText(dtype_str)
            
            unique_count = result.nunique()
            self.stats_unique_label.setText(str(unique_count))
            
            null_count = result.isnull().sum()
            self.stats_null_label.setText(str(null_count))
            
            if pd.api.types.is_numeric_dtype(result):
                self.stats_min_label.setText(f"{result.min():.4f}")
                self.stats_max_label.setText(f"{result.max():.4f}")
                self.stats_mean_label.setText(f"{result.mean():.4f}")
            else:
                self.stats_min_label.setText("N/A")
                self.stats_max_label.setText("N/A") 
                self.stats_mean_label.setText("N/A")
                
        except Exception as e:
            logger.warning(f"Error updating preview stats: {e}")
            
    def clearPreviewResults(self):
        """Clear preview table and statistics"""
        self.preview_table.setRowCount(0)
        self.preview_table.setColumnCount(0)
        
        self.stats_type_label.setText("-")
        self.stats_unique_label.setText("-")
        self.stats_null_label.setText("-")
        self.stats_min_label.setText("-")
        self.stats_max_label.setText("-")
        self.stats_mean_label.setText("-")
        
        self.preview_info.setText("")