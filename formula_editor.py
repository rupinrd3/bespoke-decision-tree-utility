#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Formula Editor Module for Bespoke Utility
Provides UI for creating new variables through formulas
"""

import logging
import re
from typing import Dict, List, Tuple, Optional, Any, Set, Callable

from PyQt5.QtCore import Qt, pyqtSignal, QRect, QSize
from PyQt5.QtGui import (QFont, QSyntaxHighlighter, QTextCharFormat, 
                        QColor, QPalette, QFontMetrics)
from PyQt5.QtWidgets import (QWidget, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
                           QLabel, QLineEdit, QTextEdit, QPushButton, QComboBox,
                           QListWidget, QListWidgetItem, QTabWidget, QSplitter,
                           QMessageBox, QToolButton, QSpinBox, QFrame,
                           QGroupBox, QRadioButton, QCheckBox, QMenu, QAction,
                           QToolBar)

from data.feature_engineering import FormulaParser

logger = logging.getLogger(__name__)

class FormulaHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for the formula editor"""
    
    def __init__(self, document):
        """
        Initialize the highlighter
        
        Args:
            document: Text document to highlight
        """
        super().__init__(document)
        
        self.highlighting_rules = []
        
        function_format = QTextCharFormat()
        function_format.setForeground(QColor(0, 0, 180))
        function_format.setFontWeight(QFont.Bold)
        self.highlighting_rules.append((
            r'\\b[A-Z]+\\(', function_format
        ))
        
        variable_format = QTextCharFormat()
        variable_format.setForeground(QColor(0, 120, 120))
        self.highlighting_rules.append((
            r'\\b[A-Za-z][A-Za-z0-9_]*\\b', variable_format
        ))
        
        number_format = QTextCharFormat()
        number_format.setForeground(QColor(120, 0, 0))
        self.highlighting_rules.append((
            r'\\b[-+]?[0-9]*\\.?[0-9]+\\b', number_format
        ))
        
        operator_format = QTextCharFormat()
        operator_format.setForeground(QColor(120, 0, 120))
        operator_format.setFontWeight(QFont.Bold)
        self.highlighting_rules.append((
            r'[+\\-*/\\^%=<>!]', operator_format
        ))
        
        parentheses_format = QTextCharFormat()
        parentheses_format.setForeground(QColor(0, 0, 0))
        parentheses_format.setFontWeight(QFont.Bold)
        self.highlighting_rules.append((
            r'[\\(\\)]', parentheses_format
        ))
        
        self.highlighting_rules = [(re.compile(pattern), format) 
                                 for pattern, format in self.highlighting_rules]
    
    def highlightBlock(self, text):
        """
        Apply highlighting rules to a block of text
        
        Args:
            text: Text to highlight
        """
        for pattern, format in self.highlighting_rules:
            for match in pattern.finditer(text):
                self.setFormat(match.start(), match.length(), format)

class FormulaEditor(QWidget):
    """Widget for editing formulas and creating new variables"""
    
    formulaApplied = pyqtSignal(str, str, str)  # formula, variable_name, variable_type
    
    def __init__(self, available_columns=None, parent=None):
        """
        Initialize the formula editor
        
        Args:
            available_columns: List of available column names
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.available_columns = available_columns or []
        self.formula_parser = FormulaParser()
        
        self.init_ui()
        
        self.connect_signals()
    
    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout()
        
        formula_group = QGroupBox("Formula Definition")
        formula_layout = QVBoxLayout()
        
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Variable Name:"))
        self.var_name_edit = QLineEdit()
        self.var_name_edit.setPlaceholderText("Enter new variable name")
        name_layout.addWidget(self.var_name_edit)
        
        name_layout.addWidget(QLabel("Type:"))
        self.var_type_combo = QComboBox()
        self.var_type_combo.addItems(["float", "int", "str", "category"])
        name_layout.addWidget(self.var_type_combo)
        
        formula_layout.addLayout(name_layout)
        
        formula_layout.addWidget(QLabel("Formula:"))
        self.formula_edit = QTextEdit()
        self.formula_edit.setPlaceholderText("Enter formula (e.g., Income / 12)")
        self.formula_highlighter = FormulaHighlighter(self.formula_edit.document())
        formula_layout.addWidget(self.formula_edit)
        
        self.status_label = QLabel("")
        formula_layout.addWidget(self.status_label)
        
        formula_group.setLayout(formula_layout)
        main_layout.addWidget(formula_group)
        
        helper_splitter = QSplitter(Qt.Horizontal)
        
        function_group = QGroupBox("Functions")
        function_layout = QVBoxLayout()
        
        self.function_list = QListWidget()
        self.populate_function_list()
        function_layout.addWidget(self.function_list)
        
        self.function_help = QLabel("Select a function to see its description")
        self.function_help.setWordWrap(True)
        function_layout.addWidget(self.function_help)
        
        function_group.setLayout(function_layout)
        helper_splitter.addWidget(function_group)
        
        column_group = QGroupBox("Available Columns")
        column_layout = QVBoxLayout()
        
        self.column_filter = QLineEdit()
        self.column_filter.setPlaceholderText("Filter columns...")
        column_layout.addWidget(self.column_filter)
        
        self.column_list = QListWidget()
        self.populate_column_list()
        column_layout.addWidget(self.column_list)
        
        column_group.setLayout(column_layout)
        helper_splitter.addWidget(column_group)
        
        helper_splitter.setSizes([200, 200])
        
        main_layout.addWidget(helper_splitter)
        
        button_layout = QHBoxLayout()
        self.validate_btn = QPushButton("Validate Formula")
        self.apply_btn = QPushButton("Apply Formula")
        self.apply_btn.setEnabled(False)  # Disabled until validation
        
        button_layout.addWidget(self.validate_btn)
        button_layout.addWidget(self.apply_btn)
        
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
    
    def connect_signals(self):
        """Connect signals to slots"""
        self.function_list.itemClicked.connect(self.function_selected)
        self.function_list.itemDoubleClicked.connect(self.insert_function)
        
        self.column_list.itemDoubleClicked.connect(self.insert_column)
        self.column_filter.textChanged.connect(self.filter_columns)
        
        self.validate_btn.clicked.connect(self.validate_formula)
        self.apply_btn.clicked.connect(self.apply_formula)
        
        self.formula_edit.textChanged.connect(self.on_formula_changed)
        self.var_name_edit.textChanged.connect(self.check_inputs)
    
    def populate_function_list(self):
        """Populate the function list with available functions"""
        functions = self.formula_parser.functions.keys()
        
        self.function_list.clear()
        for func in sorted(functions):
            item = QListWidgetItem(func)
            item.setToolTip(f"Double-click to insert {func}()")
            self.function_list.addItem(item)
    
    def populate_column_list(self):
        """Populate the column list with available columns"""
        self.column_list.clear()
        
        for column in sorted(self.available_columns):
            item = QListWidgetItem(column)
            item.setToolTip(f"Double-click to insert '{column}'")
            self.column_list.addItem(item)
    
    def set_available_columns(self, columns: List[str]):
        """
        Set the available columns
        
        Args:
            columns: List of column names
        """
        self.available_columns = columns
        self.populate_column_list()
    
    def function_selected(self, item):
        """
        Display help for the selected function
        
        Args:
            item: Selected list item
        """
        function_name = item.text()
        
        descriptions = {
            'LOG': "LOG(x): Base-10 logarithm of x",
            'LN': "LN(x): Natural logarithm of x",
            'EXP': "EXP(x): e raised to the power x",
            'SQRT': "SQRT(x): Square root of x",
            'ABS': "ABS(x): Absolute value of x",
            'ROUND': "ROUND(x, n): Round x to n decimal places",
            'FLOOR': "FLOOR(x): Round x down to the nearest integer",
            'CEIL': "CEIL(x): Round x up to the nearest integer",
            'SIN': "SIN(x): Sine of x (in radians)",
            'COS': "COS(x): Cosine of x (in radians)",
            'TAN': "TAN(x): Tangent of x (in radians)",
            'MIN': "MIN(x, y, ...): Minimum of the arguments",
            'MAX': "MAX(x, y, ...): Maximum of the arguments",
            'MEAN': "MEAN(x, y, ...): Arithmetic mean of the arguments",
            'SUM': "SUM(x, y, ...): Sum of the arguments",
            'IF': "IF(condition, true_value, false_value): Returns true_value if condition is true, otherwise false_value"
        }
        
        self.function_help.setText(descriptions.get(function_name, f"{function_name}(): No description available"))
    
    def insert_function(self, item):
        """
        Insert a function into the formula editor
        
        Args:
            item: List item containing function name
        """
        function_name = item.text()
        
        cursor = self.formula_edit.textCursor()
        cursor.insertText(f"{function_name}()")
        
        cursor.movePosition(cursor.Left, cursor.MoveAnchor, 1)
        self.formula_edit.setTextCursor(cursor)
        
        self.formula_edit.setFocus()
    
    def insert_column(self, item):
        """
        Insert a column name into the formula editor
        
        Args:
            item: List item containing column name
        """
        column_name = item.text()
        
        cursor = self.formula_edit.textCursor()
        cursor.insertText(column_name)
        
        self.formula_edit.setFocus()
    
    def filter_columns(self, text):
        """
        Filter the column list based on input text
        
        Args:
            text: Filter text
        """
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            if text.lower() in item.text().lower():
                item.setHidden(False)
            else:
                item.setHidden(True)
    
    def on_formula_changed(self):
        """Handle formula text changes"""
        self.status_label.setText("")
        self.apply_btn.setEnabled(False)
    
    def check_inputs(self):
        """Check if inputs are valid"""
        var_name = self.var_name_edit.text().strip()
        
        if not var_name:
            self.apply_btn.setEnabled(False)
            return
        
        if not var_name.isidentifier():
            self.status_label.setText("Invalid variable name (use letters, numbers, underscore; must start with letter)")
            self.status_label.setStyleSheet("color: red")
            self.apply_btn.setEnabled(False)
            return
        
        if var_name in self.available_columns:
            self.status_label.setText(f"Warning: Variable '{var_name}' already exists and will be overwritten")
            self.status_label.setStyleSheet("color: orange")
        
        if getattr(self, 'formula_is_valid', False):
            self.apply_btn.setEnabled(True)
    
    def validate_formula(self):
        """Validate the formula"""
        formula = self.formula_edit.toPlainText().strip()
        
        if not formula:
            self.status_label.setText("Formula cannot be empty")
            self.status_label.setStyleSheet("color: red")
            self.formula_is_valid = False
            self.apply_btn.setEnabled(False)
            return
        
        valid, error_msg = self.formula_parser.validate_syntax(formula)
        
        if valid:
            self.status_label.setText("Formula is valid!")
            self.status_label.setStyleSheet("color: green")
            self.formula_is_valid = True
            
            if self.var_name_edit.text().strip():
                self.apply_btn.setEnabled(True)
        else:
            self.status_label.setText(f"Invalid formula: {error_msg}")
            self.status_label.setStyleSheet("color: red")
            self.formula_is_valid = False
            self.apply_btn.setEnabled(False)
    
    def apply_formula(self):
        """Apply the formula to create a new variable"""
        formula = self.formula_edit.toPlainText().strip()
        var_name = self.var_name_edit.text().strip()
        var_type = self.var_type_combo.currentText()
        
        valid, error_msg = self.formula_parser.validate_syntax(formula)
        
        if not valid:
            QMessageBox.warning(self, "Invalid Formula", 
                              f"Cannot apply invalid formula: {error_msg}")
            return
        
        self.formulaApplied.emit(formula, var_name, var_type)
        
        self.status_label.setText(f"Variable '{var_name}' created successfully!")
        self.status_label.setStyleSheet("color: green")

class FormulaHistoryItem:
    """Class representing a formula history item"""
    
    def __init__(self, formula, var_name, var_type, timestamp=None):
        """
        Initialize a formula history item
        
        Args:
            formula: Formula expression
            var_name: Variable name
            var_type: Variable type
            timestamp: Creation timestamp
        """
        import datetime
        
        self.formula = formula
        self.var_name = var_name
        self.var_type = var_type
        self.timestamp = timestamp or datetime.datetime.now()
    
    def __str__(self):
        """String representation"""
        return f"{self.var_name} = {self.formula} ({self.var_type})"

class FormulaHistoryWidget(QListWidget):
    """Widget for displaying formula history"""
    
    formulaSelected = pyqtSignal(str, str, str)  # formula, variable_name, variable_type
    
    def __init__(self, parent=None):
        """
        Initialize the formula history widget
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.history = []
        
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
        self.itemDoubleClicked.connect(self.on_item_double_clicked)
    
    def add_formula(self, formula, var_name, var_type):
        """
        Add a formula to the history
        
        Args:
            formula: Formula expression
            var_name: Variable name
            var_type: Variable type
        """
        history_item = FormulaHistoryItem(formula, var_name, var_type)
        
        self.history.append(history_item)
        
        item = QListWidgetItem(str(history_item))
        item.setToolTip(f"Double-click to reuse this formula\n{formula}")
        self.insertItem(0, item)  # Add at the top (most recent first)
    
    def on_item_double_clicked(self, item):
        """
        Handle double click on history item
        
        Args:
            item: List widget item
        """
        index = self.row(item)
        history_index = len(self.history) - 1 - index  # Reverse index (most recent first)
        
        if 0 <= history_index < len(self.history):
            history_item = self.history[history_index]
            
            self.formulaSelected.emit(
                history_item.formula,
                history_item.var_name,
                history_item.var_type
            )
    
    def show_context_menu(self, position):
        """
        Show context menu for a history item
        
        Args:
            position: Menu position
        """
        item = self.itemAt(position)
        
        if item:
            menu = QMenu()
            
            reuse_action = QAction("Reuse Formula", self)
            copy_action = QAction("Copy Formula", self)
            remove_action = QAction("Remove from History", self)
            
            menu.addAction(reuse_action)
            menu.addAction(copy_action)
            menu.addSeparator()
            menu.addAction(remove_action)
            
            index = self.row(item)
            history_index = len(self.history) - 1 - index  # Reverse index (most recent first)
            
            def on_reuse():
                if 0 <= history_index < len(self.history):
                    history_item = self.history[history_index]
                    self.formulaSelected.emit(
                        history_item.formula,
                        history_item.var_name,
                        history_item.var_type
                    )
            
            def on_copy():
                if 0 <= history_index < len(self.history):
                    from PyQt5.QtGui import QClipboard
                    from PyQt5.QtWidgets import QApplication
                    
                    history_item = self.history[history_index]
                    QApplication.clipboard().setText(history_item.formula)
            
            def on_remove():
                if 0 <= history_index < len(self.history):
                    self.history.pop(history_index)
                    
                    self.takeItem(index)
            
            reuse_action.triggered.connect(on_reuse)
            copy_action.triggered.connect(on_copy)
            remove_action.triggered.connect(on_remove)
            
            menu.exec_(self.mapToGlobal(position))

class FormulaEditorDialog(QDialog):
    """Dialog for creating new variables through formulas"""
    
    formulaApplied = pyqtSignal(str, str, str)  # formula, variable_name, variable_type
    
    def __init__(self, available_columns=None, parent=None):
        """
        Initialize the formula editor dialog
        
        Args:
            available_columns: List of available column names
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.available_columns = available_columns or []
        
        self.init_ui()
        
        self.connect_signals()
        
        self.setWindowTitle("Formula Editor")
        self.resize(800, 600)
    
    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout()
        
        tab_widget = QTabWidget()
        
        self.formula_editor = FormulaEditor(self.available_columns)
        tab_widget.addTab(self.formula_editor, "Formula Editor")
        
        self.history_widget = FormulaHistoryWidget()
        tab_widget.addTab(self.history_widget, "Formula History")
        
        main_layout.addWidget(tab_widget)
        
        button_layout = QHBoxLayout()
        self.close_btn = QPushButton("Close")
        button_layout.addStretch()
        button_layout.addWidget(self.close_btn)
        
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
    
    def connect_signals(self):
        """Connect signals to slots"""
        self.formula_editor.formulaApplied.connect(self.on_formula_applied)
        
        self.history_widget.formulaSelected.connect(self.on_formula_selected)
        
        self.close_btn.clicked.connect(self.close)
    
    def set_available_columns(self, columns: List[str]):
        """
        Set the available columns
        
        Args:
            columns: List of column names
        """
        self.available_columns = columns
        self.formula_editor.set_available_columns(columns)
    
    def on_formula_applied(self, formula, var_name, var_type):
        """
        Handle formula application
        
        Args:
            formula: Formula expression
            var_name: Variable name
            var_type: Variable type
        """
        self.history_widget.add_formula(formula, var_name, var_type)
        
        self.formulaApplied.emit(formula, var_name, var_type)
    
    def on_formula_selected(self, formula, var_name, var_type):
        """
        Handle formula selection from history
        
        Args:
            formula: Formula expression
            var_name: Variable name
            var_type: Variable type
        """
        self.formula_editor.formula_edit.setPlainText(formula)
        self.formula_editor.var_name_edit.setText(var_name)
        
        index = self.formula_editor.var_type_combo.findText(var_type)
        if index >= 0:
            self.formula_editor.var_type_combo.setCurrentIndex(index)
        
        self.parent().setCurrentWidget(self.formula_editor)
        
        self.formula_editor.validate_formula()
