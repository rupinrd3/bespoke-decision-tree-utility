#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Transformation Dialog for Bespoke Utility
Comprehensive dialog for creating new features and transforming data
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout, QLabel,
    QDialogButtonBox, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QComboBox, QLineEdit, QTextEdit,
    QWidget, QSizePolicy, QGroupBox, QTabWidget, QSpinBox,
    QDoubleSpinBox, QCheckBox, QListWidget, QListWidgetItem,
    QSplitter, QProgressBar, QSlider, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont, QValidator, QDoubleValidator

from data.feature_engineering import FeatureEngineering

logger = logging.getLogger(__name__)


class DataTransformationDialog(QDialog):
    """Comprehensive data transformation dialog"""
    
    transformationApplied = pyqtSignal(pd.DataFrame, str)  # dataframe, operation_name
    
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        
        self.original_df = df.copy()
        self.current_df = df.copy()
        self.feature_engineering = FeatureEngineering()
        self.transformation_history = []
        
        self.setWindowTitle("Data Transformation")
        self.resize(1200, 800)
        
        self.init_ui()
        
        self.feature_engineering.featureCreated.connect(self.on_feature_created)
        self.feature_engineering.featureError.connect(self.on_feature_error)
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        left_panel = self.create_transformation_panel()
        splitter.addWidget(left_panel)
        
        right_panel = self.create_preview_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([500, 700])
        
        button_layout = QHBoxLayout()
        
        self.apply_button = QPushButton("Apply Transformation")
        self.apply_button.clicked.connect(self.apply_transformation)
        
        self.reset_button = QPushButton("Reset All")
        self.reset_button.clicked.connect(self.reset_transformations)
        
        self.preview_button = QPushButton("Update Preview")
        self.preview_button.clicked.connect(self.update_preview)
        
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.preview_button)
        button_layout.addStretch()
        
        dialog_buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        dialog_buttons.accepted.connect(self.accept)
        dialog_buttons.rejected.connect(self.reject)
        
        button_layout.addWidget(dialog_buttons)
        layout.addLayout(button_layout)
        
    def create_transformation_panel(self) -> QWidget:
        """Create the transformation configuration panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        self.transformation_tabs = QTabWidget()
        
        formula_tab = self.create_formula_tab()
        self.transformation_tabs.addTab(formula_tab, "Formula")
        
        binning_tab = self.create_binning_tab()
        self.transformation_tabs.addTab(binning_tab, "Binning")
        
        interaction_tab = self.create_interaction_tab()
        self.transformation_tabs.addTab(interaction_tab, "Interactions")
        
        binary_tab = self.create_binary_tab()
        self.transformation_tabs.addTab(binary_tab, "Binary Variables")
        
        math_tab = self.create_math_transformations_tab()
        self.transformation_tabs.addTab(math_tab, "Math Functions")
        
        text_tab = self.create_text_transformations_tab()
        self.transformation_tabs.addTab(text_tab, "Text Processing")
        
        date_tab = self.create_date_transformations_tab()
        self.transformation_tabs.addTab(date_tab, "Date/Time")
        
        layout.addWidget(self.transformation_tabs)
        
        return panel
        
    def create_formula_tab(self) -> QWidget:
        """Create the formula transformation tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        formula_group = QGroupBox("Custom Formula")
        formula_layout = QVBoxLayout()
        
        name_layout = QFormLayout()
        self.formula_name_edit = QLineEdit()
        self.formula_name_edit.setPlaceholderText("Enter variable name")
        name_layout.addRow("Variable Name:", self.formula_name_edit)
        formula_layout.addLayout(name_layout)
        
        formula_layout.addWidget(QLabel("Formula:"))
        self.formula_edit = QTextEdit()
        self.formula_edit.setMaximumHeight(100)
        self.formula_edit.setPlaceholderText(
            "Examples:\n"
            "Age * 2 + Income / 1000\n"
            "LOG(Sales) + SQRT(Profit)\n"
            "IF(Age > 30, 1, 0)\n"
            "MIN(Score1, Score2, Score3)"
        )
        formula_layout.addWidget(self.formula_edit)
        
        type_layout = QFormLayout()
        self.formula_type_combo = QComboBox()
        self.formula_type_combo.addItems(['float', 'int', 'str', 'category'])
        type_layout.addRow("Data Type:", self.formula_type_combo)
        formula_layout.addLayout(type_layout)
        
        validate_btn = QPushButton("Validate Formula")
        validate_btn.clicked.connect(self.validate_formula)
        formula_layout.addWidget(validate_btn)
        
        formula_group.setLayout(formula_layout)
        layout.addWidget(formula_group)
        
        variables_group = QGroupBox("Available Variables")
        variables_layout = QVBoxLayout()
        
        self.variables_list = QListWidget()
        self.variables_list.itemDoubleClicked.connect(self.insert_variable)
        variables_layout.addWidget(self.variables_list)
        
        variables_group.setLayout(variables_layout)
        layout.addWidget(variables_group)
        
        functions_group = QGroupBox("Available Functions")
        functions_layout = QVBoxLayout()
        
        functions_text = QLabel("""
        <b>Mathematical:</b> LOG, LN, EXP, SQRT, ABS, ROUND, FLOOR, CEIL<br>
        <b>Trigonometric:</b> SIN, COS, TAN<br>
        <b>Statistical:</b> MIN, MAX, MEAN, SUM<br>
        <b>Conditional:</b> IF(condition, true_value, false_value)<br>
        <b>Operators:</b> +, -, *, /, ^, %
        """)
        functions_text.setWordWrap(True)
        functions_layout.addWidget(functions_text)
        
        functions_group.setLayout(functions_layout)
        layout.addWidget(functions_group)
        
        layout.addStretch()
        return widget
        
    def create_binning_tab(self) -> QWidget:
        """Create the binning transformation tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        binning_group = QGroupBox("Binning Configuration")
        binning_layout = QFormLayout()
        
        self.binning_name_edit = QLineEdit()
        self.binning_name_edit.setPlaceholderText("Enter binned variable name")
        binning_layout.addRow("Variable Name:", self.binning_name_edit)
        
        self.binning_source_combo = QComboBox()
        binning_layout.addRow("Source Column:", self.binning_source_combo)
        
        self.binning_num_bins_spin = QSpinBox()
        self.binning_num_bins_spin.setRange(2, 20)
        self.binning_num_bins_spin.setValue(5)
        binning_layout.addRow("Number of Bins:", self.binning_num_bins_spin)
        
        self.binning_method_combo = QComboBox()
        self.binning_method_combo.addItems(['equal_width', 'equal_freq', 'custom'])
        binning_layout.addRow("Binning Method:", self.binning_method_combo)
        
        binning_group.setLayout(binning_layout)
        layout.addWidget(binning_group)
        
        labels_group = QGroupBox("Custom Labels (Optional)")
        labels_layout = QVBoxLayout()
        
        self.binning_labels_edit = QTextEdit()
        self.binning_labels_edit.setMaximumHeight(80)
        self.binning_labels_edit.setPlaceholderText("Enter comma-separated labels (e.g., Low, Medium, High)")
        labels_layout.addWidget(self.binning_labels_edit)
        
        labels_group.setLayout(labels_layout)
        layout.addWidget(labels_group)
        
        layout.addStretch()
        return widget
        
    def create_interaction_tab(self) -> QWidget:
        """Create the interaction transformation tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        interaction_group = QGroupBox("Interaction Configuration")
        interaction_layout = QFormLayout()
        
        self.interaction_name_edit = QLineEdit()
        self.interaction_name_edit.setPlaceholderText("Enter interaction variable name")
        interaction_layout.addRow("Variable Name:", self.interaction_name_edit)
        
        self.interaction_type_combo = QComboBox()
        self.interaction_type_combo.addItems(['multiply', 'add', 'subtract', 'divide'])
        interaction_layout.addRow("Interaction Type:", self.interaction_type_combo)
        
        interaction_group.setLayout(interaction_layout)
        layout.addWidget(interaction_group)
        
        columns_group = QGroupBox("Select Columns")
        columns_layout = QVBoxLayout()
        
        self.interaction_available_list = QListWidget()
        self.interaction_available_list.setSelectionMode(QListWidget.MultiSelection)
        columns_layout.addWidget(QLabel("Available Columns:"))
        columns_layout.addWidget(self.interaction_available_list)
        
        self.interaction_selected_list = QListWidget()
        columns_layout.addWidget(QLabel("Selected Columns:"))
        columns_layout.addWidget(self.interaction_selected_list)
        
        button_layout = QHBoxLayout()
        add_btn = QPushButton("Add →")
        add_btn.clicked.connect(self.add_interaction_column)
        remove_btn = QPushButton("← Remove")
        remove_btn.clicked.connect(self.remove_interaction_column)
        
        button_layout.addWidget(add_btn)
        button_layout.addWidget(remove_btn)
        columns_layout.addLayout(button_layout)
        
        columns_group.setLayout(columns_layout)
        layout.addWidget(columns_group)
        
        layout.addStretch()
        return widget
        
    def create_binary_tab(self) -> QWidget:
        """Create the binary variable transformation tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        binary_group = QGroupBox("Binary Variable Configuration")
        binary_layout = QFormLayout()
        
        self.binary_name_edit = QLineEdit()
        self.binary_name_edit.setPlaceholderText("Enter binary variable name")
        binary_layout.addRow("Variable Name:", self.binary_name_edit)
        
        self.binary_source_combo = QComboBox()
        binary_layout.addRow("Source Column:", self.binary_source_combo)
        
        self.binary_operator_combo = QComboBox()
        self.binary_operator_combo.addItems(['>', '>=', '<', '<=', '=', '!=', 'contains'])
        binary_layout.addRow("Operator:", self.binary_operator_combo)
        
        self.binary_value_edit = QLineEdit()
        self.binary_value_edit.setPlaceholderText("Enter comparison value")
        binary_layout.addRow("Value:", self.binary_value_edit)
        
        binary_group.setLayout(binary_layout)
        layout.addWidget(binary_group)
        
        layout.addStretch()
        return widget
        
    def create_math_transformations_tab(self) -> QWidget:
        """Create mathematical transformations tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        math_group = QGroupBox("Mathematical Transformation")
        math_layout = QFormLayout()
        
        self.math_name_edit = QLineEdit()
        self.math_name_edit.setPlaceholderText("Enter transformed variable name")
        math_layout.addRow("Variable Name:", self.math_name_edit)
        
        self.math_source_combo = QComboBox()
        math_layout.addRow("Source Column:", self.math_source_combo)
        
        self.math_transform_combo = QComboBox()
        self.math_transform_combo.addItems([
            'log', 'log10', 'sqrt', 'square', 'abs', 'exp',
            'reciprocal', 'standardize', 'normalize', 'rank'
        ])
        math_layout.addRow("Transformation:", self.math_transform_combo)
        
        math_group.setLayout(math_layout)
        layout.addWidget(math_group)
        
        scaling_group = QGroupBox("Scaling Options")
        scaling_layout = QVBoxLayout()
        
        minmax_layout = QHBoxLayout()
        self.minmax_check = QCheckBox("Min-Max Scaling")
        self.minmax_min_spin = QDoubleSpinBox()
        self.minmax_min_spin.setRange(-1000, 1000)
        self.minmax_min_spin.setValue(0)
        self.minmax_max_spin = QDoubleSpinBox()
        self.minmax_max_spin.setRange(-1000, 1000)
        self.minmax_max_spin.setValue(1)
        
        minmax_layout.addWidget(self.minmax_check)
        minmax_layout.addWidget(QLabel("Min:"))
        minmax_layout.addWidget(self.minmax_min_spin)
        minmax_layout.addWidget(QLabel("Max:"))
        minmax_layout.addWidget(self.minmax_max_spin)
        minmax_layout.addStretch()
        
        scaling_layout.addLayout(minmax_layout)
        scaling_group.setLayout(scaling_layout)
        layout.addWidget(scaling_group)
        
        layout.addStretch()
        return widget
        
    def create_text_transformations_tab(self) -> QWidget:
        """Create text transformations tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        text_group = QGroupBox("Text Transformation")
        text_layout = QFormLayout()
        
        self.text_name_edit = QLineEdit()
        self.text_name_edit.setPlaceholderText("Enter transformed variable name")
        text_layout.addRow("Variable Name:", self.text_name_edit)
        
        self.text_source_combo = QComboBox()
        text_layout.addRow("Source Column:", self.text_source_combo)
        
        self.text_transform_combo = QComboBox()
        self.text_transform_combo.addItems([
            'length', 'word_count', 'upper', 'lower', 'title',
            'extract_numbers', 'remove_punctuation', 'encode_categorical'
        ])
        text_layout.addRow("Transformation:", self.text_transform_combo)
        
        text_group.setLayout(text_layout)
        layout.addWidget(text_group)
        
        pattern_group = QGroupBox("Pattern Extraction")
        pattern_layout = QFormLayout()
        
        self.text_pattern_edit = QLineEdit()
        self.text_pattern_edit.setPlaceholderText("Enter regex pattern (optional)")
        pattern_layout.addRow("Pattern:", self.text_pattern_edit)
        
        pattern_group.setLayout(pattern_layout)
        layout.addWidget(pattern_group)
        
        layout.addStretch()
        return widget
        
    def create_date_transformations_tab(self) -> QWidget:
        """Create date/time transformations tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        date_group = QGroupBox("Date/Time Transformation")
        date_layout = QFormLayout()
        
        self.date_name_edit = QLineEdit()
        self.date_name_edit.setPlaceholderText("Enter transformed variable name")
        date_layout.addRow("Variable Name:", self.date_name_edit)
        
        self.date_source_combo = QComboBox()
        date_layout.addRow("Source Column:", self.date_source_combo)
        
        self.date_transform_combo = QComboBox()
        self.date_transform_combo.addItems([
            'year', 'month', 'day', 'weekday', 'quarter',
            'days_since', 'age_in_years', 'is_weekend', 'is_holiday'
        ])
        date_layout.addRow("Transformation:", self.date_transform_combo)
        
        date_group.setLayout(date_layout)
        layout.addWidget(date_group)
        
        ref_group = QGroupBox("Reference Date (for calculations)")
        ref_layout = QFormLayout()
        
        self.ref_date_edit = QLineEdit()
        self.ref_date_edit.setPlaceholderText("YYYY-MM-DD (optional, defaults to today)")
        ref_layout.addRow("Reference Date:", self.ref_date_edit)
        
        ref_group.setLayout(ref_layout)
        layout.addWidget(ref_group)
        
        layout.addStretch()
        return widget
        
    def create_preview_panel(self) -> QWidget:
        """Create the data preview and history panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        preview_tabs = QTabWidget()
        
        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        
        info_layout = QHBoxLayout()
        self.preview_info_label = QLabel("Current: 0 rows, 0 cols")
        self.preview_info_label.setFont(QFont("", 9, QFont.Bold))
        info_layout.addWidget(self.preview_info_label)
        info_layout.addStretch()
        preview_layout.addLayout(info_layout)
        
        self.preview_table = QTableWidget()
        self.preview_table.setSortingEnabled(False)
        self.preview_table.setAlternatingRowColors(True)
        preview_layout.addWidget(self.preview_table)
        
        preview_tabs.addTab(preview_tab, "Data Preview")
        
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        
        history_layout.addWidget(QLabel("Transformation History:"))
        self.history_list = QListWidget()
        history_layout.addWidget(self.history_list)
        
        undo_btn = QPushButton("Undo Last Transformation")
        undo_btn.clicked.connect(self.undo_transformation)
        history_layout.addWidget(undo_btn)
        
        preview_tabs.addTab(history_tab, "History")
        
        layout.addWidget(preview_tabs)
        
        return panel
        
    def showEvent(self, event):
        """Initialize UI when dialog is shown"""
        super().showEvent(event)
        self.populate_column_lists()
        self.update_preview()
        
    def populate_column_lists(self):
        """Populate all column combo boxes and lists"""
        columns = list(self.current_df.columns)
        numeric_columns = list(self.current_df.select_dtypes(include=[np.number]).columns)
        text_columns = list(self.current_df.select_dtypes(include=[object]).columns)
        date_columns = list(self.current_df.select_dtypes(include=['datetime']).columns)
        
        self.variables_list.clear()
        for col in columns:
            self.variables_list.addItem(col)
            
        self.binning_source_combo.clear()
        self.binning_source_combo.addItems(numeric_columns)
        
        self.interaction_available_list.clear()
        for col in numeric_columns:
            self.interaction_available_list.addItem(col)
            
        self.binary_source_combo.clear()
        self.binary_source_combo.addItems(columns)
        
        self.math_source_combo.clear()
        self.math_source_combo.addItems(numeric_columns)
        
        self.text_source_combo.clear()
        self.text_source_combo.addItems(text_columns)
        
        self.date_source_combo.clear()
        self.date_source_combo.addItems(date_columns + text_columns)
        
    def insert_variable(self, item):
        """Insert variable name into formula"""
        variable = item.text()
        cursor = self.formula_edit.textCursor()
        cursor.insertText(variable)
        
    def add_interaction_column(self):
        """Add column to interaction selection"""
        for item in self.interaction_available_list.selectedItems():
            self.interaction_selected_list.addItem(item.text())
            
    def remove_interaction_column(self):
        """Remove column from interaction selection"""
        for item in self.interaction_selected_list.selectedItems():
            row = self.interaction_selected_list.row(item)
            self.interaction_selected_list.takeItem(row)
            
    def validate_formula(self):
        """Validate the current formula"""
        formula = self.formula_edit.toPlainText().strip()
        if not formula:
            QMessageBox.warning(self, "Validation", "Please enter a formula.")
            return
            
        is_valid, error_msg = self.feature_engineering.formula_parser.validate_syntax(formula)
        if is_valid:
            QMessageBox.information(self, "Validation", "Formula syntax is valid.")
        else:
            QMessageBox.warning(self, "Validation Error", f"Invalid formula: {error_msg}")
            
    def apply_transformation(self):
        """Apply the current transformation"""
        current_tab = self.transformation_tabs.currentIndex()
        
        try:
            if current_tab == 0:  # Formula
                self.apply_formula_transformation()
            elif current_tab == 1:  # Binning
                self.apply_binning_transformation()
            elif current_tab == 2:  # Interaction
                self.apply_interaction_transformation()
            elif current_tab == 3:  # Binary
                self.apply_binary_transformation()
            elif current_tab == 4:  # Math
                self.apply_math_transformation()
            elif current_tab == 5:  # Text
                self.apply_text_transformation()
            elif current_tab == 6:  # Date
                self.apply_date_transformation()
                
        except Exception as e:
            QMessageBox.critical(self, "Transformation Error", f"Error applying transformation: {str(e)}")
            
    def apply_formula_transformation(self):
        """Apply formula transformation"""
        var_name = self.formula_name_edit.text().strip()
        formula = self.formula_edit.toPlainText().strip()
        var_type = self.formula_type_combo.currentText()
        
        if not var_name or not formula:
            QMessageBox.warning(self, "Input Error", "Please enter both variable name and formula.")
            return
            
        self.current_df = self.feature_engineering.create_formula_variable(
            self.current_df, formula, var_name, var_type, "current"
        )
        
        self.add_to_history(f"Formula: {var_name} = {formula}")
        self.update_preview()
        self.populate_column_lists()
        
    def apply_binning_transformation(self):
        """Apply binning transformation"""
        var_name = self.binning_name_edit.text().strip()
        source_col = self.binning_source_combo.currentText()
        num_bins = self.binning_num_bins_spin.value()
        method = self.binning_method_combo.currentText()
        
        if not var_name or not source_col:
            QMessageBox.warning(self, "Input Error", "Please enter variable name and select source column.")
            return
            
        labels_text = self.binning_labels_edit.toPlainText().strip()
        labels = None
        if labels_text:
            labels = [label.strip() for label in labels_text.split(',')]
            if len(labels) != num_bins:
                QMessageBox.warning(self, "Label Error", f"Number of labels ({len(labels)}) must match number of bins ({num_bins}).")
                return
                
        self.current_df = self.feature_engineering.create_binned_variable(
            self.current_df, source_col, var_name, num_bins, method, labels, "current"
        )
        
        self.add_to_history(f"Binning: {var_name} from {source_col} ({method}, {num_bins} bins)")
        self.update_preview()
        self.populate_column_lists()
        
    def apply_interaction_transformation(self):
        """Apply interaction transformation"""
        var_name = self.interaction_name_edit.text().strip()
        interaction_type = self.interaction_type_combo.currentText()
        
        if not var_name:
            QMessageBox.warning(self, "Input Error", "Please enter variable name.")
            return
            
        columns = []
        for i in range(self.interaction_selected_list.count()):
            columns.append(self.interaction_selected_list.item(i).text())
            
        if len(columns) < 2:
            QMessageBox.warning(self, "Selection Error", "Please select at least 2 columns for interaction.")
            return
            
        self.current_df = self.feature_engineering.create_interaction_term(
            self.current_df, columns, var_name, interaction_type, "current"
        )
        
        self.add_to_history(f"Interaction: {var_name} = {' '.join(columns)} ({interaction_type})")
        self.update_preview()
        self.populate_column_lists()
        
    def apply_binary_transformation(self):
        """Apply binary transformation"""
        var_name = self.binary_name_edit.text().strip()
        source_col = self.binary_source_combo.currentText()
        operator = self.binary_operator_combo.currentText()
        value = self.binary_value_edit.text().strip()
        
        if not var_name or not source_col or not value:
            QMessageBox.warning(self, "Input Error", "Please fill in all fields.")
            return
            
        if operator != 'contains':
            try:
                if self.current_df[source_col].dtype in ['int64', 'float64']:
                    value = float(value)
            except ValueError:
                pass  # Keep as string
                
        condition = {
            'column': source_col,
            'operator': operator,
            'value': value
        }
        
        self.current_df = self.feature_engineering.create_binary_variable(
            self.current_df, condition, var_name, "current"
        )
        
        self.add_to_history(f"Binary: {var_name} = {source_col} {operator} {value}")
        self.update_preview()
        self.populate_column_lists()
        
    def apply_math_transformation(self):
        """Apply mathematical transformation"""
        var_name = self.math_name_edit.text().strip()
        source_col = self.math_source_combo.currentText()
        transform = self.math_transform_combo.currentText()
        
        if not var_name or not source_col:
            QMessageBox.warning(self, "Input Error", "Please enter variable name and select source column.")
            return
            
        try:
            result_df = self.current_df.copy()
            series = self.current_df[source_col]
            
            if transform == 'log':
                result = np.log(series.replace(0, np.nan))
            elif transform == 'log10':
                result = np.log10(series.replace(0, np.nan))
            elif transform == 'sqrt':
                result = np.sqrt(series.abs())
            elif transform == 'square':
                result = series ** 2
            elif transform == 'abs':
                result = series.abs()
            elif transform == 'exp':
                result = np.exp(series)
            elif transform == 'reciprocal':
                result = 1 / series.replace(0, np.nan)
            elif transform == 'standardize':
                result = (series - series.mean()) / series.std()
            elif transform == 'normalize':
                result = (series - series.min()) / (series.max() - series.min())
            elif transform == 'rank':
                result = series.rank()
            else:
                QMessageBox.warning(self, "Transform Error", f"Unknown transformation: {transform}")
                return
                
            if self.minmax_check.isChecked():
                min_val = self.minmax_min_spin.value()
                max_val = self.minmax_max_spin.value()
                result = min_val + (result - result.min()) * (max_val - min_val) / (result.max() - result.min())
                
            result_df[var_name] = result
            self.current_df = result_df
            
            self.add_to_history(f"Math: {var_name} = {transform}({source_col})")
            self.update_preview()
            self.populate_column_lists()
            
        except Exception as e:
            QMessageBox.critical(self, "Transformation Error", f"Error applying {transform}: {str(e)}")
            
    def apply_text_transformation(self):
        """Apply text transformation"""
        var_name = self.text_name_edit.text().strip()
        source_col = self.text_source_combo.currentText()
        transform = self.text_transform_combo.currentText()
        pattern = self.text_pattern_edit.text().strip()
        
        if not var_name or not source_col:
            QMessageBox.warning(self, "Input Error", "Please enter variable name and select source column.")
            return
            
        try:
            result_df = self.current_df.copy()
            series = self.current_df[source_col].astype(str)
            
            if transform == 'length':
                result = series.str.len()
            elif transform == 'word_count':
                result = series.str.split().str.len()
            elif transform == 'upper':
                result = series.str.upper()
            elif transform == 'lower':
                result = series.str.lower()
            elif transform == 'title':
                result = series.str.title()
            elif transform == 'extract_numbers':
                result = series.str.extract(r'(\d+)')[0].astype(float)
            elif transform == 'remove_punctuation':
                result = series.str.replace(r'[^\w\s]', '', regex=True)
            elif transform == 'encode_categorical':
                result = pd.Categorical(series).codes
            else:
                QMessageBox.warning(self, "Transform Error", f"Unknown transformation: {transform}")
                return
                
            result_df[var_name] = result
            self.current_df = result_df
            
            self.add_to_history(f"Text: {var_name} = {transform}({source_col})")
            self.update_preview()
            self.populate_column_lists()
            
        except Exception as e:
            QMessageBox.critical(self, "Transformation Error", f"Error applying {transform}: {str(e)}")
            
    def apply_date_transformation(self):
        """Apply date transformation"""
        var_name = self.date_name_edit.text().strip()
        source_col = self.date_source_combo.currentText()
        transform = self.date_transform_combo.currentText()
        ref_date = self.ref_date_edit.text().strip()
        
        if not var_name or not source_col:
            QMessageBox.warning(self, "Input Error", "Please enter variable name and select source column.")
            return
            
        try:
            result_df = self.current_df.copy()
            
            if not pd.api.types.is_datetime64_any_dtype(self.current_df[source_col]):
                series = pd.to_datetime(self.current_df[source_col], errors='coerce')
            else:
                series = self.current_df[source_col]
                
            if transform == 'year':
                result = series.dt.year
            elif transform == 'month':
                result = series.dt.month
            elif transform == 'day':
                result = series.dt.day
            elif transform == 'weekday':
                result = series.dt.weekday
            elif transform == 'quarter':
                result = series.dt.quarter
            elif transform == 'days_since':
                ref = pd.to_datetime(ref_date) if ref_date else pd.Timestamp.now()
                result = (ref - series).dt.days
            elif transform == 'age_in_years':
                ref = pd.to_datetime(ref_date) if ref_date else pd.Timestamp.now()
                result = (ref - series).dt.days / 365.25
            elif transform == 'is_weekend':
                result = (series.dt.weekday >= 5).astype(int)
            elif transform == 'is_holiday':
                result = ((series.dt.month == 12) & (series.dt.day == 25)).astype(int)
            else:
                QMessageBox.warning(self, "Transform Error", f"Unknown transformation: {transform}")
                return
                
            result_df[var_name] = result
            self.current_df = result_df
            
            self.add_to_history(f"Date: {var_name} = {transform}({source_col})")
            self.update_preview()
            self.populate_column_lists()
            
        except Exception as e:
            QMessageBox.critical(self, "Transformation Error", f"Error applying {transform}: {str(e)}")
            
    def add_to_history(self, operation: str):
        """Add operation to history"""
        self.transformation_history.append({
            'operation': operation,
            'dataframe': self.current_df.copy()
        })
        self.history_list.addItem(operation)
        
    def undo_transformation(self):
        """Undo the last transformation"""
        if len(self.transformation_history) > 1:
            self.transformation_history.pop()
            previous = self.transformation_history[-1]
            self.current_df = previous['dataframe'].copy()
            
            self.history_list.takeItem(self.history_list.count() - 1)
            self.update_preview()
            self.populate_column_lists()
        else:
            QMessageBox.information(self, "Undo", "No more transformations to undo.")
            
    def reset_transformations(self):
        """Reset all transformations"""
        self.current_df = self.original_df.copy()
        self.transformation_history = []
        self.history_list.clear()
        self.update_preview()
        self.populate_column_lists()
        
    def update_preview(self):
        """Update the data preview"""
        shape = self.current_df.shape
        self.preview_info_label.setText(f"Current: {shape[0]} rows, {shape[1]} cols")
        
        preview_df = self.current_df.head(100)
        
        self.preview_table.setRowCount(len(preview_df))
        self.preview_table.setColumnCount(len(preview_df.columns))
        self.preview_table.setHorizontalHeaderLabels([str(col) for col in preview_df.columns])
        
        for i in range(len(preview_df)):
            for j, col in enumerate(preview_df.columns):
                value = preview_df.iloc[i, j]
                if pd.isna(value):
                    item_text = "<NA>"
                else:
                    item_text = str(value)
                    
                item = QTableWidgetItem(item_text)
                if pd.isna(value):
                    item.setBackground(Qt.lightGray)
                self.preview_table.setItem(i, j, item)
                
        self.preview_table.resizeColumnsToContents()
        
    def on_feature_created(self, dataset_name: str, column_name: str):
        """Handle feature created signal"""
        logger.info(f"Feature created: {column_name}")
        
    def on_feature_error(self, error_message: str):
        """Handle feature error signal"""
        QMessageBox.warning(self, "Feature Engineering Error", error_message)
        
    def get_transformed_dataframe(self) -> pd.DataFrame:
        """Get the transformed dataframe"""
        return self.current_df.copy()
        
    def get_transformation_history(self) -> List[str]:
        """Get the transformation history"""
        return [item['operation'] for item in self.transformation_history]