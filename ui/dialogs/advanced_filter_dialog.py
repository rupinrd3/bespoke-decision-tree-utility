#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Filter Dialog for Bespoke Utility
Provides comprehensive filtering capabilities with complex conditions and data preview
"""

import logging
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QDialogButtonBox, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QComboBox, QLineEdit, QMenu, QAction,
    QWidget, QSizePolicy, QGroupBox, QTabWidget, QTextEdit,
    QCheckBox, QSpinBox, QDoubleSpinBox, QDateEdit, QSlider,
    QListWidget, QListWidgetItem, QSplitter, QProgressBar
)
from PyQt5.QtCore import Qt, pyqtSignal, QDate
from PyQt5.QtGui import QFont

logger = logging.getLogger(__name__)


class AdvancedFilterDialog(QDialog):
    """Advanced filtering dialog with complex conditions and preview"""
    
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        
        self.original_df = df.copy()
        self.filtered_df = df.copy()
        self.filter_conditions = []
        
        self.setWindowTitle("Advanced Data Filter")
        self.resize(1000, 700)
        
        self.init_ui()
        self.update_preview()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        left_panel = self.create_filter_panel()
        splitter.addWidget(left_panel)
        
        right_panel = self.create_preview_panel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([400, 600])
        
        button_layout = QHBoxLayout()
        
        self.apply_button = QPushButton("Apply Filters")
        self.apply_button.clicked.connect(self.apply_filters)
        
        self.reset_button = QPushButton("Reset All")
        self.reset_button.clicked.connect(self.reset_filters)
        
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
        
    def create_filter_panel(self) -> QWidget:
        """Create the filter configuration panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        self.filter_tabs = QTabWidget()
        
        basic_tab = self.create_basic_filters_tab()
        self.filter_tabs.addTab(basic_tab, "Basic Filters")
        
        advanced_tab = self.create_advanced_filters_tab()
        self.filter_tabs.addTab(advanced_tab, "Advanced")
        
        column_tab = self.create_column_filters_tab()
        self.filter_tabs.addTab(column_tab, "Column Filters")
        
        layout.addWidget(self.filter_tabs)
        
        return panel
        
    def create_basic_filters_tab(self) -> QWidget:
        """Create basic filtering interface"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        quick_group = QGroupBox("Quick Filters")
        quick_layout = QVBoxLayout()
        
        self.remove_duplicates_check = QCheckBox("Remove duplicate rows")
        quick_layout.addWidget(self.remove_duplicates_check)
        
        self.remove_empty_rows_check = QCheckBox("Remove rows with all missing values")
        quick_layout.addWidget(self.remove_empty_rows_check)
        
        sample_layout = QHBoxLayout()
        self.sample_check = QCheckBox("Sample data:")
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, len(self.original_df))
        self.sample_spin.setValue(1000)
        self.sample_type_combo = QComboBox()
        self.sample_type_combo.addItems(["rows", "percent"])
        
        sample_layout.addWidget(self.sample_check)
        sample_layout.addWidget(self.sample_spin)
        sample_layout.addWidget(self.sample_type_combo)
        sample_layout.addStretch()
        
        quick_layout.addLayout(sample_layout)
        quick_group.setLayout(quick_layout)
        layout.addWidget(quick_group)
        
        conditions_group = QGroupBox("Filter Conditions")
        conditions_layout = QVBoxLayout()
        
        add_condition_btn = QPushButton("Add Condition")
        add_condition_btn.clicked.connect(self.add_basic_condition)
        conditions_layout.addWidget(add_condition_btn)
        
        self.basic_conditions_layout = QVBoxLayout()
        conditions_layout.addLayout(self.basic_conditions_layout)
        
        conditions_group.setLayout(conditions_layout)
        layout.addWidget(conditions_group)
        
        layout.addStretch()
        return widget
        
    def create_advanced_filters_tab(self) -> QWidget:
        """Create advanced filtering interface"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        query_group = QGroupBox("Custom Query")
        query_layout = QVBoxLayout()
        
        query_layout.addWidget(QLabel("Enter pandas query expression:"))
        self.query_text = QTextEdit()
        self.query_text.setPlaceholderText(
            "Examples:\n"
            "Age > 25 & Income < 50000\n"
            "Category.isin(['A', 'B'])\n"
            "Name.str.contains('John')\n"
            "Date >= '2020-01-01'"
        )
        self.query_text.setMaximumHeight(150)
        query_layout.addWidget(self.query_text)
        
        validate_btn = QPushButton("Validate Query")
        validate_btn.clicked.connect(self.validate_query)
        query_layout.addWidget(validate_btn)
        
        query_group.setLayout(query_layout)
        layout.addWidget(query_group)
        
        stats_group = QGroupBox("Statistical Filters")
        stats_layout = QFormLayout()
        
        self.outlier_check = QCheckBox()
        self.outlier_method_combo = QComboBox()
        self.outlier_method_combo.addItems(["IQR", "Z-score", "Modified Z-score"])
        self.outlier_threshold_spin = QDoubleSpinBox()
        self.outlier_threshold_spin.setRange(1.0, 5.0)
        self.outlier_threshold_spin.setValue(3.0)
        self.outlier_threshold_spin.setSingleStep(0.1)
        
        outlier_layout = QHBoxLayout()
        outlier_layout.addWidget(self.outlier_check)
        outlier_layout.addWidget(self.outlier_method_combo)
        outlier_layout.addWidget(QLabel("Threshold:"))
        outlier_layout.addWidget(self.outlier_threshold_spin)
        outlier_layout.addStretch()
        
        stats_layout.addRow("Remove outliers:", outlier_layout)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        layout.addStretch()
        return widget
        
    def create_column_filters_tab(self) -> QWidget:
        """Create column-specific filtering interface"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        selection_group = QGroupBox("Column Selection")
        selection_layout = QVBoxLayout()
        
        select_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        deselect_all_btn = QPushButton("Deselect All")
        select_all_btn.clicked.connect(self.select_all_columns)
        deselect_all_btn.clicked.connect(self.deselect_all_columns)
        
        select_layout.addWidget(select_all_btn)
        select_layout.addWidget(deselect_all_btn)
        select_layout.addStretch()
        selection_layout.addLayout(select_layout)
        
        self.column_list = QListWidget()
        self.column_list.setSelectionMode(QListWidget.MultiSelection)
        
        for col in self.original_df.columns:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.column_list.addItem(item)
            
        selection_layout.addWidget(self.column_list)
        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)
        
        column_filters_group = QGroupBox("Column-Specific Filters")
        column_filters_layout = QVBoxLayout()
        
        self.column_filter_combo = QComboBox()
        self.column_filter_combo.addItems(self.original_df.columns.tolist())
        self.column_filter_combo.currentTextChanged.connect(self.update_column_filter_details)
        column_filters_layout.addWidget(self.column_filter_combo)
        
        self.column_filter_details = QWidget()
        column_filters_layout.addWidget(self.column_filter_details)
        
        column_filters_group.setLayout(column_filters_layout)
        layout.addWidget(column_filters_group)
        
        self.update_column_filter_details()
        
        layout.addStretch()
        return widget
        
    def create_preview_panel(self) -> QWidget:
        """Create the data preview panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        info_layout = QHBoxLayout()
        self.info_label = QLabel("Original: 0 rows, 0 cols | Filtered: 0 rows, 0 cols")
        self.info_label.setFont(QFont("", 9, QFont.Bold))
        info_layout.addWidget(self.info_label)
        info_layout.addStretch()
        
        layout.addLayout(info_layout)
        
        self.preview_table = QTableWidget()
        self.preview_table.setSortingEnabled(True)
        self.preview_table.setAlternatingRowColors(True)
        layout.addWidget(self.preview_table)
        
        stats_group = QGroupBox("Summary Statistics")
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setReadOnly(True)
        stats_group_layout = QVBoxLayout()
        stats_group_layout.addWidget(self.stats_text)
        stats_group.setLayout(stats_group_layout)
        layout.addWidget(stats_group)
        
        return panel
        
    def add_basic_condition(self):
        """Add a new basic filter condition"""
        condition_widget = BasicConditionWidget(self.original_df.columns.tolist(), self.original_df.dtypes.to_dict())
        condition_widget.remove_requested.connect(lambda w=condition_widget: self.remove_basic_condition(w))
        condition_widget.condition_changed.connect(self.update_preview)
        
        self.basic_conditions_layout.addWidget(condition_widget)
        
    def remove_basic_condition(self, widget):
        """Remove a basic filter condition"""
        self.basic_conditions_layout.removeWidget(widget)
        widget.deleteLater()
        self.update_preview()
        
    def update_column_filter_details(self):
        """Update column-specific filter details"""
        if self.column_filter_details.layout():
            for i in reversed(range(self.column_filter_details.layout().count())):
                self.column_filter_details.layout().itemAt(i).widget().setParent(None)
        else:
            layout = QVBoxLayout(self.column_filter_details)
            self.column_filter_details.setLayout(layout)
            
        layout = self.column_filter_details.layout()
        
        column = self.column_filter_combo.currentText()
        if not column:
            return
            
        series = self.original_df[column]
        
        info_text = f"Column: {column}\nType: {series.dtype}\nNon-null: {series.notna().sum()}/{len(series)}"
        layout.addWidget(QLabel(info_text))
        
        if pd.api.types.is_numeric_dtype(series):
            self.add_numeric_column_filters(layout, column, series)
        elif pd.api.types.is_datetime64_any_dtype(series):
            self.add_datetime_column_filters(layout, column, series)
        else:
            self.add_categorical_column_filters(layout, column, series)
            
    def add_numeric_column_filters(self, layout, column: str, series: pd.Series):
        """Add numeric column filters"""
        range_group = QGroupBox("Value Range")
        range_layout = QFormLayout()
        
        min_val, max_val = series.min(), series.max()
        
        self.range_min_spin = QDoubleSpinBox()
        self.range_min_spin.setRange(min_val, max_val)
        self.range_min_spin.setValue(min_val)
        self.range_min_spin.setSingleStep((max_val - min_val) / 100)
        
        self.range_max_spin = QDoubleSpinBox()
        self.range_max_spin.setRange(min_val, max_val)
        self.range_max_spin.setValue(max_val)
        self.range_max_spin.setSingleStep((max_val - min_val) / 100)
        
        range_layout.addRow("Minimum:", self.range_min_spin)
        range_layout.addRow("Maximum:", self.range_max_spin)
        
        range_group.setLayout(range_layout)
        layout.addWidget(range_group)
        
        percentile_group = QGroupBox("Percentile Filter")
        percentile_layout = QFormLayout()
        
        self.percentile_low = QSpinBox()
        self.percentile_low.setRange(0, 50)
        self.percentile_low.setValue(0)
        self.percentile_low.setSuffix("%")
        
        self.percentile_high = QSpinBox()
        self.percentile_high.setRange(50, 100)
        self.percentile_high.setValue(100)
        self.percentile_high.setSuffix("%")
        
        percentile_layout.addRow("Lower percentile:", self.percentile_low)
        percentile_layout.addRow("Upper percentile:", self.percentile_high)
        
        percentile_group.setLayout(percentile_layout)
        layout.addWidget(percentile_group)
        
    def add_datetime_column_filters(self, layout, column: str, series: pd.Series):
        """Add datetime column filters"""
        date_group = QGroupBox("Date Range")
        date_layout = QFormLayout()
        
        min_date = series.min()
        max_date = series.max()
        
        self.date_from = QDateEdit()
        self.date_from.setDate(QDate.fromString(str(min_date.date()), Qt.ISODate))
        
        self.date_to = QDateEdit()
        self.date_to.setDate(QDate.fromString(str(max_date.date()), Qt.ISODate))
        
        date_layout.addRow("From:", self.date_from)
        date_layout.addRow("To:", self.date_to)
        
        date_group.setLayout(date_layout)
        layout.addWidget(date_group)
        
    def add_categorical_column_filters(self, layout, column: str, series: pd.Series):
        """Add categorical column filters"""
        values_group = QGroupBox("Value Selection")
        values_layout = QVBoxLayout()
        
        select_layout = QHBoxLayout()
        select_all_values_btn = QPushButton("Select All")
        deselect_all_values_btn = QPushButton("Deselect All")
        select_layout.addWidget(select_all_values_btn)
        select_layout.addWidget(deselect_all_values_btn)
        select_layout.addStretch()
        values_layout.addLayout(select_layout)
        
        self.values_list = QListWidget()
        self.values_list.setMaximumHeight(200)
        
        unique_values = series.dropna().unique()
        if len(unique_values) <= 100:  # Only show if not too many unique values
            for value in sorted(unique_values):
                item = QListWidgetItem(str(value))
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                self.values_list.addItem(item)
        else:
            values_layout.addWidget(QLabel(f"Too many unique values ({len(unique_values)}) to display"))
            
        values_layout.addWidget(self.values_list)
        
        select_all_values_btn.clicked.connect(lambda: self.select_all_items(self.values_list))
        deselect_all_values_btn.clicked.connect(lambda: self.deselect_all_items(self.values_list))
        
        values_group.setLayout(values_layout)
        layout.addWidget(values_group)
        
    def select_all_columns(self):
        """Select all columns"""
        for i in range(self.column_list.count()):
            self.column_list.item(i).setCheckState(Qt.Checked)
            
    def deselect_all_columns(self):
        """Deselect all columns"""
        for i in range(self.column_list.count()):
            self.column_list.item(i).setCheckState(Qt.Unchecked)
            
    def select_all_items(self, list_widget):
        """Select all items in a list widget"""
        for i in range(list_widget.count()):
            list_widget.item(i).setCheckState(Qt.Checked)
            
    def deselect_all_items(self, list_widget):
        """Deselect all items in a list widget"""
        for i in range(list_widget.count()):
            list_widget.item(i).setCheckState(Qt.Unchecked)
            
    def validate_query(self):
        """Validate the custom query"""
        query = self.query_text.toPlainText().strip()
        if not query:
            QMessageBox.information(self, "Validation", "Query is empty.")
            return
            
        try:
            result = self.original_df.query(query)
            QMessageBox.information(self, "Validation", f"Query is valid. Would return {len(result)} rows.")
        except Exception as e:
            QMessageBox.warning(self, "Validation Error", f"Invalid query: {str(e)}")
            
    def apply_filters(self):
        """Apply all configured filters"""
        try:
            df = self.original_df.copy()
            
            if self.remove_duplicates_check.isChecked():
                df = df.drop_duplicates()
                
            if self.remove_empty_rows_check.isChecked():
                df = df.dropna(how='all')
                
            if self.sample_check.isChecked():
                sample_size = self.sample_spin.value()
                if self.sample_type_combo.currentText() == "percent":
                    sample_size = int(len(df) * sample_size / 100)
                df = df.sample(n=min(sample_size, len(df)), random_state=42)
                
            basic_conditions = self.get_basic_conditions()
            for condition in basic_conditions:
                df = self.apply_condition(df, condition)
                
            query = self.query_text.toPlainText().strip()
            if query:
                df = df.query(query)
                
            if self.outlier_check.isChecked():
                df = self.remove_outliers(df)
                
            selected_columns = []
            for i in range(self.column_list.count()):
                item = self.column_list.item(i)
                if item.checkState() == Qt.Checked:
                    selected_columns.append(item.text())
                    
            if selected_columns:
                df = df[selected_columns]
                
            self.filtered_df = df
            self.update_preview()
            
        except Exception as e:
            QMessageBox.critical(self, "Filter Error", f"Error applying filters: {str(e)}")
            
    def get_basic_conditions(self) -> List[Dict[str, Any]]:
        """Get all basic filter conditions"""
        conditions = []
        for i in range(self.basic_conditions_layout.count()):
            widget = self.basic_conditions_layout.itemAt(i).widget()
            if isinstance(widget, BasicConditionWidget):
                condition = widget.get_condition()
                if condition:
                    conditions.append(condition)
        return conditions
        
    def apply_condition(self, df: pd.DataFrame, condition: Dict[str, Any]) -> pd.DataFrame:
        """Apply a single filter condition"""
        column = condition['column']
        operator = condition['operator']
        value = condition.get('value')
        
        if operator == '=':
            return df[df[column] == value]
        elif operator == '!=':
            return df[df[column] != value]
        elif operator == '>':
            return df[df[column] > value]
        elif operator == '<':
            return df[df[column] < value]
        elif operator == '>=':
            return df[df[column] >= value]
        elif operator == '<=':
            return df[df[column] <= value]
        elif operator == 'contains':
            return df[df[column].str.contains(str(value), na=False)]
        elif operator == 'starts with':
            return df[df[column].str.startswith(str(value), na=False)]
        elif operator == 'ends with':
            return df[df[column].str.endswith(str(value), na=False)]
        elif operator == 'is null':
            return df[df[column].isna()]
        elif operator == 'is not null':
            return df[df[column].notna()]
        else:
            return df
            
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using the selected method"""
        method = self.outlier_method_combo.currentText()
        threshold = self.outlier_threshold_spin.value()
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if method == "IQR":
            for col in numeric_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        elif method in ["Z-score", "Modified Z-score"]:
            for col in numeric_columns:
                if method == "Z-score":
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                else:  # Modified Z-score
                    median = df[col].median()
                    mad = np.median(np.abs(df[col] - median))
                    z_scores = 0.6745 * (df[col] - median) / mad
                    z_scores = np.abs(z_scores)
                df = df[z_scores < threshold]
                
        return df
        
    def reset_filters(self):
        """Reset all filters to default state"""
        self.remove_duplicates_check.setChecked(False)
        self.remove_empty_rows_check.setChecked(False)
        self.sample_check.setChecked(False)
        
        while self.basic_conditions_layout.count():
            child = self.basic_conditions_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        self.query_text.clear()
        
        self.outlier_check.setChecked(False)
        
        self.select_all_columns()
        
        self.filtered_df = self.original_df.copy()
        self.update_preview()
        
    def update_preview(self):
        """Update the data preview"""
        orig_shape = self.original_df.shape
        filt_shape = self.filtered_df.shape
        self.info_label.setText(f"Original: {orig_shape[0]} rows, {orig_shape[1]} cols | "
                               f"Filtered: {filt_shape[0]} rows, {filt_shape[1]} cols")
        
        preview_df = self.filtered_df.head(100)
        
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
        
        self.update_summary_stats()
        
    def update_summary_stats(self):
        """Update summary statistics display"""
        try:
            stats_text = f"Dataset Summary (Filtered):\n"
            stats_text += f"Shape: {self.filtered_df.shape[0]} rows, {self.filtered_df.shape[1]} columns\n"
            stats_text += f"Memory usage: {self.filtered_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n"
            
            numeric_df = self.filtered_df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                stats_text += "Numeric Columns:\n"
                desc = numeric_df.describe()
                stats_text += desc.to_string()
                stats_text += "\n\n"
                
            categorical_df = self.filtered_df.select_dtypes(include=[object])
            if not categorical_df.empty:
                stats_text += "Categorical Columns:\n"
                for col in categorical_df.columns[:5]:  # Show first 5 categorical columns
                    unique_count = categorical_df[col].nunique()
                    stats_text += f"{col}: {unique_count} unique values\n"
                    
            self.stats_text.setPlainText(stats_text)
            
        except Exception as e:
            self.stats_text.setPlainText(f"Error generating statistics: {str(e)}")
            
    def get_filtered_dataframe(self) -> pd.DataFrame:
        """Get the filtered DataFrame"""
        return self.filtered_df.copy()


class BasicConditionWidget(QWidget):
    """Widget for a single basic filter condition"""
    
    condition_changed = pyqtSignal()
    remove_requested = pyqtSignal()
    
    def __init__(self, columns: List[str], dtypes: Dict[str, Any]):
        super().__init__()
        
        self.columns = columns
        self.dtypes = dtypes
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.logic_combo = QComboBox()
        self.logic_combo.addItems(["AND", "OR"])
        self.logic_combo.setVisible(False)  # Hidden for first condition
        layout.addWidget(self.logic_combo)
        
        self.column_combo = QComboBox()
        self.column_combo.addItems(self.columns)
        layout.addWidget(self.column_combo)
        
        self.operator_combo = QComboBox()
        layout.addWidget(self.operator_combo)
        
        self.value_edit = QLineEdit()
        layout.addWidget(self.value_edit)
        
        self.remove_btn = QPushButton("✕")
        self.remove_btn.setMaximumWidth(30)
        self.remove_btn.clicked.connect(self.remove_requested.emit)
        layout.addWidget(self.remove_btn)
        
        self.column_combo.currentTextChanged.connect(self.update_operators)
        self.operator_combo.currentTextChanged.connect(self.condition_changed.emit)
        self.value_edit.textChanged.connect(self.condition_changed.emit)
        
        self.update_operators()
        
    def update_operators(self):
        """Update available operators based on selected column"""
        column = self.column_combo.currentText()
        if not column:
            return
            
        dtype = self.dtypes.get(column, 'object')
        
        self.operator_combo.clear()
        
        if pd.api.types.is_numeric_dtype(dtype):
            operators = ["=", "!=", ">", "<", ">=", "<=", "is null", "is not null"]
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            operators = ["=", "!=", ">", "<", ">=", "<=", "is null", "is not null"]
        else:
            operators = ["=", "!=", "contains", "starts with", "ends with", "is null", "is not null"]
            
        self.operator_combo.addItems(operators)
        
    def set_logic_visible(self, visible: bool):
        """Set logic operator visibility"""
        self.logic_combo.setVisible(visible)
        
    def get_condition(self) -> Optional[Dict[str, Any]]:
        """Get the filter condition"""
        column = self.column_combo.currentText()
        operator = self.operator_combo.currentText()
        
        if not column or not operator:
            return None
            
        condition = {
            'logic': self.logic_combo.currentText() if self.logic_combo.isVisible() else 'AND',
            'column': column,
            'operator': operator
        }
        
        if operator not in ["is null", "is not null"]:
            value_text = self.value_edit.text()
            
            dtype = self.dtypes.get(column, 'object')
            if pd.api.types.is_numeric_dtype(dtype):
                try:
                    condition['value'] = float(value_text)
                except ValueError:
                    condition['value'] = value_text
            else:
                condition['value'] = value_text
                
        return condition