#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Data Filter Dialog for Bespoke Utility
Comprehensive filtering interface with logical operators and preview
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QTabWidget, QWidget, QListWidget,
    QSplitter, QProgressBar, QMessageBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QFrame, QSizePolicy, QDateEdit, QTimeEdit, QDateTimeEdit,
    QButtonGroup, QRadioButton
)

logger = logging.getLogger(__name__)


class FilterCondition:
    """Represents a single filter condition"""
    
    def __init__(self, column: str = "", operator: str = "=", value: str = "", 
                 logic: str = "AND", condition_type: str = "simple"):
        self.column = column
        self.operator = operator  
        self.value = value
        self.logic = logic  # AND/OR connection to next condition
        self.condition_type = condition_type  # simple/range/list/null
        
        self.value2 = ""
        
        self.value_list = []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert condition to dictionary"""
        return {
            'column': self.column,
            'operator': self.operator,
            'value': self.value,
            'value2': self.value2,
            'value_list': self.value_list,
            'logic': self.logic,
            'condition_type': self.condition_type
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FilterCondition':
        """Create condition from dictionary"""
        condition = cls()
        condition.column = data.get('column', '')
        condition.operator = data.get('operator', '=')
        condition.value = data.get('value', '')
        condition.value2 = data.get('value2', '')
        condition.value_list = data.get('value_list', [])
        condition.logic = data.get('logic', 'AND')
        condition.condition_type = data.get('condition_type', 'simple')
        return condition


class ConditionWidget(QWidget):
    """Widget for editing a single filter condition"""
    
    conditionChanged = pyqtSignal()
    removeRequested = pyqtSignal()
    
    def __init__(self, condition: FilterCondition, column_info: Dict[str, Any], parent=None):
        super().__init__(parent)
        
        self.condition = condition
        self.column_info = column_info
        
        self.setupUI()
        self.connectSignals()
        self.updateOperators()
        
    def setupUI(self):
        """Setup the UI components"""
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.logic_combo = QComboBox()
        self.logic_combo.addItems(['AND', 'OR'])
        self.logic_combo.setCurrentText(self.condition.logic)
        self.logic_combo.setFixedWidth(60)
        layout.addWidget(self.logic_combo)
        
        self.column_combo = QComboBox()
        self.column_combo.addItems(list(self.column_info.keys()))
        if self.condition.column:
            self.column_combo.setCurrentText(self.condition.column)
        self.column_combo.setMinimumWidth(120)
        layout.addWidget(self.column_combo)
        
        self.operator_combo = QComboBox()
        self.operator_combo.setMinimumWidth(80)
        layout.addWidget(self.operator_combo)
        
        self.value_widget = QLineEdit()
        self.value_widget.setText(self.condition.value)
        self.value_widget.setMinimumWidth(120)
        layout.addWidget(self.value_widget)
        
        self.value2_widget = QLineEdit()
        self.value2_widget.setText(self.condition.value2)
        self.value2_widget.setMinimumWidth(120)
        self.value2_widget.setVisible(False)
        layout.addWidget(self.value2_widget)
        
        self.range_label = QLabel("AND")
        self.range_label.setVisible(False)
        layout.addWidget(self.range_label)
        
        self.remove_button = QPushButton("✕")
        self.remove_button.setFixedSize(25, 25)
        self.remove_button.setToolTip("Remove condition")
        self.remove_button.clicked.connect(self.removeRequested.emit)
        layout.addWidget(self.remove_button)
        
        self.setLayout(layout)
        
    def connectSignals(self):
        """Connect widget signals"""
        self.logic_combo.currentTextChanged.connect(self.onConditionChanged)
        self.column_combo.currentTextChanged.connect(self.onColumnChanged)
        self.operator_combo.currentTextChanged.connect(self.onOperatorChanged)
        self.value_widget.textChanged.connect(self.onConditionChanged)
        self.value2_widget.textChanged.connect(self.onConditionChanged)
        
    def onColumnChanged(self):
        """Handle column selection change"""
        self.updateOperators()
        self.onConditionChanged()
        
    def onOperatorChanged(self):
        """Handle operator selection change"""
        self.updateValueWidgets()
        self.onConditionChanged()
        
    def onConditionChanged(self):
        """Handle any condition change"""
        self.updateConditionFromUI()
        self.conditionChanged.emit()
        
    def updateOperators(self):
        """Update operator options based on selected column"""
        column = self.column_combo.currentText()
        if not column or column not in self.column_info:
            return
            
        column_type = self.column_info[column].get('dtype', 'object')
        
        self.operator_combo.clear()
        
        operators = ['=', '!=']
        
        if pd.api.types.is_numeric_dtype(column_type):
            operators.extend(['>', '<', '>=', '<=', 'between', 'not between'])
        elif pd.api.types.is_datetime64_any_dtype(column_type):
            operators.extend(['>', '<', '>=', '<=', 'between', 'not between'])
        
        if column_type == 'object':
            operators.extend(['contains', 'not contains', 'starts with', 'ends with',
                            'in list', 'not in list'])
        
        operators.extend(['is null', 'is not null'])
        
        self.operator_combo.addItems(operators)
        
        if self.condition.operator in operators:
            self.operator_combo.setCurrentText(self.condition.operator)
            
    def updateValueWidgets(self):
        """Update value input widgets based on operator"""
        operator = self.operator_combo.currentText()
        
        is_range = operator in ['between', 'not between']
        is_null = operator in ['is null', 'is not null']
        is_list = operator in ['in list', 'not in list']
        
        self.value_widget.setVisible(not is_null)
        self.value2_widget.setVisible(is_range)
        self.range_label.setVisible(is_range)
        
        if is_list:
            self.value_widget.setPlaceholderText("value1, value2, value3")
        elif is_range:
            self.value_widget.setPlaceholderText("From")
            self.value2_widget.setPlaceholderText("To")
        else:
            self.value_widget.setPlaceholderText("Value")
            
    def updateConditionFromUI(self):
        """Update the condition object from UI values"""
        self.condition.logic = self.logic_combo.currentText()
        self.condition.column = self.column_combo.currentText()
        self.condition.operator = self.operator_combo.currentText()
        self.condition.value = self.value_widget.text()
        self.condition.value2 = self.value2_widget.text()
        
        operator = self.condition.operator
        if operator in ['between', 'not between']:
            self.condition.condition_type = 'range'
        elif operator in ['in list', 'not in list']:
            self.condition.condition_type = 'list'
            self.condition.value_list = [v.strip() for v in self.condition.value.split(',') if v.strip()]
        elif operator in ['is null', 'is not null']:
            self.condition.condition_type = 'null'
        else:
            self.condition.condition_type = 'simple'
            
    def setLogicVisible(self, visible: bool):
        """Set visibility of logic connector"""
        self.logic_combo.setVisible(visible)


class AdvancedDataFilterDialog(QDialog):
    """Advanced data filtering dialog with comprehensive options"""
    
    def __init__(self, dataframe: pd.DataFrame, parent=None):
        super().__init__(parent)
        
        self.dataframe = dataframe
        self.filtered_dataframe = None
        self.conditions = []
        self.condition_widgets = []
        
        self.column_info = self._analyze_columns()
        
        self.setWindowTitle("Advanced Data Filter")
        self.setModal(True)
        self.resize(1000, 700)
        
        self.setupUI()
        self.addCondition()  # Start with one condition
        
    def setupUI(self):
        """Setup the main UI"""
        layout = QVBoxLayout()
        
        header_label = QLabel("Advanced Data Filter")
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header_label)
        
        info_label = QLabel(f"Dataset: {len(self.dataframe)} rows, {len(self.dataframe.columns)} columns")
        layout.addWidget(info_label)
        
        splitter = QSplitter(Qt.Vertical)
        
        conditions_widget = self.createConditionsPanel()
        splitter.addWidget(conditions_widget)
        
        preview_widget = self.createPreviewPanel()
        splitter.addWidget(preview_widget)
        
        splitter.setSizes([280, 420])
        
        layout.addWidget(splitter)
        
        button_layout = QHBoxLayout()
        
        self.save_filter_button = QPushButton("Save Filter")
        self.load_filter_button = QPushButton("Load Filter")
        self.clear_button = QPushButton("Clear All")
        
        button_layout.addWidget(self.save_filter_button)
        button_layout.addWidget(self.load_filter_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        
        self.preview_button = QPushButton("Preview")
        self.apply_button = QPushButton("Apply Filter")
        self.cancel_button = QPushButton("Cancel")
        
        self.preview_button.clicked.connect(self.previewFilter)
        self.apply_button.clicked.connect(self.applyFilter)
        self.cancel_button.clicked.connect(self.reject)
        self.clear_button.clicked.connect(self.clearConditions)
        
        button_layout.addWidget(self.preview_button)
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def createConditionsPanel(self) -> QWidget:
        """Create the conditions configuration panel"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        title_label = QLabel("Filter Conditions")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        
        self.conditions_scroll = QScrollArea()
        self.conditions_scroll.setWidgetResizable(True)
        self.conditions_scroll.setMinimumHeight(200)
        
        self.conditions_widget = QWidget()
        self.conditions_layout = QVBoxLayout()
        self.conditions_widget.setLayout(self.conditions_layout)
        self.conditions_scroll.setWidget(self.conditions_widget)
        
        layout.addWidget(self.conditions_scroll)
        
        add_button = QPushButton("+ Add Condition")
        add_button.clicked.connect(self.addCondition)
        layout.addWidget(add_button)
        
        widget.setLayout(layout)
        return widget
        
    def createPreviewPanel(self) -> QWidget:
        """Create the data preview panel"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        preview_header = QHBoxLayout()
        title_label = QLabel("Preview")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        preview_header.addWidget(title_label)
        
        self.preview_info_label = QLabel("")
        preview_header.addStretch()
        preview_header.addWidget(self.preview_info_label)
        
        layout.addLayout(preview_header)
        
        self.preview_table = QTableWidget()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setSortingEnabled(True)
        layout.addWidget(self.preview_table)
        
        widget.setLayout(layout)
        return widget
        
    def _analyze_columns(self) -> Dict[str, Dict[str, Any]]:
        """Analyze dataframe columns for type information"""
        column_info = {}
        
        for col in self.dataframe.columns:
            series = self.dataframe[col]
            
            info = {
                'dtype': series.dtype,
                'unique_count': series.nunique(),
                'null_count': series.isnull().sum(),
                'sample_values': series.dropna().head(10).tolist()
            }
            
            if pd.api.types.is_numeric_dtype(series):
                info['min_value'] = series.min()
                info['max_value'] = series.max()
                
            column_info[col] = info
            
        return column_info
        
    def addCondition(self):
        """Add a new filter condition"""
        condition = FilterCondition()
        condition_widget = ConditionWidget(condition, self.column_info)
        
        condition_widget.conditionChanged.connect(self.onConditionChanged)
        condition_widget.removeRequested.connect(lambda: self.removeCondition(condition_widget))
        
        if len(self.condition_widgets) == 0:
            condition_widget.setLogicVisible(False)
            
        self.conditions.append(condition)
        self.condition_widgets.append(condition_widget)
        self.conditions_layout.addWidget(condition_widget)
        
        if len(self.condition_widgets) == 1:
            QTimer.singleShot(100, self.previewFilter)  # Delay to allow UI to update
            
    def removeCondition(self, condition_widget: ConditionWidget):
        """Remove a filter condition"""
        if len(self.condition_widgets) <= 1:
            return  # Keep at least one condition
            
        index = self.condition_widgets.index(condition_widget)
        
        self.conditions.pop(index)
        self.condition_widgets.pop(index)
        
        self.conditions_layout.removeWidget(condition_widget)
        condition_widget.deleteLater()
        
        if self.condition_widgets:
            self.condition_widgets[0].setLogicVisible(False)
            
        self.previewFilter()
        
    def clearConditions(self):
        """Clear all conditions"""
        while self.condition_widgets:
            widget = self.condition_widgets.pop()
            self.conditions_layout.removeWidget(widget)
            widget.deleteLater()
            
        self.conditions.clear()
        
        self.addCondition()
        
    def onConditionChanged(self):
        """Handle condition change"""
        QTimer.singleShot(500, self.previewFilter)
        
    def previewFilter(self):
        """Preview the filtered data"""
        try:
            query = self.buildFilterQuery()
            
            if not query.strip():
                self.filtered_dataframe = self.dataframe.copy()
            else:
                self.filtered_dataframe = self.dataframe.query(query)
                
            self.updatePreviewTable()
            
            original_count = len(self.dataframe)
            filtered_count = len(self.filtered_dataframe)
            percentage = (filtered_count / original_count * 100) if original_count > 0 else 0
            
            self.preview_info_label.setText(
                f"Showing {filtered_count:,} of {original_count:,} rows ({percentage:.1f}%)"
            )
            
        except Exception as e:
            logger.error(f"Error previewing filter: {e}")
            self.preview_info_label.setText(f"Filter error: {str(e)}")
            
    def buildFilterQuery(self) -> str:
        """Build pandas query string from conditions"""
        query_parts = []
        
        for i, condition in enumerate(self.conditions):
            if not condition.column or not condition.operator:
                continue
                
            condition_str = self.buildConditionString(condition)
            if not condition_str:
                continue
                
            if i > 0 and query_parts:
                query_parts.append(f" {condition.logic.upper()} ")
                
            query_parts.append(f"({condition_str})")
            
        return "".join(query_parts)
        
    def buildConditionString(self, condition: FilterCondition) -> str:
        """Build query string for a single condition"""
        column = condition.column
        operator = condition.operator
        value = condition.value
        value2 = condition.value2
        
        if operator == 'is null':
            return f"`{column}`.isnull()"
        elif operator == 'is not null':
            return f"`{column}`.notnull()"
            
        column_info = self.column_info.get(column, {})
        is_numeric = pd.api.types.is_numeric_dtype(column_info.get('dtype', 'object'))
        
        if is_numeric:
            try:
                value = float(value) if value else 0
                value2 = float(value2) if value2 else 0
            except ValueError:
                return ""  # Invalid numeric value
        else:
            value = f"'{value}'" if value else "''"
            value2 = f"'{value2}'" if value2 else "''"
            
        if operator == '=':
            return f"`{column}` == {value}"
        elif operator == '!=':
            return f"`{column}` != {value}"
        elif operator == '>':
            return f"`{column}` > {value}"
        elif operator == '<':
            return f"`{column}` < {value}"
        elif operator == '>=':
            return f"`{column}` >= {value}"
        elif operator == '<=':
            return f"`{column}` <= {value}"
        elif operator == 'between':
            return f"({value} <= `{column}` <= {value2})"
        elif operator == 'not between':
            return f"not ({value} <= `{column}` <= {value2})"
        elif operator == 'contains':
            return f"`{column}`.str.contains({value}, na=False)"
        elif operator == 'not contains':
            return f"not `{column}`.str.contains({value}, na=False)"
        elif operator == 'starts with':
            return f"`{column}`.str.startswith({value}, na=False)"
        elif operator == 'ends with':
            return f"`{column}`.str.endswith({value}, na=False)"
        elif operator == 'in list':
            values = [v.strip().strip("'\"") for v in condition.value.split(',') if v.strip()]
            if is_numeric:
                try:
                    values = [float(v) for v in values]
                except ValueError:
                    return ""
            else:
                values = [f"'{v}'" for v in values]
            return f"`{column}` in [{', '.join(map(str, values))}]"
        elif operator == 'not in list':
            values = [v.strip().strip("'\"") for v in condition.value.split(',') if v.strip()]
            if is_numeric:
                try:
                    values = [float(v) for v in values]
                except ValueError:
                    return ""
            else:
                values = [f"'{v}'" for v in values]
            return f"`{column}` not in [{', '.join(map(str, values))}]"
            
        return ""
        
    def updatePreviewTable(self):
        """Update the preview table with filtered data"""
        if self.filtered_dataframe is None or self.filtered_dataframe.empty:
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
            return
            
        preview_df = self.filtered_dataframe.head(100).iloc[:, :10]
        
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
        
    def applyFilter(self):
        """Apply the filter and close dialog"""
        self.previewFilter()  # Ensure latest filter is applied
        self.accept()
        
    def getFilteredData(self) -> pd.DataFrame:
        """Get the filtered dataframe"""
        return self.filtered_dataframe if self.filtered_dataframe is not None else self.dataframe.copy()
        
    def getFilterConditions(self) -> List[Dict[str, Any]]:
        """Get the filter conditions as a list of dictionaries"""
        return [condition.to_dict() for condition in self.conditions]
        
    def setFilterConditions(self, conditions: List[Dict[str, Any]]):
        """Set filter conditions from a list of dictionaries"""
        self.clearConditions()
        
        for condition_data in conditions:
            condition = FilterCondition.from_dict(condition_data)
            condition_widget = ConditionWidget(condition, self.column_info)
            
            condition_widget.conditionChanged.connect(self.onConditionChanged)
            condition_widget.removeRequested.connect(lambda: self.removeCondition(condition_widget))
            
            if len(self.condition_widgets) == 0:
                condition_widget.setLogicVisible(False)
                
            self.conditions.append(condition)
            self.condition_widgets.append(condition_widget)
            self.conditions_layout.addWidget(condition_widget)
            
        self.previewFilter()