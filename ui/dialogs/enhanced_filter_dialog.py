#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Filter Dialog for Bespoke Utility
Provides an advanced interface for filtering data with complex conditions

"""

import logging
from typing import Dict, List, Any, Optional
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, 
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QGroupBox,
    QScrollArea, QWidget, QFrame, QSplitter, QTreeWidget, QTreeWidgetItem,
    QSpacerItem, QSizePolicy, QButtonGroup, QRadioButton, QCheckBox,
    QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit, QTextEdit
)
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor

logger = logging.getLogger(__name__)

class FilterConditionWidget(QWidget):
    """Widget representing a single filter condition with modern UI"""
    
    condition_changed = pyqtSignal()
    delete_requested = pyqtSignal()
    
    def __init__(self, columns: List[str], condition_data: Dict = None, column_types: Dict[str, str] = None, parent=None):
        super().__init__(parent)
        self.columns = columns
        self.condition_data = condition_data or {}
        self.column_types = column_types or {}
        self.setup_ui()
        self.load_condition_data()
        
    def setup_ui(self):
        """Setup the UI for a single filter condition"""
        self.setFixedHeight(80)
        self.setStyleSheet("""
            FilterConditionWidget {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                margin: 2px;
            }
            FilterConditionWidget:hover {
                border-color: #007bff;
                background-color: #f0f8ff;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)
        
        self.group_indicator = QLabel()
        self.group_indicator.setFixedWidth(20)
        self.group_indicator.setAlignment(Qt.AlignCenter)
        self.group_indicator.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #6c757d;
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 2px;
            }
        """)
        layout.addWidget(self.group_indicator)
        
        self.logic_combo = QComboBox()
        self.logic_combo.addItems(['AND', 'OR'])
        self.logic_combo.setFixedSize(60, 32)
        self.logic_combo.setStyleSheet("""
            QComboBox {
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                font-size: 12px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                color: white;
            }
        """)
        self.logic_combo.currentTextChanged.connect(self.condition_changed.emit)
        self.logic_combo.currentTextChanged.connect(self.update_visual_grouping)
        layout.addWidget(self.logic_combo)
        
        self.column_combo = QComboBox()
        self.column_combo.addItems(self.columns)
        self.column_combo.setMinimumWidth(150)
        self.column_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 6px 8px;
                font-size: 13px;
            }
            QComboBox:focus {
                border-color: #007bff;
            }
        """)
        self.column_combo.currentTextChanged.connect(self.condition_changed.emit)
        layout.addWidget(self.column_combo)
        
        self.operator_combo = QComboBox()
        self.operator_combo.addItems([
            '==', '!=', '>', '<', '>=', '<=',
            'contains', 'starts_with', 'ends_with',
            'is_null', 'not_null', 'in', 'not_in'
        ])
        self.operator_combo.setFixedWidth(120)
        self.operator_combo.setStyleSheet("""
            QComboBox {
                background-color: white;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 6px 8px;
                font-size: 13px;
            }
            QComboBox:focus {
                border-color: #007bff;
            }
        """)
        self.operator_combo.currentTextChanged.connect(self._on_operator_changed)
        self.operator_combo.currentTextChanged.connect(self.condition_changed.emit)
        layout.addWidget(self.operator_combo)
        
        self.value_widget = QLineEdit()
        self.value_widget.setPlaceholderText("Enter value...")
        self.value_widget.setMinimumWidth(120)
        self.value_widget.setStyleSheet("""
            QLineEdit {
                background-color: white;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 6px 8px;
                font-size: 13px;
            }
            QLineEdit:focus {
                border-color: #007bff;
            }
        """)
        self.value_widget.textChanged.connect(self.condition_changed.emit)
        self.value_widget.textChanged.connect(self._validate_value)
        self.column_combo.currentTextChanged.connect(self._update_value_placeholder)
        layout.addWidget(self.value_widget)
        
        self.delete_btn = QPushButton("🗑️")
        self.delete_btn.setFixedSize(32, 32)
        self.delete_btn.setToolTip("Delete this condition")
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                border-radius: 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:pressed {
                background-color: #bd2130;
            }
        """)
        self.delete_btn.clicked.connect(self.delete_requested.emit)
        layout.addWidget(self.delete_btn)
        
    def _on_operator_changed(self, operator):
        """Handle operator change to update value input widget"""
        if operator in ['is_null', 'not_null']:
            self.value_widget.setEnabled(False)
            self.value_widget.setPlaceholderText("No value needed")
        elif operator in ['in', 'not_in']:
            self.value_widget.setEnabled(True)
            self.value_widget.setPlaceholderText("Enter comma-separated values")
        else:
            self.value_widget.setEnabled(True)
            self._update_value_placeholder()
    
    def _update_value_placeholder(self):
        """Update placeholder text based on column type"""
        column = self.column_combo.currentText()
        if column and column in self.column_types:
            column_type = self.column_types[column]
            
            if 'int' in column_type.lower() or 'uint' in column_type.lower():
                if column_type.lower() in ['uint8', 'int8']:
                    self.value_widget.setPlaceholderText("Enter 0/1 or Y/N for flags")
                else:
                    self.value_widget.setPlaceholderText("Enter integer value")
            elif 'float' in column_type.lower():
                self.value_widget.setPlaceholderText("Enter numeric value")
            elif 'bool' in column_type.lower():
                self.value_widget.setPlaceholderText("Enter True/False, 1/0, or Y/N")
            else:
                self.value_widget.setPlaceholderText("Enter text value")
        else:
            self.value_widget.setPlaceholderText("Enter value...")
    
    def _validate_value(self):
        """Validate the entered value and provide visual feedback"""
        try:
            self._get_typed_value()
            self.value_widget.setStyleSheet("""
                QLineEdit {
                    background-color: white;
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    padding: 6px 8px;
                    font-size: 13px;
                }
                QLineEdit:focus {
                    border-color: #007bff;
                }
            """)
        except ValueError as e:
            self.value_widget.setStyleSheet("""
                QLineEdit {
                    background-color: #fff5f5;
                    border: 1px solid #dc3545;
                    border-radius: 4px;
                    padding: 6px 8px;
                    font-size: 13px;
                }
                QLineEdit:focus {
                    border-color: #dc3545;
                }
            """)
            self.value_widget.setToolTip(str(e))
    
    def set_logic_visible(self, visible: bool):
        """Show/hide the logic connector and group indicator"""
        self.logic_combo.setVisible(visible)
        self.group_indicator.setVisible(visible)
        if visible:
            self.update_visual_grouping()
    
    def update_visual_grouping(self):
        """Update visual grouping indicators based on logical operators"""
        if not self.logic_combo.isVisible():
            return
            
        current_logic = self.logic_combo.currentText()
        
        if current_logic == 'AND':
            self.group_indicator.setText('∧')
            self.group_indicator.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    color: #28a745;
                    background-color: #d4edda;
                    border: 2px solid #28a745;
                    border-radius: 4px;
                    padding: 2px;
                }
            """)
            self.setStyleSheet("""
                FilterConditionWidget {
                    background-color: #f0fff0;
                    border: 2px solid #28a745;
                    border-left: 6px solid #28a745;
                    border-radius: 8px;
                    margin: 2px 5px 2px 10px;
                    padding: 4px;
                }
                FilterConditionWidget:before {
                    content: "(";
                    color: #28a745;
                    font-weight: bold;
                }
            """)
        else:  # OR
            self.group_indicator.setText('∨')
            self.group_indicator.setStyleSheet("""
                QLabel {
                    font-size: 16px;
                    font-weight: bold;
                    color: #fd7e14;
                    background-color: #fff3cd;
                    border: 2px solid #fd7e14;
                    border-radius: 4px;
                    padding: 2px;
                }
            """)
            self.setStyleSheet("""
                FilterConditionWidget {
                    background-color: #fffbf0;
                    border: 2px dashed #fd7e14;
                    border-left: 6px solid #fd7e14;
                    border-radius: 8px;
                    margin: 2px 5px 2px 5px;
                    padding: 4px;
                }
            """)
            
        if hasattr(self.parent(), 'update_grouping_preview'):
            self.parent().update_grouping_preview()
    
    def load_condition_data(self):
        """Load condition data into the widgets"""
        if self.condition_data:
            if 'column' in self.condition_data:
                index = self.column_combo.findText(self.condition_data['column'])
                if index >= 0:
                    self.column_combo.setCurrentIndex(index)
            
            if 'operator' in self.condition_data:
                index = self.operator_combo.findText(self.condition_data['operator'])
                if index >= 0:
                    self.operator_combo.setCurrentIndex(index)
            
            if 'value' in self.condition_data:
                self.value_widget.setText(str(self.condition_data['value']))
            
            if 'logic' in self.condition_data:
                index = self.logic_combo.findText(self.condition_data['logic'])
                if index >= 0:
                    self.logic_combo.setCurrentIndex(index)
    
    def get_condition_data(self) -> Dict:
        """Get the current condition data"""
        return {
            'column': self.column_combo.currentText(),
            'operator': self.operator_combo.currentText(),
            'value': self._get_typed_value(),
            'logic': self.logic_combo.currentText()
        }
    
    def _get_typed_value(self):
        """Get the value with appropriate type conversion based on column type"""
        text = self.value_widget.text().strip()
        operator = self.operator_combo.currentText()
        column = self.column_combo.currentText()
        
        if operator in ['is_null', 'not_null']:
            return None
        elif operator in ['in', 'not_in']:
            return [v.strip() for v in text.split(',') if v.strip()]
        else:
            if not text:
                return None
                
            column_type = self.column_types.get(column, 'object')
            
            if 'int' in column_type.lower() or 'uint' in column_type.lower():
                try:
                    return int(text)
                except ValueError:
                    if column_type.lower() in ['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32']:
                        return text  # Let data processor handle categorical text values
                    raise ValueError(f"'{text}' is not a valid integer value for column '{column}' (type: {column_type})")
            
            elif 'float' in column_type.lower():
                try:
                    return float(text)
                except ValueError:
                    raise ValueError(f"'{text}' is not a valid numeric value for column '{column}' (type: {column_type})")
            
            elif 'bool' in column_type.lower():
                if text.upper() in ['TRUE', '1', 'Y', 'YES']:
                    return True
                elif text.upper() in ['FALSE', '0', 'N', 'NO']:
                    return False
                else:
                    raise ValueError(f"'{text}' is not a valid boolean value. Use: True/False, 1/0, Y/N")
            
            else:
                operator = self.operator_combo.currentText()
                if operator in ['in', 'not_in']:
                    if ',' in text:
                        return [v.strip() for v in text.split(',') if v.strip()]
                    else:
                        return [text.strip()]
                else:
                    return text.strip()


class EnhancedFilterDialog(QDialog):
    """Enhanced Filter Dialog with modern UI and advanced functionality"""
    
    def __init__(self, filter_node, available_columns: List[str], parent=None, column_types: Dict[str, str] = None):
        super().__init__(parent)
        self.filter_node = filter_node
        self.available_columns = available_columns
        self.column_types = column_types or {}
        self.condition_widgets = []
        
        self.setWindowTitle("🔍 Configure Filter")
        self.setModal(True)
        self.resize(800, 600)
        self.setup_ui()
        self.load_existing_conditions()
        self.apply_modern_styling()
    
    def setup_ui(self):
        """Setup the main UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        header_layout = QHBoxLayout()
        
        title_label = QLabel("Filter Configuration")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        self.add_condition_btn = QPushButton("➕ Add Condition")
        self.add_condition_btn.clicked.connect(self.add_condition)
        header_layout.addWidget(self.add_condition_btn)
        
        layout.addLayout(header_layout)
        
        splitter = QSplitter(Qt.Horizontal)
        
        conditions_group = QGroupBox("Filter Conditions")
        conditions_layout = QVBoxLayout(conditions_group)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(300)
        
        self.conditions_widget = QWidget()
        self.conditions_layout = QVBoxLayout(self.conditions_widget)
        self.conditions_layout.setSpacing(8)
        self.conditions_layout.addStretch()
        
        self.scroll_area.setWidget(self.conditions_widget)
        conditions_layout.addWidget(self.scroll_area)
        
        condition_buttons_layout = QHBoxLayout()
        
        self.clear_all_btn = QPushButton("🧹 Clear All")
        self.clear_all_btn.clicked.connect(self.clear_all_conditions)
        condition_buttons_layout.addWidget(self.clear_all_btn)
        
        condition_buttons_layout.addStretch()
        
        self.validate_btn = QPushButton("✓ Validate")
        self.validate_btn.clicked.connect(self.validate_conditions)
        condition_buttons_layout.addWidget(self.validate_btn)
        
        conditions_layout.addLayout(condition_buttons_layout)
        splitter.addWidget(conditions_group)
        
        preview_group = QGroupBox("Filter Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_text = QTextEdit()
        self.preview_text.setMaximumHeight(150)
        self.preview_text.setReadOnly(True)
        self.preview_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                padding: 8px;
            }
        """)
        preview_layout.addWidget(QLabel("Filter Expression:"))
        preview_layout.addWidget(self.preview_text)
        
        stats_label = QLabel("Filter Statistics")
        stats_font = QFont()
        stats_font.setBold(True)
        stats_label.setFont(stats_font)
        preview_layout.addWidget(stats_label)
        
        self.stats_text = QLabel("No conditions defined")
        self.stats_text.setStyleSheet("""
            QLabel {
                background-color: #e9ecef;
                border-radius: 4px;
                padding: 8px;
                color: #6c757d;
            }
        """)
        preview_layout.addWidget(self.stats_text)
        
        preview_layout.addStretch()
        splitter.addWidget(preview_group)
        
        splitter.setSizes([500, 300])
        layout.addWidget(splitter)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setDefault(True)
        button_layout.addWidget(self.ok_btn)
        
        layout.addLayout(button_layout)
    
    def apply_modern_styling(self):
        """Apply modern styling to the dialog"""
        self.setStyleSheet("""
            QDialog {
                background-color: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                margin: 8px 0px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px 0 8px;
                background-color: #ffffff;
                color: #495057;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 500;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
            QPushButton#add_condition_btn {
                background-color: #28a745;
            }
            QPushButton#add_condition_btn:hover {
                background-color: #218838;
            }
            QPushButton#clear_all_btn {
                background-color: #6c757d;
            }
            QPushButton#clear_all_btn:hover {
                background-color: #545b62;
            }
            QPushButton#validate_btn {
                background-color: #17a2b8;
            }
            QPushButton#validate_btn:hover {
                background-color: #138496;
            }
        """)
        
        self.add_condition_btn.setObjectName("add_condition_btn")
        self.clear_all_btn.setObjectName("clear_all_btn")
        self.validate_btn.setObjectName("validate_btn")
    
    def add_condition(self, condition_data: Dict = None):
        """Add a new condition widget"""
        if not self.available_columns:
            QMessageBox.warning(self, "No Columns", "No columns available for filtering.")
            return
        
        self.conditions_layout.takeAt(self.conditions_layout.count() - 1)
        
        condition_widget = FilterConditionWidget(self.available_columns, condition_data, self.column_types, self)
        condition_widget.condition_changed.connect(self.update_preview)
        condition_widget.delete_requested.connect(lambda: self.delete_condition(condition_widget))
        
        if len(self.condition_widgets) == 0:
            condition_widget.set_logic_visible(False)
        
        self.condition_widgets.append(condition_widget)
        self.conditions_layout.addWidget(condition_widget)
        
        self.conditions_layout.addStretch()
        
        self.update_preview()
        
        logger.info(f"Added filter condition widget. Total conditions: {len(self.condition_widgets)}")
    
    def delete_condition(self, condition_widget: FilterConditionWidget):
        """Delete a condition widget"""
        if condition_widget in self.condition_widgets:
            self.condition_widgets.remove(condition_widget)
            condition_widget.deleteLater()
            
            if self.condition_widgets:
                self.condition_widgets[0].set_logic_visible(False)
            
            self.update_preview()
            
            logger.info(f"Deleted filter condition widget. Remaining conditions: {len(self.condition_widgets)}")
    
    def clear_all_conditions(self):
        """Clear all conditions"""
        reply = QMessageBox.question(
            self, "Clear All Conditions",
            "Are you sure you want to clear all filter conditions?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            for widget in self.condition_widgets:
                widget.deleteLater()
            self.condition_widgets.clear()
            self.update_preview()
            
            logger.info("Cleared all filter conditions")
    
    def validate_conditions(self):
        """Validate all conditions"""
        issues = []
        
        for i, widget in enumerate(self.condition_widgets):
            condition = widget.get_condition_data()
            
            operator = condition['operator']
            if operator not in ['is_null', 'not_null'] and not condition['value']:
                issues.append(f"Condition {i+1}: Missing value for operator '{operator}'")
            
            if not condition['column']:
                issues.append(f"Condition {i+1}: No column selected")
        
        if issues:
            QMessageBox.warning(self, "Validation Issues", "\\n".join(issues))
        else:
            QMessageBox.information(self, "Validation", "All conditions are valid!")
    
    def update_preview(self):
        """Update the filter preview text"""
        if not self.condition_widgets:
            self.preview_text.setText("No conditions defined")
            self.stats_text.setText("No conditions defined")
            return
        
        expression_parts = []
        
        for i, widget in enumerate(self.condition_widgets):
            condition = widget.get_condition_data()
            
            if i > 0:
                expression_parts.append(f" {condition['logic']} ")
            
            column = condition['column']
            operator = condition['operator']
            value = condition['value']
            
            if operator == 'is_null':
                expr = f"{column} IS NULL"
            elif operator == 'not_null':
                expr = f"{column} IS NOT NULL"
            elif operator in ['in', 'not_in']:
                value_str = f"({', '.join(map(str, value)) if value else ''})"
                expr = f"{column} {'IN' if operator == 'in' else 'NOT IN'} {value_str}"
            elif operator == 'contains':
                expr = f"{column} LIKE '%{value}%'"
            elif operator == 'starts_with':
                expr = f"{column} LIKE '{value}%'"
            elif operator == 'ends_with':
                expr = f"{column} LIKE '%{value}'"
            else:
                expr = f"{column} {operator} {repr(value) if isinstance(value, str) else value}"
            
            expression_parts.append(expr)
        
        full_expression = "".join(expression_parts)
        self.preview_text.setText(f"WHERE {full_expression}")
        
        self.stats_text.setText(f"Total conditions: {len(self.condition_widgets)}")
    
    def load_existing_conditions(self):
        """Load existing conditions from the filter node"""
        if hasattr(self.filter_node, 'conditions') and self.filter_node.conditions:
            for condition_data in self.filter_node.conditions:
                self.add_condition(condition_data)
            
            logger.info(f"Loaded {len(self.filter_node.conditions)} existing conditions")
    
    def get_conditions(self) -> List[Dict]:
        """Get all conditions from the dialog"""
        conditions = []
        for widget in self.condition_widgets:
            condition_data = widget.get_condition_data()
            if len(conditions) == 0:
                condition_data.pop('logic', None)
            conditions.append(condition_data)
        return conditions
    
    def accept(self):
        """Accept the dialog and save conditions"""
        conditions = self.get_conditions()
        
        if hasattr(self.filter_node, 'clear_conditions'):
            self.filter_node.clear_conditions()
        
        for condition in conditions:
            if hasattr(self.filter_node, 'add_condition'):
                logic = condition.pop('logic', None)
                self.filter_node.add_condition(
                    condition['column'],
                    condition['operator'],
                    condition['value'],
                    logic
                )
        
        logger.info(f"Saved {len(conditions)} filter conditions to node {self.filter_node.node_id}")
        super().accept()