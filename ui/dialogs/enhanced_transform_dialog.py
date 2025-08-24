#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Data Transformation Dialog for Bespoke Utility
Provides a comprehensive interface for data transformations with real-time previews

"""

import logging
from typing import Dict, List, Any, Optional
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, 
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QGroupBox,
    QScrollArea, QWidget, QFrame, QSplitter, QTreeWidget, QTreeWidgetItem,
    QSpacerItem, QSizePolicy, QButtonGroup, QRadioButton, QCheckBox,
    QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PyQt5.QtGui import QFont, QIcon, QPalette, QColor

logger = logging.getLogger(__name__)

class TransformationWidget(QWidget):
    """Widget representing a single transformation with modern UI"""
    
    transformation_changed = pyqtSignal()
    delete_requested = pyqtSignal()
    edit_requested = pyqtSignal()
    
    def __init__(self, columns: List[str], transform_data: Dict = None, parent=None):
        super().__init__(parent)
        self.columns = columns
        self.transform_data = transform_data or {}
        self.setup_ui()
        self.load_transform_data()
        
    def setup_ui(self):
        """Setup the UI for a single transformation"""
        self.setFixedHeight(100)
        self.setStyleSheet("""
            TransformationWidget {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                margin: 2px;
            }
            TransformationWidget:hover {
                border-color: #28a745;
                background-color: #f0fff4;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        
        top_layout = QHBoxLayout()
        
        self.type_label = QLabel("create_variable")
        self.type_label.setStyleSheet("""
            QLabel {
                background-color: #28a745;
                color: white;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                font-size: 11px;
                max-width: 120px;
            }
        """)
        top_layout.addWidget(self.type_label)
        
        arrow_label = QLabel("→")
        arrow_label.setStyleSheet("font-size: 16px; color: #6c757d; font-weight: bold;")
        top_layout.addWidget(arrow_label)
        
        self.target_label = QLabel("new_variable")
        self.target_label.setStyleSheet("""
            QLabel {
                background-color: #007bff;
                color: white;
                border-radius: 4px;
                padding: 4px 8px;
                font-weight: bold;
                font-size: 11px;
                min-width: 100px;
            }
        """)
        top_layout.addWidget(self.target_label)
        
        top_layout.addStretch()
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(4)
        
        self.edit_btn = QPushButton("✏️")
        self.edit_btn.setFixedSize(28, 28)
        self.edit_btn.setToolTip("Edit transformation")
        self.edit_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                border: none;
                border-radius: 14px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        self.edit_btn.clicked.connect(self.edit_requested.emit)
        button_layout.addWidget(self.edit_btn)
        
        self.delete_btn = QPushButton("🗑️")
        self.delete_btn.setFixedSize(28, 28)
        self.delete_btn.setToolTip("Delete transformation")
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                border-radius: 14px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        self.delete_btn.clicked.connect(self.delete_requested.emit)
        button_layout.addWidget(self.delete_btn)
        
        top_layout.addLayout(button_layout)
        layout.addLayout(top_layout)
        
        self.details_label = QLabel("Source: column1, column2 | Formula: column1 + column2")
        self.details_label.setStyleSheet("""
            QLabel {
                color: #6c757d;
                font-size: 11px;
                font-style: italic;
                margin-top: 2px;
                padding: 2px 4px;
                background-color: #e9ecef;
                border-radius: 3px;
            }
        """)
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)
        
    def load_transform_data(self):
        """Load transformation data into the widgets"""
        if self.transform_data:
            transform_type = self.transform_data.get('type', 'unknown')
            self.type_label.setText(transform_type)
            
            type_colors = {
                'create_variable': '#28a745',
                'derive_ratio': '#fd7e14',
                'derive_difference': '#6f42c1',
                'derive_sum': '#20c997',
                'encode_categorical': '#e83e8c',
                'log_transform': '#ffc107',
                'standardize': '#6c757d',
                'binning': '#17a2b8'
            }
            color = type_colors.get(transform_type, '#28a745')
            self.type_label.setStyleSheet(f"""
                QLabel {{
                    background-color: {color};
                    color: white;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-weight: bold;
                    font-size: 11px;
                    max-width: 120px;
                }}
            """)
            
            target_column = self.transform_data.get('target_column', 'new_variable')
            self.target_label.setText(target_column)
            
            source_columns = self.transform_data.get('source_columns', [])
            formula = self.transform_data.get('formula', '')
            
            details_parts = []
            if source_columns:
                details_parts.append(f"Source: {', '.join(source_columns)}")
            if formula:
                details_parts.append(f"Formula: {formula}")
            
            if details_parts:
                self.details_label.setText(" | ".join(details_parts))
            else:
                self.details_label.setText("No details available")
    
    def get_transform_data(self) -> Dict:
        """Get the current transformation data"""
        return self.transform_data.copy()
    
    def update_transform_data(self, new_data: Dict):
        """Update the transformation data"""
        self.transform_data = new_data
        self.load_transform_data()
        self.transformation_changed.emit()


class TransformationEditDialog(QDialog):
    """Dialog for editing individual transformations"""
    
    def __init__(self, columns: List[str], transform_data: Dict = None, parent=None):
        super().__init__(parent)
        self.columns = columns
        self.transform_data = transform_data or {}
        
        self.setWindowTitle("✏️ Edit Transformation")
        self.setModal(True)
        self.resize(600, 500)
        self.setup_ui()
        self.load_data()
        self.apply_styling()
    
    def setup_ui(self):
        """Setup the edit dialog UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        header_label = QLabel("Transformation Configuration")
        header_font = QFont()
        header_font.setPointSize(14)
        header_font.setBold(True)
        header_label.setFont(header_font)
        layout.addWidget(header_label)
        
        form_layout = QVBoxLayout()
        
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems([
            'create_variable', 'derive_ratio', 'derive_difference', 'derive_sum',
            'encode_categorical', 'log_transform', 'standardize', 'binning'
        ])
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        type_layout.addWidget(self.type_combo)
        type_layout.addStretch()
        form_layout.addLayout(type_layout)
        
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("New Column Name:"))
        self.target_edit = QLineEdit()
        self.target_edit.setPlaceholderText("Enter new column name...")
        target_layout.addWidget(self.target_edit)
        form_layout.addLayout(target_layout)
        
        source_group = QGroupBox("Source Columns")
        source_layout = QVBoxLayout(source_group)
        
        self.columns_list = QListWidget()
        self.columns_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.columns_list.addItems(self.columns)
        source_layout.addWidget(self.columns_list)
        
        selected_layout = QHBoxLayout()
        selected_layout.addWidget(QLabel("Selected:"))
        self.selected_label = QLabel("None")
        self.selected_label.setStyleSheet("""
            QLabel {
                background-color: #e9ecef;
                border-radius: 4px;
                padding: 4px 8px;
                font-style: italic;
            }
        """)
        selected_layout.addWidget(self.selected_label)
        selected_layout.addStretch()
        source_layout.addLayout(selected_layout)
        
        self.columns_list.itemSelectionChanged.connect(self._update_selected_columns)
        form_layout.addWidget(source_group)
        
        self.config_group = QGroupBox("Configuration")
        self.config_layout = QVBoxLayout(self.config_group)
        
        self.formula_widget = QWidget()
        formula_layout = QVBoxLayout(self.formula_widget)
        
        formula_header = QHBoxLayout()
        formula_header.addWidget(QLabel("Formula Editor:"))
        
        help_btn = QPushButton("?")
        help_btn.setFixedSize(20, 20)
        help_btn.setToolTip("Drag variables from Source Columns or click functions below")
        formula_header.addWidget(help_btn)
        formula_header.addStretch()
        formula_layout.addLayout(formula_header)
        
        self.formula_edit = QTextEdit()
        self.formula_edit.setMaximumHeight(120)
        self.formula_edit.setPlaceholderText("Enter formula or drag variables and functions...\nExamples: Income / Age, IF(Credit_Score > 700, 'Good', 'Bad')")
        self.formula_edit.setStyleSheet("""
            QTextEdit {
                border: 2px solid #007bff;
                border-radius: 6px;
                padding: 8px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                background-color: #f8f9fa;
            }
        """)
        formula_layout.addWidget(self.formula_edit)
        
        functions_label = QLabel("Function Palette:")
        functions_label.setFont(QFont("Arial", 10, QFont.Bold))
        formula_layout.addWidget(functions_label)
        
        math_functions = QHBoxLayout()
        math_functions.addWidget(QLabel("Math:"))
        
        for func in ["+", "-", "*", "/", "**", "ABS", "SQRT", "LOG"]:
            btn = QPushButton(func)
            btn.setFixedSize(40, 25)
            btn.clicked.connect(lambda checked, f=func: self._insert_function(f))
            btn.setToolTip(f"Insert {func} function")
            math_functions.addWidget(btn)
        math_functions.addStretch()
        formula_layout.addLayout(math_functions)
        
        logical_functions = QHBoxLayout()
        logical_functions.addWidget(QLabel("Logic:"))
        
        for func in ["IF", "AND", "OR", "NOT", ">", "<", ">=", "<=", "==", "!="]:
            btn = QPushButton(func)
            btn.setFixedSize(40, 25)
            btn.clicked.connect(lambda checked, f=func: self._insert_function(f))
            btn.setToolTip(f"Insert {func} operator")
            logical_functions.addWidget(btn)
        logical_functions.addStretch()
        formula_layout.addLayout(logical_functions)
        
        var_helper = QHBoxLayout()
        insert_var_btn = QPushButton("← Insert Selected Variables")
        insert_var_btn.clicked.connect(self._insert_selected_variables)
        insert_var_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        var_helper.addWidget(insert_var_btn)
        
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(lambda: self.formula_edit.clear())
        var_helper.addWidget(clear_btn)
        var_helper.addStretch()
        formula_layout.addLayout(var_helper)
        
        self.config_layout.addWidget(self.formula_widget)
        
        self.bins_widget = QWidget()
        bins_layout = QHBoxLayout(self.bins_widget)
        bins_layout.addWidget(QLabel("Number of bins:"))
        self.bins_spin = QSpinBox()
        self.bins_spin.setMinimum(2)
        self.bins_spin.setMaximum(20)
        self.bins_spin.setValue(5)
        bins_layout.addWidget(self.bins_spin)
        bins_layout.addStretch()
        self.config_layout.addWidget(self.bins_widget)
        
        # Categorical encoding options
        self.encoding_widget = QWidget()
        encoding_layout = QVBoxLayout(self.encoding_widget)
        encoding_layout.addWidget(QLabel("Encoding method:"))
        self.encoding_combo = QComboBox()
        self.encoding_combo.addItems(["One-Hot", "Label", "Target"])
        encoding_layout.addWidget(self.encoding_combo)
        self.config_layout.addWidget(self.encoding_widget)
        
        self.log_widget = QWidget()
        log_layout = QVBoxLayout(self.log_widget)
        self.log_plus_one = QCheckBox("Add 1 before log transform (for zero values)")
        self.log_plus_one.setChecked(True)
        log_layout.addWidget(self.log_plus_one)
        self.config_layout.addWidget(self.log_widget)
        
        self.standardize_widget = QWidget()
        std_layout = QVBoxLayout(self.standardize_widget)
        std_layout.addWidget(QLabel("Standardization method:"))
        self.standardize_combo = QComboBox()
        self.standardize_combo.addItems(["Z-score (mean=0, std=1)", "Min-Max (0 to 1)", "Robust (median, IQR)"])
        std_layout.addWidget(self.standardize_combo)
        self.config_layout.addWidget(self.standardize_widget)
        
        self.simple_widget = QWidget()
        simple_layout = QVBoxLayout(self.simple_widget)
        simple_layout.addWidget(QLabel("This transformation will be applied to selected columns."))
        simple_layout.addWidget(QLabel("No additional configuration required."))
        self.config_layout.addWidget(self.simple_widget)
        
        form_layout.addWidget(self.config_group)
        layout.addLayout(form_layout)
        
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
        
        self._on_type_changed(self.type_combo.currentText())
    
    def _on_type_changed(self, transform_type):
        """Handle transformation type change"""
        try:
            if hasattr(self, 'formula_widget') and self.formula_widget:
                self.formula_widget.setVisible(False)
            if hasattr(self, 'bins_widget') and self.bins_widget:
                self.bins_widget.setVisible(False)
            if hasattr(self, 'encoding_widget') and self.encoding_widget:
                self.encoding_widget.setVisible(False)
            if hasattr(self, 'log_widget') and self.log_widget:
                self.log_widget.setVisible(False)
            if hasattr(self, 'standardize_widget') and self.standardize_widget:
                self.standardize_widget.setVisible(False)
            if hasattr(self, 'simple_widget') and self.simple_widget:
                self.simple_widget.setVisible(False)
            
            if transform_type == 'create_variable' and hasattr(self, 'formula_widget') and self.formula_widget:
                self.formula_widget.setVisible(True)
            elif transform_type == 'binning' and hasattr(self, 'bins_widget') and self.bins_widget:
                self.bins_widget.setVisible(True)
            elif transform_type == 'encode_categorical' and hasattr(self, 'encoding_widget') and self.encoding_widget:
                self.encoding_widget.setVisible(True)
            elif transform_type == 'log_transform' and hasattr(self, 'log_widget') and self.log_widget:
                self.log_widget.setVisible(True)
            elif transform_type == 'standardize' and hasattr(self, 'standardize_widget') and self.standardize_widget:
                self.standardize_widget.setVisible(True)
            elif transform_type in ['derive_ratio', 'derive_difference', 'derive_sum'] and hasattr(self, 'simple_widget') and self.simple_widget:
                self.simple_widget.setVisible(True)
            
            min_columns = {
                'create_variable': 0,  # Can use formula
                'derive_ratio': 2,
                'derive_difference': 2,
                'derive_sum': 1,
                'encode_categorical': 1,
                'log_transform': 1,
                'standardize': 1,
                'binning': 1
            }
            
            self.min_columns = min_columns.get(transform_type, 1)
                
        except Exception as e:
            logger.error(f"Error changing transformation type to {transform_type}: {e}")
    
    def _insert_function(self, func_name):
        """Insert function or operator into formula editor"""
        try:
            cursor = self.formula_edit.textCursor()
            
            if func_name in ['IF']:
                text = f"{func_name}(condition, true_value, false_value)"
            elif func_name in ['ABS', 'SQRT', 'LOG']:
                text = f"{func_name}(value)"
            elif func_name in ['AND', 'OR']:
                text = f" {func_name} "
            elif func_name in ['>', '<', '>=', '<=', '==', '!=']:
                text = f" {func_name} "
            elif func_name in ['+', '-', '*', '/', '**']:
                text = f" {func_name} "
            else:
                text = func_name
            
            cursor.insertText(text)
            self.formula_edit.setTextCursor(cursor)
            self.formula_edit.setFocus()
            
        except Exception as e:
            logger.error(f"Error inserting function {func_name}: {e}")
    
    def _insert_selected_variables(self):
        """Insert selected variables from the source columns list into formula"""
        try:
            selected_items = self.columns_list.selectedItems()
            if not selected_items:
                QMessageBox.information(self, "No Selection", "Please select variables from the Source Columns list first.")
                return
            
            variable_names = [item.text() for item in selected_items]
            
            cursor = self.formula_edit.textCursor()
            
            if len(variable_names) == 1:
                cursor.insertText(variable_names[0])
            elif len(variable_names) == 2:
                var1, var2 = variable_names[0], variable_names[1]
                cursor.insertText(f"{var1} / {var2}")
            else:
                variables_text = ", ".join(variable_names)
                cursor.insertText(variables_text)
            
            self.formula_edit.setTextCursor(cursor)
            self.formula_edit.setFocus()
            
            self._update_selected_columns()
            
        except Exception as e:
            logger.error(f"Error inserting selected variables: {e}")
            QMessageBox.warning(self, "Error", f"Failed to insert variables: {str(e)}")
    
    def _update_selected_columns(self):
        """Update the selected columns display"""
        selected_items = self.columns_list.selectedItems()
        if selected_items:
            selected_text = ", ".join([item.text() for item in selected_items])
            self.selected_label.setText(selected_text)
        else:
            self.selected_label.setText("None")
    
    def load_data(self):
        """Load existing transformation data"""
        if self.transform_data:
            transform_type = self.transform_data.get('type', 'create_variable')
            index = self.type_combo.findText(transform_type)
            if index >= 0:
                self.type_combo.setCurrentIndex(index)
            
            target_column = self.transform_data.get('target_column', '')
            self.target_edit.setText(target_column)
            
            source_columns = self.transform_data.get('source_columns', [])
            for i in range(self.columns_list.count()):
                item = self.columns_list.item(i)
                if item.text() in source_columns:
                    item.setSelected(True)
            
            formula = self.transform_data.get('formula', '')
            self.formula_edit.setPlainText(formula)
            
            bins = self.transform_data.get('bins', 5)
            self.bins_spin.setValue(bins)
    
    def get_transform_data(self) -> Dict:
        """Get the transformation data from the dialog"""
        selected_items = self.columns_list.selectedItems()
        source_columns = [item.text() for item in selected_items]
        
        data = {
            'type': self.type_combo.currentText(),
            'target_column': self.target_edit.text().strip(),
            'source_columns': source_columns
        }
        
        if data['type'] == 'create_variable':
            data['formula'] = self.formula_edit.toPlainText().strip()
        elif data['type'] == 'binning':
            data['bins'] = self.bins_spin.value()
        
        return data
    
    def accept(self):
        """Validate and accept the dialog"""
        data = self.get_transform_data()
        
        if not data['target_column']:
            QMessageBox.warning(self, "Validation Error", "Please enter a target column name.")
            return
        
        if data['type'] == 'create_variable' and not data['formula']:
            QMessageBox.warning(self, "Validation Error", "Please enter a formula for variable creation.")
            return
        
        if data['type'] != 'create_variable' and len(data['source_columns']) < self.min_columns:
            QMessageBox.warning(self, "Validation Error", 
                              f"This transformation type requires at least {self.min_columns} source column(s).")
            return
        
        super().accept()
    
    def apply_styling(self):
        """Apply modern styling"""
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
            QLineEdit, QTextEdit, QComboBox, QSpinBox {
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 6px 8px;
                font-size: 13px;
            }
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus {
                border-color: #007bff;
            }
            QListWidget {
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
            }
            QListWidget::item {
                padding: 4px 8px;
                border-bottom: 1px solid #e9ecef;
            }
            QListWidget::item:selected {
                background-color: #007bff;
                color: white;
            }
        """)


class EnhancedTransformDialog(QDialog):
    """Enhanced Transform Dialog with modern UI and advanced functionality"""
    
    def __init__(self, transform_node, available_columns: List[str], parent=None):
        super().__init__(parent)
        self.transform_node = transform_node
        self.available_columns = available_columns
        self.transformation_widgets = []
        
        self.setWindowTitle("🔧 Configure Transformations")
        self.setModal(True)
        self.resize(900, 700)
        self.setup_ui()
        self.load_existing_transformations()
        self.apply_modern_styling()
    
    def setup_ui(self):
        """Setup the main UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        header_layout = QHBoxLayout()
        
        title_label = QLabel("Transform Configuration")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        self.add_transform_btn = QPushButton("➕ Add Transformation")
        self.add_transform_btn.clicked.connect(self.add_transformation)
        header_layout.addWidget(self.add_transform_btn)
        
        layout.addLayout(header_layout)
        
        splitter = QSplitter(Qt.Horizontal)
        
        transforms_group = QGroupBox("Transformations")
        transforms_layout = QVBoxLayout(transforms_group)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(400)
        
        self.transforms_widget = QWidget()
        self.transforms_layout = QVBoxLayout(self.transforms_widget)
        self.transforms_layout.setSpacing(8)
        self.transforms_layout.addStretch()
        
        self.scroll_area.setWidget(self.transforms_widget)
        transforms_layout.addWidget(self.scroll_area)
        
        transform_buttons_layout = QHBoxLayout()
        
        self.clear_all_btn = QPushButton("🧹 Clear All")
        self.clear_all_btn.clicked.connect(self.clear_all_transformations)
        transform_buttons_layout.addWidget(self.clear_all_btn)
        
        transform_buttons_layout.addStretch()
        
        self.validate_btn = QPushButton("✓ Validate")
        self.validate_btn.clicked.connect(self.validate_transformations)
        transform_buttons_layout.addWidget(self.validate_btn)
        
        transforms_layout.addLayout(transform_buttons_layout)
        splitter.addWidget(transforms_group)
        
        preview_group = QGroupBox("Transform Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        order_label = QLabel("Execution Order:")
        order_font = QFont()
        order_font.setBold(True)
        order_label.setFont(order_font)
        preview_layout.addWidget(order_label)
        
        self.order_list = QListWidget()
        self.order_list.setMaximumHeight(150)
        self.order_list.setStyleSheet("""
            QListWidget {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 4px 8px;
                border-bottom: 1px solid #e9ecef;
            }
        """)
        preview_layout.addWidget(self.order_list)
        
        stats_label = QLabel("Summary")
        stats_label.setFont(order_font)
        preview_layout.addWidget(stats_label)
        
        self.stats_text = QLabel("No transformations defined")
        self.stats_text.setStyleSheet("""
            QLabel {
                background-color: #e9ecef;
                border-radius: 4px;
                padding: 8px;
                color: #6c757d;
            }
        """)
        preview_layout.addWidget(self.stats_text)
        
        help_label = QLabel("Transformation Types")
        help_label.setFont(order_font)
        preview_layout.addWidget(help_label)
        
        help_text = QLabel("""
• create_variable: Custom formula-based variables
• derive_ratio: Ratio of two columns (A/B)
• derive_difference: Difference of two columns (A-B)
• derive_sum: Sum of multiple columns
• encode_categorical: Convert text to numbers
• log_transform: Natural logarithm transformation
• standardize: Z-score standardization
• binning: Group continuous values into bins
        """)
        help_text.setWordWrap(True)
        help_text.setStyleSheet("""
            QLabel {
                background-color: #e7f3ff;
                border-radius: 4px;
                padding: 8px;
                font-size: 11px;
                color: #004085;
            }
        """)
        preview_layout.addWidget(help_text)
        
        preview_layout.addStretch()
        splitter.addWidget(preview_group)
        
        splitter.setSizes([600, 300])
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
            QPushButton#add_transform_btn {
                background-color: #28a745;
            }
            QPushButton#add_transform_btn:hover {
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
        
        self.add_transform_btn.setObjectName("add_transform_btn")
        self.clear_all_btn.setObjectName("clear_all_btn")
        self.validate_btn.setObjectName("validate_btn")
    
    def add_transformation(self, transform_data: Dict = None):
        """Add a new transformation"""
        if not self.available_columns:
            QMessageBox.warning(self, "No Columns", "No columns available for transformation.")
            return
        
        dialog = TransformationEditDialog(self.available_columns, transform_data, self)
        if dialog.exec_() == QDialog.Accepted:
            new_transform_data = dialog.get_transform_data()
            
            self.transforms_layout.takeAt(self.transforms_layout.count() - 1)
            
            transform_widget = TransformationWidget(self.available_columns, new_transform_data, self)
            transform_widget.transformation_changed.connect(self.update_preview)
            transform_widget.delete_requested.connect(lambda: self.delete_transformation(transform_widget))
            transform_widget.edit_requested.connect(lambda: self.edit_transformation(transform_widget))
            
            self.transformation_widgets.append(transform_widget)
            self.transforms_layout.addWidget(transform_widget)
            
            self.transforms_layout.addStretch()
            
            self.update_preview()
            
            logger.info(f"Added transformation widget. Total transformations: {len(self.transformation_widgets)}")
    
    def edit_transformation(self, transform_widget: TransformationWidget):
        """Edit an existing transformation"""
        current_data = transform_widget.get_transform_data()
        
        dialog = TransformationEditDialog(self.available_columns, current_data, self)
        if dialog.exec_() == QDialog.Accepted:
            new_data = dialog.get_transform_data()
            transform_widget.update_transform_data(new_data)
            self.update_preview()
            
            logger.info(f"Edited transformation: {new_data['target_column']}")
    
    def delete_transformation(self, transform_widget: TransformationWidget):
        """Delete a transformation widget"""
        if transform_widget in self.transformation_widgets:
            self.transformation_widgets.remove(transform_widget)
            transform_widget.deleteLater()
            self.update_preview()
            
            logger.info(f"Deleted transformation widget. Remaining transformations: {len(self.transformation_widgets)}")
    
    def clear_all_transformations(self):
        """Clear all transformations"""
        reply = QMessageBox.question(
            self, "Clear All Transformations",
            "Are you sure you want to clear all transformations?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            for widget in self.transformation_widgets:
                widget.deleteLater()
            self.transformation_widgets.clear()
            self.update_preview()
            
            logger.info("Cleared all transformations")
    
    def validate_transformations(self):
        """Validate all transformations"""
        issues = []
        target_columns = set()
        
        for i, widget in enumerate(self.transformation_widgets):
            transform_data = widget.get_transform_data()
            
            target = transform_data['target_column']
            if target in target_columns:
                issues.append(f"Transformation {i+1}: Duplicate target column '{target}'")
            target_columns.add(target)
            
            if not target:
                issues.append(f"Transformation {i+1}: Missing target column name")
            
            transform_type = transform_data['type']
            if transform_type == 'create_variable' and not transform_data.get('formula'):
                issues.append(f"Transformation {i+1}: Missing formula for variable creation")
            elif transform_type != 'create_variable' and not transform_data.get('source_columns'):
                issues.append(f"Transformation {i+1}: Missing source columns")
        
        if issues:
            QMessageBox.warning(self, "Validation Issues", "\\n".join(issues))
        else:
            QMessageBox.information(self, "Validation", "All transformations are valid!")
    
    def update_preview(self):
        """Update the transformation preview"""
        self.order_list.clear()
        
        if not self.transformation_widgets:
            self.stats_text.setText("No transformations defined")
            return
        
        for i, widget in enumerate(self.transformation_widgets):
            transform_data = widget.get_transform_data()
            item_text = f"{i+1}. {transform_data['type']} → {transform_data['target_column']}"
            self.order_list.addItem(item_text)
        
        total_transforms = len(self.transformation_widgets)
        type_counts = {}
        for widget in self.transformation_widgets:
            transform_type = widget.get_transform_data()['type']
            type_counts[transform_type] = type_counts.get(transform_type, 0) + 1
        
        type_summary = ", ".join([f"{count} {ttype}" for ttype, count in type_counts.items()])
        self.stats_text.setText(f"Total: {total_transforms} transformations\\n{type_summary}")
    
    def load_existing_transformations(self):
        """Load existing transformations from the transform node"""
        if hasattr(self.transform_node, 'transformations') and self.transform_node.transformations:
            for transform_data in self.transform_node.transformations:
                self.add_transformation(transform_data)
            
            logger.info(f"Loaded {len(self.transform_node.transformations)} existing transformations")
    
    def get_transformations(self) -> List[Dict]:
        """Get all transformations from the dialog"""
        transformations = []
        if not hasattr(self, 'transformation_widgets') or not self.transformation_widgets:
            return transformations
        
        for widget in self.transformation_widgets:
            try:
                if widget is not None and hasattr(widget, 'get_transform_data'):
                    transform_data = widget.get_transform_data()
                    if transform_data:
                        transformations.append(transform_data)
            except Exception as e:
                logger.error(f"Error getting transform data from widget: {e}")
                continue
        return transformations
    
    def accept(self):
        """Accept the dialog and save transformations"""
        try:
            transformations = self.get_transformations()
            
            if not hasattr(self, 'transform_node') or self.transform_node is None:
                QMessageBox.warning(self, "Error", "Transform node is not available")
                return
            
            try:
                if hasattr(self.transform_node, 'clear_transformations'):
                    self.transform_node.clear_transformations()
            except Exception as e:
                logger.error(f"Error clearing transformations: {e}")
            
            successful_transforms = 0
            for transform_data in transformations:
                try:
                    if hasattr(self.transform_node, 'add_transformation') and transform_data:
                        if not transform_data.get('type') or not transform_data.get('target_column'):
                            logger.warning(f"Skipping invalid transformation: {transform_data}")
                            continue
                            
                        transformation_params = {
                            'transform_type': transform_data['type'],
                            'target_column': transform_data['target_column'],
                            'source_columns': transform_data.get('source_columns', []),
                            'formula': transform_data.get('formula', '')
                        }
                        
                        if transform_data['type'] == 'binning' and 'bins' in transform_data:
                            transformation_params['bins'] = transform_data['bins']
                        
                        self.transform_node.add_transformation(**transformation_params)
                        successful_transforms += 1
                except Exception as e:
                    logger.error(f"Error adding transformation {transform_data}: {e}")
                    continue
            
            logger.info(f"Saved {successful_transforms} transformations to node {self.transform_node.node_id}")
            super().accept()
            
        except Exception as e:
            logger.error(f"Error in transform dialog accept: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save transformations: {str(e)}")
            return