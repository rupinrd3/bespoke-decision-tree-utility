#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Variable Selection Dialog for Bespoke Utility
Interface for selecting variables based on importance thresholds and criteria
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QIcon, QColor
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QTabWidget, QWidget, QListWidget, QListWidgetItem,
    QSplitter, QProgressBar, QMessageBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QFrame, QSizePolicy, QSlider, QButtonGroup, QRadioButton
)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from analytics.variable_importance import VariableImportance, ImportanceMethod

logger = logging.getLogger(__name__)


class VariableSelectionDialog(QDialog):
    """Dialog for selecting variables based on importance and other criteria"""
    
    def __init__(self, dataframe: pd.DataFrame, importance_scores: Dict[str, float] = None,
                 target_column: str = None, parent=None):
        super().__init__(parent)
        
        self.dataframe = dataframe
        self.importance_scores = importance_scores or {}
        self.target_column = target_column
        self.selected_variables = []
        
        if not self.importance_scores and target_column:
            self._calculate_basic_importance()
        
        self.setWindowTitle("Variable Selection")
        self.setModal(True)
        self.resize(1200, 800)
        
        self.setupUI()
        self.populateVariables()
        
    def setupUI(self):
        """Setup the main UI"""
        layout = QVBoxLayout()
        
        header_label = QLabel("Variable Selection")
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header_label)
        
        info_text = (f"Dataset: {len(self.dataframe)} rows, {len(self.dataframe.columns)} columns")
        if self.target_column:
            info_text += f" | Target: {self.target_column}"
        info_label = QLabel(info_text)
        layout.addWidget(info_label)
        
        splitter = QSplitter(Qt.Horizontal)
        
        left_panel = self.createSelectionPanel()
        splitter.addWidget(left_panel)
        
        right_panel = self.createVariablePanel()
        splitter.addWidget(right_panel)
        
        splitter.setSizes([480, 720])
        
        layout.addWidget(splitter)
        
        summary_group = self.createSummaryPanel()
        layout.addWidget(summary_group)
        
        button_layout = QHBoxLayout()
        
        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(self.selectAllVariables)
        
        self.clear_selection_button = QPushButton("Clear Selection")
        self.clear_selection_button.clicked.connect(self.clearSelection)
        
        self.auto_select_button = QPushButton("Auto Select")
        self.auto_select_button.clicked.connect(self.autoSelectVariables)
        
        button_layout.addWidget(self.select_all_button)
        button_layout.addWidget(self.clear_selection_button)
        button_layout.addWidget(self.auto_select_button)
        button_layout.addStretch()
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.acceptSelection)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def createSelectionPanel(self) -> QWidget:
        """Create the selection criteria panel"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        threshold_group = QGroupBox("Importance Threshold")
        threshold_layout = QFormLayout()
        
        self.threshold_method_combo = QComboBox()
        self.threshold_method_combo.addItems([
            'Minimum Score', 'Top N Variables', 'Top Percentage', 'Cumulative Importance'
        ])
        self.threshold_method_combo.currentTextChanged.connect(self.updateThresholdControls)
        threshold_layout.addRow("Method:", self.threshold_method_combo)
        
        self.threshold_value_spin = QDoubleSpinBox()
        self.threshold_value_spin.setRange(0.0, 1.0)
        self.threshold_value_spin.setValue(0.01)
        self.threshold_value_spin.setDecimals(4)
        self.threshold_value_spin.setSingleStep(0.001)
        self.threshold_value_spin.valueChanged.connect(self.applyThreshold)
        threshold_layout.addRow("Threshold:", self.threshold_value_spin)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 1000)
        self.threshold_slider.setValue(10)
        self.threshold_slider.valueChanged.connect(self.onSliderChanged)
        threshold_layout.addRow("", self.threshold_slider)
        
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)
        
        stats_group = QGroupBox("Statistical Filters")
        stats_layout = QFormLayout()
        
        self.correlation_enabled_checkbox = QCheckBox("Filter by correlation with target")
        stats_layout.addRow("", self.correlation_enabled_checkbox)
        
        self.correlation_threshold_spin = QDoubleSpinBox()
        self.correlation_threshold_spin.setRange(0.0, 1.0)
        self.correlation_threshold_spin.setValue(0.1)
        self.correlation_threshold_spin.setDecimals(3)
        self.correlation_threshold_spin.setEnabled(False)
        stats_layout.addRow("Min Correlation:", self.correlation_threshold_spin)
        
        self.variance_enabled_checkbox = QCheckBox("Filter by variance")
        stats_layout.addRow("", self.variance_enabled_checkbox)
        
        self.variance_threshold_spin = QDoubleSpinBox()
        self.variance_threshold_spin.setRange(0.0, 1000.0)
        self.variance_threshold_spin.setValue(0.0)
        self.variance_threshold_spin.setDecimals(4)
        self.variance_threshold_spin.setEnabled(False)
        stats_layout.addRow("Min Variance:", self.variance_threshold_spin)
        
        self.missing_enabled_checkbox = QCheckBox("Filter by missing values")
        stats_layout.addRow("", self.missing_enabled_checkbox)
        
        self.missing_threshold_spin = QDoubleSpinBox()
        self.missing_threshold_spin.setRange(0.0, 1.0)
        self.missing_threshold_spin.setValue(0.5)
        self.missing_threshold_spin.setDecimals(2)
        self.missing_threshold_spin.setEnabled(False)
        stats_layout.addRow("Max Missing %:", self.missing_threshold_spin)
        
        self.correlation_enabled_checkbox.toggled.connect(self.correlation_threshold_spin.setEnabled)
        self.variance_enabled_checkbox.toggled.connect(self.variance_threshold_spin.setEnabled)
        self.missing_enabled_checkbox.toggled.connect(self.missing_threshold_spin.setEnabled)
        
        self.correlation_enabled_checkbox.toggled.connect(self.applyFilters)
        self.variance_enabled_checkbox.toggled.connect(self.applyFilters)
        self.missing_enabled_checkbox.toggled.connect(self.applyFilters)
        self.correlation_threshold_spin.valueChanged.connect(self.applyFilters)
        self.variance_threshold_spin.valueChanged.connect(self.applyFilters)
        self.missing_threshold_spin.valueChanged.connect(self.applyFilters)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        dtype_group = QGroupBox("Data Type Filters")
        dtype_layout = QVBoxLayout()
        
        self.include_numeric_checkbox = QCheckBox("Include Numeric Variables")
        self.include_numeric_checkbox.setChecked(True)
        self.include_numeric_checkbox.toggled.connect(self.applyFilters)
        dtype_layout.addWidget(self.include_numeric_checkbox)
        
        self.include_categorical_checkbox = QCheckBox("Include Categorical Variables")
        self.include_categorical_checkbox.setChecked(True)
        self.include_categorical_checkbox.toggled.connect(self.applyFilters)
        dtype_layout.addWidget(self.include_categorical_checkbox)
        
        self.include_datetime_checkbox = QCheckBox("Include DateTime Variables")
        self.include_datetime_checkbox.setChecked(True)
        self.include_datetime_checkbox.toggled.connect(self.applyFilters)
        dtype_layout.addWidget(self.include_datetime_checkbox)
        
        dtype_group.setLayout(dtype_layout)
        layout.addWidget(dtype_group)
        
        exclusion_group = QGroupBox("Manual Exclusions")
        exclusion_layout = QVBoxLayout()
        
        self.exclusion_list = QListWidget()
        self.exclusion_list.setMaximumHeight(150)
        exclusion_layout.addWidget(QLabel("Variables to exclude:"))
        exclusion_layout.addWidget(self.exclusion_list)
        
        exclusion_buttons = QHBoxLayout()
        self.add_exclusion_button = QPushButton("Add")
        self.remove_exclusion_button = QPushButton("Remove")
        self.add_exclusion_button.clicked.connect(self.addExclusion)
        self.remove_exclusion_button.clicked.connect(self.removeExclusion)
        
        exclusion_buttons.addWidget(self.add_exclusion_button)
        exclusion_buttons.addWidget(self.remove_exclusion_button)
        exclusion_layout.addLayout(exclusion_buttons)
        
        exclusion_group.setLayout(exclusion_layout)
        layout.addWidget(exclusion_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def createVariablePanel(self) -> QWidget:
        """Create the variable list and visualization panel"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        table_group = QGroupBox("Variables")
        table_layout = QVBoxLayout()
        
        controls_layout = QHBoxLayout()
        
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search variables...")
        self.search_edit.textChanged.connect(self.filterVariableTable)
        controls_layout.addWidget(QLabel("Search:"))
        controls_layout.addWidget(self.search_edit)
        
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(['Importance (Desc)', 'Importance (Asc)', 'Name (A-Z)', 'Name (Z-A)', 'Correlation'])
        self.sort_combo.currentTextChanged.connect(self.sortVariableTable)
        controls_layout.addWidget(QLabel("Sort by:"))
        controls_layout.addWidget(self.sort_combo)
        
        table_layout.addLayout(controls_layout)
        
        self.variable_table = QTableWidget()
        self.variable_table.setColumnCount(7)
        self.variable_table.setHorizontalHeaderLabels([
            'Select', 'Variable', 'Importance', 'Type', 'Correlation', 'Missing %', 'Variance'
        ])
        
        header = self.variable_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        
        self.variable_table.setAlternatingRowColors(True)
        self.variable_table.setSortingEnabled(False)  # We'll handle sorting manually
        
        table_layout.addWidget(self.variable_table)
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)
        
        viz_group = QGroupBox("Importance Visualization")
        viz_layout = QVBoxLayout()
        
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        widget.setLayout(layout)
        return widget
        
    def createSummaryPanel(self) -> QWidget:
        """Create the selection summary panel"""
        summary_group = QGroupBox("Selection Summary")
        layout = QHBoxLayout()
        
        self.total_variables_label = QLabel("Total: 0")
        self.selected_variables_label = QLabel("Selected: 0")
        self.excluded_variables_label = QLabel("Excluded: 0")
        self.selection_percentage_label = QLabel("Percentage: 0%")
        
        layout.addWidget(self.total_variables_label)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.selected_variables_label)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.excluded_variables_label)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.selection_percentage_label)
        layout.addStretch()
        
        summary_group.setLayout(layout)
        return summary_group
        
    def populateVariables(self):
        """Populate the variable table"""
        columns = [col for col in self.dataframe.columns if col != self.target_column]
        
        self.variable_table.setRowCount(len(columns))
        
        for i, col in enumerate(columns):
            checkbox = QCheckBox()
            checkbox.setChecked(True)  # Start with all selected
            checkbox.stateChanged.connect(self.updateSelection)
            self.variable_table.setCellWidget(i, 0, checkbox)
            
            name_item = QTableWidgetItem(col)
            self.variable_table.setItem(i, 1, name_item)
            
            importance = self.importance_scores.get(col, 0.0)
            importance_item = QTableWidgetItem(f"{importance:.4f}")
            importance_item.setTextAlignment(Qt.AlignRight)
            self.variable_table.setItem(i, 2, importance_item)
            
            dtype = str(self.dataframe[col].dtype)
            if pd.api.types.is_numeric_dtype(self.dataframe[col]):
                dtype_display = "Numeric"
            elif pd.api.types.is_datetime64_any_dtype(self.dataframe[col]):
                dtype_display = "DateTime"
            else:
                dtype_display = "Categorical"
            type_item = QTableWidgetItem(dtype_display)
            self.variable_table.setItem(i, 3, type_item)
            
            correlation = 0.0
            if self.target_column and pd.api.types.is_numeric_dtype(self.dataframe[col]):
                try:
                    correlation = abs(self.dataframe[col].corr(self.dataframe[self.target_column]))
                    if pd.isna(correlation):
                        correlation = 0.0
                except (ValueError, TypeError, KeyError) as e:
                    logger.debug(f"Error calculating correlation for {col}: {e}")
                    correlation = 0.0
            correlation_item = QTableWidgetItem(f"{correlation:.3f}")
            correlation_item.setTextAlignment(Qt.AlignRight)
            self.variable_table.setItem(i, 4, correlation_item)
            
            missing_pct = (self.dataframe[col].isna().sum() / len(self.dataframe)) * 100
            missing_item = QTableWidgetItem(f"{missing_pct:.1f}%")
            missing_item.setTextAlignment(Qt.AlignRight)
            self.variable_table.setItem(i, 5, missing_item)
            
            variance = 0.0
            if pd.api.types.is_numeric_dtype(self.dataframe[col]):
                try:
                    variance = self.dataframe[col].var()
                    if pd.isna(variance):
                        variance = 0.0
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error calculating variance for {col}: {e}")
                    variance = 0.0
            variance_item = QTableWidgetItem(f"{variance:.4f}")
            variance_item.setTextAlignment(Qt.AlignRight)
            self.variable_table.setItem(i, 6, variance_item)
            
        self.updateVisualization()
        self.updateSummary()
        
    def updateThresholdControls(self):
        """Update threshold controls based on method"""
        method = self.threshold_method_combo.currentText()
        
        if method == 'Minimum Score':
            self.threshold_value_spin.setRange(0.0, 1.0)
            self.threshold_value_spin.setValue(0.01)
            self.threshold_value_spin.setDecimals(4)
            self.threshold_value_spin.setSingleStep(0.001)
            self.threshold_value_spin.setSuffix("")
        elif method == 'Top N Variables':
            self.threshold_value_spin.setRange(1, len(self.dataframe.columns))
            self.threshold_value_spin.setValue(10)
            self.threshold_value_spin.setDecimals(0)
            self.threshold_value_spin.setSingleStep(1)
            self.threshold_value_spin.setSuffix(" vars")
        elif method == 'Top Percentage':
            self.threshold_value_spin.setRange(1, 100)
            self.threshold_value_spin.setValue(20)
            self.threshold_value_spin.setDecimals(0)
            self.threshold_value_spin.setSingleStep(1)
            self.threshold_value_spin.setSuffix("%")
        elif method == 'Cumulative Importance':
            self.threshold_value_spin.setRange(0.1, 1.0)
            self.threshold_value_spin.setValue(0.9)
            self.threshold_value_spin.setDecimals(2)
            self.threshold_value_spin.setSingleStep(0.05)
            self.threshold_value_spin.setSuffix("")
            
        self.applyThreshold()
        
    def onSliderChanged(self, value):
        """Handle slider value change"""
        method = self.threshold_method_combo.currentText()
        
        if method == 'Minimum Score':
            threshold = value / 1000.0
        elif method == 'Top N Variables':
            threshold = int((value / 1000.0) * len(self.dataframe.columns))
        elif method == 'Top Percentage':
            threshold = int((value / 1000.0) * 100)
        elif method == 'Cumulative Importance':
            threshold = 0.1 + (value / 1000.0) * 0.9
        else:
            threshold = value / 1000.0
            
        self.threshold_value_spin.setValue(threshold)
        
    def applyThreshold(self):
        """Apply importance threshold to variable selection"""
        method = self.threshold_method_combo.currentText()
        threshold = self.threshold_value_spin.value()
        
        scored_variables = [(col, score) for col, score in self.importance_scores.items() 
                           if col != self.target_column]
        scored_variables.sort(key=lambda x: x[1], reverse=True)
        
        selected_vars = set()
        
        if method == 'Minimum Score':
            selected_vars = {col for col, score in scored_variables if score >= threshold}
        elif method == 'Top N Variables':
            selected_vars = {col for col, score in scored_variables[:int(threshold)]}
        elif method == 'Top Percentage':
            n_vars = int(len(scored_variables) * threshold / 100)
            selected_vars = {col for col, score in scored_variables[:n_vars]}
        elif method == 'Cumulative Importance':
            cumsum = 0.0
            total_importance = sum(score for _, score in scored_variables)
            for col, score in scored_variables:
                cumsum += score
                selected_vars.add(col)
                if cumsum / total_importance >= threshold:
                    break
                    
        for i in range(self.variable_table.rowCount()):
            name_item = self.variable_table.item(i, 1)
            if name_item:
                var_name = name_item.text()
                checkbox = self.variable_table.cellWidget(i, 0)
                if checkbox:
                    checkbox.setChecked(var_name in selected_vars)
                    
        self.updateSelection()
        
    def applyFilters(self):
        """Apply statistical and data type filters"""
        for i in range(self.variable_table.rowCount()):
            name_item = self.variable_table.item(i, 1)
            if not name_item:
                continue
                
            var_name = name_item.text()
            checkbox = self.variable_table.cellWidget(i, 0)
            if not checkbox:
                continue
                
            should_exclude = False
            
            type_item = self.variable_table.item(i, 3)
            if type_item:
                var_type = type_item.text()
                if var_type == "Numeric" and not self.include_numeric_checkbox.isChecked():
                    should_exclude = True
                elif var_type == "Categorical" and not self.include_categorical_checkbox.isChecked():
                    should_exclude = True
                elif var_type == "DateTime" and not self.include_datetime_checkbox.isChecked():
                    should_exclude = True
                    
            if (self.correlation_enabled_checkbox.isChecked() and 
                self.target_column and not should_exclude):
                correlation_item = self.variable_table.item(i, 4)
                if correlation_item:
                    try:
                        correlation = float(correlation_item.text())
                        if correlation < self.correlation_threshold_spin.value():
                            should_exclude = True
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Error parsing correlation value: {e}")
                        pass
                        
            if self.variance_enabled_checkbox.isChecked() and not should_exclude:
                variance_item = self.variable_table.item(i, 6)
                if variance_item:
                    try:
                        variance = float(variance_item.text())
                        if variance < self.variance_threshold_spin.value():
                            should_exclude = True
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Error parsing variance value: {e}")
                        pass
                        
            if self.missing_enabled_checkbox.isChecked() and not should_exclude:
                missing_item = self.variable_table.item(i, 5)
                if missing_item:
                    try:
                        missing_pct = float(missing_item.text().rstrip('%'))
                        if missing_pct > self.missing_threshold_spin.value() * 100:
                            should_exclude = True
                    except (ValueError, TypeError, AttributeError) as e:
                        logger.debug(f"Error parsing missing percentage: {e}")
                        pass
                        
            if should_exclude:
                checkbox.setChecked(False)
                
        self.updateSelection()
        
    def filterVariableTable(self):
        """Filter variable table based on search text"""
        search_text = self.search_edit.text().lower()
        
        for i in range(self.variable_table.rowCount()):
            name_item = self.variable_table.item(i, 1)
            if name_item:
                var_name = name_item.text().lower()
                should_show = search_text in var_name
                self.variable_table.setRowHidden(i, not should_show)
                
    def sortVariableTable(self):
        """Sort variable table based on selected criteria"""
        sort_method = self.sort_combo.currentText()
        
        rows_data = []
        for i in range(self.variable_table.rowCount()):
            row_data = []
            checkbox = self.variable_table.cellWidget(i, 0)
            row_data.append(checkbox.isChecked() if checkbox else False)
            
            for j in range(1, self.variable_table.columnCount()):
                item = self.variable_table.item(i, j)
                row_data.append(item.text() if item else "")
                
            rows_data.append((i, row_data))
            
        if sort_method == 'Importance (Desc)':
            rows_data.sort(key=lambda x: float(x[1][2]) if x[1][2] else 0, reverse=True)
        elif sort_method == 'Importance (Asc)':
            rows_data.sort(key=lambda x: float(x[1][2]) if x[1][2] else 0)
        elif sort_method == 'Name (A-Z)':
            rows_data.sort(key=lambda x: x[1][1])
        elif sort_method == 'Name (Z-A)':
            rows_data.sort(key=lambda x: x[1][1], reverse=True)
        elif sort_method == 'Correlation':
            rows_data.sort(key=lambda x: float(x[1][4]) if x[1][4] else 0, reverse=True)
            
        for new_i, (old_i, row_data) in enumerate(rows_data):
            checkbox = QCheckBox()
            checkbox.setChecked(row_data[0])
            checkbox.stateChanged.connect(self.updateSelection)
            self.variable_table.setCellWidget(new_i, 0, checkbox)
            
            for j in range(1, len(row_data)):
                item = QTableWidgetItem(row_data[j])
                if j in [2, 4, 5, 6]:  # Numeric columns
                    item.setTextAlignment(Qt.AlignRight)
                self.variable_table.setItem(new_i, j, item)
                
    def updateSelection(self):
        """Update selection based on checkboxes"""
        self.selected_variables = []
        
        for i in range(self.variable_table.rowCount()):
            checkbox = self.variable_table.cellWidget(i, 0)
            name_item = self.variable_table.item(i, 1)
            
            if checkbox and name_item and checkbox.isChecked():
                self.selected_variables.append(name_item.text())
                
        self.updateSummary()
        self.updateVisualization()
        
    def updateSummary(self):
        """Update the selection summary"""
        total = self.variable_table.rowCount()
        selected = len(self.selected_variables)
        excluded = total - selected
        percentage = (selected / total * 100) if total > 0 else 0
        
        self.total_variables_label.setText(f"Total: {total}")
        self.selected_variables_label.setText(f"Selected: {selected}")
        self.excluded_variables_label.setText(f"Excluded: {excluded}")
        self.selection_percentage_label.setText(f"Percentage: {percentage:.1f}%")
        
    def updateVisualization(self):
        """Update the importance visualization"""
        if not self.importance_scores:
            return
            
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        selected_importance = {var: self.importance_scores.get(var, 0.0) 
                             for var in self.selected_variables}
        
        if not selected_importance:
            ax.text(0.5, 0.5, 'No variables selected', 
                   ha='center', va='center', transform=ax.transAxes)
        else:
            sorted_vars = sorted(selected_importance.items(), key=lambda x: x[1], reverse=True)
            
            if len(sorted_vars) > 15:
                sorted_vars = sorted_vars[:15]
                
            variables = [var for var, _ in sorted_vars]
            importance = [imp for _, imp in sorted_vars]
            
            y_pos = np.arange(len(variables))
            bars = ax.barh(y_pos, importance, color='skyblue')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(variables)
            ax.set_xlabel('Importance Score')
            ax.set_title(f'Selected Variables Importance (Top {len(sorted_vars)})')
            
            for i, (bar, imp) in enumerate(zip(bars, importance)):
                width = bar.get_width()
                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{imp:.3f}', ha='left', va='center', fontsize=8)
                       
        ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()
        
    def addExclusion(self):
        """Add variable to exclusion list"""
        pass
        
    def removeExclusion(self):
        """Remove variable from exclusion list"""
        current_item = self.exclusion_list.currentItem()
        if current_item:
            self.exclusion_list.takeItem(self.exclusion_list.row(current_item))
            
    def selectAllVariables(self):
        """Select all variables"""
        for i in range(self.variable_table.rowCount()):
            checkbox = self.variable_table.cellWidget(i, 0)
            if checkbox:
                checkbox.setChecked(True)
                
    def clearSelection(self):
        """Clear all selections"""
        for i in range(self.variable_table.rowCount()):
            checkbox = self.variable_table.cellWidget(i, 0)
            if checkbox:
                checkbox.setChecked(False)
                
    def autoSelectVariables(self):
        """Auto-select variables based on current criteria"""
        self.applyThreshold()
        self.applyFilters()
        
    def acceptSelection(self):
        """Accept the current selection"""
        if not self.selected_variables:
            reply = QMessageBox.question(self, "No Variables Selected", 
                                       "No variables are selected. Do you want to proceed?",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
                
        self.accept()
        
    def getSelectedVariables(self) -> List[str]:
        """Get the list of selected variables"""
        return self.selected_variables.copy()
        
    def _calculate_basic_importance(self):
        """Calculate basic importance scores if not provided"""
        if not self.target_column:
            for col in self.dataframe.columns:
                if pd.api.types.is_numeric_dtype(self.dataframe[col]):
                    try:
                        variance = self.dataframe[col].var()
                        self.importance_scores[col] = variance if not pd.isna(variance) else 0.0
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Error calculating variance for {col}: {e}")
                        self.importance_scores[col] = 0.0
                else:
                    self.importance_scores[col] = 0.0
        else:
            for col in self.dataframe.columns:
                if col == self.target_column:
                    continue
                    
                if pd.api.types.is_numeric_dtype(self.dataframe[col]):
                    try:
                        correlation = abs(self.dataframe[col].corr(self.dataframe[self.target_column]))
                        self.importance_scores[col] = correlation if not pd.isna(correlation) else 0.0
                    except (ValueError, TypeError, KeyError) as e:
                        logger.debug(f"Error calculating correlation for {col}: {e}")
                        self.importance_scores[col] = 0.0
                else:
                    self.importance_scores[col] = 0.0
                    
        max_score = max(self.importance_scores.values()) if self.importance_scores else 1.0
        if max_score > 0:
            self.importance_scores = {k: v/max_score for k, v in self.importance_scores.items()}