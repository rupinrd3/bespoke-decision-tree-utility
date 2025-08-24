#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Categorical Split Grouping Dialog for Bespoke Utility
Interface for grouping categorical feature values for optimal splits
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QTabWidget, QWidget, QListWidget, QListWidgetItem,
    QSplitter, QProgressBar, QMessageBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QFrame, QSizePolicy, QTreeWidget, QTreeWidgetItem
)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class CategoricalGroupingWorker(QThread):
    """Worker thread for calculating optimal categorical groupings"""
    
    progress_updated = pyqtSignal(int)
    grouping_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, data: pd.DataFrame, feature: str, target: str, method: str):
        super().__init__()
        self.data = data
        self.feature = feature
        self.target = target
        self.method = method
        
    def run(self):
        """Run the grouping calculation"""
        try:
            if self.method == "mutual_information":
                result = self._calculate_mutual_information_groups()
            elif self.method == "chi_square":
                result = self._calculate_chi_square_groups()
            elif self.method == "target_mean":
                result = self._calculate_target_mean_groups()
            elif self.method == "frequency_based":
                result = self._calculate_frequency_groups()
            else:
                result = self._calculate_simple_groups()
                
            self.grouping_completed.emit(result)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
            
    def _calculate_mutual_information_groups(self) -> Dict[str, Any]:
        """Calculate groups based on mutual information"""
        self.progress_updated.emit(20)
        
        feature_values = self.data[self.feature].unique()
        target_values = self.data[self.target].unique()
        
        mi_scores = {}
        for value in feature_values:
            subset = self.data[self.data[self.feature] == value]
            if len(subset) > 0:
                target_dist = subset[self.target].value_counts(normalize=True)
                mi_score = -sum(p * np.log2(p) for p in target_dist if p > 0)
                mi_scores[value] = mi_score
                
        self.progress_updated.emit(60)
        
        sorted_values = sorted(mi_scores.items(), key=lambda x: x[1])
        groups = self._create_groups_from_scores(sorted_values)
        
        self.progress_updated.emit(100)
        
        return {
            'method': 'mutual_information',
            'groups': groups,
            'scores': mi_scores,
            'feature': self.feature,
            'target': self.target
        }
        
    def _calculate_chi_square_groups(self) -> Dict[str, Any]:
        """Calculate groups based on chi-square statistics"""
        self.progress_updated.emit(20)
        
        from scipy.stats import chi2_contingency
        
        feature_values = self.data[self.feature].unique()
        chi_scores = {}
        
        for value in feature_values:
            value_data = self.data[self.feature] == value
            contingency = pd.crosstab(value_data, self.data[self.target])
            
            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                chi2, p_value, _, _ = chi2_contingency(contingency)
                chi_scores[value] = chi2
            else:
                chi_scores[value] = 0.0
                
        self.progress_updated.emit(60)
        
        sorted_values = sorted(chi_scores.items(), key=lambda x: x[1])
        groups = self._create_groups_from_scores(sorted_values)
        
        self.progress_updated.emit(100)
        
        return {
            'method': 'chi_square',
            'groups': groups,
            'scores': chi_scores,
            'feature': self.feature,
            'target': self.target
        }
        
    def _calculate_target_mean_groups(self) -> Dict[str, Any]:
        """Calculate groups based on target variable means"""
        self.progress_updated.emit(20)
        
        if pd.api.types.is_numeric_dtype(self.data[self.target]):
            target_means = self.data.groupby(self.feature)[self.target].mean()
        else:
            target_means = self.data.groupby(self.feature)[self.target].apply(
                lambda x: x.value_counts().index[0] if len(x) > 0 else None
            )
            
        self.progress_updated.emit(60)
        
        if pd.api.types.is_numeric_dtype(self.data[self.target]):
            sorted_values = sorted(target_means.items(), key=lambda x: x[1])
            groups = self._create_groups_from_scores(sorted_values)
        else:
            groups = {}
            for target_val in self.data[self.target].unique():
                group_name = f"Target_{target_val}"
                groups[group_name] = [
                    cat for cat, tgt in target_means.items() if tgt == target_val
                ]
                
        self.progress_updated.emit(100)
        
        return {
            'method': 'target_mean',
            'groups': groups,
            'scores': dict(target_means),
            'feature': self.feature,
            'target': self.target
        }
        
    def _calculate_frequency_groups(self) -> Dict[str, Any]:
        """Calculate groups based on frequency distribution"""
        self.progress_updated.emit(20)
        
        frequencies = self.data[self.feature].value_counts()
        
        self.progress_updated.emit(60)
        
        groups = {}
        high_freq = frequencies.quantile(0.75)
        med_freq = frequencies.quantile(0.25)
        
        groups['High_Frequency'] = [
            cat for cat, freq in frequencies.items() if freq >= high_freq
        ]
        groups['Medium_Frequency'] = [
            cat for cat, freq in frequencies.items() 
            if med_freq <= freq < high_freq
        ]
        groups['Low_Frequency'] = [
            cat for cat, freq in frequencies.items() if freq < med_freq
        ]
        
        self.progress_updated.emit(100)
        
        return {
            'method': 'frequency_based',
            'groups': groups,
            'scores': dict(frequencies),
            'feature': self.feature,
            'target': self.target
        }
        
    def _calculate_simple_groups(self) -> Dict[str, Any]:
        """Create simple alphabetical or numerical groups"""
        self.progress_updated.emit(50)
        
        feature_values = sorted(self.data[self.feature].unique())
        n_groups = min(5, len(feature_values))  # Max 5 groups
        group_size = len(feature_values) // n_groups
        
        groups = {}
        for i in range(n_groups):
            start_idx = i * group_size
            if i == n_groups - 1:  # Last group gets remaining values
                end_idx = len(feature_values)
            else:
                end_idx = (i + 1) * group_size
                
            group_name = f"Group_{i+1}"
            groups[group_name] = feature_values[start_idx:end_idx]
            
        self.progress_updated.emit(100)
        
        return {
            'method': 'simple',
            'groups': groups,
            'scores': {},
            'feature': self.feature,
            'target': self.target
        }
        
    def _create_groups_from_scores(self, sorted_values: List[Tuple], 
                                  n_groups: int = 3) -> Dict[str, List]:
        """Create groups from sorted value-score pairs"""
        groups = {}
        group_size = len(sorted_values) // n_groups
        
        for i in range(n_groups):
            start_idx = i * group_size
            if i == n_groups - 1:  # Last group gets remaining values
                end_idx = len(sorted_values)
            else:
                end_idx = (i + 1) * group_size
                
            group_name = f"Group_{i+1}"
            groups[group_name] = [item[0] for item in sorted_values[start_idx:end_idx]]
            
        return groups


class CategoricalSplitGroupingDialog(QDialog):
    """Dialog for grouping categorical feature values for optimal splits"""
    
    def __init__(self, dataframe: pd.DataFrame, feature_column: str, 
                 target_column: str = None, parent=None):
        super().__init__(parent)
        
        self.dataframe = dataframe
        self.feature_column = feature_column
        self.target_column = target_column
        self.current_groups = {}
        self.grouping_result = {}
        
        self.setWindowTitle(f"Categorical Split Grouping - {feature_column}")
        self.setModal(True)
        self.resize(1000, 700)
        
        self.setupUI()
        self.loadFeatureData()
        
    def setupUI(self):
        """Setup the main UI"""
        layout = QVBoxLayout()
        
        header_label = QLabel(f"Categorical Split Grouping: {self.feature_column}")
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header_label)
        
        unique_count = self.dataframe[self.feature_column].nunique()
        total_count = len(self.dataframe)
        info_text = f"Feature: {self.feature_column} | Unique values: {unique_count} | Total samples: {total_count}"
        if self.target_column:
            info_text += f" | Target: {self.target_column}"
        info_label = QLabel(info_text)
        layout.addWidget(info_label)
        
        main_splitter = QSplitter(Qt.Horizontal)
        
        left_panel = self.createGroupingPanel()
        main_splitter.addWidget(left_panel)
        
        right_panel = self.createGroupsPanel()
        main_splitter.addWidget(right_panel)
        
        main_splitter.setSizes([400, 600])
        
        layout.addWidget(main_splitter)
        
        button_layout = QHBoxLayout()
        
        self.apply_grouping_button = QPushButton("Apply Grouping")
        self.apply_grouping_button.clicked.connect(self.applyGrouping)
        self.apply_grouping_button.setEnabled(False)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.resetGroups)
        
        button_layout.addWidget(self.apply_grouping_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def createGroupingPanel(self) -> QWidget:
        """Create the grouping methods and controls panel"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        methods_group = QGroupBox("Grouping Methods")
        methods_layout = QVBoxLayout()
        
        self.grouping_method_combo = QComboBox()
        
        methods = [
            ("simple", "Simple Grouping"),
            ("frequency_based", "Frequency-Based"),
            ("target_mean", "Target Mean-Based"),
            ("mutual_information", "Mutual Information"),
            ("chi_square", "Chi-Square Test")
        ]
        
        for method_id, method_name in methods:
            self.grouping_method_combo.addItem(method_name, method_id)
            
        methods_layout.addWidget(QLabel("Method:"))
        methods_layout.addWidget(self.grouping_method_combo)
        
        self.num_groups_spin = QSpinBox()
        self.num_groups_spin.setRange(2, 10)
        self.num_groups_spin.setValue(3)
        methods_layout.addWidget(QLabel("Number of Groups:"))
        methods_layout.addWidget(self.num_groups_spin)
        
        self.generate_groups_button = QPushButton("Generate Groups")
        self.generate_groups_button.clicked.connect(self.generateGroups)
        methods_layout.addWidget(self.generate_groups_button)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        methods_layout.addWidget(self.progress_bar)
        
        methods_group.setLayout(methods_layout)
        layout.addWidget(methods_group)
        
        manual_group = QGroupBox("Manual Grouping")
        manual_layout = QVBoxLayout()
        
        manual_layout.addWidget(QLabel("Available Values:"))
        
        self.available_values_list = QListWidget()
        self.available_values_list.setMaximumHeight(150)
        self.available_values_list.setSelectionMode(QListWidget.MultiSelection)
        manual_layout.addWidget(self.available_values_list)
        
        manual_controls = QHBoxLayout()
        
        self.new_group_name_edit = QLineEdit()
        self.new_group_name_edit.setPlaceholderText("Group name...")
        manual_controls.addWidget(self.new_group_name_edit)
        
        self.create_group_button = QPushButton("Create Group")
        self.create_group_button.clicked.connect(self.createManualGroup)
        manual_controls.addWidget(self.create_group_button)
        
        manual_layout.addLayout(manual_controls)
        
        manual_group.setLayout(manual_layout)
        layout.addWidget(manual_group)
        
        stats_group = QGroupBox("Group Statistics")
        stats_layout = QFormLayout()
        
        self.total_groups_label = QLabel("0")
        self.largest_group_label = QLabel("N/A")
        self.smallest_group_label = QLabel("N/A")
        self.coverage_label = QLabel("0%")
        
        stats_layout.addRow("Total Groups:", self.total_groups_label)
        stats_layout.addRow("Largest Group:", self.largest_group_label)
        stats_layout.addRow("Smallest Group:", self.smallest_group_label)
        stats_layout.addRow("Coverage:", self.coverage_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def createGroupsPanel(self) -> QWidget:
        """Create the current groups and visualization panel"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        groups_group = QGroupBox("Current Groups")
        groups_layout = QVBoxLayout()
        
        self.groups_table = QTableWidget()
        self.groups_table.setColumnCount(4)
        self.groups_table.setHorizontalHeaderLabels([
            'Group Name', 'Values', 'Count', 'Actions'
        ])
        
        header = self.groups_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        
        self.groups_table.setAlternatingRowColors(True)
        groups_layout.addWidget(self.groups_table)
        
        groups_group.setLayout(groups_layout)
        layout.addWidget(groups_group)
        
        viz_group = QGroupBox("Group Visualization")
        viz_layout = QVBoxLayout()
        
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        widget.setLayout(layout)
        return widget
        
    def loadFeatureData(self):
        """Load the feature data into available values list"""
        unique_values = sorted(self.dataframe[self.feature_column].unique())
        
        self.available_values_list.clear()
        for value in unique_values:
            item = QListWidgetItem(str(value))
            item.setData(Qt.UserRole, value)
            self.available_values_list.addItem(item)
            
    def generateGroups(self):
        """Generate groups using the selected method"""
        if not self.target_column:
            QMessageBox.warning(self, "No Target Column", 
                              "Target column is required for most grouping methods.")
            return
            
        method = self.grouping_method_combo.currentData()
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.generate_groups_button.setEnabled(False)
        
        self.worker = CategoricalGroupingWorker(
            self.dataframe, self.feature_column, self.target_column, method
        )
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.grouping_completed.connect(self.onGroupingCompleted)
        self.worker.error_occurred.connect(self.onGroupingError)
        self.worker.start()
        
    def onGroupingCompleted(self, result: Dict[str, Any]):
        """Handle completion of automatic grouping"""
        self.progress_bar.setVisible(False)
        self.generate_groups_button.setEnabled(True)
        
        self.grouping_result = result
        self.current_groups = result['groups']
        
        self.updateGroupsTable()
        self.updateGroupStatistics()
        self.updateVisualization()
        
        self.apply_grouping_button.setEnabled(True)
        
    def onGroupingError(self, error_message: str):
        """Handle grouping error"""
        self.progress_bar.setVisible(False)
        self.generate_groups_button.setEnabled(True)
        
        QMessageBox.critical(self, "Grouping Error", 
                           f"Error occurred during grouping: {error_message}")
        
    def createManualGroup(self):
        """Create a manual group from selected values"""
        group_name = self.new_group_name_edit.text().strip()
        if not group_name:
            QMessageBox.warning(self, "Invalid Group Name", 
                              "Please enter a valid group name.")
            return
            
        selected_items = self.available_values_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Values Selected", 
                              "Please select values to group.")
            return
            
        if group_name in self.current_groups:
            QMessageBox.warning(self, "Group Exists", 
                              "A group with this name already exists.")
            return
            
        selected_values = [item.data(Qt.UserRole) for item in selected_items]
        self.current_groups[group_name] = selected_values
        
        for existing_group, values in self.current_groups.items():
            if existing_group != group_name:
                self.current_groups[existing_group] = [
                    v for v in values if v not in selected_values
                ]
                
        self.current_groups = {
            name: values for name, values in self.current_groups.items() 
            if values
        }
        
        self.updateGroupsTable()
        self.updateGroupStatistics()
        self.updateVisualization()
        
        self.available_values_list.clearSelection()
        self.new_group_name_edit.clear()
        
        self.apply_grouping_button.setEnabled(True)
        
    def updateGroupsTable(self):
        """Update the groups table display"""
        self.groups_table.setRowCount(len(self.current_groups))
        
        for i, (group_name, values) in enumerate(self.current_groups.items()):
            name_item = QTableWidgetItem(group_name)
            self.groups_table.setItem(i, 0, name_item)
            
            values_str = ', '.join(str(v) for v in values[:5])
            if len(values) > 5:
                values_str += f" ... (+{len(values) - 5} more)"
            values_item = QTableWidgetItem(values_str)
            values_item.setToolTip(', '.join(str(v) for v in values))
            self.groups_table.setItem(i, 1, values_item)
            
            count_item = QTableWidgetItem(str(len(values)))
            count_item.setTextAlignment(Qt.AlignRight)
            self.groups_table.setItem(i, 2, count_item)
            
            delete_button = QPushButton("Delete")
            delete_button.clicked.connect(lambda checked, gn=group_name: self.deleteGroup(gn))
            self.groups_table.setCellWidget(i, 3, delete_button)
            
    def deleteGroup(self, group_name: str):
        """Delete a group"""
        if group_name in self.current_groups:
            del self.current_groups[group_name]
            self.updateGroupsTable()
            self.updateGroupStatistics()
            self.updateVisualization()
            
    def updateGroupStatistics(self):
        """Update group statistics display"""
        if not self.current_groups:
            self.total_groups_label.setText("0")
            self.largest_group_label.setText("N/A")
            self.smallest_group_label.setText("N/A")
            self.coverage_label.setText("0%")
            return
            
        group_sizes = [len(values) for values in self.current_groups.values()]
        total_values = sum(group_sizes)
        unique_values = self.dataframe[self.feature_column].nunique()
        
        self.total_groups_label.setText(str(len(self.current_groups)))
        self.largest_group_label.setText(str(max(group_sizes)))
        self.smallest_group_label.setText(str(min(group_sizes)))
        self.coverage_label.setText(f"{(total_values / unique_values * 100):.1f}%")
        
    def updateVisualization(self):
        """Update the group visualization"""
        if not self.current_groups:
            return
            
        self.figure.clear()
        
        ax = self.figure.add_subplot(111)
        
        group_names = list(self.current_groups.keys())
        group_sizes = [len(values) for values in self.current_groups.values()]
        
        if self.target_column:
            target_values = self.dataframe[self.target_column].unique()
            target_colors = plt.cm.Set3(np.linspace(0, 1, len(target_values)))
            
            group_target_counts = {}
            for group_name, values in self.current_groups.items():
                group_data = self.dataframe[self.dataframe[self.feature_column].isin(values)]
                target_counts = group_data[self.target_column].value_counts()
                group_target_counts[group_name] = target_counts
                
            bottom = np.zeros(len(group_names))
            for i, target_val in enumerate(target_values):
                heights = [group_target_counts[group].get(target_val, 0) for group in group_names]
                ax.bar(group_names, heights, bottom=bottom, 
                      label=str(target_val), color=target_colors[i])
                bottom += heights
                
            ax.legend(title=self.target_column, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_ylabel('Sample Count')
            
        else:
            ax.bar(group_names, group_sizes, color='skyblue')
            ax.set_ylabel('Number of Categories')
            
        ax.set_xlabel('Groups')
        ax.set_title(f'Categorical Groups for {self.feature_column}')
        
        if len(max(group_names, key=len)) > 10:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
        self.figure.tight_layout()
        self.canvas.draw()
        
    def resetGroups(self):
        """Reset all groups"""
        reply = QMessageBox.question(self, "Reset Groups", 
                                   "Are you sure you want to reset all groups?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.current_groups.clear()
            self.grouping_result.clear()
            self.updateGroupsTable()
            self.updateGroupStatistics()
            self.updateVisualization()
            self.apply_grouping_button.setEnabled(False)
            
    def applyGrouping(self):
        """Apply the current grouping"""
        if not self.current_groups:
            QMessageBox.warning(self, "No Groups", 
                              "Please create groups before applying.")
            return
            
        all_values = set()
        for values in self.current_groups.values():
            all_values.update(values)
            
        unique_values = set(self.dataframe[self.feature_column].unique())
        unassigned = unique_values - all_values
        
        if unassigned:
            reply = QMessageBox.question(
                self, "Unassigned Values", 
                f"The following values are not assigned to any group: {list(unassigned)}\n"
                "Do you want to create an 'Other' group for them?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Yes:
                self.current_groups['Other'] = list(unassigned)
            elif reply == QMessageBox.Cancel:
                return
                
        QMessageBox.information(self, "Grouping Applied", 
                              f"Successfully created {len(self.current_groups)} groups "
                              f"for feature '{self.feature_column}'.")
        
    def getGroupingResult(self) -> Dict[str, Any]:
        """Get the final grouping result"""
        return {
            'feature': self.feature_column,
            'target': self.target_column,
            'groups': self.current_groups.copy(),
            'method': self.grouping_result.get('method', 'manual'),
            'statistics': {
                'total_groups': len(self.current_groups),
                'group_sizes': {name: len(values) for name, values in self.current_groups.items()},
                'coverage': len(set().union(*self.current_groups.values())) if self.current_groups else 0
            }
        }