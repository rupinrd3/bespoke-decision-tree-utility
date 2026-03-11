#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Node Editor Module for Bespoke Utility
Provides an interface for editing decision tree nodes

"""

import logging
import math
from typing import Dict, List, Tuple, Optional, Any, Set, Union

from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal, pyqtSlot, QObject
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPixmap, QIcon
from PyQt5.QtWidgets import (QDialog, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                           QLabel, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
                           QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
                           QCheckBox, QRadioButton, QButtonGroup, QGroupBox, QTabWidget,
                           QMessageBox, QTextEdit, QSplitter, QScrollArea, QFrame,
                           QListWidget, QListWidgetItem, QToolButton, QMenu, QAction,
                           QApplication)

from models.node import TreeNode
from models.decision_tree import BespokeDecisionTree, SplitCriterion
from analytics.node_statistics import NodeAnalyzer
from utils.metrics_calculator import CentralMetricsCalculator

logger = logging.getLogger(__name__)

class NodePropertiesWidget(QWidget):
    """Widget for displaying and editing node properties"""
    
    propertiesChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        """
        Initialize the node properties widget
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.node = None
        self.model = None
        self.read_only = False
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        form_layout = QFormLayout()
        
        self.id_label = QLabel("None")
        form_layout.addRow("Node ID:", self.id_label)
        
        self.depth_label = QLabel("0")
        form_layout.addRow("Depth:", self.depth_label)
        
        self.type_label = QLabel("Terminal")
        form_layout.addRow("Type:", self.type_label)
        
        self.samples_label = QLabel("0")
        form_layout.addRow("Samples:", self.samples_label)
        
        self.impurity_label = QLabel("0.0")
        form_layout.addRow("Impurity:", self.impurity_label)
        
        self.class_label = QLabel("None")
        form_layout.addRow("Class Distribution:", self.class_label)
        
        self.prediction_label = QLabel("None")
        form_layout.addRow("Prediction:", self.prediction_label)
        
        self.probability_label = QLabel("0.0")
        form_layout.addRow("Probability:", self.probability_label)
        
        layout.addLayout(form_layout)
        
        split_group = QGroupBox("Split Information")
        self.split_layout = QFormLayout()
        split_group.setLayout(self.split_layout)
        
        self.feature_combo = QComboBox()
        self.split_layout.addRow("Split Feature:", self.feature_combo)
        
        self.split_type_combo = QComboBox()
        self.split_type_combo.addItems(["Numeric", "Categorical"])
        self.split_layout.addRow("Split Type:", self.split_type_combo)
        
        self.split_value_spin = QDoubleSpinBox()
        self.split_value_spin.setDecimals(6)
        self.split_value_spin.setRange(-1e6, 1e6)
        self.split_layout.addRow("Split Value:", self.split_value_spin)
        
        self.categories_label = QLabel("Categories:")
        self.split_layout.addRow(self.categories_label)
        
        self.categories_table = QTableWidget(0, 2)
        self.categories_table.setHorizontalHeaderLabels(["Category", "Child"])
        self.categories_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.categories_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.split_layout.addRow(self.categories_table)
        
        buttons_layout = QHBoxLayout()
        self.add_category_btn = QPushButton("Add Category")
        self.remove_category_btn = QPushButton("Remove Category")
        buttons_layout.addWidget(self.add_category_btn)
        buttons_layout.addWidget(self.remove_category_btn)
        self.split_layout.addRow(buttons_layout)
        
        self.terminal_check = QCheckBox("Make this node terminal (leaf)")
        self.split_layout.addRow(self.terminal_check)
        
        layout.addWidget(split_group)
        
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QFormLayout()
        metrics_group.setLayout(metrics_layout)
        
        self.accuracy_label = QLabel("N/A")
        metrics_layout.addRow("Accuracy:", self.accuracy_label)
        
        self.precision_label = QLabel("N/A")
        metrics_layout.addRow("Precision:", self.precision_label)
        
        self.recall_label = QLabel("N/A")
        metrics_layout.addRow("Recall:", self.recall_label)
        
        self.f1_label = QLabel("N/A")
        metrics_layout.addRow("F1 Score:", self.f1_label)
        
        layout.addWidget(metrics_group)
        
        buttons_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Changes")
        self.reset_btn = QPushButton("Reset")
        buttons_layout.addWidget(self.apply_btn)
        buttons_layout.addWidget(self.reset_btn)
        layout.addLayout(buttons_layout)
        
        self.connect_signals()
        
        self.set_split_controls_enabled(False)
    
    def connect_signals(self):
        """Connect widget signals to slots"""
        self.split_type_combo.currentIndexChanged.connect(self.on_split_type_changed)
        
        self.terminal_check.toggled.connect(self.on_terminal_toggled)
        
        self.apply_btn.clicked.connect(self.apply_changes)
        self.reset_btn.clicked.connect(self.reset)
        
        self.add_category_btn.clicked.connect(self.add_category)
        self.remove_category_btn.clicked.connect(self.remove_category)
    
    def set_node(self, node: TreeNode, feature_names: List[str] = None, read_only: bool = False, model = None):
        """
        Set the node to display/edit
        
        Args:
            node: TreeNode to edit
            feature_names: List of available feature names
            read_only: Whether to disable editing
            model: BespokeDecisionTree model for applying splits
        """
        self.node = node
        self.model = model
        self.read_only = read_only
        
        if feature_names:
            self.feature_combo.clear()
            self.feature_combo.addItems(feature_names)
        
        self.update_ui()
        
        self.set_controls_enabled(not read_only)
    
    def update_ui(self):
        """Update UI components with node properties"""
        if not self.node:
            return
        
        self.id_label.setText(self.node.node_id)
        self.depth_label.setText(str(self.node.depth))
        self.type_label.setText("Terminal" if self.node.is_terminal else "Internal")
        self.samples_label.setText(str(self.node.samples))
        
        if self.node.impurity is not None:
            self.impurity_label.setText(f"{self.node.impurity:.6f}")
        else:
            self.impurity_label.setText("N/A")
        
        if self.node.class_counts:
            dist_text = ", ".join([f"{cls}: {count}" for cls, count in self.node.class_counts.items()])
            self.class_label.setText(dist_text)
        else:
            self.class_label.setText("None")
        
        if self.node.majority_class is not None:
            self.prediction_label.setText(str(self.node.majority_class))
        else:
            self.prediction_label.setText("None")
            
        if self.node.probability is not None:
            self.probability_label.setText(f"{self.node.probability:.4f}")
        else:
            self.probability_label.setText("0.0")
        
        if not self.node.is_terminal and self.node.split_feature:
            self.terminal_check.setChecked(False)
            
            self.set_split_controls_enabled(True)
            
            index = self.feature_combo.findText(self.node.split_feature)
            if index >= 0:
                self.feature_combo.setCurrentIndex(index)
            
            split_type_index = 0 if self.node.split_type == 'numeric' else 1
            self.split_type_combo.setCurrentIndex(split_type_index)
            
            if self.node.split_type == 'numeric':
                if self.node.split_value is not None:
                    self.split_value_spin.setValue(self.node.split_value)
                
                self.categories_label.hide()
                self.categories_table.hide()
                self.add_category_btn.hide()
                self.remove_category_btn.hide()
                
                self.split_value_spin.show()
            else:
                self.categories_label.show()
                self.categories_table.show()
                self.add_category_btn.show()
                self.remove_category_btn.show()
                
                self.split_value_spin.hide()
                
                self.update_categories_table()
        else:
            self.terminal_check.setChecked(True)
            self.set_split_controls_enabled(False)
        
        if self.node.accuracy is not None:
            self.accuracy_label.setText(f"{self.node.accuracy:.4f}")
        else:
            self.accuracy_label.setText("N/A")
            
        if self.node.precision is not None:
            self.precision_label.setText(f"{self.node.precision:.4f}")
        else:
            self.precision_label.setText("N/A")
            
        if self.node.recall is not None:
            self.recall_label.setText(f"{self.node.recall:.4f}")
        else:
            self.recall_label.setText("N/A")
            
        if self.node.f1_score is not None:
            self.f1_label.setText(f"{self.node.f1_score:.4f}")
        else:
            self.f1_label.setText("N/A")
    
    def update_categories_table(self):
        """Update the categories table for categorical splits"""
        if not self.node or not self.node.split_categories:
            self.categories_table.setRowCount(0)
            return
        
        categories = list(self.node.split_categories.items())
        categories.sort(key=lambda x: x[0])
        
        self.categories_table.setRowCount(len(categories))
        
        for i, (category, child_idx) in enumerate(categories):
            cat_item = QTableWidgetItem(str(category))
            self.categories_table.setItem(i, 0, cat_item)
            
            child_item = QTableWidgetItem(str(child_idx))
            self.categories_table.setItem(i, 1, child_item)
    
    def set_controls_enabled(self, enabled: bool):
        """
        Enable or disable all editable controls
        
        Args:
            enabled: Whether to enable controls
        """
        self.feature_combo.setEnabled(enabled)
        self.split_type_combo.setEnabled(enabled)
        self.split_value_spin.setEnabled(enabled)
        self.categories_table.setEnabled(enabled)
        self.add_category_btn.setEnabled(enabled)
        self.remove_category_btn.setEnabled(enabled)
        self.terminal_check.setEnabled(enabled)
        
        self.apply_btn.setEnabled(enabled)
        self.reset_btn.setEnabled(enabled)
    
    def set_split_controls_enabled(self, enabled: bool):
        """
        Enable or disable split-related controls
        
        Args:
            enabled: Whether to enable controls
        """
        self.feature_combo.setEnabled(enabled and not self.read_only)
        self.split_type_combo.setEnabled(enabled and not self.read_only)
        self.split_value_spin.setEnabled(enabled and not self.read_only)
        self.categories_table.setEnabled(enabled and not self.read_only)
        self.add_category_btn.setEnabled(enabled and not self.read_only)
        self.remove_category_btn.setEnabled(enabled and not self.read_only)
        
        if enabled:
            split_type = self.split_type_combo.currentText().lower()
            
            if split_type == "numeric":
                self.categories_label.hide()
                self.categories_table.hide()
                self.add_category_btn.hide()
                self.remove_category_btn.hide()
                
                self.split_value_spin.show()
            else:
                self.categories_label.show()
                self.categories_table.show()
                self.add_category_btn.show()
                self.remove_category_btn.show()
                
                self.split_value_spin.hide()
        else:
            self.split_value_spin.hide()
            self.categories_label.hide()
            self.categories_table.hide()
            self.add_category_btn.hide()
            self.remove_category_btn.hide()
    
    def on_split_type_changed(self, index: int):
        """
        Handle split type change
        
        Args:
            index: Combo box index
        """
        split_type = self.split_type_combo.currentText().lower()
        
        if split_type == "numeric":
            self.categories_label.hide()
            self.categories_table.hide()
            self.add_category_btn.hide()
            self.remove_category_btn.hide()
            
            self.split_value_spin.show()
        else:
            self.categories_label.show()
            self.categories_table.show()
            self.add_category_btn.show()
            self.remove_category_btn.show()
            
            self.split_value_spin.hide()
            
            if self.node and not self.node.split_categories:
                self.node.split_categories = {}
                self.update_categories_table()
    
    def on_terminal_toggled(self, checked: bool):
        """
        Handle terminal node checkbox toggle
        
        Args:
            checked: Whether the checkbox is checked
        """
        self.set_split_controls_enabled(not checked)
    
    def add_category(self):
        """Add a new category to the table"""
        row_count = self.categories_table.rowCount()
        
        self.categories_table.setRowCount(row_count + 1)
        
        cat_item = QTableWidgetItem("")
        child_item = QTableWidgetItem("0")
        
        self.categories_table.setItem(row_count, 0, cat_item)
        self.categories_table.setItem(row_count, 1, child_item)
    
    def remove_category(self):
        """Remove the selected category from the table"""
        selected_rows = set()
        for item in self.categories_table.selectedItems():
            selected_rows.add(item.row())
        
        for row in sorted(selected_rows, reverse=True):
            self.categories_table.removeRow(row)
    
    def apply_changes(self):
        """Apply changes to the node"""
        if not self.node or self.read_only:
            return
        
        try:
            terminal_checked = self.terminal_check.isChecked()
            
            if terminal_checked != self.node.is_terminal:
                self.node.is_terminal = terminal_checked
                
                if terminal_checked:
                    self.node.split_feature = None
                    self.node.split_value = None
                    self.node.split_type = None
                    self.node.split_categories = {}
                    self.node.split_rule = None
                    self.node.children = []
                    
                    logger.info(f"Node {self.node.node_id} converted to terminal node")
                    
                else:
                    success = self._apply_split_from_ui()
                    if not success:
                        self.terminal_check.setChecked(True)
                        return
                        
            elif not terminal_checked and not self.node.children:
                success = self._apply_split_from_ui()
                if not success:
                    return
        
        except Exception as e:
            logger.error(f"Error applying changes to node {self.node.node_id}: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self, "Apply Changes Error",
                f"Could not apply changes: {str(e)}"
            )
            return
        
        self.propertiesChanged.emit()
        
        self.update_ui()
    
    def _apply_split_from_ui(self):
        """Apply split based on current UI settings"""
        feature = self.feature_combo.currentText()
        split_type = self.split_type_combo.currentText().lower()
        
        if not feature:
            QMessageBox.warning(self, "Apply Split", "Please select a feature for the split.")
            return False
        
        split_info = {
            'feature': feature,
            'split_type': split_type
        }
        
        if split_type == "numeric":
            value = self.split_value_spin.value()
            split_info['split_value'] = value
            split_info['split_operator'] = '<='
            
        else:
            categories = {}
            
            for row in range(self.categories_table.rowCount()):
                cat_item = self.categories_table.item(row, 0)
                child_item = self.categories_table.item(row, 1)
                
                if cat_item and child_item:
                    cat = cat_item.text().strip()
                    child_text = child_item.text().strip()
                    child_idx = int(child_text) if child_text.isdigit() else 0
                    
                    if cat:
                        categories[cat] = child_idx
            
            if not categories:
                QMessageBox.warning(self, "Apply Split", "Please add categories for the categorical split.")
                return False
                
            split_info['split_categories'] = categories
        
        if self.model and hasattr(self.model, 'apply_manual_split'):
            success = self.model.apply_manual_split(self.node.node_id, split_info)
            if success:
                logger.info(f"Successfully applied {split_type} split on {feature} to node {self.node.node_id}")
                if split_type == "numeric":
                    logger.info(f"Split creates child nodes: {self.node.node_id}_L (≤{value}) and {self.node.node_id}_R (>{value})")
                else:
                    logger.info(f"Split creates {len(set(categories.values()))} child nodes for categories: {list(categories.keys())}")
                return True
            else:
                QMessageBox.warning(self, "Apply Split", "Failed to apply split. Check the log for details.")
                return False
        else:
            if split_type == "numeric":
                self.node.set_split(feature, value, 'numeric')
            else:
                self.node.set_categorical_split(feature, categories)
            
            logger.warning(f"Applied split to node {self.node.node_id} without model - child nodes not created")
            return True
    
    def reset(self):
        """Reset controls to match the node's current state"""
        self.update_ui()


class SplitFinderWidget(QWidget):
    """Widget for finding optimal splits for a node"""
    
    splitSelected = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """
        Initialize the split finder widget
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.node = None
        self.model = None
        self.potential_splits = []
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        info_layout = QFormLayout()
        
        self.id_label = QLabel("None")
        info_layout.addRow("Node ID:", self.id_label)
        
        self.samples_label = QLabel("0")
        info_layout.addRow("Samples:", self.samples_label)
        
        self.impurity_label = QLabel("0.0")
        info_layout.addRow("Impurity:", self.impurity_label)
        
        layout.addLayout(info_layout)
        
        find_layout = QHBoxLayout()
        self.find_split_btn = QPushButton("Find Optimal Splits")
        find_layout.addWidget(self.find_split_btn)
        layout.addLayout(find_layout)
        
        self.split_table = QTableWidget(0, 6)
        self.split_table.setHorizontalHeaderLabels([
            "Feature", "Split Condition", "Gini Decrease", 
            "Left Samples", "Right Samples", "Class Distribution"
        ])
        header = self.split_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Feature
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Split Condition
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Gini Decrease
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Left Samples
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Right Samples
        header.setSectionResizeMode(5, QHeaderView.Stretch)           # Class Distribution
        self.split_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.split_table)
        
        self.apply_btn = QPushButton("Apply Selected Split")
        layout.addWidget(self.apply_btn)
        
        help_text = QLabel(
            "<b>Find Split Guide:</b><br>"
            "• <b>Gini Decrease:</b> Higher values (e.g., 0.15) indicate better class separation<br>"
            "• <b>Sample Size:</b> Avoid splits with very small nodes (&lt;10 records)<br>"
            "• <b>Class Distribution:</b> Shows percentage breakdown in child nodes<br>"
            "• <b>Best Practice:</b> Choose splits with high Gini decrease and balanced sample sizes"
        )
        help_text.setStyleSheet("background-color: #f0f8ff; padding: 8px; border: 1px solid #ccc; border-radius: 4px;")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        
        self.find_split_btn.clicked.connect(self.find_splits)
        self.apply_btn.clicked.connect(self.apply_split)
        
        self.apply_btn.setEnabled(False)
    
    def set_node(self, node: TreeNode, model: Optional[BespokeDecisionTree] = None):
        """
        Set the node to find splits for
        
        Args:
            node: TreeNode to find splits for
            model: Decision tree model
        """
        self.node = node
        self.model = model
        self.potential_splits = []
        
        self.split_table.setRowCount(0)
        
        if node:
            self.id_label.setText(node.node_id)
            self.samples_label.setText(str(node.samples))
            
            if node.impurity is not None:
                self.impurity_label.setText(f"{node.impurity:.6f}")
            else:
                self.impurity_label.setText("N/A")
        else:
            self.id_label.setText("None")
            self.samples_label.setText("0")
            self.impurity_label.setText("0.0")
        
        self.find_split_btn.setEnabled(node is not None and model is not None)
        
        self.apply_btn.setEnabled(False)
    
    def find_splits(self):
        """Find optimal splits for the node"""
        if not self.node or not self.model:
            return
        
        if not hasattr(self.model, 'find_split_for_node'):
            QMessageBox.warning(
                self, "Find Split", 
                "The model does not support finding splits for a specific node."
            )
            return
        
        try:
            from PyQt5.QtWidgets import QProgressDialog
            progress = QProgressDialog("Finding optimal splits...", "Cancel", 0, 100, self)
            progress.setWindowTitle("Find Split")
            progress.setModal(True)
            progress.setValue(0)
            progress.show()
            
            self.setEnabled(False)
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            QApplication.processEvents()
            
            progress.setValue(20)
            progress.setLabelText("Analyzing features and samples...")
            QApplication.processEvents()
            
            progress.setValue(50)
            progress.setLabelText("Evaluating potential splits...")
            QApplication.processEvents()
            
            self.potential_splits = self.model.find_split_for_node(self.node.node_id)
            
            progress.setValue(80)
            progress.setLabelText("Updating results table...")
            QApplication.processEvents()
            
            self.update_split_table()
            
            progress.setValue(100)
            progress.close()
            
            self.apply_btn.setEnabled(len(self.potential_splits) > 0)
            
            if len(self.potential_splits) == 0:
                QMessageBox.information(
                    self, "Find Split", 
                    "No beneficial splits found for this node. The node may already be well-separated or have too few samples."
                )
            else:
                QMessageBox.information(
                    self, "Find Split", 
                    f"Found {len(self.potential_splits)} potential splits. Best gain: {self.potential_splits[0]['gain']:.4f}"
                )
            
        except Exception as e:
            logger.error(f"Error finding splits: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self, "Find Split Error", 
                f"Could not find splits: {str(e)}"
            )
        finally:
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
            if 'progress' in locals():
                progress.close()
    
    def update_split_table(self):
        """Update the split table with potential splits"""
        self.split_table.setRowCount(0)
        
        if not self.potential_splits:
            return
        
        self.split_table.setRowCount(len(self.potential_splits))
        
        for i, split in enumerate(self.potential_splits):
            feature_item = QTableWidgetItem(split.get('feature', ''))
            self.split_table.setItem(i, 0, feature_item)
            
            split_desc = split.get('split_desc', '')
            condition_item = QTableWidgetItem(split_desc)
            self.split_table.setItem(i, 1, condition_item)
            
            gain = split.get('gain', 0.0)
            gain_item = QTableWidgetItem(f"{gain:.4f}")
            if gain >= 0.05:
                gain_item.setBackground(QColor(144, 238, 144))  # Light green
            elif gain >= 0.02:
                gain_item.setBackground(QColor(255, 255, 224))  # Light yellow  
            else:
                gain_item.setBackground(QColor(255, 182, 193))  # Light pink
            self.split_table.setItem(i, 2, gain_item)
            
            split_info = split.get('split_info', {})
            
            left_samples = split_info.get('left_samples', 0)
            left_item = QTableWidgetItem(str(left_samples))
            if left_samples < 10:
                left_item.setBackground(QColor(255, 182, 193))  # Light pink
            self.split_table.setItem(i, 3, left_item)
            
            right_samples = split_info.get('right_samples', 0)
            right_item = QTableWidgetItem(str(right_samples))
            if right_samples < 10:
                right_item.setBackground(QColor(255, 182, 193))  # Light pink
            self.split_table.setItem(i, 4, right_item)
            
            class_dist_text = self._format_class_distribution(split_info)
            dist_item = QTableWidgetItem(class_dist_text)
            self.split_table.setItem(i, 5, dist_item)
        
        self.split_table.sortItems(2, Qt.DescendingOrder)
        
        if self.split_table.rowCount() > 0:
            self.split_table.selectRow(0)
    
    def _format_class_distribution(self, split_info):
        """Format class distribution for display"""
        try:
            left_dist = split_info.get('left_class_distribution', {})
            right_dist = split_info.get('right_class_distribution', {})
            
            if not left_dist or not right_dist:
                return "N/A"
            
            dist_text = "Left: "
            for cls, count in left_dist.items():
                total = sum(left_dist.values())
                pct = CentralMetricsCalculator.calculate_percentage(count, total, 1)
                dist_text += f"{cls}:{pct:.1f}% "
            
            dist_text += "| Right: "
            for cls, count in right_dist.items():
                total = sum(right_dist.values())
                pct = CentralMetricsCalculator.calculate_percentage(count, total, 1)
                dist_text += f"{cls}:{pct:.1f}% "
                
            return dist_text.strip()
            
        except Exception as e:
            logger.error(f"Error formatting class distribution: {e}")
            return "N/A"
    
    def apply_split(self):
        """Apply the selected split to the node"""
        if not self.node or not self.model or not self.potential_splits:
            return
        
        selected_rows = self.split_table.selectedIndexes()
        if not selected_rows:
            return
        
        row = selected_rows[0].row()
        
        if not hasattr(self.model, 'apply_split_to_node'):
            QMessageBox.warning(
                self, "Apply Split", 
                "The model does not support applying splits to a specific node."
            )
            return
        
        try:
            split_info = self.potential_splits[row].get('split_info', {})
            
            if not split_info:
                QMessageBox.warning(
                    self, "Apply Split", 
                    "No split information available for the selected split."
                )
                return
            
            self.splitSelected.emit(split_info)
            
        except Exception as e:
            logger.error(f"Error applying split: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self, "Apply Split Error", 
                f"Could not apply split: {str(e)}"
            )


class NodeReportWidget(QWidget):
    """Widget for displaying node reports and statistics"""
    
    def __init__(self, parent=None):
        """
        Initialize the node report widget
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.node = None
        self.node_analyzer = NodeAnalyzer()
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout()
        self.summary_tab.setLayout(summary_layout)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text)
        
        self.tab_widget.addTab(self.summary_tab, "Summary")
        
        self.rules_tab = QWidget()
        rules_layout = QVBoxLayout()
        self.rules_tab.setLayout(rules_layout)
        
        self.rules_text = QTextEdit()
        self.rules_text.setReadOnly(True)
        rules_layout.addWidget(self.rules_text)
        
        self.tab_widget.addTab(self.rules_tab, "Rules")
        
        self.dist_tab = QWidget()
        dist_layout = QVBoxLayout()
        self.dist_tab.setLayout(dist_layout)
        
        self.dist_text = QTextEdit()
        self.dist_text.setReadOnly(True)
        dist_layout.addWidget(self.dist_text)
        
        self.tab_widget.addTab(self.dist_tab, "Distribution")
    
    def set_node(self, node: TreeNode):
        """
        Set the node to display reports for
        
        Args:
            node: TreeNode to report on
        """
        self.node = node
        
        self.update_reports()
    
    def update_reports(self):
        """Update all report tabs with node information"""
        if not self.node:
            self.summary_text.clear()
            self.rules_text.clear()
            self.dist_text.clear()
            return
        
        try:
            report = self.node_analyzer.get_node_report(self.node)
            
            if 'error' in report:
                logger.warning(f"Node statistics error: {report['error']}")
                self.show_statistics_error(report['error'])
                return
            
            self.update_summary_tab(report)
            
            self.update_rules_tab(report)
            
            self.update_distribution_tab(report)
            
        except Exception as e:
            logger.error(f"Failed to generate node statistics: {e}", exc_info=True)
            self.show_statistics_error(f"Unable to calculate node statistics: {str(e)}")
    
    def show_statistics_error(self, error_message: str):
        """Show error message in statistics tabs"""
        error_text = f"⚠️ Statistics Error\n\n{error_message}\n\nNode statistics are temporarily unavailable."
        
        if hasattr(self, 'summary_text'):
            self.summary_text.clear()
            self.summary_text.append(error_text)
        
        if hasattr(self, 'rules_text'):
            self.rules_text.clear()
            self.rules_text.append(error_text)
        
        if hasattr(self, 'dist_text'):
            self.dist_text.clear()
            self.dist_text.append(error_text)
    
    def update_summary_tab(self, report: Dict[str, Any]):
        """
        Update the summary tab with node information
        
        Args:
            report: Node report dictionary
        """
        html = "<html><body style='font-family: Arial, sans-serif;'>"
        
        html += f"<h2>Node {report.get('node_id', 'Unknown')}</h2>"
        html += f"<p><b>Type:</b> {'Terminal' if report.get('is_terminal', False) else 'Internal'}</p>"
        
        html += f"<p><b>Depth:</b> {report.get('depth', 0)}</p>"
        html += f"<p><b>Samples:</b> {report.get('samples', 0)}</p>"
        
        if 'impurity' in report and report['impurity'] is not None:
            html += f"<p><b>Impurity:</b> {report['impurity']:.6f}</p>"
        
        if report.get('is_terminal', False) and 'prediction' in report:
            html += f"<p><b>Prediction:</b> {report['prediction']}</p>"
            
            if 'prediction_confidence' in report:
                html += f"<p><b>Confidence:</b> {report['prediction_confidence']:.4f}</p>"
        
        if not report.get('is_terminal', True) and 'split_feature' in report:
            html += "<h3>Split Information</h3>"
            html += f"<p><b>Feature:</b> {report['split_feature']}</p>"
            
            if 'split_type' in report and report['split_type']:
                html += f"<p><b>Split Type:</b> {report['split_type'].capitalize()}</p>"
            
            if 'split_rule' in report:
                html += f"<p><b>Split Rule:</b> {report['split_rule']}</p>"
        
        if any(metric in report for metric in ['accuracy', 'precision', 'recall', 'f1_score']):
            html += "<h3>Performance Metrics</h3>"
            
            if 'accuracy' in report:
                html += f"<p><b>Accuracy:</b> {report['accuracy']:.4f}</p>"
            
            if 'precision' in report:
                html += f"<p><b>Precision:</b> {report['precision']:.4f}</p>"
            
            if 'recall' in report:
                html += f"<p><b>Recall:</b> {report['recall']:.4f}</p>"
            
            if 'f1_score' in report:
                html += f"<p><b>F1 Score:</b> {report['f1_score']:.4f}</p>"
        
        html += "</body></html>"
        
        self.summary_text.setHtml(html)
    
    def update_rules_tab(self, report: Dict[str, Any]):
        """
        Update the rules tab with node path rules
        
        Args:
            report: Node report dictionary
        """
        html = "<html><body style='font-family: Arial, sans-serif;'>"
        
        path = report.get('path', [])
        
        if path:
            html += "<h3>Path to Node (Rules)</h3>"
            html += "<ol>"
            
            for rule in path:
                html += f"<li>{rule.get('rule', '')}</li>"
            
            html += "</ol>"
        else:
            html += "<p>No path rules available (this may be the root node).</p>"
        
        html += "</body></html>"
        
        self.rules_text.setHtml(html)
    
    def update_distribution_tab(self, report: Dict[str, Any]):
        """
        Update the distribution tab with class distribution
        
        Args:
            report: Node report dictionary
        """
        html = "<html><body style='font-family: Arial, sans-serif;'>"
        
        class_dist = report.get('class_distribution', {})
        
        if class_dist:
            html += "<h3>Class Distribution</h3>"
            
            total = sum(class_dist.values())
            
            html += "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse: collapse;'>"
            html += "<tr><th>Class</th><th>Count</th><th>Percentage</th></tr>"
            
            for cls, count in sorted(class_dist.items()):
                percent = CentralMetricsCalculator.calculate_percentage(count, total, 2)
                html += f"<tr><td>{cls}</td><td>{count}</td><td>{percent:.2f}%</td></tr>"
            
            html += "</table>"
            
            # Add note about majority class
            prediction = report.get('prediction')
            if prediction is not None:
                html += f"<p><b>Majority Class (Prediction):</b> {prediction}</p>"
        else:
            html += "<p>No class distribution available.</p>"
        
        html += "</body></html>"
        
        self.dist_text.setHtml(html)


class NodeEditorDialog(QDialog):
    """Dialog for editing tree nodes"""
    
    def __init__(self, node: TreeNode, model: Optional[BespokeDecisionTree] = None, 
               feature_names: List[str] = None, parent=None):
        """
        Initialize the node editor dialog
        
        Args:
            node: TreeNode to edit
            model: Decision tree model (optional)
            feature_names: List of available feature names (optional)
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.node = node
        self.model = model
        self.feature_names = feature_names or []
        
        self.init_ui()
        
        self.properties_widget.set_node(node, feature_names)
        self.split_finder_widget.set_node(node, model)
        self.report_widget.set_node(node)
        
        if node:
            self.setWindowTitle(f"Edit Node {node.node_id}")
        else:
            self.setWindowTitle("Node Editor")
    
    def init_ui(self):
        """Initialize the user interface"""
        self.resize(800, 600)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)
        
        self.tab_widget = QTabWidget()
        
        self.properties_widget = NodePropertiesWidget()
        self.tab_widget.addTab(self.properties_widget, "Properties")
        
        self.split_finder_widget = SplitFinderWidget()
        self.tab_widget.addTab(self.split_finder_widget, "Find Split")
        
        splitter.addWidget(self.tab_widget)
        
        self.report_widget = NodeReportWidget()
        splitter.addWidget(self.report_widget)
        
        splitter.setSizes([400, 200])
        
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.connect_signals()
    
    def connect_signals(self):
        """Connect widget signals"""
        self.properties_widget.propertiesChanged.connect(self.on_properties_changed)
        
        self.split_finder_widget.splitSelected.connect(self.on_split_selected)
    
    def on_properties_changed(self):
        """Handle properties changed signal"""
        self.report_widget.set_node(self.node)
    
    def on_split_selected(self, split_info: Dict[str, Any]):
        """
        Handle split selected signal
        
        Args:
            split_info: Split information dictionary
        """
        if self.model and hasattr(self.model, 'apply_manual_split'):
            try:
                logger.info(f"NodeEditorDialog: Applying split {split_info} to node {self.node.node_id}")
                
                success = self.model.apply_manual_split(self.node.node_id, split_info)
                
                if success:
                    logger.info(f"NodeEditorDialog: Split applied successfully to node {self.node.node_id}")
                    
                    self.properties_widget.update_ui()
                    self.report_widget.set_node(self.node)
                    
                    QMessageBox.information(
                        self, "Apply Split", 
                        "Split was successfully applied to the node."
                    )
                else:
                    logger.warning(f"NodeEditorDialog: Failed to apply split to node {self.node.node_id}")
                    QMessageBox.warning(
                        self, "Apply Split", 
                        "Failed to apply the split to the node."
                    )
            
            except Exception as e:
                logger.error(f"NodeEditorDialog: Error applying split to node {self.node.node_id}: {str(e)}", exc_info=True)
                QMessageBox.critical(
                    self, "Apply Split Error", 
                    f"Could not apply split: {str(e)}"
                )


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    
    test_node = TreeNode(node_id="test")
    test_node.update_stats(
        samples=1000,
        class_counts={"Yes": 400, "No": 600},
        impurity=0.48
    )
    test_node.set_split("Income", 50000, "numeric")
    
    child = TreeNode(node_id="test_L", parent=test_node, depth=1)
    child.update_stats(
        samples=500,
        class_counts={"Yes": 300, "No": 200},
        impurity=0.48
    )
    test_node.add_child(child)
    
    app = QApplication(sys.argv)
    dialog = NodeEditorDialog(test_node, feature_names=["Age", "Income", "Credit_Score"])
    dialog.show()
    
    sys.exit(app.exec_())