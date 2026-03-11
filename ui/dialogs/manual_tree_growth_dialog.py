#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manual Tree Growth Dialog for Bespoke Utility
Interactive interface for manual decision tree construction
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont, QIcon, QColor, QPen, QBrush
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QTabWidget, QWidget, QListWidget, QListWidgetItem,
    QSplitter, QProgressBar, QMessageBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QFrame, QSizePolicy, QTreeWidget, QTreeWidgetItem, QGraphicsView,
    QGraphicsScene, QGraphicsItem, QGraphicsEllipseItem, QGraphicsRectItem,
    QGraphicsTextItem, QGraphicsLineItem
)

from models.node import TreeNode
from models.split_finder import SplitFinder
from models.decision_tree import SplitCriterion

logger = logging.getLogger(__name__)


class TreeGraphicsNode(QGraphicsRectItem):
    """Graphics item representing a tree node"""
    
    def __init__(self, tree_node: TreeNode, x: float, y: float, width: float, height: float):
        super().__init__(x, y, width, height)
        self.tree_node = tree_node
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        
        if tree_node.is_terminal:
            self.setBrush(QBrush(QColor(144, 238, 144)))  # Light green for leaves
        else:
            self.setBrush(QBrush(QColor(173, 216, 230)))  # Light blue for internal nodes
            
        self.setPen(QPen(QColor(0, 0, 0), 2))
        
        self.text_item = QGraphicsTextItem(self.get_node_text(), self)
        self.text_item.setPos(x + 5, y + 5)
        
    def get_node_text(self) -> str:
        """Get display text for the node"""
        if self.tree_node.is_terminal:
            return f"Leaf\\nPred: {self.tree_node.prediction}\\nSamples: {self.tree_node.samples}"
        else:
            feature = self.tree_node.split_feature or "Unknown"
            value = self.tree_node.split_value or 0
            return f"{feature}\\n<= {value}\\nSamples: {self.tree_node.samples}"
            
    def update_text(self):
        """Update the text display"""
        self.text_item.setPlainText(self.get_node_text())


class ManualTreeGrowthDialog(QDialog):
    """Dialog for manual tree construction and editing"""
    
    nodeSelected = pyqtSignal(str)  # node_id
    splitRequested = pyqtSignal(str)  # node_id
    
    def __init__(self, tree_model, training_data: pd.DataFrame, 
                 target_column: str, parent=None):
        super().__init__(parent)
        
        self.tree_model = tree_model
        self.training_data = training_data
        self.target_column = target_column
        self.X = training_data.drop(columns=[target_column])
        self.y = training_data[target_column]
        
        self.split_finder = SplitFinder()
        
        self.selected_node = None
        self.graphics_nodes = {}  # node_id -> TreeGraphicsNode
        
        self.setWindowTitle("🌳 Manual Tree Growth")
        self.setModal(False)  # Allow interaction with main window
        self.resize(1400, 900)
        
        self.setStyleSheet("""
            QDialog {
                background-color: #f8fafc;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                color: #1e293b;
                font-size: 14px;
            }
            QLabel[styleClass=\"header\"] {
                color: #0f172a;
                font-size: 18px;
                font-weight: 600;
                padding: 12px 0px;
            }
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                font-weight: 600;
                font-size: 13px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
            QPushButton:disabled {
                background-color: #94a3b8;
                color: #cbd5e1;
            }
            QPushButton[styleClass=\"secondary\"] {
                background-color: #f1f5f9;
                color: #475569;
                border: 1px solid #cbd5e1;
            }
            QPushButton[styleClass=\"secondary\"]:hover {
                background-color: #e2e8f0;
            }
            QPushButton[styleClass=\"danger\"] {
                background-color: #ef4444;
            }
            QPushButton[styleClass=\"danger\"]:hover {
                background-color: #dc2626;
            }
            QGroupBox {
                font-weight: 600;
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 16px;
                background-color: white;
                color: #374151;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
                background-color: #f8fafc;
            }
            QTableWidget {
                background-color: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                gridline-color: #f1f5f9;
                font-size: 13px;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #f1f5f9;
            }
            QTableWidget::item:selected {
                background-color: #eff6ff;
                color: #1e40af;
            }
            QHeaderView::section {
                background-color: #f8fafc;
                border: none;
                border-bottom: 2px solid #e2e8f0;
                padding: 12px 8px;
                font-weight: 600;
                color: #374151;
            }
        """)
        
        self.setupUI()
        self.buildTreeVisualization()
        
    def setupUI(self):
        """Setup the main UI"""
        layout = QVBoxLayout()
        
        header_label = QLabel("Manual Tree Growth")
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header_label)
        
        main_splitter = QSplitter(Qt.Horizontal)
        
        left_panel = self.createNavigationPanel()
        main_splitter.addWidget(left_panel)
        
        center_panel = self.createVisualizationPanel()
        main_splitter.addWidget(center_panel)
        
        right_panel = self.createDetailsPanel()
        main_splitter.addWidget(right_panel)
        
        main_splitter.setSizes([300, 700, 400])
        
        layout.addWidget(main_splitter)
        
        button_layout = QHBoxLayout()
        
        self.save_tree_button = QPushButton("Save Tree")
        self.save_tree_button.clicked.connect(self.saveTree)
        
        self.reset_tree_button = QPushButton("Reset Tree")
        self.reset_tree_button.clicked.connect(self.resetTree)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        
        button_layout.addWidget(self.save_tree_button)
        button_layout.addWidget(self.reset_tree_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def createNavigationPanel(self) -> QWidget:
        """Create the tree navigation panel"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        nav_group = QGroupBox("Tree Navigator")
        nav_layout = QVBoxLayout()
        
        search_layout = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search nodes...")
        self.search_edit.textChanged.connect(self.searchNodes)
        
        self.search_button = QPushButton("Go to Node")
        self.search_button.clicked.connect(self.goToNode)
        
        search_layout.addWidget(self.search_edit)
        search_layout.addWidget(self.search_button)
        nav_layout.addLayout(search_layout)
        
        self.tree_navigator = QTreeWidget()
        self.tree_navigator.setHeaderLabel("Tree Structure")
        self.tree_navigator.itemClicked.connect(self.onNavigatorItemClicked)
        nav_layout.addWidget(self.tree_navigator)
        
        nav_group.setLayout(nav_layout)
        layout.addWidget(nav_group)
        
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QVBoxLayout()
        
        self.expand_all_button = QPushButton("Expand All Nodes")
        self.expand_all_button.clicked.connect(self.expandAllNodes)
        actions_layout.addWidget(self.expand_all_button)
        
        self.collapse_all_button = QPushButton("Collapse All Nodes")
        self.collapse_all_button.clicked.connect(self.collapseAllNodes)
        actions_layout.addWidget(self.collapse_all_button)
        
        self.auto_layout_button = QPushButton("Auto Layout Tree")
        self.auto_layout_button.clicked.connect(self.autoLayoutTree)
        actions_layout.addWidget(self.auto_layout_button)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        stats_group = QGroupBox("Tree Statistics")
        stats_layout = QFormLayout()
        
        self.depth_label = QLabel("0")
        self.nodes_label = QLabel("1")
        self.leaves_label = QLabel("1")
        self.accuracy_label = QLabel("N/A")
        
        stats_layout.addRow("Depth:", self.depth_label)
        stats_layout.addRow("Nodes:", self.nodes_label)
        stats_layout.addRow("Leaves:", self.leaves_label)
        stats_layout.addRow("Accuracy:", self.accuracy_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def createVisualizationPanel(self) -> QWidget:
        """Create the tree visualization panel"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        controls_layout = QHBoxLayout()
        
        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(self.zoomIn)
        
        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(self.zoomOut)
        
        self.fit_view_button = QPushButton("Fit to View")
        self.fit_view_button.clicked.connect(self.fitToView)
        
        controls_layout.addWidget(self.zoom_in_button)
        controls_layout.addWidget(self.zoom_out_button)
        controls_layout.addWidget(self.fit_view_button)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        self.graphics_view = QGraphicsView()
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.setRenderHint(self.graphics_view.Antialiasing)
        
        layout.addWidget(self.graphics_view)
        
        widget.setLayout(layout)
        return widget
        
    def createDetailsPanel(self) -> QWidget:
        """Create the node details and actions panel"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        details_group = QGroupBox("Node Details")
        details_layout = QFormLayout()
        
        self.node_id_label = QLabel("None")
        self.node_depth_label = QLabel("0")
        self.node_samples_label = QLabel("0")
        self.node_impurity_label = QLabel("0.0")
        self.node_prediction_label = QLabel("None")
        
        details_layout.addRow("Node ID:", self.node_id_label)
        details_layout.addRow("Depth:", self.node_depth_label)
        details_layout.addRow("Samples:", self.node_samples_label)
        details_layout.addRow("Impurity:", self.node_impurity_label)
        details_layout.addRow("Prediction:", self.node_prediction_label)
        
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)
        
        self.split_group = QGroupBox("Split Information")
        split_layout = QFormLayout()
        
        self.split_feature_label = QLabel("None")
        self.split_value_label = QLabel("None")
        self.split_operator_label = QLabel("<=")
        self.split_improvement_label = QLabel("0.0")
        
        split_layout.addRow("Feature:", self.split_feature_label)
        split_layout.addRow("Value:", self.split_value_label)
        split_layout.addRow("Operator:", self.split_operator_label)
        split_layout.addRow("Improvement:", self.split_improvement_label)
        
        self.split_group.setLayout(split_layout)
        layout.addWidget(self.split_group)
        
        actions_group = QGroupBox("Node Actions")
        actions_layout = QVBoxLayout()
        
        self.find_split_button = QPushButton("Find Split")
        self.find_split_button.clicked.connect(self.findSplit)
        self.find_split_button.setEnabled(False)
        actions_layout.addWidget(self.find_split_button)
        
        self.edit_split_button = QPushButton("Edit Split")
        self.edit_split_button.clicked.connect(self.editSplit)
        self.edit_split_button.setEnabled(False)
        actions_layout.addWidget(self.edit_split_button)
        
        self.delete_split_button = QPushButton("Delete Split")
        self.delete_split_button.clicked.connect(self.deleteSplit)
        self.delete_split_button.setEnabled(False)
        actions_layout.addWidget(self.delete_split_button)
        
        self.make_leaf_button = QPushButton("Convert to Leaf")
        self.make_leaf_button.clicked.connect(self.makeLeaf)
        self.make_leaf_button.setEnabled(False)
        actions_layout.addWidget(self.make_leaf_button)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        candidates_group = QGroupBox("Split Candidates")
        candidates_layout = QVBoxLayout()
        
        self.candidates_table = QTableWidget()
        self.candidates_table.setColumnCount(4)
        self.candidates_table.setHorizontalHeaderLabels([
            'Feature', 'Value', 'Improvement', 'Select'
        ])
        self.candidates_table.setMaximumHeight(200)
        self.candidates_table.setAlternatingRowColors(True)
        
        header = self.candidates_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        
        candidates_layout.addWidget(self.candidates_table)
        candidates_group.setLayout(candidates_layout)
        layout.addWidget(candidates_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def buildTreeVisualization(self):
        """Build the initial tree visualization"""
        self.graphics_scene.clear()
        self.graphics_nodes.clear()
        
        if not self.tree_model or not self.tree_model.root:
            return
            
        self.updateTreeNavigator()
        
        self.layoutTreeNodes(self.tree_model.root, 0, 0, 800)
        
        self.updateTreeStatistics()
        
    def layoutTreeNodes(self, node: TreeNode, x: float, y: float, width: float):
        """Layout tree nodes in the graphics scene"""
        node_width = 120
        node_height = 60
        level_height = 100
        
        graphics_node = TreeGraphicsNode(node, x, y, node_width, node_height)
        self.graphics_scene.addItem(graphics_node)
        self.graphics_nodes[node.node_id] = graphics_node
        
        graphics_node.mousePressEvent = lambda event: self.selectNode(node.node_id)
        
        if node.children:
            child_width = width / len(node.children)
            for i, child in enumerate(node.children):
                child_x = x + i * child_width + (child_width - node_width) / 2
                child_y = y + level_height
                
                line = QGraphicsLineItem(
                    x + node_width/2, y + node_height,
                    child_x + node_width/2, child_y
                )
                line.setPen(QPen(QColor(0, 0, 0), 2))
                self.graphics_scene.addItem(line)
                
                self.layoutTreeNodes(child, child_x, child_y, child_width)
                
    def updateTreeNavigator(self):
        """Update the tree structure navigator"""
        self.tree_navigator.clear()
        
        if not self.tree_model or not self.tree_model.root:
            return
            
        root_item = self.createNavigatorItem(self.tree_model.root)
        self.tree_navigator.addTopLevelItem(root_item)
        self.tree_navigator.expandAll()
        
    def createNavigatorItem(self, node: TreeNode) -> QTreeWidgetItem:
        """Create a navigator item for a tree node"""
        if node.is_terminal:
            text = f"Leaf [{node.node_id}]: {node.prediction} ({node.samples} samples)"
        else:
            feature = node.split_feature or "Unknown"
            value = node.split_value or 0
            text = f"Split [{node.node_id}]: {feature} <= {value} ({node.samples} samples)"
            
        item = QTreeWidgetItem([text])
        item.setData(0, Qt.UserRole, node.node_id)
        
        for child in node.children:
            child_item = self.createNavigatorItem(child)
            item.addChild(child_item)
            
        return item
        
    def selectNode(self, node_id: str):
        """Select a node for editing"""
        node = self.findNodeById(node_id)
        if not node:
            return
            
        self.selected_node = node
        
        self.updateNodeDetails(node)
        self.updateActionButtons(node)
        
        for gnode in self.graphics_nodes.values():
            gnode.setSelected(False)
            
        if node_id in self.graphics_nodes:
            self.graphics_nodes[node_id].setSelected(True)
            
    def updateNodeDetails(self, node: TreeNode):
        """Update node details display"""
        self.node_id_label.setText(node.node_id)
        self.node_depth_label.setText(str(node.depth))
        self.node_samples_label.setText(str(node.samples))
        self.node_impurity_label.setText(f"{getattr(node, 'impurity', 0.0):.4f}")
        
        if node.is_terminal:
            self.node_prediction_label.setText(str(node.prediction))
            self.split_group.setVisible(False)
        else:
            self.node_prediction_label.setText("N/A")
            self.split_group.setVisible(True)
            
            self.split_feature_label.setText(str(node.split_feature))
            self.split_value_label.setText(str(node.split_value))
            self.split_operator_label.setText(getattr(node, 'split_operator', '<='))
            
            improvement = getattr(node, 'split_improvement', 0.0)
            self.split_improvement_label.setText(f"{improvement:.4f}")
            
    def updateActionButtons(self, node: TreeNode):
        """Update action button states"""
        is_leaf = node.is_terminal
        
        self.find_split_button.setEnabled(is_leaf)
        self.edit_split_button.setEnabled(not is_leaf)
        self.delete_split_button.setEnabled(not is_leaf)
        self.make_leaf_button.setEnabled(not is_leaf)
        
    def findSplit(self):
        """Find optimal splits for the selected node"""
        if not self.selected_node:
            return
            
        node_data = self.getNodeData(self.selected_node)
        if node_data is None or len(node_data) < 2:
            QMessageBox.warning(self, "Insufficient Data", 
                              "Not enough data to find splits for this node.")
            return
            
        X_node = node_data.drop(columns=[self.target_column])
        y_node = node_data[self.target_column]
        
        try:
            splits = self.split_finder.find_all_splits(
                X_node, y_node, 
                criterion=SplitCriterion.GINI,
                max_splits=10
            )
            
            self.populateCandidatesTable(splits)
            
        except Exception as e:
            QMessageBox.critical(self, "Error Finding Splits", 
                               f"Error occurred while finding splits: {str(e)}")
            
    def populateCandidatesTable(self, splits: List[Dict[str, Any]]):
        """Populate the split candidates table"""
        self.candidates_table.setRowCount(len(splits))
        
        for i, split in enumerate(splits):
            feature_item = QTableWidgetItem(str(split.get('feature', 'Unknown')))
            self.candidates_table.setItem(i, 0, feature_item)
            
            value_item = QTableWidgetItem(f"{split.get('value', 0):.4f}")
            self.candidates_table.setItem(i, 1, value_item)
            
            improvement = split.get('improvement', 0.0)
            improvement_item = QTableWidgetItem(f"{improvement:.4f}")
            self.candidates_table.setItem(i, 2, improvement_item)
            
            select_button = QPushButton("Apply")
            select_button.clicked.connect(lambda checked, s=split: self.applySplit(s))
            self.candidates_table.setCellWidget(i, 3, select_button)
            
    def applySplit(self, split_info: Dict[str, Any]):
        """Apply a split to the selected node"""
        if not self.selected_node:
            return
            
        try:
            self.split_finder.apply_split(
                self.selected_node, 
                split_info['feature'],
                split_info['value'],
                split_info.get('operator', '<=')
            )
            
            self.updateNodeAfterSplit(self.selected_node)
            
            self.buildTreeVisualization()
            
            self.selectNode(self.selected_node.node_id)
            
        except Exception as e:
            QMessageBox.critical(self, "Error Applying Split", 
                               f"Error occurred while applying split: {str(e)}")
            
    def editSplit(self):
        """Edit the split for the selected node"""
        if not self.selected_node or self.selected_node.is_terminal:
            return
            
        dialog = EditSplitDialog(self.selected_node, self.X.columns.tolist(), self)
        if dialog.exec_() == QDialog.Accepted:
            new_feature, new_value, new_operator = dialog.getSplitParameters()
            
            try:
                self.selected_node.split_feature = new_feature
                self.selected_node.split_value = new_value
                self.selected_node.split_operator = new_operator
                
                self.rebuildNodeChildren(self.selected_node)
                
                self.buildTreeVisualization()
                self.selectNode(self.selected_node.node_id)
                
            except Exception as e:
                QMessageBox.critical(self, "Error Editing Split", 
                                   f"Error occurred while editing split: {str(e)}")
                                   
    def deleteSplit(self):
        """Delete the split and convert node to leaf"""
        if not self.selected_node or self.selected_node.is_terminal:
            return
            
        reply = QMessageBox.question(self, "Delete Split", 
                                   "Are you sure you want to delete this split and convert the node to a leaf?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.makeLeaf()
            
    def makeLeaf(self):
        """Convert the selected node to a leaf"""
        if not self.selected_node:
            return
            
        node_data = self.getNodeData(self.selected_node)
        if node_data is not None and len(node_data) > 0:
            prediction = node_data[self.target_column].mode().iloc[0]
        else:
            prediction = "Unknown"
            
        self.selected_node.is_terminal = True
        self.selected_node.prediction = prediction
        self.selected_node.children = []
        self.selected_node.split_feature = None
        self.selected_node.split_value = None
        self.selected_node.split_operator = None
        
        self.buildTreeVisualization()
        self.selectNode(self.selected_node.node_id)
        
    def getNodeData(self, node: TreeNode) -> Optional[pd.DataFrame]:
        """Get the data subset for a specific node"""
        
        if node == self.tree_model.root:
            return self.training_data
        else:
            sample_size = min(node.samples, len(self.training_data))
            return self.training_data.sample(n=sample_size, random_state=42)
            
    def updateNodeAfterSplit(self, node: TreeNode):
        """Update node properties after applying a split"""
        node_data = self.getNodeData(node)
        if node_data is None:
            return
            
        node.samples = len(node_data)
        
        if not node.is_terminal:
            y_node = node_data[self.target_column]
            class_counts = y_node.value_counts()
            total = len(y_node)
            gini = 1.0 - sum((count/total)**2 for count in class_counts)
            node.impurity = gini
            
    def rebuildNodeChildren(self, node: TreeNode):
        """Rebuild child nodes after split parameters change"""
        if node.is_terminal:
            return
            
        node_data = self.getNodeData(node)
        if node_data is None:
            return
            
        feature = node.split_feature
        value = node.split_value
        operator = getattr(node, 'split_operator', '<=')
        
        if operator == '<=':
            left_data = node_data[node_data[feature] <= value]
            right_data = node_data[node_data[feature] > value]
        elif operator == '==':
            left_data = node_data[node_data[feature] == value]
            right_data = node_data[node_data[feature] != value]
        else:
            left_data = node_data[node_data[feature] <= value]
            right_data = node_data[node_data[feature] > value]
            
        if len(node.children) == 0:
            left_child = TreeNode(f"{node.node_id}_L", node.depth + 1)
            right_child = TreeNode(f"{node.node_id}_R", node.depth + 1)
            node.children = [left_child, right_child]
        else:
            left_child = node.children[0]
            right_child = node.children[1] if len(node.children) > 1 else None
            
        left_child.samples = len(left_data)
        if len(left_data) > 0:
            left_child.prediction = left_data[self.target_column].mode().iloc[0]
        left_child.is_terminal = True
        
        if right_child:
            right_child.samples = len(right_data)
            if len(right_data) > 0:
                right_child.prediction = right_data[self.target_column].mode().iloc[0]
            right_child.is_terminal = True
            
    def searchNodes(self):
        """Search for nodes by ID or feature"""
        search_text = self.search_edit.text().lower()
        
        self.highlightNavigatorItems(self.tree_navigator.invisibleRootItem(), search_text)
        
    def highlightNavigatorItems(self, item: QTreeWidgetItem, search_text: str):
        """Highlight navigator items that match search"""
        for i in range(item.childCount()):
            child = item.child(i)
            text = child.text(0).lower()
            
            if search_text in text:
                child.setBackground(0, QColor(255, 255, 0, 100))  # Yellow highlight
            else:
                child.setBackground(0, QColor(255, 255, 255, 0))  # No highlight
                
            self.highlightNavigatorItems(child, search_text)
            
    def goToNode(self):
        """Go to the node specified in search"""
        search_text = self.search_edit.text()
        
        node = self.findNodeById(search_text)
        if node:
            self.selectNode(node.node_id)
            
            if node.node_id in self.graphics_nodes:
                graphics_node = self.graphics_nodes[node.node_id]
                self.graphics_view.centerOn(graphics_node)
        else:
            QMessageBox.information(self, "Node Not Found", 
                                  f"No node found with ID: {search_text}")
            
    def onNavigatorItemClicked(self, item: QTreeWidgetItem, column: int):
        """Handle navigator item click"""
        node_id = item.data(0, Qt.UserRole)
        if node_id:
            self.selectNode(node_id)
            
    def findNodeById(self, node_id: str) -> Optional[TreeNode]:
        """Find a node by its ID"""
        return self._findNodeByIdRecursive(self.tree_model.root, node_id)
        
    def _findNodeByIdRecursive(self, node: TreeNode, target_id: str) -> Optional[TreeNode]:
        """Recursively search for node by ID"""
        if node.node_id == target_id:
            return node
            
        for child in node.children:
            result = self._findNodeByIdRecursive(child, target_id)
            if result:
                return result
                
        return None
        
    def expandAllNodes(self):
        """Expand all nodes in navigator"""
        self.tree_navigator.expandAll()
        
    def collapseAllNodes(self):
        """Collapse all nodes in navigator"""
        self.tree_navigator.collapseAll()
        
    def autoLayoutTree(self):
        """Auto-layout the tree visualization"""
        self.buildTreeVisualization()
        self.fitToView()
        
    def zoomIn(self):
        """Zoom in the graphics view"""
        self.graphics_view.scale(1.2, 1.2)
        
    def zoomOut(self):
        """Zoom out the graphics view"""
        self.graphics_view.scale(0.8, 0.8)
        
    def fitToView(self):
        """Fit tree to view"""
        self.graphics_view.fitInView(self.graphics_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        
    def updateTreeStatistics(self):
        """Update tree statistics display"""
        if not self.tree_model or not self.tree_model.root:
            return
            
        depth = self.calculateTreeDepth(self.tree_model.root)
        nodes = self.countNodes(self.tree_model.root)
        leaves = self.countLeaves(self.tree_model.root)
        
        self.depth_label.setText(str(depth))
        self.nodes_label.setText(str(nodes))
        self.leaves_label.setText(str(leaves))
        
        try:
            predictions = self.tree_model.predict(self.X)
            accuracy = (predictions == self.y).mean()
            self.accuracy_label.setText(f"{accuracy:.3f}")
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Error calculating accuracy: {e}")
            self.accuracy_label.setText("N/A")
            
    def calculateTreeDepth(self, node: TreeNode) -> int:
        """Calculate maximum depth of tree"""
        if node.is_terminal:
            return 0
        
        max_child_depth = 0
        for child in node.children:
            child_depth = self.calculateTreeDepth(child)
            max_child_depth = max(max_child_depth, child_depth)
            
        return max_child_depth + 1
        
    def countNodes(self, node: TreeNode) -> int:
        """Count total nodes in tree"""
        count = 1
        for child in node.children:
            count += self.countNodes(child)
        return count
        
    def countLeaves(self, node: TreeNode) -> int:
        """Count leaf nodes in tree"""
        if node.is_terminal:
            return 1
            
        count = 0
        for child in node.children:
            count += self.countLeaves(child)
        return count
        
    def saveTree(self):
        """Save the current tree state"""
        try:
            QMessageBox.information(self, "Tree Saved", "Tree has been saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Error saving tree: {str(e)}")
            
    def resetTree(self):
        """Reset tree to initial state"""
        reply = QMessageBox.question(self, "Reset Tree", 
                                   "Are you sure you want to reset the tree to its initial state?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.tree_model.root.is_terminal = True
            self.tree_model.root.children = []
            self.tree_model.root.prediction = self.y.mode().iloc[0]
            
            self.buildTreeVisualization()


class EditSplitDialog(QDialog):
    """Dialog for editing split parameters"""
    
    def __init__(self, node: TreeNode, feature_names: List[str], parent=None):
        super().__init__(parent)
        
        self.node = node
        self.feature_names = feature_names
        
        self.setWindowTitle("Edit Split")
        self.setModal(True)
        self.resize(400, 300)
        
        self.setupUI()
        self.loadCurrentSplit()
        
    def setupUI(self):
        """Setup the dialog UI"""
        layout = QVBoxLayout()
        
        form_layout = QFormLayout()
        
        self.feature_combo = QComboBox()
        self.feature_combo.addItems(self.feature_names)
        form_layout.addRow("Feature:", self.feature_combo)
        
        self.value_spin = QDoubleSpinBox()
        self.value_spin.setRange(-999999, 999999)
        self.value_spin.setDecimals(4)
        form_layout.addRow("Value:", self.value_spin)
        
        self.operator_combo = QComboBox()
        self.operator_combo.addItems(['<=', '>=', '==', '!='])
        form_layout.addRow("Operator:", self.operator_combo)
        
        layout.addLayout(form_layout)
        
        button_layout = QHBoxLayout()
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def loadCurrentSplit(self):
        """Load current split parameters"""
        if self.node.split_feature:
            index = self.feature_combo.findText(self.node.split_feature)
            if index >= 0:
                self.feature_combo.setCurrentIndex(index)
                
        if self.node.split_value is not None:
            self.value_spin.setValue(self.node.split_value)
            
        operator = getattr(self.node, 'split_operator', '<=')
        index = self.operator_combo.findText(operator)
        if index >= 0:
            self.operator_combo.setCurrentIndex(index)
            
    def getSplitParameters(self) -> Tuple[str, float, str]:
        """Get the edited split parameters"""
        return (
            self.feature_combo.currentText(),
            self.value_spin.value(),
            self.operator_combo.currentText()
        )