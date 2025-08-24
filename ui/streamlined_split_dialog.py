#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlined Split Dialog Module
Provides a simplified, intuitive interface for applying splits to decision tree nodes
"""

import logging
from typing import Dict, List, Optional, Any

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QColor
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QTabWidget, QWidget, QTextEdit, QProgressBar,
    QFrame, QSplitter
)

from models.node import TreeNode
from models.decision_tree import BespokeDecisionTree

logger = logging.getLogger(__name__)

class StreamlinedSplitDialog(QDialog):
    """Streamlined dialog for finding and applying splits to tree nodes"""
    
    splitApplied = pyqtSignal(str)  # node_id
    
    def __init__(self, node: TreeNode, model: BespokeDecisionTree, parent=None):
        """
        Initialize the streamlined split dialog
        
        Args:
            node: TreeNode to split
            model: Decision tree model
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.node = node
        self.model = model
        self.potential_splits = []
        self.selected_split = None
        
        self.init_ui()
        self.find_splits_automatically()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle(f"Split Node: {self.node.node_id}")
        self.setMinimumSize(800, 600)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #f0f8ff; border: 1px solid #ccc; border-radius: 4px; padding: 8px;")
        header_layout = QVBoxLayout(header_frame)
        
        title_label = QLabel(f"🌳 Split Node: {self.node.node_id}")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)
        
        info_label = QLabel(f"Samples: {getattr(self.node, 'sample_count', 'Unknown')} | "
                           f"Impurity: {getattr(self.node, 'impurity', 0.0):.4f} | "
                           f"Current: {'Terminal Node' if self.node.is_terminal else 'Internal Node'}")
        header_layout.addWidget(info_label)
        
        layout.addWidget(header_frame)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        content_splitter = QSplitter(Qt.Vertical)
        layout.addWidget(content_splitter)
        
        splits_group = QGroupBox("🎯 Optimal Split Options")
        splits_layout = QVBoxLayout(splits_group)
        
        instructions = QLabel(
            "💡 <b>How to Split:</b> Review the splits below ranked by quality. "
            "Higher 'Gini Decrease' means better separation. Click 'Apply Split' to create child nodes."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #2c3e50; margin: 5px; padding: 5px;")
        splits_layout.addWidget(instructions)
        
        self.split_table = QTableWidget(0, 5)
        self.split_table.setHorizontalHeaderLabels([
            "Rank", "Feature", "Split Condition", "Quality Score", "Sample Split"
        ])
        
        header = self.split_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Rank
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Feature
        header.setSectionResizeMode(2, QHeaderView.Stretch)           # Split Condition
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Quality Score
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Sample Split
        
        self.split_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.split_table.setAlternatingRowColors(True)
        self.split_table.itemSelectionChanged.connect(self.on_split_selection_changed)
        
        splits_layout.addWidget(self.split_table)
        
        content_splitter.addWidget(splits_group)
        
        preview_group = QGroupBox("📊 Split Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_text = QTextEdit()
        self.preview_text.setMaximumHeight(150)
        self.preview_text.setReadOnly(True)
        self.preview_text.setHtml("<i>Select a split above to see preview...</i>")
        preview_layout.addWidget(self.preview_text)
        
        content_splitter.addWidget(preview_group)
        
        content_splitter.setSizes([400, 150])
        
        button_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("🔄 Refresh Splits")
        self.refresh_btn.clicked.connect(self.find_splits_automatically)
        button_layout.addWidget(self.refresh_btn)
        
        button_layout.addStretch()
        
        self.apply_btn = QPushButton("✅ Apply Selected Split")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.apply_selected_split)
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        button_layout.addWidget(self.apply_btn)
        
        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        self.status_label = QLabel("Ready to find splits...")
        self.status_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        layout.addWidget(self.status_label)
    
    def find_splits_automatically(self):
        """Find optimal splits for the node"""
        self.status_label.setText("🔍 Finding optimal splits...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        
        try:
            self.potential_splits = self.model.find_split_for_node(self.node.node_id)
            
            if self.potential_splits:
                self.populate_split_table()
                self.status_label.setText(f"✅ Found {len(self.potential_splits)} potential splits")
            else:
                self.status_label.setText("⚠️ No optimal splits found for this node")
                QMessageBox.information(
                    self, "No Splits Found",
                    f"No optimal splits were found for node {self.node.node_id}. "
                    "This could mean:\n"
                    "• The node is already pure (all samples have same class)\n"
                    "• Minimum split requirements are not met\n"
                    "• The data quality is insufficient for splitting"
                )
            
        except Exception as e:
            logger.error(f"Error finding splits: {e}", exc_info=True)
            self.status_label.setText("❌ Error finding splits")
            QMessageBox.critical(
                self, "Split Finding Error",
                f"Could not find splits for node {self.node.node_id}:\n{str(e)}"
            )
        
        finally:
            self.progress_bar.setVisible(False)
    
    def populate_split_table(self):
        """Populate the split table with found splits"""
        self.split_table.setRowCount(len(self.potential_splits))
        
        for i, split in enumerate(self.potential_splits):
            rank_item = QTableWidgetItem(f"#{i+1}")
            rank_item.setTextAlignment(Qt.AlignCenter)
            if i == 0:  # Best split
                rank_item.setBackground(QColor("#d4edda"))
            self.split_table.setItem(i, 0, rank_item)
            
            feature_item = QTableWidgetItem(split.get('feature', 'Unknown'))
            feature_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.split_table.setItem(i, 1, feature_item)
            
            condition = split.get('split_desc', 'Unknown condition')
            condition_item = QTableWidgetItem(condition)
            condition_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.split_table.setItem(i, 2, condition_item)
            
            gain = split.get('gain', 0.0)
            quality_item = QTableWidgetItem(f"{gain:.4f}")
            quality_item.setTextAlignment(Qt.AlignCenter)
            if gain > 0.1:
                quality_item.setBackground(QColor("#d4edda"))  # Green for excellent
            elif gain > 0.05:
                quality_item.setBackground(QColor("#fff3cd"))  # Yellow for good
            self.split_table.setItem(i, 3, quality_item)
            
            split_info_dict = split.get('split_info', {})
            left_samples = split_info_dict.get('left_samples', 0)
            right_samples = split_info_dict.get('right_samples', 0)
            sample_info = f"{left_samples} | {right_samples}"
            sample_item = QTableWidgetItem(sample_info)
            sample_item.setTextAlignment(Qt.AlignCenter)
            self.split_table.setItem(i, 4, sample_item)
            
            self.split_table.setRowData(i, split)
    
    def on_split_selection_changed(self):
        """Handle split selection change"""
        current_row = self.split_table.currentRow()
        
        if current_row >= 0 and current_row < len(self.potential_splits):
            self.selected_split = self.potential_splits[current_row]
            self.apply_btn.setEnabled(True)
            self.update_split_preview()
        else:
            self.selected_split = None
            self.apply_btn.setEnabled(False)
            self.preview_text.setHtml("<i>Select a split above to see preview...</i>")
    
    def update_split_preview(self):
        """Update the split preview section"""
        if not self.selected_split:
            return
        
        split_info = self.selected_split.get('split_info', {})
        
        html = f"""
        <h3>🎯 Split Preview</h3>
        <p><b>Feature:</b> {self.selected_split.get('feature', 'Unknown')}</p>
        <p><b>Condition:</b> {self.selected_split.get('split_desc', 'Unknown')}</p>
        <p><b>Quality Score:</b> {self.selected_split.get('gain', 0.0):.4f}</p>
        
        <h4>📊 Resulting Child Nodes:</h4>
        <table border="1" cellpadding="4" style="border-collapse: collapse; width: 100%;">
        <tr style="background-color: #f8f9fa;">
            <th>Child</th>
            <th>Samples</th>
            <th>Class Distribution</th>
        </tr>
        """
        
        left_samples = split_info.get('left_samples', 0)
        left_dist = split_info.get('left_class_distribution', {})
        left_dist_str = ", ".join([f"Class {k}: {v}" for k, v in left_dist.items()])
        
        html += f"""
        <tr>
            <td><b>Left Child</b></td>
            <td>{left_samples}</td>
            <td>{left_dist_str}</td>
        </tr>
        """
        
        right_samples = split_info.get('right_samples', 0)
        right_dist = split_info.get('right_class_distribution', {})
        right_dist_str = ", ".join([f"Class {k}: {v}" for k, v in right_dist.items()])
        
        html += f"""
        <tr>
            <td><b>Right Child</b></td>
            <td>{right_samples}</td>
            <td>{right_dist_str}</td>
        </tr>
        </table>
        """
        
        self.preview_text.setHtml(html)
    
    def apply_selected_split(self):
        """Apply the selected split to the node"""
        if not self.selected_split:
            QMessageBox.warning(self, "No Split Selected", "Please select a split to apply.")
            return
        
        try:
            split_info = self.selected_split.get('split_info', {})
            
            logger.info(f"StreamlinedSplitDialog: Applying split to node {self.node.node_id}")
            
            success = self.model.apply_manual_split(self.node.node_id, split_info)
            
            if success:
                logger.info(f"StreamlinedSplitDialog: Split applied successfully")
                
                self.splitApplied.emit(self.node.node_id)
                
                QMessageBox.information(
                    self, "Split Applied Successfully!",
                    f"The split has been applied to node {self.node.node_id}.\n\n"
                    f"Feature: {self.selected_split.get('feature', 'Unknown')}\n"
                    f"Condition: {self.selected_split.get('split_desc', 'Unknown')}\n\n"
                    "Child nodes have been created and the tree has been updated."
                )
                
                self.accept()
                
            else:
                logger.warning(f"StreamlinedSplitDialog: Failed to apply split")
                QMessageBox.warning(
                    self, "Split Failed",
                    f"Could not apply the split to node {self.node.node_id}. "
                    "Please check the logs for more details."
                )
        
        except Exception as e:
            logger.error(f"StreamlinedSplitDialog: Error applying split: {e}", exc_info=True)
            QMessageBox.critical(
                self, "Split Error",
                f"An error occurred while applying the split:\n{str(e)}"
            )

def setRowData(self, row, data):
    """Set data for a table row"""
    if not hasattr(self, '_row_data'):
        self._row_data = {}
    self._row_data[row] = data

def getRowData(self, row):
    """Get data for a table row"""
    if not hasattr(self, '_row_data'):
        return None
    return self._row_data.get(row)

QTableWidget.setRowData = setRowData
QTableWidget.getRowData = getRowData