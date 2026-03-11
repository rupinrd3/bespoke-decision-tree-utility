#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Streamlined Manual Split Dialog
Provides an intuitive interface for manually creating splits with optimal mouse/keyboard input
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple

from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QBrush, QKeySequence
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QPushButton, QComboBox, QDoubleSpinBox, QSlider,
    QListWidget, QListWidgetItem, QMessageBox, QFrame, QSplitter,
    QTextEdit, QShortcut, QWidget, QGridLayout, QCheckBox,
    QScrollArea, QApplication, QProgressBar, QSpacerItem, QSizePolicy
)

from models.node import TreeNode
from models.decision_tree import BespokeDecisionTree

logger = logging.getLogger(__name__)

class DataDistributionWidget(QWidget):
    """Widget to show data distribution for threshold selection"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.threshold = 0.0
        self.setMinimumHeight(80)
        
    def set_data(self, data: np.ndarray, threshold: float = None):
        """Set data for visualization"""
        self.data = data
        if threshold is not None:
            self.threshold = threshold
        self.update()
    
    def set_threshold(self, threshold: float):
        """Update threshold line"""
        self.threshold = threshold
        self.update()
    
    def paintEvent(self, event):
        """Paint the distribution histogram"""
        if self.data is None or len(self.data) == 0:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width() - 20
        height = self.height() - 20
        margin = 10
        
        try:
            hist, bin_edges = np.histogram(self.data, bins=20)
            max_count = max(hist) if len(hist) > 0 else 1
            
            bar_width = width / len(hist)
            for i, count in enumerate(hist):
                bar_height = (count / max_count) * height * 0.8
                x = margin + i * bar_width
                y = margin + height - bar_height
                
                bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
                if bin_center <= self.threshold:
                    painter.fillRect(int(x), int(y), int(bar_width-1), int(bar_height), 
                                   QColor("#3498db"))  # Blue for left
                else:
                    painter.fillRect(int(x), int(y), int(bar_width-1), int(bar_height), 
                                   QColor("#e74c3c"))  # Red for right
            
            data_min, data_max = self.data.min(), self.data.max()
            if data_max > data_min:
                threshold_x = margin + ((self.threshold - data_min) / (data_max - data_min)) * width
                painter.setPen(QPen(QColor("#2c3e50"), 2))
                painter.drawLine(int(threshold_x), margin, int(threshold_x), margin + height)
                
                painter.drawText(int(threshold_x + 5), margin + 15, f"{self.threshold:.2f}")
                
        except Exception as e:
            logger.error(f"Error painting distribution: {e}")

class CategoryGroupWidget(QWidget):
    """Widget for grouping categories with drag-and-drop"""
    
    groupingChanged = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.categories = []
        self.grouping = {}
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        instructions = QLabel("💡 Drag categories between groups to create your split:")
        instructions.setStyleSheet("color: #7f8c8d; font-style: italic; margin: 5px;")
        layout.addWidget(instructions)
        
        groups_layout = QHBoxLayout()
        
        left_group = QGroupBox("📍 Left Child")
        left_group.setStyleSheet("QGroupBox { color: #3498db; font-weight: bold; }")
        left_layout = QVBoxLayout(left_group)
        self.left_list = QListWidget()
        self.left_list.setDragDropMode(QListWidget.DragDrop)
        self.left_list.setDefaultDropAction(Qt.MoveAction)
        self.left_list.itemChanged.connect(self.on_grouping_changed)
        left_layout.addWidget(self.left_list)
        groups_layout.addWidget(left_group)
        
        right_group = QGroupBox("📍 Right Child")
        right_group.setStyleSheet("QGroupBox { color: #e74c3c; font-weight: bold; }")
        right_layout = QVBoxLayout(right_group)
        self.right_list = QListWidget()
        self.right_list.setDragDropMode(QListWidget.DragDrop)
        self.right_list.setDefaultDropAction(Qt.MoveAction)
        self.right_list.itemChanged.connect(self.on_grouping_changed)
        right_layout.addWidget(self.right_list)
        groups_layout.addWidget(right_group)
        
        layout.addLayout(groups_layout)
        
        button_layout = QHBoxLayout()
        
        self.auto_group_btn = QPushButton("🤖 Auto Group")
        self.auto_group_btn.setToolTip("Automatically group categories based on target distribution")
        self.auto_group_btn.clicked.connect(self.auto_group)
        button_layout.addWidget(self.auto_group_btn)
        
        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.clicked.connect(self.reset_grouping)
        button_layout.addWidget(self.reset_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
    
    def set_categories(self, categories: List[str], target_data: pd.Series = None):
        """Set categories to group"""
        self.categories = categories
        self.target_data = target_data
        self.reset_grouping()
    
    def reset_grouping(self):
        """Reset to default grouping"""
        self.left_list.clear()
        self.right_list.clear()
        
        mid = len(self.categories) // 2
        
        for i, cat in enumerate(self.categories):
            item = QListWidgetItem(str(cat))
            if i < mid:
                self.left_list.addItem(item)
            else:
                self.right_list.addItem(item)
        
        self.on_grouping_changed()
    
    def auto_group(self):
        """Automatically group based on target distribution"""
        if self.target_data is None:
            QMessageBox.information(self, "Auto Group", "Target data not available for auto grouping")
            return
        
        category_rates = {}
        for cat in self.categories:
            cat_data = self.target_data[self.target_data.index.isin([cat])]
            if len(cat_data) > 0:
                category_rates[cat] = cat_data.mean()
        
        sorted_cats = sorted(category_rates.keys(), key=lambda x: category_rates.get(x, 0))
        mid = len(sorted_cats) // 2
        
        self.left_list.clear()
        self.right_list.clear()
        
        for i, cat in enumerate(sorted_cats):
            item = QListWidgetItem(str(cat))
            if i < mid:
                self.left_list.addItem(item)
            else:
                self.right_list.addItem(item)
        
        self.on_grouping_changed()
    
    def on_grouping_changed(self):
        """Handle grouping change"""
        self.grouping = {}
        
        for i in range(self.left_list.count()):
            cat = self.left_list.item(i).text()
            self.grouping[cat] = 0
        
        for i in range(self.right_list.count()):
            cat = self.right_list.item(i).text()
            self.grouping[cat] = 1
        
        self.groupingChanged.emit(self.grouping)
    
    def get_grouping(self) -> Dict[str, int]:
        """Get current grouping"""
        return self.grouping.copy()

class StreamlinedManualSplitDialog(QDialog):
    """Streamlined dialog for manual split creation with optimal UX"""
    
    splitApplied = pyqtSignal(str)  # node_id
    
    def __init__(self, node: TreeNode, model: BespokeDecisionTree, parent=None):
        super().__init__(parent)
        
        self.node = node
        self.model = model
        self.data = None
        self.target_data = None
        self.selected_feature = None
        self.split_info = {}
        
        self.init_ui()
        self.setup_keyboard_shortcuts()
        self.load_data()
        
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle(f"🛠️ Manual Split: {self.node.node_id}")
        self.setMinimumSize(900, 700)
        
        layout = QVBoxLayout(self)
        
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame { 
                background-color: #f8f9fa; 
                border: 1px solid #dee2e6; 
                border-radius: 6px; 
                padding: 10px; 
            }
        """)
        header_layout = QVBoxLayout(header_frame)
        
        title = QLabel(f"🛠️ Create Manual Split for Node: {self.node.node_id}")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        header_layout.addWidget(title)
        
        info = QLabel(f"Samples: {getattr(self.node, 'sample_count', 'Unknown')} | "
                     f"Impurity: {getattr(self.node, 'impurity', 0.0):.4f}")
        header_layout.addWidget(info)
        
        layout.addWidget(header_frame)
        
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        feature_group = QGroupBox("🎯 1. Select Feature")
        feature_layout = QVBoxLayout(feature_group)
        
        self.feature_combo = QComboBox()
        self.feature_combo.setEditable(True)  # Allow typing to search
        self.feature_combo.currentTextChanged.connect(self.on_feature_changed)
        feature_layout.addWidget(self.feature_combo)
        
        self.feature_stats = QLabel("Select a feature to see statistics...")
        self.feature_stats.setStyleSheet("color: #6c757d; font-style: italic; padding: 5px;")
        feature_layout.addWidget(self.feature_stats)
        
        left_layout.addWidget(feature_group)
        
        self.split_config_group = QGroupBox("⚙️ 2. Configure Split")
        self.split_config_layout = QVBoxLayout(self.split_config_group)
        left_layout.addWidget(self.split_config_group)
        
        preview_group = QGroupBox("📊 3. Split Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_text = QTextEdit()
        self.preview_text.setMaximumHeight(120)
        self.preview_text.setReadOnly(True)
        self.preview_text.setHtml("<i>Configure split above to see preview...</i>")
        preview_layout.addWidget(self.preview_text)
        
        left_layout.addWidget(preview_group)
        
        button_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("✅ Apply Split")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self.apply_split)
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
                border: none;
            }
            QPushButton:hover { background-color: #218838; }
            QPushButton:disabled { background-color: #6c757d; }
        """)
        button_layout.addWidget(self.apply_btn)
        
        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        left_layout.addLayout(button_layout)
        
        splitter.addWidget(left_panel)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        viz_group = QGroupBox("📈 Data Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        self.distribution_widget = DataDistributionWidget()
        viz_layout.addWidget(self.distribution_widget)
        
        self.grouping_widget = CategoryGroupWidget()
        self.grouping_widget.groupingChanged.connect(self.on_grouping_changed)
        self.grouping_widget.setVisible(False)
        viz_layout.addWidget(self.grouping_widget)
        
        right_layout.addWidget(viz_group)
        
        self.status_label = QLabel("Ready to create manual split...")
        self.status_label.setStyleSheet("color: #6c757d; font-style: italic;")
        right_layout.addWidget(self.status_label)
        
        splitter.addWidget(right_panel)
        
        splitter.setSizes([400, 500])
    
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts"""
        apply_shortcut = QShortcut(QKeySequence.InsertParagraphSeparator, self)
        apply_shortcut.activated.connect(self.try_apply_split)
        
        cancel_shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        cancel_shortcut.activated.connect(self.reject)
        
        up_shortcut = QShortcut(QKeySequence(Qt.Key_Up), self)
        up_shortcut.activated.connect(self.threshold_up)
        
        down_shortcut = QShortcut(QKeySequence(Qt.Key_Down), self)
        down_shortcut.activated.connect(self.threshold_down)
    
    def load_data(self):
        """Load data for the node"""
        try:
            if not hasattr(self.model, '_cached_X') or self.model._cached_X is None:
                QMessageBox.warning(self, "No Data", "No training data available. Please train the model first.")
                return
            
            self.data = self.model._cached_X
            self.target_data = self.model._cached_y
            
            features = list(self.data.columns)
            self.feature_combo.addItems(features)
            
            if features:
                self.feature_combo.setCurrentIndex(0)
                
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            QMessageBox.critical(self, "Data Error", f"Could not load data: {str(e)}")
    
    def on_feature_changed(self):
        """Handle feature selection change"""
        feature = self.feature_combo.currentText()
        if not feature or feature not in self.data.columns:
            return
        
        self.selected_feature = feature
        feature_data = self.data[feature]
        
        try:
            unique_count = feature_data.nunique()
            missing_count = feature_data.isnull().sum()
            
            if pd.api.types.is_numeric_dtype(feature_data):
                stats_text = f"📊 Numeric | Unique: {unique_count} | Missing: {missing_count} | Range: {feature_data.min():.2f} - {feature_data.max():.2f}"
                self.setup_numeric_split(feature_data)
            else:
                stats_text = f"📋 Categorical | Categories: {unique_count} | Missing: {missing_count}"
                self.setup_categorical_split(feature_data)
            
            self.feature_stats.setText(stats_text)
            
        except Exception as e:
            logger.error(f"Error analyzing feature {feature}: {e}")
            self.feature_stats.setText("❌ Error analyzing feature")
    
    def setup_numeric_split(self, feature_data: pd.Series):
        """Setup UI for numeric split"""
        self.clear_split_config()
        
        self.distribution_widget.setVisible(True)
        self.grouping_widget.setVisible(False)
        
        threshold_layout = QHBoxLayout()
        
        data_min, data_max = feature_data.min(), feature_data.max()
        data_range = data_max - data_min
        
        threshold_label = QLabel("Threshold:")
        threshold_layout.addWidget(threshold_label)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(1000)
        self.threshold_slider.setValue(500)  # Start at median
        self.threshold_slider.valueChanged.connect(self.on_threshold_slider_changed)
        threshold_layout.addWidget(self.threshold_slider)
        
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(data_min, data_max)
        self.threshold_input.setDecimals(4)
        self.threshold_input.setValue((data_min + data_max) / 2)
        self.threshold_input.valueChanged.connect(self.on_threshold_input_changed)
        threshold_layout.addWidget(self.threshold_input)
        
        self.split_config_layout.addLayout(threshold_layout)
        
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Quick sets:"))
        
        for pct, label in [(25, "Q1"), (50, "Median"), (75, "Q3")]:
            btn = QPushButton(label)
            value = feature_data.quantile(pct/100)
            btn.clicked.connect(lambda checked, v=value: self.set_threshold(v))
            preset_layout.addWidget(btn)
        
        self.split_config_layout.addLayout(preset_layout)
        
        clean_data = feature_data.dropna()
        self.distribution_widget.set_data(clean_data.values, self.threshold_input.value())
        
        self.apply_btn.setEnabled(True)
        self.update_split_preview()
    
    def setup_categorical_split(self, feature_data: pd.Series):
        """Setup UI for categorical split"""
        self.clear_split_config()
        
        self.distribution_widget.setVisible(False)
        self.grouping_widget.setVisible(True)
        
        categories = feature_data.dropna().unique().tolist()
        
        if len(categories) < 2:
            self.status_label.setText("❌ Feature has less than 2 categories")
            self.apply_btn.setEnabled(False)
            return
        
        if len(categories) > 20:
            self.status_label.setText("⚠️ Feature has many categories - consider grouping similar ones")
        
        self.grouping_widget.set_categories(categories, self.target_data)
        
        self.apply_btn.setEnabled(True)
        
    def clear_split_config(self):
        """Clear split configuration layout"""
        while self.split_config_layout.count():
            child = self.split_config_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def on_threshold_slider_changed(self):
        """Handle threshold slider change"""
        if not hasattr(self, 'threshold_input'):
            return
        
        feature_data = self.data[self.selected_feature]
        data_min, data_max = feature_data.min(), feature_data.max()
        
        slider_pct = self.threshold_slider.value() / 1000.0
        threshold_value = data_min + slider_pct * (data_max - data_min)
        
        self.threshold_input.blockSignals(True)
        self.threshold_input.setValue(threshold_value)
        self.threshold_input.blockSignals(False)
        
        clean_data = feature_data.dropna()
        self.distribution_widget.set_threshold(threshold_value)
        
        self.update_split_preview()
    
    def on_threshold_input_changed(self):
        """Handle threshold input change"""
        if not hasattr(self, 'threshold_slider'):
            return
        
        feature_data = self.data[self.selected_feature]
        data_min, data_max = feature_data.min(), feature_data.max()
        
        threshold_value = self.threshold_input.value()
        slider_pct = (threshold_value - data_min) / (data_max - data_min) if data_max > data_min else 0.5
        slider_value = int(slider_pct * 1000)
        
        self.threshold_slider.blockSignals(True)
        self.threshold_slider.setValue(slider_value)
        self.threshold_slider.blockSignals(False)
        
        clean_data = feature_data.dropna()
        self.distribution_widget.set_threshold(threshold_value)
        
        self.update_split_preview()
    
    def on_grouping_changed(self, grouping: Dict[str, int]):
        """Handle categorical grouping change"""
        self.update_split_preview()
    
    def set_threshold(self, value: float):
        """Set threshold value"""
        if hasattr(self, 'threshold_input'):
            self.threshold_input.setValue(value)
    
    def threshold_up(self):
        """Increase threshold (keyboard shortcut)"""
        if hasattr(self, 'threshold_input'):
            current = self.threshold_input.value()
            step = self.threshold_input.singleStep()
            self.threshold_input.setValue(current + step)
    
    def threshold_down(self):
        """Decrease threshold (keyboard shortcut)"""
        if hasattr(self, 'threshold_input'):
            current = self.threshold_input.value()
            step = self.threshold_input.singleStep()
            self.threshold_input.setValue(current - step)
    
    def update_split_preview(self):
        """Update split preview"""
        if not self.selected_feature:
            return
        
        try:
            feature_data = self.data[self.selected_feature]
            
            if pd.api.types.is_numeric_dtype(feature_data):
                threshold = self.threshold_input.value()
                
                left_mask = feature_data <= threshold
                right_mask = feature_data > threshold
                
                left_count = left_mask.sum()
                right_count = right_mask.sum()
                
                left_dist = "N/A"
                right_dist = "N/A"
                
                if self.target_data is not None:
                    try:
                        left_targets = self.target_data[left_mask]
                        right_targets = self.target_data[right_mask]
                        
                        if len(left_targets) > 0:
                            left_dist = left_targets.value_counts().to_dict()
                        if len(right_targets) > 0:
                            right_dist = right_targets.value_counts().to_dict()
                    except:
                        pass
                
                preview_html = f"""
                <h3>🔢 Numeric Split Preview</h3>
                <p><b>Feature:</b> {self.selected_feature}</p>
                <p><b>Condition:</b> {self.selected_feature} ≤ {threshold:.4f}</p>
                
                <table border="1" style="border-collapse: collapse; width: 100%;">
                <tr style="background-color: #f8f9fa;">
                    <th>Child</th><th>Condition</th><th>Samples</th><th>Distribution</th>
                </tr>
                <tr>
                    <td><b>Left</b></td>
                    <td>≤ {threshold:.4f}</td>
                    <td>{left_count}</td>
                    <td>{left_dist}</td>
                </tr>
                <tr>
                    <td><b>Right</b></td>
                    <td>> {threshold:.4f}</td>
                    <td>{right_count}</td>
                    <td>{right_dist}</td>
                </tr>
                </table>
                """
                
                self.split_info = {
                    'feature': self.selected_feature,
                    'split_type': 'numeric',
                    'split_value': threshold,
                    'threshold': threshold,
                    'split_operator': '<='
                }
                
            else:
                grouping = self.grouping_widget.get_grouping()
                
                if not grouping:
                    preview_html = "<i>Configure grouping above...</i>"
                    self.split_info = {}
                else:
                    left_cats = [cat for cat, group in grouping.items() if group == 0]
                    right_cats = [cat for cat, group in grouping.items() if group == 1]
                    
                    left_mask = feature_data.isin(left_cats)
                    right_mask = feature_data.isin(right_cats)
                    
                    left_count = left_mask.sum()
                    right_count = right_mask.sum()
                    
                    preview_html = f"""
                    <h3>📋 Categorical Split Preview</h3>
                    <p><b>Feature:</b> {self.selected_feature}</p>
                    
                    <table border="1" style="border-collapse: collapse; width: 100%;">
                    <tr style="background-color: #f8f9fa;">
                        <th>Child</th><th>Categories</th><th>Samples</th>
                    </tr>
                    <tr>
                        <td><b>Left</b></td>
                        <td>{', '.join(left_cats[:3])}{'...' if len(left_cats) > 3 else ''}</td>
                        <td>{left_count}</td>
                    </tr>
                    <tr>
                        <td><b>Right</b></td>
                        <td>{', '.join(right_cats[:3])}{'...' if len(right_cats) > 3 else ''}</td>
                        <td>{right_count}</td>
                    </tr>
                    </table>
                    """
                    
                    self.split_info = {
                        'feature': self.selected_feature,
                        'split_type': 'categorical',
                        'split_categories': grouping
                    }
            
            self.preview_text.setHtml(preview_html)
            
        except Exception as e:
            logger.error(f"Error updating preview: {e}")
            self.preview_text.setHtml(f"<i>Error creating preview: {str(e)}</i>")
    
    def try_apply_split(self):
        """Try to apply split (keyboard shortcut)"""
        if self.apply_btn.isEnabled():
            self.apply_split()
    
    def apply_split(self):
        """Apply the configured split"""
        if not self.split_info:
            QMessageBox.warning(self, "No Split", "Please configure a split first.")
            return
        
        try:
            logger.info(f"Applying manual split: {self.split_info}")
            
            success = self.model.apply_manual_split(self.node.node_id, self.split_info)
            
            if success:
                logger.info(f"Manual split applied successfully")
                
                self.splitApplied.emit(self.node.node_id)
                
                QMessageBox.information(
                    self, "Split Applied!",
                    f"Manual split applied successfully!\n\n"
                    f"Feature: {self.split_info.get('feature', 'Unknown')}\n"
                    f"Type: {self.split_info.get('split_type', 'Unknown')}\n\n"
                    "Child nodes have been created."
                )
                
                self.accept()
            else:
                QMessageBox.warning(self, "Split Failed", "Could not apply the split. Check the logs for details.")
                
        except Exception as e:
            logger.error(f"Error applying manual split: {e}", exc_info=True)
            QMessageBox.critical(self, "Split Error", f"Error applying split:\n{str(e)}")

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    
    from models.node import TreeNode
    node = TreeNode("test_node")
    dialog = StreamlinedManualSplitDialog(node, None)
    dialog.show()
    
    sys.exit(app.exec_())