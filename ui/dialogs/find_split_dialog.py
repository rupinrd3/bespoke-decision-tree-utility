#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Find Split Dialog - Dedicated interface for finding optimal splits
Shows ranked split candidates and allows direct application or manual editing

"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QComboBox, QSpinBox, QProgressBar, QMessageBox,
    QHeaderView, QTabWidget, QWidget, QTextEdit, QSplitter, QFrame,
    QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, pyqtSlot
from PyQt5.QtGui import QFont, QIcon, QColor

logger = logging.getLogger(__name__)


class SplitFinderWorker(QThread):
    """Background worker for finding optimal splits"""
    
    split_found = pyqtSignal(list)  # List of split candidates
    progress_updated = pyqtSignal(int)  # Progress percentage
    error_occurred = pyqtSignal(str)  # Error message
    
    def __init__(self, model, node_id: str, parent=None):
        super().__init__(parent)
        self.model = model
        self.node_id = node_id
        self.is_cancelled = False
        
    def run(self):
        """Find optimal splits in background with proper statistics calculation"""
        try:
            self.progress_updated.emit(10)
            
            node = self.model.get_node(self.node_id)
            if not node:
                self.error_occurred.emit(f"Node {self.node_id} not found")
                return
                
            self.progress_updated.emit(30)
            
            node_data = None
            target_data = None
            
            if hasattr(node, 'data') and node.data is not None:
                node_data = node.data
            elif hasattr(self.model, '_cached_X') and hasattr(self.model, '_cached_y'):
                if hasattr(node, 'sample_indices') and node.sample_indices is not None:
                    node_data = self.model._cached_X.iloc[node.sample_indices]
                    target_data = self.model._cached_y.iloc[node.sample_indices]
                else:
                    node_data = self.model._cached_X
                    target_data = self.model._cached_y
            
            if node_data is None or node_data.empty:
                self.error_occurred.emit("No data available for split calculation")
                return
                
            self.progress_updated.emit(50)
            
            candidates = self._calculate_split_candidates(node_data, target_data, node)
                    
            self.progress_updated.emit(100)
            self.split_found.emit(candidates)
            
        except Exception as e:
            logger.error(f"Error finding splits: {e}", exc_info=True)
            self.error_occurred.emit(str(e))
            
    def _calculate_split_candidates(self, X_data, y_data, node):
        """Calculate split candidates with proper statistics"""
        candidates = []
        
        try:
            if y_data is not None:
                if hasattr(y_data, 'values'):
                    y_values = y_data.values
                else:
                    y_values = y_data
            else:
                if hasattr(node, 'target_distribution') and node.target_distribution:
                    total_samples = sum(node.target_distribution.values())
                    y_values = []
                    for class_val, count in node.target_distribution.items():
                        y_values.extend([class_val] * count)
                    y_values = np.array(y_values)
                else:
                    logger.warning("No target data available for split calculation")
                    return candidates
            
            criterion_name = "Information Gain"  # Default
            if hasattr(self.parent(), 'criterion_combo'):
                criterion_text = self.parent().criterion_combo.currentText()
                criterion_name = criterion_text
            
            if criterion_name == "Information Gain":
                baseline_metric = self._calculate_entropy(y_values)
            elif criterion_name == "Gini Gain":
                baseline_metric = self._calculate_gini_impurity(y_values)
            elif criterion_name == "Deviance Reduction":
                baseline_metric = self._calculate_deviance(y_values)
            else:
                baseline_metric = self._calculate_gini_impurity(y_values)
            
            feature_list = [(name, X_data[name]) for name in X_data.columns]
            logger.info(f"Starting parallel processing of {len(feature_list)} features using {criterion_name} criterion")
            
            candidates = Parallel(n_jobs=-2, backend='threading')(
                delayed(self._process_single_feature)(
                    feature_name, feature_data, y_values, baseline_metric, criterion_name
                ) for feature_name, feature_data in feature_list
            )
            
            candidates = [c for c in candidates if c is not None]
            logger.info(f"Parallel processing completed. Found {len(candidates)} valid split candidates from {len(feature_list)} features")
            
            candidates.sort(key=lambda x: x.get('stat_value', 0), reverse=True)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error calculating split candidates: {e}")
            return []
    
    def _process_single_feature(self, feature_name, feature_data, y_values, baseline_metric, criterion_name):
        """Process a single feature for split finding - designed for parallel execution"""
        try:
            if feature_data.isna().all():
                logger.debug(f"Skipping feature {feature_name}: all missing values")
                return None
            
            logger.debug(f"Processing feature {feature_name} ({'numeric' if pd.api.types.is_numeric_dtype(feature_data) else 'categorical'})")
            
            if pd.api.types.is_numeric_dtype(feature_data):
                result = self._find_best_numeric_split(feature_name, feature_data, y_values, baseline_metric, criterion_name)
            else:
                result = self._find_best_categorical_split(feature_name, feature_data, y_values, baseline_metric, criterion_name)
            
            if result:
                logger.debug(f"Feature {feature_name}: found split with stat_value={result.get('stat_value', 0):.4f}")
            else:
                logger.debug(f"Feature {feature_name}: no valid split found")
                
            return result
                
        except Exception as e:
            logger.error(f"Error processing feature {feature_name}: {e}")
            return None
    
    def _calculate_gini_impurity(self, y_values):
        """Calculate Gini impurity for target values"""
        if len(y_values) == 0:
            return 0
        
        _, counts = np.unique(y_values, return_counts=True)
        probabilities = counts / len(y_values)
        return 1.0 - np.sum(probabilities ** 2)
    
    def _calculate_entropy(self, y_values):
        """Calculate entropy for target values"""
        if len(y_values) == 0:
            return 0
        
        _, counts = np.unique(y_values, return_counts=True)
        probabilities = counts / len(y_values)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _calculate_deviance(self, y_values):
        """Calculate deviance for target values"""
        if len(y_values) == 0:
            return 0
        
        p = np.mean(y_values)  # proportion of positive class
        n = len(y_values)
        
        if p <= 0 or p >= 1:
            return 0
        
        return -2 * n * (p * np.log(p) + (1 - p) * np.log(1 - p))
    
    def _find_best_numeric_split(self, feature_name, feature_data, y_values, baseline_metric, criterion_name):
        """Find best numeric split for a feature"""
        try:
            valid_mask = ~feature_data.isna()
            if valid_mask.sum() < 10:  # Need minimum samples
                return None
                
            valid_feature = feature_data[valid_mask].values
            valid_target = y_values[valid_mask]
            
            unique_values = np.unique(valid_feature)
            if len(unique_values) < 2:
                return None
            
            max_unique_values = 1000  # Reasonable limit for split point evaluation
            if len(unique_values) > max_unique_values:
                logger.debug(f"Feature {feature_name} has {len(unique_values)} unique values, sampling {max_unique_values} for split evaluation")
                step = len(unique_values) // max_unique_values
                unique_values = unique_values[::step]
            
            split_points = []
            for i in range(len(unique_values) - 1):
                try:
                    val1, val2 = unique_values[i], unique_values[i + 1]
                    if abs(val1) > 1e10 or abs(val2) > 1e10:
                        midpoint = val1 + (val2 - val1) / 2
                    else:
                        midpoint = (val1 + val2) / 2
                    
                    if np.isfinite(midpoint):
                        split_points.append(midpoint)
                except (OverflowError, RuntimeWarning):
                    logger.warning(f"Skipping split point between {unique_values[i]} and {unique_values[i + 1]} due to overflow")
                    continue
            
            if not split_points:
                logger.warning(f"No valid split points found for feature {feature_name} due to overflow issues")
                return None
            
            best_gain = 0
            best_split = None
            
            for split_value in split_points:
                left_mask = valid_feature <= split_value
                right_mask = valid_feature > split_value
                
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                
                left_target = valid_target[left_mask]
                right_target = valid_target[right_mask]
                
                n_left = len(left_target)
                n_right = len(right_target)
                n_total = n_left + n_right
                
                if criterion_name == "Information Gain":
                    left_metric = self._calculate_entropy(left_target)
                    right_metric = self._calculate_entropy(right_target)
                    weighted_metric = (n_left / n_total) * left_metric + (n_right / n_total) * right_metric
                    stat_value = baseline_metric - weighted_metric  # Information Gain
                elif criterion_name == "Gini Gain":
                    left_metric = self._calculate_gini_impurity(left_target)
                    right_metric = self._calculate_gini_impurity(right_target)
                    weighted_metric = (n_left / n_total) * left_metric + (n_right / n_total) * right_metric
                    stat_value = baseline_metric - weighted_metric  # Gini Gain
                elif criterion_name == "Deviance Reduction":
                    left_deviance = self._calculate_deviance(left_target)
                    right_deviance = self._calculate_deviance(right_target)
                    stat_value = baseline_metric - (left_deviance + right_deviance)  # Deviance Reduction
                    left_metric = left_deviance
                    right_metric = right_deviance
                else:
                    left_metric = self._calculate_gini_impurity(left_target)
                    right_metric = self._calculate_gini_impurity(right_target)
                    weighted_metric = (n_left / n_total) * left_metric + (n_right / n_total) * right_metric
                    stat_value = baseline_metric - weighted_metric
                
                if stat_value > best_gain:
                    best_gain = stat_value
                    best_split = {
                        'feature': feature_name,
                        'split_type': 'numeric',
                        'split_value': split_value,
                        'stat_value': stat_value,
                        'criterion': criterion_name,
                        'left_metric': left_metric,
                        'right_metric': right_metric,
                        'left_samples': n_left,
                        'right_samples': n_right,
                        'n_samples': n_total
                    }
            
            return best_split
            
        except Exception as e:
            logger.error(f"Error finding numeric split for {feature_name}: {e}")
            return None
    
    def _find_best_categorical_split(self, feature_name, feature_data, y_values, baseline_metric, criterion_name):
        """Find best categorical split for a feature"""
        try:
            valid_mask = ~feature_data.isna()
            if valid_mask.sum() < 10:  # Need minimum samples
                return None
                
            valid_feature = feature_data[valid_mask].values
            valid_target = y_values[valid_mask]
            
            unique_categories = np.unique(valid_feature)
            if len(unique_categories) < 2:
                return None
                
            best_gain = 0
            best_split = None
            
            max_categories = min(len(unique_categories), 10)  # Limit for performance
            categories_to_try = unique_categories[:max_categories]
            
            for category in categories_to_try:
                left_mask = valid_feature == category
                right_mask = valid_feature != category
                
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                
                left_target = valid_target[left_mask]
                right_target = valid_target[right_mask]
                
                n_left = len(left_target)
                n_right = len(right_target)
                n_total = n_left + n_right
                
                if criterion_name == "Information Gain":
                    left_metric = self._calculate_entropy(left_target)
                    right_metric = self._calculate_entropy(right_target)
                    weighted_metric = (n_left / n_total) * left_metric + (n_right / n_total) * right_metric
                    stat_value = baseline_metric - weighted_metric  # Information Gain
                elif criterion_name == "Gini Gain":
                    left_metric = self._calculate_gini_impurity(left_target)
                    right_metric = self._calculate_gini_impurity(right_target)
                    weighted_metric = (n_left / n_total) * left_metric + (n_right / n_total) * right_metric
                    stat_value = baseline_metric - weighted_metric  # Gini Gain
                elif criterion_name == "Deviance Reduction":
                    left_deviance = self._calculate_deviance(left_target)
                    right_deviance = self._calculate_deviance(right_target)
                    stat_value = baseline_metric - (left_deviance + right_deviance)  # Deviance Reduction
                    left_metric = left_deviance
                    right_metric = right_deviance
                else:
                    left_metric = self._calculate_gini_impurity(left_target)
                    right_metric = self._calculate_gini_impurity(right_target)
                    weighted_metric = (n_left / n_total) * left_metric + (n_right / n_total) * right_metric
                    stat_value = baseline_metric - weighted_metric
                
                if stat_value > best_gain:
                    best_gain = stat_value
                    best_split = {
                        'feature': feature_name,
                        'split_type': 'categorical',
                        'split_value': f"= {category}",
                        'left_categories': [category],
                        'right_categories': [cat for cat in unique_categories if cat != category],
                        'stat_value': stat_value,
                        'criterion': criterion_name,
                        'left_metric': left_metric,
                        'right_metric': right_metric,
                        'left_samples': n_left,
                        'right_samples': n_right,
                        'n_samples': n_total
                    }
            
            return best_split
            
        except Exception as e:
            logger.error(f"Error finding categorical split for {feature_name}: {e}")
            return None
    
    def cancel(self):
        """Cancel the search"""
        self.is_cancelled = True


class FindSplitDialog(QDialog):
    """Dialog for finding and selecting optimal splits"""
    
    split_selected = pyqtSignal(dict)  # Selected split configuration
    
    def __init__(self, node, model, parent=None):
        super().__init__(parent)
        self.node = node
        self.model = model
        self.split_candidates = []
        self.worker = None
        
        self.setWindowTitle(f"🔍 Find Optimal Split - Node {node.node_id}")
        self.setModal(True)
        self.resize(1200, 800)
        
        self.setStyleSheet("""
            QDialog {
                background-color: #f8fafc;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                color: #1e293b;
                font-size: 13px;
            }
            QComboBox, QSpinBox {
                background-color: white;
                border: 2px solid #e2e8f0;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 13px;
                min-height: 20px;
            }
            QComboBox:focus, QSpinBox:focus {
                border-color: #3b82f6;
                outline: none;
            }
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                font-weight: 600;
                font-size: 13px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
            QPushButton[styleClass=\"secondary\"] {
                background-color: #f1f5f9;
                color: #475569;
                border: 1px solid #cbd5e1;
            }
            QPushButton[styleClass=\"secondary\"]:hover {
                background-color: #e2e8f0;
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
            QProgressBar {
                border: 2px solid #e2e8f0;
                border-radius: 6px;
                text-align: center;
                font-weight: 600;
                font-size: 12px;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 4px;
            }
        """)
        
        self.init_ui()
        self.start_split_search()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        header_layout = QHBoxLayout()
        
        node_info = QLabel(f"Finding optimal splits for Node: {self.node.node_id}")
        node_info.setFont(QFont("Arial", 12, QFont.Bold))
        header_layout.addWidget(node_info)
        
        header_layout.addStretch()
        
        self.criterion_combo = QComboBox()
        self.criterion_combo.addItems(['Information Gain', 'Gini Gain', 'Deviance Reduction'])
        self.criterion_combo.setMinimumWidth(150)  # Increase width for new labels
        
        criterion_value = getattr(self.model, 'criterion', 'gini')
        if hasattr(criterion_value, 'value'):
            criterion_str = criterion_value.value
        else:
            criterion_str = str(criterion_value)
        
        criterion_mapping = {
            'gini': 'Gini Gain',
            'entropy': 'Information Gain', 
            'log_loss': 'Deviance Reduction'
        }
        display_text = criterion_mapping.get(criterion_str, 'Information Gain')
        self.criterion_combo.setCurrentText(display_text)
        self.criterion_combo.currentTextChanged.connect(self.restart_search)
        
        header_layout.addWidget(QLabel("Criterion:"))
        header_layout.addWidget(self.criterion_combo)
        
        layout.addLayout(header_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        splitter = QSplitter(Qt.Horizontal)
        
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        
        left_layout.addWidget(QLabel("Split Candidates (ranked by information gain):"))
        
        self.splits_table = QTableWidget()
        self.splits_table.setColumnCount(3)
        self.splits_table.setHorizontalHeaderLabels([
            'Variable', 'Type', 'Stat Value'
        ])
        
        header = self.splits_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)           # Variable
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Type
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Stat Value
        
        self.splits_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.splits_table.itemSelectionChanged.connect(self.on_split_selection_changed)
        self.splits_table.itemDoubleClicked.connect(self.apply_selected_split)
        
        left_layout.addWidget(self.splits_table)
        
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        
        
        preview_group = QGroupBox("Split Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumHeight(200)
        
        preview_layout.addWidget(self.preview_text)
        right_layout.addWidget(preview_group)
        
        distribution_group = QGroupBox("Variable Distribution")
        distribution_layout = QVBoxLayout(distribution_group)
        
        self.distribution_text = QTextEdit()
        self.distribution_text.setReadOnly(True)
        self.distribution_text.setMaximumHeight(150)
        
        distribution_layout.addWidget(self.distribution_text)
        right_layout.addWidget(distribution_group)
        
        right_layout.addStretch()
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 300])
        
        layout.addWidget(splitter)
        
        button_layout = QHBoxLayout()
        
        self.apply_button = QPushButton("Apply Split")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self.apply_selected_split)
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.restart_search)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.apply_button)
        button_layout.addStretch()
        button_layout.addWidget(self.refresh_button)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
    def start_split_search(self):
        """Start the split search in background"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.splits_table.setRowCount(0)
        self.split_candidates = []
        
        self.worker = SplitFinderWorker(self.model, self.node.node_id, self)
        self.worker.split_found.connect(self.on_splits_found)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.error_occurred.connect(self.on_search_error)
        self.worker.finished.connect(lambda: self.progress_bar.setVisible(False))
        
        self.worker.start()
        
    def restart_search(self):
        """Restart the split search with new parameters"""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
            
        self.start_split_search()
        
    @pyqtSlot(list)
    def on_splits_found(self, candidates: List[Dict[str, Any]]):
        """Handle found split candidates"""
        self.split_candidates = candidates
        self.populate_splits_table()
        
    @pyqtSlot(str)
    def on_search_error(self, error_message: str):
        """Handle search errors"""
        QMessageBox.critical(self, "Split Search Error", f"Error finding splits:\n{error_message}")
        
    def populate_splits_table(self):
        """Populate the splits table with candidates"""
        self.splits_table.setRowCount(len(self.split_candidates))
        
        for row, candidate in enumerate(self.split_candidates):
            try:
                variable = candidate.get('feature', 'Unknown')
                self.splits_table.setItem(row, 0, QTableWidgetItem(str(variable)))
                
                split_type = candidate.get('split_type', 'unknown')
                if split_type == 'numeric':
                    type_display = "Numerical"
                elif split_type == 'categorical':
                    type_display = "Categorical"
                else:
                    type_display = split_type.title()
                self.splits_table.setItem(row, 1, QTableWidgetItem(type_display))
                
                stat_value = candidate.get('stat_value', 0)
                stat_item = QTableWidgetItem(f"{stat_value:.4f}")
                stat_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.splits_table.setItem(row, 2, stat_item)
                
                if len(self.split_candidates) > 1:
                    max_stat = max(c.get('stat_value', 0) for c in self.split_candidates)
                    min_stat = min(c.get('stat_value', 0) for c in self.split_candidates)
                    if max_stat > min_stat:
                        normalized_stat = (stat_value - min_stat) / (max_stat - min_stat)
                        green_component = int(255 * normalized_stat)
                        red_component = int(255 * (1 - normalized_stat))
                        color = QColor(red_component, green_component, 0, 50)
                        
                        for col in range(3):
                            if self.splits_table.item(row, col):
                                self.splits_table.item(row, col).setBackground(color)
                
            except Exception as e:
                logger.error(f"Error populating table row {row}: {e}")
                continue
                
        if len(self.split_candidates) > 0:
            self.splits_table.selectRow(0)
            
    def on_split_selection_changed(self):
        """Handle split selection change"""
        selected_rows = self.splits_table.selectionModel().selectedRows()
        
        if selected_rows:
            row = selected_rows[0].row()
            if 0 <= row < len(self.split_candidates):
                candidate = self.split_candidates[row]
                self.update_split_preview(candidate)
                self.apply_button.setEnabled(True)
            else:
                self.clear_split_preview()
                self.apply_button.setEnabled(False)
        else:
            self.clear_split_preview()
            self.apply_button.setEnabled(False)
            
    def update_split_preview(self, candidate: Dict[str, Any]):
        """Update the split preview panel"""
        try:
            
            preview_lines = []
            split_type = candidate.get('split_type', 'unknown')
            
            if split_type == 'numeric':
                split_value = candidate.get('split_value', 0)
                preview_lines.append(f"Split condition: {candidate.get('feature')} ≤ {split_value:.3f}")
                preview_lines.append("")
                preview_lines.append(f"Left branch:  {candidate.get('feature')} ≤ {split_value:.3f}")
                preview_lines.append(f"  Samples: {candidate.get('left_samples', 0)}")
                preview_lines.append(f"  Metric: {candidate.get('left_metric', 0):.4f}")
                preview_lines.append("")
                preview_lines.append(f"Right branch: {candidate.get('feature')} > {split_value:.3f}")
                preview_lines.append(f"  Samples: {candidate.get('right_samples', 0)}")
                preview_lines.append(f"  Metric: {candidate.get('right_metric', 0):.4f}")
                
            elif split_type == 'categorical':
                left_categories = candidate.get('left_categories', [])
                right_categories = candidate.get('right_categories', [])
                
                preview_lines.append(f"Split condition: {candidate.get('feature')} ∈ {left_categories}")
                preview_lines.append("")
                preview_lines.append(f"Left branch ({len(left_categories)} categories):")
                for cat in left_categories[:5]:  # Show first 5
                    preview_lines.append(f"  • {cat}")
                if len(left_categories) > 5:
                    preview_lines.append(f"  ... and {len(left_categories) - 5} more")
                preview_lines.append(f"  Samples: {candidate.get('left_samples', 0)}")
                preview_lines.append(f"  Metric: {candidate.get('left_metric', 0):.4f}")
                preview_lines.append("")
                preview_lines.append(f"Right branch ({len(right_categories)} categories):")
                for cat in right_categories[:5]:  # Show first 5
                    preview_lines.append(f"  • {cat}")
                if len(right_categories) > 5:
                    preview_lines.append(f"  ... and {len(right_categories) - 5} more")
                preview_lines.append(f"  Samples: {candidate.get('right_samples', 0)}")
                preview_lines.append(f"  Metric: {candidate.get('right_metric', 0):.4f}")
                
            self.preview_text.setPlainText("\n".join(preview_lines))
            
            self.update_distribution_info(candidate)
            
        except Exception as e:
            logger.error(f"Error updating split preview: {e}")
            self.preview_text.setPlainText(f"Error displaying preview: {e}")
            
    def update_distribution_info(self, candidate: Dict[str, Any]):
        """Update variable distribution information"""
        try:
            feature = candidate.get('feature')
            if not feature or not hasattr(self.model, '_cached_X'):
                self.distribution_text.setPlainText("Distribution information not available")
                return
                
            node_data = self.get_node_data()
            if node_data is None or feature not in node_data.columns:
                self.distribution_text.setPlainText("Feature data not available")
                return
                
            feature_data = node_data[feature].dropna()
            if len(feature_data) == 0:
                self.distribution_text.setPlainText("No valid data for this feature")
                return
                
            distribution_lines = []
            distribution_lines.append(f"Variable: {feature}")
            distribution_lines.append(f"Valid samples: {len(feature_data)}")
            
            if pd.api.types.is_numeric_dtype(feature_data):
                distribution_lines.append(f"Min: {feature_data.min():.3f}")
                distribution_lines.append(f"Max: {feature_data.max():.3f}")
                distribution_lines.append(f"Mean: {feature_data.mean():.3f}")
                distribution_lines.append(f"Std: {feature_data.std():.3f}")
                
                q25, q50, q75 = feature_data.quantile([0.25, 0.5, 0.75])
                distribution_lines.append(f"Q1: {q25:.3f}, Median: {q50:.3f}, Q3: {q75:.3f}")
            else:
                value_counts = feature_data.value_counts()
                unique_count = len(value_counts)
                distribution_lines.append(f"Unique values: {unique_count}")
                
                distribution_lines.append("\nTop categories:")
                for i, (value, count) in enumerate(value_counts.head(10).items()):
                    pct = count / len(feature_data) * 100
                    distribution_lines.append(f"  {value}: {count} ({pct:.1f}%)")
                
                if unique_count > 10:
                    distribution_lines.append(f"  ... and {unique_count - 10} more")
                    
            self.distribution_text.setPlainText("\n".join(distribution_lines))
            
        except Exception as e:
            logger.error(f"Error updating distribution info: {e}")
            self.distribution_text.setPlainText(f"Error: {e}")
            
    def clear_split_preview(self):
        """Clear the split preview panel"""
        self.preview_text.clear()
        self.distribution_text.clear()
        
    def get_node_data(self):
        """Get data for the current node"""
        try:
            if hasattr(self.model, '_cached_X') and hasattr(self.node, 'sample_indices'):
                return self.model._cached_X.iloc[self.node.sample_indices]
            elif hasattr(self.model, '_cached_X'):
                return self.model._cached_X
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting node data: {e}")
            return None
            
    def apply_selected_split(self):
        """Apply the selected split"""
        selected_rows = self.splits_table.selectionModel().selectedRows()
        
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a split to apply.")
            return
            
        row = selected_rows[0].row()
        if not (0 <= row < len(self.split_candidates)):
            QMessageBox.warning(self, "Invalid Selection", "Selected split is not valid.")
            return
            
        candidate = self.split_candidates[row]
        
        self.split_selected.emit(candidate)
        self.accept()
        
            
    def closeEvent(self, event):
        """Handle dialog close"""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        super().closeEvent(event)