#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Split Configuration Dialog for Bespoke Utility
Simplified dialog focused purely on UI - business logic extracted to separate components
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QDoubleValidator
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QSpinBox,
    QCheckBox, QTextEdit, QTabWidget, QWidget, QListWidget,
    QSplitter, QProgressBar, QMessageBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QFrame, QSizePolicy
)

from models.node import TreeNode
from models.split_configuration import SplitConfiguration, SplitConfigurationFactory
from ui.components.bin_manager import BinManager
from business.split_statistics_calculator import SplitStatisticsCalculator
from business.split_preview_generator import SplitPreviewGenerator, SplitPreview
from utils.performance_optimizer import (optimize_bin_manager, get_split_quality_cache, 
                                       performance_timed, PerformanceTimer)

logger = logging.getLogger(__name__)


class SplitConfigurationDialog(QDialog):
    """Simplified split configuration dialog with extracted business logic"""
    
    splitConfigured = pyqtSignal(object, object)  # node, split_configuration
    
    def __init__(self, node: TreeNode, dataset: pd.DataFrame, 
                 target_column: str, parent=None):
        super().__init__(parent)
        
        self.node = node
        self.dataset = dataset
        self.target_column = target_column
        self.current_feature = None
        
        base_bin_manager = BinManager()
        self.bin_manager = optimize_bin_manager(base_bin_manager)
        self.stats_calculator = SplitStatisticsCalculator()
        self.preview_generator = SplitPreviewGenerator(self.stats_calculator)
        self.quality_cache = get_split_quality_cache()
        
        self.current_preview: Optional[SplitPreview] = None
        self._update_timer: Optional[QTimer] = None
        
        self.setWindowTitle(f"Configure Split for Node {node.node_id}")
        self.setMinimumSize(800, 600)
        self.resize(900, 700)
        
        self.setup_ui()
        self.connect_signals()
        self.load_initial_configuration()
        
    def setup_ui(self):
        """Set up the user interface"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        
        self.create_header(main_layout)
        
        content_splitter = QSplitter(Qt.Horizontal)
        
        config_widget = self.create_configuration_panel()
        content_splitter.addWidget(config_widget)
        
        preview_widget = self.create_preview_panel()
        content_splitter.addWidget(preview_widget)
        
        content_splitter.setSizes([400, 500])
        main_layout.addWidget(content_splitter)
        
        self.create_button_bar(main_layout)
        
        self.setLayout(main_layout)
        
    def create_header(self, layout: QVBoxLayout):
        """Create header with node information"""
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        
        header_layout = QHBoxLayout(header_frame)
        
        title = QLabel(f"🔧 Configure Split: {self.node.node_id}")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        title.setFont(font)
        header_layout.addWidget(title)
        
        info_text = f"📊 {len(self.dataset)} samples"
        if self.target_column in self.dataset.columns:
            unique_targets = self.dataset[self.target_column].nunique()
            info_text += f" | {unique_targets} target classes"
        info_label = QLabel(info_text)
        info_label.setStyleSheet("color: #6c757d;")
        header_layout.addWidget(info_label)
        
        header_layout.addStretch()
        layout.addWidget(header_frame)
        
    def create_configuration_panel(self) -> QWidget:
        """Create the configuration panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        feature_group = QGroupBox("Feature Selection")
        feature_layout = QFormLayout()
        
        self.feature_combo = QComboBox()
        features = [col for col in self.dataset.columns if col != self.target_column]
        self.feature_combo.addItems(features)
        self.feature_combo.currentTextChanged.connect(self.on_feature_changed)
        feature_layout.addRow("Feature:", self.feature_combo)
        
        self.feature_info_label = QLabel("Select a feature to see details")
        self.feature_info_label.setStyleSheet("color: #6c757d; font-style: italic;")
        feature_layout.addRow("Info:", self.feature_info_label)
        
        feature_group.setLayout(feature_layout)
        layout.addWidget(feature_group)
        
        bin_group = QGroupBox("Bin Configuration")
        bin_layout = QVBoxLayout()
        
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Number of bins:"))
        self.num_bins_spin = QSpinBox()
        self.num_bins_spin.setMinimum(2)
        self.num_bins_spin.setMaximum(20)
        self.num_bins_spin.setValue(2)
        self.num_bins_spin.valueChanged.connect(self.on_num_bins_changed)
        controls_layout.addWidget(self.num_bins_spin)
        
        controls_layout.addStretch()
        
        equal_width_btn = QPushButton("Equal Width")
        equal_width_btn.clicked.connect(self.create_equal_width_bins)
        controls_layout.addWidget(equal_width_btn)
        
        equal_freq_btn = QPushButton("Equal Frequency")
        equal_freq_btn.clicked.connect(self.create_equal_frequency_bins)
        controls_layout.addWidget(equal_freq_btn)
        
        quantile_btn = QPushButton("Quantiles")
        quantile_btn.clicked.connect(self.create_quantile_bins)
        controls_layout.addWidget(quantile_btn)
        
        bin_layout.addLayout(controls_layout)
        
        self.bin_list_widget = QListWidget()
        self.bin_list_widget.setMaximumHeight(200)
        bin_layout.addWidget(self.bin_list_widget)
        
        manual_layout = QHBoxLayout()
        
        add_btn = QPushButton("➕ Add Bin")
        add_btn.clicked.connect(self.add_manual_bin)
        manual_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("✖️ Remove Selected")
        remove_btn.clicked.connect(self.remove_selected_bin)
        manual_layout.addWidget(remove_btn)
        
        merge_btn = QPushButton("🔗 Merge Small")
        merge_btn.clicked.connect(self.merge_small_bins)
        manual_layout.addWidget(merge_btn)
        
        manual_layout.addStretch()
        bin_layout.addLayout(manual_layout)
        
        bin_group.setLayout(bin_layout)
        layout.addWidget(bin_group)
        
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QFormLayout()
        
        self.criterion_combo = QComboBox()
        self.criterion_combo.addItems(["gini", "entropy", "log_loss"])
        self.criterion_combo.currentTextChanged.connect(self.on_criterion_changed)
        advanced_layout.addRow("Impurity Measure:", self.criterion_combo)
        
        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setMinimum(1)
        self.min_samples_spin.setMaximum(100)
        self.min_samples_spin.setValue(5)
        advanced_layout.addRow("Min Samples per Bin:", self.min_samples_spin)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        layout.addStretch()
        return widget
        
    def create_preview_panel(self) -> QWidget:
        """Create the preview panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        metrics_group = QGroupBox("Split Quality Metrics")
        metrics_layout = QFormLayout()
        
        self.impurity_before_label = QLabel("N/A")
        self.impurity_after_label = QLabel("N/A")
        self.impurity_decrease_label = QLabel("N/A")
        self.gain_ratio_label = QLabel("N/A")
        self.quality_score_label = QLabel("N/A")
        
        metrics_layout.addRow("Impurity Before:", self.impurity_before_label)
        metrics_layout.addRow("Impurity After:", self.impurity_after_label)
        metrics_layout.addRow("Impurity Decrease:", self.impurity_decrease_label)
        metrics_layout.addRow("Gain Ratio:", self.gain_ratio_label)
        metrics_layout.addRow("Quality Score:", self.quality_score_label)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        recommendations_group = QGroupBox("Recommendations")
        self.recommendations_text = QTextEdit()
        self.recommendations_text.setMaximumHeight(100)
        self.recommendations_text.setReadOnly(True)
        recommendations_layout = QVBoxLayout()
        recommendations_layout.addWidget(self.recommendations_text)
        recommendations_group.setLayout(recommendations_layout)
        layout.addWidget(recommendations_group)
        
        warnings_group = QGroupBox("Warnings")
        self.warnings_text = QTextEdit()
        self.warnings_text.setMaximumHeight(100)
        self.warnings_text.setReadOnly(True)
        warnings_layout = QVBoxLayout()
        warnings_layout.addWidget(self.warnings_text)
        warnings_group.setLayout(warnings_layout)
        layout.addWidget(warnings_group)
        
        details_group = QGroupBox("Bin Details")
        details_layout = QVBoxLayout()
        
        self.details_table = QTableWidget()
        self.details_table.setColumnCount(6)
        self.details_table.setHorizontalHeaderLabels([
            "Bin", "Range", "Samples", "% of Total", "Most Common Class", "Purity"
        ])
        
        header = self.details_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.details_table.setAlternatingRowColors(True)
        self.details_table.setMaximumHeight(200)
        
        details_layout.addWidget(self.details_table)
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)
        
        return widget
        
    def create_button_bar(self, layout: QVBoxLayout):
        """Create the button bar"""
        button_layout = QHBoxLayout()
        
        reset_btn = QPushButton("🔄 Reset")
        reset_btn.clicked.connect(self.reset_configuration)
        button_layout.addWidget(reset_btn)
        
        button_layout.addStretch()
        
        self.apply_btn = QPushButton("✅ Apply Split")
        self.apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 6px;
                border: none;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        self.apply_btn.clicked.connect(self.apply_configuration)
        button_layout.addWidget(self.apply_btn)
        
        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
    def connect_signals(self):
        """Connect signals from business logic components"""
        self.bin_manager.binsChanged.connect(self.update_bin_display)
        self.bin_manager.binsChanged.connect(self.schedule_preview_update)
        self.bin_manager.validationError.connect(self.show_validation_error)
        
    def load_initial_configuration(self):
        """Load initial configuration"""
        if self.feature_combo.count() > 0:
            if hasattr(self.node, 'split_feature') and self.node.split_feature:
                feature = self.node.split_feature
                if feature in [self.feature_combo.itemText(i) for i in range(self.feature_combo.count())]:
                    self.feature_combo.setCurrentText(feature)
                    
            self.on_feature_changed(self.feature_combo.currentText())
            
    def on_feature_changed(self, feature_name: str):
        """Handle feature selection change"""
        if not feature_name or feature_name not in self.dataset.columns:
            return
            
        self.current_feature = feature_name
        feature_data = self.dataset[feature_name].dropna()
        
        if pd.api.types.is_numeric_dtype(self.dataset[feature_name]):
            feature_type = "Numeric"
            self.num_bins_spin.setMaximum(20)
            if len(feature_data) > 0:
                info_text = f"{feature_type} | Range: {feature_data.min():.3f} to {feature_data.max():.3f} | {feature_data.nunique()} unique values"
            else:
                info_text = f"{feature_type} | No valid data"
            bin_manager_type = 'numeric'
        else:
            feature_type = "Categorical"
            unique_count = feature_data.nunique()
            self.num_bins_spin.setMaximum(min(10, unique_count))
            if len(feature_data) > 0:
                mode_val = feature_data.mode().iloc[0] if len(feature_data.mode()) > 0 else 'N/A'
                info_text = f"{feature_type} | {unique_count} categories | Most common: {mode_val}"
            else:
                info_text = f"{feature_type} | No valid data"
            bin_manager_type = 'categorical'
            
        self.feature_info_label.setText(info_text)
        
        self.bin_manager.set_feature_data(feature_data, bin_manager_type)
        
        if bin_manager_type == 'numeric':
            self.create_equal_width_bins()
        else:
            self.create_categorical_bins()
            
    def on_num_bins_changed(self, num_bins: int):
        """Handle number of bins change"""
        current_bins = self.bin_manager.get_bin_count()
        
        if num_bins != current_bins:
            if self.current_feature and pd.api.types.is_numeric_dtype(self.dataset[self.current_feature]):
                self.bin_manager.create_equal_width_bins(num_bins)
            else:
                self.create_categorical_bins()
                
    def on_criterion_changed(self, criterion: str):
        """Handle impurity criterion change"""
        self.stats_calculator.set_criterion(criterion)
        self.schedule_preview_update()
        
    def create_equal_width_bins(self):
        """Create equal width bins"""
        num_bins = self.num_bins_spin.value()
        if self.bin_manager.create_equal_width_bins(num_bins):
            self.num_bins_spin.setValue(self.bin_manager.get_bin_count())
            
    def create_equal_frequency_bins(self):
        """Create equal frequency bins"""
        num_bins = self.num_bins_spin.value()
        if self.bin_manager.create_equal_frequency_bins(num_bins):
            self.num_bins_spin.setValue(self.bin_manager.get_bin_count())
            
    def create_quantile_bins(self):
        """Create quantile bins"""
        num_bins = self.num_bins_spin.value()
        if self.bin_manager.create_quantile_bins(num_bins):
            self.num_bins_spin.setValue(self.bin_manager.get_bin_count())
            
    def create_categorical_bins(self):
        """Create bins for categorical features"""
        if not self.current_feature:
            return
            
        feature_data = self.dataset[self.current_feature].dropna()
        unique_values = sorted(feature_data.unique())
        
        self.bin_manager.clear_bins()
        
        max_categories = min(len(unique_values), 10)
        for value in unique_values[:max_categories]:
            self.bin_manager.add_bin(value, value)
            
        self.num_bins_spin.setValue(self.bin_manager.get_bin_count())
        
    def add_manual_bin(self):
        """Add a manual bin"""
        if not self.current_feature:
            return
            
        bins = self.bin_manager.get_bins()
        if not bins:
            return
            
        last_min, last_max = bins[-1]
        feature_data = self.dataset[self.current_feature].dropna()
        data_max = feature_data.max()
        
        if last_max >= data_max:
            new_min = last_max
            new_max = last_max + (last_max - last_min) * 0.1
        else:
            new_min = last_max
            new_max = data_max
            
        self.bin_manager.add_bin(new_min, new_max)
        self.num_bins_spin.setValue(self.bin_manager.get_bin_count())
        
    def remove_selected_bin(self):
        """Remove selected bin"""
        current_row = self.bin_list_widget.currentRow()
        if current_row >= 0:
            self.bin_manager.remove_bin(current_row)
            self.num_bins_spin.setValue(self.bin_manager.get_bin_count())
            
    def merge_small_bins(self):
        """Merge small bins"""
        min_samples = self.min_samples_spin.value()
        merged_count = self.bin_manager.merge_small_bins(min_samples)
        
        if merged_count > 0:
            self.num_bins_spin.setValue(self.bin_manager.get_bin_count())
            QMessageBox.information(
                self, "Bins Merged",
                f"Merged {merged_count} small bins with adjacent bins."
            )
        else:
            QMessageBox.information(
                self, "No Merge Needed",
                "No bins were small enough to require merging."
            )
            
    def update_bin_display(self):
        """Update the bin list display"""
        self.bin_list_widget.clear()
        
        bins = self.bin_manager.get_bins()
        for i, (min_val, max_val) in enumerate(bins):
            if self.current_feature and pd.api.types.is_numeric_dtype(self.dataset[self.current_feature]):
                text = f"Bin {i+1}: [{min_val:.3f}, {max_val:.3f}]"
            else:
                text = f"Bin {i+1}: {min_val}"
                
            self.bin_list_widget.addItem(text)
            
    def schedule_preview_update(self):
        """Schedule a preview update with debouncing"""
        if self._update_timer:
            self._update_timer.stop()
            self._update_timer.deleteLater()
            
        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self.update_preview)
        self._update_timer.start(200)  # 200ms delay
        
    @performance_timed("preview_update", 0.05)
    def update_preview(self):
        """Update the split preview with performance optimization"""
        if not self.current_feature:
            self.clear_preview()
            return
            
        try:
            with PerformanceTimer("preview_generation"):
                bins = self.bin_manager.get_bins()
                if len(bins) < 2:
                    self.clear_preview()
                    return
                    
                feature_data = self.dataset[self.current_feature].dropna()
                target_data = self.dataset[self.target_column].loc[feature_data.index]
                
                feature_type = 'numeric' if pd.api.types.is_numeric_dtype(feature_data) else 'categorical'
                
                quality_metrics = self.quality_cache.get_or_calculate_quality(
                    feature_data, target_data, bins, self.stats_calculator, feature_type
                )
                
                self.current_preview = self.preview_generator.generate_preview(
                    feature_data, target_data, bins, self.current_feature, feature_type
                )
                
                self.display_preview(self.current_preview)
            
        except Exception as e:
            logger.error(f"Error updating preview: {e}")
            self.clear_preview()
            
    def display_preview(self, preview: SplitPreview):
        """Display the split preview"""
        metrics = preview.quality_metrics
        
        self.impurity_before_label.setText(f"{metrics.impurity_before:.4f}")
        self.impurity_after_label.setText(f"{metrics.impurity_after:.4f}")
        self.impurity_decrease_label.setText(f"{metrics.impurity_decrease:.4f}")
        self.gain_ratio_label.setText(f"{metrics.gain_ratio:.4f}")
        self.quality_score_label.setText(f"{preview.evaluation['quality_score']:.1f}")
        
        if metrics.impurity_decrease > 0:
            self.impurity_decrease_label.setStyleSheet("color: #28a745; font-weight: bold;")
        else:
            self.impurity_decrease_label.setStyleSheet("color: #dc3545; font-weight: bold;")
            
        if preview.recommendations:
            self.recommendations_text.setText("\n".join(f"• {rec}" for rec in preview.recommendations))
        else:
            self.recommendations_text.setText("No specific recommendations.")
            
        if preview.warnings:
            self.warnings_text.setText("\n".join(preview.warnings))
            self.warnings_text.setStyleSheet("color: #856404; background-color: #fff3cd;")
        else:
            self.warnings_text.setText("No warnings.")
            self.warnings_text.setStyleSheet("")
            
        self.details_table.setRowCount(len(preview.bin_details))
        
        for i, detail in enumerate(preview.bin_details):
            self.details_table.setItem(i, 0, QTableWidgetItem(f"Bin {detail['bin_number']}"))
            self.details_table.setItem(i, 1, QTableWidgetItem(detail['range_text']))
            self.details_table.setItem(i, 2, QTableWidgetItem(str(detail['sample_count'])))
            self.details_table.setItem(i, 3, QTableWidgetItem(f"{detail['percentage']:.1f}%"))
            self.details_table.setItem(i, 4, QTableWidgetItem(detail['most_common_class']))
            self.details_table.setItem(i, 5, QTableWidgetItem(f"{detail['purity']:.3f}"))
            
            if detail['status'] == 'empty':
                for col in range(6):
                    item = self.details_table.item(i, col)
                    if item:
                        item.setBackground(QColor("#f8d7da"))  # Light red
            elif detail['status'] == 'small':
                for col in range(6):
                    item = self.details_table.item(i, col)
                    if item:
                        item.setBackground(QColor("#fff3cd"))  # Light yellow
                        
        has_critical_issues = any("⚠️" in warning for warning in preview.warnings)
        self.apply_btn.setEnabled(not has_critical_issues and preview.evaluation['quality_score'] > 0)
        
    def clear_preview(self):
        """Clear the preview display"""
        self.impurity_before_label.setText("N/A")
        self.impurity_after_label.setText("N/A")
        self.impurity_decrease_label.setText("N/A")
        self.gain_ratio_label.setText("N/A")
        self.quality_score_label.setText("N/A")
        
        self.recommendations_text.clear()
        self.warnings_text.clear()
        self.details_table.setRowCount(0)
        
        self.apply_btn.setEnabled(False)
        
    def show_validation_error(self, message: str):
        """Show validation error"""
        QMessageBox.warning(self, "Validation Error", message)
        
    def reset_configuration(self):
        """Reset to initial configuration"""
        if self.current_feature:
            self.on_feature_changed(self.current_feature)
            
    def apply_configuration(self):
        """Apply the current configuration"""
        if not self.current_feature or not self.current_preview:
            QMessageBox.warning(self, "Invalid Configuration", "Please configure the split properly.")
            return
            
        try:
            bins = self.bin_manager.get_bins()
            if len(bins) < 2:
                QMessageBox.warning(self, "Invalid Bins", "At least 2 bins are required.")
                return
                
            if pd.api.types.is_numeric_dtype(self.dataset[self.current_feature]):
                if len(bins) == 2:
                    threshold = bins[0][1]  # Split point
                    split_config = SplitConfigurationFactory.from_dict({
                        'feature': self.current_feature,
                        'split_type': 'numeric',
                        'split_value': threshold
                    })
                else:
                    # TODO: Implement proper multi-bin support
                    threshold = bins[0][1]
                    split_config = SplitConfigurationFactory.from_dict({
                        'feature': self.current_feature,
                        'split_type': 'numeric',
                        'split_value': threshold
                    })
            else:
                if len(bins) == 2:
                    left_categories = [bins[0][0]]
                    right_categories = [bins[1][0]]
                    split_config = SplitConfigurationFactory.from_dict({
                        'feature': self.current_feature,
                        'split_type': 'categorical',
                        'left_categories': left_categories,
                        'right_categories': right_categories
                    })
                else:
                    split_categories = {}
                    for i, (min_val, max_val) in enumerate(bins):
                        split_categories[min_val] = i
                    split_config = SplitConfigurationFactory.from_dict({
                        'feature': self.current_feature,
                        'split_type': 'categorical',
                        'split_categories': split_categories
                    })
                    
            validation_result = split_config.validate()
            if not validation_result.is_valid:
                QMessageBox.warning(
                    self, "Invalid Configuration",
                    f"Split configuration is invalid:\n" + "\n".join(validation_result.errors)
                )
                return
                
            reply = QMessageBox.question(
                self, "Apply Split",
                f"Apply split on '{self.current_feature}' with {len(bins)} bins?\n\n"
                f"Quality Score: {self.current_preview.evaluation['quality_score']:.1f}\n"
                f"Gain Ratio: {self.current_preview.quality_metrics.gain_ratio:.3f}",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.splitConfigured.emit(self.node, split_config)
                self.accept()
                
        except Exception as e:
            logger.error(f"Error applying configuration: {e}")
            QMessageBox.critical(self, "Error", f"Failed to apply configuration: {e}")