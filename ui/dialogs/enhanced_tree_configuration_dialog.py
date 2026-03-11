#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Tree Configuration Dialog for Bespoke Utility
Comprehensive interface for configuring decision tree parameters
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QTabWidget, QWidget, QListWidget,
    QSplitter, QProgressBar, QMessageBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QFrame, QSizePolicy, QSlider, QButtonGroup, QRadioButton
)

from models.decision_tree import SplitCriterion, TreeGrowthMode

logger = logging.getLogger(__name__)


class EnhancedTreeConfigurationDialog(QDialog):
    """Enhanced dialog for configuring decision tree parameters"""
    
    def __init__(self, current_config: Dict[str, Any] = None, 
                 dataset_info: Dict[str, Any] = None, parent=None):
        super().__init__(parent)
        
        self.current_config = current_config or {}
        self.dataset_info = dataset_info or {}
        self.final_config = {}
        
        self.setWindowTitle("Enhanced Tree Configuration")
        self.setModal(True)
        self.resize(800, 700)
        
        self.setupUI()
        self.loadCurrentConfig()
        
    def setupUI(self):
        """Setup the main UI"""
        layout = QVBoxLayout()
        
        header_label = QLabel("Decision Tree Configuration")
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header_label)
        
        if self.dataset_info:
            info_text = (f"Dataset: {self.dataset_info.get('n_samples', '?')} samples, "
                        f"{self.dataset_info.get('n_features', '?')} features")
            info_label = QLabel(info_text)
            layout.addWidget(info_label)
        
        self.tab_widget = QTabWidget()
        
        basic_tab = self.createBasicParametersTab()
        self.tab_widget.addTab(basic_tab, "Basic Parameters")
        
        advanced_tab = self.createAdvancedParametersTab()
        self.tab_widget.addTab(advanced_tab, "Advanced Parameters")
        
        splitting_tab = self.createSplittingCriteriaTab()
        self.tab_widget.addTab(splitting_tab, "Splitting Criteria")
        
        pruning_tab = self.createPruningTab()
        self.tab_widget.addTab(pruning_tab, "Pruning Options")
        
        performance_tab = self.createPerformanceTab()
        self.tab_widget.addTab(performance_tab, "Performance & Memory")
        
        layout.addWidget(self.tab_widget)
        
        presets_group = self.createPresetsGroup()
        layout.addWidget(presets_group)
        
        button_layout = QHBoxLayout()
        
        self.validate_button = QPushButton("Validate Configuration")
        self.validate_button.clicked.connect(self.validateConfiguration)
        
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.resetToDefaults)
        
        button_layout.addWidget(self.validate_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        
        self.apply_button = QPushButton("Apply Configuration")
        self.apply_button.setDefault(True)
        self.apply_button.clicked.connect(self.acceptConfiguration)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def createBasicParametersTab(self) -> QWidget:
        """Create the basic parameters tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        criterion_group = QGroupBox("Splitting Criterion")
        criterion_layout = QVBoxLayout()
        
        self.criterion_group = QButtonGroup()
        
        criteria = [
            (SplitCriterion.GINI, "Gini Impurity", 
             "Standard measure of node impurity, good for balanced datasets"),
            (SplitCriterion.ENTROPY, "Entropy (Information Gain)", 
             "Information-theoretic measure, good for feature selection"),
            (SplitCriterion.MISCLASSIFICATION, "Misclassification Error", 
             "Simple error rate, less sensitive to class imbalance")
        ]
        
        for i, (criterion, name, description) in enumerate(criteria):
            radio = QRadioButton(name)
            radio.setToolTip(description)
            if i == 0:  # Default to Gini
                radio.setChecked(True)
            
            radio.criterion = criterion
            self.criterion_group.addButton(radio, i)
            criterion_layout.addWidget(radio)
            
            desc_label = QLabel(f"  {description}")
            desc_label.setFont(QFont("Arial", 8))
            desc_label.setStyleSheet("color: gray;")
            criterion_layout.addWidget(desc_label)
            
        criterion_group.setLayout(criterion_layout)
        layout.addWidget(criterion_group)
        
        structure_group = QGroupBox("Tree Structure")
        structure_layout = QFormLayout()
        
        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(1, 50)
        self.max_depth_spin.setValue(10)
        self.max_depth_spin.setSpecialValueText("Unlimited")
        self.max_depth_spin.setToolTip("Maximum depth of the tree (0 = unlimited)")
        structure_layout.addRow("Maximum Depth:", self.max_depth_spin)
        
        self.min_samples_split_spin = QSpinBox()
        self.min_samples_split_spin.setRange(2, 1000)
        self.min_samples_split_spin.setValue(2)
        self.min_samples_split_spin.setToolTip("Minimum samples required to split an internal node")
        structure_layout.addRow("Min Samples to Split:", self.min_samples_split_spin)
        
        self.min_samples_leaf_spin = QSpinBox()
        self.min_samples_leaf_spin.setRange(1, 1000)
        self.min_samples_leaf_spin.setValue(1)
        self.min_samples_leaf_spin.setToolTip("Minimum samples required to be at a leaf node")
        structure_layout.addRow("Min Samples per Leaf:", self.min_samples_leaf_spin)
        
        self.max_leaf_nodes_spin = QSpinBox()
        self.max_leaf_nodes_spin.setRange(0, 10000)
        self.max_leaf_nodes_spin.setValue(0)
        self.max_leaf_nodes_spin.setSpecialValueText("Unlimited")
        self.max_leaf_nodes_spin.setToolTip("Maximum number of leaf nodes (0 = unlimited)")
        structure_layout.addRow("Max Leaf Nodes:", self.max_leaf_nodes_spin)
        
        structure_group.setLayout(structure_layout)
        layout.addWidget(structure_group)
        
        growth_group = QGroupBox("Tree Growth Mode")
        growth_layout = QVBoxLayout()
        
        self.growth_mode_group = QButtonGroup()
        
        growth_modes = [
            (TreeGrowthMode.AUTOMATIC, "Automatic", 
             "Build tree automatically using configured parameters"),
            (TreeGrowthMode.MANUAL, "Manual", 
             "Build tree interactively with user guidance"),
            (TreeGrowthMode.HYBRID, "Hybrid", 
             "Start automatic, then allow manual refinement")
        ]
        
        for i, (mode, name, description) in enumerate(growth_modes):
            radio = QRadioButton(name)
            radio.setToolTip(description)
            if i == 0:  # Default to automatic
                radio.setChecked(True)
            
            radio.growth_mode = mode
            self.growth_mode_group.addButton(radio, i)
            growth_layout.addWidget(radio)
            
            desc_label = QLabel(f"  {description}")
            desc_label.setFont(QFont("Arial", 8))
            desc_label.setStyleSheet("color: gray;")
            growth_layout.addWidget(desc_label)
            
        growth_group.setLayout(growth_layout)
        layout.addWidget(growth_group)
        
        # Add clarifying note about manual tree building availability
        note_group = QGroupBox("Important Note")
        note_layout = QVBoxLayout()
        
        note_text = QLabel(
            "Manual tree building mode configures the model for interactive tree construction, "
            "but the actual manual tree building interface becomes available only after "
            "initial model training is complete."
        )
        note_text.setStyleSheet("color: #666; font-size: 9pt;")
        note_text.setWordWrap(True)
        note_layout.addWidget(note_text)
        
        note_group.setLayout(note_layout)
        layout.addWidget(note_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def createAdvancedParametersTab(self) -> QWidget:
        """Create the advanced parameters tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        features_group = QGroupBox("Feature Selection")
        features_layout = QFormLayout()
        
        self.max_features_combo = QComboBox()
        self.max_features_combo.addItems(['All Features', 'sqrt', 'log2', 'Custom'])
        self.max_features_combo.setToolTip("Number of features to consider for each split")
        features_layout.addRow("Max Features:", self.max_features_combo)
        
        self.max_features_custom_spin = QSpinBox()
        self.max_features_custom_spin.setRange(1, 1000)
        self.max_features_custom_spin.setValue(1)
        self.max_features_custom_spin.setEnabled(False)
        features_layout.addRow("Custom Max Features:", self.max_features_custom_spin)
        
        self.max_features_combo.currentTextChanged.connect(
            lambda text: self.max_features_custom_spin.setEnabled(text == 'Custom')
        )
        
        features_group.setLayout(features_layout)
        layout.addWidget(features_group)
        
        quality_group = QGroupBox("Split Quality Thresholds")
        quality_layout = QFormLayout()
        
        self.min_impurity_decrease_spin = QDoubleSpinBox()
        self.min_impurity_decrease_spin.setRange(0.0, 1.0)
        self.min_impurity_decrease_spin.setValue(0.0)
        self.min_impurity_decrease_spin.setDecimals(6)
        self.min_impurity_decrease_spin.setSingleStep(0.001)
        self.min_impurity_decrease_spin.setToolTip("Minimum impurity decrease required for a split")
        quality_layout.addRow("Min Impurity Decrease:", self.min_impurity_decrease_spin)
        
        self.min_weight_fraction_spin = QDoubleSpinBox()
        self.min_weight_fraction_spin.setRange(0.0, 0.5)
        self.min_weight_fraction_spin.setValue(0.0)
        self.min_weight_fraction_spin.setDecimals(4)
        self.min_weight_fraction_spin.setSingleStep(0.01)
        self.min_weight_fraction_spin.setToolTip("Minimum weighted fraction of samples required at a leaf")
        quality_layout.addRow("Min Weight Fraction Leaf:", self.min_weight_fraction_spin)
        
        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)
        
        balance_group = QGroupBox("Class Balancing")
        balance_layout = QVBoxLayout()
        
        self.class_weight_group = QButtonGroup()
        
        weight_options = [
            ('none', "No Weighting", "All classes have equal weight"),
            ('balanced', "Balanced", "Automatically balance class weights"),
            ('custom', "Custom", "Specify custom class weights")
        ]
        
        for i, (weight_type, name, description) in enumerate(weight_options):
            radio = QRadioButton(name)
            radio.setToolTip(description)
            if i == 0:  # Default to no weighting
                radio.setChecked(True)
            
            radio.weight_type = weight_type
            self.class_weight_group.addButton(radio, i)
            balance_layout.addWidget(radio)
            
        self.custom_weights_edit = QLineEdit()
        self.custom_weights_edit.setPlaceholderText("e.g., {0: 1.0, 1: 2.0}")
        self.custom_weights_edit.setEnabled(False)
        balance_layout.addWidget(QLabel("Custom Weights (dict format):"))
        balance_layout.addWidget(self.custom_weights_edit)
        
        self.class_weight_group.buttonClicked.connect(
            lambda btn: self.custom_weights_edit.setEnabled(btn.weight_type == 'custom')
        )
        
        balance_group.setLayout(balance_layout)
        layout.addWidget(balance_group)
        
        random_group = QGroupBox("Reproducibility")
        random_layout = QFormLayout()
        
        self.random_state_spin = QSpinBox()
        self.random_state_spin.setRange(0, 99999)
        self.random_state_spin.setValue(42)
        self.random_state_spin.setToolTip("Random seed for reproducible results")
        random_layout.addRow("Random Seed:", self.random_state_spin)
        
        random_group.setLayout(random_layout)
        layout.addWidget(random_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def createSplittingCriteriaTab(self) -> QWidget:
        """Create the splitting criteria tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        details_group = QGroupBox("Splitting Criterion Details")
        details_layout = QVBoxLayout()
        
        criteria_info = QTextEdit()
        criteria_info.setReadOnly(True)
        criteria_info.setMaximumHeight(200)
        criteria_info.setHtml("""
        <h3>Splitting Criteria Explained</h3>
        
        <h4>Gini Impurity</h4>
        <p>Measures the probability of misclassifying a randomly chosen element. 
        Formula: Gini = 1 - Σ(p_i²) where p_i is the probability of class i.</p>
        <p><b>Best for:</b> Balanced datasets, computational efficiency</p>
        
        <h4>Entropy (Information Gain)</h4>
        <p>Measures the amount of information needed to classify samples.
        Formula: Entropy = -Σ(p_i * log2(p_i))</p>
        <p><b>Best for:</b> Feature selection, interpretability</p>
        
        <h4>Misclassification Error</h4>
        <p>Simple error rate measure.
        Formula: Error = 1 - max(p_i)</p>
        <p><b>Best for:</b> Simple models, less sensitive to class imbalance</p>
        """)
        
        details_layout.addWidget(criteria_info)
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)
        
        params_group = QGroupBox("Criterion Parameters")
        params_layout = QFormLayout()
        
        self.splitter_threshold_spin = QDoubleSpinBox()
        self.splitter_threshold_spin.setRange(0.0, 1.0)
        self.splitter_threshold_spin.setValue(0.5)
        self.splitter_threshold_spin.setDecimals(4)
        self.splitter_threshold_spin.setToolTip("Threshold for binary splits in categorical features")
        params_layout.addRow("Binary Split Threshold:", self.splitter_threshold_spin)
        
        self.allow_multiway_checkbox = QCheckBox("Allow Multi-way Splits")
        self.allow_multiway_checkbox.setToolTip("Allow splits with more than two branches for categorical features")
        params_layout.addRow("", self.allow_multiway_checkbox)
        
        self.categorical_method_combo = QComboBox()
        self.categorical_method_combo.addItems(['Binary', 'One-vs-Rest', 'Optimal'])
        self.categorical_method_combo.setToolTip("Method for handling categorical features")
        params_layout.addRow("Categorical Method:", self.categorical_method_combo)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def createPruningTab(self) -> QWidget:
        """Create the pruning options tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.pruning_enabled_checkbox = QCheckBox("Enable Pruning")
        self.pruning_enabled_checkbox.setChecked(True)
        self.pruning_enabled_checkbox.setToolTip("Enable post-pruning to reduce overfitting")
        layout.addWidget(self.pruning_enabled_checkbox)
        
        method_group = QGroupBox("Pruning Method")
        method_layout = QVBoxLayout()
        
        self.pruning_method_group = QButtonGroup()
        
        pruning_methods = [
            ('cost_complexity', "Cost-Complexity Pruning", 
             "Minimal cost-complexity pruning (recommended)"),
            ('reduced_error', "Reduced Error Pruning", 
             "Prune branches that don't improve validation accuracy"),
            ('critical_value', "Critical Value Pruning", 
             "Prune based on statistical significance"),
            ('pessimistic', "Pessimistic Error Pruning", 
             "Conservative pruning based on error estimates")
        ]
        
        for i, (method, name, description) in enumerate(pruning_methods):
            radio = QRadioButton(name)
            radio.setToolTip(description)
            if i == 0:  # Default to cost-complexity
                radio.setChecked(True)
            
            radio.method = method
            self.pruning_method_group.addButton(radio, i)
            method_layout.addWidget(radio)
            
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)
        
        params_group = QGroupBox("Pruning Parameters")
        params_layout = QFormLayout()
        
        self.pruning_alpha_spin = QDoubleSpinBox()
        self.pruning_alpha_spin.setRange(0.0, 1.0)
        self.pruning_alpha_spin.setValue(0.01)
        self.pruning_alpha_spin.setDecimals(6)
        self.pruning_alpha_spin.setSingleStep(0.001)
        self.pruning_alpha_spin.setToolTip("Complexity parameter for cost-complexity pruning")
        params_layout.addRow("Alpha (Complexity):", self.pruning_alpha_spin)
        
        self.validation_split_spin = QDoubleSpinBox()
        self.validation_split_spin.setRange(0.1, 0.5)
        self.validation_split_spin.setValue(0.2)
        self.validation_split_spin.setDecimals(2)
        self.validation_split_spin.setSingleStep(0.05)
        self.validation_split_spin.setToolTip("Fraction of data to use for pruning validation")
        params_layout.addRow("Validation Split:", self.validation_split_spin)
        
        self.pruning_strategy_combo = QComboBox()
        self.pruning_strategy_combo.addItems(['Conservative', 'Moderate', 'Aggressive'])
        self.pruning_strategy_combo.setCurrentText('Moderate')
        self.pruning_strategy_combo.setToolTip("Pruning aggressiveness level")
        params_layout.addRow("Pruning Strategy:", self.pruning_strategy_combo)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        self.pruning_enabled_checkbox.toggled.connect(method_group.setEnabled)
        self.pruning_enabled_checkbox.toggled.connect(params_group.setEnabled)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def createPerformanceTab(self) -> QWidget:
        """Create the performance and memory tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        memory_group = QGroupBox("Memory Management")
        memory_layout = QFormLayout()
        
        self.memory_optimization_checkbox = QCheckBox("Enable Memory Optimization")
        self.memory_optimization_checkbox.setChecked(True)
        self.memory_optimization_checkbox.setToolTip("Enable memory optimization for large datasets")
        memory_layout.addRow("", self.memory_optimization_checkbox)
        
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(1000, 1000000)
        self.chunk_size_spin.setValue(50000)
        self.chunk_size_spin.setSuffix(" samples")
        self.chunk_size_spin.setToolTip("Chunk size for processing large datasets")
        memory_layout.addRow("Chunk Size:", self.chunk_size_spin)
        
        self.memory_limit_spin = QSpinBox()
        self.memory_limit_spin.setRange(100, 8000)
        self.memory_limit_spin.setValue(800)
        self.memory_limit_spin.setSuffix(" MB")
        self.memory_limit_spin.setToolTip("Maximum memory usage before optimization")
        memory_layout.addRow("Memory Limit:", self.memory_limit_spin)
        
        memory_group.setLayout(memory_layout)
        layout.addWidget(memory_group)
        
        parallel_group = QGroupBox("Parallel Processing")
        parallel_layout = QFormLayout()
        
        self.parallel_enabled_checkbox = QCheckBox("Enable Parallel Processing")
        self.parallel_enabled_checkbox.setChecked(True)
        self.parallel_enabled_checkbox.setToolTip("Use multiple CPU cores for training")
        parallel_layout.addRow("", self.parallel_enabled_checkbox)
        
        self.n_jobs_spin = QSpinBox()
        self.n_jobs_spin.setRange(-1, 32)
        self.n_jobs_spin.setValue(-1)
        self.n_jobs_spin.setSpecialValueText("All Cores")
        self.n_jobs_spin.setToolTip("Number of CPU cores to use (-1 = all available)")
        parallel_layout.addRow("Number of Jobs:", self.n_jobs_spin)
        
        parallel_group.setLayout(parallel_layout)
        layout.addWidget(parallel_group)
        
        cache_group = QGroupBox("Caching")
        cache_layout = QFormLayout()
        
        self.caching_enabled_checkbox = QCheckBox("Enable Caching")
        self.caching_enabled_checkbox.setChecked(True)
        self.caching_enabled_checkbox.setToolTip("Cache intermediate results for faster training")
        cache_layout.addRow("", self.caching_enabled_checkbox)
        
        self.cache_size_spin = QSpinBox()
        self.cache_size_spin.setRange(10, 1000)
        self.cache_size_spin.setValue(100)
        self.cache_size_spin.setSuffix(" MB")
        self.cache_size_spin.setToolTip("Maximum cache size")
        cache_layout.addRow("Cache Size:", self.cache_size_spin)
        
        cache_group.setLayout(cache_layout)
        layout.addWidget(cache_group)
        
        progress_group = QGroupBox("Progress Reporting")
        progress_layout = QFormLayout()
        
        self.verbose_checkbox = QCheckBox("Verbose Output")
        self.verbose_checkbox.setToolTip("Show detailed progress information")
        progress_layout.addRow("", self.verbose_checkbox)
        
        self.progress_interval_spin = QSpinBox()
        self.progress_interval_spin.setRange(1, 100)
        self.progress_interval_spin.setValue(10)
        self.progress_interval_spin.setSuffix(" nodes")
        self.progress_interval_spin.setToolTip("Report progress every N nodes")
        progress_layout.addRow("Progress Interval:", self.progress_interval_spin)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def createPresetsGroup(self) -> QWidget:
        """Create preset configurations group"""
        presets_group = QGroupBox("Preset Configurations")
        layout = QHBoxLayout()
        
        self.fast_button = QPushButton("Fast Training")
        self.fast_button.setToolTip("Optimized for speed, may sacrifice some accuracy")
        self.fast_button.clicked.connect(lambda: self.applyPreset('fast'))
        
        self.balanced_button = QPushButton("Balanced")
        self.balanced_button.setToolTip("Good balance between speed and accuracy")
        self.balanced_button.clicked.connect(lambda: self.applyPreset('balanced'))
        
        self.accurate_button = QPushButton("High Accuracy")
        self.accurate_button.setToolTip("Optimized for maximum accuracy")
        self.accurate_button.clicked.connect(lambda: self.applyPreset('accurate'))
        
        self.memory_efficient_button = QPushButton("Memory Efficient")
        self.memory_efficient_button.setToolTip("Optimized for low memory usage")
        self.memory_efficient_button.clicked.connect(lambda: self.applyPreset('memory'))
        
        layout.addWidget(self.fast_button)
        layout.addWidget(self.balanced_button)
        layout.addWidget(self.accurate_button)
        layout.addWidget(self.memory_efficient_button)
        layout.addStretch()
        
        presets_group.setLayout(layout)
        return presets_group
        
    def loadCurrentConfig(self):
        """Load current configuration into UI"""
        if not self.current_config:
            return
            
        if 'criterion' in self.current_config:
            criterion = self.current_config['criterion']
            for button in self.criterion_group.buttons():
                if hasattr(button, 'criterion') and button.criterion.value == criterion:
                    button.setChecked(True)
                    break
                    
        if 'max_depth' in self.current_config:
            self.max_depth_spin.setValue(self.current_config['max_depth'])
            
        if 'min_samples_split' in self.current_config:
            self.min_samples_split_spin.setValue(self.current_config['min_samples_split'])
            
        if 'min_samples_leaf' in self.current_config:
            self.min_samples_leaf_spin.setValue(self.current_config['min_samples_leaf'])
            
        
    def applyPreset(self, preset_type: str):
        """Apply a preset configuration"""
        presets = {
            'fast': {
                'max_depth': 5,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'max_features': 'sqrt',
                'pruning_enabled': False,
                'memory_optimization': True,
                'parallel_processing': True
            },
            'balanced': {
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'max_features': 'All Features',
                'pruning_enabled': True,
                'pruning_alpha': 0.01,
                'memory_optimization': True,
                'parallel_processing': True
            },
            'accurate': {
                'max_depth': 20,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'All Features',
                'pruning_enabled': True,
                'pruning_alpha': 0.001,
                'memory_optimization': False,
                'parallel_processing': True
            },
            'memory': {
                'max_depth': 8,
                'min_samples_split': 50,
                'min_samples_leaf': 20,
                'max_features': 'sqrt',
                'pruning_enabled': True,
                'chunk_size': 10000,
                'memory_optimization': True,
                'parallel_processing': False
            }
        }
        
        if preset_type not in presets:
            return
            
        preset = presets[preset_type]
        
        self.max_depth_spin.setValue(preset.get('max_depth', 10))
        self.min_samples_split_spin.setValue(preset.get('min_samples_split', 2))
        self.min_samples_leaf_spin.setValue(preset.get('min_samples_leaf', 1))
        
        max_features = preset.get('max_features', 'All Features')
        index = self.max_features_combo.findText(max_features)
        if index >= 0:
            self.max_features_combo.setCurrentIndex(index)
            
        self.pruning_enabled_checkbox.setChecked(preset.get('pruning_enabled', True))
        
        if 'pruning_alpha' in preset:
            self.pruning_alpha_spin.setValue(preset['pruning_alpha'])
            
        if 'chunk_size' in preset:
            self.chunk_size_spin.setValue(preset['chunk_size'])
            
        self.memory_optimization_checkbox.setChecked(preset.get('memory_optimization', True))
        self.parallel_enabled_checkbox.setChecked(preset.get('parallel_processing', True))
        
        QMessageBox.information(self, "Preset Applied", 
                              f"Applied {preset_type.title()} preset configuration.")
        
    def validateConfiguration(self):
        """Validate the current configuration"""
        errors = []
        warnings = []
        
        max_depth = self.max_depth_spin.value()
        min_samples_split = self.min_samples_split_spin.value()
        min_samples_leaf = self.min_samples_leaf_spin.value()
        
        if min_samples_leaf >= min_samples_split:
            errors.append("Min samples per leaf must be less than min samples to split")
            
        if max_depth == 1 and min_samples_split > 2:
            warnings.append("Very shallow trees with high min_samples_split may not split at all")
            
        chunk_size = self.chunk_size_spin.value()
        memory_limit = self.memory_limit_spin.value()
        
        if self.dataset_info.get('n_samples', 0) > 0:
            if chunk_size > self.dataset_info['n_samples']:
                warnings.append("Chunk size is larger than dataset size")
                
        custom_weights_radio = None
        for button in self.class_weight_group.buttons():
            if hasattr(button, 'weight_type') and button.weight_type == 'custom' and button.isChecked():
                custom_weights_radio = button
                break
                
        if custom_weights_radio and custom_weights_radio.isChecked():
            weights_text = self.custom_weights_edit.text().strip()
            if weights_text:
                try:
                    eval(weights_text)  # Basic validation
                except (SyntaxError, NameError, ValueError) as e:
                    logger.debug(f"Error validating custom weights: {e}")
                    errors.append("Invalid custom class weights format")
                    
        if errors:
            error_text = "Configuration Errors:\\n" + "\\n".join(f"• {error}" for error in errors)
            QMessageBox.critical(self, "Configuration Errors", error_text)
        elif warnings:
            warning_text = "Configuration Warnings:\\n" + "\\n".join(f"• {warning}" for warning in warnings)
            QMessageBox.warning(self, "Configuration Warnings", warning_text)
        else:
            QMessageBox.information(self, "Validation Successful", "Configuration is valid!")
            
    def resetToDefaults(self):
        """Reset all parameters to default values"""
        reply = QMessageBox.question(self, "Reset Configuration", 
                                   "Are you sure you want to reset all parameters to default values?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.criterion_group.buttons()[0].setChecked(True)  # Gini
            self.max_depth_spin.setValue(10)
            self.min_samples_split_spin.setValue(2)
            self.min_samples_leaf_spin.setValue(1)
            self.max_leaf_nodes_spin.setValue(0)
            
            self.growth_mode_group.buttons()[0].setChecked(True)  # Automatic
            
            self.max_features_combo.setCurrentText('All Features')
            self.min_impurity_decrease_spin.setValue(0.0)
            self.min_weight_fraction_spin.setValue(0.0)
            self.class_weight_group.buttons()[0].setChecked(True)  # No weighting
            self.random_state_spin.setValue(42)
            
            self.pruning_enabled_checkbox.setChecked(True)
            self.pruning_method_group.buttons()[0].setChecked(True)  # Cost-complexity
            self.pruning_alpha_spin.setValue(0.01)
            self.validation_split_spin.setValue(0.2)
            self.pruning_strategy_combo.setCurrentText('Moderate')
            
            self.memory_optimization_checkbox.setChecked(True)
            self.chunk_size_spin.setValue(50000)
            self.memory_limit_spin.setValue(800)
            self.parallel_enabled_checkbox.setChecked(True)
            self.n_jobs_spin.setValue(-1)
            self.caching_enabled_checkbox.setChecked(True)
            self.cache_size_spin.setValue(100)
            self.verbose_checkbox.setChecked(False)
            self.progress_interval_spin.setValue(10)
            
    def acceptConfiguration(self):
        """Accept the configuration and close dialog"""
        self.validateConfiguration()
        
        config = {}
        
        selected_criterion = None
        for button in self.criterion_group.buttons():
            if button.isChecked() and hasattr(button, 'criterion'):
                selected_criterion = button.criterion
                break
        config['criterion'] = selected_criterion.value if selected_criterion else 'gini'
        
        config['max_depth'] = self.max_depth_spin.value()
        config['min_samples_split'] = self.min_samples_split_spin.value()
        config['min_samples_leaf'] = self.min_samples_leaf_spin.value()
        config['max_leaf_nodes'] = self.max_leaf_nodes_spin.value() if self.max_leaf_nodes_spin.value() > 0 else None
        
        selected_growth_mode = None
        for button in self.growth_mode_group.buttons():
            if button.isChecked() and hasattr(button, 'growth_mode'):
                selected_growth_mode = button.growth_mode
                break
        config['growth_mode'] = selected_growth_mode.value if selected_growth_mode else 'automatic'
        
        max_features_text = self.max_features_combo.currentText()
        if max_features_text == 'All Features':
            config['max_features'] = None
        elif max_features_text == 'Custom':
            config['max_features'] = self.max_features_custom_spin.value()
        else:
            config['max_features'] = max_features_text
            
        config['min_impurity_decrease'] = self.min_impurity_decrease_spin.value()
        config['min_weight_fraction_leaf'] = self.min_weight_fraction_spin.value()
        
        selected_weight_type = None
        for button in self.class_weight_group.buttons():
            if button.isChecked() and hasattr(button, 'weight_type'):
                selected_weight_type = button.weight_type
                break
                
        if selected_weight_type == 'none':
            config['class_weight'] = None
        elif selected_weight_type == 'balanced':
            config['class_weight'] = 'balanced'
        elif selected_weight_type == 'custom':
            weights_text = self.custom_weights_edit.text().strip()
            if weights_text:
                try:
                    config['class_weight'] = eval(weights_text)
                except (SyntaxError, NameError, ValueError) as e:
                    logger.debug(f"Error parsing custom weights: {e}")
                    config['class_weight'] = None
            else:
                config['class_weight'] = None
                
        config['random_state'] = self.random_state_spin.value()
        
        config['pruning_enabled'] = self.pruning_enabled_checkbox.isChecked()
        
        selected_pruning_method = None
        for button in self.pruning_method_group.buttons():
            if button.isChecked() and hasattr(button, 'method'):
                selected_pruning_method = button.method
                break
        config['pruning_method'] = selected_pruning_method if selected_pruning_method else 'cost_complexity'
        
        config['pruning_alpha'] = self.pruning_alpha_spin.value()
        config['validation_split'] = self.validation_split_spin.value()
        config['pruning_strategy'] = self.pruning_strategy_combo.currentText()
        
        config['use_memory_optimization'] = self.memory_optimization_checkbox.isChecked()
        config['chunk_size'] = self.chunk_size_spin.value()
        config['memory_limit'] = self.memory_limit_spin.value()
        config['parallel_enabled'] = self.parallel_enabled_checkbox.isChecked()
        config['n_jobs'] = self.n_jobs_spin.value()
        config['caching_enabled'] = self.caching_enabled_checkbox.isChecked()
        config['cache_size'] = self.cache_size_spin.value()
        config['verbose'] = self.verbose_checkbox.isChecked()
        config['progress_interval'] = self.progress_interval_spin.value()
        
        self.final_config = config
        self.accept()
        
    def getConfiguration(self) -> Dict[str, Any]:
        """Get the final configuration"""
        return self.final_config.copy()