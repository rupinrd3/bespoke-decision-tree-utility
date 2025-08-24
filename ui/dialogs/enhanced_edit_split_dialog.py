#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Edit Split Dialog for Bespoke Utility
Advanced interface for editing splits with multi-bin support and automatic recomputation

"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette, QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QTabWidget, QWidget, QListWidget, QListWidgetItem,
    QSplitter, QProgressBar, QMessageBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QFrame, QSizePolicy, QButtonGroup, QRadioButton, QSlider, QStackedWidget
)

from models.node import TreeNode

from ui.components.numerical_bin_manager import NumericalBinManager
from ui.components.categorical_bin_manager import CategoricalBinManager

logger = logging.getLogger(__name__)


class BinConfigWidget(QWidget):
    """Widget for configuring individual bins with improved layout"""
    
    binChanged = pyqtSignal()
    removeRequested = pyqtSignal(int)  # bin_id
    
    def __init__(self, bin_id: int, min_val: Union[float, str], max_val: Union[float, str], 
                 count: int = 0, target_dist: Dict = None, is_categorical: bool = False):
        super().__init__()
        self.bin_id = bin_id
        self.min_val = min_val
        self.max_val = max_val
        self.count = count
        self.target_dist = target_dist or {}
        self.is_categorical = is_categorical
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the bin configuration UI with better spacing"""
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        
        self.bin_label = QLabel(f"Bin {self.bin_id + 1}")
        self.bin_label.setFixedWidth(50)
        self.bin_label.setStyleSheet("QLabel { font-weight: bold; }")
        layout.addWidget(self.bin_label)
        
        range_layout = QHBoxLayout()
        range_layout.setSpacing(4)
        
        if self.is_categorical:
            self.min_edit = QLineEdit(str(self.min_val))
            self.min_edit.setReadOnly(True)  # Categorical values shouldn't be directly edited
            self.min_edit.setFixedWidth(120)
            self.min_edit.setToolTip("Category value for this bin")
            range_layout.addWidget(self.min_edit)
            
            self.max_edit = None
        else:
            self.min_edit = QLineEdit(f"{self.min_val:.3f}")
            self.min_edit.setValidator(QDoubleValidator())
            self.min_edit.textChanged.connect(self.on_value_changed)
            self.min_edit.setFixedWidth(70)
            self.min_edit.setToolTip("Minimum value for this bin")
            range_layout.addWidget(self.min_edit)
            
            range_layout.addWidget(QLabel("≤ x <"))
            
            self.max_edit = QLineEdit(f"{self.max_val:.3f}")
            self.max_edit.setValidator(QDoubleValidator())
            self.max_edit.textChanged.connect(self.on_value_changed)
            self.max_edit.setFixedWidth(70)
            self.max_edit.setToolTip("Maximum value for this bin")
            range_layout.addWidget(self.max_edit)
        
        layout.addLayout(range_layout)
        
        self.count_label = QLabel(f"{self.count} samples")
        self.count_label.setFixedWidth(80)
        self.count_label.setAlignment(Qt.AlignCenter)
        self.count_label.setStyleSheet("QLabel { color: #2c3e50; font-weight: bold; }")
        layout.addWidget(self.count_label)
        
        dist_text = ", ".join([f"{k}:{v}" for k, v in self.target_dist.items()])
        self.dist_label = QLabel(dist_text or "No data")
        self.dist_label.setStyleSheet("QLabel { color: #7f8c8d; }")
        self.dist_label.setWordWrap(True)
        self.dist_label.setMaximumWidth(150)
        self.dist_label.setMinimumWidth(150)
        layout.addWidget(self.dist_label)
        
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(2)
        
        split_btn = QPushButton("⚡")
        split_btn.setFixedSize(25, 25)
        if self.is_categorical:
            split_btn.setToolTip("Cannot split categorical bins")
            split_btn.setEnabled(False)
            split_btn.setStyleSheet("QPushButton { color: #6c757d; }")
        else:
            split_btn.setToolTip("Split this bin into two")
            split_btn.clicked.connect(self.split_bin)
        buttons_layout.addWidget(split_btn)
        
        if self.bin_id >= 2:
            remove_btn = QPushButton("✕")
            remove_btn.setFixedSize(25, 25)
            remove_btn.setToolTip("Remove this bin")
            remove_btn.setStyleSheet("QPushButton { color: red; font-weight: bold; }")
            remove_btn.clicked.connect(lambda: self.removeRequested.emit(self.bin_id))
            buttons_layout.addWidget(remove_btn)
        
        layout.addLayout(buttons_layout)
        
        self.setStyleSheet("""
            BinConfigWidget {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                margin: 2px;
            }
            BinConfigWidget:hover {
                background-color: #e9ecef;
            }
        """)
        
        self.setLayout(layout)
        self.setFixedHeight(40)  # Fixed height to prevent overlap
        
    def split_bin(self):
        """Split this bin into two equal parts"""
        try:
            if self.is_categorical:
                logger.info("Cannot split categorical bins")
                return
                
            min_val, max_val = self.get_range()
            try:
                min_float = float(min_val)
                max_float = float(max_val)
                mid_val = (min_float + max_float) / 2
                
                if self.max_edit is not None:
                    self.max_edit.setText(f"{mid_val:.3f}")
                
                self.binChanged.emit()
            except (ValueError, TypeError):
                logger.warning(f"Cannot split non-numeric bin: {min_val} to {max_val}")
            
        except Exception as e:
            logger.error(f"Error splitting bin: {e}")
            
    def on_value_changed(self):
        """ENHANCED: Handle value change with immediate validation and feedback"""
        try:
            if hasattr(self, 'min_edit') and hasattr(self, 'max_edit'):
                min_text = self.min_edit.text().strip()
                max_text = self.max_edit.text().strip()
                
                try:
                    if min_text and max_text:
                        min_val = float(min_text)
                        max_val = float(max_text)
                        
                        if min_val >= max_val:
                            self.min_edit.setStyleSheet("QLineEdit { background-color: #ffebee; border: 1px solid #f44336; }")
                            self.max_edit.setStyleSheet("QLineEdit { background-color: #ffebee; border: 1px solid #f44336; }")
                            return  # Don't emit signal for invalid ranges
                        else:
                            self.min_edit.setStyleSheet("")
                            self.max_edit.setStyleSheet("")
                    
                except ValueError:
                    self.min_edit.setStyleSheet("QLineEdit { background-color: #fff3e0; border: 1px solid #ff9800; }")
                    self.max_edit.setStyleSheet("QLineEdit { background-color: #fff3e0; border: 1px solid #ff9800; }")
                    return
            
            self.binChanged.emit()
            
        except Exception as e:
            logger.error(f"Error in bin value change handler: {e}")
            self.binChanged.emit()  # Emit anyway to prevent UI freeze
        
    def get_range(self) -> Tuple[Union[float, str], Union[float, str]]:
        """Get the current range"""
        try:
            if self.is_categorical:
                if self.min_edit is None:
                    return self.min_val, self.max_val
                return self.min_edit.text(), self.min_edit.text()
            else:
                if self.min_edit is None or self.max_edit is None:
                    return self.min_val, self.max_val
                min_val = float(self.min_edit.text())
                max_val = float(self.max_edit.text())
                return min_val, max_val
        except (ValueError, AttributeError):
            return self.min_val, self.max_val
            
    def update_stats(self, count: int, target_dist: Dict):
        """Update the statistics display"""
        self.count = count
        self.target_dist = target_dist
        
        self.count_label.setText(f"{count} samples")
        dist_text = ", ".join([f"{k}:{v}" for k, v in target_dist.items()])
        self.dist_label.setText(dist_text or "No data")
        
    def set_range(self, min_val: Union[float, str], max_val: Union[float, str]):
        """Set the range values"""
        self.min_val = min_val
        self.max_val = max_val
        
        if self.is_categorical:
            self.min_edit.setText(str(min_val))
        else:
            self.min_edit.setText(f"{min_val:.3f}")
            if self.max_edit is not None:
                self.max_edit.setText(f"{max_val:.3f}")


class EnhancedEditSplitDialog(QDialog):
    """Enhanced dialog for editing splits with multi-bin support and improved UI"""
    
    splitModified = pyqtSignal(object, str, object)  # node, feature, new_value
    splitApplied = pyqtSignal(str)  # node_id - emitted when split is successfully applied
    
    def __init__(self, node: TreeNode, dataset: pd.DataFrame, 
                 target_column: str, model=None, initial_split=None, parent=None):
        super().__init__(parent)
        
        self.node = node
        self.dataset = dataset
        self.target_column = target_column
        self.model = model
        self.initial_split = initial_split  # NEW: Support for pre-populated splits
        self.original_feature = getattr(node, 'split_feature', None)
        self.original_value = getattr(node, 'split_value', None)
        
        self.bins = []  # Legacy - kept for compatibility but not used in UI
        self.current_feature = None
        self._initializing = True  # CRITICAL: Flag to prevent premature validation
        self._data_loaded = False  # NEW: Flag to track data loading state
        self._ui_ready = False     # NEW: Flag to track UI readiness
        
        self.setWindowTitle(f"🔧 Enhanced Split Editor - Node {node.node_id}")
        self.setMinimumSize(950, 800)
        self.resize(1100, 850)
        
        self.setStyleSheet("""
            QDialog {
                background-color: #fafafa;
                font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #2c3e50;
                background-color: #fafafa;
            }
        """)
        
        self.setup_ui()
        self._ui_ready = True
        
        self._validate_inputs()
        
        if self._data_loaded:
            self.load_current_split()
            
            if self.initial_split:
                self.populate_from_initial_split()
        
        self._initializing = False
        
        logger.info(f"Enhanced Edit Split Dialog initialized for node {node.node_id}")
        
    def setup_ui(self):
        """Set up the user interface with improved layout"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        
        
        top_section = QWidget()
        top_layout = QHBoxLayout(top_section)
        top_layout.setSpacing(10)
        
        left_top_panel = QWidget()
        left_top_layout = QVBoxLayout(left_top_panel)
        left_top_layout.setSpacing(8)
        
        self.create_feature_selection(left_top_layout)
        
        self.bin_controls_container = QWidget()
        self.bin_controls_container.setMinimumHeight(160)  # Balanced height for all controls
        self.bin_controls_layout = QVBoxLayout(self.bin_controls_container)
        self.bin_controls_layout.setSpacing(10)  # Add spacing between sections
        left_top_layout.addWidget(self.bin_controls_container)
        
        right_top_panel = QWidget()
        right_top_layout = QVBoxLayout(right_top_panel)
        right_top_layout.setSpacing(8)
        
        self.create_preview_panel(right_top_layout)
        
        top_layout.addWidget(left_top_panel)
        top_layout.addWidget(right_top_panel)
        
        top_section.setMaximumHeight(280)
        main_layout.addWidget(top_section)
        
        main_layout.setSpacing(5)
        
        content_splitter = QSplitter(Qt.Horizontal)
        
        self._setup_adaptive_ui()
        content_splitter.addWidget(self.variable_type_stack)
        
        content_splitter.setSizes([500, 400])
        main_layout.addWidget(content_splitter)
        
        self.create_button_bar(main_layout)
        
        self.setLayout(main_layout)
        
    def create_feature_selection(self, layout: QVBoxLayout):
        """Create feature selection section"""
        feature_group = QGroupBox("Feature Selection")
        feature_layout = QVBoxLayout()
        feature_layout.setSpacing(6)
        
        feature_row = QHBoxLayout()
        feature_row.addWidget(QLabel("Feature:"))
        
        self.feature_combo = QComboBox()
        features = [col for col in self.dataset.columns if col != self.target_column]
        self.feature_combo.addItems(features)
        self.feature_combo.currentTextChanged.connect(self.on_feature_changed)
        self.feature_combo.setMinimumWidth(200)
        feature_row.addWidget(self.feature_combo)
        feature_row.addStretch()
        
        feature_layout.addLayout(feature_row)
        
        self.feature_type_label = QLabel("Select a feature to see details")
        self.feature_type_label.setStyleSheet("QLabel { color: #6c757d; font-style: italic; }")
        feature_layout.addWidget(self.feature_type_label)
        
        feature_group.setLayout(feature_layout)
        layout.addWidget(feature_group)
        
        
    def create_preview_panel(self, layout: QVBoxLayout):
        """Create the preview panel"""
        metrics_group = QGroupBox("Split Quality Metrics")
        metrics_layout = QFormLayout()
        metrics_layout.setSpacing(6)
        
        self.impurity_before_label = QLabel("N/A")
        self.impurity_after_label = QLabel("N/A")
        self.impurity_decrease_label = QLabel("N/A")
        self.gain_ratio_label = QLabel("N/A")
        
        metrics_layout.addRow("Impurity Before:", self.impurity_before_label)
        metrics_layout.addRow("Impurity After:", self.impurity_after_label)
        metrics_layout.addRow("Impurity Decrease:", self.impurity_decrease_label)
        metrics_layout.addRow("Gain Ratio:", self.gain_ratio_label)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QFormLayout()
        advanced_layout.setSpacing(6)
        
        self.criterion_combo = QComboBox()
        self.criterion_combo.addItems(["Information Gain", "Gini Gain", "Deviance Reduction"])
        self.criterion_combo.currentTextChanged.connect(self.schedule_preview_update)
        advanced_layout.addRow("Impurity Measure:", self.criterion_combo)
        
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
    def create_button_bar(self, layout: QVBoxLayout):
        """Create the button bar"""
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        reset_btn = QPushButton("🔄 Reset")
        reset_btn.clicked.connect(self.reset_to_original)
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
                color: #fff;
            }
        """)
        self.apply_btn.clicked.connect(self.apply_changes)
        button_layout.addWidget(self.apply_btn)
        
        cancel_btn = QPushButton("❌ Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 20px;
                border-radius: 6px;
                border: 1px solid #6c757d;
                min-width: 100px;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)

    def _setup_adaptive_ui(self):
        """Setup adaptive UI that changes based on variable type"""
        self.variable_type_stack = QStackedWidget()
        
        self.numerical_panel = self._create_numerical_panel()
        self.variable_type_stack.addWidget(self.numerical_panel)
        
        self.categorical_panel = self._create_categorical_panel()
        self.variable_type_stack.addWidget(self.categorical_panel)
        
        # Note: The stacked widget will be added to layout in setup_ui
        
    def _create_numerical_panel(self) -> QWidget:
        """Create panel specifically for numerical variables"""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.numerical_manager_container = QWidget()
        self.numerical_manager_layout = QVBoxLayout(self.numerical_manager_container)
        layout.addWidget(self.numerical_manager_container)
        
        numerical_controls = self._create_numerical_controls()
        layout.addLayout(numerical_controls)
        
        panel.setLayout(layout)
        return panel
        
    def _create_categorical_panel(self) -> QWidget:
        """Create panel specifically for categorical variables using single-widget approach"""
        panel = QWidget()
        main_layout = QVBoxLayout(panel)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        self.categorical_manager_container = QWidget()
        self.categorical_manager_layout = QVBoxLayout(self.categorical_manager_container)
        main_layout.addWidget(self.categorical_manager_container)
        
        categorical_controls = self._create_categorical_controls()
        main_layout.addLayout(categorical_controls)
        
        logger.info("Created categorical panel with manager container")
        return panel
        
    def _create_numerical_controls(self) -> QHBoxLayout:
        """Create controls specific to numerical variables"""
        layout = QHBoxLayout()
        
        layout.addStretch()
        
        return layout
        
    def _create_categorical_controls(self) -> QHBoxLayout:
        """Create controls specific to categorical variables"""
        layout = QHBoxLayout()
        
        layout.addStretch()
        return layout

    def load_current_split(self):
        """Load the current split information"""
        logger.info(f"Loading current split - original_feature: {self.original_feature}")
        
        if self.original_feature and self.original_feature in self.dataset.columns:
            self.feature_combo.setCurrentText(self.original_feature)
            self.on_feature_changed(self.original_feature)
            
            if self.original_value is not None:
                node_data = self.get_node_data()
                if node_data is not None and self.original_feature in node_data.columns:
                    if pd.api.types.is_numeric_dtype(node_data[self.original_feature]):
                        node_children_count = len(getattr(self.node, 'children', []))
                        if node_children_count > 2:
                            if hasattr(self, 'numerical_bin_manager') and self.numerical_bin_manager:
                                self.numerical_bin_manager.set_num_bins(node_children_count)
                                self.numerical_bin_manager.create_equal_width_bins()
                        else:
                            self.create_simple_split()
                    else:
                        self.create_categorical_split()
        else:
            if self.feature_combo.count() > 0:
                feature_name = self.feature_combo.itemText(0)
                logger.info(f"Using default first feature: {feature_name}")
                self.on_feature_changed(feature_name)

    def _validate_inputs(self):
        """
        CRITICAL FIX: Validate all inputs before proceeding
        Maintains UI functionality while preventing KeyError crashes
        """
        try:
            if self.node is None:
                raise ValueError("Node cannot be None")
            
            if self.dataset is None:
                logger.warning("None dataset provided, creating empty DataFrame")
                self.dataset = pd.DataFrame()
            elif len(self.dataset) == 0:
                logger.warning("Empty dataset provided")
                
            if self.target_column and len(self.dataset) > 0:
                if self.target_column not in self.dataset.columns:
                    logger.error(f"Target column '{self.target_column}' not found in dataset columns: {list(self.dataset.columns)}")
                    
                    target_recovered = False
                    if hasattr(self, 'model') and self.model:
                        try:
                            if (hasattr(self.model, '_cached_X') and self.model._cached_X is not None and
                                hasattr(self.model, '_cached_y') and self.model._cached_y is not None):
                                
                                logger.info("Attempting to recover dataset from model cache")
                                self.dataset = self.model._cached_X.copy()
                                self.dataset[self.target_column] = self.model._cached_y
                                target_recovered = True
                                logger.info(f"Successfully recovered dataset with target column '{self.target_column}'")
                                
                        except Exception as recovery_error:
                            logger.warning(f"Failed to recover from model cache: {recovery_error}")
                    
                    if not target_recovered:
                        logger.warning(f"Target column '{self.target_column}' not available, disabling target-based statistics")
                        self.target_column = None
                else:
                    logger.info(f"Target column '{self.target_column}' validated successfully")
            
            self._data_loaded = len(self.dataset) > 0
            
            dataset_info = f"{len(self.dataset)} rows, {len(self.dataset.columns)} cols" if len(self.dataset) > 0 else "empty"
            logger.info(f"Input validation completed - Dataset: {dataset_info}, Target: {self.target_column}, Data loaded: {self._data_loaded}")
            
        except Exception as e:
            logger.error(f"Error during input validation: {e}")
            if not hasattr(self, 'dataset') or self.dataset is None:
                self.dataset = pd.DataFrame()
            self.target_column = None
            self._data_loaded = False
                
    def on_feature_changed(self, feature_name: str):
        """
        CRITICAL FIX: Handle feature change with proper validation sequencing
        """
        try:
            if not self._ui_ready:
                logger.debug(f"Skipping feature change processing - UI not ready: {feature_name}")
                return
            
            if self._initializing:
                logger.debug(f"Feature change during initialization: {feature_name} - allowing critical setup")
        
            if not feature_name:
                logger.warning("Empty feature name provided")
                return
                
            if not self._data_loaded or self.dataset is None or len(self.dataset) == 0:
                logger.warning("No dataset available for feature change")
                return
            
            if feature_name not in self.dataset.columns:
                logger.error(f"Feature '{feature_name}' not found in dataset columns: {list(self.dataset.columns)}")
                return
                
            self.current_feature = feature_name
            logger.info(f"Feature changed to: {feature_name}")
            
            node_data = self.get_node_data()
            if node_data is None or feature_name not in node_data.columns:
                logger.warning(f"Feature '{feature_name}' not found in node data")
                return
                
            feature_data = node_data[feature_name]  # CRITICAL FIX: Include missing values to match actual split behavior
            if len(feature_data) == 0:
                logger.warning(f"No valid data found for feature '{feature_name}'")
                return
                
            variable_info = self._analyze_variable_characteristics(feature_data)
            
            self._adapt_ui_for_variable_type(variable_info)
            
            if variable_info['is_numerical']:
                if not hasattr(self, 'numerical_bin_manager') or self.numerical_bin_manager is None:
                    from ui.components.numerical_bin_manager import NumericalBinManager
                    self.numerical_bin_manager = NumericalBinManager(parent=self)
                    
                    if hasattr(self, 'numerical_manager_layout'):
                        while self.numerical_manager_layout.count():
                            child = self.numerical_manager_layout.takeAt(0)
                            if child.widget():
                                child.widget().deleteLater()
                        self.numerical_manager_layout.addWidget(self.numerical_bin_manager)
                    
                    self.numerical_bin_manager.bins_changed.connect(self._on_numerical_bins_changed)
                    self.numerical_bin_manager.bins_changed.connect(self.schedule_preview_update)
                    logger.info(f"Created and integrated numerical bin manager for {feature_name}")
                else:
                    logger.info(f"Reusing existing numerical bin manager for {feature_name}")
                    
                self._setup_numerical_variable(feature_data, variable_info)
                
                if hasattr(self, 'numerical_bin_manager') and self.numerical_bin_manager:
                    self.numerical_bin_manager.update()
                    self.numerical_bin_manager.show()
                    logger.info("Forced numerical bin manager update and show")
            else:
                if not hasattr(self, 'categorical_bin_manager') or self.categorical_bin_manager is None:
                    logger.info(f"Creating new categorical bin manager for {feature_name}")
                    from ui.components.categorical_bin_manager import CategoricalBinManager
                    self.categorical_bin_manager = CategoricalBinManager(parent=self)
                    
                    if hasattr(self, 'categorical_manager_layout'):
                        while self.categorical_manager_layout.count():
                            child = self.categorical_manager_layout.takeAt(0)
                            if child.widget():
                                child.widget().deleteLater()
                        self.categorical_manager_layout.addWidget(self.categorical_bin_manager)
                        logger.info(f"Added categorical bin manager to layout container")
                        
                        if hasattr(self, 'bin_controls_layout') and hasattr(self.categorical_bin_manager, 'controls_widgets'):
                            while self.bin_controls_layout.count():
                                child = self.bin_controls_layout.takeAt(0)
                                if child.widget():
                                    child.widget().deleteLater()
                            self.bin_controls_layout.addWidget(self.categorical_bin_manager.controls_widgets)
                            logger.info("Moved categorical bin controls to top section")
                    else:
                        logger.warning("categorical_manager_layout not found - cannot add bin manager")
                    
                    if hasattr(self.categorical_bin_manager, 'bins_changed'):
                        self.categorical_bin_manager.bins_changed.connect(self._on_categorical_bins_changed)
                        self.categorical_bin_manager.bins_changed.connect(self.schedule_preview_update)
                        logger.info("Connected categorical bin manager signals")
                    if hasattr(self.categorical_bin_manager, 'validation_error'):
                        self.categorical_bin_manager.validation_error.connect(self._show_validation_error)
                    logger.info(f"Created and integrated categorical bin manager for {feature_name}")
                else:
                    logger.info(f"Reusing existing categorical bin manager for {feature_name}")
                
                self._setup_categorical_variable(feature_data, variable_info)
                
                if hasattr(self, 'initial_split') and self.initial_split and not getattr(self, '_populating_split', False):
                    self._populating_split = True  # Prevent recursion
                    self._populate_existing_categorical_split()
                    self._populating_split = False
                
                if hasattr(self, 'categorical_bin_manager') and self.categorical_bin_manager:
                    self.categorical_bin_manager.update()
                    self.categorical_bin_manager.show()
                    logger.info("Forced categorical bin manager update and show")
                
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(100, self.schedule_preview_update)
            
        except Exception as e:
            logger.error(f"Error handling feature change to '{feature_name}': {e}")
            self.current_feature = None

    def _analyze_variable_type(self, data: pd.Series) -> Dict[str, Any]:
        """Comprehensive variable type analysis"""
        total_count = len(data)
        unique_count = data.nunique()
        
        is_numerical = pd.api.types.is_numeric_dtype(data)
        
        if is_numerical:
            analysis = {
                'is_numerical': True,
                'type_name': 'Numerical',
                'unique_count': unique_count,
                'total_count': total_count,
                'min_value': float(data.min()),
                'max_value': float(data.max()),
                'mean_value': float(data.mean()),
                'std_value': float(data.std()),
                'has_decimals': not all(data == data.astype(int)),
                'is_highly_skewed': False,
                'has_outliers': False,
                'recommended_bins': min(10, max(3, int(np.sqrt(total_count))))
            }
            
            try:
                from scipy import stats
                skewness = abs(stats.skew(data))
                analysis['is_highly_skewed'] = skewness > 1.0
                analysis['skewness'] = skewness
            except ImportError:
                q25, q75 = data.quantile([0.25, 0.75])
                median = data.median()
                analysis['is_highly_skewed'] = abs((q75 - median) - (median - q25)) / (q75 - q25) > 0.5
                
            q1, q3 = data.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            analysis['has_outliers'] = len(outliers) > 0
            analysis['outlier_count'] = len(outliers)
            
        else:
            value_counts = data.value_counts()
            most_common_freq = value_counts.iloc[0] if len(value_counts) > 0 else 0
            
            analysis = {
                'is_numerical': False,
                'type_name': 'Categorical',
                'unique_count': unique_count,
                'total_count': total_count,
                'most_common_value': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_common_freq': most_common_freq,
                'most_common_pct': (most_common_freq / total_count * 100) if total_count > 0 else 0,
                'is_high_cardinality': unique_count > 50,
                'is_binary': unique_count == 2,
                'cardinality_level': 'low' if unique_count <= 10 else 'medium' if unique_count <= 50 else 'high',
                'recommended_bins': min(5, max(2, unique_count // 10))
            }
            
        return analysis
        
    def _adapt_ui_for_variable_type(self, variable_info: Dict[str, Any]):
        """Adapt UI based on detected variable type"""
        try:
            if variable_info['is_numerical']:
                logger.info("Switching to numerical panel for variable")
                self.variable_type_stack.setCurrentWidget(self.numerical_panel)
                
                if hasattr(self, 'numerical_bin_manager') and self.numerical_bin_manager:
                    if variable_info.get('has_decimals', False):
                        self.numerical_bin_manager.set_precision(3)
                    else:
                        self.numerical_bin_manager.set_precision(0)
                    
            else:
                logger.info("Switching to categorical panel for variable")
                self.variable_type_stack.setCurrentWidget(self.categorical_panel)
                
                if variable_info.get('is_high_cardinality', False):
                    self._show_high_cardinality_warning(variable_info)
                elif variable_info.get('is_binary', False):
                    self._show_suggestion("Binary variable detected. Consider keeping both categories separate.")
                elif variable_info.get('has_rare_categories', False):
                    self._show_suggestion("Rare categories detected. Consider grouping low-frequency categories.")
                    
        except Exception as e:
            logger.error(f"Error adapting UI for variable type: {e}")
                
    def _setup_numerical_variable(self, data: pd.Series, variable_info: Dict[str, Any]):
        """Setup numerical variable in the numerical bin manager"""
        if not hasattr(self, 'numerical_bin_manager') or self.numerical_bin_manager is None:
            logger.info("Creating numerical bin manager")
            self.numerical_bin_manager = NumericalBinManager(parent=self)
            self.numerical_bin_manager.bins_changed.connect(self._on_numerical_bins_changed)
            self.numerical_bin_manager.validation_error.connect(self._show_validation_error)
            
            self._clear_container(self.numerical_manager_layout)
            self.numerical_manager_layout.addWidget(self.numerical_bin_manager)
        
        recommended_bins = variable_info.get('recommended_bins', 3)
        self.numerical_bin_manager.set_num_bins(recommended_bins)
        
        self.numerical_bin_manager.load_data(data)
        
        if variable_info.get('is_highly_skewed', False):
            self._show_suggestion("Highly skewed data detected. Using quantile-based binning for better distribution.")
            QTimer.singleShot(100, self.numerical_bin_manager.create_quantile_bins)
        elif variable_info.get('has_outliers', False):
            self._show_suggestion("Outliers detected. Consider using 'Handle Outliers' before binning.")
        else:
            self._show_suggestion("Using equal width binning for normally distributed data.")
            
    def _setup_categorical_variable(self, data: pd.Series, variable_info: Dict[str, Any]):
        """Setup categorical variable in the categorical bin manager"""
        if not hasattr(self, 'categorical_bin_manager') or self.categorical_bin_manager is None:
            logger.warning("Categorical bin manager not found in _setup_categorical_variable - creating as fallback")
            self.categorical_bin_manager = CategoricalBinManager(parent=self)
            self.categorical_bin_manager.bins_changed.connect(self._on_categorical_bins_changed)
            self.categorical_bin_manager.validation_error.connect(self._show_validation_error)
            
            self._clear_container(self.categorical_manager_layout)
            self.categorical_manager_layout.addWidget(self.categorical_bin_manager)
        
        target_data = None
        node_data = self.get_node_data()
        if self.target_column and node_data is not None and self.target_column in node_data.columns:
            target_data = node_data[self.target_column]
        self.categorical_bin_manager.load_data(data, target_data)
        
        if variable_info.get('is_high_cardinality', False):
            self._show_suggestion(f"High cardinality variable ({variable_info['unique_count']} categories) detected. "
                                f"Consider using 'Auto Group' for intelligent grouping.")
        elif variable_info.get('is_binary', False):
            self._show_suggestion("Binary variable detected. Consider keeping both categories as separate bins.")
            
    def _show_high_cardinality_warning(self, variable_info: Dict[str, Any]):
        """Show warning for high cardinality categorical variables"""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("High Cardinality Variable")
        msg.setText(f"This categorical variable has {variable_info['unique_count']} unique categories.")
        msg.setInformativeText("For better model performance and interpretability, consider grouping "
                              "similar categories together using the Auto Group feature or manual grouping.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        
    def _show_suggestion(self, message: str):
        """Show a suggestion message to the user"""
        if hasattr(self, 'suggestions_label'):
            self.suggestions_label.setText(f"💡 Suggestion: {message}")
            self.suggestions_label.setStyleSheet("color: #0066cc; background-color: #e6f3ff; "
                                               "padding: 8px; border-radius: 4px; margin: 5px;")
    
    def _show_validation_error(self, error_message: str):
        """Show validation error from categorical bin manager"""
        QMessageBox.warning(self, "Validation Error", error_message)
    
    def _clear_container(self, layout):
        """Clear all widgets from a container layout"""
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
    
    def _on_numerical_bins_changed(self):
        """Handle changes to numerical bins"""
        try:
            logger.debug("Numerical bins changed - updating preview")
        except Exception as e:
            logger.error(f"Error handling numerical bins change: {e}")
    
    def _on_categorical_bins_changed(self):
        """Handle changes to categorical bins"""
        try:
            logger.debug("Categorical bins changed - updating preview")
        except Exception as e:
            logger.error(f"Error handling categorical bins change: {e}")

    def _update_feature_info_display(self, variable_info: Dict[str, Any]):
        """Update the feature information display"""
        if variable_info['is_numerical']:
            stats_text = (f"Range: {variable_info['min_value']:.3f} - {variable_info['max_value']:.3f} | "
                         f"Mean: {variable_info['mean_value']:.3f} | "
                         f"Unique: {variable_info['unique_count']}")
            if variable_info.get('has_outliers'):
                stats_text += f" | Outliers: {variable_info.get('outlier_count', 0)}"
        else:
            stats_text = (f"Categories: {variable_info['unique_count']} | "
                         f"Most common: {variable_info['most_common_value']} "
                         f"({variable_info['most_common_pct']:.1f}%)")
            
        self.feature_type_label.setText(f"{variable_info['type_name']} | {stats_text}")
        
    def _analyze_variable_characteristics(self, data: pd.Series) -> Dict[str, Any]:
        """Comprehensive variable analysis for intelligent UI adaptation"""
        is_numeric = pd.api.types.is_numeric_dtype(data)
        unique_count = data.nunique()
        
        analysis = {
            'is_numerical': is_numeric,
            'unique_count': unique_count,
            'has_missing': data.isna().any(),
            'missing_rate': data.isna().sum() / len(data),
            'sample_size': len(data)
        }
        
        if is_numeric:
            try:
                from scipy import stats
                
                analysis.update({
                    'has_decimals': (data % 1 != 0).any(),
                    'data_range': (data.min(), data.max()),
                    'mean': data.mean(),
                    'std': data.std(),
                    'skewness': stats.skew(data),
                    'is_highly_skewed': abs(stats.skew(data)) > 2,
                    'has_outliers': self._detect_outliers_iqr(data),
                    'recommended_bins': self._calculate_optimal_bins(data),
                    'distribution_type': self._classify_distribution(data)
                })
            except ImportError:
                analysis.update({
                    'has_decimals': (data % 1 != 0).any(),
                    'data_range': (data.min(), data.max()),
                    'mean': data.mean(),
                    'std': data.std(),
                    'recommended_bins': min(10, max(3, int(np.sqrt(len(data)))))
                })
        else:
            value_counts = data.value_counts()
            analysis.update({
                'is_binary': unique_count == 2,
                'is_high_cardinality': unique_count > 50,
                'cardinality_level': self._classify_cardinality(unique_count),
                'recommended_bins': self._recommend_categorical_bins(unique_count),
                'top_categories': value_counts.head(10).to_dict(),
                'frequency_distribution': self._analyze_frequency_distribution(value_counts),
                'has_rare_categories': (value_counts < len(data) * 0.01).any()
            })
            
        return analysis
        
                
    
                                               
    def _detect_outliers_iqr(self, data: pd.Series) -> bool:
        """Detect outliers using IQR method"""
        if len(data) < 4:
            return False
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return ((data < lower_bound) | (data > upper_bound)).any()
        
    def _calculate_optimal_bins(self, data: pd.Series) -> int:
        """Calculate optimal number of bins using multiple methods"""
        n = len(data)
        
        sturges = int(np.ceil(np.log2(n) + 1))
        
        sqrt_choice = int(np.ceil(np.sqrt(n)))
        
        rice = int(np.ceil(2 * n**(1/3)))
        
        optimal = np.median([sturges, sqrt_choice, rice])
        return int(max(3, min(20, optimal)))
        
    def _classify_distribution(self, data: pd.Series) -> str:
        """Classify the distribution type of numerical data"""
        try:
            from scipy import stats
            skewness = abs(stats.skew(data))
            
            if skewness > 2:
                return 'highly_skewed'
            elif skewness > 1:
                return 'moderately_skewed'
            else:
                return 'normal'
        except ImportError:
            q25, q75 = data.quantile([0.25, 0.75])
            median = data.median()
            asymmetry = abs((q75 - median) - (median - q25)) / (q75 - q25) if q75 != q25 else 0
            
            if asymmetry > 0.5:
                return 'highly_skewed'
            elif asymmetry > 0.3:
                return 'moderately_skewed'
            else:
                return 'normal'
                
    def _classify_cardinality(self, unique_count: int) -> str:
        """Classify cardinality level"""
        if unique_count <= 10:
            return 'low'
        elif unique_count <= 50:
            return 'medium'
        else:
            return 'high'
            
    def _recommend_categorical_bins(self, unique_count: int) -> int:
        """Recommend number of bins for categorical variables"""
        if unique_count <= 5:
            return unique_count  # Keep all categories separate
        elif unique_count <= 20:
            return max(2, unique_count // 2)  # Group into fewer bins
        else:
            return max(3, min(10, unique_count // 10))  # Aggressive grouping
            
    def _analyze_frequency_distribution(self, value_counts: pd.Series) -> Dict[str, Any]:
        """Analyze frequency distribution of categorical values"""
        total = value_counts.sum()
        
        return {
            'entropy': -sum((count/total) * np.log2(count/total) for count in value_counts if count > 0),
            'concentration_ratio': value_counts.head(3).sum() / total,  # Top 3 categories
            'uniformity': len(value_counts[value_counts > total * 0.05]) / len(value_counts)  # Categories with >5% frequency
        }
        
        
    def schedule_preview_update(self):
        """Schedule preview update - FIXED to actually update metrics"""
        try:
            if hasattr(self, 'current_feature') and self.current_feature:
                current_panel = self.variable_type_stack.currentWidget()
                
                if current_panel == self.numerical_panel and hasattr(self, 'numerical_bin_manager'):
                    bin_configs = self.numerical_bin_manager.get_bin_configuration()
                    self.bins = []
                    for config in bin_configs:
                        mock_bin = type('MockBin', (), {})()
                        mock_bin.get_range = lambda c=config: (c.get('min_value', 0), c.get('max_value', 0))
                        self.bins.append(mock_bin)
                        
                elif current_panel == self.categorical_panel and hasattr(self, 'categorical_bin_manager'):
                    bin_configs = self.categorical_bin_manager.get_bin_configuration()
                    self.bins = []
                    for config in bin_configs:
                        mock_bin = type('MockBin', (), {})()
                        categories = config.get('categories', [])
                        if categories:
                            mock_bin.get_range = lambda c=categories[0]: (c, c)
                        else:
                            mock_bin.get_range = lambda: (None, None)
                        self.bins.append(mock_bin)
                
                self.update_preview()
                
        except Exception as e:
            logger.error(f"Error in schedule_preview_update: {e}")
            try:
                self.update_preview()
            except:
                pass
    
    
    
    def _handle_outliers(self):
        """Handle outliers in numerical data"""
        if not self.current_feature or not hasattr(self, 'numerical_bin_manager'):
            return
        
        node_data = self.get_node_data()
        if node_data is None or self.current_feature not in node_data.columns:
            logger.warning(f"No node data available for feature {self.current_feature}")
            return
            
        feature_data = node_data[self.current_feature].dropna()
        
        q1, q3 = feature_data.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_bins = [
            (feature_data.min(), lower_bound),  # Lower outliers
            (lower_bound, upper_bound),         # Normal range
            (upper_bound, feature_data.max())   # Upper outliers
        ]
        
        self.numerical_bin_manager.set_bins(outlier_bins)
        self._show_suggestion(f"Created 3 bins: outliers below {lower_bound:.2f}, normal range {lower_bound:.2f}-{upper_bound:.2f}, and outliers above {upper_bound:.2f}")
            
        
    def create_equal_width_bins(self):
        """Create equal width bins"""
        if not self.current_feature:
            return
            
        self.clear_bins()
        
        node_data = self.get_node_data()
        if node_data is None or self.current_feature not in node_data.columns:
            logger.warning(f"No node data available for feature {self.current_feature}")
            return
            
        data = node_data[self.current_feature].dropna()
        if len(data) == 0:
            return
            
        min_val = data.min()
        max_val = data.max()
        num_bins = self.num_bins_spinner.value()
        
        if max_val == min_val:
            range_extend = max(0.001, abs(min_val) * 0.001)
            max_val = min_val + range_extend
        
        bin_width = (max_val - min_val) / num_bins
        
        for i in range(num_bins):
            bin_min = min_val + i * bin_width
            bin_max = min_val + (i + 1) * bin_width
            
            if i == num_bins - 1:  # Last bin includes max value
                bin_max = max_val
                
            self.add_bin_widget(i, bin_min, bin_max)
            
        self.update_all_bin_stats()
        
    def create_equal_frequency_bins(self):
        """Create equal frequency bins"""
        if not self.current_feature:
            return
            
        self.clear_bins()
        
        node_data = self.get_node_data()
        if node_data is None or self.current_feature not in node_data.columns:
            logger.warning(f"No node data available for feature {self.current_feature}")
            return
            
        data = node_data[self.current_feature].dropna()
        if len(data) == 0:
            return
            
        num_bins = self.num_bins_spinner.value()
        
        try:
            quantiles = np.linspace(0, 1, num_bins + 1)
            bin_edges = data.quantile(quantiles).values
            
            unique_edges = []
            for edge in bin_edges:
                if not unique_edges or edge != unique_edges[-1]:
                    unique_edges.append(edge)
            
            for i in range(len(unique_edges) - 1):
                self.add_bin_widget(i, unique_edges[i], unique_edges[i + 1])
                
            self.num_bins_spinner.setValue(len(unique_edges) - 1)
            
        except Exception as e:
            logger.error(f"Error creating equal frequency bins: {e}")
            self.create_equal_width_bins()
            
        self.update_all_bin_stats()
        
    def create_quantile_bins(self):
        """Create quantile-based bins"""
        if not self.current_feature:
            return
            
        self.clear_bins()
        
        node_data = self.get_node_data()
        if node_data is None or self.current_feature not in node_data.columns:
            logger.warning(f"No node data available for feature {self.current_feature}")
            return
            
        data = node_data[self.current_feature].dropna()
        if len(data) == 0:
            return
            
        num_bins = self.num_bins_spinner.value()
        
        if num_bins == 2:
            quantiles = [0, 0.5, 1]
        elif num_bins == 3:
            quantiles = [0, 0.33, 0.67, 1]
        elif num_bins == 4:
            quantiles = [0, 0.25, 0.5, 0.75, 1]
        elif num_bins == 5:
            quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1]
        else:
            quantiles = np.linspace(0, 1, num_bins + 1)
            
        try:
            bin_edges = data.quantile(quantiles).values
            bin_edges = np.unique(bin_edges)
            
            for i in range(len(bin_edges) - 1):
                self.add_bin_widget(i, bin_edges[i], bin_edges[i + 1])
                
        except Exception as e:
            logger.error(f"Error creating quantile bins: {e}")
            self.create_equal_width_bins()
            
        self.update_all_bin_stats()
        
    def create_categorical_bins(self):
        """Create bins for categorical features"""
        if not self.current_feature:
            return
            
        self.clear_bins()
        
        node_data = self.get_node_data()
        if node_data is None or self.current_feature not in node_data.columns:
            logger.warning(f"No node data available for feature {self.current_feature}")
            return
            
        feature_data = node_data[self.current_feature]
        unique_values = sorted(feature_data.dropna().unique().astype(str))
        
        if feature_data.isna().any():
            unique_values.append("Missing")
            logger.info(f"Added 'Missing' category for {feature_data.isna().sum()} missing values")
        
        max_categories = min(len(unique_values), 10)
        
        for i, value in enumerate(unique_values[:max_categories]):
            self.add_bin_widget(i, value, value, is_categorical=True)
            
        self.num_bins_spinner.setValue(len(self.bins))
        self.update_all_bin_stats()
        
    def create_simple_split(self):
        """Create a simple 2-bin split from existing split value using advanced bin managers"""
        if not self.current_feature or self.original_value is None:
            return
            
        node_data = self.get_node_data()
        if node_data is None:
            return
            
        if pd.api.types.is_numeric_dtype(node_data[self.current_feature]):
            if hasattr(self, 'numerical_bin_manager'):
                data = node_data[self.current_feature].dropna()
                min_val = data.min()
                max_val = data.max()
                
                bins = [(min_val, self.original_value), (self.original_value, max_val)]
                self.numerical_bin_manager.bins = bins
                self.numerical_bin_manager._populate_bins_table()
    
    
    def create_categorical_split(self):
        """Legacy method - no longer used with advanced bin managers"""
        pass
    
        
    def add_bin_widget(self, bin_id: int, min_val: Union[float, str], max_val: Union[float, str], is_categorical: bool = False):
        """Add a new bin widget with comprehensive validation"""
        try:
            if min_val is None or max_val is None:
                logger.warning(f"Invalid bin values: min_val={min_val}, max_val={max_val}")
                return
            
            if not is_categorical:
                try:
                    min_float = float(min_val)
                    max_float = float(max_val)
                    if min_float > max_float:
                        logger.error(f"Invalid bin range: min_val ({min_val}) must be <= max_val ({max_val})")
                        QMessageBox.warning(self, "Invalid Bin Range", 
                                          f"Minimum value ({min_float:.3f}) must be less than or equal to maximum value ({max_float:.3f})")
                        return
                except (ValueError, TypeError):
                    logger.error(f"Invalid numeric values for bin: min_val={min_val}, max_val={max_val}")
                    QMessageBox.warning(self, "Invalid Values", 
                                      f"Invalid numeric values: min={min_val}, max={max_val}")
                    return
                
                if self._bins_overlap(min_val, max_val):
                    logger.error(f"Bin range [{min_val}, {max_val}] overlaps with existing bins")
                    try:
                        min_float = float(min_val)
                        max_float = float(max_val)
                        QMessageBox.warning(self, "Overlapping Bins", 
                                          f"Bin range [{min_float:.3f}, {max_float:.3f}] overlaps with existing bins.\n"
                                          "Overlapping bins would cause ambiguous data assignment.")
                    except (ValueError, TypeError):
                        QMessageBox.warning(self, "Overlapping Bins", 
                                          f"Bin range [{min_val}, {max_val}] overlaps with existing bins.\n"
                                          "Overlapping bins would cause ambiguous data assignment.")
                    return
                
            bin_widget = BinConfigWidget(bin_id, min_val, max_val, is_categorical=is_categorical)
            bin_widget.binChanged.connect(self.update_all_bin_stats)
            bin_widget.removeRequested.connect(self.remove_bin)
            
            self.bins.append(bin_widget)
            if self.bins_layout is not None:
                self.bins_layout.addWidget(bin_widget)
        except Exception as e:
            logger.error(f"Error creating bin widget: {e}")
            
    def _bins_overlap(self, new_min: float, new_max: float) -> bool:
        """Check if new bin overlaps with existing bins - enhanced validation"""
        if new_min is None or new_max is None:
            logger.warning(f"Invalid bin range for overlap check: [{new_min}, {new_max}]")
            return True  # Treat invalid ranges as overlapping
            
        try:
            float(new_min)
            float(new_max)
            if new_min >= new_max:
                logger.warning(f"Invalid numeric bin range: [{new_min}, {new_max}]")
                return True
        except (ValueError, TypeError):
            return False
            
        for bin_widget in self.bins:
            try:
                existing_min, existing_max = bin_widget.get_range()
                
                if existing_min is None or existing_max is None:
                    logger.warning(f"Invalid existing bin range: [{existing_min}, {existing_max}]")
                    continue
                
                try:
                    if float(existing_min) > float(existing_max):
                        logger.warning(f"Invalid existing numeric bin range: [{existing_min}, {existing_max}]")
                        continue
                except (ValueError, TypeError):
                    pass
                    
                tolerance = 1e-10
                try:
                    new_min_float = float(new_min)
                    new_max_float = float(new_max)
                    existing_min_float = float(existing_min)
                    existing_max_float = float(existing_max)
                    
                    if (new_min_float < existing_max_float - tolerance) and (new_max_float > existing_min_float + tolerance):
                        logger.debug(f"Overlap detected: new[{new_min_float:.6f}, {new_max_float:.6f}] overlaps with existing[{existing_min_float:.6f}, {existing_max_float:.6f}]")
                        return True
                except (ValueError, TypeError):
                    continue
            except Exception as e:
                logger.warning(f"Error checking bin overlap: {e}")
                continue
        return False
        
    def validate_bin_coverage(self) -> bool:
        """Validate that bins provide complete coverage without gaps"""
        if not self.bins or not self.current_feature:
            return False
            
        try:
            node_data = self.get_node_data()
            if node_data is None or self.current_feature not in node_data.columns:
                return False
                
            feature_data = node_data[self.current_feature].dropna()
            if len(feature_data) == 0:
                return True  # No data to validate
                
            data_min = feature_data.min()
            data_max = feature_data.max()
            
            bin_ranges = []
            for bin_widget in self.bins:
                try:
                    min_val, max_val = bin_widget.get_range()
                    bin_ranges.append((min_val, max_val))
                except Exception as e:
                    logger.warning(f"Error getting bin range: {e}")
                    continue
                    
            if not bin_ranges:
                return False
                
            bin_ranges.sort()
            
            if bin_ranges[0][0] > data_min:
                logger.warning(f"Bins don't cover minimum data value: {data_min} < {bin_ranges[0][0]}")
                return False
                
            if bin_ranges[-1][1] < data_max:
                logger.warning(f"Bins don't cover maximum data value: {data_max} > {bin_ranges[-1][1]}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating bin coverage: {e}")
            return False
        
    def add_bin(self):
        """ENHANCED: Add a new bin with intelligent positioning and validation"""
        if not self.current_feature:
            logger.warning("No feature selected for adding bin")
            return
            
        try:
            node_data = self.get_node_data()
            if node_data is None or self.current_feature not in node_data.columns:
                logger.warning(f"No node data available for feature {self.current_feature}")
                return
                
            if not pd.api.types.is_numeric_dtype(node_data[self.current_feature]):
                data = node_data[self.current_feature].dropna()
                all_categories = set(data.unique())
                
                assigned_categories = set()
                for bin_widget in self.bins:
                    min_val, max_val = bin_widget.get_range()
                    assigned_categories.add(min_val)  # For categorical, min=max=category
                
                unassigned = all_categories - assigned_categories
                if unassigned:
                    new_category = sorted(list(unassigned))[0]
                    bin_id = len(self.bins)
                    self.add_bin_widget(bin_id, new_category, new_category, is_categorical=True)
                else:
                    logger.info("All categories are already assigned to bins")
                    return
            else:
                if not self.bins:
                    data = node_data[self.current_feature].dropna()
                    if len(data) == 0:
                        return
                    data_min, data_max = data.min(), data.max()
                    range_size = data_max - data_min
                    new_min = data_min
                    new_max = data_min + range_size * 0.5  # Half the range
                else:
                    last_bin = self.bins[-1]
                    last_min, last_max = last_bin.get_range()
                    
                    data = node_data[self.current_feature].dropna()
                    data_max = data.max()
                    
                    if last_max >= data_max:
                        avg_bin_size = sum([(self.bins[i].get_range()[1] - self.bins[i].get_range()[0]) 
                                          for i in range(len(self.bins))]) / len(self.bins)
                        new_min = last_max
                        new_max = last_max + avg_bin_size * 0.5  # Half average bin size
                    else:
                        remaining_range = data_max - last_max
                        new_min = last_max
                        new_max = min(data_max, last_max + max(remaining_range * 0.3, 
                                                              (data_max - data.min()) * 0.1))
                
                bin_id = len(self.bins)
                self.add_bin_widget(bin_id, new_min, new_max)
            
            self.num_bins_spinner.setValue(len(self.bins))
            self.update_all_bin_stats()
            
            feature_data = self.get_safe_feature_data(self.current_feature)
            if feature_data is not None:
                logger.info(f"Added bin {len(self.bins)}: {'categorical' if not pd.api.types.is_numeric_dtype(feature_data) else 'numeric'}")
            else:
                logger.warning(f"Could not determine feature type for {self.current_feature}")
            
        except Exception as e:
            logger.error(f"Error adding bin: {e}")
            try:
                bin_id = len(self.bins)
                self.add_bin_widget(bin_id, 0, 1)
                self.num_bins_spinner.setValue(len(self.bins))
            except:
                pass
        
    def remove_bin(self, bin_id: int):
        """ENHANCED: Remove a bin with intelligent merging and validation"""
        if len(self.bins) <= 2:  # Keep at least 2 bins
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Cannot Remove Bin", 
                              "At least 2 bins are required for a valid split.\n"
                              "Consider merging with adjacent bins instead.")
            return
            
        try:
            bin_to_remove = None
            for bin_widget in self.bins:
                if bin_widget.bin_id == bin_id:
                    bin_to_remove = bin_widget
                    break
                    
            if not bin_to_remove:
                logger.warning(f"Bin {bin_id} not found for removal")
                return
            
            feature_data = self.get_safe_feature_data(self.current_feature)
            if feature_data is not None and pd.api.types.is_numeric_dtype(feature_data) and len(self.bins) > 2:
                remove_min, remove_max = bin_to_remove.get_range()
                
                left_bin = None
                right_bin = None
                
                for i, bin_widget in enumerate(self.bins):
                    if bin_widget.bin_id == bin_id:
                        if i > 0:
                            left_bin = self.bins[i-1]
                        if i < len(self.bins) - 1:
                            right_bin = self.bins[i+1]
                        break
                
                if left_bin:
                    left_min, left_max = left_bin.get_range()
                    left_bin.set_range(left_min, remove_max)
                    logger.info(f"Merged removed bin {bin_id} with left bin {left_bin.bin_id}")
                elif right_bin:
                    right_min, right_max = right_bin.get_range()
                    right_bin.set_range(remove_min, right_max)
                    logger.info(f"Merged removed bin {bin_id} with right bin {right_bin.bin_id}")
            
            bin_to_remove.setParent(None)
            self.bins.remove(bin_to_remove)
            logger.info(f"Removed bin {bin_id}")
                
            for i, bin_widget in enumerate(self.bins):
                bin_widget.bin_id = i
                bin_widget.bin_label.setText(f"Bin {i + 1}")
                
            self.num_bins_spinner.setValue(len(self.bins))
            self.update_all_bin_stats()
            
        except Exception as e:
            logger.error(f"Error removing bin {bin_id}: {e}")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", f"Failed to remove bin: {e}")
        
    def merge_small_bins(self):
        """Merge bins with very few samples"""
        if len(self.bins) <= 2:
            return
            
        node_data = self.get_node_data()
        if node_data is None:
            return
            
        total_samples = len(node_data)
        min_samples_threshold = max(1, total_samples // 20)  # At least 5% of data
        
        merged_any = False
        i = 0
        while i < len(self.bins) - 1:
            bin_widget = self.bins[i]
            min_val, max_val = bin_widget.get_range()
            
            if pd.api.types.is_numeric_dtype(node_data[self.current_feature]):
                mask = (node_data[self.current_feature] >= min_val) & (node_data[self.current_feature] < max_val)
                if bin_widget == self.bins[-1]:  # Last bin includes max
                    mask = (node_data[self.current_feature] >= min_val) & (node_data[self.current_feature] <= max_val)
            else:
                mask = node_data[self.current_feature] == min_val
                
            count = np.sum(mask)
            
            if count < min_samples_threshold and i < len(self.bins) - 1:
                next_bin = self.bins[i + 1]
                _, next_max = next_bin.get_range()
                
                bin_widget.set_range(min_val, next_max)
                
                self.remove_bin(next_bin.bin_id)
                merged_any = True
            else:
                i += 1
                
        if merged_any:
            self.update_all_bin_stats()
            QMessageBox.information(
                self, "Bins Merged",
                f"Small bins (< {min_samples_threshold} samples) have been merged with adjacent bins."
            )
        else:
            QMessageBox.information(self, "No Merge Needed", "No bins were small enough to require merging.")
        
    def on_num_bins_changed(self, num_bins: int):
        """Handle change in number of bins"""
        current_bins = len(self.bins)
        
        if num_bins > current_bins:
            for _ in range(num_bins - current_bins):
                self.add_bin()
        elif num_bins < current_bins and num_bins >= 2:
            while len(self.bins) > num_bins:
                self.remove_bin(len(self.bins) - 1)
                
    def update_all_bin_stats(self):
        """ENHANCED: Update statistics for all bins with real-time feedback and validation"""
        try:
            if not self.current_feature or not self.bins:
                logger.debug("No feature or bins to update")
                return
                
            node_data = self.get_node_data()
            if node_data is None or self.current_feature not in node_data.columns:
                logger.warning(f"No data available for feature {self.current_feature}")
                return
            
            total_samples = len(node_data)
            assigned_samples = 0
            overlapping_bins = []
            empty_bins = []
            
            for i, bin_widget in enumerate(self.bins):
                try:
                    min_val, max_val = bin_widget.get_range()
                    
                    if pd.api.types.is_numeric_dtype(node_data[self.current_feature]):
                        if i == len(self.bins) - 1:  # Last bin includes max value
                            mask = (node_data[self.current_feature] >= min_val) & (node_data[self.current_feature] <= max_val)
                        else:
                            mask = (node_data[self.current_feature] >= min_val) & (node_data[self.current_feature] < max_val)
                            
                        if i > 0:
                            prev_min, prev_max = self.bins[i-1].get_range()
                            if min_val < prev_max:
                                overlapping_bins.append((i-1, i))
                                
                    else:
                        mask = node_data[self.current_feature] == min_val
                        
                    bin_data = node_data[mask]
                    count = len(bin_data)
                    assigned_samples += count
                    
                    if count == 0:
                        empty_bins.append(i)
                    
                    if self.target_column in bin_data.columns and count > 0:
                        target_dist = bin_data[self.target_column].value_counts().to_dict()
                        target_dist_pct = {k: f"{v} ({100*v/count:.1f}%)" for k, v in target_dist.items()}
                    else:
                        target_dist = {}
                        target_dist_pct = {}
                        
                    bin_widget.update_stats(count, target_dist_pct)
                    
                    if count == 0:
                        bin_widget.setStyleSheet("""
                            BinConfigWidget {
                                background-color: #ffebee;
                                border: 2px solid #f44336;
                                border-radius: 4px;
                                margin: 2px;
                            }
                        """)
                    elif count < max(1, total_samples * 0.01):  # Less than 1% of data
                        bin_widget.setStyleSheet("""
                            BinConfigWidget {
                                background-color: #fff3e0;
                                border: 2px solid #ff9800;
                                border-radius: 4px;
                                margin: 2px;
                            }
                        """)
                    else:
                        bin_widget.setStyleSheet("""
                            BinConfigWidget {
                                background-color: #f8f9fa;
                                border: 1px solid #dee2e6;
                                border-radius: 4px;
                                margin: 2px;
                            }
                            BinConfigWidget:hover {
                                background-color: #e9ecef;
                            }
                        """)
                        
                except Exception as e:
                    logger.error(f"Error updating bin {i}: {e}")
                    continue
            
            coverage_pct = (assigned_samples / total_samples * 100) if total_samples > 0 else 0
            
            status_messages = []
            if empty_bins:
                status_messages.append(f"⚠️ {len(empty_bins)} empty bins")
            if overlapping_bins:
                status_messages.append(f"⚠️ {len(overlapping_bins)} overlapping ranges")
            if coverage_pct < 95:
                unassigned = total_samples - assigned_samples
                status_messages.append(f"⚠️ {unassigned} samples unassigned ({100-coverage_pct:.1f}%)")
            
            if status_messages:
                logger.info(f"Bin validation: {'; '.join(status_messages)}")
            else:
                logger.debug(f"All bins valid: {len(self.bins)} bins covering {coverage_pct:.1f}% of data")
                
            if hasattr(self, '_preview_update_timer') and self._preview_update_timer:
                self._preview_update_timer.stop()
                try:
                    self._preview_update_timer.deleteLater()
                except RuntimeError:
                    pass
                self._preview_update_timer = None
            
            delay = 150 if len(self.bins) <= 3 else 300
            self._preview_update_timer = QTimer()
            self._preview_update_timer.setSingleShot(True)
            self._preview_update_timer.timeout.connect(self.update_preview)
            self._preview_update_timer.start(delay)
            
        except Exception as e:
            logger.error(f"Error updating bin stats: {e}")
            for bin_widget in self.bins:
                try:
                    bin_widget.update_stats(0, {'Error': 'Unable to calculate'})
                except:
                    pass
        
    def update_preview(self):
        """CRITICAL FIX: Update preview with proper error handling"""
        try:
            if self._initializing or not self._ui_ready or not self._data_loaded:
                logger.debug("Skipping preview update - not ready")
                return
                
            if not self.current_feature:
                logger.debug("No current feature for preview update")
                return
                
            node_data = self.get_node_data()
            if node_data is None:
                return
                
            current_impurity = self.calculate_impurity(node_data)
            
            total_samples = len(node_data)
            weighted_impurity = 0.0
            
            if hasattr(self, 'impurity_before_label'):
                self.impurity_before_label.setText(f"{current_impurity:.4f}")
            
            if not hasattr(self, 'bins') or not self.bins:
                return
            
            for i, bin_widget in enumerate(self.bins):
                min_val, max_val = bin_widget.get_range()
                
                if pd.api.types.is_numeric_dtype(node_data[self.current_feature]):
                    mask = (node_data[self.current_feature] >= min_val) & (node_data[self.current_feature] < max_val)
                    if bin_widget == self.bins[-1]:
                        mask = (node_data[self.current_feature] >= min_val) & (node_data[self.current_feature] <= max_val)
                    try:
                        min_float = float(min_val)
                        max_float = float(max_val)
                        range_text = f"[{min_float:.3f}, {max_float:.3f}]"
                    except (ValueError, TypeError):
                        range_text = f"[{min_val}, {max_val}]"
                else:
                    mask = node_data[self.current_feature] == min_val
                    range_text = str(min_val)
                    
                bin_data = node_data[mask]
                count = len(bin_data)
                
                bin_impurity = self.calculate_impurity(bin_data)
                if total_samples > 0:
                    weighted_impurity += (count / total_samples) * bin_impurity
                
                if self.target_column in bin_data.columns and count > 0:
                    target_dist = bin_data[self.target_column].value_counts()
                    dist_text = ", ".join([f"{k}:{v}" for k, v in target_dist.items()])
                    purity = 1.0 - bin_impurity
                else:
                    dist_text = "No data"
                    purity = 0.0
                    
                pass
                
                pass
                            
            criterion = self.criterion_combo.currentText() if self.criterion_combo else "Gini Gain"
            
            if criterion == "Information Gain":
                stat_value = current_impurity - weighted_impurity
                stat_label = "Information Gain"
            elif criterion == "Gini Gain":
                stat_value = current_impurity - weighted_impurity
                stat_label = "Gini Gain"
            elif criterion == "Deviance Reduction":
                # Note: For deviance reduction, we don't weight the child deviances
                child_deviances_sum = 0
                for bin_widget in self.bins:
                    try:
                        min_val, max_val = bin_widget.get_range()
                        if pd.api.types.is_numeric_dtype(node_data[self.current_feature]):
                            mask = (node_data[self.current_feature] >= min_val) & (node_data[self.current_feature] < max_val)
                        else:
                            mask = node_data[self.current_feature] == min_val
                        bin_data = node_data[mask]
                        if len(bin_data) > 0:
                            child_deviances_sum += self.calculate_impurity(bin_data)
                    except:
                        continue
                stat_value = current_impurity - child_deviances_sum
                stat_label = "Deviance Reduction"
            else:
                stat_value = current_impurity - weighted_impurity
                stat_label = "Gini Gain"
            
            gain_ratio = stat_value / current_impurity if current_impurity > 0 else 0
            
            self.impurity_before_label.setText(f"{current_impurity:.4f}")
            self.impurity_after_label.setText(f"{weighted_impurity:.4f}")
            self.impurity_decrease_label.setText(f"{stat_value:.4f} ({stat_label})")
            self.gain_ratio_label.setText(f"{gain_ratio:.4f}")
            
            if stat_value > 0:
                self.impurity_decrease_label.setStyleSheet("QLabel { color: #28a745; font-weight: bold; }")
            else:
                self.impurity_decrease_label.setStyleSheet("QLabel { color: #dc3545; font-weight: bold; }")
                
        except Exception as e:
            logger.error(f"Error updating preview: {e}")
            
    def get_node_data(self) -> Optional[pd.DataFrame]:
        """Get the data for this node - FIXED to return actual node-specific data with comprehensive validation"""
        try:
            if self.dataset is None or self.dataset.empty:
                logger.warning("No dataset available")
                return None
            
            if not hasattr(self, 'node') or self.node is None:
                logger.warning("No node available, using full dataset")
                return self.dataset.copy()
            
            if hasattr(self.model, 'get_node_data') and hasattr(self.node, 'node_id'):
                try:
                    node_data = self.model.get_node_data(self.node.node_id)
                    if node_data is not None and len(node_data) > 0:
                        logger.info(f"Retrieved {len(node_data)} rows for node {self.node.node_id} from model")
                        return node_data
                except Exception as e:
                    logger.warning(f"Failed to get node data from model: {e}")
            
            if (hasattr(self.model, '_get_node_sample_indices') and 
                hasattr(self.model, '_cached_X') and self.model._cached_X is not None):
                try:
                    sample_indices = self.model._get_node_sample_indices(self.model._cached_X, self.node)
                    if (sample_indices and len(sample_indices) > 0 and
                        all(isinstance(idx, (int, np.integer)) and 0 <= idx < len(self.dataset) for idx in sample_indices)):
                        
                        filtered_data = self.dataset.iloc[sample_indices].copy()
                        logger.info(f"Retrieved {len(filtered_data)} rows for node {self.node.node_id} using calculated indices")
                        return filtered_data
                except Exception as e:
                    logger.warning(f"Failed to calculate sample indices: {e}")
            
            if hasattr(self.node, 'sample_indices') and self.node.sample_indices is not None:
                try:
                    indices = self.node.sample_indices
                    if (len(indices) > 0 and 
                        all(0 <= idx < len(self.dataset) for idx in indices)):
                        filtered_data = self.dataset.iloc[indices].copy()
                        logger.info(f"Retrieved {len(filtered_data)} rows for node {self.node.node_id} using cached sample_indices")
                        return filtered_data
                    else:
                        logger.warning(f"Invalid cached sample_indices for node {self.node.node_id}")
                except Exception as e:
                    logger.warning(f"Failed to filter using cached sample_indices: {e}")
            
            if hasattr(self.node, 'node_id') and self.node.node_id == 'root':
                logger.info(f"Root node detected, returning full dataset: {len(self.dataset)} rows")
                return self.dataset.copy()
            
            logger.warning(f"Could not determine node-specific data for node {getattr(self.node, 'node_id', 'unknown')}, falling back to full dataset")
            return self.dataset.copy()
            
        except Exception as e:
            logger.error(f"Error getting node data: {e}")
            try:
                if self.dataset is not None and not self.dataset.empty:
                    logger.warning("Falling back to full dataset due to error")
                    return self.dataset.copy()
                else:
                    logger.error("No valid dataset available for fallback")
                    return None
            except Exception as fallback_error:
                logger.error(f"Critical error in fallback: {fallback_error}")
                return None
    
    def get_safe_feature_data(self, feature_name: str, node_data: Optional[pd.DataFrame] = None) -> Optional[pd.Series]:
        """Safely get feature data with comprehensive validation for all variable types"""
        try:
            if node_data is None:
                node_data = self.get_node_data()
            
            if not feature_name:
                logger.error("No feature name provided")
                return None
            
            if node_data is not None and not node_data.empty and feature_name in node_data.columns:
                feature_data = node_data[feature_name]
                logger.debug(f"Using node-specific data for feature '{feature_name}': {len(feature_data)} rows")
            elif self.dataset is not None and feature_name in self.dataset.columns:
                feature_data = self.dataset[feature_name]
                logger.warning(f"Falling back to full dataset for feature '{feature_name}': {len(feature_data)} rows")
            else:
                logger.error(f"Feature '{feature_name}' not found in any dataset")
                return None
            
            if len(feature_data) == 0:
                logger.warning(f"Feature '{feature_name}' contains no data")
                return feature_data  # Return empty series for consistency
            
            if pd.api.types.is_numeric_dtype(feature_data):
                unique_vals = feature_data.dropna().nunique()
                if unique_vals <= 2:
                    logger.debug(f"Feature '{feature_name}' detected as Binary (unique values: {unique_vals})")
                else:
                    logger.debug(f"Feature '{feature_name}' detected as Numerical (unique values: {unique_vals})")
            else:
                unique_vals = feature_data.dropna().nunique()
                logger.debug(f"Feature '{feature_name}' detected as Categorical/Ordinal (unique values: {unique_vals})")
            
            return feature_data
            
        except Exception as e:
            logger.error(f"Error getting safe feature data for '{feature_name}': {e}")
            return None
        
    def calculate_impurity(self, data: pd.DataFrame) -> float:
        """Calculate impurity based on selected criterion using the new formulas"""
        if self.target_column not in data.columns or len(data) == 0:
            return 0.0
            
        class_counts = data[self.target_column].value_counts()
        total = len(data)
        
        if total == 0:
            return 0.0
            
        if self.criterion_combo is None:
            criterion = "Gini Gain"
        else:
            criterion = self.criterion_combo.currentText()
        
        try:
            if criterion == "Information Gain":
                probabilities = class_counts / total
                probabilities = probabilities[probabilities > 0]  # Avoid log(0)
                entropy = -np.sum(probabilities * np.log2(probabilities))
                return entropy
            elif criterion == "Gini Gain":
                probabilities = class_counts / total
                gini = 1.0 - np.sum(probabilities ** 2)
                return gini
            elif criterion == "Deviance Reduction":
                p = np.mean(data[self.target_column])  # proportion of positive class
                if p <= 0 or p >= 1:
                    return 0.0
                deviance = -2 * total * (p * np.log(p) + (1 - p) * np.log(1 - p))
                return deviance
            else:
                probabilities = class_counts / total
                gini = 1.0 - np.sum(probabilities ** 2)
                return gini
        except Exception as e:
            logger.error(f"Error calculating {criterion} impurity: {e}")
            return 0.0
        
    def reset_to_original(self):
        """Reset to original split configuration"""
        try:
            if self.original_feature and self.original_feature in [self.feature_combo.itemText(i) for i in range(self.feature_combo.count())]:
                self.feature_combo.setCurrentText(self.original_feature)
                self.on_feature_changed(self.original_feature)
                
                if self.original_value is not None:
                    QTimer.singleShot(100, self.create_simple_split)
                else:
                    QTimer.singleShot(100, self._create_default_bins)
            else:
                if self.feature_combo.count() > 0:
                    first_feature = self.feature_combo.itemText(0)
                    self.feature_combo.setCurrentText(first_feature)
                    self.on_feature_changed(first_feature)
                    QTimer.singleShot(100, self._create_default_bins)
                    
            QMessageBox.information(self, "Reset Complete", "Split configuration has been reset.")
            
        except Exception as e:
            logger.error(f"Error resetting to original: {e}")
            QMessageBox.warning(self, "Reset Error", f"Could not reset configuration: {str(e)}")
    
    def _create_default_bins(self):
        """Create default bins for the current feature"""
        if not self.current_feature:
            return
            
        try:
            node_data = self.get_node_data()
            if node_data is None or self.current_feature not in node_data.columns:
                return
                
            is_numeric = pd.api.types.is_numeric_dtype(node_data[self.current_feature])
            
            if is_numeric:
                if hasattr(self, 'numerical_bin_manager'):
                    self.numerical_bin_manager.create_equal_width_bins()
            else:
                if hasattr(self, 'categorical_bin_manager'):
                    self.categorical_bin_manager.create_default_bins()
                    
        except Exception as e:
            logger.error(f"Error creating default bins: {e}")
            
    def apply_changes(self):
        """CRITICAL FIX: Apply changes with proper validation sequencing"""
        try:
            if self._initializing:
                logger.debug("Skipping apply_changes during initialization")
                return
                
            if not self.current_feature:
                QMessageBox.warning(self, "Invalid Configuration", "Please select a feature first.")
                return
                
            if not self._data_loaded:
                QMessageBox.warning(self, "No Data", "No data available for validation.")
                return
                
            node_data = self.get_node_data()
            if node_data is None:
                QMessageBox.warning(self, "No Data", "No data available for validation.")
                return
                
            is_numeric = pd.api.types.is_numeric_dtype(node_data[self.current_feature])
            
            bin_definitions = []
            bin_configs = []
            valid_bins = []
            category_groups = []  # Initialize category_groups for both paths
            
            if is_numeric:
                if not hasattr(self, 'numerical_bin_manager'):
                    QMessageBox.warning(self, "Configuration Error", "Numerical bin manager not available.")
                    return
                    
                bin_definitions = self.numerical_bin_manager.get_bin_definitions()
                if len(bin_definitions) < 2:
                    QMessageBox.warning(self, "Invalid Bins", "At least 2 bins are required.")
                    return
            else:
                if not hasattr(self, 'categorical_bin_manager'):
                    QMessageBox.warning(self, "Configuration Error", "Categorical bin manager not available.")
                    return
                    
                bin_configs = self.categorical_bin_manager.get_bin_configuration()
                
                valid_bins = [config for config in bin_configs if config.get('categories')]
                
                if hasattr(self, 'initial_split') and self.initial_split:
                    logger.info("Editing existing categorical split - bypassing initial bin count validation")
                elif len(valid_bins) < 2:
                    QMessageBox.warning(self, "Invalid Bins", 
                                      f"At least 2 bins with categories are required. Currently have {len(valid_bins)} valid bins.\n\n"
                                      f"Please use the 'Auto Group' button or drag categories to bins.")
                    return
                
            if is_numeric:
                empty_bins = []
                for i, (min_val, max_val) in enumerate(bin_definitions):
                    if i == len(bin_definitions) - 1:  # Last bin includes max
                        mask = (node_data[self.current_feature] >= min_val) & (node_data[self.current_feature] <= max_val)
                    else:
                        mask = (node_data[self.current_feature] >= min_val) & (node_data[self.current_feature] < max_val)
                    
                    if np.sum(mask) == 0:
                        empty_bins.append(i + 1)
                        
                if empty_bins:
                    reply = QMessageBox.question(
                        self, "Empty Bins Warning",
                        f"Bins {empty_bins} contain no samples. Continue anyway?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.No:
                        return
                
                thresholds = []
                for i in range(len(bin_definitions) - 1):
                    _, max_val = bin_definitions[i]
                    thresholds.append(max_val)
                
                # Log the threshold extraction for debugging
                logger.info(f"Bin definitions received: {bin_definitions}")
                logger.info(f"Extracted thresholds: {thresholds}")
                
                if len(bin_definitions) == 2 and len(thresholds) == 1:
                    split_config = {
                        "type": "numeric_binary",
                        "feature": self.current_feature,
                        "split_value": thresholds[0]
                    }
                else:
                    split_config = {
                        "feature": self.current_feature,
                        "type": "numeric_multi_bin",
                        "thresholds": thresholds,
                        "num_bins": len(bin_definitions)
                    }
            else:
                empty_bins = []
                category_groups = []
                for i, bin_config in enumerate(bin_configs):
                    categories = bin_config.get('categories', [])
                    if not categories:
                        empty_bins.append(i + 1)
                        continue
                    
                    mask = node_data[self.current_feature].isin(categories)
                    if np.sum(mask) == 0:
                        empty_bins.append(i + 1)
                        
                    category_groups.append(categories)
                    
                valid_bins = [config for config in bin_configs if config.get('categories')]
                
                if empty_bins:
                    reply = QMessageBox.question(
                        self, "Empty Bins Warning",
                        f"Bins {empty_bins} contain no samples. Continue anyway?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if reply == QMessageBox.No:
                        return
                
                if len(category_groups) == 2:
                    split_config = {
                        "type": "categorical_binary",
                        "feature": self.current_feature,
                        "left_categories": category_groups[0],
                        "right_categories": category_groups[1]
                    }
                else:
                    split_categories = {}
                    for i, group in enumerate(category_groups):
                        for category in group:
                            split_categories[category] = i
                    
                    split_config = {
                        "feature": self.current_feature,
                        "type": "categorical_multi_bin", 
                        "split_categories": split_categories
                    }
            
            num_bins = len(bin_definitions) if is_numeric else len(valid_bins)
            reply = QMessageBox.question(
                self, "Apply Split",
                f"Apply {num_bins}-bin split on {self.current_feature}?\n\n"
                f"This will replace the current split on node {self.node.node_id}.",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                success = self._apply_split_to_model(split_config)
                if success:
                    self.splitModified.emit(self.node, self.current_feature, split_config)
                    self.splitApplied.emit(self.node.node_id)
                    self.accept()
                else:
                    QMessageBox.critical(self, "Split Application Failed", 
                                       "Failed to apply the split. Please check the logs for details.")
                                   
        except Exception as e:
            logger.error(f"Error applying changes: {e}")
            QMessageBox.critical(self, "Application Error", f"Failed to apply changes: {e}")
    
    def _apply_split_to_model(self, split_config: Dict[str, Any]) -> bool:
        """Apply the split configuration to the model"""
        try:
            logger.debug(f"Checking model availability - hasattr: {hasattr(self, 'model')}, model: {getattr(self, 'model', 'NOT_SET')}")
            if not hasattr(self, 'model') or self.model is None:
                logger.error(f"No model available to apply split - hasattr: {hasattr(self, 'model')}, model is None: {getattr(self, 'model', 'NOT_SET') is None}")
                return False
                
            model_split_config = self._convert_to_model_format(split_config)
            
            success = self.model.apply_manual_split(self.node.node_id, model_split_config)
            
            if success:
                logger.info(f"Successfully applied {split_config.get('type', 'unknown')} split to node {self.node.node_id}")
                return True
            else:
                logger.error(f"Failed to apply split to node {self.node.node_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error applying split to model: {e}")
            return False
            
    def _convert_to_model_format(self, split_config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert enhanced dialog split config to model format"""
        config_type = split_config.get('type', 'unknown')
        
        if config_type == 'numeric_binary':
            return {
                'feature': split_config['feature'],
                'split_type': 'numeric',
                'split_value': split_config['split_value'],
                'threshold': split_config['split_value'],
                'split_operator': '<='
            }
        elif config_type == 'categorical_binary':
            return {
                'feature': split_config['feature'],
                'split_type': 'categorical',
                'left_categories': split_config['left_categories'],
                'right_categories': split_config['right_categories']
            }
        elif config_type == 'numeric_multi_bin':
            return {
                'feature': split_config['feature'],
                'split_type': 'numeric_multi_bin',
                'thresholds': split_config['thresholds'],
                'num_bins': split_config['num_bins']
            }
        elif config_type == 'categorical_multi_bin':
            return {
                'feature': split_config['feature'],
                'split_type': 'categorical_multi_bin',
                'split_categories': split_config['split_categories']
            }
        else:
            return split_config
            
    def _analyze_feature_characteristics(self, data: pd.Series) -> dict:
        """ENHANCED: Analyze feature characteristics for intelligent auto-recognition"""
        try:
            analysis = {
                'is_numeric': False,
                'type': 'Unknown',
                'stats_text': 'No data',
                'icon': '❓',
                'color': '#666',
                'is_highly_skewed': False,
                'has_outliers': False
            }
            
            if len(data) == 0:
                return analysis
                
            is_numeric = pd.api.types.is_numeric_dtype(data)
            
            if is_numeric:
                analysis['is_numeric'] = True
                analysis['type'] = 'Numerical'
                analysis['icon'] = '🔢'
                analysis['color'] = '#28a745'
                
                data_min, data_max = data.min(), data.max()
                data_mean = data.mean()
                data_std = data.std()
                unique_count = data.nunique()
                
                try:
                    skewness = data.skew()
                    analysis['is_highly_skewed'] = abs(skewness) > 1.5
                except:
                    skewness = 0
                    
                try:
                    q1, q3 = data.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = ((data < lower_bound) | (data > upper_bound)).sum()
                    analysis['has_outliers'] = outliers > len(data) * 0.05  # More than 5% outliers
                except:
                    analysis['has_outliers'] = False
                    
                if data_max == data_min:
                    analysis['stats_text'] = f"Constant value: {data_min:.3f}"
                    analysis['icon'] = '⚠️'
                    analysis['color'] = '#ff9800'
                else:
                    range_str = f"{data_min:.3f} to {data_max:.3f}"
                    if data_max - data_min > 1000:
                        range_str = f"{data_min:.0f} to {data_max:.0f}"
                    
                    analysis['stats_text'] = f"Range: {range_str} | μ={data_mean:.2f}, σ={data_std:.2f} | {unique_count} unique"
                    
                    quality_flags = []
                    if analysis['is_highly_skewed']:
                        quality_flags.append(f"skewed({skewness:.1f})")
                    if analysis['has_outliers']:
                        quality_flags.append(f"outliers({outliers})")
                    if unique_count < 10:
                        quality_flags.append("discrete")
                        
                    if quality_flags:
                        analysis['stats_text'] += f" | {', '.join(quality_flags)}"
                        
            else:
                analysis['is_numeric'] = False
                analysis['type'] = 'Categorical'
                analysis['icon'] = '🏷️'
                analysis['color'] = '#007bff'
                
                unique_count = data.nunique()
                total_count = len(data)
                
                try:
                    mode_values = data.mode()
                    most_common = mode_values.iloc[0] if len(mode_values) > 0 else 'N/A'
                    most_common_count = (data == most_common).sum()
                    most_common_pct = (most_common_count / total_count * 100)
                except:
                    most_common = 'N/A'
                    most_common_pct = 0
                    
                if unique_count == 2:
                    analysis['type'] = 'Binary Categorical'
                    analysis['icon'] = '⚔️'
                elif unique_count <= 5:
                    analysis['type'] = 'Low-Cardinality Categorical'
                elif unique_count > total_count * 0.8:
                    analysis['type'] = 'High-Cardinality Categorical'
                    analysis['icon'] = '📊'
                    analysis['color'] = '#6f42c1'
                    
                if most_common != 'N/A':
                    most_common_str = str(most_common)[:20] + ('...' if len(str(most_common)) > 20 else '')
                    analysis['stats_text'] = f"{unique_count} categories | Most common: '{most_common_str}' ({most_common_pct:.1f}%)"
                else:
                    analysis['stats_text'] = f"{unique_count} categories | No dominant category"
                    
                if unique_count > 50:
                    analysis['stats_text'] += " | ⚠️ High cardinality"
                elif unique_count == total_count:
                    analysis['stats_text'] += " | ⚠️ All unique (ID-like)"
                    analysis['color'] = '#dc3545'
                    
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing feature characteristics: {e}")
            return {
                'is_numeric': pd.api.types.is_numeric_dtype(data),
                'type': 'Error in Analysis',
                'stats_text': f'Analysis failed: {str(e)[:50]}',
                'icon': '❌',
                'color': '#dc3545',
                'is_highly_skewed': False,
                'has_outliers': False
            }

    def _on_numerical_bins_changed(self, bins: List[Tuple[float, float]]):
        """Handle numerical bins changed with performance optimization"""
        if not self.current_feature or not bins:
            return
            
        cache_key = f"{self.current_feature}_{hash(tuple(bins))}"
        
        try:
            from utils.performance_optimizer import get_split_quality_cache
            cache = get_split_quality_cache()
        except ImportError:
            class MockCache:
                def __init__(self):
                    self.cache = self
                    self._data = {}
                def get(self, key):
                    return self._data.get(key)
                def set(self, key, value):
                    self._data[key] = value
            cache = MockCache()
        
        cached_result = cache.get(cache_key)
        if cached_result:
            self._update_numerical_preview(cached_result)
            return
            
        QTimer.singleShot(0, lambda: self._calculate_numerical_statistics(bins, cache_key))
        
    def _calculate_numerical_statistics(self, bins: List[Tuple[float, float]], cache_key: str):
        """
        CRITICAL FIX: Calculate numerical split statistics with robust error handling
        Prevents KeyError crashes while maintaining functionality
        """
        try:
            if not self.current_feature:
                logger.error("Current feature not set")
                return
            if not self.target_column:
                logger.warning("Target column not set - skipping statistics calculation")
                return
            if self.dataset is None or len(self.dataset) == 0:
                logger.error("Dataset not available or empty")
                return
                
            if self.current_feature not in self.dataset.columns:
                logger.error(f"Feature '{self.current_feature}' not found in dataset columns: {list(self.dataset.columns)}")
                return
                
            if self.target_column not in self.dataset.columns:
                logger.error(f"Target column '{self.target_column}' not found in dataset columns: {list(self.dataset.columns)}")
                return
                
            cache = None
            try:
                from utils.performance_optimizer import get_split_quality_cache
                cache = get_split_quality_cache()
            except ImportError:
                class MockCache:
                    def __init__(self):
                        self._data = {}
                    def get(self, key):
                        return self._data.get(key)
                    def set(self, key, value):
                        self._data[key] = value
                cache = MockCache()
            except Exception as cache_error:
                logger.warning(f"Cache initialization failed: {cache_error}")
                cache = None
            
            node_data = self.get_node_data()
            if node_data is None:
                logger.warning("No node data available for statistics calculation")
                return
                
            if self.current_feature not in node_data.columns:
                logger.warning(f"Feature {self.current_feature} not found in node data")
                return
                
            if self.target_column not in node_data.columns:
                logger.warning(f"Target column {self.target_column} not found in node data")
                return
                
            feature_data = node_data[self.current_feature]
            target_data = node_data[self.target_column]
            
            valid_feature_mask = ~feature_data.isna()
            valid_feature_data = feature_data[valid_feature_mask]
            
            if len(valid_feature_data) == 0:
                logger.warning("No valid feature data available for statistics calculation")
                return
            
            bin_stats = []
            total_samples = len(node_data)  # Use full dataset count
            missing_count = (~valid_feature_mask).sum()
            
            bin_counts = []
            
            for i, (min_val, max_val) in enumerate(bins):
                try:
                    if i == len(bins) - 1:
                        mask = (valid_feature_data >= min_val) & (valid_feature_data <= max_val)
                    else:
                        mask = (valid_feature_data >= min_val) & (valid_feature_data < max_val)
                    
                    valid_count = mask.sum()
                    bin_counts.append(valid_count)
                    
                except Exception as e:
                    logger.warning(f"Error calculating bin {i}: {e}")
                    bin_counts.append(0)
            
            if missing_count > 0 and bin_counts:
                largest_bin_idx = bin_counts.index(max(bin_counts))
                bin_counts[largest_bin_idx] += missing_count
                logger.info(f"Assigned {missing_count} missing values to largest bin (bin {largest_bin_idx + 1})")
            
            for i, (min_val, max_val) in enumerate(bins):
                try:
                    final_count = bin_counts[i]
                    
                    if i == len(bins) - 1:
                        mask = valid_feature_mask & (feature_data >= min_val) & (feature_data <= max_val)
                    else:
                        mask = valid_feature_mask & (feature_data >= min_val) & (feature_data < max_val)
                    
                    if i == largest_bin_idx and missing_count > 0:
                        missing_mask = ~valid_feature_mask
                        mask = mask | missing_mask
                    
                    bin_feature_data = feature_data[mask]
                    bin_target_data = target_data[mask]
                    
                    if final_count > 0:
                        if len(bin_target_data) > 0:
                            target_counts = bin_target_data.value_counts().to_dict()
                            majority_class = bin_target_data.mode().iloc[0]
                        else:
                            target_counts = {}
                            majority_class = None
                        
                        bin_stats.append({
                            'range': (min_val, max_val),
                            'count': final_count,  # Use final count including missing values
                            'percentage': (final_count / total_samples * 100) if total_samples > 0 else 0,
                            'target_counts': target_counts,
                            'majority_class': majority_class
                        })
                    else:
                        bin_stats.append({
                            'range': (min_val, max_val),
                            'count': 0,
                            'percentage': 0.0,
                            'target_counts': {},
                            'majority_class': None
                        })
                        
                except Exception as bin_error:
                    logger.warning(f"Error calculating statistics for bin {i}: {bin_error}")
                    continue
            
            if cache:
                try:
                    cache.set(cache_key, bin_stats)
                except Exception as cache_set_error:
                    logger.warning(f"Failed to cache results: {cache_set_error}")
            
            preview_total = sum(bin_stat['count'] for bin_stat in bin_stats)
            logger.info(f"Preview total: {preview_total}, Expected: {total_samples}, Match: {preview_total == total_samples}")
            
            if hasattr(self, '_update_numerical_preview'):
                self._update_numerical_preview(bin_stats)
            
            logger.info(f"Successfully calculated statistics for {len(bin_stats)} bins")
            
        except Exception as e:
            logger.error(f"Error calculating numerical statistics: {e}")
            try:
                if hasattr(self, 'bin_details_table') and self.bin_details_table is not None:
                    try:
                        self.bin_details_table.setRowCount(0)
                    except AttributeError:
                        pass  # Widget may have been removed
                if hasattr(self, '_clear_statistics_displays'):
                    self._clear_statistics_displays()
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}")
            
        
    def _calculate_categorical_statistics(self, bin_configs: List[Dict[str, Any]], cache_key: str):
        """Calculate categorical split statistics with caching"""
        try:
            node_data = self.get_node_data()
            if node_data is None:
                logger.warning("No node data available for categorical statistics calculation")
                return
                
            if self.current_feature not in node_data.columns:
                logger.warning(f"Feature {self.current_feature} not found in node data")
                return
                
            if self.target_column not in node_data.columns:
                logger.warning(f"Target column {self.target_column} not found in node data")
                return
                
            feature_data = node_data[self.current_feature]
            target_data = node_data[self.target_column]
            total_samples = len(node_data)
            
            missing_mask = feature_data.isna()
            missing_count = missing_mask.sum()
            
            logger.info(f"Categorical statistics: {total_samples} total samples, {missing_count} missing values")
            
            bin_counts = []
            bin_stats = []
            valid_feature_mask = ~missing_mask
            
            for i, config in enumerate(bin_configs):
                categories = config['categories']
                valid_mask = valid_feature_mask & feature_data.isin(categories)
                valid_count = valid_mask.sum()
                bin_counts.append(valid_count)
            
            largest_bin_idx = 0
            if missing_count > 0 and bin_counts:
                largest_bin_idx = bin_counts.index(max(bin_counts))
                bin_counts[largest_bin_idx] += missing_count
                logger.info(f"Assigned {missing_count} missing values to largest categorical bin (bin {largest_bin_idx + 1})")
            
            for i, config in enumerate(bin_configs):
                categories = config['categories']
                mask = valid_feature_mask & feature_data.isin(categories)
                
                if i == largest_bin_idx and missing_count > 0:
                    mask = mask | missing_mask
                
                bin_data = target_data[mask]
                final_count = bin_counts[i]
                
                stats = {
                    'bin_name': config.get('bin_id', f'bin_{i}'),
                    'categories': categories,
                    'count': final_count,  # Use final count including missing values
                    'target_distribution': bin_data.value_counts().to_dict() if len(bin_data) > 0 else {}
                }
                
                if len(bin_data) > 0:
                    target_counts = bin_data.value_counts()
                    if len(target_counts) > 0:
                        max_count = target_counts.max()
                        stats['purity'] = max_count / len(bin_data)
                    else:
                        stats['purity'] = 0.0
                else:
                    stats['purity'] = 0.0
                    
                bin_stats.append(stats)
                
            try:
                from utils.performance_optimizer import get_split_quality_cache
                cache = get_split_quality_cache()
            except ImportError:
                class MockCache:
                    def __init__(self):
                        self.cache = self
                        self._data = {}
                    def get(self, key):
                        return self._data.get(key)
                    def set(self, key, value):
                        self._data[key] = value
                cache = MockCache()
            cache.set(cache_key, bin_stats)
            
            preview_total = sum(stat['count'] for stat in bin_stats)
            logger.info(f"Categorical preview total: {preview_total}, Expected: {total_samples}, Match: {preview_total == total_samples}")
            
            self._update_categorical_preview(bin_stats)
            
        except Exception as e:
            logger.error(f"Error calculating categorical statistics: {e}")

    def _update_numerical_preview(self, bin_stats: List[Dict[str, Any]]):
        """Update preview for numerical bins"""
        pass

    def _update_categorical_preview(self, bin_stats: List[Dict[str, Any]]):
        """Update preview for categorical bins"""
        pass





            
    def _enable_numeric_controls(self, enable: bool):
        """Enable/disable controls specific to numeric features"""
        try:
            if hasattr(self, 'create_equal_width_btn'):
                self.create_equal_width_btn.setEnabled(enable)
            if hasattr(self, 'create_equal_frequency_btn'):
                self.create_equal_frequency_btn.setEnabled(enable)
            if hasattr(self, 'create_quantile_btn'):
                self.create_quantile_btn.setEnabled(enable)
                
        except Exception as e:
            logger.warning(f"Error configuring numeric controls: {e}")

    def populate_from_initial_split(self):
        """Populate dialog with initial split configuration"""
        if not self.initial_split:
            return
            
        try:
            self._initializing = True
            
            feature = self.initial_split.get('feature')
            split_type = self.initial_split.get('split_type', 'numeric')
            
            if not feature or feature not in self.dataset.columns:
                logger.warning(f"Feature {feature} not found in dataset")
                return
            
            feature_index = self.feature_combo.findText(feature)
            if feature_index >= 0:
                self.feature_combo.setCurrentIndex(feature_index)
                self.on_feature_changed(feature)
            
            if split_type == 'numeric':
                split_value = self.initial_split.get('split_value')
                if split_value is not None and hasattr(self, 'numerical_bin_manager'):
                    feature_data = self.get_safe_feature_data(feature)
                    if feature_data is not None and len(feature_data) > 0:
                        data_min = feature_data.min()
                        data_max = feature_data.max()
                    else:
                        logger.error(f"Cannot calculate min/max for feature '{feature}' - no valid data")
                        return
                    
                    bins = [(data_min, split_value), (split_value, data_max)]
                    self.numerical_bin_manager.bins = bins
                    self.numerical_bin_manager._populate_bins_table()
                    
            elif split_type == 'categorical':
                left_categories = self.initial_split.get('left_categories', [])
                right_categories = self.initial_split.get('right_categories', [])
                
                if not hasattr(self, 'categorical_bin_manager') or self.categorical_bin_manager is None:
                    from ui.components.categorical_bin_manager import CategoricalBinManager
                    self.categorical_bin_manager = CategoricalBinManager(
                        self.dataset, feature, self.target_column, parent=self
                    )
                    logger.info(f"Created categorical bin manager for {feature}")
                
                if not right_categories and left_categories:
                    feature_data = self.get_safe_feature_data(feature)
                    if feature_data is not None:
                        all_categories = feature_data.unique().tolist()
                    else:
                        logger.error(f"Cannot get categories for feature '{feature}' - no valid data")
                        all_categories = []
                    right_categories = [cat for cat in all_categories if cat not in left_categories]
                
                if left_categories or right_categories:
                    node_data = self.get_node_data()
                    if node_data is not None and feature in node_data.columns:
                        feature_data = node_data[feature]
                        target_data = node_data[self.target_column] if self.target_column in node_data.columns else None
                        self.categorical_bin_manager.load_data(feature_data, target_data)
                    
                    bin_configs = []
                    if left_categories:
                        bin_configs.append({'categories': left_categories})
                    if right_categories:
                        bin_configs.append({'categories': right_categories})
                    
                    logger.info(f"Setting categorical bins: {len(bin_configs)} bins with categories {left_categories} | {right_categories}")
                    self.categorical_bin_manager.set_bin_configuration(bin_configs)
                    
                    if hasattr(self.categorical_bin_manager, 'bins_changed'):
                        self.categorical_bin_manager.bins_changed.emit()
                else:
                    logger.warning(f"No valid categories found for split: left={left_categories}, right={right_categories}")
            
            logger.info(f"Populated dialog with initial split for {feature}")
            
        except Exception as e:
            logger.error(f"Error populating from initial split: {e}")
        finally:
            self._initializing = False
    
    def _populate_existing_categorical_split(self):
        """Helper method to populate existing categorical split data"""
        try:
            if not self.initial_split or not self.categorical_bin_manager:
                return
                
            left_categories = self.initial_split.get('left_categories', [])
            right_categories = self.initial_split.get('right_categories', [])
            
            if left_categories or right_categories:
                if not right_categories and left_categories:
                    feature_data = self.get_safe_feature_data(self.current_feature)
                    if feature_data is not None:
                        all_categories = feature_data.unique().tolist()
                    else:
                        logger.error(f"Cannot get categories for feature '{self.current_feature}' - no valid data")
                        all_categories = []
                    right_categories = [cat for cat in all_categories if cat not in left_categories]
                
                bin_configs = []
                if left_categories:
                    bin_configs.append({'categories': left_categories})
                if right_categories:
                    bin_configs.append({'categories': right_categories})
                
                self.categorical_bin_manager.set_bin_configuration(bin_configs)
                logger.info(f"Populated existing categorical split: {len(bin_configs)} bins")
                
        except Exception as e:
            logger.error(f"Error populating existing categorical split: {e}")