#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Find Split Dialog - Dedicated interface for finding optimal splits
Shows ranked split candidates and allows direct application or manual editing

[SplitFinderWorker.__init__ -> Initializes the SplitFinderWorker -> dependent functions are None]
[SplitFinderWorker.run -> Finds optimal splits in the background -> dependent functions are _find_splits_via_model]
[SplitFinderWorker._find_splits_via_model -> Delegates to model.find_split_for_node with criterion overrides -> dependent functions are _resolve_override]
[SplitFinderWorker._resolve_override -> Normalizes criterion selections from the UI -> dependent functions are None]
[SplitFinderWorker.cancel -> Cancels the search -> dependent functions are None]
[FindSplitDialog.__init__ -> Initializes the FindSplitDialog -> dependent functions are init_ui, start_split_search]
[FindSplitDialog.init_ui -> Initializes the user interface -> dependent functions are restart_search, on_split_selection_changed, apply_selected_split, reject]
[FindSplitDialog.start_split_search -> Starts the split search in the background -> dependent functions are on_splits_found, on_search_error]
[FindSplitDialog.restart_search -> Restarts the split search -> dependent functions are start_split_search]
[FindSplitDialog.on_splits_found -> Handles found split candidates -> dependent functions are populate_splits_table]
[FindSplitDialog.on_search_error -> Handles search errors -> dependent functions are None]
[FindSplitDialog._populate_splits_table -> Populates the splits table with candidates -> dependent functions are None]
[FindSplitDialog.on_split_selection_changed -> Handles split selection change -> dependent functions are update_split_preview, apply_button.setEnabled]
[FindSplitDialog.update_split_preview -> Updates the split preview panel -> dependent functions are update_distribution_info]
[FindSplitDialog.update_distribution_info -> Updates the variable distribution information -> dependent functions are None]
[FindSplitDialog.clear_split_preview -> Clears the split preview panel -> dependent functions are None]
[FindSplitDialog.get_node_data -> Gets the data for the current node -> dependent functions are None]
[FindSplitDialog.apply_selected_split -> Applies the selected split -> dependent functions are None]
[FindSplitDialog.closeEvent -> Handles the dialog close event -> dependent functions are None]
"""

import logging
from copy import deepcopy
from typing import Dict, Any, List, Optional

import pandas as pd
from models.decision_tree import SplitCriterion
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QComboBox, QSpinBox, QProgressBar, QMessageBox,
    QHeaderView, QTabWidget, QWidget, QTextEdit, QSplitter, QFrame,
    QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QFont, QColor

logger = logging.getLogger(__name__)


class SplitFinderWorker(QThread):
    """Background worker that delegates split discovery to the model."""

    split_found = pyqtSignal(list)  # List of split candidates
    progress_updated = pyqtSignal(int)  # Progress percentage
    error_occurred = pyqtSignal(str)  # Error message

    def __init__(
        self,
        model,
        node_id: str,
        criterion_override: Optional[SplitCriterion] = None,
        split_overrides: Optional[Dict[str, Any]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.model = model
        self.node_id = node_id
        self.criterion_override = criterion_override
        self.split_overrides = split_overrides or {}
        self.is_cancelled = False

    def run(self):
        """Execute the model's split finder without blocking the UI."""
        try:
            self.progress_updated.emit(5)

            if not hasattr(self.model, "find_split_for_node"):
                self.error_occurred.emit(
                    "Model does not support finding splits for nodes."
                )
                return

            node = self.model.get_node(self.node_id)
            if not node:
                self.error_occurred.emit(f"Node {self.node_id} not found")
                return

            self.progress_updated.emit(25)
            splits = self._find_splits_via_model()
            if self.is_cancelled:
                return

            self.progress_updated.emit(100)
            self.split_found.emit(splits or [])
        except Exception as exc:
            logger.error("Error finding splits via model: %s", exc, exc_info=True)
            self.error_occurred.emit(str(exc))

    def _find_splits_via_model(self) -> List[Dict[str, Any]]:
        """Call the model's split finder with optional overrides."""
        original_criterion = getattr(self.model, "criterion", None)
        override = self._resolve_override()
        split_cfg_backup = None
        split_cfg_ref = None

        if self.split_overrides:
            model_config = getattr(self.model, "config", None)
            if isinstance(model_config, dict):
                decision_cfg = model_config.setdefault("decision_tree", {})
                split_cfg_ref = decision_cfg.setdefault("split_finding", {})
                split_cfg_backup = deepcopy(split_cfg_ref)
                split_cfg_ref.update(self.split_overrides)
                logger.debug(
                    "Applied split overrides for worker: %s", self.split_overrides
                )

        try:
            if override is not None and original_criterion != override:
                logger.debug(
                    "Temporarily overriding model criterion to %s for split search",
                    override,
                )
                self.model.criterion = override

            return self.model.find_split_for_node(self.node_id) or []
        finally:
            if (
                override is not None
                and original_criterion is not None
                and self.model.criterion != original_criterion
            ):
                logger.debug("Restoring model criterion to %s", original_criterion)
                self.model.criterion = original_criterion

            if split_cfg_backup is not None and split_cfg_ref is not None:
                split_cfg_ref.clear()
                split_cfg_ref.update(split_cfg_backup)
                logger.debug("Restored original split-finding configuration")

    def _resolve_override(self) -> Optional[SplitCriterion]:
        """Normalize the criterion override to a SplitCriterion value."""
        override = self.criterion_override
        if isinstance(override, SplitCriterion):
            return override
        if isinstance(override, str):
            try:
                return SplitCriterion(override)
            except ValueError:
                return None
        return None

    def cancel(self):
        """Cancel the search."""
        self.is_cancelled = True
class FindSplitDialog(QDialog):
    """Dialog for finding and selecting optimal splits"""
    
    CRITERION_OPTIONS = [
        ("Information Gain", SplitCriterion.ENTROPY),
        ("Gini Gain", SplitCriterion.GINI),
    ]

    split_selected = pyqtSignal(dict)  # Selected split configuration
    
    def __init__(self, node, model, parent=None):
        super().__init__(parent)
        self.node = node
        self.model = model
        self.split_candidates = []
        self.worker = None
        self.config = getattr(parent, 'config', getattr(model, 'config', {}))
        self.split_overrides = self._build_split_overrides()
        
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
        for label, _ in self.CRITERION_OPTIONS:
            self.criterion_combo.addItem(label)
        self.criterion_combo.setMinimumWidth(150)

        self.criterion_combo.setCurrentText(self._criterion_label_for_model())
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
        
        selected_criterion = self._selected_criterion()
        self.worker = SplitFinderWorker(
            self.model,
            self.node.node_id,
            selected_criterion,
            self.split_overrides,
            self,
        )
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
                
                stat_value = self._extract_gain(candidate)
                stat_item = QTableWidgetItem(f"{stat_value:.4f}")
                stat_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.splits_table.setItem(row, 2, stat_item)
                
                if len(self.split_candidates) > 1:
                    max_stat = max(self._extract_gain(c) for c in self.split_candidates)
                    min_stat = min(self._extract_gain(c) for c in self.split_candidates)
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
        """Update the split preview panel."""
        try:
            split_info = self._split_payload(candidate)
            split_type = split_info.get('split_type', candidate.get('split_type', 'unknown'))
            feature = candidate.get('feature') or split_info.get('feature', 'Unknown')
            preview_lines: List[str] = []

            if candidate.get('split_desc'):
                preview_lines.append(f"Split: {candidate['split_desc']}")
                preview_lines.append("")

            coverage = split_info.get('sample_coverage')
            if coverage:
                preview_lines.append(f"Sample coverage: {coverage * 100:.1f}% of node samples")
                preview_lines.append("")

            if split_type == 'numeric':
                threshold = split_info.get('threshold', candidate.get('split_value'))
                if threshold is not None:
                    preview_lines.append(f"Condition: {feature} <= {threshold:.4f}")
                else:
                    preview_lines.append(f"Condition: {feature} (numeric)")
                preview_lines.append("")
                preview_lines.extend(self._format_branch_summary(
                    title="Left branch",
                    samples=self._safe_int(split_info.get('left_samples')),
                    metric=split_info.get('left_impurity', split_info.get('left_metric')),
                ))
                preview_lines.append("")
                preview_lines.extend(self._format_branch_summary(
                    title="Right branch",
                    samples=self._safe_int(split_info.get('right_samples')),
                    metric=split_info.get('right_impurity', split_info.get('right_metric')),
                ))

            elif split_type == 'categorical':
                left_categories = list(split_info.get('left_categories', []))
                right_categories = list(split_info.get('right_categories', []))
                preview_lines.append(f"Condition: {feature} in selected categories")
                preview_lines.append("")
                preview_lines.extend(self._format_category_branch(
                    "Left branch",
                    left_categories,
                    samples=self._safe_int(split_info.get('left_samples')),
                    metric=split_info.get('left_impurity', split_info.get('left_metric')),
                ))
                preview_lines.append("")
                preview_lines.extend(self._format_category_branch(
                    "Right branch",
                    right_categories,
                    samples=self._safe_int(split_info.get('right_samples')),
                    metric=split_info.get('right_impurity', split_info.get('right_metric')),
                ))

            else:
                preview_lines.append(f"Split type: {split_type}")
                if candidate.get('split_desc'):
                    preview_lines.append(candidate['split_desc'])

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

    def _build_split_overrides(self) -> Optional[Dict[str, Any]]:
        """Determine split configuration overrides to ensure multi-core execution."""
        config_source = self.config if isinstance(self.config, dict) else {}
        split_cfg = (
            config_source.get('decision_tree', {}).get('split_finding', {})
            if isinstance(config_source, dict) else {}
        )

        backend_value = split_cfg.get('backend')
        use_threading = split_cfg.get('use_threading', True)

        should_force_loky = False
        if isinstance(backend_value, str):
            should_force_loky = backend_value.lower() == 'loky'
        elif backend_value in (None, '') and use_threading is False:
            should_force_loky = True

        if should_force_loky:
            overrides = {
                'backend': 'loky',
                'use_threading': False,
                'n_jobs': split_cfg.get('n_jobs', -2),
            }
            logger.debug("Prepared split overrides for dialog: %s", overrides)
            return overrides

        return None

    def _format_branch_summary(self, title: str, samples: int, metric: Optional[float]) -> List[str]:
        lines = [title + ":"]
        lines.append(f"  Samples: {samples}")
        if metric is not None:
            lines.append(f"  Impurity: {metric:.4f}")
        return lines

    def _format_category_branch(
        self,
        title: str,
        categories: List[Any],
        samples: int,
        metric: Optional[float],
    ) -> List[str]:
        lines = [f"{title} ({len(categories)} categories):"]
        for cat in categories[:5]:
            lines.append(f"  - {cat}")
        if len(categories) > 5:
            lines.append(f"  ... and {len(categories) - 5} more")
        lines.append(f"  Samples: {samples}")
        if metric is not None:
            lines.append(f"  Impurity: {metric:.4f}")
        return lines

    def _split_payload(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(candidate, dict):
            return {}
        payload = candidate.get('split_info')
        if isinstance(payload, dict):
            return payload
        return candidate

    def _safe_int(self, value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _extract_gain(self, candidate: Dict[str, Any]) -> float:
        if not isinstance(candidate, dict):
            return 0.0
        gain = candidate.get('gain')
        if gain is None:
            gain = candidate.get('split_info', {}).get('gain')
        try:
            return float(gain)
        except (TypeError, ValueError):
            return 0.0

    def _criterion_label_for_model(self) -> str:
        criterion_value = getattr(self.model, 'criterion', None)
        if isinstance(criterion_value, SplitCriterion):
            criterion_key = criterion_value.value
        else:
            criterion_key = str(criterion_value).lower() if criterion_value else ''

        if criterion_key in ('entropy', 'information_gain'):
            return 'Information Gain'
        if criterion_key == 'gini':
            return 'Gini Gain'
        return 'Information Gain'

    def _selected_criterion(self) -> SplitCriterion:
        label = self.criterion_combo.currentText()
        for option_label, criterion in self.CRITERION_OPTIONS:
            if option_label == label:
                return criterion
        return SplitCriterion.ENTROPY
            
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


