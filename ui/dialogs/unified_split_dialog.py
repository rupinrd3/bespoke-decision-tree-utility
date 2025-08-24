#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified Split Dialog System
Modern, unified dialog for all split operations with consistent UX
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QLabel, 
    QPushButton, QTableWidget, QTableWidgetItem, QTextEdit, QProgressBar,
    QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox, QSpinBox,
    QCheckBox, QSplitter, QFrame, QHeaderView, QAbstractItemView,
    QScrollArea, QGridLayout, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, QObject
from PyQt5.QtGui import QFont, QColor, QPalette

from models.node import TreeNode
from models.decision_tree import BespokeDecisionTree
from ui.base_detail_window import BaseDetailWindow

logger = logging.getLogger(__name__)

class SplitFinderWorker(QObject):
    """Worker thread for finding splits to avoid blocking UI"""
    
    splitsFound = pyqtSignal(list)  # List of split candidates
    progressUpdate = pyqtSignal(int, str)  # progress, message
    finished = pyqtSignal()
    error = pyqtSignal(str)  # error message
    
    def __init__(self, model, node, parent=None):
        super().__init__(parent)
        self.model = model
        self.node = node
        self.is_cancelled = False
        
    def find_splits(self):
        """Find optimal splits for the node"""
        try:
            self.progressUpdate.emit(10, "Analyzing node data...")
            
            if self.is_cancelled:
                return
                
            self.progressUpdate.emit(30, "Finding split candidates...")
            splits = self.model.find_split_for_node(self.node.node_id)
            
            if self.is_cancelled:
                return
                
            self.progressUpdate.emit(70, "Evaluating split quality...")
            
            processed_splits = self._process_splits(splits)
            
            self.progressUpdate.emit(100, "Complete")
            self.splitsFound.emit(processed_splits)
            
        except Exception as e:
            logger.error(f"Error finding splits: {e}", exc_info=True)
            self.error.emit(str(e))
        finally:
            self.finished.emit()
            
    def _process_splits(self, splits: List[Dict]) -> List[Dict]:
        """Process and enhance split information"""
        processed = []
        
        for i, split in enumerate(splits):
            if self.is_cancelled:
                break
                
            gain = split.get('gain', 0.0)
            samples = split.get('samples', 0)
            
            quality = min(gain * 2, 1.0) if gain > 0 else 0
            
            processed_split = {
                'rank': i + 1,
                'feature': split.get('feature', 'Unknown'),
                'threshold': split.get('value', 0.0),
                'condition': split.get('split_desc', 'Unknown condition'),
                'gain': gain,
                'quality': quality,
                'samples_left': split.get('split_info', {}).get('left_samples', 0),
                'samples_right': split.get('split_info', {}).get('right_samples', 0),
                'original_split': split
            }
            
            processed.append(processed_split)
            
        return processed
        
    def cancel(self):
        """Cancel the split finding operation"""
        self.is_cancelled = True


class ModernSplitCandidateTable(QTableWidget):
    """Enhanced table for displaying split candidates with modern styling"""
    
    candidateSelected = pyqtSignal(dict)  # split_info
    candidateDoubleClicked = pyqtSignal(dict)  # split_info
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.split_data = []
        self.setup_table()
        self.apply_styling()
        
    def setup_table(self):
        """Setup table structure and behavior"""
        headers = [
            "Rank", "Feature", "Threshold", "Condition", 
            "Information Gain", "Quality Score", "Left Samples", "Right Samples"
        ]
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        
        self.setSortingEnabled(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        
        self.itemSelectionChanged.connect(self.on_selection_changed)
        self.itemDoubleClicked.connect(self.on_item_double_clicked)
        
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Rank
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Feature
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Threshold
        header.setSectionResizeMode(3, QHeaderView.Stretch)           # Condition
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Gain
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)  # Quality
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)  # Left
        header.setSectionResizeMode(7, QHeaderView.ResizeToContents)  # Right
        
    def apply_styling(self):
        """Apply modern table styling"""
        self.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                gridline-color: #f1f5f9;
                font-size: 11px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QHeaderView::section {
                background-color: #f8fafc;
                border: none;
                border-bottom: 1px solid #e2e8f0;
                border-right: 1px solid #e2e8f0;
                padding: 8px 12px;
                font-weight: 600;
                color: #374151;
                font-size: 10px;
            }
            QTableWidget::item {
                padding: 8px 12px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: #dbeafe;
                color: #1e293b;
            }
            QTableWidget::item:hover {
                background-color: #f1f5f9;
            }
        """)
        
    def populate_candidates(self, candidates: List[Dict]):
        """Populate table with split candidates"""
        self.split_data = candidates
        self.setRowCount(len(candidates))
        
        for row, candidate in enumerate(candidates):
            rank_item = QTableWidgetItem(f"#{candidate.get('rank', row+1)}")
            rank_item.setTextAlignment(Qt.AlignCenter)
            if row == 0:  # Best split
                rank_item.setBackground(QColor("#dcfce7"))
            self.setItem(row, 0, rank_item)
            
            feature_item = QTableWidgetItem(str(candidate.get('feature', 'Unknown')))
            self.setItem(row, 1, feature_item)
            
            threshold = candidate.get('threshold', 0.0)
            threshold_item = QTableWidgetItem(f"{threshold:.4f}")
            threshold_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.setItem(row, 2, threshold_item)
            
            condition_item = QTableWidgetItem(str(candidate.get('condition', 'Unknown')))
            self.setItem(row, 3, condition_item)
            
            gain = candidate.get('gain', 0.0)
            gain_item = QTableWidgetItem(f"{gain:.6f}")
            gain_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.setItem(row, 4, gain_item)
            
            quality = candidate.get('quality', 0.0)
            quality_item = QTableWidgetItem(f"{quality:.3f}")
            quality_item.setTextAlignment(Qt.AlignCenter)
            quality_item.setBackground(QColor(self.get_quality_color(quality)))
            self.setItem(row, 5, quality_item)
            
            left_samples = candidate.get('samples_left', 0)
            right_samples = candidate.get('samples_right', 0)
            
            left_item = QTableWidgetItem(str(left_samples))
            left_item.setTextAlignment(Qt.AlignCenter)
            self.setItem(row, 6, left_item)
            
            right_item = QTableWidgetItem(str(right_samples))
            right_item.setTextAlignment(Qt.AlignCenter)
            self.setItem(row, 7, right_item)
            
        if candidates:
            self.selectRow(0)
            
    def get_quality_color(self, score: float) -> str:
        """Get background color based on quality score"""
        if score >= 0.8:
            return "#dcfce7"  # Green for excellent
        elif score >= 0.6:
            return "#fef3c7"  # Yellow for good
        elif score >= 0.4:
            return "#fed7aa"  # Orange for fair
        else:
            return "#fecaca"  # Red for poor
            
    def on_selection_changed(self):
        """Handle selection change"""
        current_row = self.currentRow()
        if 0 <= current_row < len(self.split_data):
            self.candidateSelected.emit(self.split_data[current_row])
            
    def on_item_double_clicked(self, item):
        """Handle double-click on item"""
        row = item.row()
        if 0 <= row < len(self.split_data):
            self.candidateDoubleClicked.emit(self.split_data[row])
            
    def get_selected_candidate(self) -> Optional[Dict]:
        """Get currently selected candidate"""
        current_row = self.currentRow()
        if 0 <= current_row < len(self.split_data):
            return self.split_data[current_row]
        return None


class UnifiedSplitDialog(BaseDetailWindow):
    """Modern, unified dialog for all split operations"""
    
    splitApplied = pyqtSignal(str, dict)  # node_id, split_config
    splitCanceled = pyqtSignal()
    
    def __init__(self, mode="find", node=None, model=None, dataset=None, parent=None):
        super().__init__(parent)
        
        self.mode = mode  # "find", "edit", "manual"
        self.node = node
        self.model = model
        self.dataset = dataset
        self.current_split_config = {}
        self.split_worker = None
        self.split_thread = None
        
        self.setup_split_dialog()
        self.apply_modern_styling()
        
        logger.info(f"UnifiedSplitDialog initialized in {mode} mode for node {node.node_id if node else 'None'}")
        
    def setup_split_dialog(self):
        """Setup dialog based on mode"""
        if self.mode == "find":
            self.set_window_title("🔍 Find Optimal Split")
            self.setup_find_split_interface()
        elif self.mode == "edit":
            self.set_window_title("✂️ Edit Split")
            self.setup_edit_split_interface()
        elif self.mode == "manual":
            self.set_window_title("🎯 Manual Split Configuration")
            self.setup_manual_split_interface()
            
        self.add_action_button("apply", "✅ Apply Split", self.apply_selected_split, "success")
        self.add_action_button("cancel", "❌ Cancel", self.reject, "secondary")
        
        self.enable_action_button("apply", False)
        
    def setup_find_split_interface(self):
        """Setup interface for finding optimal splits"""
        main_splitter = QSplitter(Qt.Vertical)
        
        candidates_group = QGroupBox("🎯 Split Candidates")
        candidates_layout = QVBoxLayout(candidates_group)
        
        instructions = QLabel(
            "💡 <b>How to use:</b> The table below shows potential splits ranked by quality. "
            "Higher information gain and quality scores indicate better splits. "
            "Select a split to see detailed preview below."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #374151; margin: 8px; padding: 8px; background-color: #f8fafc; border-radius: 4px;")
        candidates_layout.addWidget(instructions)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel()
        self.progress_label.setVisible(False)
        candidates_layout.addWidget(self.progress_label)
        candidates_layout.addWidget(self.progress_bar)
        
        self.candidates_table = ModernSplitCandidateTable()
        self.candidates_table.candidateSelected.connect(self.on_candidate_selected)
        self.candidates_table.candidateDoubleClicked.connect(self.on_candidate_double_clicked)
        candidates_layout.addWidget(self.candidates_table)
        
        main_splitter.addWidget(candidates_group)
        
        preview_group = QGroupBox("📊 Split Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_text = QTextEdit()
        self.preview_text.setMaximumHeight(200)
        self.preview_text.setReadOnly(True)
        self.preview_text.setHtml("<i>Select a split above to see detailed preview...</i>")
        preview_layout.addWidget(self.preview_text)
        
        main_splitter.addWidget(preview_group)
        
        main_splitter.setSizes([400, 200])
        
        self.add_content_widget(main_splitter)
        
        self.find_splits_automatically()
        
    def setup_edit_split_interface(self):
        """Setup interface for editing existing splits"""
        form_group = QGroupBox("✂️ Split Configuration")
        form_layout = QFormLayout(form_group)
        
        self.feature_combo = QComboBox()
        self.feature_combo.addItems(self.get_available_features())
        form_layout.addRow("Feature:", self.feature_combo)
        
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(-999999, 999999)
        self.threshold_spin.setDecimals(6)
        form_layout.addRow("Threshold:", self.threshold_spin)
        
        self.operator_combo = QComboBox()
        self.operator_combo.addItems(['<=', '>', '==', '!='])
        form_layout.addRow("Operator:", self.operator_combo)
        
        self.add_content_widget(form_group)
        
        self.load_current_split()
        
        self.enable_action_button("apply", True)
        
    def setup_manual_split_interface(self):
        """Setup interface for manual split configuration"""
        tab_widget = QTabWidget()
        
        feature_tab = QWidget()
        feature_layout = QFormLayout(feature_tab)
        
        self.manual_feature_combo = QComboBox()
        self.manual_feature_combo.addItems(self.get_available_features())
        self.manual_feature_combo.currentTextChanged.connect(self.on_manual_feature_changed)
        feature_layout.addRow("Select Feature:", self.manual_feature_combo)
        
        tab_widget.addTab(feature_tab, "Feature Selection")
        
        config_tab = QWidget()
        config_layout = QFormLayout(config_tab)
        
        self.manual_threshold_spin = QDoubleSpinBox()
        self.manual_threshold_spin.setRange(-999999, 999999)
        self.manual_threshold_spin.setDecimals(6)
        config_layout.addRow("Threshold:", self.manual_threshold_spin)
        
        self.manual_operator_combo = QComboBox()
        self.manual_operator_combo.addItems(['<=', '>', '==', '!='])
        config_layout.addRow("Operator:", self.manual_operator_combo)
        
        tab_widget.addTab(config_tab, "Split Configuration")
        
        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        
        self.manual_preview_text = QTextEdit()
        self.manual_preview_text.setReadOnly(True)
        preview_layout.addWidget(self.manual_preview_text)
        
        refresh_preview_btn = QPushButton("🔄 Refresh Preview")
        refresh_preview_btn.clicked.connect(self.refresh_manual_preview)
        preview_layout.addWidget(refresh_preview_btn)
        
        tab_widget.addTab(preview_tab, "Preview")
        
        self.add_content_widget(tab_widget)
        
        self.enable_action_button("apply", True)
        
    def apply_modern_styling(self):
        """Apply consistent modern styling"""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: 600;
                font-size: 12px;
                color: #1e293b;
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                background-color: #ffffff;
                padding-top: 16px;
                margin-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
                background-color: #f8fafc;
            }
            QComboBox, QDoubleSpinBox, QSpinBox {
                border: 1px solid #d1d5db;
                border-radius: 4px;
                padding: 6px 8px;
                background-color: #ffffff;
                font-size: 11px;
            }
            QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {
                border-color: #3b82f6;
                outline: none;
            }
            QTextEdit {
                border: 1px solid #d1d5db;
                border-radius: 6px;
                background-color: #ffffff;
                font-size: 11px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QProgressBar {
                border: 1px solid #e2e8f0;
                border-radius: 4px;
                background-color: #f1f5f9;
                height: 20px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 3px;
            }
        """)
        
    def find_splits_automatically(self):
        """Find optimal splits for the node"""
        if not self.model or not self.node:
            self.show_error_state("No model or node available for split finding")
            return
            
        self.show_progress("Finding optimal splits...")
        
        self.split_worker = SplitFinderWorker(self.model, self.node)
        self.split_thread = QThread()
        
        self.split_worker.moveToThread(self.split_thread)
        
        self.split_thread.started.connect(self.split_worker.find_splits)
        self.split_worker.splitsFound.connect(self.on_splits_found)
        self.split_worker.progressUpdate.connect(self.update_progress)
        self.split_worker.error.connect(self.on_split_error)
        self.split_worker.finished.connect(self.split_thread.quit)
        self.split_worker.finished.connect(self.split_worker.deleteLater)
        self.split_thread.finished.connect(self.split_thread.deleteLater)
        
        self.split_thread.start()
        
    def show_progress(self, message: str):
        """Show progress indicator"""
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate
        if hasattr(self, 'progress_label'):
            self.progress_label.setText(message)
            self.progress_label.setVisible(True)
            
    def update_progress(self, value: int, message: str):
        """Update progress indicator"""
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(value)
        if hasattr(self, 'progress_label'):
            self.progress_label.setText(message)
            
    def hide_progress(self):
        """Hide progress indicator"""
        if hasattr(self, 'progress_bar'):
            self.progress_bar.setVisible(False)
        if hasattr(self, 'progress_label'):
            self.progress_label.setVisible(False)
            
    def on_splits_found(self, splits: List[Dict]):
        """Handle splits found by worker"""
        self.hide_progress()
        
        if splits:
            self.candidates_table.populate_candidates(splits)
            logger.info(f"Found {len(splits)} split candidates")
        else:
            self.show_empty_state("No optimal splits found for this node")
            
    def on_split_error(self, error_message: str):
        """Handle error from split worker"""
        self.hide_progress()
        self.show_error_state(f"Error finding splits: {error_message}")
        logger.error(f"Split finding error: {error_message}")
        
    def on_candidate_selected(self, candidate: Dict):
        """Handle candidate selection"""
        self.current_split_config = candidate
        self.update_split_preview(candidate)
        self.enable_action_button("apply", True)
        
    def on_candidate_double_clicked(self, candidate: Dict):
        """Handle candidate double-click (apply immediately)"""
        self.current_split_config = candidate
        self.apply_selected_split()
        
    def update_split_preview(self, candidate: Dict):
        """Update split preview display"""
        if not hasattr(self, 'preview_text'):
            return
            
        original_split = candidate.get('original_split', {})
        split_info = original_split.get('split_info', {})
        
        html = f"""
        <h3>🎯 Split Preview</h3>
        <table border="0" cellpadding="6" style="width: 100%; font-family: 'Segoe UI', Arial, sans-serif;">
        <tr><td><b>Feature:</b></td><td>{candidate.get('feature', 'Unknown')}</td></tr>
        <tr><td><b>Threshold:</b></td><td>{candidate.get('threshold', 0.0):.6f}</td></tr>
        <tr><td><b>Condition:</b></td><td>{candidate.get('condition', 'Unknown')}</td></tr>
        <tr><td><b>Information Gain:</b></td><td>{candidate.get('gain', 0.0):.6f}</td></tr>
        <tr><td><b>Quality Score:</b></td><td>{candidate.get('quality', 0.0):.3f}</td></tr>
        </table>
        
        <h4>📊 Child Node Distribution:</h4>
        <table border="1" cellpadding="6" style="border-collapse: collapse; width: 100%; font-size: 11px;">
        <tr style="background-color: #f8fafc; font-weight: 600;">
            <th>Child Node</th>
            <th>Sample Count</th>
            <th>Class Distribution</th>
        </tr>
        <tr>
            <td><b>Left Child</b><br/><small>(condition = True)</small></td>
            <td style="text-align: center;">{candidate.get('samples_left', 0)}</td>
            <td>{self.format_class_distribution(split_info.get('left_class_distribution', {}))}</td>
        </tr>
        <tr>
            <td><b>Right Child</b><br/><small>(condition = False)</small></td>
            <td style="text-align: center;">{candidate.get('samples_right', 0)}</td>
            <td>{self.format_class_distribution(split_info.get('right_class_distribution', {}))}</td>
        </tr>
        </table>
        """
        
        self.preview_text.setHtml(html)
        
    def format_class_distribution(self, distribution: Dict) -> str:
        """Format class distribution for display"""
        if not distribution:
            return "No data"
            
        formatted = []
        for class_name, count in distribution.items():
            formatted.append(f"{class_name}: {count}")
            
        return "<br/>".join(formatted)
        
    def apply_selected_split(self):
        """Apply the selected or configured split"""
        if not self.current_split_config:
            QMessageBox.warning(self, "No Split Selected", "Please select or configure a split to apply.")
            return
            
        try:
            if self.mode == "find":
                original_split = self.current_split_config.get('original_split')
                if original_split:
                    success = self.model.apply_manual_split(self.node.node_id, original_split.get('split_info', {}))
                else:
                    success = False
            else:
                split_config = self.get_manual_split_config()
                success = self.model.apply_manual_split(self.node.node_id, split_config)
                
            if success:
                self.splitApplied.emit(self.node.node_id, self.current_split_config)
                
                QMessageBox.information(
                    self, "Split Applied Successfully!",
                    f"The split has been applied to node {self.node.node_id}.\n\n"
                    f"Feature: {self.current_split_config.get('feature', 'Unknown')}\n"
                    f"Threshold: {self.current_split_config.get('threshold', 0.0):.6f}\n\n"
                    "Child nodes have been created and the tree has been updated."
                )
                
                self.accept()
            else:
                QMessageBox.warning(
                    self, "Split Failed",
                    f"Could not apply the split to node {self.node.node_id}. "
                    "Please check the configuration and try again."
                )
                
        except Exception as e:
            logger.error(f"Error applying split: {e}", exc_info=True)
            QMessageBox.critical(
                self, "Split Error",
                f"An error occurred while applying the split:\n{str(e)}"
            )
            
    def get_manual_split_config(self) -> Dict:
        """Get manually configured split parameters"""
        if self.mode == "edit":
            return {
                'feature': self.feature_combo.currentText(),
                'value': self.threshold_spin.value(),
                'operator': self.operator_combo.currentText()
            }
        elif self.mode == "manual":
            return {
                'feature': self.manual_feature_combo.currentText(),
                'value': self.manual_threshold_spin.value(),
                'operator': self.manual_operator_combo.currentText()
            }
        return {}
        
    def get_available_features(self) -> List[str]:
        """Get list of available features for splitting"""
        if self.dataset is not None:
            return list(self.dataset.columns)
        elif self.model and hasattr(self.model, 'feature_names'):
            return self.model.feature_names
        else:
            return ["Feature1", "Feature2", "Feature3"]  # Fallback
            
    def load_current_split(self):
        """Load current split configuration for editing"""
        if self.node and not self.node.is_terminal:
            if hasattr(self, 'feature_combo') and self.node.split_feature:
                index = self.feature_combo.findText(self.node.split_feature)
                if index >= 0:
                    self.feature_combo.setCurrentIndex(index)
                    
            if hasattr(self, 'threshold_spin') and self.node.split_value is not None:
                self.threshold_spin.setValue(self.node.split_value)
                
            if hasattr(self, 'operator_combo'):
                operator = getattr(self.node, 'split_operator', '<=')
                index = self.operator_combo.findText(operator)
                if index >= 0:
                    self.operator_combo.setCurrentIndex(index)
                    
    def on_manual_feature_changed(self, feature_name: str):
        """Handle manual feature selection change"""
        self.refresh_manual_preview()
        
    def refresh_manual_preview(self):
        """Refresh manual split preview"""
        if hasattr(self, 'manual_preview_text'):
            config = self.get_manual_split_config()
            
            preview_html = f"""
            <h3>🎯 Manual Split Configuration</h3>
            <p><b>Feature:</b> {config.get('feature', 'Not selected')}</p>
            <p><b>Threshold:</b> {config.get('value', 0.0):.6f}</p>
            <p><b>Operator:</b> {config.get('operator', 'Not selected')}</p>
            
            <h4>Split Condition:</h4>
            <p><code>{config.get('feature', 'Feature')} {config.get('operator', 'op')} {config.get('value', 0.0):.6f}</code></p>
            
            <p><i>Note: This preview shows the configured split condition. 
            Apply the split to see actual data distribution.</i></p>
            """
            
            self.manual_preview_text.setHtml(preview_html)
            
    def closeEvent(self, event):
        """Handle dialog close event"""
        if self.split_worker:
            self.split_worker.cancel()
            
        if self.split_thread and self.split_thread.isRunning():
            self.split_thread.quit()
            self.split_thread.wait(3000)  # Wait up to 3 seconds
            
        super().closeEvent(event)