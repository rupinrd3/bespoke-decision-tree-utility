#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Categorical Bin Manager - Single Widget Approach
Complete self-contained widget as per old implementation requirements

"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Optional, Any, Callable
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QGroupBox, QScrollArea, QSplitter, QFrame,
    QMessageBox, QLineEdit, QCheckBox, QSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QInputDialog, QComboBox, QProgressBar, QTextEdit, QMenu, QAction,
    QFormLayout
)
from PyQt5.QtCore import Qt, pyqtSignal, QMimeData, QTimer, QThread, pyqtSlot
from PyQt5.QtGui import QDrag, QPainter, QFont, QColor, QCursor, QPixmap

logger = logging.getLogger(__name__)


class DraggableCategoryList(QListWidget):
    """List widget supporting drag-and-drop for categories"""
    
    categories_dropped = pyqtSignal(list, int)  # categories, target_bin_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragDropMode(QListWidget.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        
    def startDrag(self, supportedActions):
        """Start drag operation"""
        selected_items = self.selectedItems()
        if not selected_items:
            return
            
        drag = QDrag(self)
        mime_data = QMimeData()
        
        categories = [item.text() for item in selected_items]
        mime_data.setText(';'.join(categories))
        drag.setMimeData(mime_data)
        
        pixmap = QPixmap(200, 30 * len(categories))
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setPen(Qt.black)
        for i, category in enumerate(categories):
            painter.drawText(10, 20 + i * 30, category[:25])  # Truncate long names
        painter.end()
        
        drag.setPixmap(pixmap)
        drag.exec_(Qt.MoveAction)
        
    def dropEvent(self, event):
        """Handle drop event"""
        if event.mimeData().hasText():
            categories = event.mimeData().text().split(';')
            for category in categories:
                items = self.findItems(category, Qt.MatchExactly)
                for item in items:
                    self.takeItem(self.row(item))
            event.accept()
        else:
            event.ignore()


class BinWidget(QGroupBox):
    """Widget representing a single categorical bin with drop support"""
    
    categories_received = pyqtSignal(list, int)  # categories, bin_id
    bin_deleted = pyqtSignal(int)  # bin_id
    
    def __init__(self, bin_id: int, title: str = None, parent=None):
        super().__init__(title or f"Bin {bin_id + 1}", parent)
        self.bin_id = bin_id
        self.categories = []
        
        self.setAcceptDrops(True)
        self.setMinimumHeight(150)
        self.setMaximumHeight(200)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the bin widget UI (simplified per requirement 22)"""
        layout = QVBoxLayout(self)
        
        self.categories_list = QListWidget()
        self.categories_list.setMaximumHeight(120)
        self.categories_list.setDragDropMode(QListWidget.DragDrop)
        self.categories_list.setWordWrap(True)
        layout.addWidget(self.categories_list)
        
        self.stats_label = QLabel("No categories")
        self.stats_label.setStyleSheet("color: gray; font-size: 9pt;")
        layout.addWidget(self.stats_label)
        
        if self.bin_id >= 2:
            self.delete_btn = QPushButton("Delete Bin")
            self.delete_btn.setMaximumHeight(25)
            self.delete_btn.clicked.connect(lambda: self.bin_deleted.emit(self.bin_id))
            layout.addWidget(self.delete_btn)
        
    def dragEnterEvent(self, event):
        """Handle drag enter"""
        if event.mimeData().hasText():
            event.accept()
        else:
            event.ignore()
            
    def dropEvent(self, event):
        """Handle drop event"""
        if event.mimeData().hasText():
            categories = event.mimeData().text().split(';')
            self.add_categories(categories)
            self.categories_received.emit(categories, self.bin_id)
            event.accept()
        else:
            event.ignore()
            
    def add_categories(self, categories: List[str]):
        """Add categories to this bin"""
        for category in categories:
            if category not in self.categories:
                self.categories.append(category)
                item = QListWidgetItem(category)
                self.categories_list.addItem(item)
                
        self.update_stats_display()
        
    def remove_categories(self, categories: List[str]):
        """Remove categories from this bin"""
        for category in categories:
            if category in self.categories:
                self.categories.remove(category)
                items = self.categories_list.findItems(category, Qt.MatchExactly)
                for item in items:
                    self.categories_list.takeItem(self.categories_list.row(item))
                    
        self.update_stats_display()
        
    def clear_categories(self):
        """Clear all categories from this bin"""
        self.categories.clear()
        self.categories_list.clear()
        self.update_stats_display()
        
    def get_categories(self) -> List[str]:
        """Get list of categories in this bin"""
        return self.categories.copy()
        
    def set_statistics(self, stats: Dict[str, Any]):
        """Set statistics for display (simplified per requirement 22)"""
        count = stats.get('count', 0)
        self.stats_label.setText(f"{count} samples")
        
    def update_stats_display(self):
        """Update statistics display (simplified per requirement 22)"""
        if self.categories:
            self.stats_label.setText(f"{len(self.categories)} categories")
        else:
            self.stats_label.setText("No categories")


class CategoricalBinManager(QWidget):
    """
    Complete self-contained categorical bin manager widget
    RESTORED: Single widget approach from old implementation
    """
    
    bins_changed = pyqtSignal()
    validation_error = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.target_data = None
        self.bins = []  # List of BinWidget objects
        self.unassigned_categories = []
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the complete UI in single widget"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(8)
        
        self.controls_widgets = self._create_controls_widgets()
        
        main_layout.addWidget(self.controls_widgets)
        
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setContentsMargins(0, 0, 0, 0)
        
        left_panel = QGroupBox("Unassigned Categories")
        left_layout = QVBoxLayout(left_panel)
        
        self.unassigned_table = QTableWidget()
        self.unassigned_table.setColumnCount(2)
        self.unassigned_table.setHorizontalHeaderLabels(["Category", "Frequency"])
        
        header = self.unassigned_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        
        self.unassigned_table.setAlternatingRowColors(True)
        self.unassigned_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.unassigned_table.setDragDropMode(QTableWidget.DragOnly)
        self.unassigned_table.setDragEnabled(True)
        
        self.unassigned_table.startDrag = self._table_start_drag
        
        left_layout.addWidget(self.unassigned_table)
        
        self.unassigned_count_label = QLabel("0 categories")
        left_layout.addWidget(self.unassigned_count_label)
        
        content_splitter.addWidget(left_panel)
        
        right_panel = QGroupBox("Bin Assignment")
        right_layout = QVBoxLayout(right_panel)
        
        self.bins_scroll_area = QScrollArea()
        self.bins_scroll_area.setWidgetResizable(True)
        self.bins_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.bins_widget = QWidget()
        self.bins_layout = QVBoxLayout(self.bins_widget)
        self.bins_layout.addStretch()  # Add stretch at the end
        
        self.bins_scroll_area.setWidget(self.bins_widget)
        right_layout.addWidget(self.bins_scroll_area)
        
        content_splitter.addWidget(right_panel)
        
        content_splitter.setSizes([500, 500])
        
        main_layout.addWidget(content_splitter)
        
    def _create_controls_widgets(self):
        """Create the control widgets for external use"""
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        
        controls_group = QGroupBox("Bin Controls")
        controls_group.setMinimumHeight(90)  # Reasonable height for form controls
        controls_form_layout = QFormLayout()
        controls_form_layout.setVerticalSpacing(8)  # Balanced vertical spacing
        controls_form_layout.setContentsMargins(10, 10, 10, 10)  # Balanced margins
        
        self.num_bins_spinner = QSpinBox()
        self.num_bins_spinner.setRange(2, 10)
        self.num_bins_spinner.setValue(2)
        self.num_bins_spinner.setMinimumHeight(25)  # Reduce height
        self.num_bins_spinner.valueChanged.connect(self.set_num_bins)
        controls_form_layout.addRow("Number of bins:", self.num_bins_spinner)
        
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search categories...")
        self.search_edit.setMinimumHeight(25)  # Reduce height
        self.search_edit.textChanged.connect(self.filter_unassigned_categories)
        controls_form_layout.addRow("Search:", self.search_edit)
        
        controls_group.setLayout(controls_form_layout)
        controls_layout.addWidget(controls_group)
        
        controls_layout.addSpacing(5)
        
        auto_group_layout = QHBoxLayout()
        
        self.auto_group_combo = QComboBox()
        self.auto_group_combo.addItems(['frequency', 'target_rate', 'similarity'])
        auto_group_layout.addWidget(QLabel("Auto-group by:"))
        auto_group_layout.addWidget(self.auto_group_combo)
        
        self.auto_group_btn = QPushButton("Auto Group")
        self.auto_group_btn.clicked.connect(self.auto_group_categories)
        auto_group_layout.addWidget(self.auto_group_btn)
        
        self.ungroup_btn = QPushButton("Ungroup All")
        self.ungroup_btn.clicked.connect(self.ungroup_all_categories)
        auto_group_layout.addWidget(self.ungroup_btn)
        
        auto_group_layout.addStretch()
        
        self.add_bin_btn = QPushButton("Add Bin")
        self.add_bin_btn.clicked.connect(self.add_bin)
        auto_group_layout.addWidget(self.add_bin_btn)
        
        controls_layout.addLayout(auto_group_layout)
        
        return controls_container
        
    def load_data(self, data: pd.Series, target_data: pd.Series = None):
        """Load categorical data for bin management"""
        try:
            self.data = data
            self.target_data = target_data
            
            missing_count = data.isna().sum()
            logger.info(f"Categorical data loaded: {len(data)} total samples, {missing_count} missing values")
            
            valid_data = data.dropna()
            unique_categories = sorted(valid_data.unique().astype(str))
            
            if missing_count > 0:
                unique_categories.append("Missing")
                logger.info(f"Added 'Missing' category for {missing_count} NaN values")
            
            self.unassigned_categories = unique_categories.copy()
            
            self.populate_unassigned_table()
            
            self.create_initial_bins()
            
            logger.info(f"Loaded {len(unique_categories)} categories for binning")
            
        except Exception as e:
            logger.error(f"Error loading categorical data: {e}")
            self.validation_error.emit(f"Failed to load data: {e}")

    def populate_unassigned_table(self, filter_text: str = ""):
        """Populate the unassigned categories table with frequencies"""
        try:
            filtered_categories = self.unassigned_categories
            if filter_text:
                filtered_categories = [cat for cat in self.unassigned_categories 
                                     if filter_text.lower() in cat.lower()]
            
            self.unassigned_table.setRowCount(len(filtered_categories))
            
            for row, category in enumerate(filtered_categories):
                cat_item = QTableWidgetItem(category)
                self.unassigned_table.setItem(row, 0, cat_item)
                
                if self.data is not None:
                    if category == "Missing":
                        count = self.data.isna().sum()  # Count NaN values for "Missing" category
                    else:
                        count = sum(self.data == category)
                    total = len(self.data)
                    frequency = count / total if total > 0 else 0
                    freq_text = f"{count} ({frequency*100:.1f}%)"
                    
                    color_intensity = int(255 * frequency)
                    color = QColor(255 - color_intensity, color_intensity, 0, 100)
                    cat_item.setBackground(color)
                    cat_item.setToolTip(f"{category}: {count} samples ({frequency*100:.1f}%)")
                else:
                    freq_text = "N/A"
                    
                freq_item = QTableWidgetItem(freq_text)
                freq_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.unassigned_table.setItem(row, 1, freq_item)
                
            self.update_unassigned_count()
            
        except Exception as e:
            logger.error(f"Error populating unassigned table: {e}")
        
    def filter_unassigned_categories(self, filter_text: str):
        """Filter unassigned categories by search text"""
        self.populate_unassigned_table(filter_text)
        
    def _table_start_drag(self, supportedActions):
        """Custom drag implementation for the unassigned table"""
        try:
            selected_items = self.unassigned_table.selectedItems()
            if not selected_items:
                return
                
            categories = []
            for item in selected_items:
                if item.column() == 0:  # Category column
                    categories.append(item.text())
            
            if not categories:
                return
                
            drag = QDrag(self.unassigned_table)
            mime_data = QMimeData()
            mime_data.setText(';'.join(categories))
            drag.setMimeData(mime_data)
            
            pixmap = QPixmap(200, 30 * len(categories))
            pixmap.fill(Qt.transparent)
            
            painter = QPainter(pixmap)
            painter.setPen(Qt.black)
            for i, category in enumerate(categories):
                painter.drawText(10, 20 + i * 30, category[:25])  # Truncate long names
            painter.end()
            
            drag.setPixmap(pixmap)
            drag.exec_(Qt.MoveAction)
            
        except Exception as e:
            logger.error(f"Error in table drag: {e}")
        
    def update_unassigned_count(self):
        """Update unassigned categories count display"""
        count = len(self.unassigned_categories)
        self.unassigned_count_label.setText(f"{count} categories")
        
    def create_initial_bins(self):
        """Create initial empty bins"""
        self.clear_all_bins()
        
        from PyQt5.QtCore import QCoreApplication
        QCoreApplication.processEvents()
        
        num_bins = self.num_bins_spinner.value()
        for i in range(num_bins):
            self.add_bin()
            
    def set_num_bins(self, num_bins: int):
        """Set number of bins with proper validation"""
        try:
            current_bins = len(self.bins)
            
            if num_bins > current_bins:
                for i in range(current_bins, num_bins):
                    self.add_bin()
            elif num_bins < current_bins and num_bins >= 2:
                bins_to_remove = []
                for i in range(num_bins, current_bins):
                    if i < len(self.bins):
                        bin_widget = self.bins[i]
                        categories = bin_widget.get_categories()
                        self.unassigned_categories.extend(categories)
                        bins_to_remove.append(i)
                        
                for i in reversed(bins_to_remove):
                    if i < len(self.bins):
                        self.remove_bin(i)
                        
                self.populate_unassigned_table()
                
            self.bins_changed.emit()
            
        except Exception as e:
            logger.error(f"Error setting number of bins: {e}")
            self.validation_error.emit(f"Failed to set bins: {e}")
            
    def add_bin(self):
        """Add a new bin with proper error handling"""
        try:
            bin_id = len(self.bins)
            bin_widget = BinWidget(bin_id, f"Bin {bin_id + 1}")
            bin_widget.categories_received.connect(self.on_categories_moved_to_bin)
            bin_widget.bin_deleted.connect(self.remove_bin)
            
            self.bins_layout.insertWidget(self.bins_layout.count() - 1, bin_widget)
            self.bins.append(bin_widget)
            
            logger.debug(f"Added bin {bin_id}")
            self.bins_changed.emit()
            
        except Exception as e:
            logger.error(f"Error adding bin: {e}")
            self.validation_error.emit(f"Failed to add bin: {e}")
            
    def remove_bin(self, bin_id: int):
        """Remove a bin and return its categories to unassigned"""
        try:
            if bin_id < len(self.bins) and len(self.bins) > 2:  # Keep at least 2 bins
                bin_widget = self.bins[bin_id]
                
                categories = bin_widget.get_categories()
                self.unassigned_categories.extend(categories)
                
                self.bins_layout.removeWidget(bin_widget)
                bin_widget.deleteLater()
                self.bins.pop(bin_id)
                
                for i, bin_widget in enumerate(self.bins):
                    bin_widget.bin_id = i
                    bin_widget.setTitle(f"Bin {i + 1}")
                    
                self.populate_unassigned_table()
                self.bins_changed.emit()
                
                logger.debug(f"Removed bin {bin_id}")
                
        except Exception as e:
            logger.error(f"Error removing bin {bin_id}: {e}")
            
    def clear_all_bins(self):
        """Clear all bins and return categories to unassigned"""
        try:
            for bin_widget in self.bins:
                categories = bin_widget.get_categories()
                self.unassigned_categories.extend(categories)
                
            for bin_widget in self.bins:
                self.bins_layout.removeWidget(bin_widget)
                bin_widget.setParent(None)  # Immediate removal
                bin_widget.deleteLater()
                
            self.bins.clear()
            
            self.bins_layout.update()
            self.populate_unassigned_table()
            
        except Exception as e:
            logger.error(f"Error clearing bins: {e}")
            
    def on_categories_moved_to_bin(self, categories: List[str], bin_id: int):
        """Handle categories moved to a bin with validation"""
        try:
            for category in categories:
                if category in self.unassigned_categories:
                    self.unassigned_categories.remove(category)
                    
            for i, bin_widget in enumerate(self.bins):
                if i != bin_id:
                    bin_widget.remove_categories(categories)
                    
            self.populate_unassigned_table()
            self.bins_changed.emit()
            
            logger.debug(f"Moved {len(categories)} categories to bin {bin_id}")
            
        except Exception as e:
            logger.error(f"Error moving categories to bin {bin_id}: {e}")
            
    def auto_group_categories(self):
        """Auto-group categories using intelligent methods (simplified per requirement 22)"""
        try:
            if self.data is None or len(self.unassigned_categories) == 0:
                QMessageBox.information(self, "Auto Group", "No unassigned categories to group.")
                return
                
            method = self.auto_group_combo.currentText()
            max_groups = self.num_bins_spinner.value()
            
            if method == 'frequency':
                groups = self._group_by_frequency(max_groups)
            elif method == 'target_rate':
                groups = self._group_by_target_rate(max_groups)
            else:
                groups = self._group_by_frequency(max_groups)  # Default
                
            self.ungroup_all_categories()
            
            if len(groups) > len(self.bins):
                self.num_bins_spinner.setValue(len(groups))
                
            for i, group in enumerate(groups):
                if i < len(self.bins):
                    available_categories = [cat for cat in group if cat in self.unassigned_categories]
                    if available_categories:
                        self.bins[i].add_categories(available_categories)
                        for cat in available_categories:
                            if cat in self.unassigned_categories:
                                self.unassigned_categories.remove(cat)
                        
            self.populate_unassigned_table()
            self.bins_changed.emit()
                        
            QMessageBox.information(self, "Auto Grouping Complete", 
                                  f"Created {len(groups)} groups using {method} method.")
                                  
        except Exception as e:
            logger.error(f"Error in auto grouping: {e}")
            QMessageBox.critical(self, "Auto Grouping Error", f"Grouping failed: {e}")
            
    def _group_by_frequency(self, max_groups: int) -> List[List[str]]:
        """Group categories by frequency"""
        try:
            value_counts = self.data.value_counts()
            categories = value_counts.index.tolist()
            
            if len(categories) <= max_groups:
                return [[cat] for cat in categories]
                
            groups = []
            categories_per_group = max(1, len(categories) // max_groups)
            
            for i in range(0, len(categories), categories_per_group):
                group = categories[i:i + categories_per_group]
                if group:
                    groups.append(group)
                    
            if len(groups) > max_groups:
                groups[-2].extend(groups[-1])
                groups.pop()
                
            return groups
            
        except Exception as e:
            logger.error(f"Error in frequency grouping: {e}")
            return [[cat] for cat in self.unassigned_categories[:max_groups]]
            
    def _group_by_target_rate(self, max_groups: int) -> List[List[str]]:
        """Group categories by target rate (simplified)"""
        try:
            if self.target_data is None:
                return self._group_by_frequency(max_groups)
                
            category_stats = []
            for category in self.data.unique():
                mask = self.data == category
                target_subset = self.target_data[mask]
                
                if len(target_subset) > 0:
                    target_rate = target_subset.mean()
                    count = len(target_subset)
                    category_stats.append((category, target_rate, count))
                    
            category_stats.sort(key=lambda x: x[1])
            
            groups = []
            categories_per_group = max(1, len(category_stats) // max_groups)
            
            for i in range(0, len(category_stats), categories_per_group):
                group = [stat[0] for stat in category_stats[i:i + categories_per_group]]
                if group:
                    groups.append(group)
                    
            if len(groups) > max_groups:
                groups[-2].extend(groups[-1])
                groups.pop()
                
            return groups
            
        except Exception as e:
            logger.error(f"Error in target rate grouping: {e}")
            return self._group_by_frequency(max_groups)
            
    def ungroup_all_categories(self):
        """Ungroup all categories back to unassigned"""
        try:
            all_categories = []
            for bin_widget in self.bins:
                all_categories.extend(bin_widget.get_categories())
                bin_widget.clear_categories()
                
            self.unassigned_categories.extend(all_categories)
            self.unassigned_categories = list(set(self.unassigned_categories))  # Remove duplicates
            
            self.populate_unassigned_table()
            self.bins_changed.emit()
            
            logger.debug("Ungrouped all categories")
            
        except Exception as e:
            logger.error(f"Error ungrouping categories: {e}")
            
    def get_bin_configuration(self) -> List[Dict[str, Any]]:
        """Get current bin configuration"""
        try:
            bin_configs = []
            for i, bin_widget in enumerate(self.bins):
                categories = bin_widget.get_categories()
                bin_configs.append({
                    'bin_id': i,
                    'categories': categories
                })
            return bin_configs
            
        except Exception as e:
            logger.error(f"Error getting bin configuration: {e}")
            return []
        
    def set_bin_configuration(self, bin_configs: List[Dict[str, Any]]):
        """Set bin configuration from external source"""
        try:
            self.ungroup_all_categories()
            
            if len(bin_configs) > len(self.bins):
                self.num_bins_spinner.setValue(len(bin_configs))
            
            for i, config in enumerate(bin_configs):
                if i < len(self.bins):
                    categories = config.get('categories', [])
                    if categories:
                        available_categories = [cat for cat in categories if cat in self.unassigned_categories]
                        if available_categories:
                            self.bins[i].add_categories(available_categories)
                            self.on_categories_moved_to_bin(available_categories, i)
                            
            logger.info(f"Set configuration with {len(bin_configs)} bins")
            
        except Exception as e:
            logger.error(f"Error setting bin configuration: {e}")
            self.validation_error.emit(f"Failed to set configuration: {e}")
            
    def validate_bins(self) -> tuple:
        """Validate current bin configuration"""
        try:
            if not self.bins:
                return False, "No bins defined"
                
            non_empty_bins = [bin_widget for bin_widget in self.bins if bin_widget.get_categories()]
            
            if len(non_empty_bins) < 2:
                return False, "At least 2 bins with categories are required"
                
            assigned_categories = set()
            for bin_widget in self.bins:
                assigned_categories.update(bin_widget.get_categories())
                
            if self.data is not None:
                data_categories = set(self.data.unique().astype(str))
                unassigned = data_categories - assigned_categories
                
                if unassigned:
                    unassigned_list = list(unassigned)[:5]  # Show first 5
                    suffix = "..." if len(unassigned) > 5 else ""
                    return False, f"Unassigned categories: {unassigned_list}{suffix}"
                    
            return True, "Bins are valid"
            
        except Exception as e:
            logger.error(f"Error validating bins: {e}")
            return False, f"Validation error: {e}"
            
    def get_bin_count(self) -> int:
        """Get current number of bins"""
        return len(self.bins)