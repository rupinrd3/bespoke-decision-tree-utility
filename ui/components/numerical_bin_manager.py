#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Numerical Bin Manager with automatic boundary adjustment

"""

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Callable, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QHeaderView, QMessageBox,
    QComboBox, QCheckBox, QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor

logger = logging.getLogger(__name__)


class EnhancedNumericalBinManager(QWidget):
    """Enhanced numerical bin manager with auto-adjustment and real-time updates"""
    
    bins_changed = pyqtSignal(list)  # Emitted when bin configuration changes
    statistics_updated = pyqtSignal(list)  # Emitted when statistics are recalculated
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = None
        self.target_data = None
        self.bins = []  # List of (min_val, max_val) tuples
        self.precision = 3
        self.auto_adjust = True
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._delayed_update_statistics)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        controls_group = QGroupBox("Bin Controls")
        controls_layout = QFormLayout(controls_group)
        
        self.num_bins_spinner = QSpinBox()
        self.num_bins_spinner.setRange(2, 20)
        self.num_bins_spinner.setValue(4)  # Changed from 3 to 4 as per requirement 18
        self.num_bins_spinner.valueChanged.connect(self.on_num_bins_changed)
        controls_layout.addRow("Number of bins:", self.num_bins_spinner)
        
        self.precision_spinner = QSpinBox()
        self.precision_spinner.setRange(0, 6)
        self.precision_spinner.setValue(3)
        self.precision_spinner.valueChanged.connect(self.set_precision)
        controls_layout.addRow("Decimal precision:", self.precision_spinner)
        
        self.auto_adjust_checkbox = QCheckBox("Auto-adjust adjacent bins")
        self.auto_adjust_checkbox.setChecked(True)
        self.auto_adjust_checkbox.toggled.connect(self.set_auto_adjust)
        controls_layout.addRow(self.auto_adjust_checkbox)
        
        layout.addWidget(controls_group)
        
        buttons_layout = QHBoxLayout()
        
        self.equal_width_btn = QPushButton("Equal Width")
        self.equal_width_btn.clicked.connect(self.create_equal_width_bins)
        buttons_layout.addWidget(self.equal_width_btn)
        
        self.equal_freq_btn = QPushButton("Equal Frequency")
        self.equal_freq_btn.clicked.connect(self.create_equal_frequency_bins)
        buttons_layout.addWidget(self.equal_freq_btn)
        
        
        buttons_layout.addStretch()
        
        self.add_bin_btn = QPushButton("Add Bin")
        self.add_bin_btn.clicked.connect(self.add_bin)
        buttons_layout.addWidget(self.add_bin_btn)
        
        self.remove_bin_btn = QPushButton("Remove Bin")
        self.remove_bin_btn.clicked.connect(self.remove_selected_bin)
        buttons_layout.addWidget(self.remove_bin_btn)
        
        layout.addLayout(buttons_layout)
        
        self.bins_table = QTableWidget()
        self.bins_table.setColumnCount(5)
        self.bins_table.setHorizontalHeaderLabels([
            'Bin ID', 'Min Value', 'Max Value', 'Count', 'Percentage'
        ])
        
        header = self.bins_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.bins_table.setColumnWidth(3, 80)
        
        self.bins_table.setAlternatingRowColors(True)
        self.bins_table.itemChanged.connect(self.on_table_item_changed)
        
        layout.addWidget(self.bins_table)
        
    def load_data(self, data: pd.Series, target_data: pd.Series = None):
        """Load data for bin management"""
        logger.info(f"Loading data for numerical bin manager: {len(data)} samples")
        
        if data is None:
            logger.error("No data provided to numerical bin manager")
            QMessageBox.warning(self, "No Data", "No data provided for binning.")
            return
            
        self.data = data
        self.target_data = target_data
        
        valid_data = data.dropna()
        logger.info(f"Total samples: {len(self.data)}, Valid (non-NaN): {len(valid_data)} samples")
        
        if len(valid_data) == 0:
            logger.warning("No valid data available - all values are NaN")
            QMessageBox.warning(self, "No Data", "All data values are missing.")
            return
        
        self._valid_data = valid_data
            
        # Log data statistics for debugging
        logger.info(f"Data range: {self.data.min():.3f} to {self.data.max():.3f}")
        logger.info(f"Data type: {self.data.dtype}")
        
        logger.info("Creating initial equal width bins")
        self.create_equal_width_bins()
        
    def set_precision(self, precision: int):
        """Set decimal precision for boundaries"""
        self.precision = precision
        self._populate_bins_table()
        
    def set_auto_adjust(self, enabled: bool):
        """Enable/disable automatic adjustment"""
        self.auto_adjust = enabled
        
    def set_num_bins(self, num_bins: int):
        """Set number of bins"""
        self.num_bins_spinner.setValue(num_bins)
        
    def on_num_bins_changed(self, num_bins: int):
        """Handle bin count changes"""
        if self.data is not None:
            self.create_equal_width_bins()
            
    def create_equal_width_bins(self, num_bins=None):
        """Create equal width bins maintaining existing interface and signal names"""
        try:
            logger.info("Starting create_equal_width_bins")
            if not hasattr(self, 'data') or self.data is None or len(self.data) == 0:
                logger.error("No data available for creating bins")
                if hasattr(self, 'validationError'):
                    self.validationError.emit("No data available for binning")
                return False
                
            if num_bins is None:
                if hasattr(self, 'num_bins_spinner'):
                    num_bins = self.num_bins_spinner.value()
                elif hasattr(self, 'num_bins_spinbox'):
                    num_bins = self.num_bins_spinbox.value()
                else:
                    num_bins = 3  # Default fallback
                
            if num_bins < 2:
                if hasattr(self, 'validationError'):
                    self.validationError.emit("At least 2 bins are required")
                return False
                
            if hasattr(self, '_valid_data') and len(self._valid_data) > 0:
                valid_data = self._valid_data
            else:
                valid_data = self.data.dropna()
                
            if len(valid_data) == 0:
                if hasattr(self, 'validationError'):
                    self.validationError.emit("All data values are missing")
                return False
                
            data_min = float(valid_data.min())
            data_max = float(valid_data.max())
            
            if data_min == data_max:
                if hasattr(self, 'validationError'):
                    self.validationError.emit("Cannot create bins: all values are the same")
                return False
                
            bin_width = (data_max - data_min) / num_bins
            bins = []
            
            for i in range(num_bins):
                bin_min = data_min + i * bin_width
                bin_max = data_min + (i + 1) * bin_width
                
                if i == num_bins - 1:
                    bin_max = data_max
                    
                bins.append((bin_min, bin_max))
                
            self.bins = bins
            logger.info(f"Set bins attribute with {len(bins)} bins: {bins}")
                
            if hasattr(self, '_populate_bins_table'):
                logger.info("Calling _populate_bins_table")
                self._populate_bins_table()
                logger.info(f"Table now has {self.bins_table.rowCount()} rows")
            else:
                logger.error("_populate_bins_table method not found")
            
            if hasattr(self, 'schedule_statistics_update'):
                logger.info("Scheduling statistics update")
                self.schedule_statistics_update()
                
            if hasattr(self, 'bins_changed'):
                logger.info("Emitting bins_changed signal")
                self.bins_changed.emit(self.bins)
            
            logger.info(f"Successfully created {len(bins)} equal width bins")
            return True
            
        except Exception as e:
            error_msg = f"Error creating equal width bins: {str(e)}"
            logger.error(error_msg)
            if hasattr(self, 'validationError'):
                self.validationError.emit(error_msg)
            return False
        
    def create_equal_frequency_bins(self):
        """Create equal frequency bins - FIXED to handle missing values correctly"""
        if self.data is None:
            return
            
        num_bins = self.num_bins_spinner.value()
        
        try:
            valid_data = self.data.dropna() if hasattr(self.data, 'dropna') else self.data
            
            if len(valid_data) == 0:
                QMessageBox.warning(self, "Binning Error", "No valid data available for binning.")
                return
            
            quantiles = np.linspace(0, 1, num_bins + 1)
            bin_edges = valid_data.quantile(quantiles).values
            
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                QMessageBox.warning(self, "Binning Error", "Cannot create bins with current data.")
                return
                
            self.bins = []
            for i in range(len(bin_edges) - 1):
                self.bins.append((bin_edges[i], bin_edges[i + 1]))
            
            # Comprehensive debugging for bin calculation and data verification
            logger.info(f"Data stats - Min: {valid_data.min():.6f}, Max: {valid_data.max():.6f}, Valid Count: {len(valid_data)}, Total Count: {len(self.data)}")
            logger.info(f"Quantiles used: {quantiles}")
            logger.info(f"Raw bin edges: {bin_edges}")
            logger.info(f"Equal frequency bins calculated: {self.bins}")
            
            total_valid = len(valid_data)
            for i, (min_val, max_val) in enumerate(self.bins):
                if i == len(self.bins) - 1:  # Last bin includes max value
                    mask = (valid_data >= min_val) & (valid_data <= max_val)
                else:
                    mask = (valid_data >= min_val) & (valid_data < max_val)
                count = mask.sum()
                percentage = (count / total_valid) * 100 if total_valid > 0 else 0
                logger.info(f"  Bin {i+1}: {min_val:.6f} to {max_val:.6f} ({count} samples, {percentage:.1f}%)")
                
            self._populate_bins_table()
            self.schedule_statistics_update()
            self.bins_changed.emit(self.bins)
            
        except Exception as e:
            logger.error(f"Error creating equal frequency bins: {e}")
            QMessageBox.critical(self, "Binning Error", f"Error creating bins: {e}")
            
    def create_quantile_bins(self):
        """Create quantile-based bins (alias for equal frequency)"""
        self.create_equal_frequency_bins()
        
    def add_bin(self):
        """Add a new bin"""
        if not self.bins:
            return
            
        last_bin = self.bins[-1]
        new_min = last_bin[1]
        new_max = new_min + (last_bin[1] - last_bin[0])
        
        self.bins.append((new_min, new_max))
        self._populate_bins_table()
        self.schedule_statistics_update()
        self.bins_changed.emit(self.bins)
        
    def remove_selected_bin(self):
        """Remove selected bin"""
        current_row = self.bins_table.currentRow()
        if current_row < 0 or current_row >= len(self.bins):
            QMessageBox.warning(self, "No Selection", "Please select a bin to remove.")
            return
            
        if len(self.bins) <= 2:
            QMessageBox.warning(self, "Cannot Remove", "Must have at least 2 bins.")
            return
            
        self.bins.pop(current_row)
        
        if self.auto_adjust:
            self._auto_adjust_after_removal(current_row)
            
        self._populate_bins_table()
        self.schedule_statistics_update()
        self.bins_changed.emit(self.bins)
        
    def _auto_adjust_after_removal(self, removed_index: int):
        """Auto-adjust bins after removal"""
        if not self.bins:
            return
            
        if 0 < removed_index < len(self.bins):
            left_bin = self.bins[removed_index - 1]
            right_bin = self.bins[removed_index]
            
            midpoint = (left_bin[1] + right_bin[0]) / 2
            self.bins[removed_index - 1] = (left_bin[0], midpoint)
            self.bins[removed_index] = (midpoint, right_bin[1])
            
    def on_table_item_changed(self, item: QTableWidgetItem):
        """Handle manual table edits with gentle validation"""
        if not self.auto_adjust:
            return
            
        row = item.row()
        col = item.column()
        text = item.text().strip()
        
        if not text:
            return
        
        try:
            new_value = float(text)
            
            item.setBackground(QColor(255, 255, 255))  # White background
            
            if col == 1:  # Min value changed
                self._handle_min_value_change(row, new_value)
            elif col == 2:  # Max value changed
                self._handle_max_value_change(row, new_value)
                
        except ValueError:
            item.setBackground(QColor(255, 200, 200))  # Light red background
            pass
            
    def _handle_min_value_change(self, row: int, new_min: float):
        """Handle minimum value changes with validation"""
        if row >= len(self.bins):
            return
            
        current_bin = self.bins[row]
        
        if new_min >= current_bin[1]:
            self._repopulate_table_row(row)
            return
            
        if row > 0:
            prev_bin = self.bins[row - 1]
            if new_min < prev_bin[0]:
                self._repopulate_table_row(row)
                return
        
        self.bins[row] = (new_min, current_bin[1])
        
        if row > 0 and self.auto_adjust:
            prev_bin = self.bins[row - 1]
            self.bins[row - 1] = (prev_bin[0], new_min)
            
        self._populate_bins_table()  # Refresh all cells to show adjustments
        self.schedule_statistics_update(immediate=True)  # Fast update for user edits
        self.bins_changed.emit(self.bins)
        
    def _handle_max_value_change(self, row: int, new_max: float):
        """Handle maximum value changes with validation"""
        if row >= len(self.bins):
            return
            
        current_bin = self.bins[row]
        
        if new_max <= current_bin[0]:
            self._repopulate_table_row(row)
            return
            
        if row < len(self.bins) - 1:
            next_bin = self.bins[row + 1]
            if new_max > next_bin[1]:
                self._repopulate_table_row(row)
                return
        
        self.bins[row] = (current_bin[0], new_max)
        
        if row < len(self.bins) - 1 and self.auto_adjust:
            next_bin = self.bins[row + 1]
            self.bins[row + 1] = (new_max, next_bin[1])
            
        self._populate_bins_table()  # Refresh all cells to show adjustments
        self.schedule_statistics_update(immediate=True)  # Fast update for user edits
        self.bins_changed.emit(self.bins)
        
    def _repopulate_table_row(self, row: int):
        """Repopulate a single table row with original values"""
        if row >= len(self.bins):
            return
            
        min_val, max_val = self.bins[row]
        
        self.bins_table.itemChanged.disconnect()
        
        try:
            min_item = QTableWidgetItem(f"{min_val:.{self.precision}f}")
            min_item.setFlags(min_item.flags() | Qt.ItemIsEditable)
            min_item.setBackground(QColor(255, 255, 255))  # Clear any error background
            self.bins_table.setItem(row, 1, min_item)
            
            max_item = QTableWidgetItem(f"{max_val:.{self.precision}f}")
            max_item.setFlags(max_item.flags() | Qt.ItemIsEditable)
            max_item.setBackground(QColor(255, 255, 255))  # Clear any error background
            self.bins_table.setItem(row, 2, max_item)
            
        finally:
            self.bins_table.itemChanged.connect(self.on_table_item_changed)
        
    def _populate_bins_table(self):
        """Populate the bins table"""
        self.bins_table.setRowCount(len(self.bins))
        
        self.bins_table.itemChanged.disconnect()
        
        try:
            for i, (min_val, max_val) in enumerate(self.bins):
                self.bins_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
                
                min_item = QTableWidgetItem(f"{min_val:.{self.precision}f}")
                min_item.setFlags(min_item.flags() | Qt.ItemIsEditable)
                self.bins_table.setItem(i, 1, min_item)
                
                max_item = QTableWidgetItem(f"{max_val:.{self.precision}f}")
                max_item.setFlags(max_item.flags() | Qt.ItemIsEditable)
                self.bins_table.setItem(i, 2, max_item)
                
                self.bins_table.setItem(i, 3, QTableWidgetItem("..."))
                self.bins_table.setItem(i, 4, QTableWidgetItem("..."))
                
        finally:
            self.bins_table.itemChanged.connect(self.on_table_item_changed)
            
    def schedule_statistics_update(self, immediate=False):
        """Schedule delayed statistics update"""
        self.update_timer.stop()
        if immediate:
            self.update_timer.start(50)  # 50ms delay for immediate user feedback
        else:
            self.update_timer.start(500)  # 500ms delay for other cases
        
    def _delayed_update_statistics(self):
        """Perform delayed update"""
        self.update_statistics()
        
    def update_statistics(self):
        """Update bin statistics"""
        if self.data is None or not self.bins:
            return
            
        try:
            from utils.performance_optimizer import get_background_calculator
            
            calculator = get_background_calculator()
            calculator.calculate_async(
                "numerical_bins",
                self.data,
                self.target_data if self.target_data is not None else pd.Series(dtype='float64'),
                self.bins,
                self._on_statistics_calculated,
                is_categorical=False
            )
            
        except ImportError:
            self._calculate_statistics_sync()
            
    def _calculate_statistics_sync(self):
        """Synchronous statistics calculation - MATCHES actual split behavior exactly"""
        total_count = len(self.data)
        
        missing_mask = self.data.isna()
        missing_count = missing_mask.sum()
        valid_data_mask = ~missing_mask
        
        self.bins_table.setRowCount(len(self.bins))
        
        bin_counts = []
        for i, (min_val, max_val) in enumerate(self.bins):
            if i == len(self.bins) - 1:  # Last bin includes upper bound
                mask = valid_data_mask & (self.data >= min_val) & (self.data <= max_val)
            else:
                mask = valid_data_mask & (self.data >= min_val) & (self.data < max_val)
                
            count = mask.sum()
            bin_counts.append(count)
        
        if missing_count > 0 and bin_counts:
            largest_bin_idx = bin_counts.index(max(bin_counts))
            bin_counts[largest_bin_idx] += missing_count
            logger.info(f"Assigned {missing_count} missing values to largest bin (bin {largest_bin_idx + 1})")
        
        for i, count in enumerate(bin_counts):
            percentage = (count / total_count) * 100 if total_count > 0 else 0
            self.bins_table.setItem(i, 3, QTableWidgetItem(str(count)))
            self.bins_table.setItem(i, 4, QTableWidgetItem(f"{percentage:.1f}%"))
        
        total_assigned = sum(bin_counts)
        logger.info(f"Preview totals: {total_assigned} (expected: {total_count}) - Match: {total_assigned == total_count}")
        
        if total_count > 0:
            max_count = total_count / len(self.bins)  # Expected count for equal distribution
            for i, count in enumerate(bin_counts):
                intensity = min(count / max_count, 1.0) if max_count > 0 else 0
                color = QColor(int(255 * (1 - intensity)), int(255 * intensity), 0, 100)
                
                for col in range(5):
                    item = self.bins_table.item(i, col)
                    if item:
                        item.setBackground(color)
                        
    def _on_statistics_calculated(self, request_id: str, result: List[dict], error: str):
        """Handle calculated statistics"""
        if error:
            logger.error(f"Error calculating statistics: {error}")
            return
            
        if request_id != "numerical_bins" or not result:
            return
            
        for i, stats in enumerate(result):
            if i >= self.bins_table.rowCount():
                break
                
            count = stats.get('count', 0)
            percentage = stats.get('percentage', 0)
            
            self.bins_table.setItem(i, 3, QTableWidgetItem(str(count)))
            self.bins_table.setItem(i, 4, QTableWidgetItem(f"{percentage:.1f}%"))
            
        self.statistics_updated.emit(result)
        
    def get_bin_definitions(self) -> List[Tuple[float, float]]:
        """Get current bin definitions"""
        return self.bins.copy()
        
    def get_bin_count(self) -> int:
        """Get current number of bins"""
        return len(self.bins)
        
    def validate_bins(self) -> bool:
        """Validate bin configuration"""
        if not self.bins:
            return False
            
        for i in range(len(self.bins) - 1):
            if self.bins[i][1] != self.bins[i + 1][0]:
                return False
                
        if self.data is not None:
            data_min, data_max = self.data.min(), self.data.max()
            if self.bins[0][0] > data_min or self.bins[-1][1] < data_max:
                return False
                
        return True

    def on_bin_value_changed(self):
        """CRITICAL FIX: Handle bin value changes with debounced validation to prevent multiple popups"""
        try:
            if hasattr(self, '_validation_timer'):
                self._validation_timer.stop()
                
            try:
                from PyQt5.QtCore import QTimer
            except ImportError:
                logger.warning("PyQt5 not available for validation timer")
                return
                
            self._validation_timer = QTimer()
            self._validation_timer.timeout.connect(self._perform_debounced_validation)
            self._validation_timer.setSingleShot(True)
            self._validation_timer.start(750)  # 750ms delay to reduce popup frequency
            
        except Exception as e:
            logger.error(f"Error in on_bin_value_changed: {e}")

    def _perform_debounced_validation(self):
        """CRITICAL FIX: Perform validation after debounce delay to reduce popup messages"""
        try:
            import time
            
            has_focus = False
            if hasattr(self, 'bin_widgets'):
                for widget in getattr(self, 'bin_widgets', []):
                    if hasattr(widget, 'min_edit') and hasattr(widget.min_edit, 'hasFocus'):
                        if widget.min_edit.hasFocus():
                            has_focus = True
                            break
                    if hasattr(widget, 'max_edit') and hasattr(widget.max_edit, 'hasFocus'):
                        if widget.max_edit.hasFocus():
                            has_focus = True
                            break
                            
            if not has_focus:
                errors = self._validate_all_bins()
                if errors:
                    current_time = time.time()
                    if not hasattr(self, '_last_error_time'):
                        self._last_error_time = 0
                        
                    if False:  # Disabled automatic validation popups
                        try:
                            from PyQt5.QtWidgets import QMessageBox
                            QMessageBox.warning(self, "Validation Error", errors[0])
                            self._last_error_time = current_time
                        except ImportError:
                            logger.warning(f"Validation error: {errors[0]}")
                        
        except Exception as e:
            logger.error(f"Error in _perform_debounced_validation: {e}")

    def _validate_all_bins(self):
        """Validate all bins and return list of errors maintaining existing patterns"""
        errors = []
        try:
            if not hasattr(self, 'bin_widgets'):
                return errors
                
            for i, widget in enumerate(getattr(self, 'bin_widgets', [])):
                if hasattr(widget, 'min_edit') and hasattr(widget, 'max_edit'):
                    min_text = widget.min_edit.text().strip()
                    max_text = widget.max_edit.text().strip()
                    
                    if not min_text or not max_text:
                        continue
                        
                    try:
                        min_val = float(min_text)
                        max_val = float(max_text)
                        
                        if min_val >= max_val:
                            errors.append(f"Bin {i+1}: Minimum ({min_val:.3f}) must be less than maximum ({max_val:.3f})")
                            
                    except (ValueError, TypeError):
                        errors.append(f"Bin {i+1}: Please enter valid decimal numbers")
                        
        except Exception as e:
            logger.error(f"Error validating bins: {e}")
            
        return errors


    def get_bin_configuration(self):
        """Get current bin configuration in standard format"""
        bin_configs = []
        for i, (min_val, max_val) in enumerate(self.bins):
            bin_config = {
                'bin_id': i,
                'min_value': min_val,
                'max_value': max_val,
                'bin_type': 'numerical'
            }
            
            if self.data is not None:
                if i == len(self.bins) - 1:  # Last bin includes max value
                    mask = (self.data >= min_val) & (self.data <= max_val)
                else:
                    mask = (self.data >= min_val) & (self.data < max_val)
                bin_config['count'] = mask.sum()
                bin_config['percentage'] = (mask.sum() / len(self.data) * 100) if len(self.data) > 0 else 0
            
            bin_configs.append(bin_config)
        
        return bin_configs

NumericalBinManager = EnhancedNumericalBinManager