# utility/ui/dialogs/missing_values_dialog.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dialog for Handling Missing Values in Bespoke Utility.
Allows choosing strategies (remove, fill) and applying them to selected columns.
"""

import logging
from typing import List, Dict, Any, Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLabel,
    QDialogButtonBox, QComboBox, QGroupBox, QRadioButton,
    QLineEdit, QListWidget, QListWidgetItem, QAbstractItemView,
    QPushButton, QHBoxLayout, QCheckBox, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal

logger = logging.getLogger(__name__)

class MissingValuesDialog(QDialog):
    """Dialog for configuring missing value handling."""

    def __init__(self, columns: Optional[List[str]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Handle Missing Values")
        self.setMinimumWidth(500)
        self.columns = columns or []

        self.strategy = {
            'method': 'remove_rows', # Default: remove rows with any NA in selected cols
            'columns': [],          # Default: apply to selected columns from list
            'fill_value': 0,        # Default for fill_constant
            'apply_to_all_selected': True # If true, applies one strategy to all selected columns.
        }

        self.init_ui()
        self._update_ui_for_method(self.strategy['method']) # Initial setup

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        column_group = QGroupBox("Select Columns to Process")
        column_layout = QVBoxLayout(column_group)
        self.column_list_widget = QListWidget()
        self.column_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection) # Multi-select
        for col in self.columns:
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked) # Unchecked by default
            self.column_list_widget.addItem(item)
        
        select_buttons_layout = QHBoxLayout()
        self.select_all_cols_button = QPushButton("Select All")
        self.deselect_all_cols_button = QPushButton("Deselect All")
        self.select_all_cols_button.clicked.connect(self._select_all_columns)
        self.deselect_all_cols_button.clicked.connect(self._deselect_all_columns)
        select_buttons_layout.addWidget(self.select_all_cols_button)
        select_buttons_layout.addWidget(self.deselect_all_cols_button)
        select_buttons_layout.addStretch()

        column_layout.addLayout(select_buttons_layout)
        column_layout.addWidget(self.column_list_widget)
        layout.addWidget(column_group)


        strategy_group = QGroupBox("Handling Strategy")
        strategy_form_layout = QFormLayout(strategy_group)

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Remove rows with missing values in selected columns", # remove_rows_subset
            "Remove rows with any missing values (across all columns)", # remove_rows_all (careful)
            "Fill with a constant value", # fill_constant
            "Fill with Mean (numeric columns)",    # fill_mean
            "Fill with Median (numeric columns)",  # fill_median
            "Fill with Mode (most frequent)", # fill_mode
        ])
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        strategy_form_layout.addRow("Method:", self.method_combo)

        self.fill_value_label = QLabel("Fill Value:")
        self.fill_value_edit = QLineEdit(str(self.strategy['fill_value']))
        self.fill_value_edit.setPlaceholderText("Enter constant value")
        strategy_form_layout.addRow(self.fill_value_label, self.fill_value_edit)
        
        layout.addWidget(strategy_group)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept_options)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _select_all_columns(self):
        for i in range(self.column_list_widget.count()):
            self.column_list_widget.item(i).setCheckState(Qt.Checked)

    def _deselect_all_columns(self):
        for i in range(self.column_list_widget.count()):
            self.column_list_widget.item(i).setCheckState(Qt.Unchecked)


    def _on_method_changed(self, method_text: str):
        """Update UI based on selected handling method."""
        method_key = "unknown"
        if "Remove rows with missing values in selected columns" in method_text: method_key = "remove_rows_subset"
        elif "Remove rows with any missing values" in method_text: method_key = "remove_rows_all"
        elif "Fill with a constant value" in method_text: method_key = "fill_constant"
        elif "Fill with Mean" in method_text: method_key = "fill_mean"
        elif "Fill with Median" in method_text: method_key = "fill_median"
        elif "Fill with Mode" in method_text: method_key = "fill_mode"
        
        self.strategy['method'] = method_key
        self._update_ui_for_method(method_key)

    def _update_ui_for_method(self, method_key: str):
        """Show/hide relevant input fields based on the method."""
        show_fill_value = (method_key == 'fill_constant')
        self.fill_value_label.setVisible(show_fill_value)
        self.fill_value_edit.setVisible(show_fill_value)

        enable_column_selection = method_key in ['remove_rows_subset', 'fill_constant', 'fill_mean', 'fill_median', 'fill_mode']
        self.column_list_widget.setEnabled(enable_column_selection)
        self.select_all_cols_button.setEnabled(enable_column_selection)
        self.deselect_all_cols_button.setEnabled(enable_column_selection)
        if not enable_column_selection: # e.g. "remove_rows_all"
            self._deselect_all_columns() # Clear selection if not applicable


    def accept_options(self):
        """Process and accept missing value handling options."""
        selected_cols = []
        for i in range(self.column_list_widget.count()):
            item = self.column_list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected_cols.append(item.text())

        method_key = self.strategy['method'] # Already set by _on_method_changed

        if method_key in ['remove_rows_subset', 'fill_constant', 'fill_mean', 'fill_median', 'fill_mode'] and not selected_cols:
            QMessageBox.warning(self, "No Columns Selected",
                                "Please select at least one column to apply the strategy, "
                                "or choose a method that applies to all rows.")
            return

        self.strategy['columns'] = selected_cols

        if method_key == 'fill_constant':
            self.strategy['fill_value'] = self.fill_value_edit.text()
            try:
                val = float(self.strategy['fill_value'])
                if isinstance(val, float) and val.is_integer():
                    val = int(val)
                self.strategy['fill_value'] = val
            except (ValueError, TypeError):
                pass
        else:
            self.strategy['fill_value'] = None # Not used for other methods

        if method_key == 'remove_rows_all':
            self.strategy['columns'] = 'all_with_na_in_row' # Special marker

        logger.info(f"Missing value handling strategy accepted: {self.strategy}")
        self.accept()

    def get_strategy(self) -> Dict[str, Any]:
        """Return the configured strategy."""
        return self.strategy

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)

    sample_cols = ["Age", "Income", "Gender", "Education", "LastLogin"]
    dialog = MissingValuesDialog(columns=sample_cols)
    if dialog.exec_() == QDialog.Accepted:
        print("Selected Strategy:", dialog.get_strategy())
    else:
        print("Dialog cancelled.")
    sys.exit(app.exec_())