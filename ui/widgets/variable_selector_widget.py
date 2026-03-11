# utility/ui/widgets/variable_selector_widget.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Variable Selector Widget for Bespoke Utility
Allows users to select a subset of variables for modeling or other operations.
"""

import logging
import pandas as pd
from typing import List, Dict, Any, Optional, Set

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QLineEdit, QCheckBox, QAbstractItemView,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox
)
from PyQt5.QtCore import Qt, pyqtSignal

logger = logging.getLogger(__name__)

class VariableSelectorWidget(QWidget):
    """
    A widget that allows users to view available variables and select a subset.
    It can also display associated information like importance scores.
    """
    selectionChanged = pyqtSignal(list) # Emits list of selected variable names

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_variables_info: List[Dict[str, Any]] = [] # List of dicts, e.g., {'name': str, 'type': str, 'importance': float}
        self._selected_variables: Set[str] = set()

        self.init_ui()
        self.connect_signals()

    def init_ui(self):
        """Initialize the user interface components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0,0,0,0)

        control_layout = QHBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter variables...")
        control_layout.addWidget(QLabel("Filter:"))
        control_layout.addWidget(self.filter_edit)

        self.select_all_button = QPushButton("Select All")
        self.deselect_all_button = QPushButton("Deselect All")
        control_layout.addWidget(self.select_all_button)
        control_layout.addWidget(self.deselect_all_button)
        main_layout.addLayout(control_layout)

        self.variables_table = QTableWidget()
        self.variables_table.setColumnCount(4) # Checkbox, Name, Type, Importance
        self.variables_table.setHorizontalHeaderLabels([" ", "Variable Name", "Data Type", "Importance"])
        self.variables_table.setSelectionMode(QAbstractItemView.NoSelection) # Selection via checkboxes
        self.variables_table.setEditTriggers(QAbstractItemView.NoEditTriggers) # Read-only cells

        header = self.variables_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents) # Checkbox
        header.setSectionResizeMode(1, QHeaderView.Stretch) # Name
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents) # Type
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents) # Importance
        self.variables_table.verticalHeader().setVisible(False)
        
        main_layout.addWidget(self.variables_table)

    def connect_signals(self):
        """Connect widget signals."""
        self.filter_edit.textChanged.connect(self._filter_variables)
        self.select_all_button.clicked.connect(self._select_all)
        self.deselect_all_button.clicked.connect(self._deselect_all)

    def set_variables(self, variables_info: List[Dict[str, Any]], selected_variables: Optional[List[str]] = None):
        """
        Populates the widget with variable information.

        Args:
            variables_info: A list of dictionaries, where each dictionary
                            contains info about a variable (e.g., 'name', 'type', 'importance').
            selected_variables: A list of variable names that should be initially selected.
        """
        self._all_variables_info = sorted(variables_info, key=lambda x: x.get('name','').lower())
        if selected_variables is not None:
            self._selected_variables = set(selected_variables)
        else:
            self._selected_variables = {var_info['name'] for var_info in self._all_variables_info}
            
        self._populate_table()

    def _populate_table(self):
        """Fills the QTableWidget with variable information."""
        self.variables_table.setRowCount(0) # Clear existing rows
        self.variables_table.setRowCount(len(self._all_variables_info))

        self.variables_table.blockSignals(True) # Block signals during population

        for row, var_info in enumerate(self._all_variables_info):
            var_name = var_info.get('name', 'N/A')
            var_type = var_info.get('type', 'N/A')
            var_importance = var_info.get('importance') # Can be None

            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk_item.setCheckState(Qt.Checked if var_name in self._selected_variables else Qt.Unchecked)
            self.variables_table.setItem(row, 0, chk_item)

            name_item = QTableWidgetItem(var_name)
            self.variables_table.setItem(row, 1, name_item)

            type_item = QTableWidgetItem(var_type)
            self.variables_table.setItem(row, 2, type_item)
            
            if var_importance is not None:
                importance_item = QTableWidgetItem(f"{var_importance:.4f}")
            else:
                importance_item = QTableWidgetItem("N/A")
            self.variables_table.setItem(row, 3, importance_item)
        
        self.variables_table.blockSignals(False)
        try: self.variables_table.itemChanged.disconnect(self._on_item_changed)
        except TypeError: pass # If not connected yet
        self.variables_table.itemChanged.connect(self._on_item_changed)


    def _on_item_changed(self, item: QTableWidgetItem):
        """Handles changes to items in the table, specifically checkboxes."""
        if item.column() == 0: # Checkbox column
            row = item.row()
            var_name_item = self.variables_table.item(row, 1)
            if var_name_item:
                var_name = var_name_item.text()
                if item.checkState() == Qt.Checked:
                    self._selected_variables.add(var_name)
                else:
                    self._selected_variables.discard(var_name)
                self.selectionChanged.emit(self.get_selected_variables())
                logger.debug(f"Variable '{var_name}' selection changed. New selection: {self._selected_variables}")


    def _filter_variables(self, text: str):
        """Filters the list of variables based on the search text."""
        filter_text = text.lower()
        for row in range(self.variables_table.rowCount()):
            name_item = self.variables_table.item(row, 1)
            type_item = self.variables_table.item(row, 2)
            if name_item and type_item:
                matches_name = filter_text in name_item.text().lower()
                matches_type = filter_text in type_item.text().lower()
                self.variables_table.setRowHidden(row, not (matches_name or matches_type))

    def _select_all(self):
        """Selects all variables."""
        self.variables_table.blockSignals(True)
        for row in range(self.variables_table.rowCount()):
            chk_item = self.variables_table.item(row, 0)
            if chk_item:
                chk_item.setCheckState(Qt.Checked)
                var_name_item = self.variables_table.item(row, 1)
                if var_name_item:
                    self._selected_variables.add(var_name_item.text())
        self.variables_table.blockSignals(False)
        self.selectionChanged.emit(self.get_selected_variables())
        logger.debug("All variables selected.")


    def _deselect_all(self):
        """Deselects all variables."""
        self.variables_table.blockSignals(True)
        for row in range(self.variables_table.rowCount()):
            chk_item = self.variables_table.item(row, 0)
            if chk_item:
                chk_item.setCheckState(Qt.Unchecked)
        self._selected_variables.clear()
        self.variables_table.blockSignals(False)
        self.selectionChanged.emit(self.get_selected_variables())
        logger.debug("All variables deselected.")

    def get_selected_variables(self) -> List[str]:
        """Returns a list of currently selected variable names."""
        current_selection = set()
        for row in range(self.variables_table.rowCount()):
            chk_item = self.variables_table.item(row,0)
            name_item = self.variables_table.item(row,1)
            if chk_item and name_item and chk_item.checkState() == Qt.Checked:
                current_selection.add(name_item.text())
        self._selected_variables = current_selection # Update internal set
        return sorted(list(self._selected_variables))


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)

    example_vars_info = [
        {'name': 'Age', 'type': 'int64', 'importance': 0.25},
        {'name': 'Income', 'type': 'float64', 'importance': 0.35},
        {'name': 'Gender', 'type': 'object', 'importance': 0.10},
        {'name': 'ProductCategory', 'type': 'category', 'importance': 0.15},
        {'name': 'LastPurchaseDate', 'type': 'datetime64[ns]', 'importance': 0.05},
        {'name': 'IsSubscribed', 'type': 'bool', 'importance': 0.10},
    ]
    initially_selected = ['Age', 'Income']

    widget = VariableSelectorWidget()
    widget.set_variables(example_vars_info, initially_selected)
    widget.selectionChanged.connect(lambda selected: print("Selection changed:", selected))
    widget.setWindowTitle("Variable Selector Test")
    widget.resize(400, 300)
    widget.show()

    sys.exit(app.exec_())