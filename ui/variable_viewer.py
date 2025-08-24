# utility/ui/variable_viewer.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Variable Viewer Widget for Bespoke Utility
Displays information about columns in a DataFrame.

"""

import logging
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
                           QHeaderView, QLabel)
from PyQt5.QtCore import Qt

logger = logging.getLogger(__name__)

class VariableViewerWidget(QWidget):
    """Widget to display variable information."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df = None

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)

        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(5) # Variable, Type, Missing, Unique, Example
        self.table_widget.setHorizontalHeaderLabels(
            ["Variable", "Data Type", "Missing (%)", "Unique Values", "Example Value"]
        )
        self.table_widget.setEditTriggers(QTableWidget.NoEditTriggers) # Read-only
        self.table_widget.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_widget.setSelectionMode(QTableWidget.SingleSelection)
        self.table_widget.setSortingEnabled(True)
        self.table_widget.verticalHeader().setVisible(False)

        header = self.table_widget.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch) # Variable Name
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents) # Type
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents) # Missing
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents) # Unique
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents) # Example

        self.layout.addWidget(self.table_widget)


    def update_variables(self, df: pd.DataFrame):
        """Updates the view with variables from the DataFrame."""
        self._df = df
        if df is None or df.empty:
            self.table_widget.setRowCount(0)
            return

        self.table_widget.setRowCount(len(df.columns))
        self.table_widget.setSortingEnabled(False) # Disable sorting during population

        total_rows = len(df)

        for i, col_name in enumerate(df.columns):
            col_data = df[col_name]
            dtype = str(col_data.dtype)

            missing_count = col_data.isna().sum()
            missing_percent = (missing_count / total_rows * 100) if total_rows > 0 else 0

            try:
                unique_count = col_data.nunique()
            except TypeError:
                unique_count = "N/A (unhashable)"


            first_valid_index = col_data.first_valid_index()
            example_value = col_data[first_valid_index] if first_valid_index is not None else "NA"
            if pd.isna(example_value): example_value = "NA" # Ensure NA display


            name_item = QTableWidgetItem(col_name)
            type_item = QTableWidgetItem(dtype)
            missing_item = QTableWidgetItem(f"{missing_percent:.2f}% ({missing_count})")
            unique_item = QTableWidgetItem(str(unique_count))
            example_item = QTableWidgetItem(str(example_value)[:50]) # Limit example length

            self.table_widget.setItem(i, 0, name_item)
            self.table_widget.setItem(i, 1, type_item)
            self.table_widget.setItem(i, 2, missing_item)
            self.table_widget.setItem(i, 3, unique_item)
            self.table_widget.setItem(i, 4, example_item)

        self.table_widget.setSortingEnabled(True) # Re-enable sorting
        logger.debug(f"Variable viewer updated with {len(df.columns)} variables.")

    def clear(self):
        """Clears the variable view."""
        self.table_widget.setRowCount(0)
        self._df = None
