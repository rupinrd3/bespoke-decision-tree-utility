# utility/ui/dialogs/dataset_properties_dialog.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset Properties Dialog for Bespoke Utility
Allows viewing dataset metadata and selecting active variables for modeling.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLabel,
    QDialogButtonBox, QTabWidget, QWidget, QScrollArea,
    QTextEdit, QTableView # Added QTextEdit, QTableView
)
from PyQt5.QtCore import Qt, QAbstractTableModel, QVariant

from ui.widgets.variable_selector_widget import VariableSelectorWidget # Assuming this exists

logger = logging.getLogger(__name__)

class BasicStatsTableModel(QAbstractTableModel):
    """A simple model to display basic descriptive statistics."""
    def __init__(self, data: Optional[pd.DataFrame] = None, parent=None):
        super().__init__(parent)
        self._dataframe = pd.DataFrame()
        if data is not None:
            self.set_data(data)

    def set_data(self, data: pd.DataFrame):
        self.beginResetModel()
        try:
            numeric_df = data.select_dtypes(include=np.number)
            if not numeric_df.empty:
                self._dataframe = numeric_df.describe().T.reset_index()
                self._dataframe.rename(columns={'index': 'Variable'}, inplace=True)
            else:
                self._dataframe = pd.DataFrame(columns=['Variable', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])

        except Exception as e:
            logger.error(f"Error generating descriptive stats: {e}")
            self._dataframe = pd.DataFrame()
        self.endResetModel()


    def rowCount(self, parent=Qt.QModelIndex()):
        return self._dataframe.shape[0]

    def columnCount(self, parent=Qt.QModelIndex()):
        return self._dataframe.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return QVariant()
        if role == Qt.DisplayRole:
            value = self._dataframe.iloc[index.row(), index.column()]
            if isinstance(value, float):
                return f"{value:.4f}" # Format floats
            return str(value)
        return QVariant()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._dataframe.columns[section])
            if orientation == Qt.Vertical:
                return str(self._dataframe.index[section])
        return QVariant()


class DatasetPropertiesDialog(QDialog):
    """
    Dialog to display dataset properties, basic statistics, and allow variable selection.
    """

    def __init__(self, dataset_name: str, dataframe: pd.DataFrame,
                 all_variables_info: List[Dict[str, Any]], # For VariableSelectorWidget
                 initially_selected_variables: Optional[List[str]] = None,
                 source_path: Optional[str] = None, # New: path where dataset was loaded from
                 parent=None):
        super().__init__(parent)
        self.dataset_name = dataset_name
        self._df = dataframe
        self._all_variables_info = all_variables_info
        self._initially_selected_variables = initially_selected_variables
        self.source_path = source_path

        self.setWindowTitle(f"Dataset Properties: {self.dataset_name}")
        self.setMinimumSize(700, 600) # Increased size for more info

        self.init_ui()
        self.populate_data()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        summary_tab_widget = QWidget()
        summary_layout = QFormLayout(summary_tab_widget)
        summary_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.name_label = QLabel(self.dataset_name)
        summary_layout.addRow("Dataset Name:", self.name_label)
        
        self.source_path_label = QLabel("N/A")
        summary_layout.addRow("Source Path:", self.source_path_label)

        self.rows_label = QLabel("N/A")
        summary_layout.addRow("Number of Rows:", self.rows_label)

        self.cols_label = QLabel("N/A")
        summary_layout.addRow("Number of Columns:", self.cols_label)

        self.memory_label = QLabel("N/A")
        summary_layout.addRow("Memory Usage:", self.memory_label)

        self.duplicates_label = QLabel("N/A")
        summary_layout.addRow("Duplicate Rows:", self.duplicates_label)
        
        self.missing_overview_label = QLabel("N/A") # Cells with missing data
        summary_layout.addRow("Missing Cells (%):", self.missing_overview_label)


        self.tab_widget.addTab(summary_tab_widget, "Summary")

        stats_tab_widget = QWidget()
        stats_layout = QVBoxLayout(stats_tab_widget)
        stats_layout.addWidget(QLabel("Descriptive Statistics (Numeric Columns Only):"))
        self.stats_table_view = QTableView()
        self.stats_table_view.setAlternatingRowColors(True)
        self.stats_table_view.setEditTriggers(QTableView.NoEditTriggers)
        self.stats_model = BasicStatsTableModel() # Custom model
        self.stats_table_view.setModel(self.stats_model)
        stats_layout.addWidget(self.stats_table_view)
        self.tab_widget.addTab(stats_tab_widget, "Descriptive Stats")


        variables_tab_widget = QWidget() # Changed from QWidget to variables_tab_widget
        variables_layout = QVBoxLayout(variables_tab_widget) # Changed from QWidget to variables_tab_widget

        selector_label = QLabel("Select variables to be active for analysis and modeling:")
        variables_layout.addWidget(selector_label)

        self.variable_selector = VariableSelectorWidget() # Assuming this is correctly imported
        variables_layout.addWidget(self.variable_selector)
        self.tab_widget.addTab(variables_tab_widget, "Variable Selection") # Changed from QWidget to variables_tab_widget


        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

    def populate_data(self):
        """Populates the dialog with data from the DataFrame and variable info."""
        if self._df is None:
            self.rows_label.setText("N/A (DataFrame not loaded)")
            self.cols_label.setText(str(len(self._all_variables_info))) # Use all_vars if df is None
            self.memory_label.setText("N/A")
            self.duplicates_label.setText("N/A")
            self.missing_overview_label.setText("N/A")
            self.source_path_label.setText(self.source_path or "N/A")
            self.stats_model.set_data(pd.DataFrame()) # Empty stats table
        else:
            self.rows_label.setText(str(len(self._df)))
            self.cols_label.setText(str(len(self._df.columns)))
            try:
                mem_usage = self._df.memory_usage(deep=True).sum() / (1024 * 1024) # MB
                self.memory_label.setText(f"{mem_usage:.2f} MB")
            except Exception:
                self.memory_label.setText("Error calculating")

            try:
                num_duplicates = self._df.duplicated().sum()
                self.duplicates_label.setText(f"{num_duplicates} ({num_duplicates/len(self._df)*100:.2f}%)")
            except Exception:
                self.duplicates_label.setText("Error calculating")

            try:
                total_cells = self._df.size
                missing_cells = self._df.isnull().sum().sum()
                percent_missing = (missing_cells / total_cells * 100) if total_cells > 0 else 0
                self.missing_overview_label.setText(f"{missing_cells} ({percent_missing:.2f}%)")
            except Exception:
                self.missing_overview_label.setText("Error calculating")

            self.source_path_label.setText(self.source_path or "N/A")
            self.stats_model.set_data(self._df) # Populate stats table

        self.variable_selector.set_variables(self._all_variables_info, self._initially_selected_variables)

    def get_selected_variables(self) -> List[str]:
        """Returns the list of variable names selected by the user."""
        return self.variable_selector.get_selected_variables()

if __name__ == '__main__':
    import sys
    import numpy as np # For BasicStatsTableModel example
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)

    data = {
        'ID': range(100),
        'Age': [i % 50 + 20 for i in range(100)],
        'Income': [np.nan if i % 10 == 0 else i * 1000 + 30000 for i in range(100)],
        'Gender': ['Male' if i % 2 == 0 else 'Female' for i in range(100)],
        'Target': [i % 2 for i in range(100)]
    }
    sample_df = pd.DataFrame(data)
    sample_df.iloc[5, sample_df.columns.get_loc('Age')] = np.nan # Add some more NaNs

    sample_variables_info = []
    import random
    for col in sample_df.columns:
        sample_variables_info.append({
            'name': col,
            'type': str(sample_df[col].dtype),
            'importance': random.uniform(0, 0.5) if col not in ['ID', 'Target'] else (0.8 if col == 'Target' else None)
        })

    initially_selected = ['Age', 'Income', 'Target']

    dialog = DatasetPropertiesDialog(
        dataset_name="Sample Customer Data",
        dataframe=sample_df,
        all_variables_info=sample_variables_info,
        initially_selected_variables=initially_selected,
        source_path="/path/to/your/datafile.csv"
    )

    if dialog.exec_() == QDialog.Accepted:
        selected_vars = dialog.get_selected_variables()
        print("Dialog accepted. Selected variables:", selected_vars)
    else:
        print("Dialog cancelled.")

    sys.exit(app.exec_())