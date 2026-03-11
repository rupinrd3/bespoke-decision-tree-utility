# utility/ui/data_viewer.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Viewer Widget for Bespoke Utility
Displays pandas DataFrames in a table view.

[PandasModel.__init__ -> Initializes the model with a pandas DataFrame -> dependent functions are None]
[PandasModel.rowCount -> Returns the number of rows in the DataFrame -> dependent functions are None]
[PandasModel.columnCount -> Returns the number of columns in the DataFrame -> dependent functions are None]
[PandasModel.data -> Returns the data for a specific cell -> dependent functions are None]
[PandasModel.headerData -> Returns the header data for rows and columns -> dependent functions are None]
[PandasModel.sort -> Sorts the table by a specific column -> dependent functions are None]
[PandasModel.dataframe -> Returns the original DataFrame -> dependent functions are None]
[PandasModel.set_dataframe -> Sets a new DataFrame for the model -> dependent functions are None]
[DataViewerWidget.__init__ -> Initializes the widget for displaying the DataFrame -> dependent functions are apply_filter]
[DataViewerWidget.set_dataframe -> Sets the DataFrame to be displayed in the widget -> dependent functions are update_info_labels]
[DataViewerWidget.update_info_labels -> Updates the labels showing the number of rows and columns -> dependent functions are None]
[DataViewerWidget.apply_filter -> Applies a filter to the data shown in the table -> dependent functions are None]
"""

import logging
import pandas as pd
from PyQt5.QtCore import (Qt, QAbstractTableModel, QModelIndex, QVariant,
                          QSortFilterProxyModel, QRegExp)
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableView,
                           QLabel, QLineEdit, QSpinBox, QPushButton,
                           QStyle, QStyleOptionHeader, QHeaderView)
from PyQt5.QtGui import QColor, QBrush

logger = logging.getLogger(__name__)

MAX_ROWS_TO_DISPLAY = 10000 # Limit rows displayed directly for performance

class PandasModel(QAbstractTableModel):
    """A model to interface a pandas DataFrame with QTableView."""

    def __init__(self, dataframe: pd.DataFrame = None, parent=None):
        super().__init__(parent)
        self._dataframe = dataframe if dataframe is not None else pd.DataFrame()
        if len(self._dataframe) > MAX_ROWS_TO_DISPLAY:
             logger.warning(f"DataFrame has {len(self._dataframe)} rows. Display limited to first {MAX_ROWS_TO_DISPLAY} rows for performance.")
             self._display_dataframe = self._dataframe.head(MAX_ROWS_TO_DISPLAY).copy()
             self._is_limited = True
        else:
             self._display_dataframe = self._dataframe.copy()
             self._is_limited = False


    def rowCount(self, parent=QModelIndex()) -> int:
        """Return the number of rows in the DataFrame."""
        return self._display_dataframe.shape[0] if not parent.isValid() else 0

    def columnCount(self, parent=QModelIndex()) -> int:
        """Return the number of columns in the DataFrame."""
        return self._display_dataframe.shape[1] if not parent.isValid() else 0

    def data(self, index: QModelIndex, role=Qt.DisplayRole) -> QVariant:
        """Return data for a specific index and role."""
        if not index.isValid():
            return QVariant()

        row = index.row()
        col = index.column()

        if role == Qt.DisplayRole:
            value = self._display_dataframe.iloc[row, col]
            if pd.isna(value):
                return "NA"
            try:
                if isinstance(value, (float, pd.Float64Dtype)):
                     return "{:.4f}".format(value)
                return str(value)
            except Exception as e:
                 logger.debug(f"Error converting data for display at ({row},{col}): {e}")
                 return "Error" # Or some other placeholder

        elif role == Qt.BackgroundRole:
             if pd.isna(self._display_dataframe.iloc[row, col]):
                 return QBrush(QColor(255, 230, 230)) # Light red background for NA


        elif role == Qt.ToolTipRole:
            value = self._display_dataframe.iloc[row, col]
            return f"Row: {row}, Col: {col}\nValue: {value}\nType: {type(value)}"

        return QVariant()

    def headerData(self, section: int, orientation: Qt.Orientation, role=Qt.DisplayRole) -> QVariant:
        """Return header data."""
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._display_dataframe.columns[section])
            if orientation == Qt.Vertical:
                return str(self._display_dataframe.index[section]) # Show original index if needed
        return QVariant()

    def sort(self, column: int, order=Qt.AscendingOrder):
        """Sort the table by a column."""
        try:
            col_name = self._display_dataframe.columns[column]
            self.layoutAboutToBeChanged.emit()
            ascending = (order == Qt.AscendingOrder)
            self._display_dataframe = self._display_dataframe.sort_values(by=col_name, ascending=ascending)
            # Note: Sorting might break alignment with original index if index is important
            self.layoutChanged.emit()
            logger.debug(f"Sorted by column '{col_name}', order: {'Asc' if ascending else 'Desc'}")
        except Exception as e:
            logger.error(f"Error sorting DataFrame: {e}", exc_info=True)


    def dataframe(self):
        """Return the original underlying DataFrame."""
        return self._dataframe

    def set_dataframe(self, dataframe: pd.DataFrame):
         """Set a new DataFrame."""
         self.layoutAboutToBeChanged.emit()
         self._dataframe = dataframe if dataframe is not None else pd.DataFrame()
         if len(self._dataframe) > MAX_ROWS_TO_DISPLAY:
              logger.warning(f"DataFrame has {len(self._dataframe)} rows. Display limited to first {MAX_ROWS_TO_DISPLAY} rows for performance.")
              self._display_dataframe = self._dataframe.head(MAX_ROWS_TO_DISPLAY).copy()
              self._is_limited = True
         else:
              self._display_dataframe = self._dataframe.copy()
              self._is_limited = False
         self.layoutChanged.emit()


class DataViewerWidget(QWidget):
    """Widget to display a pandas DataFrame with filtering."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df = pd.DataFrame() # Store the original full DataFrame

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)

        self.info_layout = QHBoxLayout()
        self.row_label = QLabel("Rows: 0")
        self.col_label = QLabel("Cols: 0")
        self.limit_label = QLabel("") # To show if view is limited
        self.info_layout.addWidget(self.row_label)
        self.info_layout.addWidget(self.col_label)
        self.info_layout.addStretch()
        self.info_layout.addWidget(self.limit_label)
        self.layout.addLayout(self.info_layout)

        self.filter_layout = QHBoxLayout()
        self.filter_label = QLabel("Filter:")
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter data (case-insensitive)...")
        self.filter_input.textChanged.connect(self.apply_filter)
        self.filter_layout.addWidget(self.filter_label)
        self.filter_layout.addWidget(self.filter_input)
        self.layout.addLayout(self.filter_layout)

        self.table_view = QTableView()
        self.table_view.setSortingEnabled(True)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.setSelectionMode(QTableView.SingleSelection) # Or ExtendedSelection
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive) # Allow resize
        self.table_view.verticalHeader().setVisible(True) # Show row numbers (index)
        self.layout.addWidget(self.table_view)

        self.model = PandasModel()
        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.model)
        self.proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.proxy_model.setFilterKeyColumn(-1)  # Filter across all columns
        self.table_view.setModel(self.proxy_model)

    def set_dataframe(self, df: pd.DataFrame):
        """Sets the DataFrame to be displayed."""
        if df is None:
             df = pd.DataFrame() # Handle None case

        self._df = df # Store original
        self.model.set_dataframe(df) # Model handles limiting display
        self.update_info_labels()
        
        if hasattr(self, 'table_view') and self.table_view is not None:
            try:
                self.table_view.resizeColumnsToContents() # Adjust columns initially
            except RuntimeError:
                logger.warning("QTableView has been deleted - skipping column resize")
                pass
        
        logger.info(f"DataFrame set in viewer. Rows: {len(df)}, Cols: {len(df.columns)}")

    def update_info_labels(self):
        """Updates the row/column count labels."""
        try:
            actual_rows, actual_cols = self._df.shape
            display_rows = self.model.rowCount()

            if hasattr(self, 'row_label') and self.row_label is not None:
                try:
                    self.row_label.setText(f"Total Rows: {actual_rows}")
                except RuntimeError:
                    pass
            
            if hasattr(self, 'col_label') and self.col_label is not None:
                try:
                    self.col_label.setText(f"Total Cols: {actual_cols}")
                except RuntimeError:
                    pass

            if hasattr(self, 'limit_label') and self.limit_label is not None and self.model._is_limited:
                try:
                    self.limit_label.setText(f"(Displaying first {display_rows})")
                    self.limit_label.setStyleSheet("QLabel { color : orange; }")
                except RuntimeError:
                    pass
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error updating info labels: {e}")
        
        if hasattr(self, 'limit_label') and self.limit_label is not None and not self.model._is_limited:
            try:
                self.limit_label.setText("")
                self.limit_label.setStyleSheet("")
            except RuntimeError:
                pass


    def apply_filter(self, text: str):
        """Applies the filter text to the proxy model."""
        search = QRegExp(text, Qt.CaseInsensitive, QRegExp.Wildcard)
        self.proxy_model.setFilterRegExp(search)
