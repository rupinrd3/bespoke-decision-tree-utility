# utility/ui/dialogs/filter_data_dialog.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dialog for Data Filtering Configuration in Bespoke Utility.
Allows users to define and apply multiple filter conditions to a dataset.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QDialogButtonBox, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QComboBox, QLineEdit, QMenu, QAction,
    QWidget, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal

logger = logging.getLogger(__name__)

class FilterConditionWidget(QWidget):
    """A widget representing a single filter condition."""
    conditionChanged = pyqtSignal()
    requestRemove = pyqtSignal(object) # self

    def __init__(self, available_columns: List[str], column_types: Dict[str, str], parent=None):
        super().__init__(parent)
        self.available_columns = available_columns
        self.column_types = column_types # e.g. {'Age': 'numeric', 'Gender': 'categorical'}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)

        self.logic_combo = QComboBox() # AND, OR (for conditions > 0)
        self.logic_combo.addItems(["AND", "OR"])
        self.logic_combo.setVisible(False) # Only visible for second condition onwards
        layout.addWidget(self.logic_combo)

        self.column_combo = QComboBox()
        self.column_combo.addItems(self.available_columns)
        layout.addWidget(self.column_combo)

        self.operator_combo = QComboBox()
        layout.addWidget(self.operator_combo)

        self.value_edit = QLineEdit()
        self.value_edit.setPlaceholderText("Enter value")
        layout.addWidget(self.value_edit)
        
        self.value_dropdown = QComboBox() # For categorical unique values
        self.value_dropdown.setVisible(False)
        layout.addWidget(self.value_dropdown)


        self.remove_button = QPushButton("X")
        self.remove_button.setToolTip("Remove this condition")
        self.remove_button.setFixedWidth(30)
        layout.addWidget(self.remove_button)

        self.column_combo.currentTextChanged.connect(self._update_operators_and_values)
        self.operator_combo.currentTextChanged.connect(self._update_value_input_visibility)
        self.remove_button.clicked.connect(lambda: self.requestRemove.emit(self))

        if self.available_columns:
            self._update_operators_and_values(self.available_columns[0])

    def _update_operators_and_values(self, column_name: str):
        """Update operators and value input based on column type and unique values."""
        self.operator_combo.clear()
        col_type = self.column_types.get(column_name, "object") # Default to object/string like

        numeric_ops = ["=", "!=", ">", "<", ">=", "<="]
        string_ops = ["=", "!=", "contains", "starts with", "ends with"]
        common_ops = ["is null", "is not null"] # Applicable to all

        if "int" in col_type or "float" in col_type or "num" in col_type.lower(): # Numeric
            self.operator_combo.addItems(numeric_ops + common_ops)
            self.value_edit.setVisible(True)
            self.value_dropdown.setVisible(False)
            self.value_edit.setPlaceholderText("Enter numeric value")
        elif "date" in col_type.lower(): # Datetime (treat as string/specific input for now)
            self.operator_combo.addItems(string_ops + common_ops + [">", "<", ">=", "<="]) # Add date comparisons
            self.value_edit.setVisible(True)
            self.value_dropdown.setVisible(False)
            self.value_edit.setPlaceholderText("Enter date (e.g., YYYY-MM-DD)")
        else: # Categorical/Object/String
            self.operator_combo.addItems(string_ops + common_ops)
            self.value_edit.setVisible(True)
            self.value_dropdown.setVisible(False) # Keep simple for now
            self.value_edit.setPlaceholderText("Enter text value")

        self._update_value_input_visibility(self.operator_combo.currentText())
        self.conditionChanged.emit()

    def _update_value_input_visibility(self, operator: str):
        """Show/hide value input based on operator."""
        if operator in ["is null", "is not null"]:
            self.value_edit.setVisible(False)
            self.value_dropdown.setVisible(False)
        else:
            current_column = self.column_combo.currentText()
            col_type = self.column_types.get(current_column, "object")
            self.value_edit.setVisible(True)
            self.value_dropdown.setVisible(False)
            
        self.conditionChanged.emit()


    def get_condition(self) -> Optional[Dict[str, Any]]:
        """Returns the filter condition as a dictionary."""
        column = self.column_combo.currentText()
        operator = self.operator_combo.currentText()
        logic = self.logic_combo.currentText() if self.logic_combo.isVisible() else "AND" # Default to AND for the first

        if not column or not operator:
            return None

        value = None
        if operator not in ["is null", "is not null"]:
            if self.value_edit.isVisible():
                value_text = self.value_edit.text()
                col_type = self.column_types.get(column, "object")
                if "int" in col_type or "float" in col_type:
                    try:
                        value = float(value_text)
                        if value == int(value): value = int(value) # Convert to int if whole
                    except ValueError:
                        value = value_text
                else:
                     value = value_text

            elif self.value_dropdown.isVisible(): # If dropdown for categoricals is used
                value = self.value_dropdown.currentText()
        
        return {
            "logic": logic, # Added logic
            "column": column,
            "operator": operator,
            "value": value
        }

    def set_condition(self, condition: Dict[str, Any], is_first: bool):
        """Sets the condition in the widget."""
        self.logic_combo.setCurrentText(condition.get("logic", "AND"))
        self.logic_combo.setVisible(not is_first)

        self.column_combo.setCurrentText(condition.get("column", ""))
        self.operator_combo.setCurrentText(condition.get("operator", ""))
        
        op = condition.get("operator", "")
        if op not in ["is null", "is not null"]:
            self.value_edit.setText(str(condition.get("value", "")))
        
        self._update_value_input_visibility(op)


class FilterDataDialog(QDialog):
    """Dialog for configuring data filtering with multiple conditions."""

    def __init__(self, dataframe: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Data")
        self.setMinimumWidth(700) # Increased width
        self.setMinimumHeight(400)

        if dataframe is None or dataframe.empty:
            QMessageBox.warning(self, "No Data", "Cannot filter an empty dataset.")
            self.parent_df_empty = True # Flag
            return
        else:
            self.parent_df_empty = False


        self.dataframe = dataframe
        self.available_columns = sorted(dataframe.columns.tolist())
        self.column_types = {col: str(dataframe[col].dtype) for col in dataframe.columns}

        self.conditions: List[FilterConditionWidget] = []

        self.init_ui()
        self._add_condition_widget() # Add the first condition row

    def init_ui(self):
        """Initialize the user interface."""
        self.main_layout = QVBoxLayout(self) # Main layout for the dialog

        self.conditions_widget_container = QWidget() # Container for condition widgets
        self.conditions_layout = QVBoxLayout(self.conditions_widget_container)
        self.conditions_layout.setContentsMargins(0,0,0,0)

        self.main_layout.addWidget(QLabel("Define filter conditions:"))
        self.main_layout.addWidget(self.conditions_widget_container)

        self.add_condition_button = QPushButton("Add Condition")
        self.add_condition_button.clicked.connect(self._add_condition_widget)
        self.main_layout.addWidget(self.add_condition_button, 0, Qt.AlignLeft)
        
        self.main_layout.addStretch(1) # Pushes buttons to the bottom



        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept_options)
        self.button_box.rejected.connect(self.reject)
        self.main_layout.addWidget(self.button_box)
        
        if self.parent_df_empty: # If dataframe was empty, schedule a close
            self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)


    def _add_condition_widget(self, condition_data: Optional[Dict[str, Any]] = None):
        """Adds a new FilterConditionWidget to the layout."""
        is_first_condition = not self.conditions
        condition_w = FilterConditionWidget(self.available_columns, self.column_types)
        condition_w.logic_combo.setVisible(not is_first_condition)

        if condition_data:
            condition_w.set_condition(condition_data, is_first_condition)

        condition_w.requestRemove.connect(self._remove_condition_widget)

        self.conditions_layout.addWidget(condition_w)
        self.conditions.append(condition_w)

    def _remove_condition_widget(self, widget_to_remove: FilterConditionWidget):
        """Removes a specific condition widget."""
        if widget_to_remove in self.conditions:
            self.conditions.remove(widget_to_remove)
            self.conditions_layout.removeWidget(widget_to_remove)
            widget_to_remove.deleteLater()

            if self.conditions:
                self.conditions[0].logic_combo.setVisible(False)
            
    def accept_options(self):
        """Process and accept filter options."""
        self.filter_conditions_data = []
        valid_conditions = True
        for i, cond_widget in enumerate(self.conditions):
            condition = cond_widget.get_condition()
            if condition:
                if i == 0: condition.pop("logic") # First condition doesn't need preceding logic
                self.filter_conditions_data.append(condition)
            else:
                QMessageBox.warning(self, "Incomplete Condition", f"Condition {i+1} is not fully specified.")
                valid_conditions = False
                break
        
        if not self.filter_conditions_data and self.conditions: # Had rows, but none were valid
             QMessageBox.warning(self, "No Conditions", "Please define at least one valid filter condition or cancel.")
             return

        if valid_conditions:
            if not self.filter_conditions_data:
                logger.info("No filter conditions defined by user.")
            else:
                logger.info(f"Filter conditions accepted: {self.filter_conditions_data}")
            self.accept()


    def get_conditions(self) -> List[Dict[str, Any]]:
        """Return the configured filter conditions."""
        return getattr(self, 'filter_conditions_data', [])

        

        # TODO: Implement preview logic


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)

    data = {
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 35, 28, 22],
        'City': ['New York', 'London', 'Paris', 'New York', 'Berlin'],
        'Salary': [70000, 80000, 90000, 60000, 50000],
        'IsSubscribed': [True, False, True, True, False]
    }
    sample_df = pd.DataFrame(data)
    sample_df['HireDate'] = pd.to_datetime(['2020-01-15', '2019-03-22', '2021-07-30', '2020-05-10', '2022-11-05'])


    dialog = FilterDataDialog(dataframe=sample_df)
    if hasattr(dialog, 'parent_df_empty') and dialog.parent_df_empty: # Check if init failed due to empty df
        logger.warning("Dialog could not be shown because the DataFrame was empty.")
    elif dialog.exec_() == QDialog.Accepted:
        print("Filter Conditions:", dialog.get_conditions())
    else:
        print("Dialog cancelled.")
    sys.exit(app.exec_())