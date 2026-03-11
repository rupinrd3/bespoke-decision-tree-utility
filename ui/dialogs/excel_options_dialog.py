# utility/ui/dialogs/excel_options_dialog.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Excel Import Options Dialog for Bespoke Utility
"""

import logging
import pandas as pd
from PyQt5.QtCore import QRegularExpression
from PyQt5.QtGui import QRegularExpressionValidator
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QSpinBox, QComboBox, QDialogButtonBox, QMessageBox, QLabel
)

logger = logging.getLogger(__name__)


def _decode_escape(value: str) -> str:
    """Decode escape sequences (e.g., \t) for numeric separators."""
    if not value:
        return value
    try:
        return bytes(value, 'utf-8').decode('unicode_escape')
    except Exception as exc:
        logger.debug(f"Could not decode value '{value}': {exc}")
        return value

class ExcelImportOptionsDialog(QDialog):
    """Dialog for configuring Excel import options."""

    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Excel Import Options")
        self.setMinimumWidth(350)
        self.file_path = file_path

        self.options = {
            'sheet_name': 0,  # Default to first sheet
            'header': 0,      # Default header row index
            'skiprows': 0,
            'skipfooter': 0,
            'decimal': '.',
            'thousands': '',
            'date_columns': [],
            'date_format': ''
        }

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        self.sheet_combo = QComboBox()
        try:
            excel_file = pd.ExcelFile(file_path)
            self.sheet_names = excel_file.sheet_names
            self.sheet_combo.addItems(self.sheet_names)
        except Exception as e:
            logger.error(f"Error reading sheet names from {file_path}: {e}")
            QMessageBox.warning(self, "Read Error", f"Could not read sheet names from the Excel file.\n{e}\n\nPlease ensure the file is valid.")
            self.sheet_names = ["Sheet1"] # Provide a default
            self.sheet_combo.addItems(self.sheet_names)

        form_layout.addRow("Sheet Name:", self.sheet_combo)

        self.header_spinbox = QSpinBox()
        self.header_spinbox.setRange(0, 100)  # Header row index (0-based)
        self.header_spinbox.setValue(0)
        self.header_spinbox.setToolTip("Row number containing header (0-based).")
        form_layout.addRow("Header Row:", self.header_spinbox)

        char_pattern = QRegularExpression(r"^$|^.$|^\\.$")
        self.char_validator = QRegularExpressionValidator(char_pattern, self)

        self.skiprows_spinbox = QSpinBox()
        self.skiprows_spinbox.setRange(0, 100000)
        form_layout.addRow("Skip Rows (Top):", self.skiprows_spinbox)

        self.skipfooter_spinbox = QSpinBox()
        self.skipfooter_spinbox.setRange(0, 100000)
        form_layout.addRow("Skip Rows (Bottom):", self.skipfooter_spinbox)

        self.decimal_edit = QLineEdit('.')
        self.decimal_edit.setMaxLength(2)
        self.decimal_edit.setPlaceholderText("Default: .")
        self.decimal_edit.setValidator(self.char_validator)
        form_layout.addRow("Decimal Separator:", self.decimal_edit)

        self.thousands_edit = QLineEdit('')
        self.thousands_edit.setMaxLength(2)
        self.thousands_edit.setPlaceholderText("Optional (e.g. ,)")
        self.thousands_edit.setValidator(self.char_validator)
        form_layout.addRow("Thousands Separator:", self.thousands_edit)

        self.date_columns_edit = QLineEdit()
        self.date_columns_edit.setPlaceholderText("Comma separated, e.g. order_date,ship_date")
        form_layout.addRow("Date Columns:", self.date_columns_edit)

        self.date_format_edit = QLineEdit()
        self.date_format_edit.setPlaceholderText("Optional strftime format, e.g. %Y-%m-%d")
        form_layout.addRow("Date Format:", self.date_format_edit)

        date_hint = QLabel("Leave format blank to auto-detect; provide columns to coerce after load.")
        date_hint.setWordWrap(True)
        form_layout.addRow("", date_hint)

        layout.addLayout(form_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept_options)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def accept_options(self):
        """Update options dictionary before accepting."""
        selected_sheet_text = self.sheet_combo.currentText()
        if selected_sheet_text in self.sheet_names:
             self.options['sheet_name'] = selected_sheet_text
        else:
             self.options['sheet_name'] = self.sheet_combo.currentIndex()

        self.options['header'] = self.header_spinbox.value()
        self.options['skiprows'] = self.skiprows_spinbox.value()
        self.options['skipfooter'] = self.skipfooter_spinbox.value()
        decimal_val = _decode_escape(self.decimal_edit.text()) or '.'
        thousands_val = _decode_escape(self.thousands_edit.text())

        if decimal_val and thousands_val and decimal_val == thousands_val:
            QMessageBox.warning(self, "Validation Error", "Decimal and thousands separators must be different.")
            return

        self.options['decimal'] = decimal_val
        self.options['thousands'] = thousands_val or ''

        date_columns_text = self.date_columns_edit.text().strip()
        if date_columns_text:
            date_columns = [col.strip() for col in date_columns_text.split(',') if col.strip()]
        else:
            date_columns = []
        self.options['date_columns'] = date_columns
        self.options['date_format'] = self.date_format_edit.text().strip()
        logger.debug(f"Excel Import options set: {self.options}")
        self.accept()

    def get_options(self):
        """Return the selected options."""
        return self.options
