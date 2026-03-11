# utility/ui/dialogs/csv_options_dialog.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CSV Import Options Dialog for Bespoke Utility
"""

import logging
from PyQt5.QtCore import QRegularExpression
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QSpinBox, QComboBox, QDialogButtonBox, QLabel, QMessageBox
)
from PyQt5.QtGui import QRegularExpressionValidator

logger = logging.getLogger(__name__)


def _decode_escape(value: str) -> str:
    """Decode user-entered escape sequences like \\t into their literal values."""
    if not value:
        return value
    try:
        return bytes(value, 'utf-8').decode('unicode_escape')
    except Exception as exc:
        logger.debug(f"Could not decode value '{value}': {exc}")
        return value

class CsvImportOptionsDialog(QDialog):
    """Dialog for configuring CSV import options."""

    def __init__(self, default_encoding='utf-8', default_delimiter=',', parent=None):
        super().__init__(parent)
        self.setWindowTitle("CSV Import Options")
        self.setMinimumWidth(350)

        self.options = {
            'encoding': default_encoding,
            'delimiter': default_delimiter,
            'header': 0,  # Default header row index
            'use_chunks': True,  # Default to chunking for safety
            'quotechar': '"',
            'escapechar': '',
            'skiprows': 0,
            'skipfooter': 0,
            'decimal': '.',
            'thousands': '',
            'date_columns': [],
            'date_format': ''
        }

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

        char_pattern = QRegularExpression(r"^$|^.$|^\\.$")
        self.char_validator = QRegularExpressionValidator(char_pattern, self)

        # Encoding
        self.encoding_combo = QComboBox()
        common_encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        self.encoding_combo.addItems(common_encodings)
        if default_encoding in common_encodings:
            self.encoding_combo.setCurrentText(default_encoding)
        else:
             self.encoding_combo.addItem(default_encoding)
             self.encoding_combo.setCurrentText(default_encoding)
        self.encoding_combo.setEditable(True)
        form_layout.addRow("Encoding:", self.encoding_combo)

        self.delimiter_edit = QLineEdit(default_delimiter)
        form_layout.addRow("Delimiter:", self.delimiter_edit)

        self.quote_char_edit = QLineEdit('"')
        self.quote_char_edit.setMaxLength(2)
        self.quote_char_edit.setValidator(self.char_validator)
        self.quote_char_edit.setPlaceholderText('e.g. "')
        form_layout.addRow("Quote Character:", self.quote_char_edit)

        self.escape_char_edit = QLineEdit("")
        self.escape_char_edit.setMaxLength(2)
        self.escape_char_edit.setValidator(self.char_validator)
        self.escape_char_edit.setPlaceholderText("Optional (e.g. \\)")
        form_layout.addRow("Escape Character:", self.escape_char_edit)

        self.header_spinbox = QSpinBox()
        self.header_spinbox.setRange(0, 100)  # Header row index (0-based)
        self.header_spinbox.setValue(0)
        self.header_spinbox.setToolTip("Row number containing header (0-based).")
        form_layout.addRow("Header Row:", self.header_spinbox)

        self.skiprows_spinbox = QSpinBox()
        self.skiprows_spinbox.setRange(0, 100000)
        form_layout.addRow("Skip Rows (Top):", self.skiprows_spinbox)

        self.skipfooter_spinbox = QSpinBox()
        self.skipfooter_spinbox.setRange(0, 100000)
        form_layout.addRow("Skip Rows (Bottom):", self.skipfooter_spinbox)

        self.decimal_edit = QLineEdit(".")
        self.decimal_edit.setMaxLength(2)
        self.decimal_edit.setPlaceholderText("Default: .")
        self.decimal_edit.setValidator(self.char_validator)
        form_layout.addRow("Decimal Separator:", self.decimal_edit)

        self.thousands_edit = QLineEdit("")
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

        date_hint = QLabel("Leave format blank to let pandas infer; specify columns to convert.")
        date_hint.setWordWrap(True)
        form_layout.addRow("", date_hint)

        layout.addLayout(form_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept_options)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def accept_options(self):
        """Update options dictionary before accepting."""
        self.options['encoding'] = self.encoding_combo.currentText()
        self.options['delimiter'] = _decode_escape(self.delimiter_edit.text())
        self.options['header'] = self.header_spinbox.value()

        quote_val = self.quote_char_edit.text()
        self.options['quotechar'] = quote_val if quote_val else None

        escape_val = self.escape_char_edit.text()
        self.options['escapechar'] = escape_val or ''

        self.options['skiprows'] = self.skiprows_spinbox.value()
        self.options['skipfooter'] = self.skipfooter_spinbox.value()

        decimal_val = _decode_escape(self.decimal_edit.text()) or '.'
        self.options['decimal'] = decimal_val

        thousands_val = _decode_escape(self.thousands_edit.text())

        if decimal_val and thousands_val and decimal_val == thousands_val:
            QMessageBox.warning(self, "Validation Error", "Decimal and thousands separators must be different.")
            return

        self.options['thousands'] = thousands_val or ''

        date_columns_text = self.date_columns_edit.text().strip()
        if date_columns_text:
            date_columns = [col.strip() for col in date_columns_text.split(',') if col.strip()]
        else:
            date_columns = []
        self.options['date_columns'] = date_columns
        self.options['date_format'] = self.date_format_edit.text().strip()

        logger.debug(f"CSV Import options set: {self.options}")
        self.accept()

    def get_options(self):
        """Return the selected options."""
        return self.options
