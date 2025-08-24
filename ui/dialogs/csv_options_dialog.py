# utility/ui/dialogs/csv_options_dialog.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CSV Import Options Dialog for Bespoke Utility
"""

import logging
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLineEdit,
                           QSpinBox, QComboBox, QPushButton, QDialogButtonBox,
                           QCheckBox)

logger = logging.getLogger(__name__)

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
            'use_chunks': True # Default to chunking for safety
        }

        layout = QVBoxLayout(self)
        form_layout = QFormLayout()

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

        self.header_spinbox = QSpinBox()
        self.header_spinbox.setRange(0, 100)  # Header row index (0-based)
        self.header_spinbox.setValue(0)
        self.header_spinbox.setToolTip("Row number containing header (0-based).")
        form_layout.addRow("Header Row:", self.header_spinbox)


        layout.addLayout(form_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept_options)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def accept_options(self):
        """Update options dictionary before accepting."""
        self.options['encoding'] = self.encoding_combo.currentText()
        delimiter = self.delimiter_edit.text()
        delimiter = delimiter.replace('\\t', '\t')
        self.options['delimiter'] = delimiter
        self.options['header'] = self.header_spinbox.value()
        logger.debug(f"CSV Import options set: {self.options}")
        self.accept()

    def get_options(self):
        """Return the selected options."""
        return self.options
