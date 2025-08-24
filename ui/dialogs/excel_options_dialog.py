# utility/ui/dialogs/excel_options_dialog.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Excel Import Options Dialog for Bespoke Utility
"""

import logging
import pandas as pd
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QLineEdit,
                           QSpinBox, QComboBox, QPushButton, QDialogButtonBox,
                           QMessageBox)

logger = logging.getLogger(__name__)

class ExcelImportOptionsDialog(QDialog):
    """Dialog for configuring Excel import options."""

    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Excel Import Options")
        self.setMinimumWidth(350)
        self.file_path = file_path

        self.options = {
            'sheet_name': 0,  # Default to first sheet
            'header': 0     # Default header row index
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
        logger.debug(f"Excel Import options set: {self.options}")
        self.accept()

    def get_options(self):
        """Return the selected options."""
        return self.options
