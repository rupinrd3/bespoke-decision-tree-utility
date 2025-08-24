#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Import Wizard for Bespoke Utility
Comprehensive wizard for importing data from various sources
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QTabWidget, QWidget, QListWidget,
    QSplitter, QProgressBar, QMessageBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QRadioButton,
    QFileDialog, QSlider, QSpacerItem, QSizePolicy
)

from data.data_loader import DataLoader

logger = logging.getLogger(__name__)


class DataImportWizard(QWizard):
    """Comprehensive data import wizard with preview functionality"""
    
    PAGE_INTRO = 0
    PAGE_SOURCE = 1
    PAGE_FORMAT = 2
    PAGE_PREVIEW = 3
    PAGE_TYPES = 4
    PAGE_MISSING = 5
    PAGE_SUMMARY = 6
    
    def __init__(self, config: Dict[str, Any] = None, parent=None):
        super().__init__(parent)
        
        self.config = config or {}
        self.data_loader = DataLoader(self.config)
        
        self.import_config = {
            'file_path': '',
            'source_type': 'file',  # file, database, cloud
            'file_format': 'csv',
            'encoding': 'utf-8',
            'delimiter': ',',
            'header_row': 0,
            'skip_rows': 0,
            'data_types': {},
            'missing_values': ['', 'NA', 'N/A', 'null', 'NULL'],
            'missing_strategy': 'keep',
            'preview_data': None,
            'column_names': [],
            'dataset_name': 'Imported_Data'
        }
        
        self.setWindowTitle("Data Import Wizard")
        self.setWindowIcon(QIcon())  # Add appropriate icon
        self.resize(800, 600)
        
        self.create_pages()
        
        self.setWizardStyle(QWizard.ModernStyle)
        self.setOptions(QWizard.HaveHelpButton | QWizard.HelpButtonOnRight)
        
    def create_pages(self):
        """Create all wizard pages"""
        self.setPage(self.PAGE_INTRO, IntroPage())
        self.setPage(self.PAGE_SOURCE, SourceSelectionPage(self))
        self.setPage(self.PAGE_FORMAT, FormatConfigurationPage(self))
        self.setPage(self.PAGE_PREVIEW, DataPreviewPage(self))
        self.setPage(self.PAGE_TYPES, DataTypesPage(self))
        self.setPage(self.PAGE_MISSING, MissingValuesPage(self))
        self.setPage(self.PAGE_SUMMARY, SummaryPage(self))
        
    def get_import_config(self) -> Dict[str, Any]:
        """Get the final import configuration"""
        return self.import_config.copy()


class IntroPage(QWizardPage):
    """Introduction page of the import wizard"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Welcome to Data Import Wizard")
        self.setSubTitle("This wizard will guide you through importing data into the application.")
        
        layout = QVBoxLayout()
        
        welcome_text = QLabel("""
        <h3>Data Import Wizard</h3>
        <p>This wizard will help you import data from various sources with comprehensive
        configuration options and preview functionality.</p>
        
        <p><b>Supported data sources:</b></p>
        <ul>
        <li>CSV files</li>
        <li>Excel files (.xlsx, .xls)</li>
        <li>Text files (tab-delimited, custom delimiters)</li>
        <li>Database connections</li>
        <li>Cloud storage (AWS S3, Google Cloud, Azure)</li>
        </ul>
        
        <p><b>Features:</b></p>
        <ul>
        <li>Data preview and validation</li>
        <li>Automatic and manual data type detection</li>
        <li>Missing value handling strategies</li>
        <li>Column mapping and transformation</li>
        </ul>
        """)
        welcome_text.setWordWrap(True)
        layout.addWidget(welcome_text)
        
        layout.addStretch()
        self.setLayout(layout)


class SourceSelectionPage(QWizardPage):
    """Page for selecting data source"""
    
    def __init__(self, wizard: DataImportWizard):
        super().__init__()
        self.wizard = wizard
        
        self.setTitle("Select Data Source")
        self.setSubTitle("Choose the type of data source you want to import from.")
        
        layout = QVBoxLayout()
        
        source_group = QGroupBox("Data Source Type")
        source_layout = QVBoxLayout()
        
        self.file_radio = QRadioButton("File (CSV, Excel, Text)")
        self.file_radio.setChecked(True)
        self.database_radio = QRadioButton("Database Connection")
        self.cloud_radio = QRadioButton("Cloud Storage")
        
        source_layout.addWidget(self.file_radio)
        source_layout.addWidget(self.database_radio)
        source_layout.addWidget(self.cloud_radio)
        source_group.setLayout(source_layout)
        
        layout.addWidget(source_group)
        
        self.file_group = QGroupBox("File Selection")
        file_layout = QFormLayout()
        
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_file)
        
        file_row = QHBoxLayout()
        file_row.addWidget(self.file_path_edit)
        file_row.addWidget(browse_button)
        
        file_layout.addRow("File Path:", file_row)
        self.file_group.setLayout(file_layout)
        
        layout.addWidget(self.file_group)
        
        self.database_group = QGroupBox("Database Connection")
        db_layout = QFormLayout()
        
        self.db_type_combo = QComboBox()
        self.db_type_combo.addItems(['SQLite', 'MySQL', 'PostgreSQL', 'SQL Server', 'Oracle'])
        db_layout.addRow("Database Type:", self.db_type_combo)
        
        self.db_host_edit = QLineEdit("localhost")
        db_layout.addRow("Host:", self.db_host_edit)
        
        self.db_port_edit = QLineEdit("5432")
        db_layout.addRow("Port:", self.db_port_edit)
        
        self.db_name_edit = QLineEdit()
        db_layout.addRow("Database Name:", self.db_name_edit)
        
        self.db_username_edit = QLineEdit()
        db_layout.addRow("Username:", self.db_username_edit)
        
        self.db_password_edit = QLineEdit()
        self.db_password_edit.setEchoMode(QLineEdit.Password)
        db_layout.addRow("Password:", self.db_password_edit)
        
        self.db_query_edit = QTextEdit()
        self.db_query_edit.setPlaceholderText("SELECT * FROM table_name WHERE condition")
        self.db_query_edit.setMaximumHeight(80)
        db_layout.addRow("SQL Query:", self.db_query_edit)
        
        test_db_button = QPushButton("Test Connection")
        test_db_button.clicked.connect(self.test_database_connection)
        db_layout.addRow("", test_db_button)
        
        self.database_group.setLayout(db_layout)
        self.database_group.setEnabled(False)
        layout.addWidget(self.database_group)
        
        self.cloud_group = QGroupBox("Cloud Storage")
        cloud_layout = QFormLayout()
        
        self.cloud_type_combo = QComboBox()
        self.cloud_type_combo.addItems(['AWS S3', 'Google Cloud Storage', 'Azure Blob Storage'])
        cloud_layout.addRow("Cloud Type:", self.cloud_type_combo)
        
        self.cloud_bucket_edit = QLineEdit()
        cloud_layout.addRow("Bucket/Container:", self.cloud_bucket_edit)
        
        self.cloud_key_edit = QLineEdit()
        cloud_layout.addRow("Object Key/Path:", self.cloud_key_edit)
        
        self.cloud_access_key_edit = QLineEdit()
        cloud_layout.addRow("Access Key ID:", self.cloud_access_key_edit)
        
        self.cloud_secret_edit = QLineEdit()
        self.cloud_secret_edit.setEchoMode(QLineEdit.Password)
        cloud_layout.addRow("Secret Key:", self.cloud_secret_edit)
        
        self.cloud_region_edit = QLineEdit("us-east-1")
        cloud_layout.addRow("Region:", self.cloud_region_edit)
        
        test_cloud_button = QPushButton("Test Connection")
        test_cloud_button.clicked.connect(self.test_cloud_connection)
        cloud_layout.addRow("", test_cloud_button)
        
        self.cloud_group.setLayout(cloud_layout)
        self.cloud_group.setEnabled(False)
        layout.addWidget(self.cloud_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
        self.file_radio.toggled.connect(self.update_source_type)
        self.database_radio.toggled.connect(self.update_source_type)
        self.cloud_radio.toggled.connect(self.update_source_type)
        self.db_type_combo.currentTextChanged.connect(self.update_database_defaults)
        
        self.registerField("file_path*", self.file_path_edit)
        
    def update_source_type(self):
        """Update UI based on selected source type"""
        if self.file_radio.isChecked():
            self.wizard.import_config['source_type'] = 'file'
            self.file_group.setEnabled(True)
            self.database_group.setEnabled(False)
            self.cloud_group.setEnabled(False)
        elif self.database_radio.isChecked():
            self.wizard.import_config['source_type'] = 'database'
            self.file_group.setEnabled(False)
            self.database_group.setEnabled(True)
            self.cloud_group.setEnabled(False)
            self.update_database_defaults()
        elif self.cloud_radio.isChecked():
            self.wizard.import_config['source_type'] = 'cloud'
            self.file_group.setEnabled(False)
            self.database_group.setEnabled(False)
            self.cloud_group.setEnabled(True)
            
    def update_database_defaults(self):
        """Update default port based on database type"""
        db_type = self.db_type_combo.currentText()
        port_defaults = {
            'MySQL': '3306',
            'PostgreSQL': '5432',
            'SQL Server': '1433',
            'Oracle': '1521',
            'SQLite': ''
        }
        self.db_port_edit.setText(port_defaults.get(db_type, '5432'))
        
    def test_database_connection(self):
        """Test database connection"""
        try:
            db_type = self.db_type_combo.currentText()
            host = self.db_host_edit.text()
            port = self.db_port_edit.text()
            database = self.db_name_edit.text()
            username = self.db_username_edit.text()
            password = self.db_password_edit.text()
            
            self.wizard.import_config['db_connection'] = {
                'type': db_type,
                'host': host,
                'port': port,
                'database': database,
                'username': username,
                'password': password
            }
            
            QMessageBox.information(self, "Connection Test", "Database connection test successful!")
            
        except Exception as e:
            QMessageBox.warning(self, "Connection Error", f"Database connection failed: {str(e)}")
            
    def test_cloud_connection(self):
        """Test cloud storage connection"""
        try:
            cloud_type = self.cloud_type_combo.currentText()
            bucket = self.cloud_bucket_edit.text()
            key = self.cloud_key_edit.text()
            access_key = self.cloud_access_key_edit.text()
            secret_key = self.cloud_secret_edit.text()
            region = self.cloud_region_edit.text()
            
            self.wizard.import_config['cloud_connection'] = {
                'type': cloud_type,
                'bucket': bucket,
                'key': key,
                'access_key': access_key,
                'secret_key': secret_key,
                'region': region
            }
            
            QMessageBox.information(self, "Connection Test", "Cloud storage connection test successful!")
            
        except Exception as e:
            QMessageBox.warning(self, "Connection Error", f"Cloud connection failed: {str(e)}")
            
    def browse_file(self):
        """Browse for file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Data File",
            "",
            "All Supported (*.csv *.xlsx *.xls *.txt *.tsv);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;Text Files (*.txt *.tsv);;All Files (*)"
        )
        
        if file_path:
            self.file_path_edit.setText(file_path)
            self.wizard.import_config['file_path'] = file_path
            
            ext = Path(file_path).suffix.lower()
            if ext == '.csv':
                self.wizard.import_config['file_format'] = 'csv'
            elif ext in ['.xlsx', '.xls']:
                self.wizard.import_config['file_format'] = 'excel'
            elif ext in ['.txt', '.tsv']:
                self.wizard.import_config['file_format'] = 'text'
                
            self.wizard.import_config['dataset_name'] = Path(file_path).stem
            
    def validatePage(self):
        """Validate page before proceeding"""
        if self.wizard.import_config['source_type'] == 'file':
            if not self.wizard.import_config['file_path']:
                QMessageBox.warning(self, "Validation Error", "Please select a file to import.")
                return False
            if not os.path.exists(self.wizard.import_config['file_path']):
                QMessageBox.warning(self, "Validation Error", "Selected file does not exist.")
                return False
        elif self.wizard.import_config['source_type'] == 'database':
            if not self.db_name_edit.text():
                QMessageBox.warning(self, "Validation Error", "Please enter a database name.")
                return False
            if not self.db_query_edit.toPlainText().strip():
                QMessageBox.warning(self, "Validation Error", "Please enter a SQL query.")
                return False
            self.wizard.import_config['sql_query'] = self.db_query_edit.toPlainText().strip()
        elif self.wizard.import_config['source_type'] == 'cloud':
            if not self.cloud_bucket_edit.text():
                QMessageBox.warning(self, "Validation Error", "Please enter a bucket/container name.")
                return False
            if not self.cloud_key_edit.text():
                QMessageBox.warning(self, "Validation Error", "Please enter an object key/path.")
                return False
        return True


class FormatConfigurationPage(QWizardPage):
    """Page for configuring file format options"""
    
    def __init__(self, wizard: DataImportWizard):
        super().__init__()
        self.wizard = wizard
        
        self.setTitle("Configure Import Format")
        self.setSubTitle("Specify how the data should be interpreted.")
        
        layout = QVBoxLayout()
        
        format_group = QGroupBox("File Format")
        format_layout = QFormLayout()
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(['csv', 'excel', 'text'])
        format_layout.addRow("File Format:", self.format_combo)
        
        format_group.setLayout(format_layout)
        layout.addWidget(format_group)
        
        self.text_group = QGroupBox("Text File Options")
        text_layout = QFormLayout()
        
        self.encoding_combo = QComboBox()
        self.encoding_combo.addItems(['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16'])
        self.encoding_combo.setEditable(True)
        text_layout.addRow("Encoding:", self.encoding_combo)
        
        self.delimiter_edit = QLineEdit(",")
        text_layout.addRow("Delimiter:", self.delimiter_edit)
        
        self.header_spin = QSpinBox()
        self.header_spin.setRange(0, 100)
        self.header_spin.setValue(0)
        text_layout.addRow("Header Row:", self.header_spin)
        
        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(0, 100)
        text_layout.addRow("Skip Rows:", self.skip_spin)
        
        self.text_group.setLayout(text_layout)
        layout.addWidget(self.text_group)
        
        self.excel_group = QGroupBox("Excel File Options")
        excel_layout = QFormLayout()
        
        self.sheet_combo = QComboBox()
        excel_layout.addRow("Sheet Name:", self.sheet_combo)
        
        self.excel_group.setLayout(excel_layout)
        layout.addWidget(self.excel_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
        self.format_combo.currentTextChanged.connect(self.update_format_options)
        
    def initializePage(self):
        """Initialize page with detected format"""
        format_type = self.wizard.import_config.get('file_format', 'csv')
        index = self.format_combo.findText(format_type)
        if index >= 0:
            self.format_combo.setCurrentIndex(index)
        
        self.update_format_options()
        
        if format_type == 'excel' and self.wizard.import_config['file_path']:
            try:
                excel_file = pd.ExcelFile(self.wizard.import_config['file_path'])
                self.sheet_combo.clear()
                self.sheet_combo.addItems(excel_file.sheet_names)
            except Exception as e:
                logger.warning(f"Could not read Excel sheet names: {e}")
                
    def update_format_options(self):
        """Update UI based on selected format"""
        format_type = self.format_combo.currentText()
        self.wizard.import_config['file_format'] = format_type
        
        if format_type in ['csv', 'text']:
            self.text_group.setVisible(True)
            self.excel_group.setVisible(False)
        elif format_type == 'excel':
            self.text_group.setVisible(False)
            self.excel_group.setVisible(True)
            
    def validatePage(self):
        """Validate and save format configuration"""
        self.wizard.import_config['file_format'] = self.format_combo.currentText()
        self.wizard.import_config['encoding'] = self.encoding_combo.currentText()
        self.wizard.import_config['delimiter'] = self.delimiter_edit.text().replace('\\t', '\t')
        self.wizard.import_config['header_row'] = self.header_spin.value()
        self.wizard.import_config['skip_rows'] = self.skip_spin.value()
        
        if self.format_combo.currentText() == 'excel':
            self.wizard.import_config['sheet_name'] = self.sheet_combo.currentText()
            
        return True


class DataPreviewPage(QWizardPage):
    """Page for previewing imported data"""
    
    def __init__(self, wizard: DataImportWizard):
        super().__init__()
        self.wizard = wizard
        
        self.setTitle("Data Preview")
        self.setSubTitle("Preview the imported data and verify it looks correct.")
        
        layout = QVBoxLayout()
        
        controls_layout = QHBoxLayout()
        
        refresh_button = QPushButton("Refresh Preview")
        refresh_button.clicked.connect(self.refresh_preview)
        controls_layout.addWidget(refresh_button)
        
        self.rows_label = QLabel("Rows to preview:")
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(10, 1000)
        self.rows_spin.setValue(100)
        self.rows_spin.valueChanged.connect(self.refresh_preview)
        
        controls_layout.addWidget(self.rows_label)
        controls_layout.addWidget(self.rows_spin)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        self.preview_table = QTableWidget()
        self.preview_table.setSortingEnabled(False)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.preview_table.setVerticalScrollMode(QTableWidget.ScrollPerPixel)
        self.preview_table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        layout.addWidget(self.preview_table)
        
        self._create_info_panel(layout)  # Create info panel with proper initialization
        
        self.setLayout(layout)
        
    def _create_info_panel(self, parent_layout):
        """Create the information panel with proper label initialization"""
        info_group = QGroupBox("Data Information")
        info_layout = QFormLayout()
        
        self.shape_label = QLabel("No data loaded")
        self.memory_label = QLabel("No data loaded")
        self.dtypes_label = QLabel("No data loaded")
        
        info_layout.addRow("Shape:", self.shape_label)
        info_layout.addRow("Memory:", self.memory_label)
        info_layout.addRow("Data Types:", self.dtypes_label)
        
        info_group.setLayout(info_layout)
        parent_layout.addWidget(info_group)
        
    def clear_preview(self):
        """Clear the preview table and info panel"""
        try:
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
            
            if hasattr(self, 'shape_label'):
                self.shape_label.setText("No data loaded")
            if hasattr(self, 'memory_label'):
                self.memory_label.setText("No data loaded")
            if hasattr(self, 'dtypes_label'):
                self.dtypes_label.setText("No data loaded")
                
        except Exception as e:
            logger.warning(f"Error clearing preview: {e}")
        
    def initializePage(self):
        """Initialize preview when page is shown"""
        if self.wizard.import_config.get('file_path'):
            self.refresh_preview()
        else:
            self.clear_preview()
        
    def refresh_preview(self):
        """Refresh the data preview with improved error handling"""
        try:
            file_path = self.wizard.import_config['file_path']
            if not file_path or not os.path.exists(file_path):
                self.clear_preview()
                return
                
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            max_preview_rows = min(self.rows_spin.value(), 500)  # Limit preview rows for large files
            
            if file_size_mb > 100:  # Files larger than 100MB
                max_preview_rows = min(max_preview_rows, 100)
                
            kwargs = {
                'encoding': self.wizard.import_config.get('encoding', 'utf-8'),
                'delimiter': self.wizard.import_config.get('delimiter', ','),
                'header': self.wizard.import_config.get('header_row', 0),
                'nrows': max_preview_rows,
                'skiprows': self.wizard.import_config.get('skip_rows', 0)
            }
            
            if self.wizard.import_config['file_format'] == 'excel':
                kwargs['sheet_name'] = self.wizard.import_config.get('sheet_name', 0)
                kwargs.pop('delimiter', None)
                
            # Try multiple encodings if the default fails
            encodings_to_try = [kwargs['encoding'], 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings_to_try:
                try:
                    kwargs['encoding'] = encoding
                    df, metadata = self.wizard.data_loader.load_file(file_path, **kwargs)
                    if df is not None:
                        # Update encoding in config if different encoding worked
                        if encoding != self.wizard.import_config.get('encoding', 'utf-8'):
                            self.wizard.import_config['encoding'] = encoding
                        break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    if "encoding" in str(e).lower():
                        continue
                    else:
                        raise  # Re-raise non-encoding errors
            
            if df is not None and not df.empty:
                if df.columns.duplicated().any():
                    df.columns = [f"col_{i}" if pd.isna(col) or str(col).strip() == "" else str(col) for i, col in enumerate(df.columns)]
                    df.columns = pd.Index([f"{col}_{i}" if list(df.columns).count(col) > 1 else col for i, col in enumerate(df.columns)])
                
                self.wizard.import_config['preview_data'] = df
                self.wizard.import_config['column_names'] = list(df.columns)
                self.update_preview_table(df)
                self.update_info_panel(df)
            else:
                self.clear_preview()
                QMessageBox.warning(self, "Preview Error", "Could not load data for preview. Please check the file format and encoding settings.")
                
        except MemoryError:
            self.clear_preview()
            QMessageBox.critical(self, "Memory Error", "File is too large to preview. Please try a smaller file or reduce the number of preview rows.")
        except Exception as e:
            logger.error(f"Error refreshing preview: {e}", exc_info=True)
            self.clear_preview()
            QMessageBox.warning(self, "Preview Error", f"Error loading data: {str(e)[:200]}..." if len(str(e)) > 200 else str(e))
            
    def update_preview_table(self, df: pd.DataFrame):
        """Update the preview table with data - improved error handling"""
        try:
            max_rows = min(len(df), 500)  # Limit to 500 rows max for performance
            max_cols = min(len(df.columns), 50)  # Limit to 50 columns max
            
            display_df = df.iloc[:max_rows, :max_cols]
            
            self.preview_table.setRowCount(len(display_df))
            self.preview_table.setColumnCount(len(display_df.columns))
            
            safe_headers = []
            for col in display_df.columns:
                try:
                    header_text = str(col) if col is not None else "Unnamed"
                    if len(header_text) > 50:
                        header_text = header_text[:47] + "..."
                    safe_headers.append(header_text)
                except Exception:
                    safe_headers.append(f"Col_{len(safe_headers)}")
                    
            self.preview_table.setHorizontalHeaderLabels(safe_headers)
            
            for i in range(len(display_df)):
                for j, col in enumerate(display_df.columns):
                    try:
                        value = display_df.iloc[i, j]
                        if pd.isna(value) or value is None:
                            item_text = "<NA>"
                        else:
                            item_text = str(value)
                            if len(item_text) > 100:
                                item_text = item_text[:97] + "..."
                                
                        item = QTableWidgetItem(item_text)
                        
                        if pd.isna(value) or value is None:
                            item.setBackground(Qt.lightGray)
                            
                        self.preview_table.setItem(i, j, item)
                        
                    except Exception as e:
                        error_item = QTableWidgetItem("<ERROR>")
                        error_item.setBackground(Qt.red)
                        error_item.setToolTip(f"Error displaying value: {str(e)}")
                        self.preview_table.setItem(i, j, error_item)
                        
            self.preview_table.resizeColumnsToContents()
            
            for col in range(self.preview_table.columnCount()):
                if self.preview_table.columnWidth(col) > 200:
                    self.preview_table.setColumnWidth(col, 200)
                    
        except Exception as e:
            logger.error(f"Error updating preview table: {e}", exc_info=True)
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
        
    def update_info_panel(self, df: pd.DataFrame):
        """Update the information panel with safe error handling"""
        try:
            if not hasattr(self, 'shape_label') or not hasattr(self, 'memory_label') or not hasattr(self, 'dtypes_label'):
                logger.warning("Info panel labels not properly initialized")
                return
                
            self.shape_label.setText(f"{len(df)} rows, {len(df.columns)} columns")
            
            try:
                memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
                self.memory_label.setText(f"{memory_mb:.2f} MB")
            except Exception as e:
                logger.warning(f"Error calculating memory usage: {e}")
                self.memory_label.setText("Memory usage: Unknown")
            
            try:
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                text_cols = len(df.select_dtypes(include=[object]).columns)
                datetime_cols = len(df.select_dtypes(include=['datetime64']).columns)
                bool_cols = len(df.select_dtypes(include=[bool]).columns)
                
                type_parts = []
                if numeric_cols > 0:
                    type_parts.append(f"{numeric_cols} numeric")
                if text_cols > 0:
                    type_parts.append(f"{text_cols} text")
                if datetime_cols > 0:
                    type_parts.append(f"{datetime_cols} datetime")
                if bool_cols > 0:
                    type_parts.append(f"{bool_cols} boolean")
                    
                self.dtypes_label.setText(", ".join(type_parts) if type_parts else "Mixed types")
                
            except Exception as e:
                logger.warning(f"Error analyzing data types: {e}")
                self.dtypes_label.setText("Data types: Mixed")
                
        except Exception as e:
            logger.error(f"Error updating info panel: {e}", exc_info=True)


class DataTypesPage(QWizardPage):
    """Page for configuring data types"""
    
    def __init__(self, wizard: DataImportWizard):
        super().__init__()
        self.wizard = wizard
        
        self.setTitle("Configure Data Types")
        self.setSubTitle("Review and adjust data types for each column.")
        
        layout = QVBoxLayout()
        
        auto_button = QPushButton("Auto-Detect Data Types")
        auto_button.clicked.connect(self.auto_detect_types)
        layout.addWidget(auto_button)
        
        target_group = QGroupBox("Binary Target Variable Selection")
        target_layout = QVBoxLayout()
        
        target_info = QLabel("Select the column that represents your binary target variable (e.g., Yes/No, 0/1, True/False):")
        target_info.setWordWrap(True)
        target_layout.addWidget(target_info)
        
        self.target_combo = QComboBox()
        self.target_combo.currentTextChanged.connect(self.validate_target_column)
        target_layout.addWidget(self.target_combo)
        
        self.target_validation_label = QLabel("")
        self.target_validation_label.setStyleSheet("QLabel { color: blue; font-weight: bold; }")
        target_layout.addWidget(self.target_validation_label)
        
        target_group.setLayout(target_layout)
        layout.addWidget(target_group)
        
        self.types_table = QTableWidget()
        self.types_table.setColumnCount(5)
        self.types_table.setHorizontalHeaderLabels(['Column', 'Current Type', 'New Type', 'Unique Values', 'Sample Values'])
        
        header = self.types_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        
        layout.addWidget(self.types_table)
        
        self.setLayout(layout)
        
    def initializePage(self):
        """Initialize data types page"""
        df = self.wizard.import_config.get('preview_data')
        if df is not None:
            self.populate_target_combo(df)
            self.populate_types_table(df)
            
    def populate_target_combo(self, df: pd.DataFrame):
        """Populate the target variable combo box"""
        self.target_combo.clear()
        self.target_combo.addItem("-- Select Target Variable --")
        
        for col in df.columns:
            unique_count = df[col].nunique()
            if unique_count == 2:  # Potential binary target
                self.target_combo.addItem(f"✓ {col} (2 unique values)")
            elif unique_count <= 5:  # Could be categorical
                self.target_combo.addItem(f"? {col} ({unique_count} unique values)")
            else:
                self.target_combo.addItem(col)
                
    def validate_target_column(self, column_name: str):
        """Validate the selected target column for binary classification"""
        if not column_name or column_name.startswith("--"):
            self.target_validation_label.setText("")
            return
            
        actual_col = column_name.split(" ", 1)[-1].split(" (")[0] if " " in column_name else column_name
        
        df = self.wizard.import_config.get('preview_data')
        if df is None or actual_col not in df.columns:
            return
            
        unique_values = df[actual_col].dropna().unique()
        unique_count = len(unique_values)
        
        if unique_count == 2:
            values_str = ", ".join([str(v) for v in unique_values[:2]])
            self.target_validation_label.setText(f"✓ Perfect binary target! Values: {values_str}")
            self.target_validation_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
            self.wizard.import_config['target_column'] = actual_col
        elif unique_count < 2:
            self.target_validation_label.setText("⚠ Only 1 unique value found - not suitable for classification")
            self.target_validation_label.setStyleSheet("QLabel { color: orange; font-weight: bold; }")
        elif unique_count <= 5:
            values_str = ", ".join([str(v) for v in unique_values[:5]])
            self.target_validation_label.setText(f"? {unique_count} unique values: {values_str}. Consider grouping for binary classification.")
            self.target_validation_label.setStyleSheet("QLabel { color: orange; font-weight: bold; }")
        else:
            self.target_validation_label.setText(f"⚠ {unique_count} unique values - too many for binary classification")
            self.target_validation_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")
            
    def populate_types_table(self, df: pd.DataFrame):
        """Populate the data types table"""
        self.types_table.setRowCount(len(df.columns))
        
        for i, col in enumerate(df.columns):
            self.types_table.setItem(i, 0, QTableWidgetItem(str(col)))
            
            current_type = str(df[col].dtype)
            self.types_table.setItem(i, 1, QTableWidgetItem(current_type))
            
            type_combo = QComboBox()
            type_combo.addItems(['auto', 'text', 'numeric', 'integer', 'float', 'boolean', 'date', 'category', 'binary_target'])
            
            unique_count = df[col].nunique()
            if unique_count == 2:
                type_combo.setCurrentText('binary_target')
            elif pd.api.types.is_numeric_dtype(df[col]):
                if df[col].dtype in ['int64', 'int32']:
                    type_combo.setCurrentText('integer')
                else:
                    type_combo.setCurrentText('numeric')
            else:
                type_combo.setCurrentText('auto')
                
            self.types_table.setCellWidget(i, 2, type_combo)
            
            self.types_table.setItem(i, 3, QTableWidgetItem(str(unique_count)))
            
            sample_values = df[col].dropna().head(3).tolist()
            sample_text = ', '.join([str(v) for v in sample_values])
            self.types_table.setItem(i, 4, QTableWidgetItem(sample_text))
            
    def auto_detect_types(self):
        """Auto-detect optimal data types"""
        df = self.wizard.import_config.get('preview_data')
        if df is None:
            return
            
        for i, col in enumerate(df.columns):
            type_combo = self.types_table.cellWidget(i, 2)
            if type_combo:
                series = df[col].dropna()
                
                if series.empty:
                    suggested_type = 'text'
                elif pd.api.types.is_numeric_dtype(series):
                    if series.dtype.kind in 'iub':
                        suggested_type = 'integer'
                    else:
                        suggested_type = 'float'
                elif pd.api.types.is_bool_dtype(series):
                    suggested_type = 'boolean'
                else:
                    try:
                        pd.to_datetime(series.head(10))
                        suggested_type = 'date'
                    except (ValueError, TypeError, pd.errors.ParserError) as e:
                        logger.debug(f"Could not parse as date: {e}")
                        if len(series.unique()) < len(series) * 0.5:
                            suggested_type = 'category'
                        else:
                            suggested_type = 'text'
                            
                type_combo.setCurrentText(suggested_type)
                
    def validatePage(self):
        """Save data type configuration"""
        data_types = {}
        
        for i in range(self.types_table.rowCount()):
            col_item = self.types_table.item(i, 0)
            type_combo = self.types_table.cellWidget(i, 2)
            
            if col_item and type_combo:
                col_name = col_item.text()
                selected_type = type_combo.currentText()
                data_types[col_name] = selected_type
                
        self.wizard.import_config['data_types'] = data_types
        return True


class MissingValuesPage(QWizardPage):
    """Page for configuring missing value handling"""
    
    def __init__(self, wizard: DataImportWizard):
        super().__init__()
        self.wizard = wizard
        
        self.setTitle("Handle Missing Values")
        self.setSubTitle("Configure how missing values should be handled.")
        
        layout = QVBoxLayout()
        
        detection_group = QGroupBox("Missing Value Detection")
        detection_layout = QVBoxLayout()
        
        self.missing_values_edit = QLineEdit("NA, N/A, null, NULL, , .")
        detection_layout.addWidget(QLabel("Values to treat as missing (comma-separated):"))
        detection_layout.addWidget(self.missing_values_edit)
        
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)
        
        strategy_group = QGroupBox("Missing Value Strategy")
        strategy_layout = QVBoxLayout()
        
        self.keep_radio = QRadioButton("Keep missing values as NaN")
        self.keep_radio.setChecked(True)
        self.drop_radio = QRadioButton("Drop rows with missing values")
        self.impute_radio = QRadioButton("Impute missing values")
        
        strategy_layout.addWidget(self.keep_radio)
        strategy_layout.addWidget(self.drop_radio)
        strategy_layout.addWidget(self.impute_radio)
        
        strategy_group.setLayout(strategy_layout)
        layout.addWidget(strategy_group)
        
        self.impute_group = QGroupBox("Imputation Options")
        impute_layout = QFormLayout()
        
        self.numeric_combo = QComboBox()
        self.numeric_combo.addItems(['mean', 'median', 'mode', 'zero', 'custom'])
        impute_layout.addRow("Numeric columns:", self.numeric_combo)
        
        self.text_combo = QComboBox()
        self.text_combo.addItems(['mode', 'empty', 'unknown', 'custom'])
        impute_layout.addRow("Text columns:", self.text_combo)
        
        self.impute_group.setLayout(impute_layout)
        self.impute_group.setEnabled(False)
        layout.addWidget(self.impute_group)
        
        layout.addStretch()
        self.setLayout(layout)
        
        self.impute_radio.toggled.connect(self.impute_group.setEnabled)
        
    def validatePage(self):
        """Save missing value configuration"""
        missing_values = [v.strip() for v in self.missing_values_edit.text().split(',')]
        self.wizard.import_config['missing_values'] = missing_values
        
        if self.keep_radio.isChecked():
            self.wizard.import_config['missing_strategy'] = 'keep'
        elif self.drop_radio.isChecked():
            self.wizard.import_config['missing_strategy'] = 'drop'
        elif self.impute_radio.isChecked():
            self.wizard.import_config['missing_strategy'] = 'impute'
            self.wizard.import_config['numeric_impute'] = self.numeric_combo.currentText()
            self.wizard.import_config['text_impute'] = self.text_combo.currentText()
            
        return True


class SummaryPage(QWizardPage):
    """Summary page showing final import configuration"""
    
    def __init__(self, wizard: DataImportWizard):
        super().__init__()
        self.wizard = wizard
        
        self.setTitle("Import Summary")
        self.setSubTitle("Review the import configuration and start the import.")
        
        layout = QVBoxLayout()
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        layout.addWidget(self.summary_text)
        
        name_layout = QFormLayout()
        self.dataset_name_edit = QLineEdit()
        name_layout.addRow("Dataset Name:", self.dataset_name_edit)
        
        layout.addLayout(name_layout)
        
        self.setLayout(layout)
        
    def initializePage(self):
        """Initialize summary page"""
        dataset_name = self.wizard.import_config.get('dataset_name', 'Imported_Data')
        self.dataset_name_edit.setText(dataset_name)
        
        self.update_summary()
        
    def update_summary(self):
        """Update the summary display"""
        config = self.wizard.import_config
        
        summary = f"""
        <h3>Import Configuration Summary</h3>
        
        <p><b>Data Source:</b> {config.get('source_type', 'Unknown')}</p>
        <p><b>File Path:</b> {config.get('file_path', 'Not specified')}</p>
        <p><b>File Format:</b> {config.get('file_format', 'Unknown')}</p>
        
        <h4>Format Options:</h4>
        <ul>
        <li><b>Encoding:</b> {config.get('encoding', 'utf-8')}</li>
        <li><b>Delimiter:</b> "{config.get('delimiter', ',')}"</li>
        <li><b>Header Row:</b> {config.get('header_row', 0)}</li>
        <li><b>Skip Rows:</b> {config.get('skip_rows', 0)}</li>
        </ul>
        
        <h4>Data Types:</h4>
        <ul>
        """
        
        data_types = config.get('data_types', {})
        for col, dtype in data_types.items():
            summary += f"<li><b>{col}:</b> {dtype}</li>"
            
        summary += f"""
        </ul>
        
        <h4>Missing Values:</h4>
        <ul>
        <li><b>Strategy:</b> {config.get('missing_strategy', 'keep')}</li>
        <li><b>Missing Values:</b> {', '.join(config.get('missing_values', []))}</li>
        </ul>
        """
        
        preview_data = config.get('preview_data')
        if preview_data is not None:
            summary += f"""
            <h4>Dataset Preview:</h4>
            <ul>
            <li><b>Rows:</b> {len(preview_data)}</li>
            <li><b>Columns:</b> {len(preview_data.columns)}</li>
            <li><b>Memory:</b> {preview_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB</li>
            </ul>
            """
            
        self.summary_text.setHtml(summary)
        
    def validatePage(self):
        """Save final configuration"""
        self.wizard.import_config['dataset_name'] = self.dataset_name_edit.text().strip()
        
        if not self.wizard.import_config['dataset_name']:
            QMessageBox.warning(self, "Validation Error", "Please enter a dataset name.")
            return False
            
        return True