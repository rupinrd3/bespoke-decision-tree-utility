#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cloud Import Options Dialog for Bespoke Utility
Provides UI for configuring cloud storage connections and importing data
"""

import logging
from typing import Dict, Any, Optional, List

from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QLabel, QLineEdit, QComboBox, QPushButton, QSpinBox,
    QCheckBox, QTextEdit, QTabWidget, QWidget, QListWidget,
    QSplitter, QProgressBar, QMessageBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QRadioButton,
    QDialogButtonBox, QFileDialog
)

from data.cloud_importer import CloudImporter, CloudStorageError

logger = logging.getLogger(__name__)


class CloudTestThread(QThread):
    """Thread for testing cloud connections"""
    
    connectionTested = pyqtSignal(bool, str)  # success, message
    bucketsLoaded = pyqtSignal(list)  # bucket list
    filesLoaded = pyqtSignal(list)  # file list
    
    def __init__(self, provider: str, connection_details: Dict[str, Any]):
        super().__init__()
        self.provider = provider
        self.connection_details = connection_details
        self.action = "test"
        self.bucket_name = None
        self.prefix = ""
        
    def set_action(self, action: str, bucket_name: str = None, prefix: str = ""):
        """Set the action to perform: 'test', 'load_buckets', 'load_files'"""
        self.action = action
        self.bucket_name = bucket_name
        self.prefix = prefix
        
    def run(self):
        """Run the cloud operation"""
        importer = CloudImporter({})
        
        try:
            if self.action == "test":
                if importer.test_connection(self.provider, **self.connection_details):
                    self.connectionTested.emit(True, "Connection successful!")
                else:
                    self.connectionTested.emit(False, "Connection test failed")
                    
            elif self.action == "load_buckets":
                buckets = importer.list_buckets(self.provider, **self.connection_details)
                self.bucketsLoaded.emit(buckets or [])
                
            elif self.action == "load_files":
                files = importer.list_files(self.provider, self.bucket_name, 
                                          prefix=self.prefix, max_files=100, 
                                          **self.connection_details)
                self.filesLoaded.emit(files or [])
                    
        except CloudStorageError as e:
            self.connectionTested.emit(False, str(e))
        except Exception as e:
            self.connectionTested.emit(False, f"Unexpected error: {str(e)}")


class CloudImportOptionsDialog(QDialog):
    """Dialog for configuring cloud import options"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Cloud Import Options")
        self.setMinimumSize(700, 500)
        self.resize(800, 600)
        
        self.connection_settings = {}
        self.selected_file = None
        self.test_thread = None
        
        self.init_ui()
        
        self.connect_signals()
        
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        self.connection_tab = self.create_connection_tab()
        self.tab_widget.addTab(self.connection_tab, "Connection")
        
        self.files_tab = self.create_files_tab()
        self.tab_widget.addTab(self.files_tab, "Files")
        
        self.options_tab = self.create_options_tab()
        self.tab_widget.addTab(self.options_tab, "Import Options")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        button_layout = QHBoxLayout()
        
        self.test_button = QPushButton("Test Connection")
        self.import_button = QPushButton("Import")
        self.cancel_button = QPushButton("Cancel")
        
        self.import_button.setEnabled(False)
        
        button_layout.addStretch()
        button_layout.addWidget(self.test_button)
        button_layout.addWidget(self.import_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.tab_widget.setTabEnabled(1, False)
        self.tab_widget.setTabEnabled(2, False)
        
    def create_connection_tab(self) -> QWidget:
        """Create the connection configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        provider_group = QGroupBox("Cloud Provider")
        provider_layout = QFormLayout(provider_group)
        
        self.provider_combo = QComboBox()
        self.provider_combo.addItems([
            "AWS S3",
            "Google Cloud Storage",
            "Azure Blob Storage"
        ])
        provider_layout.addRow("Provider:", self.provider_combo)
        
        layout.addWidget(provider_group)
        
        self.aws_group = QGroupBox("AWS S3 Configuration")
        aws_layout = QFormLayout(self.aws_group)
        
        self.aws_access_key_edit = QLineEdit()
        self.aws_secret_key_edit = QLineEdit()
        self.aws_secret_key_edit.setEchoMode(QLineEdit.Password)
        self.aws_region_edit = QLineEdit()
        self.aws_region_edit.setText("us-east-1")
        
        aws_layout.addRow("Access Key ID:", self.aws_access_key_edit)
        aws_layout.addRow("Secret Access Key:", self.aws_secret_key_edit)
        aws_layout.addRow("Region:", self.aws_region_edit)
        
        aws_info = QLabel("Leave credentials empty to use environment variables or IAM roles")
        aws_info.setWordWrap(True)
        aws_info.setStyleSheet("color: gray; font-style: italic;")
        aws_layout.addRow("", aws_info)
        
        layout.addWidget(self.aws_group)
        
        self.gcs_group = QGroupBox("Google Cloud Storage Configuration")
        gcs_layout = QFormLayout(self.gcs_group)
        
        self.gcs_project_edit = QLineEdit()
        self.gcs_credentials_edit = QLineEdit()
        self.gcs_browse_btn = QPushButton("Browse...")
        
        creds_layout = QHBoxLayout()
        creds_layout.addWidget(self.gcs_credentials_edit)
        creds_layout.addWidget(self.gcs_browse_btn)
        
        gcs_layout.addRow("Project ID:", self.gcs_project_edit)
        gcs_layout.addRow("Credentials JSON:", creds_layout)
        
        gcs_info = QLabel("Leave credentials empty to use default application credentials")
        gcs_info.setWordWrap(True)
        gcs_info.setStyleSheet("color: gray; font-style: italic;")
        gcs_layout.addRow("", gcs_info)
        
        layout.addWidget(self.gcs_group)
        
        self.azure_group = QGroupBox("Azure Blob Storage Configuration")
        azure_layout = QFormLayout(self.azure_group)
        
        self.azure_account_edit = QLineEdit()
        self.azure_key_edit = QLineEdit()
        self.azure_key_edit.setEchoMode(QLineEdit.Password)
        self.azure_connection_edit = QLineEdit()
        
        azure_layout.addRow("Account Name:", self.azure_account_edit)
        azure_layout.addRow("Account Key:", self.azure_key_edit)
        azure_layout.addRow("Connection String:", self.azure_connection_edit)
        
        azure_info = QLabel("Provide either Account Name + Key OR Connection String")
        azure_info.setWordWrap(True)
        azure_info.setStyleSheet("color: gray; font-style: italic;")
        azure_layout.addRow("", azure_info)
        
        layout.addWidget(self.azure_group)
        
        self.update_provider_fields()
        
        layout.addStretch()
        
        return widget
    
    def create_files_tab(self) -> QWidget:
        """Create the files selection tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        bucket_group = QGroupBox("Bucket/Container")
        bucket_layout = QHBoxLayout(bucket_group)
        
        self.bucket_combo = QComboBox()
        self.bucket_combo.setEditable(True)
        self.load_buckets_btn = QPushButton("Load Buckets")
        
        bucket_layout.addWidget(QLabel("Bucket:"))
        bucket_layout.addWidget(self.bucket_combo)
        bucket_layout.addWidget(self.load_buckets_btn)
        bucket_layout.addStretch()
        
        layout.addWidget(bucket_group)
        
        files_group = QGroupBox("Files")
        files_layout = QVBoxLayout(files_group)
        
        prefix_layout = QHBoxLayout()
        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText("Enter path prefix to filter files...")
        self.load_files_btn = QPushButton("Load Files")
        
        prefix_layout.addWidget(QLabel("Prefix:"))
        prefix_layout.addWidget(self.prefix_edit)
        prefix_layout.addWidget(self.load_files_btn)
        files_layout.addLayout(prefix_layout)
        
        self.files_table = QTableWidget()
        self.files_table.setColumnCount(4)
        self.files_table.setHorizontalHeaderLabels(["Name", "Size", "Modified", "Type"])
        self.files_table.horizontalHeader().setStretchLastSection(True)
        self.files_table.setSelectionBehavior(QTableWidget.SelectRows)
        files_layout.addWidget(self.files_table)
        
        self.file_info_label = QLabel("Select a file to view details")
        self.file_info_label.setWordWrap(True)
        files_layout.addWidget(self.file_info_label)
        
        layout.addWidget(files_group)
        
        return widget
    
    def create_options_tab(self) -> QWidget:
        """Create the import options tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        format_group = QGroupBox("File Format")
        format_layout = QFormLayout(format_group)
        
        self.format_combo = QComboBox()
        self.format_combo.addItems(["csv", "excel", "parquet", "json", "txt"])
        format_layout.addRow("Format:", self.format_combo)
        
        layout.addWidget(format_group)
        
        pandas_group = QGroupBox("Import Options")
        pandas_layout = QVBoxLayout(pandas_group)
        
        pandas_layout.addWidget(QLabel("Additional pandas read options (JSON format):"))
        
        self.pandas_options_edit = QTextEdit()
        self.pandas_options_edit.setPlaceholderText('{"sep": ",", "encoding": "utf-8", "nrows": 1000}')
        self.pandas_options_edit.setMaximumHeight(100)
        
        font = QFont("Courier New", 9)
        self.pandas_options_edit.setFont(font)
        
        pandas_layout.addWidget(self.pandas_options_edit)
        
        examples_label = QLabel("""
Examples:
• CSV: {"sep": ",", "encoding": "utf-8", "nrows": 1000}
• Excel: {"sheet_name": "Sheet1", "header": 0}
• JSON: {"orient": "records", "lines": true}
        """.strip())
        examples_label.setStyleSheet("color: gray; font-size: 9px;")
        examples_label.setWordWrap(True)
        pandas_layout.addWidget(examples_label)
        
        layout.addWidget(pandas_group)
        
        layout.addStretch()
        
        return widget
    
    def connect_signals(self):
        """Connect signals and slots"""
        self.provider_combo.currentTextChanged.connect(self.update_provider_fields)
        self.test_button.clicked.connect(self.test_connection)
        self.gcs_browse_btn.clicked.connect(self.browse_gcs_credentials)
        
        self.load_buckets_btn.clicked.connect(self.load_buckets)
        self.load_files_btn.clicked.connect(self.load_files)
        self.files_table.itemSelectionChanged.connect(self.file_selection_changed)
        
        self.format_combo.currentTextChanged.connect(self.format_changed)
        
        self.import_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
    
    def update_provider_fields(self):
        """Update connection fields based on selected provider"""
        provider = self.provider_combo.currentText()
        
        self.aws_group.setVisible(False)
        self.gcs_group.setVisible(False)
        self.azure_group.setVisible(False)
        
        if provider == "AWS S3":
            self.aws_group.setVisible(True)
        elif provider == "Google Cloud Storage":
            self.gcs_group.setVisible(True)
        elif provider == "Azure Blob Storage":
            self.azure_group.setVisible(True)
    
    def browse_gcs_credentials(self):
        """Browse for GCS credentials JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select GCS Credentials JSON File",
            "",
            "JSON files (*.json);;All files (*)"
        )
        
        if file_path:
            self.gcs_credentials_edit.setText(file_path)
    
    def get_connection_details(self) -> Dict[str, Any]:
        """Get connection details from UI fields"""
        provider = self.provider_combo.currentText()
        
        if provider == "AWS S3":
            details = {}
            if self.aws_access_key_edit.text().strip():
                details['aws_access_key_id'] = self.aws_access_key_edit.text().strip()
            if self.aws_secret_key_edit.text().strip():
                details['aws_secret_access_key'] = self.aws_secret_key_edit.text().strip()
            if self.aws_region_edit.text().strip():
                details['region_name'] = self.aws_region_edit.text().strip()
            return details
        
        elif provider == "Google Cloud Storage":
            details = {}
            if self.gcs_project_edit.text().strip():
                details['project'] = self.gcs_project_edit.text().strip()
            if self.gcs_credentials_edit.text().strip():
                details['credentials_path'] = self.gcs_credentials_edit.text().strip()
            return details
        
        elif provider == "Azure Blob Storage":
            details = {}
            if self.azure_connection_edit.text().strip():
                details['connection_string'] = self.azure_connection_edit.text().strip()
            else:
                if self.azure_account_edit.text().strip():
                    details['account_name'] = self.azure_account_edit.text().strip()
                if self.azure_key_edit.text().strip():
                    details['account_key'] = self.azure_key_edit.text().strip()
            return details
        
        return {}
    
    def get_provider_key(self) -> str:
        """Get the provider key for the cloud importer"""
        provider = self.provider_combo.currentText()
        if provider == "AWS S3":
            return "aws_s3"
        elif provider == "Google Cloud Storage":
            return "gcs"
        elif provider == "Azure Blob Storage":
            return "azure_blob"
        return ""
    
    def test_connection(self):
        """Test the cloud connection"""
        if hasattr(self, 'test_thread') and self.test_thread and self.test_thread.isRunning():
            return
            
        provider_key = self.get_provider_key()
        connection_details = self.get_connection_details()
        
        if not self.validate_connection_details(connection_details):
            return
        
        if hasattr(self, 'test_thread') and self.test_thread:
            self.test_thread.quit()
            self.test_thread.wait()
        
        self.test_thread = CloudTestThread(provider_key, connection_details)
        self.test_thread.connectionTested.connect(self.on_connection_tested)
        
        self.test_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        self.test_thread.start()
    
    def validate_connection_details(self, details: Dict[str, Any]) -> bool:
        """Validate connection details"""
        provider = self.provider_combo.currentText()
        
        if provider == "Azure Blob Storage":
            if not details.get('connection_string'):
                if not details.get('account_name'):
                    QMessageBox.warning(self, "Validation Error", "Please provide either Connection String or Account Name for Azure.")
                    return False
        
        return True
    
    def on_connection_tested(self, success: bool, message: str):
        """Handle connection test result"""
        self.test_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            QMessageBox.information(self, "Connection Test", message)
            
            self.tab_widget.setTabEnabled(1, True)
            self.tab_widget.setTabEnabled(2, True)
        else:
            QMessageBox.warning(self, "Connection Test Failed", message)
    
    def load_buckets(self):
        """Load available buckets"""
        provider_key = self.get_provider_key()
        connection_details = self.get_connection_details()
        
        if not self.validate_connection_details(connection_details):
            return
        
        self.test_thread = CloudTestThread(provider_key, connection_details)
        self.test_thread.set_action("load_buckets")
        self.test_thread.bucketsLoaded.connect(self.on_buckets_loaded)
        
        self.load_buckets_btn.setEnabled(False)
        self.test_thread.start()
    
    def on_buckets_loaded(self, buckets: List[str]):
        """Handle loaded buckets"""
        self.load_buckets_btn.setEnabled(True)
        
        self.bucket_combo.clear()
        if buckets:
            self.bucket_combo.addItems(buckets)
        else:
            self.bucket_combo.addItem("(No buckets found)")
    
    def load_files(self):
        """Load files from selected bucket"""
        bucket_name = self.bucket_combo.currentText()
        if not bucket_name or bucket_name == "(No buckets found)":
            QMessageBox.warning(self, "No Bucket", "Please select a bucket first.")
            return
        
        provider_key = self.get_provider_key()
        connection_details = self.get_connection_details()
        prefix = self.prefix_edit.text().strip()
        
        self.test_thread = CloudTestThread(provider_key, connection_details)
        self.test_thread.set_action("load_files", bucket_name, prefix)
        self.test_thread.filesLoaded.connect(self.on_files_loaded)
        
        self.load_files_btn.setEnabled(False)
        self.test_thread.start()
    
    def on_files_loaded(self, files: List[Dict[str, Any]]):
        """Handle loaded files"""
        self.load_files_btn.setEnabled(True)
        
        self.files_table.setRowCount(len(files))
        
        for row, file_info in enumerate(files):
            self.files_table.setItem(row, 0, QTableWidgetItem(file_info.get('name', '')))
            
            size = file_info.get('size', 0)
            if size > 1024*1024:
                size_str = f"{size/(1024*1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size} B"
            self.files_table.setItem(row, 1, QTableWidgetItem(size_str))
            
            modified = file_info.get('modified', '')
            if hasattr(modified, 'strftime'):
                modified_str = modified.strftime('%Y-%m-%d %H:%M:%S')
            else:
                modified_str = str(modified)
            self.files_table.setItem(row, 2, QTableWidgetItem(modified_str))
            
            name = file_info.get('name', '')
            if '.' in name:
                file_type = name.split('.')[-1].upper()
            else:
                file_type = 'Unknown'
            self.files_table.setItem(row, 3, QTableWidgetItem(file_type))
        
        self.files_table.resizeColumnsToContents()
    
    def file_selection_changed(self):
        """Handle file selection change"""
        selected_rows = self.files_table.selectionModel().selectedRows()
        if selected_rows:
            row = selected_rows[0].row()
            file_name = self.files_table.item(row, 0).text()
            file_size = self.files_table.item(row, 1).text()
            file_modified = self.files_table.item(row, 2).text()
            file_type = self.files_table.item(row, 3).text()
            
            self.selected_file = file_name
            self.file_info_label.setText(
                f"Selected: {file_name}\n"
                f"Size: {file_size}, Modified: {file_modified}, Type: {file_type}"
            )
            
            if '.' in file_name:
                ext = file_name.split('.')[-1].lower()
                if ext in ['csv']:
                    self.format_combo.setCurrentText('csv')
                elif ext in ['xlsx', 'xls']:
                    self.format_combo.setCurrentText('excel')
                elif ext in ['parquet']:
                    self.format_combo.setCurrentText('parquet')
                elif ext in ['json']:
                    self.format_combo.setCurrentText('json')
                elif ext in ['txt']:
                    self.format_combo.setCurrentText('txt')
            
            self.import_button.setEnabled(True)
        else:
            self.selected_file = None
            self.file_info_label.setText("Select a file to view details")
            self.import_button.setEnabled(False)
    
    def format_changed(self):
        """Handle format change"""
        format_type = self.format_combo.currentText()
        
        if format_type == 'csv':
            self.pandas_options_edit.setPlaceholderText('{"sep": ",", "encoding": "utf-8", "nrows": 1000}')
        elif format_type == 'excel':
            self.pandas_options_edit.setPlaceholderText('{"sheet_name": "Sheet1", "header": 0}')
        elif format_type == 'json':
            self.pandas_options_edit.setPlaceholderText('{"orient": "records", "lines": true}')
        elif format_type == 'parquet':
            self.pandas_options_edit.setPlaceholderText('{"columns": ["col1", "col2"]}')
        else:
            self.pandas_options_edit.setPlaceholderText('{}')
    
    def get_import_config(self) -> Dict[str, Any]:
        """Get the final import configuration"""
        if not self.selected_file:
            return {}
        
        pandas_options = {}
        try:
            import json
            options_text = self.pandas_options_edit.toPlainText().strip()
            if options_text:
                pandas_options = json.loads(options_text)
        except json.JSONDecodeError:
            QMessageBox.warning(self, "Invalid JSON", "Pandas options must be valid JSON format.")
            return {}
        
        config = {
            'provider': self.get_provider_key(),
            'connection_details': self.get_connection_details(),
            'bucket_name': self.bucket_combo.currentText(),
            'file_key': self.selected_file,
            'file_format': self.format_combo.currentText(),
            'pandas_read_options': pandas_options
        }
        
        return config
    
    def accept(self):
        """Accept the dialog and return configuration"""
        config = self.get_import_config()
        
        if not config:
            return
        
        if not config.get('bucket_name') or config['bucket_name'] == "(No buckets found)":
            QMessageBox.warning(self, "Import Error", "Please select a valid bucket.")
            return
        
        if not config.get('file_key'):
            QMessageBox.warning(self, "Import Error", "Please select a file to import.")
            return
        
        self.connection_settings = config
        super().accept()
    
    def closeEvent(self, event):
        """Handle dialog close event"""
        if hasattr(self, 'test_thread') and self.test_thread:
            if self.test_thread.isRunning():
                self.test_thread.quit()
                self.test_thread.wait()
        super().closeEvent(event)


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    dialog = CloudImportOptionsDialog()
    
    if dialog.exec_() == QDialog.Accepted:
        print("Import configuration:", dialog.connection_settings)
    
    sys.exit(app.exec_())