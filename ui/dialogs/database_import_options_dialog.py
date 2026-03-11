#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database Import Options Dialog for Bespoke Utility
Provides UI for configuring database connections and importing data
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

from data.database_importer import DatabaseImporter, DatabaseConnectionError, DatabaseQueryError

logger = logging.getLogger(__name__)


class DatabaseTestThread(QThread):
    """Thread for testing database connections"""
    
    connectionTested = pyqtSignal(bool, str)  # success, message
    tablesLoaded = pyqtSignal(list)  # table list
    schemasLoaded = pyqtSignal(list)  # schema list
    
    def __init__(self, db_type: str, connection_details: Dict[str, Any]):
        super().__init__()
        self.db_type = db_type
        self.connection_details = connection_details
        self.action = "test"
        
    def set_action(self, action: str):
        """Set the action to perform: 'test', 'load_tables', 'load_schemas'"""
        self.action = action
        
    def run(self):
        """Run the database operation"""
        importer = DatabaseImporter({})
        
        try:
            if importer.connect(self.db_type, self.connection_details):
                if self.action == "test":
                    self.connectionTested.emit(True, "Connection successful!")
                elif self.action == "load_tables":
                    tables = importer.list_tables_or_views()
                    self.tablesLoaded.emit(tables or [])
                elif self.action == "load_schemas":
                    schemas = importer.list_schemas()
                    self.schemasLoaded.emit(schemas or [])
                    
            importer.disconnect()
            
        except (DatabaseConnectionError, DatabaseQueryError) as e:
            self.connectionTested.emit(False, str(e))
        except Exception as e:
            self.connectionTested.emit(False, f"Unexpected error: {str(e)}")


class DatabaseImportOptionsDialog(QDialog):
    """Dialog for configuring database import options"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Database Import Options")
        self.setMinimumSize(700, 500)
        self.resize(800, 600)
        
        self.connection_settings = {}
        self.selected_table = None
        self.selected_schema = None
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
        
        self.tables_tab = self.create_tables_tab()
        self.tab_widget.addTab(self.tables_tab, "Tables")
        
        self.query_tab = self.create_query_tab()
        self.tab_widget.addTab(self.query_tab, "Custom Query")
        
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
        
    def create_connection_tab(self) -> QWidget:
        """Create the connection configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        type_group = QGroupBox("Database Type")
        type_layout = QFormLayout(type_group)
        
        self.db_type_combo = QComboBox()
        self.db_type_combo.addItems([
            "SQLite",
            "PostgreSQL", 
            "MySQL",
            "SQL Server",
            "Oracle"
        ])
        type_layout.addRow("Database Type:", self.db_type_combo)
        
        layout.addWidget(type_group)
        
        self.connection_group = QGroupBox("Connection Parameters")
        self.connection_layout = QFormLayout(self.connection_group)
        
        self.host_edit = QLineEdit()
        self.host_edit.setText("localhost")
        
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(5432)
        
        self.database_edit = QLineEdit()
        self.username_edit = QLineEdit()
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        
        self.sqlite_path_edit = QLineEdit()
        self.sqlite_browse_btn = QPushButton("Browse...")
        
        self.driver_combo = QComboBox()
        self.driver_combo.addItems([
            "{ODBC Driver 17 for SQL Server}",
            "{ODBC Driver 13 for SQL Server}",
            "{SQL Server Native Client 11.0}",
            "{SQL Server}"
        ])
        self.trusted_connection_check = QCheckBox("Use Windows Authentication")
        
        self.service_name_edit = QLineEdit()
        self.sid_edit = QLineEdit()
        
        self.connection_layout.addRow("Host:", self.host_edit)
        self.connection_layout.addRow("Port:", self.port_spin)
        self.connection_layout.addRow("Database:", self.database_edit)
        self.connection_layout.addRow("Username:", self.username_edit)
        self.connection_layout.addRow("Password:", self.password_edit)
        
        sqlite_layout = QHBoxLayout()
        sqlite_layout.addWidget(self.sqlite_path_edit)
        sqlite_layout.addWidget(self.sqlite_browse_btn)
        self.connection_layout.addRow("Database Path:", sqlite_layout)
        
        self.connection_layout.addRow("Driver:", self.driver_combo)
        self.connection_layout.addRow("", self.trusted_connection_check)
        
        self.connection_layout.addRow("Service Name:", self.service_name_edit)
        self.connection_layout.addRow("SID:", self.sid_edit)
        
        layout.addWidget(self.connection_group)
        
        self.update_connection_fields()
        
        layout.addStretch()
        
        return widget
    
    def create_tables_tab(self) -> QWidget:
        """Create the tables selection tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        schema_group = QGroupBox("Schema")
        schema_layout = QHBoxLayout(schema_group)
        
        self.schema_combo = QComboBox()
        self.schema_combo.setEditable(True)
        self.load_schemas_btn = QPushButton("Load Schemas")
        
        schema_layout.addWidget(QLabel("Schema:"))
        schema_layout.addWidget(self.schema_combo)
        schema_layout.addWidget(self.load_schemas_btn)
        schema_layout.addStretch()
        
        layout.addWidget(schema_group)
        
        tables_group = QGroupBox("Tables and Views")
        tables_layout = QVBoxLayout(tables_group)
        
        load_tables_layout = QHBoxLayout()
        self.load_tables_btn = QPushButton("Load Tables")
        load_tables_layout.addWidget(self.load_tables_btn)
        load_tables_layout.addStretch()
        tables_layout.addLayout(load_tables_layout)
        
        self.tables_list = QListWidget()
        tables_layout.addWidget(self.tables_list)
        
        self.table_info_label = QLabel("Select a table to view information")
        self.table_info_label.setWordWrap(True)
        tables_layout.addWidget(self.table_info_label)
        
        layout.addWidget(tables_group)
        
        return widget
    
    def create_query_tab(self) -> QWidget:
        """Create the custom query tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        query_group = QGroupBox("SQL Query")
        query_layout = QVBoxLayout(query_group)
        
        self.query_edit = QTextEdit()
        self.query_edit.setPlaceholderText("Enter your SQL query here...")
        self.query_edit.setMinimumHeight(150)
        
        font = QFont("Courier New", 10)
        self.query_edit.setFont(font)
        
        query_layout.addWidget(self.query_edit)
        
        options_layout = QHBoxLayout()
        
        self.limit_check = QCheckBox("Limit rows")
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(1, 1000000)
        self.limit_spin.setValue(1000)
        self.limit_spin.setEnabled(False)
        
        options_layout.addWidget(self.limit_check)
        options_layout.addWidget(self.limit_spin)
        options_layout.addStretch()
        
        query_layout.addLayout(options_layout)
        
        layout.addWidget(query_group)
        
        preview_group = QGroupBox("Query Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_btn = QPushButton("Preview Query")
        self.preview_table = QTableWidget()
        self.preview_table.setMaximumHeight(200)
        
        preview_layout.addWidget(self.preview_btn)
        preview_layout.addWidget(self.preview_table)
        
        layout.addWidget(preview_group)
        
        return widget
    
    def connect_signals(self):
        """Connect signals and slots"""
        self.db_type_combo.currentTextChanged.connect(self.update_connection_fields)
        self.test_button.clicked.connect(self.test_connection)
        self.sqlite_browse_btn.clicked.connect(self.browse_sqlite_file)
        self.trusted_connection_check.toggled.connect(self.toggle_sql_auth_fields)
        
        self.load_schemas_btn.clicked.connect(self.load_schemas)
        self.load_tables_btn.clicked.connect(self.load_tables)
        self.tables_list.itemSelectionChanged.connect(self.table_selection_changed)
        
        self.limit_check.toggled.connect(self.limit_spin.setEnabled)
        self.preview_btn.clicked.connect(self.preview_query)
        
        self.import_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
    
    def update_connection_fields(self):
        """Update connection fields based on selected database type"""
        db_type = self.db_type_combo.currentText().lower()
        
        for i in range(self.connection_layout.rowCount()):
            label_item = self.connection_layout.itemAt(i, QFormLayout.LabelRole)
            field_item = self.connection_layout.itemAt(i, QFormLayout.FieldRole)
            
            if label_item and label_item.widget():
                label_item.widget().setVisible(False)
            if field_item:
                if field_item.widget():
                    field_item.widget().setVisible(False)
                elif field_item.layout():
                    for j in range(field_item.layout().count()):
                        item = field_item.layout().itemAt(j)
                        if item and item.widget():
                            item.widget().setVisible(False)
        
        if db_type == "sqlite":
            self.show_field_row("Database Path:")
            
        elif db_type == "postgresql":
            self.port_spin.setValue(5432)
            self.show_field_row("Host:")
            self.show_field_row("Port:")
            self.show_field_row("Database:")
            self.show_field_row("Username:")
            self.show_field_row("Password:")
            
        elif db_type == "mysql":
            self.port_spin.setValue(3306)
            self.show_field_row("Host:")
            self.show_field_row("Port:")
            self.show_field_row("Database:")
            self.show_field_row("Username:")
            self.show_field_row("Password:")
            
        elif db_type == "sql server":
            self.port_spin.setValue(1433)
            self.show_field_row("Host:")
            self.show_field_row("Port:")
            self.show_field_row("Database:")
            self.show_field_row("Driver:")
            self.show_field_row("")  # Trusted connection checkbox
            if not self.trusted_connection_check.isChecked():
                self.show_field_row("Username:")
                self.show_field_row("Password:")
                
        elif db_type == "oracle":
            self.port_spin.setValue(1521)
            self.show_field_row("Host:")
            self.show_field_row("Port:")
            self.show_field_row("Username:")
            self.show_field_row("Password:")
            self.show_field_row("Service Name:")
            self.show_field_row("SID:")
    
    def show_field_row(self, label_text: str):
        """Show a specific field row by label text"""
        for i in range(self.connection_layout.rowCount()):
            label_item = self.connection_layout.itemAt(i, QFormLayout.LabelRole)
            field_item = self.connection_layout.itemAt(i, QFormLayout.FieldRole)
            
            if label_item and label_item.widget():
                label_widget = label_item.widget()
                if isinstance(label_widget, QLabel) and label_widget.text() == label_text:
                    label_widget.setVisible(True)
                    if field_item:
                        if field_item.widget():
                            field_item.widget().setVisible(True)
                        elif field_item.layout():
                            for j in range(field_item.layout().count()):
                                item = field_item.layout().itemAt(j)
                                if item and item.widget():
                                    item.widget().setVisible(True)
                    break
    
    def browse_sqlite_file(self):
        """Browse for SQLite database file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SQLite Database",
            "",
            "SQLite files (*.db *.sqlite *.sqlite3);;All files (*)"
        )
        
        if file_path:
            self.sqlite_path_edit.setText(file_path)
    
    def toggle_sql_auth_fields(self, checked: bool):
        """Toggle SQL Server authentication fields"""
        if self.db_type_combo.currentText().lower() == "sql server":
            if checked:
                self.hide_field_row("Username:")
                self.hide_field_row("Password:")
            else:
                self.show_field_row("Username:")
                self.show_field_row("Password:")
    
    def hide_field_row(self, label_text: str):
        """Hide a specific field row by label text"""
        for i in range(self.connection_layout.rowCount()):
            label_item = self.connection_layout.itemAt(i, QFormLayout.LabelRole)
            field_item = self.connection_layout.itemAt(i, QFormLayout.FieldRole)
            
            if label_item and label_item.widget():
                label_widget = label_item.widget()
                if isinstance(label_widget, QLabel) and label_widget.text() == label_text:
                    label_widget.setVisible(False)
                    if field_item and field_item.widget():
                        field_item.widget().setVisible(False)
                    break
    
    def get_connection_details(self) -> Dict[str, Any]:
        """Get connection details from UI fields"""
        db_type = self.db_type_combo.currentText().lower()
        
        if db_type == "sqlite":
            return {
                'database_path': self.sqlite_path_edit.text().strip()
            }
        
        elif db_type in ["postgresql", "mysql"]:
            return {
                'host': self.host_edit.text().strip(),
                'port': self.port_spin.value(),
                'database_name': self.database_edit.text().strip(),
                'user': self.username_edit.text().strip(),
                'password': self.password_edit.text()
            }
        
        elif db_type == "sql server":
            details = {
                'host': self.host_edit.text().strip(),
                'port': self.port_spin.value(),
                'database_name': self.database_edit.text().strip(),
                'driver': self.driver_combo.currentText(),
                'trusted_connection': self.trusted_connection_check.isChecked()
            }
            
            if not self.trusted_connection_check.isChecked():
                details.update({
                    'user': self.username_edit.text().strip(),
                    'password': self.password_edit.text()
                })
            
            return details
        
        elif db_type == "oracle":
            details = {
                'host': self.host_edit.text().strip(),
                'port': self.port_spin.value(),
                'user': self.username_edit.text().strip(),
                'password': self.password_edit.text()
            }
            
            if self.service_name_edit.text().strip():
                details['service_name'] = self.service_name_edit.text().strip()
            elif self.sid_edit.text().strip():
                details['sid'] = self.sid_edit.text().strip()
            
            return details
        
        return {}
    
    def test_connection(self):
        """Test the database connection"""
        if hasattr(self, 'test_thread') and self.test_thread and self.test_thread.isRunning():
            return
            
        db_type = self.db_type_combo.currentText().lower()
        connection_details = self.get_connection_details()
        
        if not self.validate_connection_details(connection_details):
            return
        
        if hasattr(self, 'test_thread') and self.test_thread:
            self.test_thread.quit()
            self.test_thread.wait()
        
        self.test_thread = DatabaseTestThread(db_type, connection_details)
        self.test_thread.connectionTested.connect(self.on_connection_tested)
        
        self.test_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        self.test_thread.start()
    
    def validate_connection_details(self, details: Dict[str, Any]) -> bool:
        """Validate connection details"""
        db_type = self.db_type_combo.currentText().lower()
        
        if db_type == "sqlite":
            if not details.get('database_path'):
                QMessageBox.warning(self, "Validation Error", "Please specify the SQLite database path.")
                return False
        
        else:
            required_fields = ['host', 'database_name']
            if db_type == "sql server" and details.get('trusted_connection'):
                required_fields = ['host', 'database_name']  # No user/password needed
            elif db_type == "oracle":
                required_fields = ['host', 'user']
                if not details.get('service_name') and not details.get('sid'):
                    QMessageBox.warning(self, "Validation Error", "Please specify either Service Name or SID for Oracle.")
                    return False
            else:
                required_fields = ['host', 'database_name', 'user']
            
            for field in required_fields:
                if not details.get(field):
                    QMessageBox.warning(self, "Validation Error", f"Please fill in the {field.replace('_', ' ').title()} field.")
                    return False
        
        return True
    
    def on_connection_tested(self, success: bool, message: str):
        """Handle connection test result"""
        self.test_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            QMessageBox.information(self, "Connection Test", message)
            self.import_button.setEnabled(True)
            
            self.tab_widget.setTabEnabled(1, True)
            self.tab_widget.setTabEnabled(2, True)
        else:
            QMessageBox.warning(self, "Connection Test Failed", message)
            self.import_button.setEnabled(False)
    
    def load_schemas(self):
        """Load available schemas"""
        db_type = self.db_type_combo.currentText().lower()
        connection_details = self.get_connection_details()
        
        if not self.validate_connection_details(connection_details):
            return
        
        self.test_thread = DatabaseTestThread(db_type, connection_details)
        self.test_thread.set_action("load_schemas")
        self.test_thread.schemasLoaded.connect(self.on_schemas_loaded)
        
        self.load_schemas_btn.setEnabled(False)
        self.test_thread.start()
    
    def on_schemas_loaded(self, schemas: List[str]):
        """Handle loaded schemas"""
        self.load_schemas_btn.setEnabled(True)
        
        self.schema_combo.clear()
        if schemas:
            self.schema_combo.addItems(schemas)
        else:
            self.schema_combo.addItem("(No schemas found)")
    
    def load_tables(self):
        """Load available tables"""
        db_type = self.db_type_combo.currentText().lower()
        connection_details = self.get_connection_details()
        
        if not self.validate_connection_details(connection_details):
            return
        
        self.test_thread = DatabaseTestThread(db_type, connection_details)
        self.test_thread.set_action("load_tables")
        self.test_thread.tablesLoaded.connect(self.on_tables_loaded)
        
        self.load_tables_btn.setEnabled(False)
        self.test_thread.start()
    
    def on_tables_loaded(self, tables: List[str]):
        """Handle loaded tables"""
        self.load_tables_btn.setEnabled(True)
        
        self.tables_list.clear()
        if tables:
            self.tables_list.addItems(tables)
        else:
            self.tables_list.addItem("(No tables found)")
    
    def table_selection_changed(self):
        """Handle table selection change"""
        current_item = self.tables_list.currentItem()
        if current_item:
            self.selected_table = current_item.text()
            self.table_info_label.setText(f"Selected table: {self.selected_table}")
            # TODO: Load table info (columns, row count, etc.)
    
    def preview_query(self):
        """Preview custom query results"""
        query = self.query_edit.toPlainText().strip()
        if not query:
            QMessageBox.warning(self, "Query Error", "Please enter a SQL query.")
            return
        
        # TODO: Implement query preview
        QMessageBox.information(self, "Preview", "Query preview not implemented yet.")
    
    def get_import_config(self) -> Dict[str, Any]:
        """Get the final import configuration"""
        config = {
            'db_type': self.db_type_combo.currentText().lower(),
            'connection_details': self.get_connection_details(),
            'import_mode': None,
            'table_name': None,
            'schema': None,
            'query': None
        }
        
        current_tab = self.tab_widget.currentIndex()
        
        if current_tab == 1:  # Tables tab
            config['import_mode'] = 'table'
            config['table_name'] = self.selected_table
            
            schema_text = self.schema_combo.currentText()
            if schema_text and schema_text != "(No schemas found)":
                config['schema'] = schema_text
                
        elif current_tab == 2:  # Query tab
            config['import_mode'] = 'query'
            config['query'] = self.query_edit.toPlainText().strip()
            
            if self.limit_check.isChecked():
                config['limit'] = self.limit_spin.value()
        
        return config
    
    def accept(self):
        """Accept the dialog and return configuration"""
        config = self.get_import_config()
        
        if config['import_mode'] == 'table' and not config['table_name']:
            QMessageBox.warning(self, "Import Error", "Please select a table to import.")
            return
        
        if config['import_mode'] == 'query' and not config['query']:
            QMessageBox.warning(self, "Import Error", "Please enter a SQL query.")
            return
        
        if config['import_mode'] is None:
            QMessageBox.warning(self, "Import Error", "Please select Tables or Custom Query tab.")
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
    dialog = DatabaseImportOptionsDialog()
    
    if dialog.exec_() == QDialog.Accepted:
        print("Import configuration:", dialog.connection_settings)
    
    sys.exit(app.exec_())