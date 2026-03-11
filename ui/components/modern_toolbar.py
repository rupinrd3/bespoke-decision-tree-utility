#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modern Toolbar Component
Context-aware toolbar with modern styling and icon support
"""

import logging
from typing import Dict, List, Optional, Callable
from PyQt5.QtWidgets import (
    QToolBar, QAction, QPushButton, QMenu, QLabel, QWidget, 
    QHBoxLayout, QFrame, QSizePolicy, QToolButton
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QFont

logger = logging.getLogger(__name__)

class ModernToolbar(QToolBar):
    """Enhanced toolbar with context-aware sections and modern styling"""
    
    contextChanged = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMovable(False)
        self.setIconSize(QSize(18, 18))
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        
        self.context_actions = {}
        self.universal_actions = []
        self.current_context = 'workflow'
        
        self.action_buttons = {}
        self.separators = []
        
        self.setup_styling()
        logger.debug("ModernToolbar initialized")
        
    def setup_styling(self):
        """Apply modern styling to the toolbar"""
        self.setStyleSheet("""
            QToolBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #f8fafc);
                border: none;
                border-bottom: 1px solid #e2e8f0;
                spacing: 8px;
                padding: 4px 12px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            QToolButton {
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 6px;
                padding: 6px 12px;
                margin: 2px;
                font-size: 11px;
                font-weight: 600;
                color: #374151;
                min-width: 60px;
            }
            
            QToolButton:hover {
                background-color: #f1f5f9;
                border-color: #e2e8f0;
            }
            
            QToolButton:pressed {
                background-color: #e2e8f0;
                border-color: #cbd5e1;
            }
            
            QToolButton:checked {
                background-color: #dbeafe;
                border-color: #3b82f6;
                color: #1e40af;
            }
            
            QToolButton.primary {
                background-color: #3b82f6;
                color: white;
                border-color: #2563eb;
            }
            
            QToolButton.primary:hover {
                background-color: #2563eb;
                border-color: #1d4ed8;
            }
            
            QToolButton.primary:pressed {
                background-color: #1d4ed8;
            }
            
            QToolButton.secondary {
                background-color: #f1f5f9;
                color: #374151;
                border-color: #d1d5db;
            }
            
            QToolButton.secondary:hover {
                background-color: #e5e7eb;
            }
            
            QToolButton.danger {
                color: #dc2626;
            }
            
            QToolButton.danger:hover {
                background-color: #fef2f2;
                border-color: #fecaca;
            }
            
            QToolButton.success {
                color: #059669;
            }
            
            QToolButton.success:hover {
                background-color: #f0fdf4;
                border-color: #bbf7d0;
            }
            
            QToolButton[popupMode="1"] {
                padding-right: 20px;
            }
            
            QToolButton::menu-indicator {
                image: none;
                width: 0px;
            }
            
            QToolButton::menu-button {
                border: none;
                width: 16px;
            }
            
            QLabel.toolbarLabel {
                color: #6b7280;
                font-size: 10px;
                font-weight: 500;
                margin: 0px 4px;
            }
            
            QFrame.toolbarSeparator {
                background-color: #e2e8f0;
                max-width: 1px;
                margin: 8px 4px;
            }
        """)
        
    def add_universal_section(self):
        """Add universal actions section (always visible)"""
        self.clear()
        
        self.add_icon_action("file-plus", "New", None, "New Project (Ctrl+N)")
        self.add_icon_action("folder-open", "Open", None, "Open Project (Ctrl+O)")
        self.add_icon_action("save", "Save", None, "Save Project (Ctrl+S)")
        
        self.add_separator()
        
        import_button = self.add_dropdown_action("download", "Import", self.create_import_menu(), "Import Data")
        
        self.add_separator()
        
        self.add_icon_action("play", "Execute", None, "Execute Workflow (F5)", style="primary")
        
        self.add_separator()
        
        self.add_icon_action("settings", "Settings", None, "Application Settings")
        
    def add_context_section(self, context: str):
        """Add context-specific actions"""
        if context == 'workflow':
            self.add_workflow_actions()
        elif context == 'data':
            self.add_data_actions()
        elif context == 'model':
            self.add_model_actions()
        elif context == 'transform':
            self.add_transform_actions()
            
    def add_workflow_actions(self):
        """Add workflow context actions"""
        self.add_icon_action("plus-circle", "Add Node", None, "Add Node to Workflow")
        self.add_icon_action("link", "Connect", None, "Connect Nodes", checkable=True)
        self.add_icon_action("mouse-pointer", "Select", None, "Select Mode", checkable=True)
        self.add_icon_action("trash-2", "Delete", None, "Delete Selected (Del)", style="danger")
        
        self.add_separator()
        
        self.add_icon_action("grid", "Auto Layout", None, "Auto Layout Workflow")
        
        self.add_separator()
        
        self.add_icon_action("zoom-in", "Zoom In", None, "Zoom In (Ctrl++)")
        self.add_icon_action("zoom-out", "Zoom Out", None, "Zoom Out (Ctrl+-)")
        self.add_icon_action("maximize", "Fit View", None, "Fit to View (Ctrl+0)")
        
    def add_data_actions(self):
        """Add data context actions"""
        self.add_icon_action("arrow-left", "Back", None, "Back to Workflow")
        
        self.add_separator()
        
        self.add_icon_action("search", "Filter", None, "Filter Data")
        self.add_icon_action("zap", "Transform", None, "Transform Data")
        self.add_icon_action("bar-chart", "Statistics", None, "Data Statistics")
        self.add_icon_action("download", "Export", None, "Export Data")
        
        self.add_separator()
        
        self.add_icon_action("tool", "Clean", None, "Clean Data")
        self.add_icon_action("type", "Types", None, "Data Types")
        
    def add_model_actions(self):
        """Add model context actions"""
        self.add_icon_action("arrow-left", "Back", None, "Back to Workflow")
        
        self.add_separator()
        
        self.add_icon_action("eye", "View Tree", None, "View Decision Tree", style="primary")
        self.add_icon_action("scissors", "Edit Splits", None, "Edit Tree Splits")
        self.add_icon_action("bar-chart-2", "Metrics", None, "Model Metrics")
        self.add_icon_action("target", "Tune", None, "Tune Parameters")
        
        self.add_separator()
        
        self.add_icon_action("scissors", "Prune", None, "Prune Tree")
        self.add_icon_action("upload", "Export", None, "Export Model", style="success")
        
    def add_transform_actions(self):
        """Add transform context actions"""
        self.add_icon_action("arrow-left", "Back", None, "Back to Workflow")
        
        self.add_separator()
        
        self.add_icon_action("filter", "Apply Filter", None, "Apply Transform")
        self.add_icon_action("eye", "Preview", None, "Preview Results")
        self.add_icon_action("check", "Confirm", None, "Confirm Transform", style="success")
        self.add_icon_action("x", "Cancel", None, "Cancel Transform", style="danger")
        
    def add_icon_action(self, icon_name: str, text: str, callback: Optional[Callable], 
                       tooltip: str = "", style: str = "default", checkable: bool = False) -> QAction:
        """Add action with icon and text"""
        action = QAction(text, self)
        
        if tooltip:
            action.setStatusTip(tooltip)
            action.setToolTip(tooltip)
            
        if callback:
            action.triggered.connect(callback)
            
        if checkable:
            action.setCheckable(True)
            
        tool_button = self.addAction(action)
        
        if style != "default":
            widget = self.widgetForAction(action)
            if widget:
                widget.setProperty("class", style)
                widget.style().unpolish(widget)
                widget.style().polish(widget)
                
        self.action_buttons[text.lower().replace(" ", "_")] = action
        
        return action
        
    def add_dropdown_action(self, icon_name: str, text: str, menu: QMenu, tooltip: str = "") -> QToolButton:
        """Add dropdown action with menu"""
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tooltip)
        button.setPopupMode(QToolButton.InstantPopup)
        button.setMenu(menu)
        
        self.addWidget(button)
        
        return button
        
    def add_separator(self):
        """Add visual separator"""
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setProperty("class", "toolbarSeparator")
        self.addWidget(separator)
        self.separators.append(separator)
        
    def add_label(self, text: str) -> QLabel:
        """Add text label"""
        label = QLabel(text)
        label.setProperty("class", "toolbarLabel")
        self.addWidget(label)
        return label
        
    def create_import_menu(self) -> QMenu:
        """Create import data menu"""
        menu = QMenu()
        
        main_window = self.parent()
        if main_window and not hasattr(main_window, 'import_data'):
            widget = main_window
            while widget and not hasattr(widget, 'import_data'):
                widget = widget.parent()
            main_window = widget
        
        csv_action = QAction("📄 CSV File", self)
        csv_action.setStatusTip("Import CSV file")
        if hasattr(main_window, 'import_data'):
            csv_action.triggered.connect(lambda: main_window.import_data('csv'))
        menu.addAction(csv_action)
        
        excel_action = QAction("📊 Excel File", self)
        excel_action.setStatusTip("Import Excel file")
        if hasattr(main_window, 'import_data'):
            excel_action.triggered.connect(lambda: main_window.import_data('excel'))
        menu.addAction(excel_action)
        
        text_action = QAction("📝 Text File", self)
        text_action.setStatusTip("Import delimited text file")
        if hasattr(main_window, 'import_data'):
            text_action.triggered.connect(lambda: main_window.import_data('text'))
        menu.addAction(text_action)
        
        menu.addSeparator()
        
        db_action = QAction("🗄️ Database", self)
        db_action.setStatusTip("Import from database")
        if hasattr(main_window, 'import_data'):
            db_action.triggered.connect(lambda: main_window.import_data('database'))
        menu.addAction(db_action)
        
        cloud_action = QAction("☁️ Cloud Storage", self)
        cloud_action.setStatusTip("Import from cloud storage")
        if hasattr(main_window, 'import_data'):
            cloud_action.triggered.connect(lambda: main_window.import_data('cloud'))
        menu.addAction(cloud_action)
        
        return menu
        
    def set_context_mode(self, context: str):
        """Switch toolbar to specific context mode"""
        if context == self.current_context:
            return
            
        logger.debug(f"Switching toolbar context from '{self.current_context}' to '{context}'")
        
        self.clear_context_actions()
        
        self.add_context_section(context)
        
        self.current_context = context
        self.contextChanged.emit(context)
        
    def clear_context_actions(self):
        """Clear context-specific actions while preserving universal ones"""
        while len(self.separators) > 3:  # Keep first 3 separators (universal section)
            separator = self.separators.pop()
            self.removeAction(separator)
            
        actions = self.actions()
        universal_count = 7  # Number of universal actions + separators
        
        for action in actions[universal_count:]:
            self.removeAction(action)
            
    def get_action(self, name: str) -> Optional[QAction]:
        """Get action by name"""
        return self.action_buttons.get(name)
        
    def enable_action(self, name: str, enabled: bool = True):
        """Enable/disable action by name"""
        action = self.get_action(name)
        if action:
            action.setEnabled(enabled)
            
    def set_action_checked(self, name: str, checked: bool):
        """Set action checked state"""
        action = self.get_action(name)
        if action and action.isCheckable():
            action.setChecked(checked)
            
    def connect_action(self, name: str, callback: Callable):
        """Connect action to callback"""
        action = self.get_action(name)
        if action:
            action.triggered.connect(callback)
            logger.info(f"Successfully connected action '{name}' to callback")
        else:
            logger.error(f"Action '{name}' not found in toolbar actions: {list(self.action_buttons.keys())}")
            
    def get_current_context(self) -> str:
        """Get current context mode"""
        return self.current_context