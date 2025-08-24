#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Widget-Based Toolbar
Custom toolbar implemented as QWidget to bypass Qt's toolbar area issues
"""

import logging
from typing import Dict, List, Optional, Callable
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QLabel, QFrame, 
    QSizePolicy, QSpacerItem, QMenu, QVBoxLayout
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QPalette, QColor

logger = logging.getLogger(__name__)

class WidgetToolbar(QWidget):
    """Custom toolbar implemented as a regular widget"""
    
    actionTriggered = pyqtSignal(str)  # action_name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.buttons = {}
        self.setup_ui()
        self.setup_styling()
        logger.info("WidgetToolbar initialized")
        
    def setup_ui(self):
        """Setup the toolbar UI structure"""
        self.setFixedHeight(50)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        self.setVisible(True)
        self.show()
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(8, 4, 8, 4)
        self.layout.setSpacing(4)
        
        self.layout.addItem(QSpacerItem(20, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))
        
    def setup_styling(self):
        """Apply modern styling to the toolbar"""
        self.setStyleSheet("""
            WidgetToolbar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #f8fafc);
                border: none;
                border-bottom: 2px solid #e2e8f0;
                margin: 0px;
                padding: 4px 8px;
            }
            
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                padding: 8px 16px;
                margin: 2px;
                font-size: 11px;
                font-weight: 600;
                color: #374151;
                min-width: 80px;
                min-height: 32px;
            }
            
            QPushButton:hover {
                background-color: #f1f5f9;
                border-color: #94a3b8;
                color: #1e293b;
            }
            
            QPushButton:pressed {
                background-color: #e2e8f0;
                border-color: #64748b;
            }
            
            QPushButton.primary {
                background-color: #3b82f6;
                color: white;
                border-color: #2563eb;
            }
            
            QPushButton.primary:hover {
                background-color: #2563eb;
                border-color: #1d4ed8;
            }
            
            QPushButton.secondary {
                background-color: #f8fafc;
                color: #475569;
                border-color: #cbd5e1;
            }
            
            QPushButton.success {
                background-color: #10b981;
                color: white;
                border-color: #059669;
            }
            
            QPushButton.success:hover {
                background-color: #059669;
                border-color: #047857;
            }
            
            QPushButton.danger {
                background-color: #ef4444;
                color: white;
                border-color: #dc2626;
            }
            
            QPushButton.danger:hover {
                background-color: #dc2626;
                border-color: #b91c1c;
            }
            
            QLabel.separator {
                color: #9ca3af;
                font-size: 16px;
                margin: 0px 4px;
            }
        """)
        
    def add_button(self, text: str, callback: Callable, 
                   tooltip: str = "", style: str = "default", 
                   icon_text: str = "") -> QPushButton:
        """Add a button to the toolbar"""
        
        display_text = f"{icon_text} {text}" if icon_text else text
        button = QPushButton(display_text, self)
        
        if tooltip:
            button.setToolTip(tooltip)
            button.setStatusTip(tooltip)
            
        if callback:
            button.clicked.connect(lambda: self._handle_button_click(text, callback))
            
        if style != "default":
            button.setProperty("class", style)
            
        self.layout.addWidget(button)
        self.buttons[text.lower().replace(" ", "_")] = button
        
        logger.debug(f"Added button: {text} with style: {style}")
        return button
        
    def add_separator(self):
        """Add a visual separator"""
        separator = QLabel("│")
        separator.setProperty("class", "separator")
        separator.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(separator)
        
    def add_spacer(self, width: int = 20):
        """Add spacing between elements"""
        spacer = QSpacerItem(width, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)
        self.layout.addItem(spacer)
        
    def _handle_button_click(self, action_name: str, callback: Callable):
        """Handle button click and emit signal"""
        try:
            callback()
            self.actionTriggered.emit(action_name)
            logger.debug(f"Toolbar action triggered: {action_name}")
        except Exception as e:
            logger.error(f"Error executing toolbar action {action_name}: {e}")
            
    def get_button(self, name: str) -> Optional[QPushButton]:
        """Get button by name"""
        return self.buttons.get(name.lower().replace(" ", "_"))
        
    def set_button_enabled(self, name: str, enabled: bool):
        """Enable/disable button by name"""
        button = self.get_button(name)
        if button:
            button.setEnabled(enabled)
            
    def set_button_visible(self, name: str, visible: bool):
        """Show/hide button by name"""
        button = self.get_button(name)
        if button:
            button.setVisible(visible)
            
    def clear_buttons(self):
        """Remove all buttons from toolbar"""
        for button in self.buttons.values():
            button.deleteLater()
        self.buttons.clear()
        
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
        self.layout.addItem(QSpacerItem(20, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))

class ToolbarContainer(QWidget):
    """Container that holds the widget toolbar and integrates with main window"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.toolbar = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup container layout"""
        self.setFixedHeight(50)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        self.setVisible(True)
        self.show()
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
    def set_toolbar(self, toolbar: WidgetToolbar):
        """Set the toolbar widget"""
        if self.toolbar:
            self.layout.removeWidget(self.toolbar)
            self.toolbar.deleteLater()
            
        self.toolbar = toolbar
        self.layout.addWidget(toolbar)
        
    def get_toolbar(self) -> Optional[WidgetToolbar]:
        """Get the current toolbar"""
        return self.toolbar