#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Window Manager for Modern Single-Window UI
Manages window transitions and state in the new single-window architecture

"""

import logging
from typing import Dict, List, Optional, Any
from PyQt5.QtCore import QObject, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtWidgets import QStackedWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QFont

logger = logging.getLogger(__name__)

class WindowManager(QObject):
    """Manages window transitions and state in single-window UI"""
    
    windowChanged = pyqtSignal(str)  # window_type
    windowSwitching = pyqtSignal(str, str)  # from_type, to_type
    backNavigationAvailable = pyqtSignal(bool)
    
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.stacked_widget = QStackedWidget()
        self.registered_windows = {}
        self.window_instances = {}
        self.current_window_type = 'workflow'
        self.window_history = ['workflow']
        self.window_states = {}
        
        self.fade_animation = None
        
        logger.info("WindowManager initialized")
        
    def get_stacked_widget(self) -> QStackedWidget:
        """Get the main stacked widget for window management"""
        return self.stacked_widget
        
    def register_window(self, window_type: str, widget_class, create_immediately=False):
        """Register a window type with its widget class"""
        self.registered_windows[window_type] = widget_class
        
        if create_immediately:
            self._create_window_instance(window_type)
            
        logger.debug(f"Registered window type: {window_type}")
        
    def _create_window_instance(self, window_type: str) -> QWidget:
        """Create an instance of the specified window type"""
        if window_type in self.window_instances:
            return self.window_instances[window_type]
            
        if window_type not in self.registered_windows:
            logger.error(f"Window type '{window_type}' not registered")
            return None
            
        widget_class = self.registered_windows[window_type]
        
        try:
            if window_type == 'workflow':
                widget = widget_class(self.main_window)
            else:
                widget = widget_class(self.main_window)
                
                if hasattr(widget, 'backRequested'):
                    widget.backRequested.connect(self.navigate_back)
                    logger.debug(f"Connected back button for {window_type}")
                
            self.window_instances[window_type] = widget
            self.stacked_widget.addWidget(widget)
            
            logger.debug(f"Created window instance: {window_type}")
            return widget
            
        except Exception as e:
            logger.error(f"Error creating window instance for '{window_type}': {e}")
            return None
        
    def show_window(self, window_type: str, data=None, animate=True):
        """Switch to specified window type with optional data"""
        if window_type == self.current_window_type:
            logger.debug(f"Already showing window: {window_type}")
            return True
            
        logger.info(f"Switching from '{self.current_window_type}' to '{window_type}'")
        
        self.windowSwitching.emit(self.current_window_type, window_type)
        
        self._save_window_state(self.current_window_type)
        
        target_widget = self._create_window_instance(window_type)
        if target_widget is None:
            logger.error(f"Failed to create window: {window_type}")
            return False
            
        if data and hasattr(target_widget, 'load_data'):
            try:
                target_widget.load_data(data)
            except Exception as e:
                logger.error(f"Error loading data into window '{window_type}': {e}")
                
        self._restore_window_state(window_type)
        
        if window_type != 'workflow':
            if len(self.window_history) == 0 or self.window_history[-1] != window_type:
                self.window_history.append(window_type)
        else:
            self.window_history = ['workflow']
            
        if animate:
            self._animate_window_transition(target_widget)
        else:
            self.stacked_widget.setCurrentWidget(target_widget)
            
        self.current_window_type = window_type
        
        self.windowChanged.emit(window_type)
        self.backNavigationAvailable.emit(self.can_navigate_back())
        
        if hasattr(self.main_window, 'switch_toolbar_context'):
            self.main_window.switch_toolbar_context(window_type)
            
        return True
        
    def navigate_back(self):
        """Navigate back to previous window"""
        if not self.can_navigate_back():
            logger.warning("Cannot navigate back - no history available")
            return False
            
        if len(self.window_history) > 1:
            self.window_history.pop()
            
        previous_window = self.window_history[-1] if self.window_history else 'workflow'
        
        logger.info(f"Navigating back to: {previous_window}")
        return self.show_window(previous_window)
        
    def navigate_to_workflow(self):
        """Navigate directly to workflow window"""
        return self.show_window('workflow')
        
    def can_navigate_back(self) -> bool:
        """Check if back navigation is possible"""
        return len(self.window_history) > 1 or self.current_window_type != 'workflow'
        
    def get_current_window_type(self) -> str:
        """Get currently active window type"""
        return self.current_window_type
        
    def get_current_window(self) -> Optional[QWidget]:
        """Get current window widget"""
        return self.window_instances.get(self.current_window_type)
        
    def get_window_history(self) -> List[str]:
        """Get navigation history"""
        return self.window_history.copy()
        
    def _save_window_state(self, window_type: str):
        """Save current state of a window"""
        if window_type in self.window_instances:
            widget = self.window_instances[window_type]
            
            state = {
                'scroll_position': None,
                'selected_items': [],
                'expanded_items': []
            }
            
            if hasattr(widget, 'save_state'):
                try:
                    custom_state = widget.save_state()
                    state.update(custom_state)
                except Exception as e:
                    logger.warning(f"Error saving state for '{window_type}': {e}")
                    
            self.window_states[window_type] = state
            logger.debug(f"Saved state for window: {window_type}")
            
    def _restore_window_state(self, window_type: str):
        """Restore saved state of a window"""
        if window_type in self.window_states and window_type in self.window_instances:
            widget = self.window_instances[window_type]
            state = self.window_states[window_type]
            
            if hasattr(widget, 'restore_state'):
                try:
                    widget.restore_state(state)
                    logger.debug(f"Restored state for window: {window_type}")
                except Exception as e:
                    logger.warning(f"Error restoring state for '{window_type}': {e}")
                    
    def _animate_window_transition(self, target_widget: QWidget):
        """Animate transition between windows"""
        try:
            self.stacked_widget.setCurrentWidget(target_widget)
            
            # TODO: Implement proper fade animation
            
        except Exception as e:
            logger.error(f"Error during window transition animation: {e}")
            self.stacked_widget.setCurrentWidget(target_widget)
            
    def clear_window_cache(self, window_type: str = None):
        """Clear cached window instances"""
        if window_type:
            if window_type in self.window_instances:
                widget = self.window_instances[window_type]
                self.stacked_widget.removeWidget(widget)
                del self.window_instances[window_type]
                logger.debug(f"Cleared window cache: {window_type}")
        else:
            for wtype in list(self.window_instances.keys()):
                if wtype != 'workflow':
                    self.clear_window_cache(wtype)
                    
    def get_window_count(self) -> int:
        """Get number of cached window instances"""
        return len(self.window_instances)
        
    def cleanup(self):
        """Cleanup resources"""
        logger.info("WindowManager cleanup started")
        
        for window_type in self.window_instances:
            self._save_window_state(window_type)
            
        self.window_instances.clear()
        self.window_states.clear()
        self.window_history.clear()
        
        logger.info("WindowManager cleanup completed")


class BreadcrumbWidget(QWidget):
    """Navigation breadcrumb widget for showing current location"""
    
    navigationRequested = pyqtSignal(str)  # window_type
    backRequested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.window_titles = {
            'workflow': '🏠 Workflow',
            'data': '📊 Data Analysis',
            'model': '🌳 Model Builder',
            'transform': '⚡ Data Transform',
            'evaluation': '📈 Model Evaluation',
            'export': '📤 Export'
        }
        
    def setup_ui(self):
        """Setup breadcrumb UI"""
        layout = QHBoxLayout()
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)
        
        self.back_button = QPushButton("⬅️")
        self.back_button.setFixedSize(32, 24)
        self.back_button.setEnabled(False)
        self.back_button.clicked.connect(self.backRequested.emit)
        self.back_button.setStyleSheet("""
            QPushButton {
                border: 1px solid #e2e8f0;
                border-radius: 4px;
                background-color: #f8fafc;
                font-size: 12px;
            }
            QPushButton:hover:enabled {
                background-color: #e2e8f0;
            }
            QPushButton:disabled {
                color: #94a3b8;
                background-color: #f1f5f9;
            }
        """)
        layout.addWidget(self.back_button)
        
        self.path_label = QLabel("🏠 Workflow")
        self.path_label.setStyleSheet("""
            QLabel {
                color: #374151;
                font-weight: 600;
                font-size: 12px;
                padding: 4px 8px;
            }
        """)
        layout.addWidget(self.path_label)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def update_breadcrumb(self, current_window: str, can_go_back: bool):
        """Update breadcrumb display"""
        self.back_button.setEnabled(can_go_back)
        
        title = self.window_titles.get(current_window, current_window.title())
        self.path_label.setText(title)
        
    def set_window_titles(self, titles: Dict[str, str]):
        """Set custom window titles for breadcrumb"""
        self.window_titles.update(titles)