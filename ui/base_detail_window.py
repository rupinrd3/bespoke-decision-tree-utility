#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base Detail Window for Bespoke Utility
Provides a base class for detail windows with common functionality

"""

import logging
from typing import Dict, Any, Optional, List
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QScrollArea, QSizePolicy, QSpacerItem
)
from PyQt5.QtGui import QFont, QIcon

logger = logging.getLogger(__name__)

class BaseDetailWindow(QWidget):
    """Base class for all detail windows with common chrome and functionality"""
    
    backRequested = pyqtSignal()
    titleChanged = pyqtSignal(str)
    actionTriggered = pyqtSignal(str, dict)  # action_name, parameters
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = "Detail Window"
        self.action_buttons = {}
        self.content_widget = None
        self.scroll_area = None
        
        self.setup_window_chrome()
        self.setup_content_area()
        self.apply_base_styling()
        
        logger.debug(f"BaseDetailWindow initialized: {self.__class__.__name__}")
        
    def setup_window_chrome(self):
        """Setup common window elements (header, back button, actions)"""
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.setLayout(self.main_layout)
        
        self.header_frame = QFrame()
        self.header_frame.setObjectName("headerFrame")
        self.header_frame.setFixedHeight(56)
        
        header_layout = QHBoxLayout(self.header_frame)
        header_layout.setContentsMargins(16, 8, 16, 8)
        header_layout.setSpacing(12)
        
        self.back_button = QPushButton("⬅️ Back")
        self.back_button.setObjectName("backButton")
        self.back_button.setFixedSize(80, 32)
        self.back_button.clicked.connect(self.backRequested.emit)
        header_layout.addWidget(self.back_button)
        
        self.title_label = QLabel(self.window_title)
        self.title_label.setObjectName("titleLabel")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        header_layout.addWidget(self.title_label)
        
        header_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        self.action_buttons_layout = QHBoxLayout()
        self.action_buttons_layout.setSpacing(8)
        header_layout.addLayout(self.action_buttons_layout)
        
        self.main_layout.addWidget(self.header_frame)
        
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setObjectName("headerSeparator")
        separator.setFixedHeight(1)
        self.main_layout.addWidget(separator)
        
    def setup_content_area(self):
        """Setup main content area with scroll support"""
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setObjectName("contentScrollArea")
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.content_widget = QWidget()
        self.content_widget.setObjectName("contentWidget")
        
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(16, 16, 16, 16)
        self.content_layout.setSpacing(16)
        
        self.scroll_area.setWidget(self.content_widget)
        
        self.main_layout.addWidget(self.scroll_area)
        
    def apply_base_styling(self):
        """Apply base styling to the window"""
        self.setStyleSheet("""
            BaseDetailWindow {
                background-color: #f8fafc;
            }
            
            QFrame#headerFrame {
                background-color: #ffffff;
                border: none;
            }
            
            QFrame#headerSeparator {
                background-color: #e2e8f0;
                border: none;
            }
            
            QLabel#titleLabel {
                color: #1e293b;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            QPushButton#backButton {
                background-color: #6b7280;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: 600;
                font-size: 11px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            QPushButton#backButton:hover {
                background-color: #4b5563;
            }
            
            QPushButton#backButton:pressed {
                background-color: #374151;
            }
            
            QScrollArea#contentScrollArea {
                border: none;
                background-color: #f8fafc;
            }
            
            QWidget#contentWidget {
                background-color: #f8fafc;
            }
            
            QPushButton.actionButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: 600;
                font-size: 11px;
                font-family: 'Segoe UI', Arial, sans-serif;
                min-width: 80px;
            }
            
            QPushButton.actionButton:hover {
                background-color: #2563eb;
            }
            
            QPushButton.actionButton:pressed {
                background-color: #1d4ed8;
            }
            
            QPushButton.secondaryAction {
                background-color: #f1f5f9;
                color: #1e293b;
                border: 1px solid #e2e8f0;
            }
            
            QPushButton.secondaryAction:hover {
                background-color: #e2e8f0;
            }
            
            QPushButton.dangerAction {
                background-color: #ef4444;
                color: white;
            }
            
            QPushButton.dangerAction:hover {
                background-color: #dc2626;
            }
            
            QPushButton.successAction {
                background-color: #10b981;
                color: white;
            }
            
            QPushButton.successAction:hover {
                background-color: #059669;
            }
        """)
        
    def set_window_title(self, title: str):
        """Set window title"""
        self.window_title = title
        self.title_label.setText(title)
        self.titleChanged.emit(title)
        
    def add_action_button(self, name: str, text: str, callback, style="primary", enabled=True):
        """Add action button to window chrome"""
        if name in self.action_buttons:
            self.remove_action_button(name)
            
        button = QPushButton(text)
        button.setObjectName(f"actionButton_{name}")
        button.clicked.connect(lambda: self._handle_action_click(name, callback))
        button.setEnabled(enabled)
        
        if style == "primary":
            button.setProperty("class", "actionButton")
        elif style == "secondary":
            button.setProperty("class", "actionButton secondaryAction")
        elif style == "danger":
            button.setProperty("class", "actionButton dangerAction")
        elif style == "success":
            button.setProperty("class", "actionButton successAction")
        else:
            button.setProperty("class", "actionButton")
            
        button.style().unpolish(button)
        button.style().polish(button)
        
        self.action_buttons[name] = button
        self.action_buttons_layout.addWidget(button)
        
        logger.debug(f"Added action button: {name}")
        
    def remove_action_button(self, name: str):
        """Remove action button from window chrome"""
        if name in self.action_buttons:
            button = self.action_buttons[name]
            self.action_buttons_layout.removeWidget(button)
            button.deleteLater()
            del self.action_buttons[name]
            logger.debug(f"Removed action button: {name}")
            
    def enable_action_button(self, name: str, enabled: bool = True):
        """Enable/disable action button"""
        if name in self.action_buttons:
            self.action_buttons[name].setEnabled(enabled)
            
    def add_zoom_controls(self, zoom_in_callback, zoom_out_callback, zoom_fit_callback):
        """Add zoom control buttons to the header (for Model Builder window)"""
        self.add_action_button("zoom_in", "🔍+", zoom_in_callback, "secondary")
        self.add_action_button("zoom_out", "🔍-", zoom_out_callback, "secondary") 
        self.add_action_button("zoom_fit", "⭘", zoom_fit_callback, "secondary")
        
        for zoom_btn_name in ["zoom_in", "zoom_out", "zoom_fit"]:
            if zoom_btn_name in self.action_buttons:
                btn = self.action_buttons[zoom_btn_name]
                btn.setFixedSize(32, 28)
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #e5e7eb;
                        color: #374151;
                        border: 1px solid #d1d5db;
                        border-radius: 4px;
                        font-size: 12px;
                        font-weight: bold;
                        padding: 2px;
                    }
                    QPushButton:hover {
                        background-color: #d1d5db;
                    }
                    QPushButton:pressed {
                        background-color: #9ca3af;
                    }
                """)
    
    def update_action_button_text(self, name: str, text: str):
        """Update action button text"""
        if name in self.action_buttons:
            self.action_buttons[name].setText(text)
            
    def _handle_action_click(self, name: str, callback):
        """Handle action button click"""
        try:
            if callable(callback):
                result = callback()
                self.actionTriggered.emit(name, {"result": result})
            else:
                logger.warning(f"Action callback for '{name}' is not callable")
        except Exception as e:
            logger.error(f"Error handling action '{name}': {e}")
            self.actionTriggered.emit(name, {"error": str(e)})
            
    def get_content_layout(self) -> QVBoxLayout:
        """Get the main content layout for subclasses to add content"""
        return self.content_layout
        
    def add_content_widget(self, widget: QWidget, stretch: int = 0):
        """Add widget to content area"""
        self.content_layout.addWidget(widget, stretch)
        
    def add_content_layout(self, layout, stretch: int = 0):
        """Add layout to content area"""
        self.content_layout.addLayout(layout, stretch)
        
    def clear_content(self):
        """Clear all content from the content area"""
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self._clear_layout(child.layout())
                
    def _clear_layout(self, layout):
        """Recursively clear a layout"""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                self._clear_layout(child.layout())
                
    def show_loading_state(self, message: str = "Loading..."):
        """Show loading state in the window"""
        self.clear_content()
        
        loading_label = QLabel(f"🔄 {message}")
        loading_label.setAlignment(Qt.AlignCenter)
        loading_label.setStyleSheet("""
            QLabel {
                color: #6b7280;
                font-size: 14px;
                font-style: italic;
                padding: 40px;
            }
        """)
        
        self.add_content_widget(loading_label)
        
    def show_error_state(self, message: str = "An error occurred"):
        """Show error state in the window"""
        self.clear_content()
        
        error_label = QLabel(f"❌ {message}")
        error_label.setAlignment(Qt.AlignCenter)
        error_label.setStyleSheet("""
            QLabel {
                color: #ef4444;
                font-size: 14px;
                padding: 40px;
            }
        """)
        
        self.add_content_widget(error_label)
        
    def show_empty_state(self, message: str = "No data available"):
        """Show empty state in the window"""
        self.clear_content()
        
        empty_label = QLabel(f"📭 {message}")
        empty_label.setAlignment(Qt.AlignCenter)
        empty_label.setStyleSheet("""
            QLabel {
                color: #6b7280;
                font-size: 14px;
                padding: 40px;
            }
        """)
        
        self.add_content_widget(empty_label)
        
    def save_state(self) -> Dict[str, Any]:
        """Save window state - override in subclasses"""
        return {
            'scroll_position': self.scroll_area.verticalScrollBar().value(),
            'window_title': self.window_title
        }
        
    def restore_state(self, state: Dict[str, Any]):
        """Restore window state - override in subclasses"""
        if 'scroll_position' in state:
            self.scroll_area.verticalScrollBar().setValue(state['scroll_position'])
            
        if 'window_title' in state:
            self.set_window_title(state['window_title'])
            
    def set_status(self, message: str):
        """Set status message for the window"""
        self.set_window_title(f"{self.window_title} - {message}")
        logger.debug(f"Status set: {message}")
        
    def load_data(self, data: Any):
        """Load data into the window - override in subclasses"""
        logger.debug(f"BaseDetailWindow.load_data called with: {type(data)}")
        self.show_loading_state("Loading data...")
        
    def refresh_content(self):
        """Refresh window content - override in subclasses"""
        logger.debug("BaseDetailWindow.refresh_content called")
        
    def cleanup(self):
        """Cleanup resources when window is destroyed"""
        logger.debug(f"BaseDetailWindow cleanup: {self.__class__.__name__}")
        
        for name in list(self.action_buttons.keys()):
            self.remove_action_button(name)
            
        self.clear_content()


class DataDetailWindow(BaseDetailWindow):
    """Detail window for data analysis and viewing"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.set_window_title("📊 Data Analysis")
        self.dataset = None
        self.dataset_name = None
        self.main_window = parent
        self.setup_data_actions()
        self.setup_data_content()
        
    def setup_data_actions(self):
        """Setup data-specific action buttons"""
        self.add_action_button("refresh", "🔄 Refresh", self.refresh_data, "secondary")
        self.add_action_button("filter", "🔍 Filter", self.filter_data, "secondary")
        self.add_action_button("export", "📤 Export", self.export_data, "primary")
        
    def setup_data_content(self):
        """Setup data viewing content area"""
        try:
            from ui.data_viewer import DataViewerWidget
            self.data_viewer = DataViewerWidget()
            self.add_content_widget(self.data_viewer, stretch=1)
            logger.info("DataDetailWindow content area setup completed")
        except ImportError as e:
            logger.error(f"Failed to import DataViewerWidget: {e}")
            self.show_error_state("Data viewer component not available")
    
    def _ensure_data_viewer(self):
        """Ensure data viewer is available and recreate if needed"""
        try:
            if not hasattr(self, 'data_viewer') or self.data_viewer is None:
                logger.info("Data viewer not found, creating new one")
                self.setup_data_content()
                return True
            
            if hasattr(self.data_viewer, 'table_view') and self.data_viewer.table_view is not None:
                try:
                    _ = self.data_viewer.table_view.isVisible()
                    return True
                except RuntimeError:
                    logger.warning("Data viewer widgets have been deleted, recreating")
                    self.clear_content()
                    self.setup_data_content()
                    return True
            
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring data viewer: {e}")
            return False
    
    def _safe_set_dataframe(self, dataframe):
        """Safely set dataframe in data viewer with error handling"""
        try:
            if not self._ensure_data_viewer():
                logger.error("Failed to ensure data viewer is available")
                return False
            
            self.data_viewer.set_dataframe(dataframe)
            return True
            
        except Exception as e:
            logger.error(f"Error setting dataframe in data viewer: {e}")
            return False
    
    def _get_input_data_for_node(self, workflow_node):
        """Get input data from connected nodes for configuration purposes"""
        try:
            import pandas as pd
            
            if hasattr(self.main_window, 'workflow_canvas') and hasattr(self.main_window.workflow_canvas, 'scene'):
                scene = self.main_window.workflow_canvas.scene
                if hasattr(scene, 'connections'):
                    for connection in scene.connections.values():
                        if connection.target_node_id == workflow_node.node_id:
                            source_node_id = connection.source_node_id
                            source_node = scene.nodes.get(source_node_id)
                            
                            if source_node:
                                source_node_title = source_node.title
                                
                                if source_node.node_type == 'dataset':
                                    output_port_name = 'Data Output'
                                elif source_node.node_type == 'filter':
                                    output_port_name = 'Filtered Data'
                                elif source_node.node_type == 'transform':
                                    output_port_name = 'Transformed Data'
                                else:
                                    continue
                                
                                compound_name = f"{source_node_title}_Output_{output_port_name}"
                                if hasattr(self.main_window, 'datasets') and compound_name in self.main_window.datasets:
                                    return self.main_window.datasets[compound_name]
                                
                                if hasattr(self.main_window, 'datasets'):
                                    for dataset_name, dataset_data in self.main_window.datasets.items():
                                        if dataset_name.startswith(compound_name):
                                            return dataset_data
                                
                                if source_node.node_type == 'dataset' and hasattr(source_node, 'dataset_name'):
                                    dataset_name = source_node.dataset_name
                                    if hasattr(self.main_window, 'datasets') and dataset_name in self.main_window.datasets:
                                        return self.main_window.datasets[dataset_name]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting input data for node: {e}")
            return None
    
    def load_data(self, data):
        """Load dataset data into the window"""
        logger.info(f"DataDetailWindow.load_data called with: {data}")
        
        try:
            if not data or 'node_id' not in data:
                self.show_error_state("No data provided")
                return
                
            node_id = data['node_id']
            self.dataset_name = data.get('name', 'Dataset')
            self.set_window_title(f"📊 {self.dataset_name}")
            
            if hasattr(self.main_window, 'workflow_canvas'):
                workflow_node = self.main_window.workflow_canvas.get_node_by_id(node_id)
                if workflow_node:
                    if hasattr(workflow_node, 'dataset_name') and workflow_node.dataset_name:
                        dataset_name = workflow_node.dataset_name
                        logger.info(f"Looking up dataset '{dataset_name}' for node {node_id}")
                        
                        if hasattr(self.main_window, 'datasets') and dataset_name in self.main_window.datasets:
                            self.dataset = self.main_window.datasets[dataset_name]
                            logger.info(f"Dataset '{dataset_name}' loaded: {len(self.dataset)} rows, {len(self.dataset.columns)} columns")
                            
                            display_data = self.dataset.head(500)
                            if self._safe_set_dataframe(display_data):
                                logger.info(f"Displaying first 500 rows of {len(self.dataset)} total rows")
                            return
                        else:
                            logger.warning(f"Dataset '{dataset_name}' not found in main window datasets")
                    
                    if hasattr(self.main_window, 'datasets') and self.main_window.datasets:
                        import pandas as pd
                        
                        node_title = workflow_node.title
                        
                        output_port_name = None
                        if workflow_node.node_type == 'filter':
                            output_port_name = 'Filtered Data'
                        elif workflow_node.node_type == 'transform':
                            output_port_name = 'Transformed Data'
                        elif workflow_node.node_type == 'dataset':
                            output_port_name = 'Data Output'
                        
                        if output_port_name:
                            compound_name = f"{node_title}_Output_{output_port_name}"
                            
                            if compound_name in self.main_window.datasets:
                                self.dataset = self.main_window.datasets[compound_name]
                                logger.info(f"Dataset loaded from compound name '{compound_name}': {len(self.dataset)} rows, {len(self.dataset.columns)} columns")
                                
                                display_data = self.dataset.head(500)
                                if self._safe_set_dataframe(display_data):
                                    logger.info(f"Displaying first 500 rows of {len(self.dataset)} total rows")
                                return
                            
                            for dataset_name, dataset_data in self.main_window.datasets.items():
                                if dataset_name.startswith(compound_name):
                                    self.dataset = dataset_data
                                    logger.info(f"Dataset loaded from compound name variant '{dataset_name}': {len(self.dataset)} rows, {len(self.dataset.columns)} columns")
                                    
                                    display_data = self.dataset.head(500)
                                    if self._safe_set_dataframe(display_data):
                                        logger.info(f"Displaying first 500 rows of {len(self.dataset)} total rows")
                                    return
                        
                        if workflow_node.node_type in ['filter', 'transform', 'model', 'evaluation', 'visualization']:
                            input_data = self._get_input_data_for_node(workflow_node)
                            if input_data is not None:
                                self.dataset = input_data
                                logger.info(f"Dataset loaded from input connection for {workflow_node.node_type} node configuration: {len(self.dataset)} rows, {len(self.dataset.columns)} columns")
                                
                                display_data = self.dataset.head(500)
                                if self._safe_set_dataframe(display_data):
                                    logger.info(f"Displaying first 500 rows of {len(self.dataset)} total rows for configuration")
                                return
                    
                    if hasattr(self.main_window, 'workflow_execution_engine') and self.main_window.workflow_execution_engine:
                        import pandas as pd
                        execution_engine = self.main_window.workflow_execution_engine
                        
                        if hasattr(execution_engine, 'node_outputs') and node_id in execution_engine.node_outputs:
                            node_outputs = execution_engine.node_outputs[node_id]
                            
                            data_output = None
                            for output_name, output_data in node_outputs.items():
                                if isinstance(output_data, pd.DataFrame):
                                    data_output = output_data
                                    break
                            
                            if data_output is not None:
                                self.dataset = data_output
                                logger.info(f"Dataset loaded from workflow execution: {len(self.dataset)} rows, {len(self.dataset.columns)} columns")
                                
                                display_data = self.dataset.head(500)
                                if self._safe_set_dataframe(display_data):
                                    logger.info(f"Displaying first 500 rows of {len(self.dataset)} total rows")
                                return
                    
                    if hasattr(workflow_node, 'dataset') and workflow_node.dataset is not None:
                        self.dataset = workflow_node.dataset
                        logger.info(f"Dataset loaded from workflow node (legacy): {len(self.dataset)} rows, {len(self.dataset.columns)} columns")
                        
                        display_data = self.dataset.head(500)
                        if self._safe_set_dataframe(display_data):
                            logger.info(f"Displaying first 500 rows of {len(self.dataset)} total rows")
                        return
                
                if hasattr(self.main_window, 'current_dataset') and self.main_window.current_dataset is not None:
                    self.dataset = self.main_window.current_dataset
                    logger.info(f"Using main window current dataset: {len(self.dataset)} rows, {len(self.dataset.columns)} columns")
                    
                    display_data = self.dataset.head(500)
                    if self._safe_set_dataframe(display_data):
                        logger.info(f"Displaying first 500 rows of {len(self.dataset)} total rows")
                    return
                    
                self.show_error_state(f"No dataset found for node {node_id}")
                logger.error(f"No dataset available for node {node_id}. Available datasets: {list(getattr(self.main_window, 'datasets', {}).keys())}")
            else:
                self.show_error_state("Workflow canvas not available")
                logger.error("Main window has no workflow_canvas attribute")
                
        except Exception as e:
            logger.error(f"Error loading dataset: {e}", exc_info=True)
            self.show_error_state(f"Error loading dataset: {str(e)}")
    
    def refresh_data(self):
        """Refresh dataset display"""
        if self.dataset is not None:
            display_data = self.dataset.head(500)
            self.data_viewer.set_dataframe(display_data)
            logger.info("Dataset refreshed")
        
    def filter_data(self):
        """Filter data action"""
        logger.info("Filter data requested")
        # TODO: Implement data filtering
        
    def export_data(self):
        """Export data action"""
        logger.info("Export data requested")
        # TODO: Implement data export
        

class ModelDetailWindow(BaseDetailWindow):
    """Detail window for model building and editing"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.set_window_title("🌳 Model Builder")
        self.model = None
        self.model_name = None
        self.tree_visualizer = None
        self.main_window = parent
        self.setup_model_actions()
        self.setup_model_content()
        
    def setup_model_actions(self):
        """Setup model-specific action buttons"""
        self.add_zoom_controls(
            self.zoom_in,
            self.zoom_out, 
            self.zoom_fit
        )
        self.add_action_button("export_model", "📤 Export", self.export_model, "success")
        
    def setup_model_content(self):
        """Setup the model builder content area"""
        from ui.tree_visualizer import TreeVisualizerWidget
        
        self.tree_visualizer = TreeVisualizerWidget(self)
        self.add_content_widget(self.tree_visualizer, stretch=1)
        
        self.connect_tree_signals()
        
        logger.info("ModelDetailWindow content area setup completed")
        
    def connect_tree_signals(self):
        """Connect tree visualizer signals to handlers"""
        if hasattr(self, 'tree_visualizer'):
            self.tree_visualizer.connect_signals()
            
            self.tree_visualizer.nodeSelected.connect(self.on_tree_node_selected)
            self.tree_visualizer.nodeDoubleClicked.connect(self.on_tree_node_double_clicked)
            self.tree_visualizer.nodeRightClicked.connect(self.on_tree_node_right_clicked)
            
            logger.info("Tree visualizer signals connected in ModelDetailWindow")
    
    def on_tree_node_selected(self, node_id):
        """Handle tree node selection"""
        logger.info(f"Tree node selected: {node_id}")
        self.set_status(f"Selected node: {node_id}")
    
    def on_tree_node_double_clicked(self, node_id):
        """Handle tree node double-click"""
        logger.info(f"Tree node double-clicked: {node_id}")
    
    def on_tree_node_right_clicked(self, node_id, scene_position, global_position):
        """Handle tree node right-click"""
        logger.info(f"Tree node right-clicked: {node_id} at scene position: {scene_position}, global position: {global_position}")
        
        try:
            if hasattr(self, 'tree_visualizer') and hasattr(self.tree_visualizer, 'view'):
                global_pos = global_position
                logger.info(f"Using provided global position: {global_pos}")
            else:
                global_pos = global_position
                logger.info("Using provided global position as fallback")
        
            from PyQt5.QtWidgets import QApplication
            main_window = None
            for widget in QApplication.topLevelWidgets():
                if widget.__class__.__name__ == 'MainWindow':
                    main_window = widget
                    break
            
            if main_window and hasattr(main_window, 'show_enhanced_context_menu'):
                if hasattr(main_window, 'tree_context_menu'):
                    model = getattr(self, 'model', None)
                    if model and hasattr(model, 'get_node'):
                        node = model.get_node(node_id)
                        if node:
                            if node.node_id == 'root' or (hasattr(node, 'parent') and node.parent is None):
                                node_type = "root"
                            elif node.is_terminal or len(node.children) == 0:
                                node_type = "terminal"
                            else:
                                node_type = "internal"
                            
                            node_state = {
                                'is_terminal': node.is_terminal,
                                'has_children': len(node.children) > 0,
                                'is_expanded': True,
                                'can_split': node.samples > 1 if hasattr(node, 'samples') else True,
                                'sample_count': getattr(node, 'samples', 0)
                            }
                            
                            main_window.tree_context_menu.show_for_node(node_id, node_type, node_state, global_pos)
                            logger.info(f"Showed context menu at global position {global_pos}")
                        else:
                            logger.warning(f"Node {node_id} not found in model")
                    else:
                        logger.warning("Model not available for context menu")
                else:
                    logger.warning("Tree context menu not available in main window")
            else:
                logger.warning("Could not find main window or context menu handler")
                
        except Exception as e:
            logger.error(f"Error handling tree node right-click: {e}", exc_info=True)
        
    def load_data(self, data):
        """Load model data into the window"""
        logger.info(f"ModelDetailWindow.load_data called with: {data}")
        
        try:
            if not data or 'node_id' not in data:
                self.show_error_state("No model data provided")
                return
                
            node_id = data['node_id']
            
            if hasattr(self.main_window, 'workflow_canvas'):
                workflow_node = self.main_window.workflow_canvas.get_node_by_id(node_id)
                if workflow_node:
                    if hasattr(workflow_node, 'model') and workflow_node.model is not None:
                        self.model = workflow_node.model
                        self.model_name = getattr(self.model, 'model_name', data.get('name', 'Unknown Model'))
                        logger.info(f"Model loaded directly from workflow node: {self.model_name}")
                    else:
                        node_config = workflow_node.get_config()
                        model_ref_id = node_config.get('model_ref_id')
                        
                        if model_ref_id and hasattr(self.main_window, 'models') and model_ref_id in self.main_window.models:
                            self.model = self.main_window.models[model_ref_id]
                            self.model_name = model_ref_id
                            logger.info(f"Model loaded from main window models: {model_ref_id}")
                        else:
                            self.show_error_state(f"Workflow node has no associated model (node_id: {node_id})")
                            return
                    
                    self.set_window_title(f"🌳 Model Builder - {self.model_name}")
                    
                    if self.model and hasattr(self.model, 'root') and self.model.root:
                        logger.info(f"Loading tree with root node into visualizer")
                        self.tree_visualizer.set_tree(self.model.root)
                        
                        self.update_action_states()
                    else:
                        self.show_empty_state("Model has no tree structure yet. Build the tree first.")
                        self.update_action_states()
                else:
                    self.show_error_state("Workflow node not found")
            else:
                self.show_error_state("Cannot access workflow canvas")
                
        except Exception as e:
            logger.error(f"Error loading model data: {e}", exc_info=True)
            self.show_error_state(f"Error loading model: {str(e)}")
            
    def update_action_states(self):
        """Update action button states based on model state"""
        if self.model:
            has_tree = hasattr(self.model, 'root') and self.model.root is not None
            is_fitted = hasattr(self.model, 'is_fitted') and self.model.is_fitted
            
            self.enable_action_button("view_tree", has_tree)
            self.enable_action_button("edit_splits", has_tree)
            self.enable_action_button("tune", is_fitted)
            self.enable_action_button("export_model", has_tree)
        else:
            for action in ["view_tree", "edit_splits", "tune", "export_model"]:
                self.enable_action_button(action, False)
        
    def view_tree(self):
        """View tree visualization"""
        logger.info("View tree requested")
        if self.model and hasattr(self.model, 'root') and self.model.root:
            self.tree_visualizer.set_tree(self.model.root)
            self.tree_visualizer.fit_to_view()
        else:
            self.show_empty_state("No tree available to view")
        
    def edit_splits(self):
        """Edit tree splits"""
        logger.info("Edit splits requested")
        
        if not self.model or not hasattr(self.model, 'root') or not self.model.root:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Tree", "No tree structure available for editing splits.")
            return
            
        if self.tree_visualizer:
            self.tree_visualizer.set_edit_mode(True)
            
        # TODO: Could also open a dedicated split editing dialog here if needed
        
    def tune_parameters(self):
        """Tune model parameters"""
        logger.info("Tune parameters requested")
        
        if not self.model:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Model", "No model available for parameter tuning.")
            return
            
        try:
            from ui.dialogs.enhanced_tree_configuration_dialog import EnhancedTreeConfigurationDialog
            
            dialog = EnhancedTreeConfigurationDialog(
                model=self.model,
                parent=self
            )
            
            if dialog.exec_() == dialog.Accepted:
                if self.tree_visualizer and self.model.root:
                    self.tree_visualizer.set_tree(self.model.root)
                    
        except ImportError as e:
            logger.error(f"Could not import tree configuration dialog: {e}")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Feature Unavailable", 
                              "Parameter tuning dialog is not available.")
        
    def export_model(self):
        """Export tree image"""
        logger.info("Export tree image requested")
        
        if not self.model:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Model", "No model available for export.")
            return
            
        if hasattr(self, 'tree_visualizer') and self.tree_visualizer:
            self.tree_visualizer.export_image()
        else:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Export Error", "Tree visualization not available for export.")
    
    def zoom_in(self):
        """Zoom in on tree visualization"""
        if hasattr(self, 'tree_visualizer') and self.tree_visualizer:
            self.tree_visualizer.zoom_in()
    
    def zoom_out(self):
        """Zoom out on tree visualization"""
        if hasattr(self, 'tree_visualizer') and self.tree_visualizer:
            self.tree_visualizer.zoom_out()
    
    def zoom_fit(self):
        """Fit tree visualization to view"""
        if hasattr(self, 'tree_visualizer') and self.tree_visualizer:
            self.tree_visualizer.zoom_fit()
            
    def refresh_content(self):
        """Refresh the model content"""
        if self.model and self.tree_visualizer:
            if hasattr(self.model, 'root') and self.model.root:
                self.tree_visualizer.set_tree(self.model.root)
            self.update_action_states()
            
    def _on_split_modified(self, node: 'TreeNode', feature: str, split_config: dict):
        """Handle split modification from enhanced edit split dialog in detail window context"""
        try:
            logger.info(f"Detail window processing split modification for node {node.node_id}")
            logger.debug(f"Split config received: {split_config}")
            
            if self.model:
                from ui.main_window import MainWindow
                main_window = None
                from PyQt5.QtWidgets import QApplication
                for widget in QApplication.topLevelWidgets():
                    if isinstance(widget, MainWindow):
                        main_window = widget
                        break
                
                if main_window and hasattr(main_window, '_convert_dialog_config_to_model_format'):
                    model_split_config = main_window._convert_dialog_config_to_model_format(split_config)
                else:
                    model_split_config = split_config
                
                success = self.model.apply_manual_split(node.node_id, model_split_config)
                
                if success:
                    logger.info(f"Successfully applied split modification for node {node.node_id} in detail window")
                    
                    logger.info(f"Refreshing detail window tree visualization for node {node.node_id}")
                    if hasattr(self, 'tree_visualizer') and self.tree_visualizer:
                        logger.info(f"Detail window tree visualizer found, refreshing with model root: {self.model.root.node_id if self.model.root else 'None'}")
                        if hasattr(self.model, 'root') and self.model.root:
                            self.tree_visualizer.set_tree(self.model.root, self.model)
                            logger.info(f"Detail window tree visualization refreshed successfully after split modification")
                        else:
                            logger.warning(f"Model root not available for detail window tree visualization refresh")
                    else:
                        logger.warning(f"Detail window tree visualizer not available for refresh")
                        
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.information(
                        self, "Split Modified",
                        f"Split successfully modified for node {node.node_id} using feature '{feature}'"
                    )
                else:
                    logger.warning(f"Failed to apply split modification for node {node.node_id} in detail window")
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.warning(
                        self, "Split Modification Failed", 
                        "Failed to apply the split modification. Check the log for details."
                    )
                    
        except Exception as e:
            logger.error(f"Error handling split modification in detail window: {e}", exc_info=True)
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(
                self, "Split Modification Error",
                f"An error occurred while modifying the split: {e}"
            )