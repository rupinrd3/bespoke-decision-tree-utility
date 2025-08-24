#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Workflow Canvas Module for Bespoke Utility
Provides a graphical canvas for creating and managing the analysis workflow

"""

import logging
import uuid
from typing import Dict, List, Tuple, Optional, Any, Set, Union

from PyQt5.QtCore import (Qt, QPointF, QRectF, QSizeF, QLineF, 
                        pyqtSignal, pyqtSlot, QObject, QEvent)
from PyQt5.QtGui import (QPainter, QPen, QBrush, QColor, QFont, 
                       QPainterPath, QIcon, QPixmap, QTransform,
                       QPainterPathStroker, QDrag)
from PyQt5.QtWidgets import (QWidget, QGraphicsView, QGraphicsScene, QGraphicsItem,
                           QGraphicsRectItem, QGraphicsTextItem, QGraphicsPathItem,
                           QGraphicsLineItem, QGraphicsEllipseItem, QGraphicsPixmapItem,
                           QVBoxLayout, QHBoxLayout, QPushButton, QToolBar,
                           QLabel, QComboBox, QMenu, QAction, QToolButton,
                           QMessageBox, QInputDialog, QGraphicsSceneDragDropEvent,
                           QGraphicsSceneMouseEvent, QGraphicsSceneContextMenuEvent,
                           QStyleOptionGraphicsItem, QDialog, QFormLayout, 
                           QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox)

from models.decision_tree import BespokeDecisionTree, TreeGrowthMode

logger = logging.getLogger(__name__)

NODE_TYPE_DATASET = "dataset"
NODE_TYPE_MODEL = "model"
NODE_TYPE_FILTER = "filter"
NODE_TYPE_TRANSFORM = "transform"
NODE_TYPE_EVALUATION = "evaluation"
NODE_TYPE_VISUALIZATION = "visualization"
NODE_TYPE_EXPORT = "export"

CONN_TYPE_DATA = "data"
CONN_TYPE_MODEL = "model"
CONN_TYPE_RESULT = "result"

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

class WorkflowNode(QGraphicsRectItem):
    """Base class for workflow nodes"""
    
    def __init__(self, node_id: str, node_type: str, title: str, parent=None):
        """
        Initialize a workflow node
        
        Args:
            node_id: Unique node identifier
            node_type: Type of node (dataset, model, etc.)
            title: Node title
            parent: Parent item
        """
        super().__init__(parent)
        
        self.node_id = node_id
        self.node_type = node_type
        self.title = title
        
        self.width = 180
        self.height = 120
        self.setRect(0, 0, self.width, self.height)
        
        self.default_color = self._get_default_color()
        self.setBrush(QBrush(self.default_color))
        self.setPen(QPen(Qt.black, 1.5))
        
        self.border_radius = 8  # Rounded corners
        self.shadow_enabled = True  # Subtle shadow
        self.icon = self._get_node_icon()  # Icon for node type
        
        self.is_selected = False
        self.is_processing = False
        self.status = STATUS_PENDING
        
        self.input_ports = []
        self.output_ports = []
        
        self.text_item = QGraphicsTextItem(self)
        self._update_text_content()
        self.text_item.setPos(10, 10)
        self.text_item.setTextWidth(self.width - 20)
        
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        
        self._create_ports()
    
    def _get_default_color(self) -> QColor:
        """
        Get the default node color based on its type
        
        Returns:
            Default node color
        """
        colors = {
            NODE_TYPE_DATASET: QColor(200, 230, 250),      # Light blue
            NODE_TYPE_MODEL: QColor(200, 250, 200),        # Light green
            NODE_TYPE_FILTER: QColor(250, 230, 200),       # Light orange
            NODE_TYPE_TRANSFORM: QColor(230, 200, 250),    # Light purple
            NODE_TYPE_EVALUATION: QColor(250, 200, 200),   # Light red
            NODE_TYPE_VISUALIZATION: QColor(240, 240, 190), # Light yellow
            NODE_TYPE_EXPORT: QColor(200, 200, 200)        # Light gray
        }
        
        return colors.get(self.node_type, QColor(240, 240, 240))
    
    def _get_node_icon(self) -> str:
        """Get icon text for node type (modern emoji icons)"""
        icons = {
            NODE_TYPE_DATASET: "📊",
            NODE_TYPE_MODEL: "🧠", 
            NODE_TYPE_FILTER: "🔍",
            NODE_TYPE_TRANSFORM: "⚡",
            NODE_TYPE_EVALUATION: "📈",
            NODE_TYPE_VISUALIZATION: "📋",
            NODE_TYPE_EXPORT: "📤"
        }
        return icons.get(self.node_type, "⚙️")
    
    def _get_status_indicator(self) -> str:
        """Get status indicator based on current status"""
        indicators = {
            STATUS_PENDING: "",
            STATUS_RUNNING: "🔄",
            STATUS_COMPLETED: "✅", 
            STATUS_FAILED: "❌"
        }
        return indicators.get(self.status, "")
    
    def _create_ports(self):
        """Create input and output ports based on node type"""
        if self.node_type == NODE_TYPE_DATASET:
            self.output_ports = [
                {'type': CONN_TYPE_DATA, 'pos': QPointF(self.width, self.height/2), 'name': 'Data Output'}
            ]
        
        elif self.node_type == NODE_TYPE_MODEL:
            self.input_ports = [
                {'type': CONN_TYPE_DATA, 'pos': QPointF(0, self.height/2), 'name': 'Data Input'}
            ]
            self.output_ports = [
                {'type': CONN_TYPE_MODEL, 'pos': QPointF(self.width, self.height/2), 'name': 'Model Output'}
            ]
        
        elif self.node_type == NODE_TYPE_FILTER:
            self.input_ports = [
                {'type': CONN_TYPE_DATA, 'pos': QPointF(0, self.height/2), 'name': 'Data Input'}
            ]
            self.output_ports = [
                {'type': CONN_TYPE_DATA, 'pos': QPointF(self.width, self.height/2), 'name': 'Filtered Data'}
            ]
        
        elif self.node_type == NODE_TYPE_TRANSFORM:
            self.input_ports = [
                {'type': CONN_TYPE_DATA, 'pos': QPointF(0, self.height/2), 'name': 'Data Input'}
            ]
            self.output_ports = [
                {'type': CONN_TYPE_DATA, 'pos': QPointF(self.width, self.height/2), 'name': 'Transformed Data'}
            ]
        
        elif self.node_type == NODE_TYPE_EVALUATION:
            self.input_ports = [
                {'type': CONN_TYPE_MODEL, 'pos': QPointF(0, self.height/3), 'name': 'Model Input'},
                {'type': CONN_TYPE_DATA, 'pos': QPointF(0, 2*self.height/3), 'name': 'Data Input'}
            ]
            self.output_ports = [
                {'type': CONN_TYPE_RESULT, 'pos': QPointF(self.width, self.height/2), 'name': 'Evaluation Results'}
            ]
        
        elif self.node_type == NODE_TYPE_VISUALIZATION:
            self.input_ports = [
                {'type': CONN_TYPE_MODEL, 'pos': QPointF(0, self.height/3), 'name': 'Model Input'},
                {'type': CONN_TYPE_RESULT, 'pos': QPointF(0, 2*self.height/3), 'name': 'Results Input'}
            ]
            self.output_ports = [
                {'type': CONN_TYPE_RESULT, 'pos': QPointF(self.width, self.height/2), 'name': 'Visualization Trigger'}
            ]
        
        elif self.node_type == NODE_TYPE_EXPORT:
            self.input_ports = [
                {'type': CONN_TYPE_MODEL, 'pos': QPointF(0, self.height/2), 'name': 'Model Input'}
            ]
        
        self._create_port_items()
    
    def _create_port_items(self):
        """Create visual items for the ports"""
        port_radius = 10
        
        for port in self.input_ports:
            port_item = QGraphicsEllipseItem(
                -port_radius, -port_radius, 2*port_radius, 2*port_radius, self
            )
            port_item.setPos(port['pos'])
            
            if port['type'] == CONN_TYPE_DATA:
                port_item.setBrush(QBrush(QColor(100, 130, 250)))  # Blue
            elif port['type'] == CONN_TYPE_MODEL:
                port_item.setBrush(QBrush(QColor(100, 250, 100)))  # Green
            elif port['type'] == CONN_TYPE_RESULT:
                port_item.setBrush(QBrush(QColor(250, 100, 100)))  # Red
            
            port_item.setPen(QPen(Qt.black, 2))
            port['item'] = port_item
            
            port_item.setZValue(10)
            
            port_item.setToolTip(port['name'])
        
        for port in self.output_ports:
            port_item = QGraphicsEllipseItem(
                -port_radius, -port_radius, 2*port_radius, 2*port_radius, self
            )
            port_item.setPos(port['pos'])
            
            if port['type'] == CONN_TYPE_DATA:
                port_item.setBrush(QBrush(QColor(100, 130, 250)))  # Blue
            elif port['type'] == CONN_TYPE_MODEL:
                port_item.setBrush(QBrush(QColor(100, 250, 100)))  # Green
            elif port['type'] == CONN_TYPE_RESULT:
                port_item.setBrush(QBrush(QColor(250, 100, 100)))  # Red
            
            port_item.setPen(QPen(Qt.black, 2))
            port['item'] = port_item
            
            port_item.setZValue(10)
            
            port_item.setToolTip(port['name'])
    
    def get_input_port_at(self, pos: QPointF) -> Optional[Dict[str, Any]]:
        """
        Get input port at a specific position
        
        Args:
            pos: Position in node coordinates
            
        Returns:
            Port dictionary or None if no port at position
        """
        for port in self.input_ports:
            port_pos = port['pos']
            
            tolerance = 15
            if (abs(pos.x() - port_pos.x()) < tolerance and 
                abs(pos.y() - port_pos.y()) < tolerance):
                return port
        
        return None
    
    def get_output_port_at(self, pos: QPointF) -> Optional[Dict[str, Any]]:
        """
        Get output port at a specific position
        
        Args:
            pos: Position in node coordinates
            
        Returns:
            Port dictionary or None if no port at position
        """
        for port in self.output_ports:
            port_pos = port['pos']
            
            tolerance = 15
            if (abs(pos.x() - port_pos.x()) < tolerance and 
                abs(pos.y() - port_pos.y()) < tolerance):
                return port
        
        return None
    
    def update_appearance(self):
        """Update node appearance based on state"""
        if self.is_selected:
            self.setBrush(QBrush(self.default_color.lighter(120)))
            self.setPen(QPen(QColor(0, 0, 150), 2.5))
        elif self.status == STATUS_RUNNING:
            self.setBrush(QBrush(QColor(250, 200, 150)))  # Running color (light orange)
            self.setPen(QPen(QColor(200, 100, 0), 2))
        elif self.status == STATUS_COMPLETED:
            self.setBrush(QBrush(QColor(200, 250, 200)))  # Completed color (light green)
            self.setPen(QPen(QColor(0, 150, 0), 2))
        elif self.status == STATUS_FAILED:
            self.setBrush(QBrush(QColor(250, 200, 200)))  # Failed color (light red)
            self.setPen(QPen(QColor(200, 0, 0), 2))
        else:  # STATUS_PENDING
            self.setBrush(QBrush(self.default_color))
            self.setPen(QPen(Qt.black, 1.5))
        
        self._update_text_content()
    
    def set_selected(self, selected: bool):
        """
        Set the selection state of the node
        
        Args:
            selected: Whether the node is selected
        """
        self.is_selected = selected
        self.update_appearance()
    
    def set_processing(self, processing: bool):
        """
        Set the processing state of the node
        
        Args:
            processing: Whether the node is processing
        """
        self.is_processing = processing
        self.update_appearance()
    
    def set_status(self, status: str):
        """
        Set the status of the node
        
        Args:
            status: Node status (pending, running, completed, failed)
        """
        self.status = status
        if status == STATUS_RUNNING:
            self.set_processing(True)
        else:
            self.set_processing(False)
        self.update_appearance()
    
    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value):
        """Handle item changes"""
        if change == QGraphicsItem.ItemSelectedChange:
            self.set_selected(bool(value))
        elif change == QGraphicsItem.ItemPositionHasChanged:
            if self.scene() and hasattr(self.scene(), 'itemMoved'):
                self.scene().itemMoved(self)
                if hasattr(self.scene(), '_update_connection_lines_for_node'):
                    self.scene()._update_connection_lines_for_node(self)
        
        return super().itemChange(change, value)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the node configuration
        
        Returns:
            Node configuration dictionary
        """
        return {
            'id': self.node_id,
            'type': self.node_type,
            'title': self.title,
            'pos': [self.pos().x(), self.pos().y()],
            'status': self.status
        }
    
    def set_config(self, config: Dict[str, Any]):
        """
        Set the node configuration
        
        Args:
            config: Node configuration dictionary
        """
        if 'pos' in config and len(config['pos']) == 2:
            self.setPos(QPointF(config['pos'][0], config['pos'][1]))
        
        if 'title' in config:
            self.title = config['title']
            self._update_text_content()
        
        if 'status' in config:
            self.set_status(config['status'])
    
    def _update_text_content(self):
        """Update text content with icon and status"""
        icon = self._get_node_icon()
        status_indicator = self._get_status_indicator()
        
        content_lines = [
            f"<b>{icon}  {self.title}</b>",
            "─" * 20,
        ]
        
        if status_indicator:
            content_lines.append(f"Status: {status_indicator}")
        
        if hasattr(self, 'dataset_name') and self.dataset_name:
            content_lines.append(f"Dataset: {self.dataset_name}")
        elif hasattr(self, 'model') and self.model and hasattr(self.model, 'model_name'):
            content_lines.append(f"Model: {self.model.model_name}")
        elif hasattr(self, 'model') and self.model:
            content_lines.append("Model: Configured")
        
        self.text_item.setHtml("<br>".join(content_lines))
    
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None):
        """Custom paint method for rounded corners and shadows"""
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        
        if self.shadow_enabled:
            shadow_rect = rect.translated(2, 2)
            shadow_path = QPainterPath()
            shadow_path.addRoundedRect(shadow_rect, self.border_radius, self.border_radius)
            painter.fillPath(shadow_path, QBrush(QColor(0, 0, 0, 30)))
        
        path = QPainterPath()
        path.addRoundedRect(rect, self.border_radius, self.border_radius)
        
        painter.fillPath(path, self.brush())
        
        painter.strokePath(path, self.pen())


class DatasetNode(WorkflowNode):
    """Node representing a dataset"""
    
    def __init__(self, node_id: str, title: str, dataset_name: str, parent=None):
        """
        Initialize a dataset node
        
        Args:
            node_id: Unique node identifier
            title: Node title
            dataset_name: Name of the dataset
            parent: Parent item
        """
        super().__init__(node_id, NODE_TYPE_DATASET, title, parent)
        
        self.dataset_name = dataset_name
        
        self.update_dataset_info()
    
    def update_dataset_info(self, rows: int = None, cols: int = None):
        """
        Update the dataset information displayed on the node
        
        Args:
            rows: Number of rows (optional)
            cols: Number of columns (optional)
        """
        html = f"<b>{self.title}</b><br>"
        html += f"Dataset: {self.dataset_name}<br>"
        
        if rows is not None and cols is not None:
            html += f"Rows: {rows}<br>"
            html += f"Columns: {cols}"
        
        self.text_item.setHtml(html)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the node configuration
        
        Returns:
            Node configuration dictionary
        """
        config = super().get_config()
        config['dataset_name'] = self.dataset_name
        return config
    
    def set_config(self, config: Dict[str, Any]):
        """
        Set the node configuration
        
        Args:
            config: Node configuration dictionary
        """
        super().set_config(config)
        
        if 'dataset_name' in config:
            self.dataset_name = config['dataset_name']
            self.update_dataset_info()


class ModelNode(WorkflowNode):
    """Node representing a model"""
    
    def __init__(self, node_id: str, title: str, model: Optional[Any] = None, parent=None):
        """
        Initialize a model node
        
        Args:
            node_id: Unique node identifier
            title: Node title
            model: Model object (optional)
            parent: Parent item
        """
        super().__init__(node_id, NODE_TYPE_MODEL, title, parent)
        
        self.model = model
        
        self.update_model_info()
    
    def _update_text_content(self):
        """Update text content with enhanced model information"""
        icon = self._get_node_icon()
        status_indicator = self._get_status_indicator()
        
        content_lines = [
            f"<b>{icon}  {self.title}</b>",
            "─" * 20,
        ]
        
        if status_indicator:
            content_lines.append(f"Status: {status_indicator}")
        
        try:
            if self.model is not None:
                if isinstance(self.model, BespokeDecisionTree):
                    content_lines.append("Model: Decision Tree")
                    
                    try:
                        if hasattr(self.model, 'is_fitted') and self.model.is_fitted:
                            num_nodes = getattr(self.model, 'num_nodes', 0)
                            max_depth = getattr(self.model, 'max_depth', 0)
                            content_lines.append(f"Nodes: {num_nodes}")
                            content_lines.append(f"Depth: {max_depth}")
                        else:
                            content_lines.append("Status: Not trained")
                    except Exception as e:
                        content_lines.append("Status: Model error")
                        logger.warning(f"Error accessing model properties: {e}")
                else:
                    content_lines.append("Model: Generic Model")
            else:
                content_lines.append("Model: Not assigned")
        except Exception as e:
            content_lines.append("Status: Display error")
            logger.error(f"Error updating model info: {e}")
        
        self.text_item.setHtml("<br>".join(content_lines))
    
    def update_model_info(self):
        """Update the model information displayed on the node"""
        self._update_text_content()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the node configuration
        
        Returns:
            Node configuration dictionary
        """
        config = super().get_config()
        
        if self.model is not None:
            config['has_model'] = True
            model_ref = getattr(self.model, 'model_name', None) or getattr(self.model, 'model_id', None) or self.title
            config['model_ref_id'] = model_ref
            config['model_id'] = getattr(self.model, 'model_id', None)
            config['target_variable'] = getattr(self.model, 'target_name', None)
            config['model_name'] = getattr(self.model, 'model_name', None)
            config['model_params'] = getattr(self.model, 'get_params', lambda: {})()
            logger.debug(f"Saving ModelNode {self.node_id} config with model_ref_id: {model_ref}")
        else:
            config['has_model'] = False
            config['model_ref_id'] = None
            config['target_variable'] = None
            config['model_name'] = None
            config['model_params'] = {}
        
        return config
    
    def set_config(self, config: Dict[str, Any]):
        """
        Set the node configuration
        
        Args:
            config: Node configuration dictionary
        """
        super().set_config(config)
        
        if config.get('has_model', False):
            if self.model is None:
                model_loaded = False
                
                model_ref_id = config.get('model_ref_id')
                if model_ref_id and hasattr(self, '_main_window_ref') and self._main_window_ref:
                    try:
                        if hasattr(self._main_window_ref, 'models') and self._main_window_ref.models:
                            if model_ref_id in self._main_window_ref.models:
                                self.model = self._main_window_ref.models[model_ref_id]
                                model_loaded = True
                                logger.info(f"Successfully connected ModelNode {self.node_id} to loaded model '{model_ref_id}' (exact match)")
                            else:
                                for model_key, model_obj in self._main_window_ref.models.items():
                                    model_name = getattr(model_obj, 'model_name', None)
                                    model_id = getattr(model_obj, 'model_id', None)
                                    if model_name == model_ref_id or model_id == model_ref_id:
                                        self.model = model_obj
                                        model_loaded = True
                                        logger.info(f"Successfully connected ModelNode {self.node_id} to loaded model '{model_ref_id}' via fuzzy match (key: {model_key})")
                                        break
                                
                                if not model_loaded:
                                    logger.warning(f"Model '{model_ref_id}' not found in loaded models {list(self._main_window_ref.models.keys())} for ModelNode {self.node_id}")
                        else:
                            logger.warning(f"No models available in main window for ModelNode {self.node_id}")
                    except Exception as e:
                        logger.error(f"Error looking up loaded model '{model_ref_id}': {e}")
                
                if not model_loaded:
                    is_project_loading = (
                        hasattr(self, '_main_window_ref') and 
                        self._main_window_ref and 
                        hasattr(self._main_window_ref, 'models') and 
                        len(self._main_window_ref.models) == 0  # Models not loaded yet
                    )
                    
                    if is_project_loading:
                        logger.info(f"Deferring model creation for ModelNode {self.node_id} - waiting for project models to load")
                        if not hasattr(self, '_pending_config'):
                            self._pending_config = {}
                        self._pending_config.update({
                            'target_variable': config.get('target_variable'),
                            'model_name': config.get('model_name'),
                            'model_params': config.get('model_params', {}),
                            'model_ref_id': config.get('model_ref_id')
                        })
                        logger.info(f"Stored config for later restoration in ModelNode {self.node_id}")
                        self.model = None  # Will be set by main window association
                    else:
                        try:
                            model_config = {}
                            
                            if hasattr(self, '_main_window_ref') and self._main_window_ref:
                                if hasattr(self._main_window_ref, 'config'):
                                    model_config = self._main_window_ref.config
                            else:
                                parent_widget = self.parent()
                                while parent_widget:
                                    if hasattr(parent_widget, 'config'):
                                        model_config = parent_widget.config
                                        break
                                    parent_widget = parent_widget.parent()
                            
                            if not model_config:
                                model_config = {
                                    'models': {
                                        'decision_tree': {
                                            'criterion': 'gini',
                                            'max_depth': None,
                                            'min_samples_split': 2,
                                            'min_samples_leaf': 1,
                                            'random_state': 42
                                        }
                                    }
                                }
                                logger.debug("Using default model config during restoration")
                            
                            self.model = BespokeDecisionTree(model_config)
                            logger.info(f"Created NEW model instance for ModelNode {self.node_id} during config restoration")
                        except Exception as e:
                            logger.error(f"Failed to create model instance during config restoration: {e}")
                            try:
                                self.model = BespokeDecisionTree({})
                                logger.info(f"Created model with empty config as fallback for ModelNode {self.node_id}")
                            except Exception as e2:
                                logger.error(f"Even fallback model creation failed: {e2}")
                                return
            
            if self.model is not None:
                if 'target_variable' in config and config['target_variable']:
                    self.model.target_name = config['target_variable']
                    logger.info(f"Restored target variable '{config['target_variable']}' for ModelNode {self.node_id}")
                
                if 'model_name' in config and config['model_name']:
                    self.model.model_name = config['model_name']
                
                if 'model_params' in config and config['model_params']:
                    try:
                        self.model.set_params(**config['model_params'])
                        logger.debug(f"Restored model parameters for ModelNode {self.node_id}")
                    except Exception as e:
                        logger.warning(f"Failed to restore model parameters for ModelNode {self.node_id}: {e}")
            else:
                self._pending_config = {
                    'target_variable': config.get('target_variable'),
                    'model_name': config.get('model_name'),
                    'model_params': config.get('model_params')
                }
                logger.info(f"Stored config for later restoration in ModelNode {self.node_id}")
        
        self.update_model_info()


class FilterNode(WorkflowNode):
    """Node representing a data filter operation"""
    
    def __init__(self, node_id: str, title: str, parent=None):
        """
        Initialize a filter node
        
        Args:
            node_id: Unique node identifier
            title: Node title
            parent: Parent item
        """
        super().__init__(node_id, NODE_TYPE_FILTER, title, parent)
        
        self.conditions = []
        
        self.original_row_count = 0
        self.filtered_row_count = 0
        self.last_applied = None
        
        self.filtered_dataframe = None
        
        self.update_filter_info()
    
    def add_condition(self, column: str, operator: str, value: Any, logic: str = 'AND'):
        """
        Add a filter condition
        
        Args:
            column: Column name to filter on
            operator: Filter operator (>, <, =, !=, contains, etc.)
            value: Value to compare against
            logic: Logic operator to combine with previous condition (AND/OR)
        """
        condition = {
            'column': column,
            'operator': operator, 
            'value': value,
            'logic': logic if self.conditions else None  # First condition has no logic
        }
        self.conditions.append(condition)
        self.update_filter_info()
        logger.info(f"Added filter condition: {column} {operator} {value}")
    
    def remove_condition(self, index: int):
        """Remove a filter condition by index"""
        if 0 <= index < len(self.conditions):
            removed = self.conditions.pop(index)
            self.update_filter_info()
            logger.info(f"Removed filter condition: {removed}")
    
    def clear_conditions(self):
        """Clear all filter conditions"""
        self.conditions.clear()
        self.update_filter_info()
        logger.info("Cleared all filter conditions")
    
    def apply_filter(self, dataframe):
        """
        Apply filter conditions to a dataframe
        
        Args:
            dataframe: Input pandas DataFrame
            
        Returns:
            Filtered pandas DataFrame
        """
        if not self.conditions or dataframe is None:
            return dataframe
        
        try:
            import pandas as pd
            
            self.original_row_count = len(dataframe)
            result_df = dataframe.copy()
            
            mask = pd.Series([True] * len(result_df), index=result_df.index)
            
            for i, condition in enumerate(self.conditions):
                column = condition['column']
                operator = condition['operator']
                value = condition['value']
                logic = condition.get('logic', 'AND')
                
                if column not in result_df.columns:
                    logger.warning(f"Filter column '{column}' not found in dataframe")
                    continue
                
                col_data = result_df[column]
                
                if operator == '>':
                    condition_mask = col_data > value
                elif operator == '<':
                    condition_mask = col_data < value
                elif operator == '>=':
                    condition_mask = col_data >= value
                elif operator == '<=':
                    condition_mask = col_data <= value
                elif operator == '==':
                    condition_mask = col_data == value
                elif operator == '!=':
                    condition_mask = col_data != value
                elif operator == 'contains':
                    condition_mask = col_data.astype(str).str.contains(str(value), na=False)
                elif operator == 'starts_with':
                    condition_mask = col_data.astype(str).str.startswith(str(value), na=False)
                elif operator == 'ends_with':
                    condition_mask = col_data.astype(str).str.endswith(str(value), na=False)
                elif operator == 'is_null':
                    condition_mask = col_data.isnull()
                elif operator == 'not_null':
                    condition_mask = col_data.notnull()
                else:
                    logger.warning(f"Unknown filter operator: {operator}")
                    continue
                
                if i == 0:
                    mask = condition_mask
                elif logic == 'AND':
                    mask = mask & condition_mask
                elif logic == 'OR':
                    mask = mask | condition_mask
            
            result_df = result_df[mask]
            self.filtered_row_count = len(result_df)
            self.filtered_dataframe = result_df
            self.last_applied = pd.Timestamp.now()
            
            logger.info(f"Filter applied: {self.original_row_count} -> {self.filtered_row_count} rows")
            self.update_filter_info()
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error applying filter: {e}")
            return dataframe
    
    def update_filter_info(self):
        """Update the filter information displayed on the node"""
        icon = self._get_node_icon()
        html = f"<b>{icon} {self.title}</b><br>"
        html += "─" * 15 + "<br>"
        
        if self.conditions:
            html += f"Conditions: {len(self.conditions)}<br>"
            first_condition = self.conditions[0]
            html += f"• {first_condition['column']} {first_condition['operator']} {first_condition['value']}"
            if len(self.conditions) > 1:
                html += f"<br>  + {len(self.conditions)-1} more..."
        else:
            html += "No conditions set"
        
        if self.filtered_row_count > 0:
            html += f"<br>Result: {self.filtered_row_count:,} rows"
            if self.original_row_count > 0:
                pct = (self.filtered_row_count / self.original_row_count) * 100
                html += f" ({pct:.1f}%)"
        
        self.text_item.setHtml(html)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the node configuration
        
        Returns:
            Node configuration dictionary
        """
        config = super().get_config()
        config['conditions'] = self.conditions.copy()
        config['original_row_count'] = self.original_row_count
        config['filtered_row_count'] = self.filtered_row_count
        return config
    
    def set_config(self, config: Dict[str, Any]):
        """
        Set the node configuration
        
        Args:
            config: Node configuration dictionary
        """
        super().set_config(config)
        
        if 'conditions' in config:
            self.conditions = config['conditions'].copy()
        
        if 'original_row_count' in config:
            self.original_row_count = config['original_row_count']
            
        if 'filtered_row_count' in config:
            self.filtered_row_count = config['filtered_row_count']
            
        self.update_filter_info()


class TransformNode(WorkflowNode):
    """Node representing a data transformation operation"""
    
    def __init__(self, node_id: str, title: str, parent=None):
        """
        Initialize a transform node
        
        Args:
            node_id: Unique node identifier
            title: Node title
            parent: Parent item
        """
        super().__init__(node_id, NODE_TYPE_TRANSFORM, title, parent)
        
        self.transformations = []
        
        self.original_column_count = 0
        self.transformed_column_count = 0
        self.last_applied = None
        
        self.transformed_dataframe = None
        
        self.update_transform_info()
    
    def add_transformation(self, transform_type: str, target_column: str, 
                          source_columns: List[str] = None, formula: str = None, 
                          parameters: Dict[str, Any] = None):
        """
        Add a transformation
        
        Args:
            transform_type: Type of transformation (create_variable, derive, encode, etc.)
            target_column: Name of the column to create/modify
            source_columns: List of source columns for the transformation
            formula: Formula string for calculations
            parameters: Additional parameters for the transformation
        """
        transformation = {
            'type': transform_type,
            'target_column': target_column,
            'source_columns': source_columns or [],
            'formula': formula or '',
            'parameters': parameters or {}
        }
        self.transformations.append(transformation)
        self.update_transform_info()
        logger.info(f"Added transformation: {transform_type} -> {target_column}")
    
    def remove_transformation(self, index: int):
        """Remove a transformation by index"""
        if 0 <= index < len(self.transformations):
            removed = self.transformations.pop(index)
            self.update_transform_info()
            logger.info(f"Removed transformation: {removed}")
    
    def clear_transformations(self):
        """Clear all transformations"""
        self.transformations.clear()
        self.update_transform_info()
        logger.info("Cleared all transformations")
    
    def apply_transformations(self, dataframe):
        """
        Apply transformations to a dataframe
        
        Args:
            dataframe: Input pandas DataFrame
            
        Returns:
            Transformed pandas DataFrame
        """
        if not self.transformations or dataframe is None:
            return dataframe
        
        try:
            import pandas as pd
            import numpy as np
            
            self.original_column_count = len(dataframe.columns)
            result_df = dataframe.copy()
            
            for transformation in self.transformations:
                transform_type = transformation['type']
                target_column = transformation['target_column']
                source_columns = transformation['source_columns']
                formula = transformation['formula']
                parameters = transformation['parameters']
                
                try:
                    if transform_type == 'create_variable':
                        if formula:
                            context = {
                                'df': result_df,
                                'np': np,
                                'pd': pd,
                                'abs': abs,
                                'max': max,
                                'min': min,
                                'round': round,
                                'len': len
                            }
                            for col in result_df.columns:
                                if col.isidentifier():  # Valid Python identifier
                                    context[col] = result_df[col]
                            
                            result_df[target_column] = eval(formula, {"__builtins__": {}}, context)
                            logger.info(f"Created variable '{target_column}' using formula: {formula}")
                    
                    elif transform_type == 'derive_ratio':
                        if len(source_columns) >= 2:
                            col1, col2 = source_columns[0], source_columns[1]
                            result_df[target_column] = result_df[col1] / result_df[col2].replace(0, np.nan)
                            logger.info(f"Created ratio '{target_column}' = {col1} / {col2}")
                    
                    elif transform_type == 'derive_difference':
                        if len(source_columns) >= 2:
                            col1, col2 = source_columns[0], source_columns[1]
                            result_df[target_column] = result_df[col1] - result_df[col2]
                            logger.info(f"Created difference '{target_column}' = {col1} - {col2}")
                    
                    elif transform_type == 'derive_sum':
                        if source_columns:
                            result_df[target_column] = result_df[source_columns].sum(axis=1)
                            logger.info(f"Created sum '{target_column}' from {len(source_columns)} columns")
                    
                    elif transform_type == 'encode_categorical':
                        if source_columns:
                            source_col = source_columns[0]
                            encoded = pd.get_dummies(result_df[source_col], prefix=target_column)
                            result_df = pd.concat([result_df, encoded], axis=1)
                            logger.info(f"One-hot encoded '{source_col}' with prefix '{target_column}'")
                    
                    elif transform_type == 'binning':
                        if source_columns:
                            source_col = source_columns[0]
                            bins = parameters.get('bins', 5)
                            labels = parameters.get('labels', None)
                            result_df[target_column] = pd.cut(result_df[source_col], bins=bins, labels=labels)
                            logger.info(f"Binned '{source_col}' into '{target_column}' with {bins} bins")
                    
                    elif transform_type == 'standardize':
                        if source_columns:
                            for col in source_columns:
                                new_col = f"{col}_standardized" if not target_column else f"{target_column}_{col}"
                                mean_val = result_df[col].mean()
                                std_val = result_df[col].std()
                                result_df[new_col] = (result_df[col] - mean_val) / std_val
                                logger.info(f"Standardized '{col}' to '{new_col}'")
                    
                    elif transform_type == 'log_transform':
                        if source_columns:
                            source_col = source_columns[0]
                            result_df[target_column] = np.log1p(result_df[source_col].clip(lower=0))
                            logger.info(f"Log transformed '{source_col}' to '{target_column}'")
                    
                    else:
                        logger.warning(f"Unknown transformation type: {transform_type}")
                        
                except Exception as e:
                    logger.error(f"Error applying transformation {transform_type}: {e}")
                    continue
            
            self.transformed_column_count = len(result_df.columns)
            self.transformed_dataframe = result_df
            self.last_applied = pd.Timestamp.now()
            
            logger.info(f"Transformations applied: {self.original_column_count} -> {self.transformed_column_count} columns")
            self.update_transform_info()
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error applying transformations: {e}")
            return dataframe
    
    def update_transform_info(self):
        """Update the transformation information displayed on the node"""
        icon = self._get_node_icon()
        html = f"<b>{icon} {self.title}</b><br>"
        html += "─" * 15 + "<br>"
        
        if self.transformations:
            html += f"Transforms: {len(self.transformations)}<br>"
            first_transform = self.transformations[0]
            html += f"• {first_transform['type']}<br>"
            html += f"  → {first_transform['target_column']}"
            if len(self.transformations) > 1:
                html += f"<br>  + {len(self.transformations)-1} more..."
        else:
            html += "No transformations set"
        
        if self.transformed_column_count > 0:
            added_cols = self.transformed_column_count - self.original_column_count
            if added_cols > 0:
                html += f"<br>Added: {added_cols} columns"
            else:
                html += f"<br>Columns: {self.transformed_column_count}"
        
        self.text_item.setHtml(html)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the node configuration
        
        Returns:
            Node configuration dictionary
        """
        config = super().get_config()
        config['transformations'] = self.transformations.copy()
        config['original_column_count'] = self.original_column_count
        config['transformed_column_count'] = self.transformed_column_count
        return config
    
    def set_config(self, config: Dict[str, Any]):
        """
        Set the node configuration
        
        Args:
            config: Node configuration dictionary
        """
        super().set_config(config)
        
        if 'transformations' in config:
            self.transformations = config['transformations'].copy()
        
        if 'original_column_count' in config:
            self.original_column_count = config['original_column_count']
            
        if 'transformed_column_count' in config:
            self.transformed_column_count = config['transformed_column_count']
            
        self.update_transform_info()


class EvaluationNode(WorkflowNode):
    """Node representing a model evaluation operation"""
    
    def __init__(self, node_id: str, title: str, parent=None):
        """
        Initialize an evaluation node
        
        Args:
            node_id: Unique node identifier
            title: Node title
            parent: Parent item
        """
        super().__init__(node_id, NODE_TYPE_EVALUATION, title, parent)
        
        self.metrics = []
        self.cross_validation = False
        self.n_folds = 5
        
        self.update_evaluation_info()
    
    def update_evaluation_info(self):
        """Update the evaluation information displayed on the node"""
        html = f"<b>{self.title}</b><br>"
        
        if self.metrics:
            html += f"Metrics: {', '.join(self.metrics)}<br>"
        else:
            html += "No metrics selected<br>"
        
        if self.cross_validation:
            html += f"CV: {self.n_folds} folds"
        
        self.text_item.setHtml(html)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the node configuration
        
        Returns:
            Node configuration dictionary
        """
        config = super().get_config()
        config['metrics'] = self.metrics.copy()
        config['cross_validation'] = self.cross_validation
        config['n_folds'] = self.n_folds
        return config
    
    def set_config(self, config: Dict[str, Any]):
        """
        Set the node configuration
        
        Args:
            config: Node configuration dictionary
        """
        super().set_config(config)
        
        if 'metrics' in config:
            self.metrics = config['metrics'].copy()
        
        if 'cross_validation' in config:
            self.cross_validation = config['cross_validation']
        
        if 'n_folds' in config:
            self.n_folds = config['n_folds']
        
        self.update_evaluation_info()


class VisualizationNode(WorkflowNode):
    """Node representing a visualization operation"""
    
    def __init__(self, node_id: str, title: str, visualization_type: str = "tree", parent=None):
        """
        Initialize a visualization node
        
        Args:
            node_id: Unique node identifier
            title: Node title
            visualization_type: Type of visualization (tree, roc, importance, etc.)
            parent: Parent item
        """
        super().__init__(node_id, NODE_TYPE_VISUALIZATION, title, parent)
        
        self.visualization_type = visualization_type
        self.viz_config = {}
        
        self.update_visualization_info()
    
    def update_visualization_info(self):
        """Update the visualization information displayed on the node"""
        html = f"<b>{self.title}</b><br>"
        html += f"Type: {self.visualization_type.capitalize()}"
        
        self.text_item.setHtml(html)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the node configuration
        
        Returns:
            Node configuration dictionary
        """
        config = super().get_config()
        config['visualization_type'] = self.visualization_type
        config['viz_config'] = self.viz_config.copy()
        return config
    
    def set_config(self, config: Dict[str, Any]):
        """
        Set the node configuration
        
        Args:
            config: Node configuration dictionary
        """
        super().set_config(config)
        
        if 'visualization_type' in config:
            self.visualization_type = config['visualization_type']
        
        if 'viz_config' in config:
            self.viz_config = config['viz_config'].copy()
        
        self.update_visualization_info()


class ExportNode(WorkflowNode):
    """Node representing an export operation"""
    
    def __init__(self, node_id: str, title: str, export_type: str = "python", parent=None):
        """
        Initialize an export node
        
        Args:
            node_id: Unique node identifier
            title: Node title
            export_type: Type of export (python, pmml, json)
            parent: Parent item
        """
        super().__init__(node_id, NODE_TYPE_EXPORT, title, parent)
        
        self.export_type = export_type
        self.export_path = ""
        
        self.update_export_info()
    
    def update_export_info(self):
        """Update the export information displayed on the node"""
        html = f"<b>{self.title}</b><br>"
        html += f"Type: {self.export_type.capitalize()}"
        
        if self.export_path:
            html += f"<br>Path: ...{self.export_path[-20:]}"
        
        self.text_item.setHtml(html)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the node configuration
        
        Returns:
            Node configuration dictionary
        """
        config = super().get_config()
        config['export_type'] = self.export_type
        config['export_path'] = self.export_path
        return config
    
    def set_config(self, config: Dict[str, Any]):
        """
        Set the node configuration
        
        Args:
            config: Node configuration dictionary
        """
        super().set_config(config)
        
        if 'export_type' in config:
            self.export_type = config['export_type']
        
        if 'export_path' in config:
            self.export_path = config['export_path']
        
        self.update_export_info()


class ConnectionData:
    """Simple data-only connection (no graphics to prevent crashes)"""
    
    def __init__(self, source_node_id: str, source_port: Dict[str, Any],
                target_node_id: str, target_port: Dict[str, Any]):
        """
        Initialize a connection data object
        
        Args:
            source_node_id: Source node ID
            source_port: Source port dict
            target_node_id: Target node ID
            target_port: Target port dict
        """
        self.connection_id = str(uuid.uuid4())
        self.source_node_id = source_node_id
        self.source_port = source_port
        self.target_node_id = target_node_id
        self.target_port = target_port
        self.connection_type = source_port.get('type', CONN_TYPE_DATA)
        
        logger.info(f"Created data connection: {source_node_id} -> {target_node_id}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get connection configuration"""
        return {
            'id': self.connection_id,
            'source_node_id': self.source_node_id,
            'source_port': self.source_port,
            'target_node_id': self.target_node_id,
            'target_port': self.target_port,
            'type': self.connection_type
        }
    
    def __str__(self) -> str:
        """String representation"""
        return f"Connection({self.source_node_id} -> {self.target_node_id}, type={self.connection_type})"
    


class WorkflowScene(QGraphicsScene):
    """Graphics scene for the workflow canvas"""
    
    nodeSelectionChanged = pyqtSignal(object)  # Selected node (or None)
    nodeDoubleClicked = pyqtSignal(WorkflowNode)     # Double-clicked node
    nodeContextMenu = pyqtSignal(WorkflowNode, QPointF)  # Node and scene position
    connectionCreated = pyqtSignal(object)   # Created connection (data only)
    connectionDeleted = pyqtSignal(object)   # Deleted connection (data only)
    sceneModified = pyqtSignal()                     # Any modification
    
    def __init__(self, parent=None, main_window_ref=None):
        """
        Initialize the workflow scene
        
        Args:
            parent: Parent object
            main_window_ref: Reference to main window for accessing loaded models
        """
        super().__init__(parent)
        
        self.main_window_ref = main_window_ref
        
        self.nodes = {}
        self.connections = {}
        
        self.temp_connection = None
        self.connection_source_node = None
        self.connection_source_port = None
        
        self.setBackgroundBrush(QBrush(QColor(240, 240, 240)))
    
    def add_node(self, node: WorkflowNode) -> str:
        """
        Add a node to the scene
        
        Args:
            node: Node to add
            
        Returns:
            Node ID
        """
        self.addItem(node)
        self.nodes[node.node_id] = node
        
        self.sceneModified.emit()
        
        return node.node_id
    
    def remove_node(self, node_id: str):
        """
        Remove a node from the scene
        
        Args:
            node_id: ID of the node to remove
        """
        if node_id in self.nodes:
            node = self.nodes[node_id]
            
            self.remove_node_connections(node)
            
            self.removeItem(node)
            del self.nodes[node_id]
            
            self.sceneModified.emit()
    
    def remove_node_connections(self, node: WorkflowNode):
        """
        Remove all connections to/from a node
        
        Args:
            node: Node to remove connections for
        """
        connections_to_remove = []
        
        for conn_id, conn in self.connections.items():
            if conn.source_node_id == node.node_id or conn.target_node_id == node.node_id:
                connections_to_remove.append(conn_id)
        
        for conn_id in connections_to_remove:
            self.remove_connection(conn_id)
    
    def add_connection(self, source_node: WorkflowNode, source_port: Dict[str, Any],
                     target_node: WorkflowNode, target_port: Dict[str, Any]) -> str:
        """
        Add a connection between nodes
        
        Args:
            source_node: Source node
            source_port: Source port
            target_node: Target node
            target_port: Target port
            
        Returns:
            Connection ID
        """
        if source_port['type'] != target_port['type']:
            logger.warning(f"Cannot connect incompatible ports: {source_port['type']} to {target_port['type']}")
            return None
        
        for conn in self.connections.values():
            if (hasattr(conn, 'target_node_id') and 
                conn.target_node_id == target_node.node_id and 
                conn.target_port == target_port):
                logger.warning(f"Target port already has a connection")
                return None
        
        try:
            connection = ConnectionData(
                source_node_id=source_node.node_id,
                source_port=source_port,
                target_node_id=target_node.node_id,
                target_port=target_port
            )
            
            self.connections[connection.connection_id] = connection
            
            self._create_connection_line(connection, source_node, target_node, source_port, target_port)
            
            source_node.setBrush(QBrush(QColor(120, 200, 120)))  # Light green
            target_node.setBrush(QBrush(QColor(120, 200, 120)))   # Light green
            
            logger.info(f"Created data connection: {source_node.title} -> {target_node.title}")
            
            self.connectionCreated.emit(connection)
            
            self.sceneModified.emit()
            
            return connection.connection_id
            
        except Exception as e:
            logger.error(f"Error creating connection: {e}")
            return None
    
    def _create_connection_line(self, connection: 'ConnectionData', source_node: WorkflowNode, 
                              target_node: WorkflowNode, source_port: Dict[str, Any], target_port: Dict[str, Any]):
        """
        Create a simple visual line for the connection
        
        Args:
            connection: Connection data object
            source_node: Source workflow node
            target_node: Target workflow node
            source_port: Source port dictionary
            target_port: Target port dictionary
        """
        try:
            source_pos = source_node.pos() + source_port['pos']
            target_pos = target_node.pos() + target_port['pos']
            
            line_item = QGraphicsLineItem(source_pos.x(), source_pos.y(), target_pos.x(), target_pos.y())
            
            if source_port['type'] == CONN_TYPE_DATA:
                line_item.setPen(QPen(QColor(100, 130, 250), 3))  # Blue line for data
            elif source_port['type'] == CONN_TYPE_MODEL:
                line_item.setPen(QPen(QColor(100, 250, 100), 3))  # Green line for model
            else:
                line_item.setPen(QPen(QColor(250, 100, 100), 3))  # Red line for results
            
            line_item.setZValue(-1)
            
            self.addItem(line_item)
            
            connection.line_item = line_item
            
            logger.info(f"Created visual connection line")
            
        except Exception as e:
            logger.warning(f"Could not create visual connection line: {e}")
    
    def remove_connection(self, connection_id: str):
        """
        Remove a connection (data only, no graphics)
        
        Args:
            connection_id: ID of the connection to remove
        """
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            
            self.connectionDeleted.emit(connection)
            
            if hasattr(connection, 'line_item') and connection.line_item:
                try:
                    self.removeItem(connection.line_item)
                except Exception as e:
                    logger.warning(f"Error removing connection line: {e}")
            
            del self.connections[connection_id]
            
            if hasattr(connection, 'source_node_id') and connection.source_node_id in self.nodes:
                source_node = self.nodes[connection.source_node_id]
                source_node.setBrush(QBrush(source_node.default_color))
            
            if hasattr(connection, 'target_node_id') and connection.target_node_id in self.nodes:
                target_node = self.nodes[connection.target_node_id]
                target_node.setBrush(QBrush(target_node.default_color))
            
            logger.info(f"Removed connection {connection_id}")
            
            self.sceneModified.emit()
    
    def itemMoved(self, item: QGraphicsItem):
        """
        Handle movement of an item and update connection lines
        
        Args:
            item: Moved item
        """
        if isinstance(item, WorkflowNode):
            logger.debug(f"Node {item.node_id} moved to position {item.pos()}")
            
            self._update_connection_lines_for_node(item)
        
        self.sceneModified.emit()
    
    def _update_connection_lines_for_node(self, node: WorkflowNode):
        """
        Update all connection lines connected to a specific node
        
        Args:
            node: The node that was moved
        """
        try:
            for connection in self.connections.values():
                if (hasattr(connection, 'line_item') and connection.line_item and 
                    (connection.source_node_id == node.node_id or connection.target_node_id == node.node_id)):
                    
                    source_node = self.nodes.get(connection.source_node_id)
                    target_node = self.nodes.get(connection.target_node_id)
                    
                    if source_node and target_node:
                        source_pos = source_node.pos() + connection.source_port['pos']
                        target_pos = target_node.pos() + connection.target_port['pos']
                        
                        line = connection.line_item.line()
                        connection.line_item.setLine(source_pos.x(), source_pos.y(), target_pos.x(), target_pos.y())
                        
        except Exception as e:
            logger.warning(f"Error updating connection lines: {e}")
    
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        """
        Handle mouse press events
        
        Args:
            event: Mouse event
        """
        try:
            item = self.itemAt(event.scenePos(), QTransform())
            
            if event.button() == Qt.LeftButton and item is not None:
                port_item = None
                
                if isinstance(item, QGraphicsEllipseItem):
                    parent_item = item.parentItem()
                    
                    if isinstance(parent_item, WorkflowNode):
                        for port in parent_item.output_ports:
                            if port.get('item') == item:
                                port_item = port
                                self.connection_source_node = parent_item
                                self.connection_source_port = port
                                logger.info(f"Starting connection from {port['name']} on {parent_item.node_id}")
                                break
                
                if port_item is not None:
                    self.start_connection(event.scenePos())
                    return
            
            super().mousePressEvent(event)
            
            selected_items = self.selectedItems()
            selected_node = None
            
            for item in selected_items:
                if isinstance(item, WorkflowNode):
                    selected_node = item
                    break
            
            self.nodeSelectionChanged.emit(selected_node)
            
        except Exception as e:
            logger.error(f"Error in mousePressEvent: {e}", exc_info=True)
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        """
        Handle mouse move events
        
        Args:
            event: Mouse event
        """
        try:
            
            if self.temp_connection and self.views():
                self.views()[0].setCursor(Qt.CrossCursor)
            
            super().mouseMoveEvent(event)
            
        except Exception as e:
            logger.error(f"Error in mouseMoveEvent: {e}", exc_info=True)
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        """
        Handle mouse release events
        
        Args:
            event: Mouse event
        """
        try:
            if self.temp_connection and event.button() == Qt.LeftButton:
                self.finish_connection(event.scenePos())
                return
            
            super().mouseReleaseEvent(event)
            
        except Exception as e:
            logger.error(f"Error in mouseReleaseEvent: {e}", exc_info=True)
            self.temp_connection = False
            self.connection_source_node = None
            self.connection_source_port = None
            super().mouseReleaseEvent(event)
    
    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent):
        """
        Handle mouse double-click events
        
        Args:
            event: Mouse event
        """
        try:
            item = self.itemAt(event.scenePos(), QTransform())
            
            if item is not None and isinstance(item, WorkflowNode):
                self.nodeDoubleClicked.emit(item)
                return
            elif item is not None:
                parent_item = item.parentItem()
                if parent_item and isinstance(parent_item, WorkflowNode):
                    self.nodeDoubleClicked.emit(parent_item)
                    return
            
            super().mouseDoubleClickEvent(event)
            
        except Exception as e:
            logger.error(f"Error in mouseDoubleClickEvent: {e}", exc_info=True)
            super().mouseDoubleClickEvent(event)
    
    def contextMenuEvent(self, event: QGraphicsSceneContextMenuEvent):
        """
        Handle context menu events
        
        Args:
            event: Context menu event
        """
        item = self.itemAt(event.scenePos(), QTransform())
        
        target_node = None
        if isinstance(item, WorkflowNode):
            target_node = item
        elif item and hasattr(item, 'parentItem') and item.parentItem():
            parent = item.parentItem()
            if isinstance(parent, WorkflowNode):
                target_node = parent
        
        if target_node:
            self.nodeContextMenu.emit(target_node, event.scenePos())
            return
        
        super().contextMenuEvent(event)
    
    def start_connection(self, pos: QPointF):
        """
        Start creating a connection (simplified - no dynamic graphics)
        
        Args:
            pos: Starting position
        """
        try:
            self.temp_connection = True  # Simple flag instead of graphics item
            
            if self.views():
                self.views()[0].setCursor(Qt.CrossCursor)
            
            if self.connection_source_port:
                logger.info(f"Starting connection from {self.connection_source_port['type']} port")
            else:
                logger.warning("Starting connection without valid source port")
                
        except Exception as e:
            logger.error(f"Error starting connection: {e}", exc_info=True)
            self.temp_connection = False
            self.connection_source_node = None
            self.connection_source_port = None
    
    def finish_connection(self, pos: QPointF):
        """
        Finish creating a connection (simplified - no dynamic graphics)
        
        Args:
            pos: Ending position
        """
        try:
            if self.temp_connection:
                self.temp_connection = False
                
                if self.views():
                    self.views()[0].setCursor(Qt.ArrowCursor)
            
            item = self.itemAt(pos, QTransform())
            
            if item is not None and isinstance(item, QGraphicsEllipseItem):
                parent_item = item.parentItem()
                
                if isinstance(parent_item, WorkflowNode):
                    for port in parent_item.input_ports:
                        if port.get('item') == item:
                            target_node = parent_item
                            target_port = port
                            
                            if (self.connection_source_node is not None and 
                                self.connection_source_port is not None):
                                logger.info(f"Creating connection: {self.connection_source_node.node_id} -> {target_node.node_id}")
                                result = self.add_connection(
                                    self.connection_source_node, 
                                    self.connection_source_port,
                                    target_node, 
                                    target_port
                                )
                                if result:
                                    logger.info(f"Successfully created connection")
                                else:
                                    logger.warning("Failed to create connection")
                            else:
                                logger.warning("Cannot create connection: missing source node or port")
                            break
            
        except Exception as e:
            logger.error(f"Error finishing connection: {e}", exc_info=True)
        finally:
            self.temp_connection = False
            self.connection_source_node = None
            self.connection_source_port = None
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the entire workflow
        
        Returns:
            Workflow configuration dictionary
        """
        config = {
            'nodes': {},
            'connections': {}
        }
        
        for node_id, node in self.nodes.items():
            config['nodes'][node_id] = node.get_config()
        
        for conn_id, conn in self.connections.items():
            config['connections'][conn_id] = conn.get_config()
        
        return config
    
    def set_config(self, config: Dict[str, Any]):
        """
        Set the configuration of the entire workflow
        
        Args:
            config: Workflow configuration dictionary
        """
        self.clear()
        self.nodes = {}
        self.connections = {}
        
        for node_id, node_config in config.get('nodes', {}).items():
            node_type = node_config.get('type')
            title = node_config.get('title', 'Node')
            
            if node_type == NODE_TYPE_DATASET:
                dataset_name = node_config.get('dataset_name', '')
                node = DatasetNode(node_id, title, dataset_name)
            elif node_type == NODE_TYPE_MODEL:
                node = ModelNode(node_id, title)
                node._main_window_ref = self.main_window_ref
            elif node_type == NODE_TYPE_FILTER:
                node = FilterNode(node_id, title)
            elif node_type == NODE_TYPE_TRANSFORM:
                node = TransformNode(node_id, title)
            elif node_type == NODE_TYPE_EVALUATION:
                node = EvaluationNode(node_id, title)
            elif node_type == NODE_TYPE_VISUALIZATION:
                viz_type = node_config.get('visualization_type', 'tree')
                node = VisualizationNode(node_id, title, viz_type)
            elif node_type == NODE_TYPE_EXPORT:
                export_type = node_config.get('export_type', 'python')
                node = ExportNode(node_id, title, export_type)
            else:
                node = WorkflowNode(node_id, node_type, title)
            
            node.set_config(node_config)
            
            self.addItem(node)
            self.nodes[node_id] = node
        
        for conn_id, conn_config in config.get('connections', {}).items():
            source_node_id = conn_config.get('source_node_id', conn_config.get('source_node'))
            target_node_id = conn_config.get('target_node_id', conn_config.get('target_node'))
            
            source_port = conn_config.get('source_port')
            target_port = conn_config.get('target_port')
            
            if source_node_id not in self.nodes or target_node_id not in self.nodes:
                logger.warning(f"Cannot create connection {conn_id}: nodes not found (source: {source_node_id}, target: {target_node_id})")
                continue
            
            source_node = self.nodes[source_node_id]
            target_node = self.nodes[target_node_id]
            
            if isinstance(source_port, int):
                source_port_idx = source_port
                target_port_idx = target_port if isinstance(target_port, int) else 0
            elif isinstance(source_port, dict):
                source_port_name = source_port.get('name', source_port.get('label', 'Output'))
                target_port_name = target_port.get('name', target_port.get('label', 'Input'))
                
                source_port_idx = None
                for i, port in enumerate(source_node.output_ports):
                    if port.get('name', port.get('label', '')) == source_port_name:
                        source_port_idx = i
                        break
                if source_port_idx is None:
                    source_port_idx = 0  # Default to first output port
                
                target_port_idx = None
                for i, port in enumerate(target_node.input_ports):
                    if port.get('name', port.get('label', '')) == target_port_name:
                        target_port_idx = i
                        break
                if target_port_idx is None:
                    target_port_idx = 0  # Default to first input port
            else:
                source_port_idx = 0
                target_port_idx = 0
            
            if (source_port_idx < 0 or source_port_idx >= len(source_node.output_ports) or
                target_port_idx < 0 or target_port_idx >= len(target_node.input_ports)):
                logger.warning(f"Cannot create connection {conn_id}: invalid port indices (source: {source_port_idx}/{len(source_node.output_ports)}, target: {target_port_idx}/{len(target_node.input_ports)})")
                continue
            
            source_port_obj = source_node.output_ports[source_port_idx]
            target_port_obj = target_node.input_ports[target_port_idx]
            
            connection = ConnectionData(
                source_node_id=source_node_id,
                source_port=source_port_obj,
                target_node_id=target_node_id,
                target_port=target_port_obj
            )
            connection.connection_id = conn_id
            
            self.connections[conn_id] = connection
            
            self._create_visual_connection(source_node, target_node, source_port_idx, target_port_idx)
            
            source_node.setBrush(QBrush(QColor(120, 200, 120)))  # Light green
            target_node.setBrush(QBrush(QColor(120, 200, 120)))   # Light green
            
            logger.info(f"Successfully restored connection {conn_id}: {source_node_id}[{source_port_idx}] -> {target_node_id}[{target_port_idx}]")
    
    def _create_visual_connection(self, source_node, target_node, source_port_idx, target_port_idx):
        """Create visual representation of connection between nodes"""
        try:
            from PyQt5.QtCore import QPointF
            from PyQt5.QtGui import QPen, QPainter
            from PyQt5.QtWidgets import QGraphicsLineItem
            
            source_rect = source_node.boundingRect()
            target_rect = target_node.boundingRect()
            
            source_point = QPointF(
                source_node.pos().x() + source_rect.width(),
                source_node.pos().y() + source_rect.height() / 2
            )
            
            target_point = QPointF(
                target_node.pos().x(),
                target_node.pos().y() + target_rect.height() / 2
            )
            
            line_item = QGraphicsLineItem(source_point.x(), source_point.y(), 
                                        target_point.x(), target_point.y())
            
            pen = QPen(QColor(60, 60, 60), 2)
            line_item.setPen(pen)
            line_item.setZValue(-1)  # Behind nodes
            
            self.addItem(line_item)
            
            if not hasattr(self, 'visual_connections'):
                self.visual_connections = []
            self.visual_connections.append(line_item)
            
            logger.debug(f"Created visual connection from {source_point} to {target_point}")
            
        except Exception as e:
            logger.warning(f"Failed to create visual connection: {e}")
    
    def clear(self):
        """Clear the scene"""
        super().clear()
        self.nodes = {}
        self.connections = {}
        self.temp_connection = None
        self.connection_source_node = None
        self.connection_source_port = None
        
        if hasattr(self, 'visual_connections'):
            self.visual_connections = []


class NodeConfigDialog(QDialog):
    """Dialog for configuring workflow nodes"""
    
    def __init__(self, node: WorkflowNode, parent=None):
        """
        Initialize the node configuration dialog
        
        Args:
            node: Node to configure
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.node = node
        
        self.setWindowTitle(f"Configure {node.title}")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        form_layout = QFormLayout()
        
        self.title_edit = QLineEdit(node.title)
        form_layout.addRow("Title:", self.title_edit)
        
        self.init_node_config(form_layout)
        
        layout.addLayout(form_layout)
        
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addStretch()
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
    
    def init_node_config(self, layout: QFormLayout):
        """
        Initialize the node-specific configuration UI
        
        Args:
            layout: Form layout to add controls to
        """
        if isinstance(self.node, DatasetNode):
            self.dataset_label = QLabel(self.node.dataset_name)
            layout.addRow("Dataset:", self.dataset_label)
        
        elif isinstance(self.node, ModelNode):
            
            self.target_combo = QComboBox()
            self.target_combo.addItem("(Select target variable)", "")
            
            available_columns = self._get_available_columns()
            for col in available_columns:
                self.target_combo.addItem(col, col)
            
            current_target = getattr(self.node.model, 'target_name', None) if hasattr(self.node, 'model') and self.node.model else None
            if current_target and current_target in available_columns:
                index = self.target_combo.findData(current_target)
                if index >= 0:
                    self.target_combo.setCurrentIndex(index)
            
            layout.addRow("Target Variable:", self.target_combo)
            
            self.model_params_button = QPushButton("Configure Model Parameters...")
            layout.addRow("", self.model_params_button)
            self.model_params_button.clicked.connect(self.configure_model_parameters)
            
            self.manual_tree_button = QPushButton("Start Manual Tree Building...")
            layout.addRow("", self.manual_tree_button)
            self.manual_tree_button.clicked.connect(self.start_manual_tree_building)
            
            self.manual_tree_button.setEnabled(self._can_start_manual_tree_building())
            
            if hasattr(self.node, 'model') and self.node.model:
                model_info = f"Model: {getattr(self.node.model, 'model_name', 'Unnamed')}"
                if hasattr(self.node.model, 'is_fitted') and self.node.model.is_fitted:
                    model_info += " (Trained)"
                else:
                    model_info += " (Not trained)"
            else:
                model_info = "No model instance"
            
            self.model_info_label = QLabel(model_info)
            layout.addRow("Status:", self.model_info_label)
        
        elif isinstance(self.node, FilterNode):
            self.conditions_label = QLabel(f"{len(self.node.conditions)} conditions")
            layout.addRow("Filter Conditions:", self.conditions_label)
            
            self.edit_conditions_button = QPushButton("Edit Conditions...")
            layout.addRow("", self.edit_conditions_button)
            
            self.edit_conditions_button.clicked.connect(self.edit_conditions)
        
        elif isinstance(self.node, TransformNode):
            self.transforms_label = QLabel(f"{len(self.node.transformations)} transformations")
            layout.addRow("Transformations:", self.transforms_label)
            
            self.edit_transforms_button = QPushButton("Edit Transformations...")
            layout.addRow("", self.edit_transforms_button)
            
            self.edit_transforms_button.clicked.connect(self.edit_transformations)
        
        elif isinstance(self.node, EvaluationNode):
            self.cv_check = QCheckBox("Use Cross-Validation")
            self.cv_check.setChecked(self.node.cross_validation)
            layout.addRow("", self.cv_check)
            
            self.folds_spin = QSpinBox()
            self.folds_spin.setRange(2, 20)
            self.folds_spin.setValue(self.node.n_folds)
            layout.addRow("CV Folds:", self.folds_spin)
            
            self.cv_check.toggled.connect(self.folds_spin.setEnabled)
            self.folds_spin.setEnabled(self.node.cross_validation)
        
        elif isinstance(self.node, VisualizationNode):
            self.viz_type_combo = QComboBox()
            self.viz_type_combo.addItems(["tree", "roc", "pr_curve", "importance", "metrics"])
            
            index = self.viz_type_combo.findText(self.node.visualization_type)
            if index >= 0:
                self.viz_type_combo.setCurrentIndex(index)
            
            layout.addRow("Visualization Type:", self.viz_type_combo)
        
        elif isinstance(self.node, ExportNode):
            self.export_type_combo = QComboBox()
            self.export_type_combo.addItems(["python", "pmml", "json", "proprietary"])
            
            index = self.export_type_combo.findText(self.node.export_type)
            if index >= 0:
                self.export_type_combo.setCurrentIndex(index)
            
            layout.addRow("Export Type:", self.export_type_combo)
            
            self.path_label = QLabel(self.node.export_path or "(Not set)")
            layout.addRow("Export Path:", self.path_label)
            
            self.set_path_button = QPushButton("Set Path...")
            layout.addRow("", self.set_path_button)
            
            self.set_path_button.clicked.connect(self.set_export_path)
    
    def edit_conditions(self):
        """Open the filter conditions editor dialog"""
        # TODO: Implement filter conditions editor
        QMessageBox.information(self, "Filter Conditions", "Filter conditions editor not implemented yet")
    
    def edit_transformations(self):
        """Open the transformations editor dialog"""
        # TODO: Implement transformations editor
        QMessageBox.information(self, "Transformations", "Transformations editor not implemented yet")
    
    def set_export_path(self):
        """Open a file dialog to set the export path"""
        export_type = self.node.export_type
        
        filters = {
            'python': "Python Files (*.py);;All Files (*)",
            'pmml': "PMML Files (*.pmml);;XML Files (*.xml);;All Files (*)",
            'json': "JSON Files (*.json);;All Files (*)",
            'csv': "CSV Files (*.csv);;All Files (*)"
        }
        
        extensions = {
            'python': '.py',
            'pmml': '.pmml', 
            'json': '.json',
            'csv': '.csv'
        }
        
        file_filter = filters.get(export_type, "All Files (*)")
        default_ext = extensions.get(export_type, '')
        
        default_name = f"exported_model{default_ext}"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, f"Set Export Path ({export_type.upper()})", 
            default_name, file_filter
        )
        
        if filename:
            self.node.export_path = filename
            self.path_label.setText(filename)
            QMessageBox.information(self, "Export Path Set", 
                                  f"Export path set to:\n{filename}")
    
    def _get_available_columns(self):
        """Get available columns from connected dataset or filtered/transformed data - FIXED to use enhanced data source handling"""
        try:
            connected_data = self._get_connected_dataset()
            
            if connected_data is not None:
                if hasattr(connected_data, 'columns'):
                    columns = list(connected_data.columns)
                    logger.debug(f"Found {len(columns)} columns from connected data source")
                    return columns
                else:
                    logger.warning("Connected data has no columns attribute")
                    return []
            else:
                logger.debug("No connected data source found for model node")
                return []
                
        except Exception as e:
            logger.warning(f"Error getting available columns: {e}")
            return []
    
    def _get_columns_from_dataset_node(self, dataset_node):
        """Get columns from a direct dataset node"""
        try:
            main_window = None
            parent = self.parent()
            while parent:
                if hasattr(parent, 'datasets'):
                    main_window = parent
                    break
                parent = parent.parent()
            
            if main_window and hasattr(main_window, 'datasets'):
                dataset_name = dataset_node.dataset_name
                if dataset_name in main_window.datasets:
                    df = main_window.datasets[dataset_name]
                    columns = list(df.columns)
                    logger.debug(f"Found {len(columns)} columns from dataset '{dataset_name}'")
                    return columns
                else:
                    logger.warning(f"Dataset '{dataset_name}' not found in main window datasets")
            else:
                logger.warning("Could not find main window with datasets attribute")
            return []
        except Exception as e:
            logger.warning(f"Error getting columns from dataset node: {e}")
            return []
    
    def _get_columns_from_processed_node(self, processed_node):
        """Get columns from a filter/transform node that has processed data"""
        try:
            import pandas as pd
            
            main_window = None
            parent = self.parent()
            while parent:
                if hasattr(parent, 'execution_engine'):
                    main_window = parent
                    break
                parent = parent.parent()
            
            if main_window and hasattr(main_window, 'execution_engine'):
                execution_engine = main_window.execution_engine
                
                if hasattr(execution_engine, 'node_outputs') and processed_node.node_id in execution_engine.node_outputs:
                    node_outputs = execution_engine.node_outputs[processed_node.node_id]
                    
                    for port_name, output_data in node_outputs.items():
                        if isinstance(output_data, pd.DataFrame):
                            columns = list(output_data.columns)
                            logger.debug(f"Found {len(columns)} columns from processed node '{processed_node.node_id}' port '{port_name}'")
                            return columns
                
                logger.debug(f"No executed data found for node {processed_node.node_id}, tracing back to source")
                return self._trace_columns_from_source(processed_node)
            else:
                logger.warning("Could not find main window with execution engine")
                return []
        except Exception as e:
            logger.warning(f"Error getting columns from processed node: {e}")
            return []
    
    def _trace_columns_from_source(self, node):
        """Trace back through the workflow to find the original dataset columns"""
        try:
            scene = node.scene()
            if not scene:
                return []
            
            for connection in scene.connections.values():
                if connection.target_node_id == node.node_id:
                    source_node = scene.nodes.get(connection.source_node_id)
                    
                    if isinstance(source_node, DatasetNode):
                        return self._get_columns_from_dataset_node(source_node)
                    elif hasattr(source_node, 'node_type') and source_node.node_type in ['filter', 'transform']:
                        return self._trace_columns_from_source(source_node)
            
            logger.debug(f"Could not trace columns from source for node {node.node_id}")
            return []
        except Exception as e:
            logger.warning(f"Error tracing columns from source: {e}")
            return []
    
    def configure_model_parameters(self):
        """Open model parameters configuration dialog"""
        try:
            if not isinstance(self.node, ModelNode):
                QMessageBox.warning(self, "Invalid Node", "This is not a model node.")
                return
            
            if not hasattr(self.node, 'model') or self.node.model is None:
                try:
                    config = {}
                    parent_widget = self.parent()
                    while parent_widget:
                        if hasattr(parent_widget, 'config'):
                            config = parent_widget.config
                            break
                        parent_widget = parent_widget.parent()
                    
                    self.node.model = BespokeDecisionTree(config)
                    self.node.model.model_name = f"DecisionTree_{self.node.node_id[:8]}"
                    logger.info(f"Created new model instance for node {self.node.node_id}")
                    
                    self.node.update_model_info()
                    
                except Exception as create_error:
                    QMessageBox.critical(self, "Model Creation Error", 
                                       f"Failed to create model instance: {str(create_error)}")
                    return
            
            if not isinstance(self.node.model, BespokeDecisionTree):
                QMessageBox.warning(self, "Invalid Model", "Model is not a BespokeDecisionTree instance.")
                return
            
            try:
                from ui.dialogs.tree_configuration_dialog import TreeConfigurationDialog
            except ImportError:
                QMessageBox.warning(self, "Configuration Dialog", 
                                  "Tree configuration dialog not available. Please check your installation.")
                return
            
            config_dialog = TreeConfigurationDialog(current_config=self.node.model.get_params(), parent=self)
            if config_dialog.exec_() == QDialog.Accepted:
                new_params = config_dialog.get_configuration()
                self.node.model.set_params(**new_params)
                logger.info(f"Updated model parameters for node {self.node.node_id}")
                
                self._update_model_info_display()
                
                self.node.update_model_info()
                
        except Exception as e:
            logger.error(f"Error configuring model parameters: {e}", exc_info=True)
            QMessageBox.critical(self, "Configuration Error", f"Error configuring model parameters: {str(e)}")
    
    def start_manual_tree_building(self):
        """Start manual tree building process"""
        try:
            if not isinstance(self.node, ModelNode):
                QMessageBox.warning(self, "Invalid Node", "This is not a model node.")
                return
            
            if not self._can_start_manual_tree_building():
                QMessageBox.warning(self, "Prerequisites Not Met", 
                                  "Cannot start manual tree building. Please ensure:\n"
                                  "1. Data is connected to the model node\n"
                                  "2. Target variable is selected\n"
                                  "3. Model is properly configured")
                return
            
            target_variable = self.target_combo.currentData()
            if not target_variable:
                QMessageBox.warning(self, "Target Variable Required", 
                                  "Please select a target variable before starting manual tree building.")
                return
            
            dataset_df = self._get_connected_dataset()
            if dataset_df is None:
                QMessageBox.warning(self, "Data Required", 
                                  "Please connect a dataset to the model node before starting manual tree building.")
                return
            
            if not hasattr(self.node, 'model') or self.node.model is None:
                try:
                    self.node.model = BespokeDecisionTree()
                    self.node.model.model_name = f"ManualTree_{self.node.node_id[:8]}"
                except Exception as create_error:
                    QMessageBox.critical(self, "Model Creation Error", 
                                       f"Failed to create model instance: {str(create_error)}")
                    return
            
            try:
                self.node.model.target_name = target_variable
                
                self.node.model.growth_mode = TreeGrowthMode.MANUAL
                
                feature_columns = [col for col in dataset_df.columns if col != target_variable]
                X = dataset_df[feature_columns]
                y = dataset_df[target_variable]
                
                self.node.model.feature_names = feature_columns
                self.node.model.target_values = y.unique()
                self.node.model.class_names = [str(cls) for cls in y.unique()]
                
                self.node.model.root.samples = len(dataset_df)
                self.node.model.root.value = y.value_counts().to_dict()
                self.node.model.root.impurity = self.node.model._calculate_impurity_from_values(y.values)
                
                self.node.model.is_manual_ready = True
                
                logger.info(f"Model {self.node.node_id} ready for manual tree building")
                
                self._signal_manual_tree_start(dataset_df, target_variable)
                
                self.accept()
                
            except Exception as setup_error:
                QMessageBox.critical(self, "Setup Error", 
                                   f"Failed to set up model for manual building: {str(setup_error)}")
                return
                
        except Exception as e:
            logger.error(f"Error starting manual tree building: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Error starting manual tree building: {str(e)}")
    
    def _can_start_manual_tree_building(self):
        """Check if manual tree building can be started"""
        try:
            if not isinstance(self.node, ModelNode):
                return False
            
            if self._get_connected_dataset() is None:
                return False
            
            available_columns = self._get_available_columns()
            if not available_columns:
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking manual tree building prerequisites: {e}")
            return False
    
    def _get_connected_dataset(self):
        """Get the dataset connected to this model node - FIXED to handle Filter/Transform nodes"""
        try:
            scene = self.node.scene()
            if not scene:
                return None
            
            for connection in scene.connections.values():
                if connection.target_node_id == self.node.node_id:
                    source_node = scene.nodes.get(connection.source_node_id)
                    
                    if isinstance(source_node, DatasetNode):
                        main_window = self._get_main_window()
                        if main_window and hasattr(main_window, 'datasets'):
                            dataset_name = source_node.dataset_name
                            if dataset_name in main_window.datasets:
                                return main_window.datasets[dataset_name]
                                
                    elif isinstance(source_node, FilterNode):
                        if hasattr(source_node, 'filtered_dataframe') and source_node.filtered_dataframe is not None:
                            logger.info(f"Model node {self.node.node_id} getting data from FilterNode {source_node.node_id}")
                            return source_node.filtered_dataframe
                        else:
                            logger.warning(f"FilterNode {source_node.node_id} has no filtered data, attempting to apply filter")
                            return self._get_data_through_filter_chain(source_node)
                            
                    elif isinstance(source_node, TransformNode):
                        if hasattr(source_node, 'transformed_dataframe') and source_node.transformed_dataframe is not None:
                            logger.info(f"Model node {self.node.node_id} getting cached data from TransformNode {source_node.node_id}")
                            return source_node.transformed_dataframe
                        else:
                            logger.info(f"TransformNode {source_node.node_id} has no cached data, applying transformations on-demand")
                            transformed_data = self._get_data_through_transform_chain(source_node)
                            if transformed_data is not None:
                                source_node.transformed_dataframe = transformed_data
                                logger.info(f"Cached transformed data for TransformNode {source_node.node_id}")
                                return transformed_data
                            else:
                                logger.warning(f"Failed to get transformed data for TransformNode {source_node.node_id}")
                                return None
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting connected dataset: {e}")
            return None
    
    def _get_main_window(self):
        """Get the main window through parent chain"""
        parent = self.parent()
        while parent:
            if hasattr(parent, 'datasets'):
                return parent
            parent = parent.parent()
        return None
    
    def _get_data_through_filter_chain(self, filter_node):
        """Get data by traversing back through filter chain to original dataset"""
        try:
            scene = filter_node.scene()
            if not scene:
                return None
            
            for connection in scene.connections.values():
                if connection.target_node_id == filter_node.node_id:
                    source_node = scene.nodes.get(connection.source_node_id)
                    
                    if isinstance(source_node, DatasetNode):
                        main_window = self._get_main_window()
                        if main_window and hasattr(main_window, 'datasets'):
                            dataset_name = source_node.dataset_name
                            if dataset_name in main_window.datasets:
                                original_data = main_window.datasets[dataset_name]
                                filtered_data = filter_node.apply_filter(original_data)
                                return filtered_data
                    
                    elif isinstance(source_node, (FilterNode, TransformNode)):
                        upstream_data = self._get_data_through_filter_chain(source_node)
                        if upstream_data is not None:
                            return filter_node.apply_filter(upstream_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting data through filter chain: {e}")
            return None
    
    def _get_data_through_transform_chain(self, transform_node):
        """Get data by traversing back through transform chain to original dataset"""
        try:
            scene = transform_node.scene()
            if not scene:
                return None
            
            for connection in scene.connections.values():
                if connection.target_node_id == transform_node.node_id:
                    source_node = scene.nodes.get(connection.source_node_id)
                    
                    if isinstance(source_node, DatasetNode):
                        main_window = self._get_main_window()
                        if main_window and hasattr(main_window, 'datasets'):
                            dataset_name = source_node.dataset_name
                            if dataset_name in main_window.datasets:
                                original_data = main_window.datasets[dataset_name]
                                if hasattr(transform_node, 'apply_transformations'):
                                    transformed_data = transform_node.apply_transformations(original_data)
                                    return transformed_data
                                else:
                                    logger.warning(f"TransformNode {transform_node.node_id} has no apply_transformations method")
                                    return original_data
                    
                    elif isinstance(source_node, (FilterNode, TransformNode)):
                        upstream_data = self._get_data_through_transform_chain(source_node)
                        if upstream_data is not None and hasattr(transform_node, 'apply_transformations'):
                            return transform_node.apply_transformations(upstream_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting data through transform chain: {e}")
            return None
    
    def _signal_manual_tree_start(self, dataset_df, target_variable):
        """Signal to the main application that manual tree building should start"""
        try:
            main_window = None
            parent = self.parent()
            while parent:
                if hasattr(parent, 'start_manual_tree_building'):
                    main_window = parent
                    break
                parent = parent.parent()
            
            if main_window:
                main_window.start_manual_tree_building(self.node.model, dataset_df, target_variable)
            else:
                self._show_tree_visualizer()
                
        except Exception as e:
            logger.warning(f"Error signaling manual tree start: {e}")
            self._show_tree_visualizer()
    
    def _show_tree_visualizer(self):
        """Show the tree visualizer for manual editing"""
        try:
            from ui.tree_visualizer import TreeVisualizerDialog
            
            visualizer = TreeVisualizerDialog(self.node.model, parent=self.parent())
            visualizer.exec_()
            
        except ImportError:
            QMessageBox.information(self, "Tree Visualizer", 
                                  "Tree visualizer not available. Manual tree building will be handled by the main window.")
        except Exception as e:
            QMessageBox.warning(self, "Visualizer Error", f"Error opening tree visualizer: {str(e)}")
    
    def _update_model_info_display(self):
        """Update the model info display"""
        try:
            if hasattr(self, 'model_info_label') and self.model_info_label:
                if hasattr(self.node, 'model') and self.node.model:
                    model_info = f"Model: {getattr(self.node.model, 'model_name', 'Unnamed')}"
                    if hasattr(self.node.model, 'is_fitted') and self.node.model.is_fitted:
                        model_info += " (Trained)"
                    elif hasattr(self.node.model, 'is_manual_ready') and self.node.model.is_manual_ready:
                        model_info += " (Ready for Manual Building)"
                    else:
                        model_info += " (Not trained)"
                else:
                    model_info = "No model instance"
                
                self.model_info_label.setText(model_info)
                
        except Exception as e:
            logger.warning(f"Error updating model info display: {e}")
    
    def accept(self):
        """Handle dialog acceptance"""
        self.node.title = self.title_edit.text()
        
        if isinstance(self.node, ModelNode):
            if hasattr(self, 'target_combo'):
                selected_target = self.target_combo.currentData()
                
                if selected_target:
                    available_columns = self._get_available_columns()
                    if not available_columns:
                        QMessageBox.warning(self, "Data Connection Required", 
                                          "Please connect a dataset to the model node before selecting a target variable.")
                        return
                    
                    if selected_target not in available_columns:
                        QMessageBox.warning(self, "Invalid Target Variable", 
                                          f"Target variable '{selected_target}' is not available in the connected dataset.\n"
                                          f"Available columns: {', '.join(available_columns[:10])}{'...' if len(available_columns) > 10 else ''}")
                        return
                    
                    if not hasattr(self.node, 'model') or self.node.model is None:
                        try:
                            config = {}
                            parent_widget = self.parent()
                            while parent_widget:
                                if hasattr(parent_widget, 'config'):
                                    config = parent_widget.config
                                    break
                                parent_widget = parent_widget.parent()
                            
                            self.node.model = BespokeDecisionTree(config)
                            self.node.model.model_name = f"DecisionTree_{self.node.node_id[:8]}"
                            logger.info(f"Created new model instance for target variable setting on node {self.node.node_id}")
                        except Exception as e:
                            QMessageBox.critical(self, "Model Creation Error", 
                                               f"Failed to create model instance: {str(e)}")
                            return
                    
                    if self.node.model:
                        old_target = self.node.model.target_name
                        
                        if old_target != selected_target:
                            was_fitted = getattr(self.node.model, 'is_fitted', False)
                            
                            self.node.model.target_name = selected_target
                            logger.info(f"Set target variable to '{selected_target}' for model node {self.node.node_id}")
                            
                            if was_fitted:
                                logger.info(f"Model was fitted with old target '{old_target}', marking for retraining")
                                self.node.model.is_fitted = False
                                
                                from PyQt5.QtWidgets import QMessageBox
                                reply = QMessageBox.question(self, "Target Variable Changed", 
                                                           f"Target variable changed from '{old_target}' to '{selected_target}'.\n"
                                                           f"The model needs to be retrained. Would you like to retrain now?",
                                                           QMessageBox.Yes | QMessageBox.No)
                                
                                if reply == QMessageBox.Yes:
                                    workflow_canvas = self.parent()
                                    if workflow_canvas and hasattr(workflow_canvas, 'trigger_workflow_rerun'):
                                        workflow_canvas.trigger_workflow_rerun(self.node.node_id)
                                    else:
                                        logger.warning("Could not trigger workflow rerun - WorkflowCanvas not accessible")
                        else:
                            self.node.model.target_name = selected_target
                else:
                    if hasattr(self.node, 'model') and self.node.model and self.node.model.target_name:
                        from PyQt5.QtWidgets import QMessageBox
                        reply = QMessageBox.question(self, "Clear Target Variable", 
                                                   f"This will clear the current target variable '{self.node.model.target_name}'.\n"
                                                   "Are you sure you want to proceed?",
                                                   QMessageBox.Yes | QMessageBox.No)
                        if reply == QMessageBox.Yes and self.node.model:
                            self.node.model.target_name = None
                            logger.info(f"Cleared target variable for model node {self.node.node_id}")
            
            self.node.update_model_info()
            
        elif isinstance(self.node, EvaluationNode):
            self.node.cross_validation = self.cv_check.isChecked()
            self.node.n_folds = self.folds_spin.value()
            self.node.update_evaluation_info()
        
        elif isinstance(self.node, VisualizationNode):
            self.node.visualization_type = self.viz_type_combo.currentText()
            self.node.update_visualization_info()
        
        elif isinstance(self.node, ExportNode):
            self.node.export_type = self.export_type_combo.currentText()
            self.node.update_export_info()
        
        self.node.text_item.setHtml(f"<b>{self.node.title}</b>")
        
        super().accept()


class WorkflowCanvas(QWidget):
    """Widget containing the workflow canvas and related controls"""
    
    nodeSelected = pyqtSignal(object)  # Can be WorkflowNode or None
    nodeConfigured = pyqtSignal(WorkflowNode)
    workflowModified = pyqtSignal()
    runWorkflowRequested = pyqtSignal()
    partialWorkflowRequested = pyqtSignal(str)  # Partial workflow run from specific node
    nodeSelectedSignal = pyqtSignal(object)  # Alias for compatibility, can be WorkflowNode or None
    sceneChanged = pyqtSignal()  # Scene modification signal
    
    def __init__(self, config: Dict[str, Any], main_window_ref=None, parent=None):
        """
        Initialize the workflow canvas
        
        Args:
            config: Application configuration
            main_window_ref: Reference to the main window (optional)
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.config = config
        self.main_window_ref = main_window_ref
        
        self.init_ui()
        
        self.connect_signals()
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        
        
        self.scene = WorkflowScene(main_window_ref=self.main_window_ref)
        self.view = QGraphicsView(self.scene)
        
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.view.setDragMode(QGraphicsView.RubberBandDrag)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        
        layout.addWidget(self.view)
    
    def connect_signals(self):
        """Connect signals and slots"""
        
        self.scene.nodeSelectionChanged.connect(self.node_selection_changed)
        self.scene.nodeDoubleClicked.connect(self.node_double_clicked)
        self.scene.nodeContextMenu.connect(self.node_context_menu)
        self.scene.sceneModified.connect(self.scene_modified)
    
    def _safe_add_node(self, node_type: str):
        """
        Safely add a node with exception handling
        
        Args:
            node_type: Type of node to add
        """
        try:
            self.add_node(node_type)
        except Exception as e:
            logger.error(f"Error adding node of type {node_type}: {e}", exc_info=True)
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", f"Failed to add {node_type} node: {str(e)}")
    
    def add_node(self, node_type: str):
        """
        Add a new node to the workflow
        
        Args:
            node_type: Type of node to add
        """
        node_id = str(uuid.uuid4())
        
        if node_type == NODE_TYPE_DATASET:
            dataset_name = self._select_dataset_for_node()
            if dataset_name is None:
                return  # User cancelled dataset selection
            
            title = "Dataset"
            node = DatasetNode(node_id, title, dataset_name)
            
            if (self.main_window_ref and hasattr(self.main_window_ref, 'datasets') and 
                dataset_name in self.main_window_ref.datasets):
                df = self.main_window_ref.datasets[dataset_name]
                node.update_dataset_info(len(df), len(df.columns))
        
        elif node_type == NODE_TYPE_MODEL:
            title = "Decision Tree"
            
            try:
                if self.main_window_ref and hasattr(self.main_window_ref, 'config'):
                    model = BespokeDecisionTree(self.main_window_ref.config)
                    node = ModelNode(node_id, title, model)
                    
                    model_name = f"DecisionTree_{len(self.main_window_ref.models) + 1}"
                    model.model_name = model_name
                    
                    if hasattr(self.main_window_ref, 'models'):
                        self.main_window_ref.models[model_name] = model
                    
                    logger.info(f"Created new model '{model_name}' for node {node_id}")
                else:
                    model = BespokeDecisionTree({})
                    model.model_name = f"DecisionTree_{node_id[:8]}"
                    node = ModelNode(node_id, title, model)
                    logger.info(f"Created fallback model for node {node_id}")
                
                if not hasattr(node, 'model') or node.model is None:
                    raise ValueError("Model was not properly assigned to node")
                
                node.update_model_info()
                
            except Exception as e:
                logger.error(f"Error creating model node: {e}", exc_info=True)
                node = ModelNode(node_id, title, None)
                QMessageBox.warning(self, "Model Creation Warning", 
                                  f"Failed to create model instance: {str(e)}. You can configure it later.")
        
        elif node_type == NODE_TYPE_FILTER:
            title = "Filter"
            node = FilterNode(node_id, title)
        
        elif node_type == NODE_TYPE_TRANSFORM:
            title = "Transform"
            node = TransformNode(node_id, title)
        
        elif node_type == NODE_TYPE_EVALUATION:
            title = "Evaluation"
            node = EvaluationNode(node_id, title)
        
        elif node_type == NODE_TYPE_VISUALIZATION:
            title = "Visualization"
            node = VisualizationNode(node_id, title)
        
        elif node_type == NODE_TYPE_EXPORT:
            title = "Export"
            node = ExportNode(node_id, title)
        
        else:
            title = "Node"
            node = WorkflowNode(node_id, node_type, title)
        
        view_center = self.view.mapToScene(self.view.viewport().rect().center())
        node.setPos(view_center)
        
        self.scene.add_node(node)
        
        node.setSelected(True)
        self.node_selection_changed(node)
    
    def add_dataset_node(self, dataset_name: str, rows: int = None, cols: int = None) -> str:
        """
        Add a dataset node with the given dataset
        
        Args:
            dataset_name: Name of the dataset
            rows: Number of rows (optional)
            cols: Number of columns (optional)
            
        Returns:
            Node ID
        """
        node_id = str(uuid.uuid4())
        
        title = "Dataset"
        node = DatasetNode(node_id, title, dataset_name)
        
        if rows is not None and cols is not None:
            node.update_dataset_info(rows, cols)
        
        view_center = self.view.mapToScene(self.view.viewport().rect().center())
        node.setPos(view_center)
        
        self.scene.add_node(node)
        
        return node_id
    
    def add_model_node(self, model: Any, model_name: str) -> str:
        """
        Add a model node with the given model
        
        Args:
            model: Model object
            model_name: Name of the model
            
        Returns:
            Node ID
        """
        node_id = str(uuid.uuid4())
        
        title = model_name
        node = ModelNode(node_id, title, model)
        
        view_center = self.view.mapToScene(self.view.viewport().rect().center())
        node.setPos(view_center.x() + 100, view_center.y())
        
        self.scene.add_node(node)
        
        return node_id
    
    def zoom_in(self):
        """Zoom in the view"""
        self.view.scale(1.2, 1.2)
    
    def zoom_out(self):
        """Zoom out the view"""
        self.view.scale(1/1.2, 1/1.2)
    
    def zoom_fit(self):
        """Fit content to view"""
        rect = self.scene.itemsBoundingRect().adjusted(-50, -50, 50, 50)
        
        if not rect.isEmpty():
            self.view.fitInView(rect, Qt.KeepAspectRatio)
    
    def run_workflow(self):
        """Run the workflow"""
        self.runWorkflowRequested.emit()
    
    def trigger_workflow_rerun(self, start_node_id: str):
        """
        Trigger workflow rerun starting from a specific node
        
        Args:
            start_node_id: ID of the node to start rerunning from
        """
        logger.info(f"Triggering workflow rerun from node {start_node_id}")
        
        if hasattr(self, 'partialWorkflowRequested'):
            self.partialWorkflowRequested.emit(start_node_id)
        else:
            logger.warning("Partial workflow execution not available, running full workflow")
            self.runWorkflowRequested.emit()
    
    def node_selection_changed(self, node: Optional[WorkflowNode]):
        """
        Handle node selection change
        
        Args:
            node: Selected node (or None)
        """
        self.nodeSelected.emit(node)
        self.nodeSelectedSignal.emit(node)  # Emit compatibility signal
    
    def node_double_clicked(self, node: WorkflowNode):
        """
        Handle node double click - Navigate to detail window instead of config dialog
        
        Args:
            node: Double-clicked node
        """
        self.scene.nodeDoubleClicked.emit(node)
        
    
    def node_context_menu(self, node: WorkflowNode, pos: QPointF):
        """
        Handle node context menu (DISABLED - using main window context menu instead)
        
        Args:
            node: Node to show menu for
            pos: Position in scene coordinates
        """
        return
        
        menu = QMenu(self.parent() if self.parent() else None)
        
        if isinstance(node, ModelNode):
            has_data = self._check_node_has_data_connection(node)
            has_target = hasattr(node, 'model') and node.model and hasattr(node.model, 'target_variable') and node.model.target_variable
            is_trained = hasattr(node, 'model') and node.model and node.model.is_fitted
            
            if not has_data:
                info_action = menu.addAction("ℹ️ Connect dataset first")
                info_action.setEnabled(False)
            elif not has_target:
                configure_action = menu.addAction("📝 Configure Model & Set Target")
            elif not is_trained:
                build_action = menu.addAction("🚀 Build Model")
                configure_action = menu.addAction("⚙️ Change Configuration")
            else:
                view_results_action = menu.addAction("📊 View Results")
                manual_tree_action = menu.addAction("🌳 Manual Tree Growth")
                rebuild_action = menu.addAction("🔄 Rebuild Model")
                menu.addSeparator()
                configure_action = menu.addAction("⚙️ Change Configuration")
            
            menu.addSeparator()
            delete_action = menu.addAction("🗑️ Delete Model")
            
        elif isinstance(node, DatasetNode):
            menu.addSeparator()
            view_data_action = menu.addAction("View Data")
            filter_data_action = menu.addAction("Filter Data...")
            
        elif isinstance(node, EvaluationNode):
            menu.addSeparator()
            evaluate_action = menu.addAction("Evaluate Model")
            
        elif isinstance(node, VisualizationNode):
            menu.addSeparator()
            show_viz_action = menu.addAction("Show Visualization")
            
        elif isinstance(node, ExportNode):
            menu.addSeparator()
            export_action = menu.addAction("Export Now")
            
        elif isinstance(node, FilterNode):
            menu.addSeparator()
            configure_filter_action = menu.addAction("⚙️ Configure Filter")
            menu.addSeparator()
            delete_filter_action = menu.addAction("🗑️ Delete")
            
        elif isinstance(node, TransformNode):
            menu.addSeparator()
            configure_transform_action = menu.addAction("⚙️ Configure Transform")
            menu.addSeparator()
            delete_transform_action = menu.addAction("🗑️ Delete")
        
        menu.addSeparator()
        duplicate_action = menu.addAction("Duplicate")
        
        view_pos = self.view.mapFromScene(pos)
        global_pos = self.view.mapToGlobal(view_pos)
        action = menu.exec_(global_pos)
        
        if isinstance(node, ModelNode):
            if 'configure_action' in locals() and action == configure_action:
                dialog = NodeConfigDialog(node, self)
                
                if dialog.exec_() == QDialog.Accepted:
                    self.nodeConfigured.emit(node)
            
            elif 'build_action' in locals() and action == build_action:
                self._build_model_streamlined(node)
            
            elif 'rebuild_action' in locals() and action == rebuild_action:
                self._build_model_streamlined(node)
            
            elif 'view_results_action' in locals() and action == view_results_action:
                self._view_model_results(node)
            
            elif 'manual_tree_action' in locals() and action == manual_tree_action:
                self._open_manual_tree_builder(node)
            
            elif 'delete_action' in locals() and action == delete_action:
                reply = QMessageBox.question(
                    self, 'Delete Model Node', 
                    f'Are you sure you want to delete the "{node.title}" model?\n\nThis will remove the model and cannot be undone.',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    self.scene.remove_node(node.node_id)
                
        elif isinstance(node, DatasetNode):
            if action == view_data_action:
                self._request_data_view(node)
            elif action == filter_data_action:
                self._request_data_filter(node)
                
        elif isinstance(node, EvaluationNode) and 'evaluate_action' in locals():
            if action == evaluate_action:
                self._request_evaluation(node)
                
        elif isinstance(node, VisualizationNode) and 'show_viz_action' in locals():
            if action == show_viz_action:
                self._request_visualization(node)
                
        elif isinstance(node, ExportNode) and 'export_action' in locals():
            if action == export_action:
                self._request_export(node)
                
        elif isinstance(node, FilterNode):
            if 'configure_filter_action' in locals() and action == configure_filter_action:
                self._configure_filter(node)
            elif 'delete_filter_action' in locals() and action == delete_filter_action:
                reply = QMessageBox.question(
                    self, 'Delete Filter Node', 
                    f'Are you sure you want to delete the "{node.title}" filter?\n\nThis will remove the filter and cannot be undone.',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self.scene.remove_node(node.node_id)
                
        elif isinstance(node, TransformNode):
            if 'configure_transform_action' in locals() and action == configure_transform_action:
                self._configure_transform(node)
            elif 'delete_transform_action' in locals() and action == delete_transform_action:
                reply = QMessageBox.question(
                    self, 'Delete Transform Node', 
                    f'Are you sure you want to delete the "{node.title}" transform?\n\nThis will remove the transform and cannot be undone.',
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self.scene.remove_node(node.node_id)
        
        if 'duplicate_action' in locals() and action == duplicate_action:
            self._duplicate_node(node)
    
    def scene_modified(self):
        """Handle scene modification"""
        self.workflowModified.emit()
        self.sceneChanged.emit()  # Emit compatibility signal
    
    def clear(self):
        """Clear the canvas"""
        self.scene.clear()
    
    def clear_canvas(self):
        """Clear the canvas (alias for compatibility)"""
        self.clear()
    
    def add_node_to_canvas(self, node_type: str, title: str = None, specific_config: Dict[str, Any] = None) -> str:
        """
        Add a node to the canvas with specific configuration
        
        Args:
            node_type: Type of node to add
            title: Node title (optional)
            specific_config: Node-specific configuration (optional)
            
        Returns:
            Node ID
        """
        if node_type == NODE_TYPE_DATASET:
            dataset_name = specific_config.get('dataset_name', 'Dataset') if specific_config else 'Dataset'
            rows = specific_config.get('df_rows') if specific_config else None
            cols = specific_config.get('df_cols') if specific_config else None
            
            return self.add_dataset_node(dataset_name, rows, cols)
            
        elif node_type == NODE_TYPE_MODEL:
            model = specific_config.get('model') if specific_config else None
            model_name = title or 'Decision Tree'
            
            return self.add_model_node(model, model_name)
            
        else:
            return self.add_node(node_type)
    
    def get_workflow_config(self) -> Dict[str, Any]:
        """
        Get the configuration of the entire workflow
        
        Returns:
            Workflow configuration dictionary
        """
        return self.scene.get_config()
    
    def get_node_by_id(self, node_id: str) -> Optional[WorkflowNode]:
        """
        Get a node by its ID
        
        Args:
            node_id: The node ID to search for
            
        Returns:
            The node if found, None otherwise
        """
        return self.scene.nodes.get(node_id)
    
    def _select_dataset_for_node(self) -> Optional[str]:
        """
        Show a dialog to select a dataset for the dataset node
        
        Returns:
            Dataset name if selected, None if cancelled
        """
        if not self.main_window_ref or not hasattr(self.main_window_ref, 'datasets'):
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Datasets", "No datasets are currently loaded. Please load a dataset first.")
            return None
        
        available_datasets = list(self.main_window_ref.datasets.keys())
        
        if not available_datasets:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No Datasets", "No datasets are currently loaded. Please load a dataset first.")
            return None
        
        from PyQt5.QtWidgets import QInputDialog
        dataset_name, ok = QInputDialog.getItem(
            self, 
            "Select Dataset", 
            "Choose a dataset for this node:", 
            available_datasets, 
            0, 
            False
        )
        
        if ok and dataset_name:
            return dataset_name
        
        return None
    
    def _check_node_has_data_connection(self, node: ModelNode) -> bool:
        """Check if a model node has a data connection"""
        for conn in self.scene.connections.values():
            if (hasattr(conn, 'target_node_id') and 
                conn.target_node_id == node.node_id and 
                conn.connection_type == CONN_TYPE_DATA):
                return True
        return False
    
    def _build_model_streamlined(self, node: ModelNode):
        """Build model with streamlined workflow - validates data connection and target"""
        if not self._check_node_has_data_connection(node):
            QMessageBox.warning(self, "Build Model", 
                               "Please connect a dataset to this model first.")
            return
        
        if not (hasattr(node, 'model') and node.model and 
                hasattr(node.model, 'target_variable') and node.model.target_variable):
            QMessageBox.warning(self, "Build Model", 
                               "Please set a target variable in the model configuration first.")
            return
        
        self._request_model_training(node)
    
    def _view_model_results(self, node: ModelNode):
        """View model results including evaluation metrics and visualizations"""
        if not (hasattr(node, 'model') and node.model and node.model.is_fitted):
            QMessageBox.warning(self, "View Results", "Model must be built first.")
            return
        
        if self.main_window_ref and hasattr(self.main_window_ref, 'show_model_results'):
            self.main_window_ref.show_model_results(node.model)
        else:
            self._request_model_evaluation(node)
    
    def _request_model_training(self, node: ModelNode):
        """Request training for a model node"""
        if self.main_window_ref and hasattr(self.main_window_ref, 'train_model_from_workflow'):
            self.main_window_ref.train_model_from_workflow(node)
        else:
            QMessageBox.information(self, "Train Model", "Model training functionality not available")
    
    def _open_manual_tree_builder(self, node: ModelNode):
        """Open the manual tree building interface"""
        if not hasattr(node, 'model') or not node.model:
            QMessageBox.warning(self, "Manual Tree Building", "No model associated with this node")
            return
            
        if not node.model.is_fitted:
            QMessageBox.warning(self, "Manual Tree Building", "Model must be trained before manual building")
            return
        
        from ui.node_editor import NodeEditorDialog
        
        try:
            root_node = node.model.root
            if not root_node:
                QMessageBox.warning(self, "Manual Tree Building", "No tree structure found in model")
                return
            
            feature_names = getattr(node.model, 'feature_names', [])
            
            dialog = NodeEditorDialog(root_node, node.model, feature_names, self)
            dialog.setWindowTitle(f"Manual Tree Building - {node.title}")
            dialog.exec_()
            
            node.update_model_info()
            
        except Exception as e:
            QMessageBox.critical(self, "Manual Tree Building Error", f"Failed to open tree builder: {str(e)}")
    
    def _request_model_evaluation(self, node: ModelNode):
        """Request evaluation for a model node"""
        if self.main_window_ref and hasattr(self.main_window_ref, 'evaluate_model_from_workflow'):
            self.main_window_ref.evaluate_model_from_workflow(node)
        else:
            QMessageBox.information(self, "Evaluate Model", "Model evaluation functionality not available")
    
    def _request_data_view(self, node: DatasetNode):
        """Request data viewing for a dataset node"""
        if self.main_window_ref and hasattr(self.main_window_ref, 'view_dataset_from_workflow'):
            self.main_window_ref.view_dataset_from_workflow(node)
        else:
            QMessageBox.information(self, "View Data", "Data viewing functionality not available")
    
    def _request_data_filter(self, node: DatasetNode):
        """Request data filtering for a dataset node"""
        if self.main_window_ref and hasattr(self.main_window_ref, 'filter_dataset_from_workflow'):
            self.main_window_ref.filter_dataset_from_workflow(node)
        else:
            QMessageBox.information(self, "Filter Data", "Data filtering functionality not available")
    
    def _request_evaluation(self, node: EvaluationNode):
        """Request evaluation execution"""
        try:
            reply = QMessageBox.question(
                self, "Run Evaluation", 
                f"This will run the workflow to evaluate the model. Continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                if self.main_window_ref and hasattr(self.main_window_ref, 'run_workflow'):
                    self.main_window_ref.run_workflow()
                elif self.main_window_ref and hasattr(self.main_window_ref, 'execute_workflow'):
                    self.main_window_ref.execute_workflow()
                else:
                    QMessageBox.information(self, "Run Evaluation", "Please click the 'Run Workflow' button to evaluate the model")
                
        except Exception as e:
            QMessageBox.critical(self, "Evaluation Error", f"Failed to run evaluation: {str(e)}")
    
    def _request_visualization(self, node: VisualizationNode):
        """Request visualization display"""
        try:
            reply = QMessageBox.question(
                self, "Generate Visualization", 
                f"This will run the workflow to generate visualizations. Continue?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                if self.main_window_ref and hasattr(self.main_window_ref, 'run_workflow'):
                    self.main_window_ref.run_workflow()
                elif self.main_window_ref and hasattr(self.main_window_ref, 'execute_workflow'):
                    self.main_window_ref.execute_workflow()
                else:
                    QMessageBox.information(self, "Generate Visualization", "Please click the 'Run Workflow' button to generate visualizations")
                
        except Exception as e:
            QMessageBox.critical(self, "Visualization Error", f"Failed to generate visualization: {str(e)}")
    
    def _request_export(self, node: ExportNode):
        """Request export execution"""
        try:
            if not hasattr(node, 'export_path') or not node.export_path:
                node.set_export_path()
                
            if not hasattr(node, 'export_path') or not node.export_path:
                return
                
            reply = QMessageBox.question(
                self, "Export Model", 
                f"This will run the workflow to export the model to:\n{node.export_path}\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                if self.main_window_ref and hasattr(self.main_window_ref, 'run_workflow'):
                    self.main_window_ref.run_workflow()
                elif self.main_window_ref and hasattr(self.main_window_ref, 'execute_workflow'):
                    self.main_window_ref.execute_workflow()
                else:
                    QMessageBox.information(self, "Export Model", "Please click the 'Run Workflow' button to export the model")
                
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")
    
    def _configure_filter(self, node: FilterNode):
        """Configure filter conditions for a FilterNode"""
        try:
            from ui.dialogs.filter_config_dialog import FilterConfigDialog
            
            dialog = FilterConfigDialog(node, self)
            
            if dialog.exec_() == QDialog.Accepted:
                self.nodeConfigured.emit(node)
                logger.info(f"Filter node '{node.title}' configured with {len(node.conditions)} conditions")
                
        except ImportError:
            QMessageBox.information(
                self, "Configure Filter", 
                "Filter configuration dialog not implemented yet.\n\n"
                "This feature will allow you to set up filter conditions for your data."
            )
        except Exception as e:
            QMessageBox.critical(self, "Filter Configuration Error", f"Failed to configure filter: {str(e)}")
    
    def _configure_transform(self, node: TransformNode):
        """Configure transformations for a TransformNode"""
        try:
            from ui.dialogs.transform_config_dialog import TransformConfigDialog
            
            dialog = TransformConfigDialog(node, self)
            
            if dialog.exec_() == QDialog.Accepted:
                self.nodeConfigured.emit(node)
                logger.info(f"Transform node '{node.title}' configured with {len(node.transformations)} transformations")
                
        except ImportError:
            QMessageBox.information(
                self, "Configure Transform", 
                "Transform configuration dialog not implemented yet.\n\n"
                "This feature will allow you to set up data transformations."
            )
        except Exception as e:
            QMessageBox.critical(self, "Transform Configuration Error", f"Failed to configure transform: {str(e)}")
    
    def _duplicate_node(self, node: WorkflowNode):
        """Duplicate a node"""
        try:
            import uuid
            new_node_id = str(uuid.uuid4())
            
            if isinstance(node, DatasetNode):
                new_node = DatasetNode(new_node_id, f"{node.title} (Copy)", node.dataset_name)
            elif isinstance(node, ModelNode):
                new_node = ModelNode(new_node_id, f"{node.title} (Copy)", None)
                if self.main_window_ref and hasattr(self.main_window_ref, 'config'):
                    from models.decision_tree import BespokeDecisionTree
                    new_model = BespokeDecisionTree(self.main_window_ref.config)
                    new_model.model_name = f"{node.model.model_name}_copy" if hasattr(node, 'model') and node.model else "Decision_Tree_Copy"
                    new_node.model = new_model
                    if hasattr(self.main_window_ref, 'models'):
                        self.main_window_ref.models[new_model.model_name] = new_model
            elif isinstance(node, FilterNode):
                new_node = FilterNode(new_node_id, f"{node.title} (Copy)")
                new_node.conditions = node.conditions.copy()
            elif isinstance(node, TransformNode):
                new_node = TransformNode(new_node_id, f"{node.title} (Copy)")
                new_node.transformations = node.transformations.copy()
            elif isinstance(node, EvaluationNode):
                new_node = EvaluationNode(new_node_id, f"{node.title} (Copy)")
                new_node.metrics = node.metrics.copy()
                new_node.cross_validation = node.cross_validation
                new_node.n_folds = node.n_folds
            elif isinstance(node, VisualizationNode):
                new_node = VisualizationNode(new_node_id, f"{node.title} (Copy)", node.visualization_type)
                new_node.viz_config = node.viz_config.copy()
            elif isinstance(node, ExportNode):
                new_node = ExportNode(new_node_id, f"{node.title} (Copy)", node.export_type)
                new_node.export_path = ""  # Don't copy the path
            else:
                new_node = WorkflowNode(new_node_id, node.node_type, f"{node.title} (Copy)")
            
            offset = QPointF(50, 50)
            new_node.setPos(node.pos() + offset)
            
            self.scene.add_node(new_node)
            
            new_node.setSelected(True)
            self.node_selection_changed(new_node)
            
        except Exception as e:
            QMessageBox.critical(self, "Duplicate Node Error", f"Failed to duplicate node: {str(e)}")