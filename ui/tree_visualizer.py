#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Tree Visualizer Module for Bespoke Utility
Provides comprehensive visualization of decision trees with detailed node and edge information

"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView, 
                           QGraphicsScene, QGraphicsItem, QGraphicsRectItem, 
                           QGraphicsLineItem, QGraphicsTextItem, QComboBox, 
                           QLabel, QPushButton, QCheckBox, QSlider, QGroupBox,
                           QGraphicsEllipseItem, QGraphicsPathItem, QToolBar,
                           QAction, QFileDialog, QMessageBox, QSpinBox)
from PyQt5.QtCore import (Qt, QRectF, QPointF, QSizeF, pyqtSignal, QTimer, 
                         QPropertyAnimation, QEasingCurve, QParallelAnimationGroup,
                         QSequentialAnimationGroup)
from PyQt5.QtGui import (QPen, QBrush, QColor, QFont, QFontMetrics, QPainter, 
                        QLinearGradient, QRadialGradient, QPainterPath, QImage,
                        QTransform, QPainterPathStroker)
import numpy as np
import pandas as pd

from models.node import TreeNode
from utils.metrics_calculator import CentralMetricsCalculator
from business.split_statistics_calculator import SplitStatisticsCalculator

logger = logging.getLogger(__name__)

class EnhancedTreeNodeItem(QGraphicsRectItem):
    """Enhanced graphics item for displaying decision tree nodes with standardized layout"""
    
    def __init__(self, node: TreeNode, config: Dict[str, Any], node_number: int):
        """
        Initialize enhanced tree node item
        
        Args:
            node: TreeNode object
            config: Visualization configuration
            node_number: Sequential node number for display
        """
        self.node = node
        self.node_id = node.node_id
        self.config = config
        self.node_number = node_number
        self.is_terminal = node.is_terminal
        self.is_highlighted = False
        
        dimensions = self._calculate_dimensions()
        self.node_width = dimensions['width']
        self.node_height = dimensions['height']
        
        super().__init__(0, 0, self.node_width, self.node_height)
        
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        self.setAcceptedMouseButtons(Qt.LeftButton | Qt.RightButton)
        
        self.setZValue(0)
        
        self._create_visual_elements()
        
        self.setToolTip(self._generate_tooltip())
    
    def _calculate_dimensions(self) -> Dict[str, int]:
        """Calculate required node dimensions based on content with improved spacing"""
        content_font = QFont("DejaVu Sans", 8)
        content_font.setStyleHint(QFont.SansSerif)
        small_font = QFont("DejaVu Sans", 7)
        small_font.setStyleHint(QFont.SansSerif)
        
        content_metrics = QFontMetrics(content_font)
        small_metrics = QFontMetrics(small_font)
        
        max_width = 180  # Wider nodes to prevent text wrapping on Windows
        total_height = 10  # Padding at top
        
        line_height = max(14, content_metrics.height() + 3)  # Conservative increase for Windows
        
        if self.node.class_counts:
            for class_label, count in self.node.class_counts.items():
                percentage = CentralMetricsCalculator.calculate_percentage(count, self.node.samples, 2)
                
                class_text = f'"{class_label}"'
                count_text = f'({count:,})'
                perc_text = f'{percentage:.2f}%'
                
                class_width = content_metrics.width(class_text)
                count_width = content_metrics.width(count_text)
                perc_width = content_metrics.width(perc_text)
                
                required_width = class_width + count_width + perc_width + 40  # 40px total spacing
                max_width = max(max_width, required_width)
                total_height += line_height
        
        if self.node.samples:
            total_label_width = small_metrics.width("Total:")
            count_width = small_metrics.width(f'{self.node.samples:,}')
            perc_width = small_metrics.width('(100.00%)')
            
            total_required_width = total_label_width + count_width + perc_width + 30
            max_width = max(max_width, total_required_width)
            total_height += line_height + 4  # Extra spacing before total
        
        total_height += 10
        
        return {'width': max_width, 'height': total_height}
    
    def _create_visual_elements(self):
        """Create internal visual elements for the node with modern design"""
        
        from PyQt5.QtWidgets import QGraphicsDropShadowEffect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(8)
        shadow.setColor(QColor(0, 0, 0, 60))  # Semi-transparent black
        shadow.setOffset(2, 2)
        self.setGraphicsEffect(shadow)
        
        if self.is_terminal:
            self.setBrush(QBrush(QColor("#faf8f1")))  # Warmer beige (#faf8f1)
            self.setPen(QPen(QColor("#a67c52"), 2))   # Warmer brown border (#a67c52)
        else:
            self.setBrush(QBrush(QColor("#f0f7ff")))  # Lighter blue background (#f0f7ff)
            self.setPen(QPen(QColor("#1e40af"), 2))   # Darker blue border (#1e40af)
        
        self._create_text_elements()
    
    def _create_text_elements(self):
        """Create text elements within the node with improved spacing and modern typography"""
        content_font = QFont("DejaVu Sans", 9, QFont.Medium)
        content_font.setStyleHint(QFont.SansSerif)
        small_font = QFont("DejaVu Sans", 8, QFont.Normal)
        small_font.setStyleHint(QFont.SansSerif)
        header_font = QFont("DejaVu Sans", 10, QFont.DemiBold)
        header_font.setStyleHint(QFont.SansSerif)
        
        y_offset = 5
        line_height = max(14, QFontMetrics(content_font).height() + 3)  # Conservative increase for Windows
        
        
        if self.node.class_counts:
            for class_label, count in self.node.class_counts.items():
                percentage = CentralMetricsCalculator.calculate_percentage(count, self.node.samples, 2)
                
                class_text = QGraphicsTextItem(f'"{class_label}"', self)
                class_text.setFont(content_font)
                class_text.setDefaultTextColor(QColor("#1e293b"))  # Modern dark gray text
                class_text.setFlag(QGraphicsItem.ItemIsSelectable, False)
                class_text.setFlag(QGraphicsItem.ItemIsFocusable, False)
                class_text.setAcceptedMouseButtons(Qt.NoButton)  # Make text non-interactive
                class_text.setZValue(1)  # Higher than node rectangle
                class_text.setPos(5, y_offset)
                
                count_text = QGraphicsTextItem(f'({count:,})', self)
                count_text.setFont(content_font)
                count_text.setDefaultTextColor(QColor("#475569"))  # Slightly lighter gray
                count_text.setFlag(QGraphicsItem.ItemIsSelectable, False)
                count_text.setFlag(QGraphicsItem.ItemIsFocusable, False)
                count_text.setAcceptedMouseButtons(Qt.NoButton)  # Make text non-interactive
                count_text.setZValue(1)  # Higher than node rectangle
                
                available_width = self.node_width - 10  # Account for margins
                count_width = count_text.boundingRect().width()
                center_x = 5 + (available_width - count_width) / 2
                count_text.setPos(center_x, y_offset)
                
                perc_text = QGraphicsTextItem(f'{percentage:.2f}%', self)
                perc_text.setFont(content_font)
                perc_text.setDefaultTextColor(QColor("#059669"))  # Modern green for percentages
                perc_text.setFlag(QGraphicsItem.ItemIsSelectable, False)
                perc_text.setFlag(QGraphicsItem.ItemIsFocusable, False)
                perc_text.setAcceptedMouseButtons(Qt.NoButton)  # Make text non-interactive
                perc_text.setZValue(1)  # Higher than node rectangle
                perc_width = perc_text.boundingRect().width()
                right_x = self.node_width - perc_width - 5  # 5px from right edge
                perc_text.setPos(right_x, y_offset)
                
                y_offset += line_height
        
        if self.node.samples:
            total_y = y_offset + 2  # Add small gap before total
            
            total_label = QGraphicsTextItem("Total:", self)
            total_label.setFont(small_font)
            total_label.setDefaultTextColor(QColor("#64748b"))  # Muted gray for labels
            total_label.setFlag(QGraphicsItem.ItemIsSelectable, False)
            total_label.setFlag(QGraphicsItem.ItemIsFocusable, False)
            total_label.setAcceptedMouseButtons(Qt.NoButton)  # Make text non-interactive
            total_label.setZValue(1)  # Higher than node rectangle
            total_label.setPos(5, total_y)
            
            count_text = QGraphicsTextItem(f'{self.node.samples:,}', self)
            count_text.setFont(small_font)
            count_text.setDefaultTextColor(QColor("#1e293b"))  # Dark gray for emphasis
            count_text.setFlag(QGraphicsItem.ItemIsSelectable, False)
            count_text.setFlag(QGraphicsItem.ItemIsFocusable, False)
            count_text.setAcceptedMouseButtons(Qt.NoButton)  # Make text non-interactive
            count_text.setZValue(1)  # Higher than node rectangle
            available_width = self.node_width - 10
            count_width = count_text.boundingRect().width()
            center_x = 5 + (available_width - count_width) / 2
            count_text.setPos(center_x, total_y)
            
            perc_text = QGraphicsTextItem('(100.00%)', self)
            perc_text.setFont(small_font)
            perc_text.setDefaultTextColor(QColor("#64748b"))  # Muted gray for 100%
            perc_text.setFlag(QGraphicsItem.ItemIsSelectable, False)
            perc_text.setFlag(QGraphicsItem.ItemIsFocusable, False)
            perc_text.setAcceptedMouseButtons(Qt.NoButton)  # Make text non-interactive
            perc_text.setZValue(1)  # Higher than node rectangle
            perc_width = perc_text.boundingRect().width()
            right_x = self.node_width - perc_width - 5
            perc_text.setPos(right_x, total_y)
    
    def _generate_tooltip(self) -> str:
        """Generate comprehensive tooltip for the node"""
        tooltip_lines = []
        
        tooltip_lines.append(f"Node ID: {self.node_id}")
        tooltip_lines.append(f"Node Number: {self.node_number}")
        tooltip_lines.append(f"Depth: {self.node.depth}")
        tooltip_lines.append(f"Type: {'Terminal' if self.is_terminal else 'Internal'}")
        
        if self.node.samples:
            tooltip_lines.append(f"Samples: {self.node.samples:,}")
        
        if self.node.class_counts:
            tooltip_lines.append("Class Distribution:")
            for class_label, count in self.node.class_counts.items():
                percentage = (count / self.node.samples * 100) if self.node.samples > 0 else 0
                tooltip_lines.append(f"  {class_label}: {count:,} ({percentage:.2f}%)")
        
        if not self.is_terminal and self.node.split_feature:
            tooltip_lines.append(f"Split Feature: {self.node.split_feature}")
            
            if self.node.split_type == 'numeric' and self.node.split_value is not None:
                tooltip_lines.append(f"Split Value: {self.node.split_value}")
            elif self.node.split_type == 'categorical' and self.node.split_categories:
                tooltip_lines.append(f"Split Categories: {len(self.node.split_categories)} groups")
        
        return "\n".join(tooltip_lines)
    
    def highlight(self, highlight: bool = True):
        """Highlight or unhighlight the node"""
        self.is_highlighted = highlight
        
        if highlight:
            self.setPen(QPen(QColor(255, 0, 0), 3))  # Red border
            self.setBrush(QBrush(QColor(255, 255, 0, 100)))  # Semi-transparent yellow
        else:
            if self.is_terminal:
                self.setBrush(QBrush(QColor("#f5f5dc")))  # Light beige
                self.setPen(QPen(QColor("#8b7355"), 2))   # Brown border
            else:
                self.setBrush(QBrush(QColor("#dbeafe")))  # Light blue
                self.setPen(QPen(QColor("#2563eb"), 2))   # Blue border
    
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        try:
            super().mousePressEvent(event)
        except RuntimeError as e:
            logger.warning(f"EnhancedTreeNodeItem {self.node_id} deleted during mouse press: {e}")
    
    def mouseDoubleClickEvent(self, event):
        """Handle mouse double-click events"""
        try:
            super().mouseDoubleClickEvent(event)
        except RuntimeError as e:
            logger.warning(f"EnhancedTreeNodeItem {self.node_id} deleted during double-click: {e}")
    
    def paint(self, painter: QPainter, option, widget):
        """Custom paint method for the node"""
        try:
            rect = self.boundingRect()
            corner_radius = 8  # 8px corner radius for modern look
            
            painter.setPen(self.pen())
            painter.setBrush(self.brush())
            
            painter.drawRoundedRect(rect, corner_radius, corner_radius)
            
            if self.isSelected():
                selection_pen = QPen(QColor("#3b82f6"), 3)  # Blue selection outline
                selection_pen.setStyle(Qt.DashLine)
                painter.setPen(selection_pen)
                painter.setBrush(QBrush())  # No fill for selection outline
                
                selection_rect = rect.adjusted(-2, -2, 2, 2)
                painter.drawRoundedRect(selection_rect, corner_radius + 2, corner_radius + 2)
        except RuntimeError as e:
            logger.warning(f"EnhancedTreeNodeItem {self.node_id} deleted during paint: {e}")
    
    def boundingRect(self) -> QRectF:
        """Return the bounding rectangle of the node"""
        try:
            return super().boundingRect()
        except RuntimeError as e:
            logger.warning(f"EnhancedTreeNodeItem {self.node_id} deleted during boundingRect: {e}")
            return QRectF(0, 0, self.node_width, self.node_height)


class EnhancedTreeEdgeItem(QGraphicsPathItem):
    """Enhanced graphics item for displaying decision tree edges with detailed information"""
    
    def __init__(self, parent_item: EnhancedTreeNodeItem, child_item: EnhancedTreeNodeItem, 
                 parent_node: TreeNode, child_node: TreeNode, config: Dict[str, Any], child_index: int):
        """
        Initialize enhanced tree edge item
        
        Args:
            parent_item: Parent node graphics item
            child_item: Child node graphics item
            parent_node: Parent TreeNode object
            child_node: Child TreeNode object
            config: Visualization configuration
            child_index: Index of this child in parent's children list
        """
        super().__init__()
        
        self.parent_item = parent_item
        self.child_item = child_item
        self.parent_node = parent_node
        self.child_node = child_node
        self.config = config
        self.child_index = child_index
        
        self.setPen(QPen(QColor(50, 50, 50), 1.5))
        self.setBrush(QBrush(Qt.NoBrush))
        
        self._create_edge_path()
    
    def _create_edge_path(self):
        """Create the edge path for both orientations with simple clean lines - FIXED to account for wrapped text height"""
        parent_rect = self.parent_item.boundingRect()
        child_rect = self.child_item.boundingRect()
        
        parent_pos = self.parent_item.pos()
        child_pos = self.child_item.pos()
        
        orientation = self.config.get('orientation', 'top_down')
        
        wrapped_text_height = self._calculate_wrapped_text_height()
        
        self._create_edge_labels()
        
        path = QPainterPath()
        
        if orientation == 'top_down':
            parent_bottom_center = QPointF(
                parent_pos.x() + parent_rect.width() / 2,
                parent_pos.y() + parent_rect.height()
            )
            
            child_top_center = QPointF(
                child_pos.x() + child_rect.width() / 2,
                child_pos.y()
            )
            
            path.moveTo(parent_bottom_center)
            
            base_vertical_distance = (child_top_center.y() - parent_bottom_center.y()) * 0.3
            text_padding = max(40, wrapped_text_height + 20)  # Minimum 40px, more if text is tall
            adjusted_vertical_distance = min(base_vertical_distance, text_padding)
            horizontal_y = child_top_center.y() - text_padding
            
            if horizontal_y < parent_bottom_center.y() + 10:
                horizontal_y = parent_bottom_center.y() + 10
            
            path.lineTo(parent_bottom_center.x(), horizontal_y)
            
            path.lineTo(child_top_center.x(), horizontal_y)
            
            final_y = child_top_center.y()
            path.lineTo(child_top_center.x(), final_y)
        
        else:  # left_right orientation
            parent_right_center = QPointF(
                parent_pos.x() + parent_rect.width(),
                parent_pos.y() + parent_rect.height() / 2
            )
            
            child_left_center = QPointF(
                child_pos.x(),
                child_pos.y() + child_rect.height() / 2
            )
            
            path.moveTo(parent_right_center)
            
            base_horizontal_distance = (child_left_center.x() - parent_right_center.x()) * 0.3
            text_width_padding = max(80, wrapped_text_height * 2)  # Estimate width from height
            adjusted_horizontal_distance = min(base_horizontal_distance, text_width_padding)
            vertical_x = child_left_center.x() - text_width_padding
            
            if vertical_x < parent_right_center.x() + 10:
                vertical_x = parent_right_center.x() + 10
            
            path.lineTo(vertical_x, parent_right_center.y())
            
            path.lineTo(vertical_x, child_left_center.y())
            
            path.lineTo(child_left_center)
        
        self.setPath(path)
    
    def _calculate_wrapped_text_height(self) -> int:
        """Calculate the height needed for wrapped text labels using font metrics"""
        try:
            bin_text = self._get_bin_value_text()
            if not bin_text:
                return 0
            
            font = QFont("DejaVu Sans", 8, QFont.Bold)
            font.setStyleHint(QFont.SansSerif)
            font_metrics = QFontMetrics(font)
            
            orientation = self.config.get('orientation', 'top_down')
            max_width_pixels = 200 if orientation == 'top_down' else 150
            
            wrapped_text = self._wrap_text_with_font_metrics(bin_text, max_width_pixels, font_metrics)
            
            line_count = wrapped_text.count('\n') + 1
            
            base_height = 15  # Height for node number label
            line_height = font_metrics.height()  # Actual font line height
            total_height = base_height + (line_count * line_height)
            
            logger.debug(f"Calculated wrapped text height: {total_height}px for {line_count} lines")
            return total_height
            
        except Exception as e:
            logger.warning(f"Error calculating wrapped text height: {e}")
            return 30  # Default fallback height
    
    def _wrap_text_with_font_metrics(self, text: str, max_width_pixels: int, font_metrics) -> str:
        """Wrap text based on actual pixel width rather than character count"""
        if not text:
            return text
            
        try:
            if '[' in text and ']' in text and ',' in text:
                content = text.strip('[]')
                parts = [part.strip() for part in content.split(',')]
                
                lines = []
                current_line = []
                
                for part in parts:
                    test_line = current_line + [part]
                    test_text = ', '.join(test_line)
                    text_width = font_metrics.width(test_text)
                    
                    if text_width > max_width_pixels and current_line:
                        lines.append(', '.join(current_line))
                        current_line = [part]
                    else:
                        current_line.append(part)
                
                if current_line:
                    lines.append(', '.join(current_line))
                
                return '\n'.join(lines)
            
            else:
                words = text.split()
                lines = []
                current_line = []
                
                for word in words:
                    test_line = current_line + [word]
                    test_text = ' '.join(test_line)
                    text_width = font_metrics.width(test_text)
                    
                    if text_width > max_width_pixels and current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        current_line.append(word)
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                return '\n'.join(lines)
                
        except Exception as e:
            logger.warning(f"Error in font metrics text wrapping: {e}")
            return self._wrap_text(text, 35 if '[' in text else 30)
    
    def _create_edge_labels(self):
        """Create properly positioned labels with background to hide overlapping lines"""
        parent_pos = self.parent_item.pos()
        child_pos = self.child_item.pos()
        parent_rect = self.parent_item.boundingRect()
        child_rect = self.child_item.boundingRect()
        
        orientation = self.config.get('orientation', 'top_down')
        
        if orientation == 'top_down':
            wrapped_height = self._calculate_wrapped_text_height()
            dynamic_offset = max(60, wrapped_height + 25)  # Moderate offset with increased level_distance
            
            label_x = child_pos.x() + child_rect.width() / 2
            label_y = child_pos.y() - dynamic_offset  # Dynamic positioning to avoid overlap
            
            node_number_text = f"[node_no: {self.child_item.node_number}]"
            node_font = QFont("DejaVu Sans", 7)
            node_font.setStyleHint(QFont.SansSerif)
            node_number_label = self._create_label_with_background(
                node_number_text, 
                node_font, 
                QColor(0, 0, 150),
                label_x, 
                label_y
            )
            
            bin_text = self._get_bin_value_text()
            if bin_text:
                font = QFont("Arial", 8, QFont.Bold)
                font.setStyleHint(QFont.SansSerif)
                font_metrics = QFontMetrics(font)
                wrapped_text = self._wrap_text_with_font_metrics(bin_text, 200, font_metrics)
                label_font = QFont("Arial", 8, QFont.Bold)
                label_font.setStyleHint(QFont.SansSerif)
                bin_label = self._create_label_with_background(
                    wrapped_text,
                    label_font,
                    QColor(0, 100, 0),
                    label_x,
                    label_y + 15  # Below node number
                )
        
        else:  # left_right orientation
            label_x = child_pos.x() + child_rect.width() + 15
            label_y = child_pos.y() + child_rect.height() / 2 - 15
            
            node_number_text = f"[node_no: {self.child_item.node_number}]"
            node_font_lr = QFont("DejaVu Sans", 7)
            node_font_lr.setStyleHint(QFont.SansSerif)
            node_number_label = self._create_label_with_background(
                node_number_text,
                node_font_lr,
                QColor(0, 0, 150),
                label_x,
                label_y
            )
            
            bin_text = self._get_bin_value_text()
            if bin_text:
                wrapped_text = self._wrap_text(bin_text, max_width=150)
                label_font = QFont("Arial", 8, QFont.Bold)
                label_font.setStyleHint(QFont.SansSerif)
                bin_label = self._create_label_with_background(
                    wrapped_text,
                    label_font,
                    QColor(0, 100, 0),
                    label_x,
                    label_y + 15
                )
    
    def _create_label_with_background(self, text: str, font: QFont, color: QColor, x: float, y: float) -> QGraphicsTextItem:
        """Create a text label with a white background to hide lines behind it"""
        text_item = QGraphicsTextItem(text, self)
        text_item.setFont(font)
        text_item.setDefaultTextColor(color)
        
        text_rect = text_item.boundingRect()
        centered_x = x - text_rect.width() / 2
        text_item.setPos(centered_x, y)
        
        background_rect = QGraphicsRectItem(
            centered_x - 3,  # Small padding
            y - 1,
            text_rect.width() + 6,
            text_rect.height() + 2,
            self
        )
        
        background_rect.setBrush(QBrush(QColor(255, 255, 255, 230)))  # Semi-transparent white
        background_rect.setPen(QPen(QColor(255, 255, 255, 230)))
        
        background_rect.setZValue(-1)
        text_item.setZValue(0)
        
        return text_item
    
    def _wrap_text(self, text: str, max_width: int) -> str:
        """Wrap text to fit within specified width"""
        if len(text) <= 30:  # Short text doesn't need wrapping
            return text
        
        if '[' in text and ']' in text and ',' in text:
            content = text.strip('[]')
            parts = [part.strip() for part in content.split(',')]
            
            lines = []
            current_line = []
            current_length = 0
            
            for part in parts:
                part_length = len(part) + 2  # +2 for comma and space
                if current_length + part_length > 35 and current_line:  # Start new line
                    lines.append(', '.join(current_line))
                    current_line = [part]
                    current_length = len(part)
                else:
                    current_line.append(part)
                    current_length += part_length
            
            if current_line:
                lines.append(', '.join(current_line))
            
            if len(lines) > 1:
                return '[' + ']\n['.join(lines) + ']'
            else:
                return '[' + lines[0] + ']'
        
        words = text.split(' ')
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) > 30 and current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + 1  # +1 for space
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    def _get_bin_value_text(self) -> str:
        """Generate the bin value text for the edge with improved validation"""
        if not self.parent_node or not hasattr(self.parent_node, 'split_feature') or not self.parent_node.split_feature:
            logger.warning(f"Edge label: No split feature for parent node {getattr(self.parent_node, 'node_id', 'unknown')}")
            return ""
        
        if not hasattr(self.parent_node, 'children') or self.child_index >= len(self.parent_node.children):
            logger.warning(f"Edge label: Invalid child index {self.child_index} for parent {self.parent_node.node_id} with {len(getattr(self.parent_node, 'children', []))} children")
            return f"invalid_idx_{self.child_index}"
        
        actual_child_index = -1
        if hasattr(self.parent_node, 'children') and self.child_node in self.parent_node.children:
            actual_child_index = self.parent_node.children.index(self.child_node)
            if actual_child_index != self.child_index:
                logger.warning(f"Edge label: Child index mismatch! Expected {self.child_index}, actual {actual_child_index} for parent {self.parent_node.node_id} -> child {self.child_node.node_id}")
                self.child_index = actual_child_index  # Use the correct index
        
        # Log debug information for troubleshooting
        logger.debug(f"Edge label: Parent {self.parent_node.node_id} ({self.parent_node.split_feature}) -> Child {self.child_node.node_id} (index {self.child_index})")
        
        if self.parent_node.split_type == 'numeric':
            if self.parent_node.split_value is not None:
                if self.child_index == 0:
                    return f"≤ {self.parent_node.split_value:.3f}"
                else:
                    return f"> {self.parent_node.split_value:.3f}"
        
        elif self.parent_node.split_type == 'categorical':
            if self.parent_node.split_categories:
                categories = []
                for cat, idx in self.parent_node.split_categories.items():
                    if idx == self.child_index:
                        categories.append(str(cat))
                
                if len(categories) <= 3:
                    return ", ".join(categories)
                elif categories:
                    if len(categories) <= 6:
                        return ", ".join(categories)
                    else:
                        first_few = categories[:3]
                        return ", ".join(first_few) + f", +{len(categories)-3} more"
        
        elif self.parent_node.split_type == 'numeric_multi_bin':
            if hasattr(self.parent_node, 'split_thresholds') and self.parent_node.split_thresholds:
                thresholds = self.parent_node.split_thresholds
                if self.child_index == 0:
                    return f"≤ {thresholds[0]:.3f}"
                elif self.child_index == len(thresholds):
                    return f"> {thresholds[-1]:.3f}"
                else:
                    return f"{thresholds[self.child_index-1]:.3f} < x ≤ {thresholds[self.child_index]:.3f}"
        
        elif self.parent_node.split_type == 'categorical_multi_bin':
            if hasattr(self.parent_node, 'split_bin_categories') and self.parent_node.split_bin_categories:
                bin_categories = self.parent_node.split_bin_categories
                if self.child_index < len(bin_categories):
                    categories = bin_categories[self.child_index]
                    if len(categories) <= 3:
                        return ", ".join(str(cat) for cat in categories)
                    elif len(categories) <= 6:
                        return ", ".join(str(cat) for cat in categories)
                    else:
                        first_few = [str(cat) for cat in categories[:3]]
                        return ", ".join(first_few) + f", +{len(categories)-3} more"
        
        logger.warning(f"Edge label: Unhandled split type {self.parent_node.split_type} for parent {self.parent_node.node_id}")
        return f"bin_{self.child_index}"


class VariableNameLabel(QGraphicsItem):
    """Special label for displaying variable names and statistics below parent nodes with background"""
    
    def __init__(self, text: str, statistic_text: str, parent_item: EnhancedTreeNodeItem, config: Dict[str, Any]):
        super().__init__()
        
        self.text = text
        self.statistic_text = statistic_text
        self.font = QFont("DejaVu Sans", 9, QFont.Bold)
        self.font.setStyleHint(QFont.SansSerif)
        self.stat_font = QFont("DejaVu Sans", 8, QFont.Normal)
        self.stat_font.setStyleHint(QFont.SansSerif)
        self.color = QColor(0, 100, 0)  # Dark green
        self.stat_color = QColor(0, 0, 150)  # Dark blue for statistics
        
        font_metrics = QFontMetrics(self.font)
        stat_metrics = QFontMetrics(self.stat_font)
        
        self.text_rect = font_metrics.boundingRect(self.text)
        self.stat_rect = stat_metrics.boundingRect(self.statistic_text)
        
        total_width = max(self.text_rect.width(), self.stat_rect.width())
        total_height = self.text_rect.height() + self.stat_rect.height() + 2  # 2px spacing
        
        parent_rect = parent_item.boundingRect()
        parent_pos = parent_item.pos()
        
        label_x = parent_pos.x() + parent_rect.width() / 2 - total_width / 2
        label_y = parent_pos.y() + parent_rect.height() + 5
        
        self.setPos(label_x, label_y)
        
        self.bg_rect = QRectF(
            -3, -2,  # Small padding
            total_width + 6,
            total_height + 4
        )
    
    def boundingRect(self) -> QRectF:
        """Return the bounding rectangle"""
        return self.bg_rect
    
    def paint(self, painter: QPainter, option, widget):
        """Custom paint method with white background for variable name and statistic"""
        painter.setBrush(QBrush(QColor(255, 255, 255, 230)))  # Semi-transparent white
        painter.setPen(QPen(QColor(255, 255, 255, 230)))
        painter.drawRect(self.bg_rect)
        
        painter.setPen(QPen(self.color))
        painter.setFont(self.font)
        text_x = (self.bg_rect.width() - 6 - self.text_rect.width()) / 2  # Center horizontally
        painter.drawText(QPointF(text_x, self.text_rect.height() - 2), self.text)
        
        painter.setPen(QPen(self.stat_color))
        painter.setFont(self.stat_font)
        stat_x = (self.bg_rect.width() - 6 - self.stat_rect.width()) / 2  # Center horizontally
        stat_y = self.text_rect.height() + self.stat_rect.height()
        painter.drawText(QPointF(stat_x, stat_y), self.statistic_text)


class EnhancedTreeScene(QGraphicsScene):
    """Enhanced graphics scene for displaying decision trees"""
    
    nodeClicked = pyqtSignal(str)
    nodeDoubleClicked = pyqtSignal(str)
    nodeRightClicked = pyqtSignal(str, QPointF, QPointF)  # node_id, scene_pos, global_pos
    
    def __init__(self, parent=None):
        """Initialize enhanced tree scene"""
        super().__init__(parent)
        
        self.node_items = {}
        self.edge_items = []
        self.label_items = []
        self.node_positions = {}
        self.node_numbers = {}
        
        self.root_node = None
        
        self.config = {}
    
    def set_configuration(self, config: Dict[str, Any]):
        """Set visualization configuration"""
        self.config = config
    
    def update_tree(self, root_node: TreeNode):
        """Update the tree visualization"""
        try:
            if not hasattr(self, 'clear') or self is None:
                logger.warning("EnhancedTreeScene has been deleted, cannot update tree")
                return
                
            self.root_node = root_node
            
            self.clearSelection()  # FIX: Clear selection state to prevent blue highlights
            self.clear()
            self.node_items.clear()
            self.edge_items.clear()
            self.label_items.clear()
            self.node_positions.clear()
            self.node_numbers.clear()
        except RuntimeError as e:
            logger.warning(f"EnhancedTreeScene deleted during update_tree: {e}")
            return
        
        if not root_node:
            return
        
        all_nodes = root_node.get_subtree_nodes()
        
        logger.info(f"EnhancedTreeScene.update_tree: Found {len(all_nodes)} nodes in tree")
        for node in all_nodes:
            logger.info(f"  Node {node.node_id}: "
                       f"is_terminal={node.is_terminal}, children={len(node.children)}, "
                       f"parent={node.parent.node_id if node.parent else None}")
        
        base_node_width = self.config.get('node_width', 140)
        base_node_height = self.config.get('node_height', 100)
        level_distance = self.config.get('level_distance', 180)
        sibling_distance = self.config.get('sibling_distance', 60)
        orientation = self.config.get('orientation', 'top_down')
        
        self._calculate_node_positions(root_node, orientation, level_distance, sibling_distance)
        
        self._create_node_items(all_nodes)
        
        self._create_edge_items(all_nodes)
        
        self._create_variable_labels(all_nodes)
        
        items_rect = self.itemsBoundingRect()
        self.setSceneRect(items_rect.adjusted(-20, -20, 20, 20))
        
        logger.info(f"EnhancedTreeScene.update_tree: Tree visualization updated with {len(all_nodes)} nodes")
    
    def _calculate_node_positions(self, root_node: TreeNode, orientation: str, level_distance: int, sibling_distance: int):
        """Calculate positions for all nodes using proper tree layout algorithm"""
        all_nodes = root_node.get_subtree_nodes()
        
        for i, node in enumerate(all_nodes):
            self.node_numbers[node.node_id] = i + 1
        
        if orientation == 'top_down':
            self._calculate_top_down_positions(root_node, level_distance, sibling_distance)
        else:
            self._calculate_left_right_positions(root_node, level_distance, sibling_distance)
    
    def _calculate_top_down_positions(self, root_node: TreeNode, level_distance: int, sibling_distance: int):
        """Calculate top-down tree positions"""
        subtree_widths = self._calculate_subtree_widths(root_node, sibling_distance)
        
        root_x = subtree_widths[root_node.node_id] / 2
        root_y = 50  # Top margin
        
        self.node_positions[root_node.node_id] = (root_x, root_y)
        
        self._position_children_top_down(root_node, root_x, root_y, level_distance, sibling_distance, subtree_widths)
    
    def _calculate_left_right_positions(self, root_node: TreeNode, level_distance: int, sibling_distance: int):
        """Calculate left-right tree positions"""
        subtree_heights = self._calculate_subtree_heights(root_node, sibling_distance)
        
        root_x = 50  # Left margin
        root_y = subtree_heights[root_node.node_id] / 2
        
        self.node_positions[root_node.node_id] = (root_x, root_y)
        
        self._position_children_left_right(root_node, root_x, root_y, level_distance, sibling_distance, subtree_heights)
    
    def _calculate_subtree_widths(self, node: TreeNode, sibling_distance: int) -> Dict[str, float]:
        """Calculate the width required for each subtree"""
        widths = {}
        
        def calculate_width(n: TreeNode) -> float:
            if n.is_terminal or not n.children:
                widths[n.node_id] = 200  # Base node width
                return 200
            
            child_widths = [calculate_width(child) for child in n.children]
            
            total_child_width = sum(child_widths) + (len(child_widths) - 1) * sibling_distance
            
            node_width = max(200, total_child_width)
            widths[n.node_id] = node_width
            
            return node_width
        
        calculate_width(node)
        return widths
    
    def _calculate_subtree_heights(self, node: TreeNode, sibling_distance: int) -> Dict[str, float]:
        """Calculate the height required for each subtree"""
        heights = {}
        
        def calculate_height(n: TreeNode) -> float:
            if n.is_terminal or not n.children:
                heights[n.node_id] = 120  # Base node height
                return 120
            
            child_heights = [calculate_height(child) for child in n.children]
            
            total_child_height = sum(child_heights) + (len(child_heights) - 1) * sibling_distance
            
            node_height = max(120, total_child_height)
            heights[n.node_id] = node_height
            
            return node_height
        
        calculate_height(node)
        return heights
    
    def _position_children_top_down(self, parent: TreeNode, parent_x: float, parent_y: float, 
                                   level_distance: int, sibling_distance: int, subtree_widths: Dict[str, float]):
        """Position children for top-down layout"""
        if not parent.children:
            return
        
        total_width = sum(subtree_widths[child.node_id] for child in parent.children)
        total_width += (len(parent.children) - 1) * sibling_distance
        
        start_x = parent_x - total_width / 2
        
        current_x = start_x
        for child in parent.children:
            child_width = subtree_widths[child.node_id]
            child_x = current_x + child_width / 2
            child_y = parent_y + level_distance
            
            self.node_positions[child.node_id] = (child_x, child_y)
            
            self._position_children_top_down(child, child_x, child_y, level_distance, sibling_distance, subtree_widths)
            
            current_x += child_width + sibling_distance
    
    def _position_children_left_right(self, parent: TreeNode, parent_x: float, parent_y: float, 
                                     level_distance: int, sibling_distance: int, subtree_heights: Dict[str, float]):
        """Position children for left-right layout"""
        if not parent.children:
            return
        
        total_height = sum(subtree_heights[child.node_id] for child in parent.children)
        total_height += (len(parent.children) - 1) * sibling_distance
        
        start_y = parent_y - total_height / 2
        
        current_y = start_y
        for child in parent.children:
            child_height = subtree_heights[child.node_id]
            child_x = parent_x + level_distance
            child_y = current_y + child_height / 2
            
            self.node_positions[child.node_id] = (child_x, child_y)
            
            self._position_children_left_right(child, child_x, child_y, level_distance, sibling_distance, subtree_heights)
            
            current_y += child_height + sibling_distance
    
    def _create_node_items(self, all_nodes: List[TreeNode]):
        """Create graphics items for all nodes"""
        for node in all_nodes:
            if node.node_id in self.node_positions:
                x, y = self.node_positions[node.node_id]
                node_number = self.node_numbers[node.node_id]
                
                node_item = EnhancedTreeNodeItem(node, self.config, node_number)
                node_item.setPos(x - node_item.node_width / 2, y - node_item.node_height / 2)
                
                self.addItem(node_item)
                self.node_items[node.node_id] = node_item
                
                logger.debug(f"Created node item for {node.node_id} at ({x}, {y})")
    
    def _create_edge_items(self, all_nodes: List[TreeNode]):
        """Create graphics items for all edges"""
        for node in all_nodes:
            if node.children and node.node_id in self.node_items:
                parent_item = self.node_items[node.node_id]
                
                for child_index, child in enumerate(node.children):
                    if child.node_id in self.node_items:
                        child_item = self.node_items[child.node_id]
                        
                        edge_item = EnhancedTreeEdgeItem(
                            parent_item, child_item, node, child, self.config, child_index
                        )
                        
                        self.addItem(edge_item)
                        self.edge_items.append(edge_item)
                        
                        logger.debug(f"Created edge from {node.node_id} to {child.node_id} (index {child_index})")
    
    def _create_variable_labels(self, all_nodes: List[TreeNode]):
        """Create variable name labels with statistics for non-terminal nodes"""
        for node in all_nodes:
            if (not node.is_terminal and 
                hasattr(node, 'split_feature') and 
                node.split_feature and 
                node.node_id in self.node_items):
                
                parent_item = self.node_items[node.node_id]
                
                statistic_text = self._calculate_node_statistic(node)
                
                var_label = VariableNameLabel(node.split_feature, statistic_text, parent_item, self.config)
                
                self.addItem(var_label)
                self.label_items.append(var_label)
                
                logger.debug(f"Created variable label '{node.split_feature}' with statistic '{statistic_text}' for node {node.node_id}")
    
    def _calculate_node_statistic(self, node: TreeNode) -> str:
        """Calculate the statistic value for a node based on the selected criterion"""
        try:
            criterion = self.config.get('criterion', 'gini')
            
            if hasattr(criterion, 'value'):
                criterion_str = criterion.value.lower()
            else:
                criterion_str = str(criterion).lower()
            
            criterion_mapping = {
                'gini': 'gini',
                'entropy': 'entropy',
                'log_loss': 'log_loss'
            }
            calc_criterion = criterion_mapping.get(criterion_str, 'gini')
            
            calc = SplitStatisticsCalculator(calc_criterion)
            
            target_data = self._get_node_target_data(node)
            if target_data is None or len(target_data) == 0:
                return "N/A"
            
            impurity = calc.calculate_impurity(target_data)
            
            if criterion_str == 'gini':
                return f"Gini: {impurity:.4f}"
            elif criterion_str == 'entropy':
                return f"Entropy: {impurity:.4f}"
            elif criterion_str == 'log_loss':
                return f"Deviance: {impurity:.4f}"
            else:
                return f"Impurity: {impurity:.4f}"
                
        except Exception as e:
            logger.warning(f"Error calculating statistic for node {node.node_id}: {e}")
            return "N/A"
    
    def _get_node_target_data(self, node: TreeNode) -> Optional[pd.Series]:
        """Get target data for a specific node"""
        try:
            if hasattr(node, 'class_counts') and node.class_counts:
                target_values = []
                for class_val, count in node.class_counts.items():
                    target_values.extend([class_val] * count)
                return pd.Series(target_values)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error getting target data for node {node.node_id}: {e}")
            return None
    
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        try:
            item = self.itemAt(event.scenePos(), QTransform())
            if isinstance(item, EnhancedTreeNodeItem):
                if event.button() == Qt.LeftButton:
                    self.nodeClicked.emit(item.node_id)
                elif event.button() == Qt.RightButton:
                    self.nodeRightClicked.emit(item.node_id, event.scenePos(), event.screenPos())
            
            super().mousePressEvent(event)
        except RuntimeError as e:
            logger.warning(f"EnhancedTreeScene deleted during mousePressEvent: {e}")
    
    def mouseDoubleClickEvent(self, event):
        """Handle mouse double-click events"""
        try:
            item = self.itemAt(event.scenePos(), QTransform())
            if isinstance(item, EnhancedTreeNodeItem):
                self.nodeDoubleClicked.emit(item.node_id)
            
            super().mouseDoubleClickEvent(event)
        except RuntimeError as e:
            logger.warning(f"EnhancedTreeScene deleted during mouseDoubleClickEvent: {e}")


class TreeVisualizerWidget(QWidget):
    """Enhanced widget for visualizing decision trees with comprehensive controls"""
    
    nodeSelected = pyqtSignal(str)
    nodeDoubleClicked = pyqtSignal(str)
    nodeRightClicked = pyqtSignal(str, QPointF, QPointF)  # node_id, scene_pos, global_pos
    
    def __init__(self, parent=None):
        """Initialize enhanced tree visualizer widget"""
        super().__init__(parent)
        
        self.config = {
            'node_width': 140,
            'node_height': 100,
            'level_distance': 220,
            'sibling_distance': 60,
            'orientation': 'top_down',
            'show_node_ids': False,
            'show_samples': True,
            'show_percentages': True,
            'show_importance': True,
            'target_variable_name': 'FLAG_EMP_PHONE'
        }
        
        self.init_ui()
        
        self.connect_signals()
        
        logger.info("TreeVisualizerWidget initialized")
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        self.create_toolbar(layout)
        
        self.scene = EnhancedTreeScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.view.setDragMode(QGraphicsView.RubberBandDrag)
        
        layout.addWidget(self.view)
        
        self.create_controls_panel(layout)
    
    def create_toolbar(self, layout):
        """Create toolbar with common actions"""
        toolbar = QToolBar("Tree Visualization")
        toolbar.setVisible(False)  # Hide as requested - functionality moved to header
        layout.addWidget(toolbar)
        
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.triggered.connect(self.zoom_in)
        toolbar.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.triggered.connect(self.zoom_out)
        toolbar.addAction(zoom_out_action)
        
        zoom_fit_action = QAction("Fit to View", self)
        zoom_fit_action.triggered.connect(self.zoom_fit)
        toolbar.addAction(zoom_fit_action)
        
        toolbar.addSeparator()
        
        export_action = QAction("Save Image", self)
        export_action.triggered.connect(self.export_image)
        toolbar.addAction(export_action)
    
    def create_controls_panel(self, layout):
        """Create controls panel for visualization options"""
        controls_group = QGroupBox("Visualization Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        orientation_label = QLabel("Orientation:")
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItem("Top-Down", "top_down")
        self.orientation_combo.addItem("Left-Right", "left_right")
        self.orientation_combo.currentIndexChanged.connect(self.change_orientation)
        
        spacing_label = QLabel("Node Spacing:")
        self.spacing_spin = QSpinBox()
        self.spacing_spin.setRange(40, 200)
        self.spacing_spin.setValue(60)
        self.spacing_spin.valueChanged.connect(self.change_spacing)
        
        level_label = QLabel("Level Distance:")
        self.level_spin = QSpinBox()
        self.level_spin.setRange(100, 300)
        self.level_spin.setValue(220)
        self.level_spin.valueChanged.connect(self.change_level_distance)
        
        self.show_ids_checkbox = QCheckBox("Show Node IDs")
        self.show_ids_checkbox.stateChanged.connect(self.toggle_show_ids)
        
        self.show_stats_checkbox = QCheckBox("Show Statistics")
        self.show_stats_checkbox.setChecked(True)
        self.show_stats_checkbox.stateChanged.connect(self.toggle_show_stats)
        
        controls_layout.addWidget(orientation_label)
        controls_layout.addWidget(self.orientation_combo)
        controls_layout.addWidget(spacing_label)
        controls_layout.addWidget(self.spacing_spin)
        controls_layout.addWidget(level_label)
        controls_layout.addWidget(self.level_spin)
        controls_layout.addWidget(self.show_ids_checkbox)
        controls_layout.addWidget(self.show_stats_checkbox)
        controls_layout.addStretch()
        
        layout.addWidget(controls_group)
    
    def connect_signals(self):
        """Connect internal signals"""
        self.scene.nodeClicked.connect(self.nodeSelected.emit)
        self.scene.nodeDoubleClicked.connect(self.nodeDoubleClicked.emit)
        self.scene.nodeRightClicked.connect(self.nodeRightClicked.emit)
    
    def set_tree(self, root_node: TreeNode, model=None):
        """Set the tree to visualize"""
        logger.info(f"TreeVisualizerWidget.set_tree called with root node: {root_node.node_id if root_node else None}")
        
        try:
            if not hasattr(self, 'scene') or self.scene is None:
                logger.warning("TreeVisualizerWidget scene has been deleted, cannot set tree")
                return
            
            if model:
                if hasattr(model, 'target_name') and model.target_name:
                    self.config['target_variable_name'] = model.target_name
                if hasattr(model, 'criterion'):
                    self.config['criterion'] = model.criterion
            
            self.scene.set_configuration(self.config)
            self.scene.update_tree(root_node)
            
            QTimer.singleShot(100, self.zoom_100)
            
        except RuntimeError as e:
            logger.warning(f"TreeVisualizerWidget scene deleted during set_tree: {e}")
        except Exception as e:
            logger.error(f"Error in TreeVisualizerWidget.set_tree: {e}", exc_info=True)
        
        logger.info(f"TreeVisualizerWidget.set_tree completed.")
    
    def update_node(self, node: TreeNode):
        """Update a specific node in the visualization"""
        self.scene.update_tree(self.scene.root_node)
    
    def get_selected_node_id(self) -> Optional[str]:
        """Get the ID of the currently selected node"""
        selected_items = self.scene.selectedItems()
        for item in selected_items:
            if isinstance(item, EnhancedTreeNodeItem):
                return item.node_id
        return None
    
    def zoom_in(self):
        """Zoom in the view"""
        self.view.scale(1.2, 1.2)
    
    def zoom_out(self):
        """Zoom out the view"""
        self.view.scale(0.8, 0.8)
    
    def zoom_fit(self):
        """Fit the entire tree in the view"""
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
    
    def zoom_100(self):
        """Set zoom to 100%"""
        self.view.resetTransform()
    
    def change_orientation(self, index):
        """Change tree orientation"""
        orientation = self.orientation_combo.itemData(index)
        self.config['orientation'] = orientation
        if self.scene.root_node:
            self.scene.set_configuration(self.config)
            self.scene.update_tree(self.scene.root_node)
            QTimer.singleShot(100, self.zoom_100)
    
    def change_spacing(self, value):
        """Change node spacing"""
        self.config['sibling_distance'] = value
        if self.scene.root_node:
            self.scene.set_configuration(self.config)
            self.scene.update_tree(self.scene.root_node)
    
    def change_level_distance(self, value):
        """Change level distance"""
        self.config['level_distance'] = value
        if self.scene.root_node:
            self.scene.set_configuration(self.config)
            self.scene.update_tree(self.scene.root_node)
    
    def toggle_show_ids(self, state):
        """Toggle node ID display"""
        self.config['show_node_ids'] = state == Qt.Checked
        if self.scene.root_node:
            self.scene.set_configuration(self.config)
            self.scene.update_tree(self.scene.root_node)
    
    def toggle_show_stats(self, state):
        """Toggle statistics display"""
        self.config['show_importance'] = state == Qt.Checked
        if self.scene.root_node:
            self.scene.set_configuration(self.config)
            self.scene.update_tree(self.scene.root_node)
    
    def export_image(self):
        """Export tree as image"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Tree Image", "decision_tree.png", 
            "PNG Files (*.png);;All Files (*)"
        )
        if filename:
            self.save_image(filename)
    
    def save_image(self, filename: str):
        """Save tree visualization as image"""
        try:
            scene_rect = self.scene.itemsBoundingRect()
            
            image = QImage(scene_rect.size().toSize(), QImage.Format_ARGB32)
            image.fill(Qt.white)
            
            painter = QPainter(image)
            painter.setRenderHint(QPainter.Antialiasing)
            
            self.scene.render(painter, QRectF(image.rect()), scene_rect)
            painter.end()
            
            image.save(filename)
            
            logger.info(f"Tree visualization exported to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to export tree image: {e}")
            return False
    
    @property
    def tree_view(self):
        """Compatibility property - return the view"""
        return self.view
    
    def select_node(self, node_id: str):
        """Select a specific node - compatibility method"""
        if node_id in self.scene.node_items:
            item = self.scene.node_items[node_id]
            self.scene.clearSelection()
            item.setSelected(True)
    
    def highlight_node(self, node_id: str, highlight: bool = True):
        """Highlight or unhighlight a specific node"""
        if node_id in self.scene.node_items:
            self.scene.node_items[node_id].highlight(highlight)
    
    def highlight_path(self, path):
        """Highlight a path - compatibility method"""
        for node_id in self.scene.node_items:
            self.highlight_node(node_id, False)
        for node_id in path:
            self.highlight_node(node_id, True)