#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Status Bar Component
Modern status bar with context information, busy indicator, and system status
"""

import logging
from typing import Optional, Dict, Any
from PyQt5.QtWidgets import (
    QStatusBar, QLabel, QProgressBar, QWidget, QHBoxLayout, 
    QFrame, QPushButton, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QRect
from PyQt5.QtGui import QFont, QMovie, QPainter, QPen, QColor
import math

logger = logging.getLogger(__name__)

class BusyIndicator(QWidget):
    """Custom busy indicator with spinning animation"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)
        
        self.angle = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_rotation)
        self.is_spinning = False
        
    def start_animation(self):
        """Start the spinning animation"""
        if not self.is_spinning:
            self.is_spinning = True
            self.timer.start(50)  # 20 FPS
            self.show()
            
    def stop_animation(self):
        """Stop the spinning animation"""
        if self.is_spinning:
            self.is_spinning = False
            self.timer.stop()
            self.hide()
            
    def update_rotation(self):
        """Update rotation angle"""
        self.angle = (self.angle + 15) % 360
        self.update()
        
    def paintEvent(self, event):
        """Custom paint event for spinning indicator"""
        if not self.is_spinning:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        pen = QPen(QColor(59, 130, 246), 2, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)
        
        center_x = self.width() // 2
        center_y = self.height() // 2
        radius = min(center_x, center_y) - 2
        
        for i in range(8):
            angle = (self.angle + i * 45) % 360
            opacity = 1.0 - (i * 0.1)
            
            pen.setColor(QColor(59, 130, 246, int(255 * opacity)))
            painter.setPen(pen)
            
            start_angle = angle * 16  # QPainter uses 1/16th of a degree
            span_angle = 30 * 16
            
            painter.drawArc(
                center_x - radius, center_y - radius,
                radius * 2, radius * 2,
                start_angle, span_angle
            )


class MemoryIndicator(QWidget):
    """Memory usage indicator widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 16)
        self.memory_percent = 0
        self.memory_text = "0%"
        
    def update_memory(self, used_mb: float, total_mb: float):
        """Update memory usage display"""
        if total_mb > 0:
            self.memory_percent = (used_mb / total_mb) * 100
            self.memory_text = f"{used_mb:.1f}GB/{total_mb:.1f}GB"
        else:
            self.memory_percent = 0
            self.memory_text = "N/A"
        self.update()
        
    def paintEvent(self, event):
        """Custom paint event for memory bar"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        bg_color = QColor(241, 245, 249)
        painter.fillRect(self.rect(), bg_color)
        
        if self.memory_percent > 0:
            if self.memory_percent < 50:
                bar_color = QColor(34, 197, 94)  # Green
            elif self.memory_percent < 80:
                bar_color = QColor(251, 191, 36)  # Yellow
            else:
                bar_color = QColor(239, 68, 68)  # Red
                
            bar_width = int((self.width() * self.memory_percent) / 100)
            painter.fillRect(0, 0, bar_width, self.height(), bar_color)
            
        painter.setPen(QPen(QColor(226, 232, 240), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))


class EnhancedStatusBar(QStatusBar):
    """Enhanced status bar with modern design and comprehensive information"""
    
    memoryWarning = pyqtSignal(float)  # memory_percent
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.context_label = None
        self.status_label = None
        self.busy_indicator = None
        self.memory_indicator = None
        self.time_label = None
        
        self.current_context = "workflow"
        self.is_busy = False
        self.busy_message = ""
        
        self.setup_status_bar()
        self.setup_styling()
        self.start_time_updates()
        
        logger.debug("EnhancedStatusBar initialized")
        
    def setup_status_bar(self):
        """Setup status bar components"""
        self.context_label = QLabel("🏠 Workflow")
        self.context_label.setStyleSheet("""
            QLabel {
                color: #374151;
                font-weight: 600;
                font-size: 11px;
                padding: 2px 8px;
                background-color: #f1f5f9;
                border-radius: 4px;
                margin: 2px;
            }
        """)
        self.addWidget(self.context_label)
        
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.VLine)
        separator1.setStyleSheet("QFrame { color: #e2e8f0; margin: 4px 2px; }")
        self.addWidget(separator1)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #6b7280;
                font-size: 11px;
                padding: 2px 8px;
            }
        """)
        self.addWidget(self.status_label, 1)  # Stretch factor
        
        self.busy_indicator = BusyIndicator()
        self.addWidget(self.busy_indicator)
        
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.VLine)
        separator2.setStyleSheet("QFrame { color: #e2e8f0; margin: 4px 2px; }")
        self.addPermanentWidget(separator2)
        
        memory_widget = QWidget()
        memory_layout = QHBoxLayout(memory_widget)
        memory_layout.setContentsMargins(4, 2, 4, 2)
        memory_layout.setSpacing(4)
        
        memory_label = QLabel("Memory:")
        memory_label.setStyleSheet("QLabel { color: #6b7280; font-size: 10px; }")
        memory_layout.addWidget(memory_label)
        
        self.memory_indicator = MemoryIndicator()
        memory_layout.addWidget(self.memory_indicator)
        
        self.addPermanentWidget(memory_widget)
        
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.VLine)
        separator3.setStyleSheet("QFrame { color: #e2e8f0; margin: 4px 2px; }")
        self.addPermanentWidget(separator3)
        
        self.time_label = QLabel()
        self.time_label.setStyleSheet("""
            QLabel {
                color: #6b7280;
                font-size: 10px;
                font-family: 'Courier New', monospace;
                padding: 2px 8px;
            }
        """)
        self.addPermanentWidget(self.time_label)
        
    def setup_styling(self):
        """Apply modern styling to status bar"""
        self.setStyleSheet("""
            QStatusBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #f8fafc);
                border-top: 1px solid #e2e8f0;
                color: #374151;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
                padding: 2px;
            }
            
            QStatusBar::item {
                border: none;
            }
        """)
        
    def start_time_updates(self):
        """Start periodic time updates"""
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)  # Update every second
        self.update_time()
        
    def update_time(self):
        """Update time display"""
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.setText(current_time)
        
    def set_context(self, context: str, context_data: Optional[Dict[str, Any]] = None):
        """Update context display"""
        self.current_context = context
        
        context_info = {
            'workflow': ('🏠', 'Workflow'),
            'data': ('📊', 'Data Analysis'),
            'model': ('🌳', 'Model Builder'),
            'transform': ('⚡', 'Data Transform'),
            'evaluation': ('📈', 'Model Evaluation'),
            'export': ('📤', 'Export')
        }
        
        icon, label = context_info.get(context, ('📋', context.title()))
        
        if context_data:
            if 'name' in context_data:
                label += f": {context_data['name']}"
            elif 'file' in context_data:
                label += f": {context_data['file']}"
                
        self.context_label.setText(f"{icon} {label}")
        
    def show_message(self, message: str, timeout: int = 0):
        """Show status message"""
        self.status_label.setText(message)
        
        if timeout > 0:
            QTimer.singleShot(timeout, lambda: self.status_label.setText("Ready"))
            
    def show_busy(self, message: str = "Processing..."):
        """Show busy state with animation"""
        if not self.is_busy:
            self.is_busy = True
            self.busy_message = message
            self.status_label.setText(f"🔄 {message}")
            self.busy_indicator.start_animation()
            
    def hide_busy(self, final_message: str = "Ready"):
        """Hide busy state"""
        if self.is_busy:
            self.is_busy = False
            self.busy_message = ""
            self.busy_indicator.stop_animation()
            self.status_label.setText(final_message)
            
    def update_memory_usage(self, used_mb: float, total_mb: float):
        """Update memory usage display"""
        used_gb = used_mb / 1024
        total_gb = total_mb / 1024
        
        self.memory_indicator.update_memory(used_gb, total_gb)
        
        if total_mb > 0:
            usage_percent = (used_mb / total_mb) * 100
            if usage_percent > 80:
                self.memoryWarning.emit(usage_percent)
                
    def show_progress(self, current: int, total: int, message: str = ""):
        """Show progress information"""
        if total > 0:
            percent = int((current / total) * 100)
            progress_message = f"{message} {percent}%" if message else f"Progress: {percent}%"
            self.show_message(progress_message)
        else:
            self.show_message(message or "Processing...")
            
    def show_error(self, message: str, timeout: int = 5000):
        """Show error message with red styling"""
        self.status_label.setStyleSheet("""
            QLabel {
                color: #dc2626;
                font-size: 11px;
                padding: 2px 8px;
                font-weight: 600;
            }
        """)
        self.show_message(f"❌ {message}", timeout)
        
        QTimer.singleShot(timeout, self.reset_status_styling)
        
    def show_success(self, message: str, timeout: int = 3000):
        """Show success message with green styling"""
        self.status_label.setStyleSheet("""
            QLabel {
                color: #059669;
                font-size: 11px;
                padding: 2px 8px;
                font-weight: 600;
            }
        """)
        self.show_message(f"✅ {message}", timeout)
        
        QTimer.singleShot(timeout, self.reset_status_styling)
        
    def show_warning(self, message: str, timeout: int = 4000):
        """Show warning message with yellow styling"""
        self.status_label.setStyleSheet("""
            QLabel {
                color: #d97706;
                font-size: 11px;
                padding: 2px 8px;
                font-weight: 600;
            }
        """)
        self.show_message(f"⚠️ {message}", timeout)
        
        QTimer.singleShot(timeout, self.reset_status_styling)
        
    def reset_status_styling(self):
        """Reset status label styling to default"""
        self.status_label.setStyleSheet("""
            QLabel {
                color: #6b7280;
                font-size: 11px;
                padding: 2px 8px;
            }
        """)
        
    def get_current_status(self) -> Dict[str, Any]:
        """Get current status information"""
        return {
            'context': self.current_context,
            'message': self.status_label.text(),
            'is_busy': self.is_busy,
            'busy_message': self.busy_message
        }
        
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'time_timer'):
            self.time_timer.stop()
            
        if self.busy_indicator:
            self.busy_indicator.stop_animation()
            
        logger.debug("EnhancedStatusBar cleanup completed")