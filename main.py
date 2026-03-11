#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for Bespoke Utility - Decision Tree Builder
This file initializes the application and launches the main window.

[main -> Main function to start the application -> dependent functions are setup_logging, load_configuration, configure_memory_settings, MainWindow]
"""

import os
import sys
import logging
import math
import time
from pathlib import Path

script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(script_dir))

from utils.logging_utils import setup_logging
from utils.config import load_configuration
from ui.main_window import MainWindow
from utils.memory_management import configure_memory_settings

def fast_out_slow_in_easing(t):
    """Cubic bezier approximation of Android's FastOutSlowInEasing"""
    if t <= 0:
        return 0
    if t >= 1:
        return 1
    return t * t * (3.0 - 2.0 * t)

def calculate_circle_state(elapsed_ms, circle_index):
    """Calculate scale and alpha for a ripple circle at given time"""
    delay = circle_index * 500  # 0ms, 500ms, 1000ms stagger
    cycle_duration = 2000  # 2 seconds per cycle
    
    if elapsed_ms < delay:
        return 0.8, 0.0  # Not started yet
    
    cycle_time = (elapsed_ms - delay) % cycle_duration
    t = cycle_time / cycle_duration
    
    scale = 0.8 + (2.0 - 0.8) * fast_out_slow_in_easing(t)
    
    alpha = 0.7 - (0.7 * t)
    
    return scale, max(0.0, alpha)

class RDSplashWindow:
    """Brand-specific splash screen with ripple animation"""
    
    def __init__(self):
        from PyQt5.QtWidgets import QWidget, QDesktopWidget
        from PyQt5.QtCore import QTimer, Qt, QTime
        from PyQt5.QtGui import QPainter, QColor, QFont, QFontMetrics
        
        self.widget = QWidget()
        self.setup_window()
        
        self.animation_timer = QTimer()
        self.start_time = time.time() * 1000  # Convert to milliseconds
        self.navy_blue = QColor(0, 0, 139)  # #00008B
        
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(16)  # ~60 FPS
        
        self.widget.paintEvent = self.paint_event
        
    def setup_window(self):
        """Setup borderless centered splash window with rounded corners"""
        from PyQt5.QtCore import Qt, QRect
        from PyQt5.QtWidgets import QDesktopWidget
        from PyQt5.QtGui import QRegion
        
        flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        self.widget.setWindowFlags(flags)
        self.widget.setFixedSize(400, 300)
        self.widget.setStyleSheet("background-color: #F8F9FA;")
        
        corner_radius = 12
        from PyQt5.QtCore import QRectF
        from PyQt5.QtGui import QPainterPath
        path = QPainterPath()
        rect = QRectF(0, 0, 400, 300)
        path.addRoundedRect(rect, corner_radius, corner_radius)
        region = QRegion(path.toFillPolygon().toPolygon())
        self.widget.setMask(region)
        
        desktop = QDesktopWidget()
        screen_rect = desktop.availableGeometry()
        window_rect = self.widget.geometry()
        x = (screen_rect.width() - window_rect.width()) // 2
        y = (screen_rect.height() - window_rect.height()) // 2
        self.widget.move(x, y)
        
    def paint_event(self, event):
        """Paint the splash screen with ripple animation"""
        from PyQt5.QtGui import QPainter, QColor, QFont, QFontMetrics
        from PyQt5.QtCore import Qt
        
        painter = QPainter(self.widget)
        painter.setRenderHint(QPainter.Antialiasing)
        
        from PyQt5.QtCore import QRectF
        from PyQt5.QtGui import QPainterPath
        background_color = QColor(248, 249, 250)  # #F8F9FA
        painter.setBrush(background_color)
        painter.setPen(Qt.NoPen)
        
        path = QPainterPath()
        rect = QRectF(self.widget.rect())
        path.addRoundedRect(rect, 12, 12)
        painter.drawPath(path)
        
        center_x = self.widget.width() // 2
        center_y = self.widget.height() // 2
        
        current_time = time.time() * 1000
        elapsed_ms = current_time - self.start_time
        
        base_radius = 40  # 80px diameter -> 40px radius (adapted from 100dp mobile)
        for i in range(3):
            scale, alpha = calculate_circle_state(elapsed_ms, i)
            
            if alpha > 0:
                radius = base_radius * scale
                circle_color = QColor(self.navy_blue)
                circle_color.setAlphaF(0.3 * alpha)  # Match Kotlin: base 30% alpha * animation alpha
                
                painter.setBrush(circle_color)
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(
                    int(center_x - radius),
                    int(center_y - radius),
                    int(radius * 2),
                    int(radius * 2)
                )
        
        font = QFont("Arial", 36, QFont.Bold)  # 36pt equivalent to 48sp mobile
        painter.setFont(font)
        painter.setPen(self.navy_blue)
        
        from PyQt5.QtCore import QRect, Qt
        text_rect = QRect(center_x - 50, center_y - 25, 100, 50)  # Centered rect around center point
        painter.drawText(text_rect, Qt.AlignCenter, "RD")
        
    def update_animation(self):
        """Update animation frame"""
        self.widget.update()  # Trigger repaint
        
    def show(self):
        """Show the splash window"""
        self.widget.show()
        self.widget.raise_()
        self.widget.activateWindow()
        
    def close(self):
        """Close the splash window"""
        self.animation_timer.stop()
        self.widget.close()

def main():
    """Main function to start the application"""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Bespoke Decision Tree Utility")
    
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import Qt
        
        app = QApplication(sys.argv)
        app.setApplicationName("Bespoke Decision Tree Utility")
        app.setOrganizationName("Bespoke Analytics")
        
        splash = None
        splash_start = time.time()
        try:
            splash = RDSplashWindow()
            splash.show()
            app.processEvents()  # Ensure splash is visible
            logger.info("Splash screen displayed")
        except Exception as splash_error:
            logger.warning(f"Failed to show splash screen: {splash_error}")
        
        logger.info("Loading configuration...")
        config = load_configuration()
        app.processEvents()  # Keep splash animated
        logger.debug("Configuration loaded successfully")
        
        logger.info("Configuring memory settings...")
        configure_memory_settings(config)
        app.processEvents()  # Keep splash animated
        logger.debug("Memory settings configured")
        
        if splash:
            min_splash_time = 2.0  # seconds - as requested by user
            elapsed = time.time() - splash_start
            if elapsed < min_splash_time:
                remaining = min_splash_time - elapsed
                end_time = time.time() + remaining
                while time.time() < end_time:
                    app.processEvents()
                    time.sleep(0.016)  # ~60 FPS
        
        if splash:
            splash.close()
            logger.info("Splash screen closed")
        
        logger.info("Creating main window...")
        main_window = MainWindow(config)
        app.processEvents()
        
        main_window.show()
        main_window.setVisible(True)
        main_window.raise_()
        main_window.activateWindow()
        main_window.setWindowState(main_window.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
        
        if os.environ.get('QT_QPA_PLATFORM') != 'wayland':
            main_window.raise_()
            main_window.activateWindow()
            main_window.setFocus()
        
        app.processEvents()
        
        logger.info(f"Main window show called - visible: {main_window.isVisible()}")
        logger.info(f"Main window size: {main_window.size()}")
        logger.info(f"Main window geometry: {main_window.geometry()}")
        logger.info(f"Platform: {app.platformName()}")
        
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}", exc_info=True)
        
        if 'splash' in locals() and splash:
            try:
                splash.close()
            except:
                pass
        
        try:
            from PyQt5.QtWidgets import QMessageBox
            error_box = QMessageBox()
            error_box.setIcon(QMessageBox.Critical)
            error_box.setWindowTitle("Application Error")
            error_box.setText(f"An error occurred while starting the application:\n{str(e)}")
            error_box.setDetailedText(f"{str(e)}")
            error_box.exec_()
        except (ImportError, RuntimeError) as qt_error:
            print(f"ERROR: {str(e)}")
            print(f"Additional Qt error: {qt_error}")
        sys.exit(1)

if __name__ == "__main__":
    main()
