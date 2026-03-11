#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Theme Manager
Manages application themes and modern styling
"""

import logging
from typing import Dict, Any, Optional, List
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QPalette, QColor

logger = logging.getLogger(__name__)

class ThemeManager(QObject):
    """Manages application themes and styling"""
    
    themeChanged = pyqtSignal(str)  # theme_name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_theme = "light"
        self.themes = {
            "light": self._get_light_theme(),
            "dark": self._get_dark_theme()
        }
        
    def _get_light_theme(self) -> Dict[str, Any]:
        """Get light theme configuration"""
        return {
            "name": "Light",
            "colors": {
                "primary": "#3b82f6",
                "primary_hover": "#2563eb",
                "primary_pressed": "#1d4ed8",
                "primary_disabled": "#93c5fd",
                
                "secondary": "#6b7280",
                "secondary_hover": "#4b5563",
                "secondary_pressed": "#374151",
                
                "success": "#10b981",
                "success_hover": "#059669",
                "success_light": "#d1fae5",
                
                "warning": "#f59e0b",
                "warning_hover": "#d97706",
                "warning_light": "#fef3c7",
                
                "danger": "#ef4444",
                "danger_hover": "#dc2626",
                "danger_light": "#fecaca",
                
                "background": "#ffffff",
                "background_secondary": "#f8fafc",
                "background_tertiary": "#f1f5f9",
                
                "surface": "#ffffff",
                "surface_hover": "#f1f5f9",
                "surface_pressed": "#e2e8f0",
                
                "border": "#e2e8f0",
                "border_light": "#f1f5f9",
                "border_dark": "#d1d5db",
                
                "text_primary": "#1e293b",
                "text_secondary": "#475569",
                "text_tertiary": "#64748b",
                "text_disabled": "#94a3b8",
                "text_inverse": "#ffffff",
                
                "overlay": "rgba(0, 0, 0, 0.5)",
                "backdrop": "rgba(0, 0, 0, 0.1)",
            },
            "spacing": {
                "xs": "2px",
                "sm": "4px",
                "md": "8px",
                "lg": "12px",
                "xl": "16px",
                "2xl": "24px",
                "3xl": "32px"
            },
            "borders": {
                "radius_sm": "4px",
                "radius_md": "6px",
                "radius_lg": "8px",
                "radius_xl": "12px"
            },
            "shadows": {
                "sm": "0 1px 2px rgba(0, 0, 0, 0.05)",
                "md": "0 4px 6px rgba(0, 0, 0, 0.07)",
                "lg": "0 10px 15px rgba(0, 0, 0, 0.1)",
                "xl": "0 20px 25px rgba(0, 0, 0, 0.1)"
            },
            "typography": {
                "font_family": "'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif",
                "font_size_xs": "10px",
                "font_size_sm": "11px",
                "font_size_md": "12px",
                "font_size_lg": "14px",
                "font_size_xl": "16px",
                "font_size_2xl": "18px",
                "font_weight_normal": "400",
                "font_weight_medium": "500",
                "font_weight_semibold": "600",
                "font_weight_bold": "700"
            }
        }
        
    def _get_dark_theme(self) -> Dict[str, Any]:
        """Get dark theme configuration"""
        return {
            "name": "Dark",
            "colors": {
                "primary": "#60a5fa",
                "primary_hover": "#3b82f6",
                "primary_pressed": "#2563eb",
                "primary_disabled": "#1e40af",
                
                "secondary": "#9ca3af",
                "secondary_hover": "#d1d5db",
                "secondary_pressed": "#f3f4f6",
                
                "success": "#34d399",
                "success_hover": "#10b981",
                "success_light": "#064e3b",
                
                "warning": "#fbbf24",
                "warning_hover": "#f59e0b",
                "warning_light": "#451a03",
                
                "danger": "#f87171",
                "danger_hover": "#ef4444",
                "danger_light": "#450a0a",
                
                "background": "#0f172a",
                "background_secondary": "#1e293b",
                "background_tertiary": "#334155",
                
                "surface": "#1e293b",
                "surface_hover": "#334155",
                "surface_pressed": "#475569",
                
                "border": "#334155",
                "border_light": "#475569",
                "border_dark": "#1e293b",
                
                "text_primary": "#f1f5f9",
                "text_secondary": "#cbd5e1",
                "text_tertiary": "#94a3b8",
                "text_disabled": "#64748b",
                "text_inverse": "#0f172a",
                
                "overlay": "rgba(0, 0, 0, 0.8)",
                "backdrop": "rgba(0, 0, 0, 0.3)",
            },
            **{k: v for k, v in self._get_light_theme().items() if k != "colors"}
        }
        
    def apply_theme(self, theme_name: str = "light"):
        """Apply a theme to the application"""
        if theme_name not in self.themes:
            logger.warning(f"Theme '{theme_name}' not found, using light theme")
            theme_name = "light"
            
        self.current_theme = theme_name
        theme = self.themes[theme_name]
        
        stylesheet = self._generate_global_stylesheet(theme)
        app = QApplication.instance()
        if app:
            app.setStyleSheet(stylesheet)
            
        self._apply_palette(theme)
        
        self.themeChanged.emit(theme_name)
        logger.info(f"Applied theme: {theme_name}")
        
    def _generate_global_stylesheet(self, theme: Dict[str, Any]) -> str:
        """Generate global application stylesheet"""
        colors = theme["colors"]
        typography = theme["typography"]
        spacing = theme["spacing"]
        borders = theme["borders"]
        shadows = theme["shadows"]
        
        stylesheet = f"""
        /* Global Application Styles */
        QMainWindow {{
            background-color: {colors['background']};
            color: {colors['text_primary']};
            font-family: {typography['font_family']};
            font-size: {typography['font_size_md']};
        }}
        
        /* Modern Button Styles */
        QPushButton {{
            background-color: {colors['primary']};
            color: {colors['text_inverse']};
            border: none;
            border-radius: {borders['radius_md']};
            padding: {spacing['md']} {spacing['lg']};
            font-weight: {typography['font_weight_semibold']};
            font-size: {typography['font_size_sm']};
            font-family: {typography['font_family']};
            min-height: 20px;
        }}
        
        QPushButton:hover {{
            background-color: {colors['primary_hover']};
        }}
        
        QPushButton:pressed {{
            background-color: {colors['primary_pressed']};
        }}
        
        QPushButton:disabled {{
            background-color: {colors['primary_disabled']};
            color: {colors['text_disabled']};
        }}
        
        QPushButton.secondary {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border']};
        }}
        
        QPushButton.secondary:hover {{
            background-color: {colors['surface_hover']};
        }}
        
        QPushButton.secondary:pressed {{
            background-color: {colors['surface_pressed']};
        }}
        
        QPushButton.danger {{
            background-color: {colors['danger']};
            color: {colors['text_inverse']};
        }}
        
        QPushButton.danger:hover {{
            background-color: {colors['danger_hover']};
        }}
        
        QPushButton.success {{
            background-color: {colors['success']};
            color: {colors['text_inverse']};
        }}
        
        QPushButton.success:hover {{
            background-color: {colors['success_hover']};
        }}
        
        /* Input Field Styles */
        QLineEdit, QTextEdit, QPlainTextEdit {{
            background-color: {colors['surface']};
            border: 1px solid {colors['border']};
            border-radius: {borders['radius_sm']};
            padding: {spacing['md']};
            color: {colors['text_primary']};
            font-size: {typography['font_size_sm']};
            selection-background-color: {colors['primary']};
        }}
        
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
            border-color: {colors['primary']};
            outline: none;
        }}
        
        QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {{
            background-color: {colors['background_tertiary']};
            color: {colors['text_disabled']};
        }}
        
        /* ComboBox Styles */
        QComboBox {{
            background-color: {colors['surface']};
            border: 1px solid {colors['border']};
            border-radius: {borders['radius_sm']};
            padding: {spacing['md']};
            color: {colors['text_primary']};
            font-size: {typography['font_size_sm']};
            min-width: 80px;
        }}
        
        QComboBox:hover {{
            border-color: {colors['primary']};
        }}
        
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        
        QComboBox::down-arrow {{
            width: 12px;
            height: 12px;
        }}
        
        QComboBox QAbstractItemView {{
            background-color: {colors['surface']};
            border: 1px solid {colors['border']};
            border-radius: {borders['radius_md']};
            selection-background-color: {colors['primary']};
            color: {colors['text_primary']};
        }}
        
        /* SpinBox Styles */
        QSpinBox, QDoubleSpinBox {{
            background-color: {colors['surface']};
            border: 1px solid {colors['border']};
            border-radius: {borders['radius_sm']};
            padding: {spacing['md']};
            color: {colors['text_primary']};
            font-size: {typography['font_size_sm']};
        }}
        
        QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {colors['primary']};
        }}
        
        /* GroupBox Styles */
        QGroupBox {{
            font-weight: {typography['font_weight_semibold']};
            font-size: {typography['font_size_md']};
            color: {colors['text_primary']};
            border: 2px solid {colors['border']};
            border-radius: {borders['radius_lg']};
            background-color: {colors['surface']};
            padding-top: {spacing['xl']};
            margin-top: {spacing['md']};
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: {spacing['lg']};
            padding: 0 {spacing['md']} 0 {spacing['md']};
            background-color: {colors['background']};
        }}
        
        /* Tab Widget Styles */
        QTabWidget::pane {{
            border: 1px solid {colors['border']};
            border-radius: {borders['radius_md']};
            background-color: {colors['surface']};
        }}
        
        QTabBar::tab {{
            background-color: {colors['background_tertiary']};
            color: {colors['text_secondary']};
            border: 1px solid {colors['border']};
            border-bottom: none;
            border-radius: {borders['radius_sm']} {borders['radius_sm']} 0 0;
            padding: {spacing['md']} {spacing['lg']};
            margin-right: 2px;
            font-size: {typography['font_size_sm']};
            font-weight: {typography['font_weight_medium']};
        }}
        
        QTabBar::tab:selected {{
            background-color: {colors['surface']};
            color: {colors['text_primary']};
            font-weight: {typography['font_weight_semibold']};
        }}
        
        QTabBar::tab:hover:!selected {{
            background-color: {colors['surface_hover']};
        }}
        
        /* Table Widget Styles */
        QTableWidget {{
            background-color: {colors['surface']};
            border: 1px solid {colors['border']};
            border-radius: {borders['radius_lg']};
            gridline-color: {colors['border_light']};
            font-size: {typography['font_size_sm']};
            selection-background-color: {colors['primary']};
            selection-color: {colors['text_inverse']};
        }}
        
        QHeaderView::section {{
            background-color: {colors['background_tertiary']};
            color: {colors['text_primary']};
            border: none;
            border-bottom: 1px solid {colors['border']};
            border-right: 1px solid {colors['border']};
            padding: {spacing['md']} {spacing['lg']};
            font-weight: {typography['font_weight_semibold']};
            font-size: {typography['font_size_xs']};
        }}
        
        QTableWidget::item {{
            padding: {spacing['md']} {spacing['lg']};
            border: none;
        }}
        
        QTableWidget::item:hover {{
            background-color: {colors['surface_hover']};
        }}
        
        /* Scroll Bar Styles */
        QScrollBar:vertical {{
            background-color: {colors['background_tertiary']};
            width: 12px;
            border-radius: 6px;
            margin: 0;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {colors['border_dark']};
            border-radius: 6px;
            min-height: 20px;
            margin: 2px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {colors['secondary']};
        }}
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
        }}
        
        QScrollBar:horizontal {{
            background-color: {colors['background_tertiary']};
            height: 12px;
            border-radius: 6px;
            margin: 0;
        }}
        
        QScrollBar::handle:horizontal {{
            background-color: {colors['border_dark']};
            border-radius: 6px;
            min-width: 20px;
            margin: 2px;
        }}
        
        QScrollBar::handle:horizontal:hover {{
            background-color: {colors['secondary']};
        }}
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            border: none;
            background: none;
        }}
        
        /* Progress Bar Styles */
        QProgressBar {{
            border: 1px solid {colors['border']};
            border-radius: {borders['radius_sm']};
            background-color: {colors['background_tertiary']};
            text-align: center;
            color: {colors['text_primary']};
            font-size: {typography['font_size_xs']};
            font-weight: {typography['font_weight_medium']};
        }}
        
        QProgressBar::chunk {{
            background-color: {colors['primary']};
            border-radius: {borders['radius_sm']};
        }}
        
        /* Splitter Styles */
        QSplitter::handle {{
            background-color: {colors['border']};
        }}
        
        QSplitter::handle:horizontal {{
            width: 2px;
        }}
        
        QSplitter::handle:vertical {{
            height: 2px;
        }}
        
        QSplitter::handle:hover {{
            background-color: {colors['primary']};
        }}
        
        /* Frame Styles */
        QFrame[frameShape="4"] {{ /* HLine */
            background-color: {colors['border']};
            max-height: 1px;
        }}
        
        QFrame[frameShape="5"] {{ /* VLine */
            background-color: {colors['border']};
            max-width: 1px;
        }}
        
        /* Label Styles */
        QLabel {{
            color: {colors['text_primary']};
            font-size: {typography['font_size_sm']};
        }}
        
        /* CheckBox and RadioButton Styles */
        QCheckBox, QRadioButton {{
            color: {colors['text_primary']};
            font-size: {typography['font_size_sm']};
            spacing: {spacing['md']};
        }}
        
        QCheckBox::indicator, QRadioButton::indicator {{
            width: 16px;
            height: 16px;
        }}
        
        QCheckBox::indicator:unchecked {{
            border: 1px solid {colors['border']};
            background-color: {colors['surface']};
            border-radius: 3px;
        }}
        
        QCheckBox::indicator:checked {{
            border: 1px solid {colors['primary']};
            background-color: {colors['primary']};
            border-radius: 3px;
        }}
        
        QRadioButton::indicator:unchecked {{
            border: 1px solid {colors['border']};
            background-color: {colors['surface']};
            border-radius: 8px;
        }}
        
        QRadioButton::indicator:checked {{
            border: 1px solid {colors['primary']};
            background-color: {colors['primary']};
            border-radius: 8px;
        }}
        """
        
        return stylesheet
        
    def _apply_palette(self, theme: Dict[str, Any]):
        """Apply QPalette for native widget colors"""
        colors = theme["colors"]
        app = QApplication.instance()
        if not app:
            return
            
        palette = QPalette()
        
        palette.setColor(QPalette.Window, QColor(colors["background"]))
        palette.setColor(QPalette.WindowText, QColor(colors["text_primary"]))
        palette.setColor(QPalette.Base, QColor(colors["surface"]))
        palette.setColor(QPalette.AlternateBase, QColor(colors["background_tertiary"]))
        palette.setColor(QPalette.ToolTipBase, QColor(colors["surface"]))
        palette.setColor(QPalette.ToolTipText, QColor(colors["text_primary"]))
        palette.setColor(QPalette.Text, QColor(colors["text_primary"]))
        palette.setColor(QPalette.Button, QColor(colors["surface"]))
        palette.setColor(QPalette.ButtonText, QColor(colors["text_primary"]))
        palette.setColor(QPalette.BrightText, QColor(colors["text_inverse"]))
        palette.setColor(QPalette.Link, QColor(colors["primary"]))
        palette.setColor(QPalette.Highlight, QColor(colors["primary"]))
        palette.setColor(QPalette.HighlightedText, QColor(colors["text_inverse"]))
        
        app.setPalette(palette)
        
    def get_current_theme(self) -> str:
        """Get current theme name"""
        return self.current_theme
        
    def get_theme_colors(self, theme_name: Optional[str] = None) -> Dict[str, str]:
        """Get color palette for a theme"""
        theme_name = theme_name or self.current_theme
        if theme_name in self.themes:
            return self.themes[theme_name]["colors"]
        return {}
        
    def get_available_themes(self) -> List[str]:
        """Get list of available theme names"""
        return list(self.themes.keys())
        
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        new_theme = "dark" if self.current_theme == "light" else "light"
        self.apply_theme(new_theme)
        
    def is_dark_theme(self) -> bool:
        """Check if current theme is dark"""
        return self.current_theme == "dark"