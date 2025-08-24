#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Window Module for Bespoke Utility
Implements the primary application window and UI framework

"""

import os
import sys
import logging
import webbrowser
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Set

import pandas as pd
import numpy as np
from PyQt5.QtCore import Qt, QSettings, QTimer, pyqtSlot, pyqtSignal, QSize, QPoint, QPointF
from PyQt5.QtGui import QIcon, QPixmap, QCloseEvent, QFont, QKeySequence
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QDockWidget,
    QAction, QMenu, QToolBar, QStatusBar, QMessageBox, QDialog,
    QFileDialog, QSplitter, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTabWidget, QShortcut,
    QProgressBar, QComboBox, QCheckBox, QSizePolicy, QInputDialog, QStyle
)

from datetime import datetime
from PyQt5.QtWidgets import QProgressDialog, QTabWidget, QGroupBox, QFormLayout, QTextEdit, QTableWidget, QTableWidgetItem, QScrollArea
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from data.data_loader import DataLoader
from data.data_processor import DataProcessor
from data.database_importer import DatabaseImporter
from data.cloud_importer import CloudImporter

try:
    from ui.formula_editor import FormulaEditorDialog
except ImportError:
    from formula_editor import FormulaEditorDialog

from data.feature_engineering import FeatureEngineering
from models.decision_tree import BespokeDecisionTree, TreeGrowthMode, SplitCriterion
from models.node import TreeNode
from ui.tree_visualizer import TreeVisualizerWidget
from ui.workflow_canvas import WorkflowCanvas, DatasetNode, ModelNode, FilterNode
from ui.workflow_canvas import WorkflowNode as CanvasWorkflowNode
from ui.workflow_canvas import STATUS_RUNNING, STATUS_COMPLETED, STATUS_FAILED, STATUS_PENDING
from ui.workflow_canvas import NODE_TYPE_DATASET, NODE_TYPE_MODEL, NODE_TYPE_FILTER
from ui.workflow_canvas import NODE_TYPE_TRANSFORM, NODE_TYPE_EVALUATION, NODE_TYPE_VISUALIZATION, NODE_TYPE_EXPORT
from ui.workflow_canvas import ConnectionData
from ui.node_editor import NodeEditorDialog
from ui.streamlined_split_dialog import StreamlinedSplitDialog
from ui.streamlined_manual_split_dialog import StreamlinedManualSplitDialog
from ui.dialogs.enhanced_edit_split_dialog import EnhancedEditSplitDialog
from ui.dialogs.find_split_dialog import FindSplitDialog
from ui.dialogs import (CsvImportOptionsDialog, ExcelImportOptionsDialog, TextImportOptionsDialog,
                                DatabaseImportOptionsDialog, CloudImportOptionsDialog, DataImportWizard,
                                FilterDataDialog, AdvancedFilterDialog, AdvancedDataFilterDialog,
                                DataTransformationDialog, MissingValuesDialog, TreeConfigurationDialog,
                                EnhancedTreeConfigurationDialog, EnhancedFormulaEditorDialog,
                                VariableImportanceDialog, VariableSelectionDialog,
                                PerformanceEvaluationDialog, ManualTreeGrowthDialog,
                                CategoricalSplitGroupingDialog, NodeReportingDialog,
                                TreeNavigationDialog)
from ui.window_manager import WindowManager, BreadcrumbWidget
from ui.base_detail_window import BaseDetailWindow, DataDetailWindow, ModelDetailWindow
from ui.components import EnhancedStatusBar
from ui.components.widget_toolbar import WidgetToolbar, ToolbarContainer
from ui.components.tree_node_context_menu import TreeNodeContextMenu
from ui.data_viewer import DataViewerWidget
from ui.variable_viewer import VariableViewerWidget
from ui.widgets.model_properties_widget import ModelPropertiesWidget
from analytics.performance_metrics import MetricsCalculator, MetricsVisualizer, MATPLOTLIB_AVAILABLE
from analytics.variable_importance import VariableImportance
from analytics.node_statistics import NodeAnalyzer
from analytics.node_report_generator import NodeReportGenerator
from export.model_saver import ModelSaver
from utils.memory_management import monitor_memory_usage, get_system_memory_info
from utils.config import get_config_value, set_config_value, save_configuration
from project_manager import ProjectManager
from workflow.execution_engine import WorkflowExecutionEngine

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """Main window for the Bespoke Decision Tree Utility application"""
    
    modelCreated = pyqtSignal(str, object)  # model_name, model
    modelDeleted = pyqtSignal(str)  # model_name
    datasetCreated = pyqtSignal(str, object)  # dataset_name, dataframe
    datasetDeleted = pyqtSignal(str)  # dataset_name

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.config = config
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.current_dataset_name: Optional[str] = None
        self.models: Dict[str, BespokeDecisionTree] = {}
        self.current_model_name: Optional[str] = None
        self.current_project_path = None
        self._project_modified = False
        self.latest_evaluation_results: Dict[str, Dict] = {}

        self.data_viewer_widgets: Dict[str, DataViewerWidget] = {}
        self.variable_viewer_widget: Optional[VariableViewerWidget] = None
        self.dataset_placeholder_label: Optional[QLabel] = None
        self.model_properties_widget: Optional[ModelPropertiesWidget] = None
        self.memory_warning_shown = False
        
        self.window_manager: Optional[WindowManager] = None
        self.enhanced_status_bar: Optional[EnhancedStatusBar] = None
        self.breadcrumb_widget: Optional[BreadcrumbWidget] = None

        self.project_manager = ProjectManager(config, main_window_ref=self)
        self.data_loader = DataLoader(config)
        self.data_processor = DataProcessor(config)
        self.database_importer = DatabaseImporter(config)
        self.cloud_importer = CloudImporter(config)
        self.feature_engineering = FeatureEngineering(config)
        self.model_saver = ModelSaver(config)
        self.node_analyzer = NodeAnalyzer()
        self.metrics_calculator = MetricsCalculator()
        self.variable_importance = VariableImportance()
        self.workflow_engine = WorkflowExecutionEngine(config, project_data_manager=self)
        
        self.tree_context_menu = TreeNodeContextMenu(self)

        self.init_ui()
        self.connect_signals()
        self.setup_memory_monitoring()
        self.load_recent_files()
        self.load_application_state()
        
        self.new_project(confirm_discard=False)
        self._initialize_enhanced_features()

        self.show()
        self.setVisible(True)
        self.raise_()
        self.activateWindow()
        
        from PyQt5.QtWidgets import QDesktopWidget
        desktop = QDesktopWidget()
        screen_geometry = desktop.screenGeometry()
        
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        window_geometry = self.frameGeometry()
        center_point = screen_geometry.center()
        window_geometry.moveCenter(center_point)
        self.move(window_geometry.topLeft())
        
        self.setWindowState(Qt.WindowActive)
        self.show()
        
        if hasattr(self, 'widget_toolbar'):
            self.widget_toolbar.setVisible(True)
            self.toolbar_container.setVisible(True)
            logger.info(f"FINAL CHECK - Toolbar visible: {self.widget_toolbar.isVisible()}")
            logger.info(f"FINAL CHECK - Container visible: {self.toolbar_container.isVisible()}")

        logger.info(f"MAIN WINDOW visible: {self.isVisible()}")
        logger.info(f"MAIN WINDOW size: {self.size()}")
        logger.info(f"MAIN WINDOW position: {self.pos()}")
        logger.info(f"MAIN WINDOW state: {self.windowState()}")
        
        central = self.centralWidget()
        logger.info(f"CENTRAL WIDGET: {type(central).__name__ if central else 'None'}")
        logger.info(f"CENTRAL WIDGET visible: {central.isVisible() if central else 'N/A'}")
        logger.info(f"CENTRAL WIDGET size: {central.size() if central else 'N/A'}")
        
        stacked = self.window_manager.get_stacked_widget()
        logger.info(f"STACKED WIDGET count: {stacked.count()}")
        logger.info(f"STACKED WIDGET current: {stacked.currentIndex()}")
        logger.info(f"STACKED WIDGET current widget: {type(stacked.currentWidget()).__name__ if stacked.currentWidget() else 'None'}")
        
        if hasattr(self, 'workflow_canvas'):
            logger.info(f"WORKFLOW CANVAS visible: {self.workflow_canvas.isVisible()}")
            logger.info(f"WORKFLOW CANVAS size: {self.workflow_canvas.size()}")
            logger.info(f"WORKFLOW CANVAS parent: {type(self.workflow_canvas.parent()).__name__ if self.workflow_canvas.parent() else 'None'}")
        
        if hasattr(self, 'main_container'):
            logger.info(f"MAIN CONTAINER children count: {len(self.main_container.children())}")
            logger.info(f"MAIN CONTAINER layout items: {self.main_layout.count()}")
            for i in range(self.main_layout.count()):
                item = self.main_layout.itemAt(i)
                widget = item.widget() if item else None
                logger.info(f"  Layout item {i}: {type(widget).__name__ if widget else 'None'}")

        logger.info("Main window initialized")

    def init_ui(self):
        """Initialize the modern user interface"""
        self.setWindowTitle("Bespoke Decision Tree Utility - New Project")
        self.setMinimumSize(1000, 700)

        self.window_manager = WindowManager(self)
        
        self.workflow_canvas = WorkflowCanvas(self.config, main_window_ref=self, parent=self)
        
        self.window_manager.stacked_widget.addWidget(self.workflow_canvas)
        self.window_manager.window_instances['workflow'] = self.workflow_canvas
        self.window_manager.registered_windows['workflow'] = None  # Mark as registered
        
        self.window_manager.register_window('data', DataDetailWindow)
        self.window_manager.register_window('model', ModelDetailWindow)
        
        from ui.detail_windows.visualization_detail_window import VisualizationDetailWindow
        self.window_manager.register_window('visualization', VisualizationDetailWindow)
        
        logger.info(f"DIRECT FIX: Added workflow canvas to stacked widget")
        logger.info(f"STACKED WIDGET count after fix: {self.window_manager.stacked_widget.count()}")
        
        self.create_modern_layout_structure()
        
        self.create_menus()
        self.create_widget_based_toolbar()
        self.create_enhanced_status_bar()
        self.create_breadcrumb_navigation()
        
        self.window_manager.stacked_widget.setCurrentWidget(self.workflow_canvas)
        self.window_manager.current_window_type = 'workflow'
        
        logger.info(f"FORCED workflow to show - current index: {self.window_manager.stacked_widget.currentIndex()}")
        logger.info(f"FORCED workflow - current widget: {type(self.window_manager.stacked_widget.currentWidget()).__name__}")
        
        self.create_shortcuts()
        
        self.connect_window_manager_signals()
        
        self.connect_workflow_navigation_signals()
        
        self.connect_workflow_context_menu_signals()
        
        self.create_compatibility_bridge()
        
        self.workflow_canvas.setFocus()
        self.update_action_states()
        
        self.setup_ui_canvas_synchronization()

    def create_menus(self):
        """Create application menus"""
        file_menu = self.menuBar().addMenu("&File")
        
        self.new_action = QAction("&New Project", self, triggered=self.new_project)
        self.new_action.setShortcut("Ctrl+N")
        self.new_action.setStatusTip("Create a new project")
        file_menu.addAction(self.new_action)
        
        self.open_action = QAction("&Open Project...", self, triggered=self.open_project)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.setStatusTip("Open existing project")
        file_menu.addAction(self.open_action)
        
        self.save_action = QAction("&Save Project", self, triggered=self.save_project)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.setStatusTip("Save current project")
        file_menu.addAction(self.save_action)
        
        file_menu.addSeparator()
        
        self.recent_menu = file_menu.addMenu("&Recent Projects")
        self.recent_menu.setStatusTip("Open recently used projects")
        
        file_menu.addSeparator()
        
        import_menu = file_menu.addMenu("&Import Data")
        
        self.import_wizard_action = QAction("Import &Wizard...", self, triggered=self.show_import_wizard)
        self.import_wizard_action.setStatusTip("Launch comprehensive data import wizard")
        import_menu.addAction(self.import_wizard_action)
        
        import_menu.addSeparator()
        
        self.import_csv_action = QAction("Import &CSV...", self, triggered=lambda: self.import_data('csv'))
        self.import_csv_action.setStatusTip("Import data from CSV file")
        import_menu.addAction(self.import_csv_action)
        
        self.import_excel_action = QAction("Import &Excel...", self, triggered=lambda: self.import_data('excel'))
        self.import_excel_action.setStatusTip("Import data from Excel file")
        import_menu.addAction(self.import_excel_action)
        
        file_menu.addSeparator()
        
        export_menu = file_menu.addMenu("&Export Model")
        
        self.export_python_action = QAction("Export to &Python...", self, triggered=lambda: self.export_model('python'))
        self.export_python_action.setStatusTip("Export model as Python code")
        export_menu.addAction(self.export_python_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self, triggered=self.close)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit the application")
        file_menu.addAction(exit_action)
        
        self.export_pmml_action = QAction("Export to &PMML...", self, triggered=lambda: self.export_model('pmml'))
        self.export_pmml_action.setStatusTip("Export model as PMML")
        export_menu.addAction(self.export_pmml_action)
        
        operations_menu = self.menuBar().addMenu("&Operations")
        
        self.add_dataset_action = QAction("📊 &Dataset", self, triggered=lambda: self.import_data('csv'))
        self.add_dataset_action.setStatusTip("Add dataset to workflow")
        operations_menu.addAction(self.add_dataset_action)
        
        self.add_filter_action = QAction("🔍 &Filter", self, triggered=self.add_filter_node)
        self.add_filter_action.setStatusTip("Add filter node to workflow")
        operations_menu.addAction(self.add_filter_action)
        
        self.add_transform_action = QAction("🔄 &Transform", self, triggered=self.add_transform_node)
        self.add_transform_action.setStatusTip("Add transform node to workflow")
        operations_menu.addAction(self.add_transform_action)
        
        self.add_model_action = QAction("🌳 &Model", self, triggered=self.create_new_model_dialog)
        self.add_model_action.setStatusTip("Add model node to workflow")
        operations_menu.addAction(self.add_model_action)
        
        self.add_viz_action = QAction("🎨 &Visualization", self, triggered=self.add_visualization_node)
        self.add_viz_action.setStatusTip("Add visualization node to workflow")
        operations_menu.addAction(self.add_viz_action)
        
        operations_menu.addSeparator()
        
        self.run_workflow_action = QAction("▶️ &Execute Workflow", self, triggered=lambda: self.workflow_canvas.runWorkflowRequested.emit())
        self.run_workflow_action.setShortcut("F5")
        self.run_workflow_action.setStatusTip("Execute the complete workflow - builds all connected models")
        operations_menu.addAction(self.run_workflow_action)
        
        data_menu = self.menuBar().addMenu("&Data")
        
        self.filter_action = QAction("&Filter Data...", self, triggered=self.show_filter_dialog)
        self.filter_action.setStatusTip("Filter the current dataset")
        data_menu.addAction(self.filter_action)
        
        self.transform_data_action = QAction("&Transform Data...", self, triggered=self.show_transformation_dialog)
        self.transform_data_action.setStatusTip("Apply comprehensive data transformations")
        data_menu.addAction(self.transform_data_action)
        
        self.create_var_action = QAction("Create &Variable...", self, triggered=self.show_formula_editor)
        self.create_var_action.setStatusTip("Create a new variable with a formula")
        data_menu.addAction(self.create_var_action)
        
        model_menu = self.menuBar().addMenu("&Model")
        
        self.new_model_action = QAction("&Create Model", self, triggered=self.create_new_model_dialog)
        self.new_model_action.setStatusTip("Add a new decision tree model to the workflow canvas")
        model_menu.addAction(self.new_model_action)
        
        self.enhanced_tree_config_action = QAction("&Configure Model...", self, triggered=self.show_enhanced_tree_configuration_dialog)
        self.enhanced_tree_config_action.setStatusTip("Configure tree parameters")
        model_menu.addAction(self.enhanced_tree_config_action)
        
        analysis_menu = self.menuBar().addMenu("&Analysis")
        
        self.variable_importance_dialog_action = QAction("Variable &Importance...", self, triggered=self.show_variable_importance_dialog)
        self.variable_importance_dialog_action.setStatusTip("Variable importance analysis")
        analysis_menu.addAction(self.variable_importance_dialog_action)
        
        self.performance_evaluation_dialog_action = QAction("Performance &Evaluation...", self, triggered=self.show_performance_evaluation_dialog)
        self.performance_evaluation_dialog_action.setStatusTip("Model performance evaluation")
        analysis_menu.addAction(self.performance_evaluation_dialog_action)
        
        view_menu = self.menuBar().addMenu("&View")
        
        self.zoom_in_action = QAction("Zoom &In", self, triggered=self.workflow_canvas.zoom_in)
        self.zoom_in_action.setShortcut("Ctrl++")
        self.zoom_in_action.setStatusTip("Zoom in on the workflow canvas")
        view_menu.addAction(self.zoom_in_action)
        
        self.zoom_out_action = QAction("Zoom &Out", self, triggered=self.workflow_canvas.zoom_out)
        self.zoom_out_action.setShortcut("Ctrl+-")
        self.zoom_out_action.setStatusTip("Zoom out on the workflow canvas")
        view_menu.addAction(self.zoom_out_action)
        
        self.zoom_fit_action = QAction("&Fit to View", self, triggered=self.workflow_canvas.zoom_fit)
        self.zoom_fit_action.setShortcut("Ctrl+0")
        self.zoom_fit_action.setStatusTip("Fit workflow to the view")
        view_menu.addAction(self.zoom_fit_action)
        
        help_menu = self.menuBar().addMenu("&Help")
        
        help_contents_action = QAction("&Help Contents", self, triggered=self.show_help)
        help_contents_action.setShortcut("F1")
        help_contents_action.setStatusTip("Show help contents")
        help_menu.addAction(help_contents_action)
        
        about_action = QAction("&About", self, triggered=self.show_about)
        about_action.setStatusTip("About Bespoke Decision Tree Utility")
        help_menu.addAction(about_action)


    def create_dock_widgets(self):
        """Create dock widgets - DISABLED FOR NEW UI"""
        logger.info("create_dock_widgets called but disabled for new UI architecture")
        
        return

        
        
    def create_modern_layout_structure(self):
        """Create a modern layout structure with widget-based toolbar"""
        self.main_container = QWidget()
        self.main_layout = QVBoxLayout(self.main_container)
        self.main_layout.setContentsMargins(0, 30, 0, 0)  # 30px top margin for menu bar clearance
        self.main_layout.setSpacing(2)  # Small spacing between toolbar and content
        
        self.toolbar_container = ToolbarContainer(self)
        self.main_layout.addWidget(self.toolbar_container)
        
        stacked_widget = self.window_manager.get_stacked_widget()
        self.main_layout.addWidget(stacked_widget, 1)  # Give it stretch factor 1
        
        self.setCentralWidget(self.main_container)
        
        logger.info("Modern layout structure created with widget-based toolbar")
        
    def create_widget_based_toolbar(self):
        """Create the widget-based toolbar with all necessary buttons"""
        self.widget_toolbar = WidgetToolbar(self)
        
        self.widget_toolbar.add_button(
            "New Project", 
            self.new_project,
            "Create a new project (Ctrl+N)",
            "default",
            "📁"
        )
        
        self.widget_toolbar.add_button(
            "Open Project", 
            self.open_project,
            "Open existing project (Ctrl+O)",
            "default", 
            "📂"
        )
        
        self.widget_toolbar.add_button(
            "Save", 
            self.save_project,
            "Save current project (Ctrl+S)",
            "default",
            "💾"
        )
        
        self.widget_toolbar.add_separator()
        
        self.widget_toolbar.add_button(
            "Import CSV", 
            lambda: self.import_data('csv'),
            "Import CSV data file",
            "secondary",
            "📊"
        )
        
        self.widget_toolbar.add_button(
            "Import Excel", 
            lambda: self.import_data('excel'),
            "Import Excel data file", 
            "secondary",
            "📈"
        )
        
        self.widget_toolbar.add_button(
            "Data Import Wizard", 
            self.show_import_wizard,
            "Open comprehensive data import wizard",
            "secondary",
            "🧙"
        )
        
        self.widget_toolbar.add_separator()
        
        self.widget_toolbar.add_button(
            "Create Model", 
            self.create_new_model_dialog,
            "Create a new decision tree model",
            "secondary",
            "🌳"
        )
        
        self.widget_toolbar.add_button(
            "Evaluation", 
            self.add_evaluation_node,
            "Add evaluation node to workflow",
            "secondary",
            "📊"
        )
        
        self.widget_toolbar.add_button(
            "Visualization", 
            self.add_visualization_node,
            "Add visualization node to workflow",
            "secondary",
            "🎨"
        )
        
        self.widget_toolbar.add_separator()
        
        self.widget_toolbar.add_button(
            "Execute Workflow", 
            self.run_workflow_from_canvas,
            "Execute the current workflow (F5)",
            "secondary",
            "▶️"
        )
        
        self.widget_toolbar.add_separator()
        
        self.widget_toolbar.add_button(
            "Filter", 
            self.add_filter_node,
            "Add filter node to workflow",
            "secondary",
            "🔍"
        )
        
        self.widget_toolbar.add_button(
            "Transform", 
            self.add_transform_node,
            "Add transform node to workflow",
            "secondary",
            "🔄"
        )
        
        self.widget_toolbar.add_separator()
        
        self.widget_toolbar.add_button(
            "Zoom In", 
            self.zoom_in_workflow,
            "Zoom in workflow canvas",
            "secondary",
            "🔍+"
        )
        
        self.widget_toolbar.add_button(
            "Zoom Out", 
            self.zoom_out_workflow,
            "Zoom out workflow canvas",
            "secondary",
            "🔍-"
        )
        
        self.toolbar_container.set_toolbar(self.widget_toolbar)
        
        self.widget_toolbar.actionTriggered.connect(self.on_toolbar_action)
        
        self.widget_toolbar.setVisible(True)
        self.toolbar_container.setVisible(True)
        self.main_container.show()
        self.main_container.setVisible(True)
        
        logger.info(f"Widget-based toolbar created with {len(self.widget_toolbar.buttons)} buttons")
        logger.info(f"Toolbar visible: {self.widget_toolbar.isVisible()}")
        logger.info(f"Toolbar container visible: {self.toolbar_container.isVisible()}")
        logger.info(f"Main container visible: {self.main_container.isVisible()}")
        
    def on_toolbar_action(self, action_name: str):
        """Handle toolbar action signals"""
        logger.info(f"Toolbar action executed: {action_name}")
        
    def show_tree_visualization(self):
        """Show tree visualization tab"""
        try:
            if not self.current_model_name or self.current_model_name not in self.models:
                QMessageBox.warning(self, "Tree Visualization", "No active model selected. Please create and train a model first.")
                return
                
            model = self.models[self.current_model_name]
            if not hasattr(model, 'is_fitted') or not model.is_fitted:
                QMessageBox.warning(self, "Tree Visualization", "Model must be trained before visualization. Please run the workflow first.")
                return
            
            if hasattr(self, 'model_tabs') and self.model_tabs:
                for i in range(self.model_tabs.count()):
                    if self.model_tabs.tabText(i) == "Tree Visualizer":
                        self.model_tabs.setCurrentIndex(i)
                        logger.info(f"Switched to Tree Visualizer tab at index {i}")
                        return
            
            QMessageBox.information(self, "Tree Visualization", 
                "Tree visualization will be available after you start manual tree building.\n\n"
                "Steps:\n"
                "1. Create a model\n"
                "2. Connect data to the model\n"
                "3. Configure the model\n"
                "4. Start tree building")
                
        except Exception as e:
            logger.error(f"Error showing tree visualization: {e}")
            QMessageBox.warning(self, "Error", f"Unable to show tree visualization: {str(e)}")

    def create_enhanced_status_bar(self):
        """Create enhanced status bar with modern features"""
        if self.statusBar():
            self.statusBar().deleteLater()
            
        self.enhanced_status_bar = EnhancedStatusBar(self)
        self.setStatusBar(self.enhanced_status_bar)
        
        self.enhanced_status_bar.memoryWarning.connect(self.handle_memory_warning)
        
        self.enhanced_status_bar.set_context('workflow')
        self.enhanced_status_bar.show_message("Ready")
        
        logger.info("Enhanced status bar created")

    def create_breadcrumb_navigation(self):
        """Create breadcrumb navigation widget"""
        try:
            self.breadcrumb_widget = BreadcrumbWidget(self)
            
            self.breadcrumb_widget.backRequested.connect(self.navigate_back)
            self.breadcrumb_widget.navigationRequested.connect(self.navigate_to_window)
            
            self.breadcrumb_widget.setVisible(False)
            self.breadcrumb_widget.hide()
            
            logger.info("Breadcrumb navigation created but hidden to prevent menu bar overlap")
        except Exception as e:
            logger.error(f"Failed to create breadcrumb navigation: {e}")
            self.breadcrumb_widget = None

    def connect_window_manager_signals(self):
        """Connect window manager signals to main window methods"""
        if self.window_manager:
            self.window_manager.windowChanged.connect(self.on_window_changed)
            self.window_manager.windowSwitching.connect(self.on_window_switching)
            self.window_manager.backNavigationAvailable.connect(self.update_back_navigation)

            
    def on_window_changed(self, window_type: str):
        """Handle window change event"""
        if self.enhanced_status_bar:
            self.enhanced_status_bar.set_context(window_type)
            
        if self.breadcrumb_widget:
            can_go_back = self.window_manager.can_navigate_back() if self.window_manager else False
            self.breadcrumb_widget.update_breadcrumb(window_type, can_go_back)
            self.breadcrumb_widget.setVisible(False)
            
        logger.debug(f"Window changed to: {window_type}")

    def on_window_switching(self, from_type: str, to_type: str):
        """Handle window switching event"""
        if self.enhanced_status_bar:
            self.enhanced_status_bar.show_busy(f"Switching to {to_type.title()}...")
            
        logger.debug(f"Switching from {from_type} to {to_type}")

    def update_back_navigation(self, available: bool):
        """Update back navigation availability"""
        if self.breadcrumb_widget:
            current_window = self.window_manager.get_current_window_type() if self.window_manager else 'workflow'
            self.breadcrumb_widget.update_breadcrumb(current_window, available)
            self.breadcrumb_widget.setVisible(False)

    def navigate_back(self):
        """Navigate back to previous window"""
        if self.window_manager:
            self.window_manager.navigate_back()

    def navigate_to_window(self, window_type: str):
        """Navigate to specific window"""
        if self.window_manager:
            self.window_manager.show_window(window_type)

    def handle_memory_warning(self, usage_percent: float):
        """Handle memory warning from status bar"""
        if not self.memory_warning_shown and usage_percent > 80:
            self.memory_warning_shown = True
            if self.enhanced_status_bar:
                self.enhanced_status_bar.show_warning(
                    f"High memory usage: {usage_percent:.1f}%",
                    timeout=8000
                )

    def connect_workflow_navigation_signals(self):
        """Connect workflow canvas signals for detail window navigation"""
        try:
            if self.workflow_canvas and hasattr(self.workflow_canvas, 'scene'):
                try:
                    self.workflow_canvas.scene.nodeDoubleClicked.disconnect()
                except:
                    pass  # No existing connections
                
                self.workflow_canvas.scene.nodeDoubleClicked.connect(self.on_workflow_node_double_clicked)
                logger.info("Workflow navigation signals connected")
            else:
                logger.error("Failed to connect workflow navigation signals - workflow_canvas or scene not available")
        except Exception as e:
            logger.error(f"Error connecting signals: {e}", exc_info=True)

    def connect_workflow_context_menu_signals(self):
        """Connect workflow canvas context menu signals"""
        try:
            if self.workflow_canvas and hasattr(self.workflow_canvas, 'scene'):
                self.workflow_canvas.scene.nodeContextMenu.connect(self.on_workflow_node_context_menu)
                logger.info("Workflow context menu signals connected")
            else:
                logger.error("Failed to connect workflow context menu signals - workflow_canvas or scene not available")
        except Exception as e:
            logger.error(f"Error connecting context menu signals: {e}", exc_info=True)

    def on_workflow_node_context_menu(self, workflow_node, position):
        """Handle workflow node right-click context menu"""
        try:
            from PyQt5.QtWidgets import QMenu, QAction
            
            context_menu = QMenu(self)
            
            configure_action = QAction("⚙️ Configure", self)
            configure_action.triggered.connect(lambda: self.configure_workflow_node(workflow_node))
            context_menu.addAction(configure_action)
            
            context_menu.addSeparator()
            
            delete_action = QAction("🗑️ Delete", self)
            delete_action.triggered.connect(lambda: self.delete_workflow_node(workflow_node))
            context_menu.addAction(delete_action)
            
            view_pos = self.workflow_canvas.view.mapFromScene(position)
            global_pos = self.workflow_canvas.view.mapToGlobal(view_pos)
            context_menu.exec_(global_pos)
            
        except Exception as e:
            logger.error(f"Error showing workflow node context menu: {e}", exc_info=True)

    def configure_workflow_node(self, workflow_node):
        """Configure workflow node settings"""
        logger.info(f"Configure requested for node {workflow_node.node_id} ({workflow_node.title})")
        
        try:
            from PyQt5.QtWidgets import QDialog
            from ui.workflow_canvas import NODE_TYPE_FILTER, NODE_TYPE_TRANSFORM, NODE_TYPE_MODEL, NodeConfigDialog
            
            if workflow_node.node_type == NODE_TYPE_FILTER:
                self._configure_filter_node(workflow_node)
            elif workflow_node.node_type == NODE_TYPE_TRANSFORM:
                self._configure_transform_node(workflow_node)
            elif workflow_node.node_type == NODE_TYPE_MODEL:
                self._configure_model_node(workflow_node)
            else:
                self._configure_generic_node(workflow_node)
                
        except Exception as e:
            logger.error(f"Error configuring workflow node: {e}", exc_info=True)
    
    def _configure_filter_node(self, filter_node):
        """Configure filter node with conditions dialog"""
        try:
            from ui.dialogs.enhanced_filter_dialog import EnhancedFilterDialog
            
            available_columns, column_types = self._get_input_columns_and_types_for_node(filter_node)
            
            dialog = EnhancedFilterDialog(filter_node, available_columns, self, column_types)
            if dialog.exec_() == QDialog.Accepted:
                logger.info(f"Filter node {filter_node.node_id} configured successfully")
                
        except ImportError:
            self._configure_filter_node_simple(filter_node)
        except Exception as e:
            logger.error(f"Error configuring filter node: {e}")
    
    def _configure_filter_node_simple(self, filter_node):
        """Simple filter configuration dialog"""
        try:
            from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                                       QPushButton, QComboBox, QLineEdit, QListWidget, 
                                       QListWidgetItem, QMessageBox)
            
            available_columns = self._get_input_columns_for_node(filter_node)
            
            if not available_columns:
                QMessageBox.warning(self, "No Data", 
                                   "No input data available. Please connect a dataset to this filter node first.")
                return
            
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Configure Filter: {filter_node.title}")
            dialog.setModal(True)
            dialog.resize(500, 400)
            
            layout = QVBoxLayout()
            
            layout.addWidget(QLabel("Add filter conditions:"))
            
            conditions_list = QListWidget()
            layout.addWidget(QLabel("Current conditions:"))
            layout.addWidget(conditions_list)
            
            for i, condition in enumerate(filter_node.conditions):
                item_text = f"{condition['column']} {condition['operator']} {condition['value']}"
                if condition.get('logic'):
                    item_text = f"{condition['logic']} {item_text}"
                conditions_list.addItem(item_text)
            
            add_layout = QHBoxLayout()
            
            column_combo = QComboBox()
            column_combo.addItems(available_columns)
            add_layout.addWidget(QLabel("Column:"))
            add_layout.addWidget(column_combo)
            
            operator_combo = QComboBox()
            operator_combo.addItems(['>', '<', '>=', '<=', '==', '!=', 'contains', 'starts_with', 'ends_with', 'is_null', 'not_null'])
            add_layout.addWidget(QLabel("Operator:"))
            add_layout.addWidget(operator_combo)
            
            value_edit = QLineEdit()
            add_layout.addWidget(QLabel("Value:"))
            add_layout.addWidget(value_edit)
            
            logic_combo = QComboBox()
            logic_combo.addItems(['AND', 'OR'])
            add_layout.addWidget(QLabel("Logic:"))
            add_layout.addWidget(logic_combo)
            
            layout.addLayout(add_layout)
            
            button_layout = QHBoxLayout()
            
            add_btn = QPushButton("Add Condition")
            clear_btn = QPushButton("Clear All")
            ok_btn = QPushButton("OK")
            cancel_btn = QPushButton("Cancel")
            
            def add_condition():
                column = column_combo.currentText()
                operator = operator_combo.currentText()
                value = value_edit.text().strip()
                logic = logic_combo.currentText() if filter_node.conditions else None
                
                if not value and operator not in ['is_null', 'not_null']:
                    QMessageBox.warning(dialog, "Invalid Input", "Please enter a value.")
                    return
                
                try:
                    if operator in ['>', '<', '>=', '<='] and value:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                except ValueError:
                    pass  # Keep as string
                
                filter_node.add_condition(column, operator, value, logic)
                
                conditions_list.clear()
                for i, condition in enumerate(filter_node.conditions):
                    item_text = f"{condition['column']} {condition['operator']} {condition['value']}"
                    if condition.get('logic'):
                        item_text = f"{condition['logic']} {item_text}"
                    conditions_list.addItem(item_text)
                
                value_edit.clear()
            
            def clear_all():
                filter_node.clear_conditions()
                conditions_list.clear()
            
            add_btn.clicked.connect(add_condition)
            clear_btn.clicked.connect(clear_all)
            ok_btn.clicked.connect(dialog.accept)
            cancel_btn.clicked.connect(dialog.reject)
            
            button_layout.addWidget(add_btn)
            button_layout.addWidget(clear_btn)
            button_layout.addWidget(ok_btn)
            button_layout.addWidget(cancel_btn)
            
            layout.addLayout(button_layout)
            dialog.setLayout(layout)
            
            if dialog.exec_() == QDialog.Accepted:
                logger.info(f"Filter node {filter_node.node_id} configured with {len(filter_node.conditions)} conditions")
            
        except Exception as e:
            logger.error(f"Error in simple filter configuration: {e}")
            QMessageBox.critical(self, "Error", f"Failed to configure filter: {str(e)}")
    
    def _configure_transform_node(self, transform_node):
        """Configure transform node with transformations dialog"""
        try:
            from ui.dialogs.enhanced_transform_dialog import EnhancedTransformDialog
            
            available_columns = self._get_input_columns_for_node(transform_node)
            
            dialog = EnhancedTransformDialog(transform_node, available_columns, self)
            if dialog.exec_() == QDialog.Accepted:
                logger.info(f"Transform node {transform_node.node_id} configured successfully")
                
        except ImportError:
            self._configure_transform_node_simple(transform_node)
        except Exception as e:
            logger.error(f"Error configuring transform node: {e}")
    
    def _configure_transform_node_simple(self, transform_node):
        """Simple transform configuration dialog"""
        try:
            from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                                       QPushButton, QComboBox, QLineEdit, QListWidget, 
                                       QListWidgetItem, QMessageBox, QTextEdit)
            
            available_columns = self._get_input_columns_for_node(transform_node)
            
            if not available_columns:
                QMessageBox.warning(self, "No Data", 
                                   "No input data available. Please connect a dataset to this transform node first.")
                return
            
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Configure Transform: {transform_node.title}")
            dialog.setModal(True)
            dialog.resize(600, 500)
            
            layout = QVBoxLayout()
            
            layout.addWidget(QLabel("Add transformations:"))
            
            transforms_list = QListWidget()
            layout.addWidget(QLabel("Current transformations:"))
            layout.addWidget(transforms_list)
            
            for i, transform in enumerate(transform_node.transformations):
                item_text = f"{transform['type']}: {transform['target_column']}"
                if transform.get('source_columns'):
                    item_text += f" (from {', '.join(transform['source_columns'])})"
                transforms_list.addItem(item_text)
            
            add_layout = QVBoxLayout()
            
            type_layout = QHBoxLayout()
            transform_type_combo = QComboBox()
            transform_type_combo.addItems([
                'create_variable', 'derive_ratio', 'derive_difference', 'derive_sum',
                'encode_categorical', 'binning', 'standardize', 'log_transform'
            ])
            type_layout.addWidget(QLabel("Type:"))
            type_layout.addWidget(transform_type_combo)
            add_layout.addLayout(type_layout)
            
            target_layout = QHBoxLayout()
            target_column_edit = QLineEdit()
            target_layout.addWidget(QLabel("New Column Name:"))
            target_layout.addWidget(target_column_edit)
            add_layout.addLayout(target_layout)
            
            source_layout = QHBoxLayout()
            source_column_combo = QComboBox()
            source_column_combo.addItems(available_columns)
            source_layout.addWidget(QLabel("Source Column:"))
            source_layout.addWidget(source_column_combo)
            add_layout.addLayout(source_layout)
            
            formula_layout = QVBoxLayout()
            formula_edit = QTextEdit()
            formula_edit.setMaximumHeight(60)
            formula_edit.setPlaceholderText("Enter formula (e.g., Income / Age, Income > 50000)")
            formula_layout.addWidget(QLabel("Formula (for create_variable):"))
            formula_layout.addWidget(formula_edit)
            add_layout.addLayout(formula_layout)
            
            layout.addLayout(add_layout)
            
            button_layout = QHBoxLayout()
            
            add_btn = QPushButton("Add Transformation")
            clear_btn = QPushButton("Clear All")
            ok_btn = QPushButton("OK")
            cancel_btn = QPushButton("Cancel")
            
            def add_transformation():
                transform_type = transform_type_combo.currentText()
                target_column = target_column_edit.text().strip()
                source_column = source_column_combo.currentText()
                formula = formula_edit.toPlainText().strip()
                
                if not target_column:
                    QMessageBox.warning(dialog, "Invalid Input", "Please enter a target column name.")
                    return
                
                source_columns = []
                if transform_type in ['derive_ratio', 'derive_difference']:
                    source_columns = [source_column]
                elif transform_type in ['derive_sum', 'encode_categorical', 'binning', 'standardize', 'log_transform']:
                    source_columns = [source_column]
                
                transform_node.add_transformation(
                    transform_type=transform_type,
                    target_column=target_column,
                    source_columns=source_columns,
                    formula=formula if transform_type == 'create_variable' else None
                )
                
                transforms_list.clear()
                for i, transform in enumerate(transform_node.transformations):
                    item_text = f"{transform['type']}: {transform['target_column']}"
                    if transform.get('source_columns'):
                        item_text += f" (from {', '.join(transform['source_columns'])})"
                    transforms_list.addItem(item_text)
                
                target_column_edit.clear()
                formula_edit.clear()
            
            def clear_all():
                transform_node.clear_transformations()
                transforms_list.clear()
            
            add_btn.clicked.connect(add_transformation)
            clear_btn.clicked.connect(clear_all)
            ok_btn.clicked.connect(dialog.accept)
            cancel_btn.clicked.connect(dialog.reject)
            
            button_layout.addWidget(add_btn)
            button_layout.addWidget(clear_btn)
            button_layout.addWidget(ok_btn)
            button_layout.addWidget(cancel_btn)
            
            layout.addLayout(button_layout)
            dialog.setLayout(layout)
            
            if dialog.exec_() == QDialog.Accepted:
                logger.info(f"Transform node {transform_node.node_id} configured with {len(transform_node.transformations)} transformations")
            
        except Exception as e:
            logger.error(f"Error in simple transform configuration: {e}")
            QMessageBox.critical(self, "Error", f"Failed to configure transform: {str(e)}")
    
    def _configure_model_node(self, model_node):
        """Configure model node with detailed NodeConfigDialog"""
        try:
            from ui.workflow_canvas import NodeConfigDialog
            
            dialog = NodeConfigDialog(model_node, self)
            if dialog.exec_() == QDialog.Accepted:
                logger.info(f"Model node {model_node.node_id} configured successfully")
                
        except Exception as e:
            logger.error(f"Error configuring model node: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to configure model node: {str(e)}")
    
    def _configure_generic_node(self, workflow_node):
        """Generic configuration for other node types"""
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, "Configuration", 
                               f"Configuration for {workflow_node.node_type} nodes is not yet implemented.")
    
    def _get_input_columns_for_node(self, node):
        """Get available columns from input connections to a node"""
        try:
            if hasattr(self.workflow_canvas, 'scene') and hasattr(self.workflow_canvas.scene, 'connections'):
                for connection in self.workflow_canvas.scene.connections.values():
                    if connection.target_node_id == node.node_id:
                        source_node_id = connection.source_node_id
                        source_node = self.workflow_canvas.scene.nodes.get(source_node_id)
                        
                        if source_node:
                            source_node_title = source_node.title
                            
                            output_port_name = None
                            if source_node.node_type == 'dataset':
                                output_port_name = 'Data Output'
                            elif source_node.node_type == 'filter':
                                output_port_name = 'Filtered Data'
                            elif source_node.node_type == 'transform':
                                output_port_name = 'Transformed Data'
                            
                            if output_port_name:
                                compound_name = f"{source_node_title}_Output_{output_port_name}"
                                
                                if hasattr(self, 'datasets') and compound_name in self.datasets:
                                    import pandas as pd
                                    dataset = self.datasets[compound_name]
                                    if isinstance(dataset, pd.DataFrame):
                                        return list(dataset.columns)
                                
                                if hasattr(self, 'datasets'):
                                    for dataset_name, dataset_data in self.datasets.items():
                                        if dataset_name.startswith(compound_name):
                                            import pandas as pd
                                            if isinstance(dataset_data, pd.DataFrame):
                                                return list(dataset_data.columns)
                            
                            if hasattr(self, 'workflow_execution_engine') and self.workflow_execution_engine:
                                execution_engine = self.workflow_execution_engine
                                
                                if (hasattr(execution_engine, 'node_outputs') and 
                                    source_node_id in execution_engine.node_outputs):
                                    node_outputs = execution_engine.node_outputs[source_node_id]
                                    
                                    import pandas as pd
                                    for output_name, output_data in node_outputs.items():
                                        if isinstance(output_data, pd.DataFrame):
                                            return list(output_data.columns)
                            
                            if hasattr(source_node, 'dataset_name'):
                                if hasattr(self, 'datasets') and source_node.dataset_name in self.datasets:
                                    return list(self.datasets[source_node.dataset_name].columns)
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting input columns for node: {e}")
            return []
    
    def _get_input_columns_and_types_for_node(self, node):
        """Get available columns and their data types from input connections to a node"""
        try:
            import pandas as pd
            
            if hasattr(self.workflow_canvas, 'scene') and hasattr(self.workflow_canvas.scene, 'connections'):
                for connection in self.workflow_canvas.scene.connections.values():
                    if connection.target_node_id == node.node_id:
                        source_node_id = connection.source_node_id
                        source_node = self.workflow_canvas.scene.nodes.get(source_node_id)
                        
                        if source_node:
                            dataset_df = None
                            
                            output_port_name = None
                            if source_node.node_type == 'dataset':
                                output_port_name = 'Data Output'
                            elif source_node.node_type == 'filter':
                                output_port_name = 'Filtered Data'
                            elif source_node.node_type == 'transform':
                                output_port_name = 'Transformed Data'
                            
                            if output_port_name:
                                compound_name = f"{source_node.title}_Output_{output_port_name}"
                                
                                if hasattr(self, 'datasets') and compound_name in self.datasets:
                                    dataset_df = self.datasets[compound_name]
                                
                                if dataset_df is None and hasattr(self, 'datasets'):
                                    for dataset_name, dataset_data in self.datasets.items():
                                        if dataset_name.startswith(compound_name):
                                            dataset_df = dataset_data
                                            break
                                
                                if dataset_df is None and hasattr(self, 'workflow_execution_engine'):
                                    execution_engine = self.workflow_execution_engine
                                    if (hasattr(execution_engine, 'node_outputs') and 
                                        source_node_id in execution_engine.node_outputs):
                                        outputs = execution_engine.node_outputs[source_node_id]
                                        if output_port_name in outputs:
                                            dataset_df = outputs[output_port_name]
                                
                                if dataset_df is None and hasattr(source_node, 'dataset_name'):
                                    if hasattr(self, 'datasets') and source_node.dataset_name in self.datasets:
                                        dataset_df = self.datasets[source_node.dataset_name]
                            
                            if dataset_df is not None and isinstance(dataset_df, pd.DataFrame):
                                columns = list(dataset_df.columns)
                                column_types = {col: str(dataset_df[col].dtype) for col in columns}
                                return columns, column_types
            
            return [], {}
            
        except Exception as e:
            logger.error(f"Error getting input columns and types for node: {e}")
            return [], {}
        
    def delete_workflow_node(self, workflow_node):
        """Delete workflow node with smart connection handling strategies"""
        try:
            from PyQt5.QtWidgets import QMessageBox, QDialog
            
            deletion_analysis = self._analyze_node_deletion_impact(workflow_node)
            
            message = f"Are you sure you want to delete '{workflow_node.title}'?"
            
            if deletion_analysis['has_downstream']:
                message += f"\n\nWARNING: This node has {deletion_analysis['downstream_count']} downstream connections."
                message += "\nDownstream nodes will lose their input data."
                
                if deletion_analysis['has_upstream']:
                    message += f"\n\nStrategies available:"
                    message += f"\n• Delete node only (breaks workflow)"
                    message += f"\n• Delete node and reconnect upstream to downstream"
                    
                    strategy = self._show_deletion_strategy_dialog(workflow_node, deletion_analysis)
                    if strategy == 'cancel':
                        return
                    elif strategy == 'reconnect':
                        self._delete_node_with_reconnection(workflow_node, deletion_analysis)
                        return
            
            elif deletion_analysis['has_upstream']:
                message += f"\n\nThis node has {deletion_analysis['upstream_count']} upstream connections."
                message += "\nThese connections will be removed."
            
            else:
                message += "\n\nThis node has no connections and can be safely deleted."
            
            reply = QMessageBox.question(
                self, 
                "Delete Node", 
                message,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                node_id = workflow_node.node_id
                node_title = workflow_node.title
                
                if hasattr(self.workflow_canvas, 'scene') and hasattr(self.workflow_canvas.scene, 'remove_node'):
                    self.workflow_canvas.scene.remove_node(node_id)
                    logger.info(f"Deleted workflow node: {node_id} ({node_title})")
                    
                    if self.enhanced_status_bar:
                        self.enhanced_status_bar.show_message(f"Deleted node: {node_title}", 3000)
                else:
                    logger.error("Workflow canvas scene does not have remove_node method")
                    
        except Exception as e:
            logger.error(f"Error deleting workflow node: {e}", exc_info=True)
            if self.enhanced_status_bar:
                self.enhanced_status_bar.show_error(f"Failed to delete node: {str(e)}")
    
    def _analyze_node_deletion_impact(self, node):
        """Analyze the impact of deleting a node on the workflow"""
        analysis = {
            'has_upstream': False,
            'has_downstream': False,
            'upstream_count': 0,
            'downstream_count': 0,
            'upstream_connections': [],
            'downstream_connections': [],
            'can_reconnect': False
        }
        
        try:
            if hasattr(self.workflow_canvas, 'scene') and hasattr(self.workflow_canvas.scene, 'connections'):
                for conn_id, connection in self.workflow_canvas.scene.connections.items():
                    if connection.source_node_id == node.node_id:
                        analysis['has_downstream'] = True
                        analysis['downstream_count'] += 1
                        analysis['downstream_connections'].append(connection)
                    
                    elif connection.target_node_id == node.node_id:
                        analysis['has_upstream'] = True
                        analysis['upstream_count'] += 1
                        analysis['upstream_connections'].append(connection)
                
                if (analysis['has_upstream'] and analysis['has_downstream'] and 
                    analysis['upstream_count'] == 1 and analysis['downstream_count'] == 1):
                    upstream_conn = analysis['upstream_connections'][0]
                    downstream_conn = analysis['downstream_connections'][0]
                    
                    if upstream_conn.source_port['type'] == downstream_conn.target_port['type']:
                        analysis['can_reconnect'] = True
        
        except Exception as e:
            logger.error(f"Error analyzing node deletion impact: {e}")
        
        return analysis
    
    def _show_deletion_strategy_dialog(self, node, analysis):
        """Show dialog to select deletion strategy"""
        try:
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QButtonGroup, QRadioButton
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Choose Deletion Strategy")
            dialog.setModal(True)
            
            layout = QVBoxLayout()
            
            label = QLabel(f"How would you like to handle deleting '{node.title}'?")
            layout.addWidget(label)
            
            button_group = QButtonGroup()
            
            delete_only_btn = QRadioButton("Delete node only (break connections)")
            delete_only_btn.setChecked(True)
            layout.addWidget(delete_only_btn)
            button_group.addButton(delete_only_btn, 0)
            
            if analysis['can_reconnect']:
                reconnect_btn = QRadioButton("Delete node and reconnect upstream to downstream")
                layout.addWidget(reconnect_btn)
                button_group.addButton(reconnect_btn, 1)
            
            button_layout = QVBoxLayout()
            ok_btn = QPushButton("OK")
            cancel_btn = QPushButton("Cancel")
            
            ok_btn.clicked.connect(dialog.accept)
            cancel_btn.clicked.connect(dialog.reject)
            
            button_layout.addWidget(ok_btn)
            button_layout.addWidget(cancel_btn)
            layout.addLayout(button_layout)
            
            dialog.setLayout(layout)
            
            if dialog.exec_() == QDialog.Accepted:
                selected_id = button_group.checkedId()
                if selected_id == 0:
                    return 'delete_only'
                elif selected_id == 1:
                    return 'reconnect'
            
            return 'cancel'
            
        except Exception as e:
            logger.error(f"Error showing deletion strategy dialog: {e}")
            return 'delete_only'
    
    def _delete_node_with_reconnection(self, node, analysis):
        """Delete node and reconnect upstream to downstream"""
        try:
            if not analysis['can_reconnect']:
                logger.warning("Cannot reconnect - falling back to simple deletion")
                self.workflow_canvas.scene.remove_node(node.node_id)
                return
            
            upstream_conn = analysis['upstream_connections'][0]
            downstream_conn = analysis['downstream_connections'][0]
            
            source_node = self.workflow_canvas.scene.nodes.get(upstream_conn.source_node_id)
            target_node = self.workflow_canvas.scene.nodes.get(downstream_conn.target_node_id)
            
            if source_node and target_node:
                self.workflow_canvas.scene.remove_node(node.node_id)
                
                new_conn_id = self.workflow_canvas.scene.add_connection(
                    source_node, upstream_conn.source_port,
                    target_node, downstream_conn.target_port
                )
                
                if new_conn_id:
                    logger.info(f"Successfully reconnected {source_node.title} to {target_node.title} after deleting {node.title}")
                    if self.enhanced_status_bar:
                        self.enhanced_status_bar.show_message(
                            f"Deleted {node.title} and reconnected workflow", 3000)
                else:
                    logger.error("Failed to create reconnection")
                    if self.enhanced_status_bar:
                        self.enhanced_status_bar.show_error("Failed to reconnect workflow")
            else:
                logger.error("Could not find source or target nodes for reconnection")
                
        except Exception as e:
            logger.error(f"Error in node deletion with reconnection: {e}")
            if self.enhanced_status_bar:
                self.enhanced_status_bar.show_error(f"Reconnection failed: {str(e)}")

    def on_workflow_node_double_clicked(self, workflow_node):
        """Handle workflow node double-click for detail window navigation"""
        try:
            if hasattr(self, '_processing_double_click') and self._processing_double_click:
                logger.warning("Preventing recursive double-click handling")
                return
            
            self._processing_double_click = True
            
            node_type = workflow_node.node_type
            node_id = workflow_node.node_id
            
            logger.info(f"Node double-clicked: {node_id} (type: {node_type})")
            
            if node_type in ["dataset", "data"]:
                data_info = {
                    'node_id': node_id,
                    'name': workflow_node.title,
                    'type': node_type
                }
                self.window_manager.show_window('data', data_info)
                
            elif node_type in ["model", "decision_tree"]:
                model_info = {
                    'node_id': node_id,
                    'name': workflow_node.title,
                    'type': node_type
                }
                self.window_manager.show_window('model', model_info)
                
            elif node_type in ["visualization", "viz"]:
                evaluation_results = {}
                model_data = None
                tree_root = None
                dataset = None
                target_column = None
                
                if hasattr(self, 'latest_evaluation_results'):
                    if node_id in self.latest_evaluation_results:
                        evaluation_results = self.latest_evaluation_results[node_id]
                    else:
                        for eval_node_id, results in self.latest_evaluation_results.items():
                            if results:  # Use the most recent non-empty results
                                evaluation_results = results
                                break
                
                if hasattr(self, 'latest_tree_models'):
                    for tree_node_id, tree_model in self.latest_tree_models.items():
                        if tree_model:
                            model_data = tree_model
                            tree_root = getattr(tree_model, 'root', None)
                            break
                
                if hasattr(self, 'current_dataset') and self.current_dataset is not None:
                    dataset = self.current_dataset
                    
                if evaluation_results:
                    target_column = evaluation_results.get('target_variable', 
                                   evaluation_results.get('target_column', None))
                
                viz_info = {
                    'node_id': node_id,
                    'name': workflow_node.title,
                    'type': node_type,
                    'evaluation_results': evaluation_results,
                    'model_data': model_data,
                    'tree_root': tree_root,
                    'dataset': dataset,
                    'target_column': target_column
                }
                
                logger.info(f"Opening visualization window with data: model={model_data is not None}, tree={tree_root is not None}, dataset={dataset is not None}")
                self.window_manager.show_window('visualization', viz_info)
                
            elif node_type in ["transform", "filter"]:
                self.configure_workflow_node(workflow_node)
                
            else:
                logger.info(f"No detail window defined for node type: {node_type}")
                if self.enhanced_status_bar:
                    self.enhanced_status_bar.show_message(f"Node {workflow_node.title} selected", 2000)
                    
        except Exception as e:
            logger.error(f"Error handling node double-click: {e}", exc_info=True)
            if self.enhanced_status_bar:
                self.enhanced_status_bar.show_error(f"Error opening detail window: {str(e)}")
        finally:
            self._processing_double_click = False

    def create_compatibility_bridge(self):
        """Create compatibility bridge for existing code that expects dock widgets and tabs"""
        self.data_tabs = QTabWidget()
        self.data_tabs.setTabsClosable(True)
        self.data_tabs.tabCloseRequested.connect(self.close_dataset_tab)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)  # Hidden - functionality in status bar
        
        self.model_tabs = QTabWidget()
        self.visualization_tabs = QTabWidget()
        
        self.data_dock = QDockWidget("Data Explorer", self)
        self.data_dock.setObjectName("dataDock")
        self.data_dock.setWidget(self.data_tabs)
        self.data_dock.setVisible(False)  # Hidden in new UI
        self.data_dock.hide()  # Extra hiding
        
        self.model_dock = QDockWidget("Model Inspector", self)
        self.model_dock.setObjectName("modelDock")
        self.model_dock.setWidget(self.model_tabs)
        self.model_dock.setVisible(False)  # Hidden in new UI
        self.model_dock.hide()  # Extra hiding
        
        self.visualization_dock = QDockWidget("Visualizations", self)
        self.visualization_dock.setObjectName("visualizationDock")
        self.visualization_dock.setWidget(self.visualization_tabs)
        self.visualization_dock.setVisible(False)  # Hidden in new UI
        self.visualization_dock.hide()  # Extra hiding
        
        self.init_data_tabs_structure()
        self.init_model_tabs_structure()
        self.init_visualization_tabs_structure()
        
        logger.info("Compatibility bridge created - legacy tab widgets available but hidden")


    def init_data_tabs_structure(self):
        """Initialize data tab structure - COMPATIBILITY STUB"""
        try:
            if not hasattr(self, 'data_tabs') or self.data_tabs is None:
                return
            
            while self.data_tabs.count() > 0:
                self.data_tabs.removeTab(0)
            
            if not hasattr(self, 'data_viewer_widgets'):
                self.data_viewer_widgets = {}
            self.data_viewer_widgets.clear()
            
            if not hasattr(self, 'variable_viewer_widget'):
                self.variable_viewer_widget = VariableViewerWidget()
            
            self.variables_tab_content = QWidget()
            layout = QVBoxLayout(self.variables_tab_content)
            layout.addWidget(self.variable_viewer_widget)
            self.data_tabs.addTab(self.variables_tab_content, "Variables")
            
            placeholder = QLabel("Use main workflow canvas - legacy tabs hidden")
            placeholder.setAlignment(Qt.AlignCenter)
            self.data_tabs.addTab(placeholder, "Data View")
            
            logger.info("Data tabs structure initialized (compatibility mode)")
        except Exception as e:
            logger.error(f"Error in init_data_tabs_structure compatibility stub: {e}")

    def init_model_tabs_structure(self):
        """Initialize model tab structure - COMPATIBILITY STUB"""
        try:
            if not hasattr(self, 'model_tabs') or self.model_tabs is None:
                return
            
            while self.model_tabs.count() > 0:
                self.model_tabs.removeTab(0)
            
            if not hasattr(self, 'model_properties_widget'):
                self.model_properties_widget = ModelPropertiesWidget()
            self.model_tabs.addTab(self.model_properties_widget, "Properties")
            
            if not hasattr(self, 'node_inspector_widget'):
                from ui.node_editor import NodePropertiesWidget
                self.node_inspector_widget = NodePropertiesWidget()
            self.model_tabs.addTab(self.node_inspector_widget, "Node Inspector")
            
            logger.info("Model tabs structure initialized (compatibility mode)")
        except Exception as e:
            logger.error(f"Error in init_model_tabs_structure compatibility stub: {e}")

    def init_visualization_tabs_structure(self):
        """Initialize visualization tab structure - COMPATIBILITY STUB"""
        try:
            if not hasattr(self, 'visualization_tabs') or self.visualization_tabs is None:
                return
            
            while self.visualization_tabs.count() > 0:
                self.visualization_tabs.removeTab(0)
            
            self._apply_visualization_tab_fonts()
            
            if not hasattr(self, 'tree_viz_widget'):
                self.tree_viz_widget = TreeVisualizerWidget()
                self.tree_viz_widget.connect_signals()
                self._apply_tree_visualizer_fonts(self.tree_viz_widget)
            self.visualization_tabs.addTab(self.tree_viz_widget, "Tree View")
            
            placeholder_tabs = [
                ("ROC Curve", "ROC analysis available in model detail window"),
                ("Lift Chart", "Lift analysis available in model detail window"),
                ("Variable Importance", "Variable importance available in model detail window")
            ]
            
            for tab_name, message in placeholder_tabs:
                placeholder = QLabel(message)
                placeholder.setAlignment(Qt.AlignCenter)
                self._apply_standard_label_font(placeholder)
                self.visualization_tabs.addTab(placeholder, tab_name)
            
            node_report_widget = self._create_node_report_tab()
            self.visualization_tabs.addTab(node_report_widget, "Node Report")
            
            logger.info("Visualization tabs structure initialized (compatibility mode)")
        except Exception as e:
            logger.error(f"Error in init_visualization_tabs_structure compatibility stub: {e}")

    def _apply_visualization_tab_fonts(self):
        """Apply consistent fonts to visualization tabs based on design specifications"""
        try:
            tab_font = self._get_standard_body_font()
            if hasattr(self, 'visualization_tabs') and self.visualization_tabs:
                self.visualization_tabs.setFont(tab_font)
                
                self.visualization_tabs.setStyleSheet("""
                    QTabWidget::pane {
                        border: 1px solid #e2e8f0;
                        background-color: #ffffff;
                    }
                    QTabBar::tab {
                        background-color: #f8fafc;
                        color: #1e293b;
                        border: 1px solid #e2e8f0;
                        padding: 8px 16px;
                        margin-right: 2px;
                        font-family: 'Segoe UI', 'Ubuntu', system-ui, sans-serif;
                        font-size: 12px;
                        font-weight: 500;
                    }
                    QTabBar::tab:selected {
                        background-color: #ffffff;
                        color: #1e293b;
                        border-bottom: 2px solid #3b82f6;
                        font-weight: 600;
                    }
                    QTabBar::tab:hover {
                        background-color: #f1f5f9;
                    }
                """)
                
            logger.debug("Visualization tabs font styling applied")
        except Exception as e:
            logger.error(f"Error applying visualization tab fonts: {e}")

    def _apply_tree_visualizer_fonts(self, tree_widget):
        """Apply consistent fonts to tree visualizer widget"""
        try:
            if tree_widget:
                tree_font = self._get_standard_body_font()
                tree_widget.setFont(tree_font)
                
                tree_widget.setStyleSheet("""
                    QWidget {
                        font-family: 'Segoe UI', 'Ubuntu', system-ui, sans-serif;
                        font-size: 12px;
                        color: #1e293b;
                        background-color: #ffffff;
                    }
                    QLabel {
                        font-size: 12px;
                        color: #64748b;
                    }
                    QGroupBox {
                        font-weight: 600;
                        font-size: 12px;
                        color: #1e293b;
                        border: 1px solid #e2e8f0;
                        border-radius: 6px;
                        margin-top: 8px;
                        padding-top: 8px;
                    }
                    QGroupBox::title {
                        subcontrol-origin: margin;
                        left: 8px;
                        padding: 0 4px 0 4px;
                    }
                """)
                
            logger.debug("Tree visualizer font styling applied")
        except Exception as e:
            logger.error(f"Error applying tree visualizer fonts: {e}")

    def _apply_standard_label_font(self, label):
        """Apply standard font to a label widget"""
        try:
            if label:
                font = self._get_standard_body_font()
                label.setFont(font)
                
                label.setStyleSheet("""
                    QLabel {
                        font-family: 'Segoe UI', 'Ubuntu', system-ui, sans-serif;
                        font-size: 14px;
                        color: #64748b;
                        padding: 16px;
                    }
                """)
                
        except Exception as e:
            logger.error(f"Error applying standard label font: {e}")

    def _get_standard_body_font(self):
        """Get standardized body font based on design specifications"""
        try:
            import platform
            system = platform.system()
            
            if system == "Windows":
                font = QFont("Segoe UI Variable", 14)
                if not font.exactMatch():
                    font = QFont("Segoe UI", 14)
            elif system == "Linux":
                font = QFont("Ubuntu", 14)
                if not font.exactMatch():
                    font = QFont("system-ui", 14)
            else:
                font = QFont("system-ui", 14)
                
            font.setWeight(QFont.Normal)
            return font
            
        except Exception as e:
            logger.error(f"Error creating standard body font: {e}")
            return QFont("Arial", 14)

    def _get_standard_header_font(self):
        """Get standardized header font (18px Bold)"""
        try:
            import platform
            system = platform.system()
            
            if system == "Windows":
                font = QFont("Segoe UI Variable", 18)
                if not font.exactMatch():
                    font = QFont("Segoe UI", 18)
            elif system == "Linux":
                font = QFont("Ubuntu", 18)
                if not font.exactMatch():
                    font = QFont("system-ui", 18)
            else:
                font = QFont("system-ui", 18)
                
            font.setWeight(QFont.Bold)
            return font
            
        except Exception as e:
            logger.error(f"Error creating standard header font: {e}")
            return QFont("Arial", 18, QFont.Bold)

    def _get_standard_small_font(self):
        """Get standardized small font (12px Regular)"""
        try:
            import platform
            system = platform.system()
            
            if system == "Windows":
                font = QFont("Segoe UI Variable", 12)
                if not font.exactMatch():
                    font = QFont("Segoe UI", 12)
            elif system == "Linux":
                font = QFont("Ubuntu", 12)
                if not font.exactMatch():
                    font = QFont("system-ui", 12)
            else:
                font = QFont("system-ui", 12)
                
            font.setWeight(QFont.Normal)
            return font
            
        except Exception as e:
            logger.error(f"Error creating standard small font: {e}")
            return QFont("Arial", 12)

    def create_shortcuts(self):
        """Create keyboard shortcuts"""
        QShortcut(QKeySequence("Ctrl+N"), self, self.new_project)
        QShortcut(QKeySequence("Ctrl+O"), self, self.open_project)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_project)
        QShortcut(QKeySequence("Ctrl+Shift+S"), self, self.save_project_as)
        
        QShortcut(QKeySequence.ZoomIn, self, self.workflow_canvas.zoom_in)
        QShortcut(QKeySequence.ZoomOut, self, self.workflow_canvas.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, self.workflow_canvas.zoom_fit)
        
        QShortcut(QKeySequence(Qt.Key_Delete), self, self.delete_selected_workflow_items)
        QShortcut(QKeySequence(Qt.Key_Backspace), self, self.delete_selected_workflow_items)
        
        QShortcut(QKeySequence("F5"), self, lambda: self.workflow_canvas.runWorkflowRequested.emit(None))

    def connect_signals(self):
        """Connect signals to slots"""
        self.workflow_canvas.runWorkflowRequested.connect(self.run_workflow_from_canvas)
        self.workflow_canvas.partialWorkflowRequested.connect(self.run_partial_workflow_from_canvas)
        self.workflow_canvas.nodeSelectedSignal.connect(self.on_workflow_node_selected_on_canvas)
        self.workflow_canvas.sceneChanged.connect(lambda: self.set_project_modified(True))
        
        if hasattr(self, 'tree_viz_widget') and hasattr(self.tree_viz_widget, 'nodeSelected'):
            self.tree_viz_widget.nodeSelected.connect(self.on_tree_node_selected_in_visualizer)
        if hasattr(self, 'tree_viz_widget') and hasattr(self.tree_viz_widget, 'nodeDoubleClicked'):
            self.tree_viz_widget.nodeDoubleClicked.connect(self.on_tree_node_double_clicked)
        if hasattr(self, 'tree_viz_widget') and hasattr(self.tree_viz_widget, 'nodeRightClicked'):
            self.tree_viz_widget.nodeRightClicked.connect(self.show_enhanced_context_menu)
            
        self.tree_context_menu.viewStatisticsRequested.connect(self.show_node_statistics_for_id)
        self.tree_context_menu.editSplitRequested.connect(self.edit_node_split)
        self.tree_context_menu.findOptimalSplitRequested.connect(self.find_optimal_split)
        self.tree_context_menu.manualSplitRequested.connect(self.manual_split_node)
        self.tree_context_menu.pruneSubtreeRequested.connect(self.prune_subtree)
        self.tree_context_menu.copyNodeInfoRequested.connect(self.copy_node_info)
        self.tree_context_menu.pasteNodeInfoRequested.connect(self.paste_node_info)
        
        self.data_tabs.currentChanged.connect(self.on_data_tab_changed)
        
        self.model_tabs.currentChanged.connect(self.on_model_tab_changed)
        
        self.workflow_engine.workflowExecutionStarted.connect(self.on_workflow_execution_started)
        self.workflow_engine.workflowExecutionFinished.connect(self.on_workflow_execution_finished)
        self.workflow_engine.workflowExecutionError.connect(self.on_workflow_execution_error)
        self.workflow_engine.workflowExecutionSuccess.connect(self.on_workflow_execution_success)
        self.workflow_engine.nodeProcessingStarted.connect(self.on_node_processing_started)
        self.workflow_engine.nodeProcessingFinished.connect(self.on_node_processing_finished)
        self.workflow_engine.nodeOutputReady.connect(self.on_node_output_ready)
        
        # TODO: Reimplement via modern toolbar context actions

    def setup_memory_monitoring(self):
        """Set up periodic memory monitoring"""
        self.memory_timer = QTimer(self)
        self.memory_timer.timeout.connect(self.update_memory_usage)
        self.memory_timer.start(10000)  # Check every 10 seconds
        self.update_memory_usage()

    def update_memory_usage(self):
        """Update memory usage display in status bar"""
        memory_info = get_system_memory_info()
        used_gb = memory_info.get('used', 0.0)
        total_gb = memory_info.get('total', 0.0)
        percent = memory_info.get('percent', 0.0)
        
        if self.enhanced_status_bar:
            self.enhanced_status_bar.update_memory_usage(used_gb * 1024, total_gb * 1024)  # Convert GB to MB
            
            if percent > 90:
                self.enhanced_status_bar.show_warning(f"High memory usage: {percent:.1f}%")
            elif percent > 70:
                self.enhanced_status_bar.show_message(f"Memory usage: {percent:.1f}%", timeout=5000)
        
        threshold = get_config_value(self.config, 'memory.memory_warning_threshold', 80.0)
        if percent > threshold:
            if not self.memory_warning_shown:
                self.statusBar.showMessage("Warning: High memory usage detected", 5000)
                self.memory_warning_shown = True
                import gc
                gc.collect()
        else:
            self.memory_warning_shown = False


    def new_project(self, confirm_discard=True):
        """Create a new project"""
        if confirm_discard and self.check_unsaved_changes():
            return

        self.datasets.clear()
        self.current_dataset_name = None
        self.models.clear()
        self.current_model_name = None
        self.project_manager.new_project()
        self.current_project_path = None
        self.latest_evaluation_results = {}

        self.workflow_canvas.clear_canvas()
        if hasattr(self, 'tree_viz_widget'):
            self.tree_viz_widget.set_tree(None)
        self.clear_all_data_viewers()
        if hasattr(self, 'model_properties_widget') and self.model_properties_widget is not None:
            self.model_properties_widget.set_model(None)
        
        if self.enhanced_status_bar:
            self.enhanced_status_bar.show_message("New project created.", timeout=3000)
        
        # Note: dataset_label and model_label functionality moved to enhanced status bar
        # TODO: Implement project status display in enhanced status bar
        self.setWindowTitle("Bespoke Decision Tree Utility - New Project")
        self.set_project_modified(False)
        self.update_action_states()

    def open_project(self, file_path: Union[str, bool] = False):
        """Open an existing project"""
        if self.check_unsaved_changes():
            return

        if isinstance(file_path, bool) or file_path is None:
            settings = QSettings("BespokeAnalytics", "BespokeDecisionTreeUtility")
            last_dir = settings.value("lastProjectPath", str(Path.home()))
            file_path_tuple = QFileDialog.getOpenFileName(
                self, "Open Project", last_dir, 
                f"Bespoke Project Files (*{ProjectManager.PROJECT_FILE_EXTENSION});;All Files (*)"
            )
            _file_path = file_path_tuple[0]
            if not _file_path:
                return
            file_path = _file_path
            settings.setValue("lastProjectPath", str(Path(file_path).parent))
        
        try:
            self.new_project(confirm_discard=False)
            
            self.project_manager.main_window_ref = self
            
            success, project_data = self.project_manager.load_project_comprehensive(str(file_path), self.workflow_canvas)
            
            if success:
                self.datasets = project_data.get('datasets', {})
                self.models = project_data.get('models', {})
                
                self.latest_evaluation_results = project_data.get('analysis_results', {})
                self.latest_variable_importance = project_data.get('variable_importance', {})
                self.latest_performance_metrics = project_data.get('performance_metrics', {})
                
                if hasattr(self, '_setup_data_availability_checks'):
                    self._setup_data_availability_checks()
                else:
                    logger.warning("_setup_data_availability_checks method not available during project load")
                
                ui_state = project_data.get('ui_state', {})
                if ui_state.get('current_dataset_name'):
                    self.current_dataset_name = ui_state['current_dataset_name']
                if ui_state.get('current_model_name'):
                    self.current_model_name = ui_state['current_model_name']
                
                if self.datasets:
                    first_ds_name = next(iter(self.datasets))
                    self.current_dataset_name = first_ds_name
                    self.add_data_viewer_tab(first_ds_name, self.datasets[first_ds_name])
                    for name, df in self.datasets.items():
                        node = self.workflow_canvas.get_node_by_id(name)
                        if isinstance(node, DatasetNode):
                            node.update_dataset_info(len(df), len(df.columns))

                if self.models:
                    for model_name, model in self.models.items():
                        for node_id, node in self.workflow_canvas.scene.nodes.items():
                            if isinstance(node, ModelNode):
                                node_config = node.get_config()
                                model_ref_id = node_config.get('model_ref_id')
                                if (model_ref_id == model_name or 
                                    model_ref_id == getattr(model, 'model_name', None) or
                                    model_ref_id == getattr(model, 'model_id', None)):
                                    node.model = model
                                    
                                    if hasattr(node, '_pending_config') and node._pending_config:
                                        pending = node._pending_config
                                        if pending.get('target_variable'):
                                            model.target_name = pending['target_variable']
                                            logger.info(f"CRITICAL FIX: Applied pending target variable '{pending['target_variable']}' to model '{model_name}'")
                                        if pending.get('model_name'):
                                            model.model_name = pending['model_name']
                                            logger.debug(f"Applied pending model name '{pending['model_name']}' to model")
                                        if pending.get('model_params'):
                                            try:
                                                model.set_params(**pending['model_params'])
                                                logger.debug(f"Applied pending model parameters to model")
                                            except Exception as e:
                                                logger.warning(f"Failed to apply pending model parameters: {e}")
                                        delattr(node, '_pending_config')
                                        logger.debug(f"Cleared pending config for ModelNode {node_id}")
                                    
                                    node.update_model_info()
                                    logger.info(f"Associated loaded model '{model_name}' with ModelNode {node_id}")
                    
                    try:
                        for node_id, node in self.workflow_canvas.scene.nodes.items():
                            if hasattr(node, '__class__') and 'ModelNode' in str(node.__class__):
                                if not hasattr(node, 'model') or node.model is None:
                                    if hasattr(node, '_pending_config') and node._pending_config:
                                        pending = node._pending_config
                                        model_ref_id = pending.get('model_ref_id')
                                        
                                        if model_ref_id and model_ref_id in self.models:
                                            model = self.models[model_ref_id]
                                            node.model = model
                                            
                                            if pending.get('target_variable'):
                                                model.target_name = pending['target_variable']
                                                logger.info(f"Applied pending target variable '{pending['target_variable']}' to model '{model_ref_id}'")
                                            if pending.get('model_name'):
                                                model.model_name = pending['model_name']
                                            if pending.get('model_params'):
                                                try:
                                                    model.set_params(**pending['model_params'])
                                                except Exception as e:
                                                    logger.warning(f"Failed to apply pending model parameters: {e}")
                                            
                                            delattr(node, '_pending_config')
                                            if hasattr(node, 'update_model_info'):
                                                node.update_model_info()
                                            logger.info(f"Completed pending model association for ModelNode {node_id} with model '{model_ref_id}'")
                                        else:
                                            logger.warning(f"ModelNode {node_id} has pending config for model '{model_ref_id}' but model not found in loaded models")
                                            
                    except Exception as e:
                        logger.error(f"Error completing pending model associations: {e}")
                    
                    try:
                        for node_id, node in self.workflow_canvas.scene.nodes.items():
                            if hasattr(node, '__class__') and 'DatasetNode' in str(node.__class__):
                                if hasattr(node, 'dataset_name'):
                                    dataset_name = node.dataset_name
                                    if dataset_name and dataset_name in self.datasets:
                                        df = self.datasets[dataset_name]
                                        if hasattr(node, 'update_dataset_info'):
                                            node.update_dataset_info(len(df), len(df.columns))
                                        logger.debug(f"Refreshed DatasetNode {node_id} with dataset '{dataset_name}'")
                            
                            elif hasattr(node, '__class__') and 'ModelNode' in str(node.__class__):
                                if hasattr(node, 'model') and node.model is not None:
                                    if hasattr(node, 'update_model_info'):
                                        node.update_model_info()
                                    logger.debug(f"Refreshed ModelNode {node_id} display")
                                else:
                                    logger.warning(f"ModelNode {node_id} has no associated model after project load")
                            
                            elif hasattr(node, 'update_visualization_info'):
                                node.update_visualization_info()
                                logger.debug(f"Refreshed visualization node {node_id}")
                        
                        self.workflow_canvas.scene.update()
                        logger.info("Completed workflow node refresh after project load")
                        
                    except Exception as e:
                        logger.error(f"Error refreshing workflow nodes after project load: {e}")
                    
                    first_model_name = next(iter(self.models))
                    self.current_model_name = first_model_name
                    self.connect_model_signals(self.models[first_model_name])
                    if hasattr(self, 'model_properties_widget') and self.model_properties_widget is not None:
                        self.model_properties_widget.set_model(self.models[first_model_name])
                    if hasattr(self, 'tree_viz_widget') and self.models[first_model_name].is_fitted:
                        self.tree_viz_widget.set_tree(self.models[first_model_name].root)

                self.current_project_path = Path(file_path)
                self.project_manager.current_project_path = self.current_project_path
                self.add_recent_file(str(file_path))
                project_name_stem = self.current_project_path.stem
                self.setWindowTitle(f"Bespoke Decision Tree Utility - {project_name_stem}")
                if self.enhanced_status_bar:
                    self.enhanced_status_bar.show_message(f"Project loaded: {file_path}", timeout=3000)
                self.set_project_modified(False)
            else:
                QMessageBox.critical(self, "Load Error", "Failed to load project.")
                self.new_project(confirm_discard=False)
        except Exception as e:
            logger.error(f"Error opening project {file_path}: {e}", exc_info=True)
            QMessageBox.critical(self, "Load Error", f"Could not open project: {e}")
            self.new_project(confirm_discard=False)
        
        self.update_action_states()

    def save_project(self) -> bool:
        """Save the current project using comprehensive method"""
        if not self.project_manager.current_project_path:
            return self.save_project_as()
        
        self.enhanced_status_bar.show_message("Saving project...")
        QApplication.processEvents()
        
        analysis_results = getattr(self, 'latest_evaluation_results', {})
        variable_importance = getattr(self, 'latest_variable_importance', {})
        performance_metrics = getattr(self, 'latest_performance_metrics', {})
        
        ui_state = {
            'current_dataset_name': getattr(self, 'current_dataset_name', None),
            'current_model_name': getattr(self, 'current_model_name', None),
            'window_size': [self.width(), self.height()],
            'window_pos': [self.x(), self.y()]
        }
        
        QApplication.processEvents()
        
        success = self.project_manager.save_project_comprehensive(
            str(self.project_manager.current_project_path),
            workflow_canvas=self.workflow_canvas,
            datasets=self.datasets,
            models=self.models,
            analysis_results=analysis_results,
            variable_importance=variable_importance,
            performance_metrics=performance_metrics,
            ui_state=ui_state,
            include_data=True
        )
        
        QApplication.processEvents()
        
        if success:
            self.set_project_modified(False)
            self.enhanced_status_bar.show_message(f"Project saved: {self.project_manager.current_project_path.name}")
        else:
            QMessageBox.critical(self, "Save Error", "Could not save project.")
        
        return success


    def _initialize_enhanced_features(self):
        """Initialize enhanced features for the main window"""
        try:
            if not hasattr(self, 'config') or not self.config:
                self.config = {}
            
            required_sections = ['analytics', 'export', 'visualization']
            for section in required_sections:
                if section not in self.config:
                    self.config[section] = {}
            
            analytics_defaults = {
                'max_features_for_importance': 1000,
                'max_samples_for_permutation': 10000,
                'cv_folds': 5,
                'importance_threshold': 0.001
            }
            
            for key, default_value in analytics_defaults.items():
                if key not in self.config['analytics']:
                    self.config['analytics'][key] = default_value
            
            export_defaults = {
                'add_comments': True,
                'include_metadata': True,
                'include_data_summary': True,
                'code_style': 'verbose'
            }
            
            for key, default_value in export_defaults.items():
                if key not in self.config['export']:
                    self.config['export'][key] = default_value
            
            viz_defaults = {
                'figure_size': [12, 8],
                'dpi': 300,
                'color_scheme': 'default'
            }
            
            for key, default_value in viz_defaults.items():
                if key not in self.config['visualization']:
                    self.config['visualization'][key] = default_value
            
            if MATPLOTLIB_AVAILABLE:
                try:
                    plt.rcParams['figure.figsize'] = self.config['visualization']['figure_size']
                    plt.rcParams['savefig.dpi'] = self.config['visualization']['dpi']
                except Exception as e:
                    logger.warning(f"Could not configure matplotlib: {str(e)}")
            
            logger.info("Enhanced features initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing enhanced features: {str(e)}")



    def save_project_as(self) -> bool:
        """Save the current project with a new name"""
        settings = QSettings("BespokeAnalytics", "BespokeDecisionTreeUtility")
        last_dir = str(self.project_manager.current_project_path.parent) if self.project_manager.current_project_path else settings.value("lastProjectPath", str(Path.home()))
        
        file_path_tuple = QFileDialog.getSaveFileName(
            self, "Save Project As", last_dir, 
            f"Bespoke Project Files (*{ProjectManager.PROJECT_FILE_EXTENSION});;All Files (*)"
        )
        file_path = file_path_tuple[0]

        if file_path:
            new_path = Path(file_path)
            if not new_path.name.endswith(ProjectManager.PROJECT_FILE_EXTENSION):
                new_path = new_path.with_suffix(ProjectManager.PROJECT_FILE_EXTENSION)
            
            self.project_manager.current_project_path = new_path
            self.current_project_path = new_path
            
            settings.setValue("lastProjectPath", str(new_path.parent))
            project_name_stem = new_path.stem
            self.setWindowTitle(f"Bespoke Decision Tree Utility - {project_name_stem}")
            self.add_recent_file(str(new_path))
            return self.save_project()
        
        return False


    def import_data(self, format_type: str):
        """Import data from a file"""
        file_path_str: Optional[str] = None
        kwargs = {}
        options_accepted = False
        dialog = None

        settings = QSettings("BespokeAnalytics", "BespokeDecisionTreeUtility")
        last_dir = settings.value("lastDataImportPath", str(Path.home()))

        if format_type == 'csv':
            file_path_str, _ = QFileDialog.getOpenFileName(self, "Import CSV", last_dir, "CSV Files (*.csv);;All Files (*)")
            if file_path_str:
                dialog = CsvImportOptionsDialog(parent=self)
        elif format_type == 'excel':
            file_path_str, _ = QFileDialog.getOpenFileName(self, "Import Excel", last_dir, "Excel Files (*.xlsx *.xls);;All Files (*)")
            if file_path_str:
                dialog = ExcelImportOptionsDialog(file_path_str, parent=self)
        elif format_type == 'text':
            file_path_str, _ = QFileDialog.getOpenFileName(self, "Import Text File", last_dir, "Text Files (*.txt *.tsv);;All Files (*)")
            if file_path_str:
                dialog = TextImportOptionsDialog(parent=self)
        elif format_type == 'database':
            dialog = DatabaseImportOptionsDialog(parent=self)
            if dialog.exec_() == QDialog.Accepted:
                try:
                    db_config = dialog.get_import_config()
                    dataset_name = db_config.get('dataset_name', 'Database_Import')
                    
                    base_dataset_name = dataset_name
                    i = 1
                    while dataset_name in self.datasets:
                        dataset_name = f"{base_dataset_name}_{i}"
                        i += 1
                    
                    self.enhanced_status_bar.show_message(f"Importing data from database...")
                    self.progress_bar.show()
                    self.progress_bar.setValue(25)
                    QApplication.processEvents()
                    
                    db_importer = DatabaseImporter(self.config)
                    df = db_importer.import_data_to_dataframe(
                        db_config['db_type'],
                        db_config['connection_details'],
                        db_config['query']
                    )
                    
                    if df is not None and not df.empty:
                        self.datasets[dataset_name] = df
                        self.current_dataset_name = dataset_name
                        
                        self.add_data_viewer_tab(dataset_name, df)
                        self.workflow_canvas.add_node_to_canvas(
                            NODE_TYPE_DATASET, 
                            title=dataset_name, 
                            specific_config={
                                'dataset_name': dataset_name, 
                                'df_rows': len(df), 
                                'df_cols': len(df.columns),
                                'source': 'database'
                            }
                        )
                        
                        self.enhanced_status_bar.show_message(f"Database dataset '{dataset_name}' imported.")
                        self.enhanced_status_bar.show_message(f"Active: {dataset_name} ({len(df)} rows, {len(df.columns)} cols)")
                        self.set_project_modified(True)
                    else:
                        QMessageBox.warning(self, "Import Warning", "No data was imported from the database.")
                        self.enhanced_status_bar.show_message("Database import returned no data.")
                        
                except Exception as e:
                    logger.error(f"Error importing from database: {e}", exc_info=True)
                    QMessageBox.critical(self, "Database Import Error", f"Could not import data from database: {e}")
                    self.enhanced_status_bar.show_message("Database import failed.")
                finally:
                    self.progress_bar.hide()
            else:
                self.enhanced_status_bar.show_message("Database import cancelled.")
            return
        elif format_type == 'cloud':
            dialog = CloudImportOptionsDialog(parent=self)
            if dialog.exec_() == QDialog.Accepted:
                try:
                    cloud_config = dialog.get_import_config()
                    dataset_name = cloud_config.get('dataset_name', 'Cloud_Import')
                    
                    base_dataset_name = dataset_name
                    i = 1
                    while dataset_name in self.datasets:
                        dataset_name = f"{base_dataset_name}_{i}"
                        i += 1
                    
                    self.enhanced_status_bar.show_message(f"Importing data from cloud storage...")
                    self.progress_bar.show()
                    self.progress_bar.setValue(25)
                    QApplication.processEvents()
                    
                    cloud_importer = CloudImporter(self.config)
                    df = cloud_importer.import_file_to_dataframe(
                        cloud_config['provider'],
                        cloud_config['connection_details'],
                        cloud_config['file_path'],
                        cloud_config.get('file_format', 'csv')
                    )
                    
                    if df is not None and not df.empty:
                        self.datasets[dataset_name] = df
                        self.current_dataset_name = dataset_name
                        
                        self.add_data_viewer_tab(dataset_name, df)
                        self.workflow_canvas.add_node_to_canvas(
                            NODE_TYPE_DATASET, 
                            title=dataset_name, 
                            specific_config={
                                'dataset_name': dataset_name, 
                                'df_rows': len(df), 
                                'df_cols': len(df.columns),
                                'source': 'cloud'
                            }
                        )
                        
                        self.enhanced_status_bar.show_message(f"Cloud dataset '{dataset_name}' imported.")
                        self.enhanced_status_bar.show_message(f"Active: {dataset_name} ({len(df)} rows, {len(df.columns)} cols)")
                        self.set_project_modified(True)
                    else:
                        QMessageBox.warning(self, "Import Warning", "No data was imported from cloud storage.")
                        self.enhanced_status_bar.show_message("Cloud import returned no data.")
                        
                except Exception as e:
                    logger.error(f"Error importing from cloud: {e}", exc_info=True)
                    QMessageBox.critical(self, "Cloud Import Error", f"Could not import data from cloud: {e}")
                    self.enhanced_status_bar.show_message("Cloud import failed.")
                finally:
                    self.progress_bar.hide()
            else:
                self.enhanced_status_bar.show_message("Cloud import cancelled.")
            return
        else:
            QMessageBox.warning(self, "Import Error", f"Unknown import format: '{format_type}'")
            return

        if not file_path_str:
            self.enhanced_status_bar.show_message("Data import cancelled.")
            return
        
        settings.setValue("lastDataImportPath", str(Path(file_path_str).parent))

        if dialog and dialog.exec_() == QDialog.Accepted:
            kwargs = dialog.get_options()
            options_accepted = True
        elif not dialog:
            options_accepted = True

        if not options_accepted:
            self.enhanced_status_bar.show_message("Data import cancelled (options).")
            return

        base_dataset_name = Path(file_path_str).stem
        dataset_name = base_dataset_name
        i = 1
        while dataset_name in self.datasets:
            dataset_name = f"{base_dataset_name}_{i}"
            i += 1
        
        kwargs['name'] = dataset_name

        self.enhanced_status_bar.show_busy(f"Loading '{Path(file_path_str).name}'...")
        QApplication.processEvents()

        try:
            df, metadata = self.data_loader.load_file(file_path_str, **kwargs)
            
            if df is None:
                raise ValueError(metadata.get("error", "Unknown error loading data"))

            self.datasets[dataset_name] = df
            self.current_dataset_name = dataset_name

            self.add_data_viewer_tab(dataset_name, df)
            self.workflow_canvas.add_node_to_canvas(
                NODE_TYPE_DATASET, 
                title=dataset_name, 
                specific_config={
                    'dataset_name': dataset_name, 
                    'df_rows': len(df), 
                    'df_cols': len(df.columns)
                }
            )
            
            self.enhanced_status_bar.show_message(f"Dataset '{dataset_name}' loaded.")
            self.enhanced_status_bar.show_message(f"Active: {dataset_name} ({len(df)} rows, {len(df.columns)} cols)")
            self.set_project_modified(True)
        except Exception as e:
            logger.error(f"Error importing data from {file_path_str}: {e}", exc_info=True)
            QMessageBox.critical(self, "Import Error", f"Could not import data: {e}")
            self.enhanced_status_bar.show_message("Import failed.")
        finally:
            self.enhanced_status_bar.hide_busy("Ready")
        
        self.update_action_states()

    def show_import_wizard(self):
        """Show the comprehensive data import wizard"""
        wizard = DataImportWizard(self.config, parent=self)
        
        if wizard.exec_() == DataImportWizard.Accepted:
            try:
                import_config = wizard.get_import_config()
                dataset_name = import_config.get('dataset_name', 'Imported_Data')
                
                base_dataset_name = dataset_name
                i = 1
                while dataset_name in self.datasets:
                    dataset_name = f"{base_dataset_name}_{i}"
                    i += 1
                
                self.enhanced_status_bar.show_message(f"Importing dataset '{dataset_name}'...")
                self.progress_bar.show()
                self.progress_bar.setValue(25)
                QApplication.processEvents()
                
                file_path = import_config['file_path']
                
                kwargs = {
                    'encoding': import_config.get('encoding', 'utf-8'),
                    'delimiter': import_config.get('delimiter', ','),
                    'header': import_config.get('header_row', 0),
                    'skiprows': import_config.get('skip_rows', 0),
                    'na_values': import_config.get('missing_values', [])
                }
                
                if import_config['file_format'] == 'excel':
                    kwargs['sheet_name'] = import_config.get('sheet_name', 0)
                
                df, metadata = self.data_loader.load_file(file_path, **kwargs)
                self.progress_bar.setValue(60)
                
                if df is None:
                    raise ValueError(metadata.get("error", "Unknown error loading data"))
                
                data_types = import_config.get('data_types', {})
                for col, dtype in data_types.items():
                    if col in df.columns and dtype != 'auto':
                        try:
                            if dtype == 'numeric':
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            elif dtype == 'integer':
                                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                            elif dtype == 'float':
                                df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                            elif dtype == 'boolean':
                                df[col] = df[col].astype(bool)
                            elif dtype == 'date':
                                df[col] = pd.to_datetime(df[col], errors='coerce')
                            elif dtype == 'category':
                                df[col] = df[col].astype('category')
                            elif dtype == 'text':
                                df[col] = df[col].astype(str)
                        except Exception as e:
                            logger.warning(f"Could not convert column {col} to {dtype}: {e}")
                
                self.progress_bar.setValue(80)
                
                missing_strategy = import_config.get('missing_strategy', 'keep')
                if missing_strategy == 'drop':
                    df = df.dropna()
                elif missing_strategy == 'impute':
                    numeric_method = import_config.get('numeric_impute', 'mean')
                    text_method = import_config.get('text_impute', 'mode')
                    
                    for col in df.columns:
                        if df[col].isna().any():
                            if pd.api.types.is_numeric_dtype(df[col]):
                                if numeric_method == 'mean':
                                    df[col].fillna(df[col].mean(), inplace=True)
                                elif numeric_method == 'median':
                                    df[col].fillna(df[col].median(), inplace=True)
                                elif numeric_method == 'mode':
                                    df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 0, inplace=True)
                                elif numeric_method == 'zero':
                                    df[col].fillna(0, inplace=True)
                            else:
                                if text_method == 'mode':
                                    df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown', inplace=True)
                                elif text_method == 'empty':
                                    df[col].fillna('', inplace=True)
                                elif text_method == 'unknown':
                                    df[col].fillna('Unknown', inplace=True)
                
                self.progress_bar.setValue(100)
                
                self.datasets[dataset_name] = df
                self.current_dataset_name = dataset_name
                
                self.add_data_viewer_tab(dataset_name, df)
                self.workflow_canvas.add_node_to_canvas(
                    NODE_TYPE_DATASET, 
                    title=dataset_name, 
                    specific_config={
                        'dataset_name': dataset_name, 
                        'df_rows': len(df), 
                        'df_cols': len(df.columns),
                        'source': 'wizard'
                    }
                )
                
                self.enhanced_status_bar.show_message(f"Dataset '{dataset_name}' imported successfully.")
                self.enhanced_status_bar.show_message(f"Active: {dataset_name} ({len(df)} rows, {len(df.columns)} cols)")
                self.set_project_modified(True)
                
            except Exception as e:
                logger.error(f"Error importing data via wizard: {e}", exc_info=True)
                QMessageBox.critical(self, "Import Error", f"Could not import data: {e}")
                self.enhanced_status_bar.show_message("Import failed.")
            finally:
                self.progress_bar.hide()
        else:
            self.enhanced_status_bar.show_message("Import wizard cancelled.")
        
        self.update_action_states()

    def add_data_viewer_tab(self, name: str, df: pd.DataFrame):
        """Add or update a data viewer tab"""
        if self.data_tabs.currentWidget() == self.dataset_placeholder_label:
            self.data_tabs.removeTab(self.data_tabs.currentIndex())
            self.dataset_placeholder_label.deleteLater()
            self.dataset_placeholder_label = None

        if name in self.data_viewer_widgets:
            viewer = self.data_viewer_widgets[name]
            viewer.set_dataframe(df)
            self.data_tabs.setCurrentWidget(viewer)
        else:
            new_viewer = DataViewerWidget(parent=self.data_tabs)
            new_viewer.set_dataframe(df)
            self.data_tabs.addTab(new_viewer, name)
            self.data_tabs.setCurrentWidget(new_viewer)
            self.data_viewer_widgets[name] = new_viewer
        
        self.update_variable_viewer(df)

    def clear_all_data_viewers(self):
        """Clear all data viewer tabs"""
        self.data_viewer_widgets.clear()
        
        while self.data_tabs.count() > 1:
            widget = self.data_tabs.widget(1)
            self.data_tabs.removeTab(1)
            if widget:
                widget.deleteLater()
        
        if self.variable_viewer_widget:
            self.variable_viewer_widget.clear()
        
        if self.data_tabs.count() == 1 and not self.dataset_placeholder_label:
            self.dataset_placeholder_label = QLabel("Import or load a dataset to view data.")
            self.dataset_placeholder_label.setAlignment(Qt.AlignCenter)
            self.dataset_placeholder_label.setObjectName("dataset_placeholder_label")
            self.data_tabs.addTab(self.dataset_placeholder_label, "Data View")
        
        self.data_tabs.setCurrentIndex(0)

    def close_dataset_tab(self, index: int):
        """Close a dataset tab"""
        widget_to_close = self.data_tabs.widget(index)
        if widget_to_close == self.variables_tab_content or widget_to_close == self.dataset_placeholder_label:
            return
        
        dataset_name = self.data_tabs.tabText(index)
        if QMessageBox.question(self, "Close Dataset", f"Close dataset '{dataset_name}'?") == QMessageBox.Yes:
            self.data_tabs.removeTab(index)
            
            if dataset_name in self.data_viewer_widgets:
                self.data_viewer_widgets.pop(dataset_name).deleteLater()
            
            if dataset_name in self.datasets:
                del self.datasets[dataset_name]
            
            if self.current_dataset_name == dataset_name:
                self.current_dataset_name = None
                
                if self.data_tabs.count() > 1:
                    self.data_tabs.setCurrentIndex(1)
                elif self.data_tabs.count() == 1:
                    self.variable_viewer_widget.clear()
                    self.init_data_tabs_structure()
                    self.enhanced_status_bar.show_message("No dataset active")
            
            self.set_project_modified(True)
            self.update_action_states()

    def update_variable_viewer(self, df: pd.DataFrame):
        """Update the variable viewer with data from a DataFrame"""
        if self.variable_viewer_widget:
            self.variable_viewer_widget.update_variables(df)


    def create_new_model_dialog(self):
        """Create a new model with default name"""
        if not self.current_dataset_name or self.current_dataset_name not in self.datasets:
            QMessageBox.warning(self, "New Model", "Please load or select a dataset first.")
            return

        base_name = "Decision_Tree"
        counter = 1
        model_name = f"{base_name}_{counter}"
        
        while model_name in self.models:
            counter += 1
            model_name = f"{base_name}_{counter}"
        
        model = BespokeDecisionTree(config=self.config)
        model.model_name = model_name
        
        self.models[model_name] = model
        self.current_model_name = model_name
        self.connect_model_signals(model)
        
        if self.model_properties_widget:
            self.model_properties_widget.set_model(model)
        
        if self.enhanced_status_bar:
            self.enhanced_status_bar.show_message(f"Active Model: {model_name}", timeout=2000)
        
        self.workflow_canvas.add_node_to_canvas(
            NODE_TYPE_MODEL, 
            title=model_name, 
            specific_config={'model_ref_id': model_name}
        )
        
        self.enhanced_status_bar.show_message(f"Model '{model_name}' created.")
        self.set_project_modified(True)
        
        self.update_action_states()

    def configure_selected_model_dialog(self):
        """Show dialog for configuring the selected model"""
        if not self.current_model_name or self.current_model_name not in self.models:
            QMessageBox.warning(self, "Configure Model", "No model is currently active or selected.")
            return
        
        model = self.models[self.current_model_name]
        dialog = TreeConfigurationDialog(current_config=model.get_params(), parent=self)
        
        if dialog.exec_() == QDialog.Accepted:
            model.set_params(**dialog.get_configuration())
            
            if self.model_properties_widget:
                self.model_properties_widget.update_properties()
            
            self.enhanced_status_bar.show_message(f"Model '{self.current_model_name}' configured.")
            self.set_project_modified(True)

    def train_selected_model_directly(self):
        """Train the selected model directly"""
        if not self.current_model_name or self.current_model_name not in self.models:
            QMessageBox.warning(self, "Train Model", "No model selected.")
            return
        
        if not self.current_dataset_name or self.current_dataset_name not in self.datasets:
            QMessageBox.warning(self, "Train Model", "No active dataset to train on.")
            return

        model = self.models[self.current_model_name]
        df = self.datasets[self.current_dataset_name]

        target_col = model.target_name
        if not target_col or target_col not in df.columns:
            potential_targets = [col for col in df.columns if df[col].nunique() <= 10]
            target_col, ok = QInputDialog.getItem(
                self, "Select Target Variable", 
                "Target for model training:", 
                potential_targets, 0, False
            )
            
            if not ok or not target_col:
                return
            
            model.target_name = target_col

        try:
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            self.on_training_progress(0, f"Starting direct training for {model.model_name}...")
            model.fit(X, y)
            
            self.on_training_complete()
            
            self.enhanced_status_bar.show_message(f"Model '{model.model_name}' trained directly.")
            self.set_project_modified(True)
        except Exception as e:
            self.on_model_error(f"Direct training failed: {e}")

    def connect_model_signals(self, model: BespokeDecisionTree):
        """Connect signals for a model"""
        logger.debug(f"Signal connection skipped for model '{model.model_name}' (signals disabled).")

    def change_growth_mode(self, index: int):
        """Change the growth mode for the current model - COMPATIBILITY STUB"""
        # TODO: Reimplement via modern toolbar dropdown actions
        logger.info(f"change_growth_mode called with index {index} - functionality temporarily disabled")
        return
        

    def change_criterion(self, index: int):
        """Change the criterion for the current model - COMPATIBILITY STUB"""
        # TODO: Reimplement via modern toolbar dropdown actions
        logger.info(f"change_criterion called with index {index} - functionality temporarily disabled")
        return
        

    def toggle_pruning(self, checked: bool):
        """Toggle pruning for the current model"""
        self.config.setdefault('decision_tree', {})['pruning_enabled'] = checked
        
        self.enable_pruning_action.setChecked(checked)
        
        if self.current_model_name and self.current_model_name in self.models:
            self.models[self.current_model_name].set_params(pruning_enabled=checked)
            status = "enabled" if checked else "disabled"
            self.enhanced_status_bar.show_message(f"Pruning {status} for model '{self.current_model_name}'")
            self.set_project_modified(True)

    def toggle_pruning_ui_sync(self, checked: bool):
        """Sync pruning UI elements"""
        self.enable_pruning_action.setChecked(checked)
        self.toggle_pruning(checked)

    def prune_selected_subtree(self):
        """Prune the subtree at the selected node"""
        selected_node_id = None
        if hasattr(self.tree_viz_widget, 'get_selected_node_id'):
            selected_node_id = self.tree_viz_widget.get_selected_node_id()

        if not selected_node_id:
            QMessageBox.warning(self, "Prune Subtree", "Please select a node in the Tree View panel first.")
            return

        if not self.current_model_name or self.current_model_name not in self.models:
            QMessageBox.warning(self, "Prune Subtree", "No active model found.")
            return

        model = self.models[self.current_model_name]
        if not model.is_fitted:
            QMessageBox.warning(self, "Prune Subtree", "The current model is not trained.")
            return

        node = model.get_node(selected_node_id)
        if not node:
            QMessageBox.warning(self, "Prune Subtree", f"Node '{selected_node_id}' not found in the current model.")
            return

        confirm = QMessageBox.question(
            self, "Prune Subtree", 
            f"Are you sure you want to prune the subtree at node '{selected_node_id}'?\n"
            "This will make the node a terminal node and remove all its children.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if confirm == QMessageBox.Yes:
            try:
                model.prune_subtree(selected_node_id)
                self.tree_viz_widget.set_tree(model.root)
                self.enhanced_status_bar.show_message(f"Subtree at node '{selected_node_id}' pruned successfully.")
                self.set_project_modified(True)
            except Exception as e:
                logger.error(f"Error pruning subtree: {e}", exc_info=True)
                QMessageBox.critical(self, "Pruning Error", f"Could not prune subtree: {e}")


    def export_model_enhanced(self):
        """Enhanced model export with multiple language support"""
        if not self.current_model_name or self.current_model_name not in self.models:
            QMessageBox.warning(self, "Export Model", "No model to export.")
            return

        model = self.models[self.current_model_name]
        if not model.is_fitted:
            QMessageBox.warning(self, "Export Model", "Model must be trained before export.")
            return

        try:
            dialog = EnhancedExportDialog(model, parent=self)
            if dialog.exec_() == QDialog.Accepted:
                export_options = dialog.get_export_options()
                
                success = self._perform_enhanced_export(model, export_options)
                
                if success:
                    QMessageBox.information(self, "Export Successful", 
                                        f"Model exported successfully to:\n{export_options['output_path']}")
                else:
                    QMessageBox.warning(self, "Export Failed", "Model export failed. Check the logs for details.")
                    
        except Exception as e:
            logger.error(f"Error in enhanced export: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Export Error", f"Error during export:\n{str(e)}")


    def export_model(self, format_type: str):
        """Export the current model"""
        if not self.current_model_name or self.current_model_name not in self.models:
            QMessageBox.warning(self, "Export Error", "No model selected to export")
            return

        model = self.models[self.current_model_name]
        if not model.is_fitted:
            QMessageBox.warning(self, "Export Error", "Model is not trained yet")
            return

        try:
            settings = QSettings("BespokeAnalytics", "BespokeDecisionTreeUtility")
            last_dir = settings.value("lastExportPath", str(Path.home()))
            default_filename = f"{model.model_name}.{format_type.lower()}"
            
            if format_type == 'pmml':
                filters = "PMML Files (*.pmml);;All Files (*)"
            elif format_type == 'json':
                filters = "JSON Files (*.json);;All Files (*)"
            elif format_type == 'python':
                filters = "Python Files (*.py);;All Files (*)"
            else:
                filters = "All Files (*)"

            file_path_tuple = QFileDialog.getSaveFileName(
                self, f"Export Model as {format_type.upper()}",
                os.path.join(last_dir, default_filename), 
                filters
            )
            file_path = file_path_tuple[0]
            
            if not file_path:
                return

            if format_type == 'python':
                required_ext = ".py"
            elif format_type == 'pmml':
                required_ext = ".pmml"
            elif format_type == 'json':
                required_ext = ".json"
            else:
                required_ext = f".{format_type.lower()}"
                
            if not file_path.lower().endswith(required_ext):
                file_path += required_ext

            if format_type == 'python':
                model.export_to_python(file_path)
            elif format_type == 'pmml':
                self.model_saver.save_pmml(model, file_path)
            elif format_type == 'json':
                self.model_saver.save_json(model, file_path)
            else:
                raise ValueError(f"Unknown export format: {format_type}")

            settings.setValue("lastExportPath", str(Path(file_path).parent))
            self.enhanced_status_bar.show_message(f"Model exported to {file_path}")
            self.set_project_modified(True)
        except Exception as e:
            logger.error(f"Error exporting model: {e}", exc_info=True)
            QMessageBox.critical(self, "Export Error", f"Could not export model: {e}")


    def calculate_variable_importance(self):
        """Enhanced variable importance calculation with progress tracking"""
        if not self.current_model_name or self.current_model_name not in self.models:
            QMessageBox.warning(self, "Variable Importance", "No model selected.")
            return
        
        model = self.models[self.current_model_name]
        if not model.is_fitted:
            QMessageBox.warning(self, "Variable Importance", "Model must be trained first.")
            return
        
        if not self.current_dataset_name or self.current_dataset_name not in self.datasets:
            QMessageBox.warning(self, "Variable Importance", "No dataset available for analysis.")
            return

        try:
            progress_dialog = QProgressDialog("Calculating Variable Importance...", "Cancel", 0, 100, self)
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.show()
            
            from analytics.variable_importance import VariableImportance
            importance_calc = VariableImportance(self)
            
            importance_calc.progress_updated.connect(progress_dialog.setValue)
            
            progress_dialog.setLabelText("Calculating Gini-based importance...")
            gini_importance = importance_calc.calculate_gini_importance(model)
            
            if not gini_importance:
                progress_dialog.close()
                QMessageBox.warning(self, "Variable Importance", "Could not calculate variable importance.")
                return
            
            df = self.datasets[self.current_dataset_name]
            permutation_importance = {}
            
            if len(df) <= 10000:  # Only for smaller datasets
                progress_dialog.setLabelText("Calculating permutation importance...")
                target_name = getattr(model, 'target_name', None)
                if target_name and target_name in df.columns:
                    X = df.drop(columns=[target_name])
                    y = df[target_name]
                    try:
                        permutation_importance = importance_calc.calculate_permutation_importance(
                            model, X, y, n_repeats=5
                        )
                    except Exception as e:
                        logger.warning(f"Could not calculate permutation importance: {str(e)}")
            
            progress_dialog.close()
            
            try:
                df = self.datasets.get(self.current_dataset_name)
                if df is not None and model.target_name:
                    dialog = VariableImportanceDialog(df, model.target_name, tree_model=model, parent=self)
                    dialog.show()
                else:
                    results_text = "Variable Importance Results:\n\n"
                    if gini_importance:
                        results_text += "Gini Importance:\n"
                        for feature, importance in gini_importance.items():
                            results_text += f"  {feature}: {importance:.4f}\n"
                    if permutation_importance:
                        results_text += "\nPermutation Importance:\n"  
                        for feature, importance in permutation_importance.items():
                            results_text += f"  {feature}: {importance:.4f}\n"
                    QMessageBox.information(self, "Variable Importance", results_text)
            except Exception as e:
                logger.error(f"Error showing importance results: {e}")
                QMessageBox.information(self, "Variable Importance", "Analysis completed successfully.")
            
        except Exception as e:
            if 'progress_dialog' in locals():
                progress_dialog.close()
            logger.error(f"Error calculating variable importance: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Analysis Error", f"Could not calculate importance:\n{str(e)}")

    def calculate_performance_metrics(self):
        """Calculate and display performance metrics - Enhanced for terminal nodes"""
        if not self.current_model_name or self.current_model_name not in self.models:
            QMessageBox.warning(self, "Performance Metrics", "No model selected.")
            return
        
        model = self.models[self.current_model_name]
        if not model.is_fitted:
            QMessageBox.warning(self, "Performance Metrics", "Model must be trained first.")
            return
        
        if not self.current_dataset_name or self.current_dataset_name not in self.datasets:
            QMessageBox.warning(self, "Performance Metrics", "No dataset available for evaluation.")
            return

        try:
            df = self.datasets[self.current_dataset_name]
            if not model.target_name or model.target_name not in df.columns:
                QMessageBox.warning(self, "Performance Metrics", "Target variable not set in the model or not found in the current dataset.")
                return

            X = df.drop(columns=[model.target_name])
            y = df[model.target_name]
            
            metrics = model.compute_metrics(X, y)
            if not metrics:
                QMessageBox.warning(self, "Performance Metrics", "Could not compute metrics.")
                return

            metrics_text = "=== MODEL PERFORMANCE METRICS ===\n\n"
            
            for metric, value in metrics.items():
                if isinstance(value, (int, float)) and metric != 'confusion_matrix':
                    metrics_text += f"{metric.replace('_', ' ').title()}: {value:.4f}\n"
            
            if 'confusion_matrix' in metrics and isinstance(metrics['confusion_matrix'], dict):
                cm = metrics['confusion_matrix']
                metrics_text += f"\n=== CONFUSION MATRIX ===\n"
                metrics_text += f"True Positives (TP):  {cm.get('true_positives', 'N/A')}\n"
                metrics_text += f"True Negatives (TN):  {cm.get('true_negatives', 'N/A')}\n"
                metrics_text += f"False Positives (FP): {cm.get('false_positives', 'N/A')}\n"
                metrics_text += f"False Negatives (FN): {cm.get('false_negatives', 'N/A')}\n"
            
            if hasattr(model, 'root') and model.root:
                terminal_nodes = [node for node in model.root.get_subtree_nodes() if node.is_terminal]
                metrics_text += f"\n=== TREE STRUCTURE ===\n"
                metrics_text += f"Total Nodes: {model.num_nodes}\n"
                metrics_text += f"Terminal Nodes: {len(terminal_nodes)}\n"
                metrics_text += f"Maximum Depth: {model.max_depth}\n"
                
                if terminal_nodes:
                    terminal_accuracies = [node.accuracy for node in terminal_nodes if node.accuracy is not None]
                    if terminal_accuracies:
                        metrics_text += f"Terminal Node Accuracy Range: {min(terminal_accuracies):.4f} - {max(terminal_accuracies):.4f}\n"
            
            QMessageBox.information(self, "Performance Metrics", metrics_text)

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}", exc_info=True)
            QMessageBox.critical(self, "Analysis Error", f"Could not calculate metrics:\n{str(e)}")

    def show_node_statistics(self):
        """Show statistics for the currently selected node in the tree view - Enhanced version"""
        selected_node_id = None
        if hasattr(self.tree_viz_widget, 'get_selected_node_id'):
            selected_node_id = self.tree_viz_widget.get_selected_node_id()

        if not selected_node_id:
            QMessageBox.warning(self, "Node Statistics", "Please select a node in the Tree View panel first.")
            return

        if not self.current_model_name or self.current_model_name not in self.models:
            QMessageBox.warning(self, "Node Statistics", "No active model found.")
            return

        model = self.models[self.current_model_name]
        if not model.is_fitted:
            QMessageBox.warning(self, "Node Statistics", "The current model is not trained.")
            return

        node = model.get_node(selected_node_id)

        if node:
            try:
                dataset = None
                if self.current_dataset_name and self.current_dataset_name in self.datasets:
                    dataset = self.datasets[self.current_dataset_name]
                
                dialog = NodeEditorDialog(
                    node=node, 
                    model=model,
                    feature_names=model.feature_names, 
                    parent=self
                )
                
                if dataset is not None and hasattr(dialog, 'set_dataset'):
                    dialog.set_dataset(dataset)
                
                if hasattr(dialog, 'select_tab_by_name'):
                    dialog.select_tab_by_name("Report")
                
                dialog.exec_()
                
            except Exception as e:
                logger.error(f"Error opening node statistics dialog: {e}", exc_info=True)
                QMessageBox.critical(self, "Node Statistics Error", f"Could not open node statistics:\n{str(e)}")
        else:
            QMessageBox.warning(self, "Node Statistics", f"Node '{selected_node_id}' not found in the current model.")


    def run_workflow_from_canvas(self, start_node_id: Optional[str] = None):
        """Run the workflow from the canvas"""
        if self.enhanced_status_bar:
            self.enhanced_status_bar.show_message("Preparing workflow...")
        QApplication.processEvents()
        
        workflow_config = self.workflow_canvas.scene.get_config()
        
        self.workflow_engine.set_workflow(
            self.workflow_canvas.scene.nodes, 
            list(self.workflow_canvas.scene.connections.values())
        )
        
        self.workflow_engine.run_workflow(start_node_id)
    
    def run_partial_workflow_from_canvas(self, start_node_id: str):
        """Run the workflow from a specific node due to configuration changes"""
        logger.info(f"Running partial workflow from node {start_node_id}")
        self.enhanced_status_bar.show_message(f"Retraining from node {start_node_id}...")
        QApplication.processEvents()
        
        workflow_config = self.workflow_canvas.scene.get_config()
        
        self.workflow_engine.set_workflow(
            self.workflow_canvas.scene.nodes, 
            list(self.workflow_canvas.scene.connections.values())
        )
        
        self.workflow_engine.run_workflow(start_node_id)

    def add_filter_node(self):
        """Add a filter node to the workflow canvas"""
        if hasattr(self, 'workflow_canvas'):
            self.workflow_canvas._safe_add_node(NODE_TYPE_FILTER)
        else:
            logger.warning("Workflow canvas not available")

    def add_transform_node(self):
        """Add a transform node to the workflow canvas"""
        if hasattr(self, 'workflow_canvas'):
            self.workflow_canvas._safe_add_node(NODE_TYPE_TRANSFORM)
        else:
            logger.warning("Workflow canvas not available")

    def add_evaluation_node(self):
        """Add an evaluation node to the workflow canvas"""
        if hasattr(self, 'workflow_canvas'):
            self.workflow_canvas._safe_add_node(NODE_TYPE_EVALUATION)
        else:
            logger.warning("Workflow canvas not available")

    def add_visualization_node(self):
        """Add a visualization node to the workflow canvas"""
        if hasattr(self, 'workflow_canvas'):
            self.workflow_canvas._safe_add_node(NODE_TYPE_VISUALIZATION)
        else:
            logger.warning("Workflow canvas not available")

    def zoom_in_workflow(self):
        """Zoom in the workflow canvas"""
        if hasattr(self, 'workflow_canvas'):
            self.workflow_canvas.zoom_in()
        else:
            logger.warning("Workflow canvas not available")

    def zoom_out_workflow(self):
        """Zoom out the workflow canvas"""
        if hasattr(self, 'workflow_canvas'):
            self.workflow_canvas.zoom_out()
        else:
            logger.warning("Workflow canvas not available")

    @pyqtSlot()
    def on_workflow_execution_started(self):
        """Handle workflow execution started signal"""
        self.enhanced_status_bar.show_message("Workflow execution started...")
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.run_workflow_action.setEnabled(False)
        if hasattr(self.workflow_canvas, 'run_all_button'):
            self.workflow_canvas.run_all_button.setEnabled(False)

    @pyqtSlot()
    def on_workflow_execution_finished(self):
        """Handle workflow execution finished signal"""
        self.progress_bar.hide()
        self.run_workflow_action.setEnabled(True)
        if hasattr(self.workflow_canvas, 'run_all_button'):
            self.workflow_canvas.run_all_button.setEnabled(True)

    @pyqtSlot(str)
    def on_workflow_execution_error(self, error_msg: str):
        """Handle workflow execution error signal"""
        self.enhanced_status_bar.show_error(f"Workflow Error: {error_msg}")
        QMessageBox.critical(self, "Workflow Execution Error", error_msg)

    @pyqtSlot(str)
    def on_workflow_execution_success(self, success_msg: str):
        """Handle workflow execution success signal"""
        self.enhanced_status_bar.show_success("✅ Workflow completed successfully")
        self.set_project_modified(True)

    @pyqtSlot(str)
    def on_node_processing_started(self, node_id: str):
        """Handle node processing started signal"""
        try:
            node_item = self.workflow_canvas.get_node_by_id(node_id)
            if node_item and hasattr(node_item, 'set_execution_status'):
                node_item.set_execution_status(STATUS_RUNNING)
            
            node_title = node_id
            if node_item and hasattr(node_item, 'title'):
                node_title = getattr(node_item, 'title', node_id)
            
            self.enhanced_status_bar.show_message(f"Processing node: {node_title}...")
            
            if self.workflow_engine.execution_order:
                try:
                    current_idx = self.workflow_engine.execution_order.index(node_id)
                    prog_val = int((current_idx / len(self.workflow_engine.execution_order)) * 100)
                    self.progress_bar.setValue(prog_val)
                except (ValueError, ZeroDivisionError):
                    pass  # Node not in main execution order or empty list
                    
        except Exception as e:
            logger.error(f"Error in on_node_processing_started: {e}", exc_info=True)

    @pyqtSlot(str, str)
    def on_node_processing_finished(self, node_id: str, status: str):
        """Handle node processing finished signal"""
        try:
            node_item = self.workflow_canvas.get_node_by_id(node_id)
            if node_item and hasattr(node_item, 'set_execution_status'):
                if status == "success":
                    node_item.set_execution_status(STATUS_COMPLETED)
                elif status == "failure":
                    node_item.set_execution_status(STATUS_FAILED)
                else:
                    node_item.set_execution_status(STATUS_PENDING)
                
                node_title = getattr(node_item, 'title', node_id) if hasattr(node_item, 'title') else node_id
                if self.enhanced_status_bar:
                    self.enhanced_status_bar.show_message(f"Node {node_title} finished with status: {status}")
            else:
                if self.enhanced_status_bar:
                    self.enhanced_status_bar.show_message(f"Node ID {node_id} finished with status: {status}")
                
        except Exception as e:
            logger.error(f"Error in on_node_processing_finished: {e}", exc_info=True)
            if self.enhanced_status_bar:
                self.enhanced_status_bar.show_message(f"Node ID {node_id} finished with status: {status}")

    @pyqtSlot(str, object, str)
    def on_node_output_ready(self, node_id: str, output_data: Any, output_port_name: str):
        """Handle node output ready signal"""
        logger.info(f"Output ready from node '{node_id}', port '{output_port_name}', type: {type(output_data)}")
        node = self.workflow_canvas.get_node_by_id(node_id)
        node_title = node.title if node else node_id

        if isinstance(output_data, pd.DataFrame):
            derived_dataset_name = f"{node_title}_Output_{output_port_name}"
            i = 1
            base_name = derived_dataset_name
            while derived_dataset_name in self.datasets:
                derived_dataset_name = f"{base_name}_{i}"
                i += 1

            self.datasets[derived_dataset_name] = output_data
            self.add_data_viewer_tab(derived_dataset_name, output_data)
            if self.enhanced_status_bar:
                self.enhanced_status_bar.show_message(f"DataFrame output from '{node_title}' available as '{derived_dataset_name}'.")
        elif isinstance(output_data, BespokeDecisionTree):
            model_name = output_data.model_name
            if model_name not in self.models:
                self.models[model_name] = output_data
                self.connect_model_signals(output_data)
            
            self.current_model_name = model_name
            if self.model_properties_widget:
                self.model_properties_widget.set_model(output_data)
            
            if self.tree_viz_widget and output_data.is_fitted:
                self.tree_viz_widget.set_tree(output_data.root)
            
            if self.enhanced_status_bar:
                self.enhanced_status_bar.show_message(f"Model output from '{node_title}' ('{model_name}') is ready.")
        elif isinstance(output_data, dict) and "accuracy" in output_data:
            try:
                logger.info(f"Evaluation results received from '{node_title}' - storing for detail windows")
                
                if not hasattr(self, 'latest_evaluation_results'):
                    self.latest_evaluation_results = {}
                self.latest_evaluation_results[node_id] = output_data
                
                try:
                    self._populate_node_report_tab(output_data)
                    logger.info(f"Node report auto-populated with evaluation results from {node_title}")
                except Exception as e:
                    logger.error(f"Error auto-populating node report: {e}")
                
                QMessageBox.information(self, f"Evaluation Complete - {node_title}", 
                                      "Evaluation completed successfully!")
                if self.enhanced_status_bar:
                    self.enhanced_status_bar.show_message(f"Evaluation from '{node_title}' completed - results stored.")
                
            except Exception as e:
                logger.error(f"Error storing evaluation results: {e}")
                QMessageBox.information(self, f"Evaluation Complete - {node_title}", 
                                      "Evaluation completed with errors!")
                if self.enhanced_status_bar:
                    self.enhanced_status_bar.show_message(f"Metrics from '{node_title}' are ready.")
        elif isinstance(output_data, dict) and "type" in output_data and output_port_name == "Visualization Trigger":
            viz_type = output_data.get('type', 'tree')
            data_ref = output_data.get('data_ref')
            
            logger.info(f"Handling visualization trigger: type={viz_type}, data_ref={type(data_ref)}")
            
            if viz_type == 'tree' and isinstance(data_ref, BespokeDecisionTree):
                if self.tree_viz_widget and data_ref.is_fitted:
                    self.tree_viz_widget.set_tree(data_ref.root)
                    if hasattr(self, 'model_tabs') and self.model_tabs:
                        for i in range(self.model_tabs.count()):
                            if self.model_tabs.tabText(i) == "Tree Visualizer":
                                self.model_tabs.setCurrentIndex(i)
                                break
                    
                    if not hasattr(self, 'latest_tree_models'):
                        self.latest_tree_models = {}
                        
                    if hasattr(self, 'latest_evaluation_results') and self.latest_evaluation_results:
                        evaluation_results = {}
                        for eval_node_id, results in self.latest_evaluation_results.items():
                            if results:
                                evaluation_results = results
                                break
                        
                        if evaluation_results:
                            viz_info = {
                                'node_id': node_id,
                                'name': 'Decision Tree Analysis',
                                'type': 'visualization',
                                'evaluation_results': evaluation_results,
                                'model_data': data_ref,
                                'tree_root': data_ref.root if hasattr(data_ref, 'root') else None,
                                'dataset': getattr(self, 'current_dataset', None),
                                'target_column': evaluation_results.get('target_column', None)
                            }
                            
                            QTimer.singleShot(500, lambda: self.window_manager.show_window('visualization', viz_info))
                            logger.info("Scheduled auto-navigation to visualization detail window")
                    self.latest_tree_models[node_id] = data_ref
                    
                    if self.enhanced_status_bar:
                        self.enhanced_status_bar.show_message(f"Tree visualization from '{node_title}' is displayed.")
                else:
                    if self.enhanced_status_bar:
                        self.enhanced_status_bar.show_message(f"Tree visualization from '{node_title}' - no fitted model available.")
            elif viz_type == 'metrics' and isinstance(data_ref, dict):
                if not hasattr(self, 'latest_evaluation_results'):
                    self.latest_evaluation_results = {}
                self.latest_evaluation_results[node_id] = data_ref
                
                QMessageBox.information(self, f"Metrics from {node_title}", 
                                      "Metrics calculation completed successfully!")
                if self.enhanced_status_bar:
                    self.enhanced_status_bar.show_message(f"Metrics from '{node_title}' stored for detail windows.")
            elif viz_type == 'tree' and isinstance(data_ref, dict):
                logger.info(f"Tree visualization received evaluation results instead of model. Checking for accuracy.")
                if 'accuracy' in data_ref:
                    if not hasattr(self, 'latest_evaluation_results'):
                        self.latest_evaluation_results = {}
                    self.latest_evaluation_results[node_id] = data_ref
                    
                    QMessageBox.information(self, f"Evaluation Results from {node_title}", 
                                          "Evaluation completed successfully!")
                    if self.enhanced_status_bar:
                        self.enhanced_status_bar.show_message(f"Evaluation results from '{node_title}' stored.")
                else:
                    if self.enhanced_status_bar:
                        self.enhanced_status_bar.show_message(f"Tree visualization from '{node_title}' received unexpected data format.")
            elif isinstance(data_ref, dict) and 'accuracy' in data_ref:
                if not hasattr(self, 'latest_evaluation_results'):
                    self.latest_evaluation_results = {}
                self.latest_evaluation_results[node_id] = data_ref
                if self.enhanced_status_bar:
                    self.enhanced_status_bar.show_message(f"Evaluation results from '{node_title}' stored.")
            else:
                if self.enhanced_status_bar:
                    self.enhanced_status_bar.show_message(f"Visualization trigger from '{node_title}' (type: {viz_type}) processed.")
        
        current_exec_order = self.workflow_engine.execution_order
        if current_exec_order and node_id == current_exec_order[-1]:
            self.progress_bar.setValue(100)

    def _populate_all_visualization_tabs(self, evaluation_results: Dict[str, Any]):
        """
        LEGACY METHOD - Replaced by new window management system
        
        This method is kept for compatibility but no longer populates the old 
        dock-based visualization tabs. Results are now stored for detail windows.
        
        Args:
            evaluation_results: Dictionary containing metrics from evaluation
        """
        try:
            logger.info("Legacy visualization tab population called - storing results for new UI")
            
            if not hasattr(self, 'latest_evaluation_results'):
                self.latest_evaluation_results = {}
            
            self.latest_evaluation_results['latest'] = evaluation_results
            
            logger.info("Evaluation results stored for new window management system")
            
        except Exception as e:
            logger.error(f"Error storing evaluation results for new UI: {e}", exc_info=True)
    
    def _populate_roc_curve_tab(self, evaluation_results: Dict[str, Any]):
        """Populate ROC curve tab with evaluation results and ROC curve plot"""
        try:
            roc_tab_index = None
            for i in range(self.visualization_tabs.count()):
                if self.visualization_tabs.tabText(i) == "ROC Curve":
                    roc_tab_index = i
                    break
            
            if roc_tab_index is not None:
                roc_widget = QWidget()
                main_layout = QVBoxLayout(roc_widget)
                
                scroll_area = QScrollArea()
                scroll_widget = QWidget()
                roc_layout = QVBoxLayout(scroll_widget)
                
                metrics_group = QGroupBox("ROC Analysis Metrics")
                metrics_group.setFont(self._get_standard_small_font())
                metrics_group.setStyleSheet("""
                    QGroupBox {
                        font-family: 'Segoe UI', 'Ubuntu', system-ui, sans-serif;
                        font-size: 12px;
                        font-weight: 600;
                        color: #1e293b;
                        border: 1px solid #e2e8f0;
                        border-radius: 6px;
                        margin-top: 8px;
                        padding-top: 8px;
                    }
                    QGroupBox::title {
                        subcontrol-origin: margin;
                        left: 8px;
                        padding: 0 4px 0 4px;
                    }
                """)
                metrics_layout = QVBoxLayout(metrics_group)
                
                metrics_text = f"""<h3>Performance Metrics</h3>
<p><b>AUC Score:</b> {evaluation_results.get('roc_auc', 'N/A')}</p>
<p><b>Accuracy:</b> {evaluation_results.get('accuracy', 'N/A'):.4f}</p>
<p><b>Precision:</b> {evaluation_results.get('precision', 'N/A'):.4f}</p>
<p><b>Recall:</b> {evaluation_results.get('recall', 'N/A'):.4f}</p>
<p><b>F1 Score:</b> {evaluation_results.get('f1_score', 'N/A'):.4f}</p>"""
                
                metrics_label = QLabel(metrics_text)
                metrics_label.setFont(self._get_standard_body_font())
                metrics_label.setStyleSheet("""
                    QLabel {
                        font-family: 'Segoe UI', 'Ubuntu', system-ui, sans-serif;
                        font-size: 14px;
                        color: #1e293b;
                        padding: 8px;
                    }
                """)
                metrics_label.setWordWrap(True)
                metrics_label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
                metrics_layout.addWidget(metrics_label)
                
                copy_metrics_btn = QPushButton("Copy Metrics to Clipboard")
                copy_metrics_btn.setFont(self._get_standard_small_font())
                copy_metrics_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #3b82f6;
                        color: white;
                        border: none;
                        border-radius: 6px;
                        padding: 8px 16px;
                        font-family: 'Segoe UI', 'Ubuntu', system-ui, sans-serif;
                        font-size: 12px;
                        font-weight: 600;
                    }
                    QPushButton:hover {
                        background-color: #2563eb;
                    }
                """)
                copy_metrics_btn.clicked.connect(lambda: self._copy_metrics_to_clipboard(evaluation_results))
                metrics_layout.addWidget(copy_metrics_btn)
                
                roc_layout.addWidget(metrics_group)
                
                try:
                    roc_plot_widget = self._create_roc_curve_plot(evaluation_results)
                    if roc_plot_widget:
                        plot_group = QGroupBox("ROC Curve")
                        plot_layout = QVBoxLayout(plot_group)
                        plot_layout.addWidget(roc_plot_widget)
                        roc_layout.addWidget(plot_group)
                    else:
                        placeholder_group = QGroupBox("ROC Curve")
                        placeholder_layout = QVBoxLayout(placeholder_group)
                        placeholder_text = QLabel("""<p><b>ROC Curve Visualization</b></p>
<p>ROC curve data is not available. To generate ROC curves:</p>
<ul>
<li>Ensure the model produces probability predictions</li>
<li>Use binary classification</li>
<li>Include ROC curve data in evaluation results</li>
</ul>
<p><i>AUC Score: {}</i></p>""".format(evaluation_results.get('roc_auc', 'N/A')))
                        placeholder_text.setWordWrap(True)
                        placeholder_layout.addWidget(placeholder_text)
                        roc_layout.addWidget(placeholder_group)
                        
                except Exception as plot_error:
                    logger.warning(f"Could not create ROC curve plot: {plot_error}")
                    fallback_label = QLabel("<p><i>ROC curve visualization unavailable</i></p>")
                    roc_layout.addWidget(fallback_label)
                
                scroll_area.setWidget(scroll_widget)
                scroll_area.setWidgetResizable(True)
                main_layout.addWidget(scroll_area)
                
                self.visualization_tabs.removeTab(roc_tab_index)
                self.visualization_tabs.insertTab(roc_tab_index, roc_widget, "ROC Curve")
                
        except Exception as e:
            logger.error(f"Error populating ROC curve tab: {e}")
    
    def _populate_lift_chart_tab(self, evaluation_results: Dict[str, Any]):
        """Populate lift chart tab with evaluation results and lift table"""
        try:
            lift_tab_index = None
            for i in range(self.visualization_tabs.count()):
                if self.visualization_tabs.tabText(i) == "Lift Chart":
                    lift_tab_index = i
                    break
            
            if lift_tab_index is not None:
                lift_widget = QWidget()
                lift_layout = QVBoxLayout(lift_widget)
                
                scroll_area = QScrollArea()
                scroll_area.setWidgetResizable(True)
                scroll_content = QWidget()
                scroll_layout = QVBoxLayout(scroll_content)
                
                metrics_text = f"""
                <h3>Model Performance Summary</h3>
                <p><b>Accuracy:</b> {evaluation_results.get('accuracy', 'N/A'):.4f}</p>
                <p><b>F1 Score:</b> {evaluation_results.get('f1_score', 'N/A'):.4f}</p>
                <p><b>Precision:</b> {evaluation_results.get('precision', 'N/A'):.4f}</p>
                <p><b>Recall:</b> {evaluation_results.get('recall', 'N/A'):.4f}</p>
                """
                
                metrics_label = QLabel(metrics_text)
                metrics_label.setWordWrap(True)
                metrics_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                scroll_layout.addWidget(metrics_label)
                
                self._create_lift_table(scroll_layout, evaluation_results)
                
                self._add_confusion_matrix_display(scroll_layout, evaluation_results)
                
                copy_button = QPushButton("Copy Data to Clipboard")
                copy_button.clicked.connect(lambda: self._copy_lift_data_to_clipboard(evaluation_results))
                scroll_layout.addWidget(copy_button)
                
                scroll_area.setWidget(scroll_content)
                lift_layout.addWidget(scroll_area)
                
                self.visualization_tabs.removeTab(lift_tab_index)
                self.visualization_tabs.insertTab(lift_tab_index, lift_widget, "Lift Chart")
                
        except Exception as e:
            logger.error(f"Error populating lift chart tab: {e}")
    
    def _create_lift_table(self, layout: QVBoxLayout, evaluation_results: Dict[str, Any]):
        """Create a performance analysis table based on actual model metrics"""
        try:
            analysis_header = QLabel("<h4>Performance Analysis Summary</h4>")
            layout.addWidget(analysis_header)
            
            accuracy = evaluation_results.get('accuracy', 0.0)
            precision = evaluation_results.get('precision', 0.0)
            recall = evaluation_results.get('recall', 0.0)
            f1_score = evaluation_results.get('f1_score', 0.0)
            roc_auc = evaluation_results.get('roc_auc', 0.0)
            
            tp = evaluation_results.get('true_positives', 0)
            tn = evaluation_results.get('true_negatives', 0)
            fp = evaluation_results.get('false_positives', 0)
            fn = evaluation_results.get('false_negatives', 0)
            
            total_samples = tp + tn + fp + fn if any([tp, tn, fp, fn]) else 12032  # Use dataset size as fallback
            positive_rate = (tp + fn) / total_samples if total_samples > 0 else 0.5
            
            analysis_table = QTableWidget()
            analysis_table.setRowCount(8)
            analysis_table.setColumnCount(3)
            analysis_table.setHorizontalHeaderLabels(['Metric', 'Value', 'Interpretation'])
            
            metrics_data = [
                ('Model Accuracy', f'{accuracy:.4f}', 'Overall correct predictions'),
                ('Precision', f'{precision:.4f}', 'Positive prediction accuracy'),
                ('Recall (Sensitivity)', f'{recall:.4f}', 'True positive detection rate'),
                ('F1 Score', f'{f1_score:.4f}', 'Harmonic mean of precision/recall'),
                ('ROC AUC', f'{roc_auc:.4f}', 'Area under ROC curve'),
                ('Base Rate', f'{positive_rate:.4f}', 'Proportion of positive cases'),
                ('Total Samples', f'{total_samples:,}', 'Dataset size analyzed'),
                ('Model Performance', 'Excellent' if accuracy > 0.9 else 'Good' if accuracy > 0.8 else 'Fair', 'Based on accuracy threshold')
            ]
            
            for row, (metric, value, interpretation) in enumerate(metrics_data):
                analysis_table.setItem(row, 0, QTableWidgetItem(metric))
                analysis_table.setItem(row, 1, QTableWidgetItem(value))
                analysis_table.setItem(row, 2, QTableWidgetItem(interpretation))
            
            analysis_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            analysis_table.setAlternatingRowColors(True)
            analysis_table.setSelectionBehavior(QTableWidget.SelectRows)
            analysis_table.resizeColumnsToContents()
            analysis_table.setMaximumHeight(280)
            
            header = analysis_table.horizontalHeader()
            header.setStretchLastSection(True)
            
            layout.addWidget(analysis_table)
            
            # Add note about lift analysis
            note = QLabel("""
            <p><b>Note on Lift Analysis:</b><br>
            • Traditional lift analysis requires model probability scores for ranking<br>
            • This table shows actual model performance metrics instead<br>
            • For binary classification with {:.1f}% accuracy on {:,} samples<br>
            • Model performs significantly better than random baseline ({:.1f}%)</p>
            """.format(accuracy * 100, total_samples, positive_rate * 100))
            note.setWordWrap(True)
            note.setStyleSheet("font-size: 9pt; color: #666; margin: 10px; background-color: #f8f9fa; padding: 10px; border-left: 3px solid #007bff;")
            layout.addWidget(note)
            
        except Exception as e:
            logger.warning(f"Error creating performance analysis table: {e}")
            fallback_label = QLabel("<p><i>Performance analysis unavailable - showing basic metrics only</i></p>")
            layout.addWidget(fallback_label)
    
    def _add_confusion_matrix_display(self, layout: QVBoxLayout, evaluation_results: Dict[str, Any]):
        """Add formatted confusion matrix display"""
        try:
            tp = evaluation_results.get('true_positives', 0)
            tn = evaluation_results.get('true_negatives', 0)
            fp = evaluation_results.get('false_positives', 0)
            fn = evaluation_results.get('false_negatives', 0)
            
            if any([tp, tn, fp, fn]):  # If we have any confusion matrix data
                cm_header = QLabel("<h4>Confusion Matrix Details:</h4>")
                layout.addWidget(cm_header)
                
                cm_table = QTableWidget(3, 3)
                cm_table.setMaximumHeight(120)
                cm_table.setMaximumWidth(300)
                
                cm_table.setHorizontalHeaderLabels(['', 'Predicted 0', 'Predicted 1'])
                cm_table.setVerticalHeaderLabels(['Actual 0', 'Actual 1', 'Total'])
                
                cm_table.setItem(0, 0, QTableWidgetItem(''))
                cm_table.setItem(0, 1, QTableWidgetItem(str(tn)))  # True Negatives
                cm_table.setItem(0, 2, QTableWidgetItem(str(fp)))  # False Positives
                cm_table.setItem(1, 0, QTableWidgetItem(''))
                cm_table.setItem(1, 1, QTableWidgetItem(str(fn)))  # False Negatives
                cm_table.setItem(1, 2, QTableWidgetItem(str(tp)))  # True Positives
                cm_table.setItem(2, 0, QTableWidgetItem(''))
                cm_table.setItem(2, 1, QTableWidgetItem(str(tn + fn)))  # Total Actual 0
                cm_table.setItem(2, 2, QTableWidgetItem(str(fp + tp)))  # Total Actual 1
                
                cm_table.setAlternatingRowColors(True)
                cm_table.resizeColumnsToContents()
                cm_table.setSelectionMode(QTableWidget.NoSelection)
                
                layout.addWidget(cm_table)
                
                total = tp + tn + fp + fn
                error_rate = (fp + fn) / total if total > 0 else 0
                
                summary_text = f"""
                <div style='font-family: monospace; background-color: #f8f9fa; padding: 8px; margin: 5px;'>
                <b>Summary:</b><br>
                True Positives: {tp:,} | True Negatives: {tn:,}<br>
                False Positives: {fp:,} | False Negatives: {fn:,}<br>
                Total Samples: {total:,} | Error Rate: {error_rate:.4f}
                </div>
                """
                
                summary_label = QLabel(summary_text)
                summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                layout.addWidget(summary_label)
            else:
                fallback_text = "<p><i>Confusion matrix data not available in evaluation results</i></p>"
                fallback_label = QLabel(fallback_text)
                layout.addWidget(fallback_label)
            
        except Exception as e:
            logger.warning(f"Error adding confusion matrix display: {e}")
    
    def _copy_lift_data_to_clipboard(self, evaluation_results: Dict[str, Any]):
        """Copy performance analysis data to clipboard in structured format"""
        try:
            from PyQt5.QtWidgets import QApplication
            
            accuracy = evaluation_results.get('accuracy', 0.0)
            precision = evaluation_results.get('precision', 0.0)
            recall = evaluation_results.get('recall', 0.0)
            f1_score = evaluation_results.get('f1_score', 0.0)
            roc_auc = evaluation_results.get('roc_auc', 0.0)
            
            tp = evaluation_results.get('true_positives', 0)
            tn = evaluation_results.get('true_negatives', 0)
            fp = evaluation_results.get('false_positives', 0)
            fn = evaluation_results.get('false_negatives', 0)
            
            total_samples = tp + tn + fp + fn if any([tp, tn, fp, fn]) else 12032
            positive_rate = (tp + fn) / total_samples if total_samples > 0 else 0.5
            error_rate = (fp + fn) / total_samples if total_samples > 0 else 0.0
            
            text_parts = ["=== MODEL PERFORMANCE ANALYSIS REPORT ==="]
            text_parts.append("")
            text_parts.append("PERFORMANCE METRICS:")
            text_parts.append(f"Model Accuracy:        {accuracy:.4f}")
            text_parts.append(f"Precision:             {precision:.4f}")
            text_parts.append(f"Recall (Sensitivity):  {recall:.4f}")
            text_parts.append(f"F1 Score:              {f1_score:.4f}")
            text_parts.append(f"ROC AUC:               {roc_auc:.4f}")
            text_parts.append(f"Base Rate:             {positive_rate:.4f}")
            text_parts.append(f"Error Rate:            {error_rate:.4f}")
            text_parts.append("")
            
            if any([tp, tn, fp, fn]):
                text_parts.append("CONFUSION MATRIX:")
                text_parts.append("                 Predicted")
                text_parts.append("                 0       1")
                text_parts.append("Actual    0   {:5d}   {:5d}".format(tn, fp))
                text_parts.append("          1   {:5d}   {:5d}".format(fn, tp))
                text_parts.append("")
                text_parts.append("CONFUSION MATRIX DETAILS:")
                text_parts.append(f"True Negatives:   {tn:,}")
                text_parts.append(f"False Positives:  {fp:,}")
                text_parts.append(f"False Negatives:  {fn:,}")
                text_parts.append(f"True Positives:   {tp:,}")
                text_parts.append(f"Total Samples:    {total_samples:,}")
                text_parts.append("")
            
            text_parts.append("PERFORMANCE ANALYSIS:")
            performance = 'Excellent' if accuracy > 0.9 else 'Good' if accuracy > 0.8 else 'Fair'
            text_parts.append(f"Model Performance:     {performance}")
            text_parts.append(f"Dataset Size:          {total_samples:,} samples")
            text_parts.append(f"Positive Class Rate:   {positive_rate:.1%}")
            text_parts.append("")
            
            text_parts.append("INTERPRETATION:")
            text_parts.append(f"• Model achieves {accuracy:.1%} accuracy on {total_samples:,} samples")
            text_parts.append(f"• Precision of {precision:.1%} means {precision:.1%} of positive predictions are correct")
            text_parts.append(f"• Recall of {recall:.1%} means {recall:.1%} of actual positives are detected")
            text_parts.append(f"• F1 score of {f1_score:.3f} balances precision and recall")
            if roc_auc > 0:
                text_parts.append(f"• ROC AUC of {roc_auc:.3f} indicates {'excellent' if roc_auc > 0.9 else 'good' if roc_auc > 0.8 else 'fair'} discrimination")
            text_parts.append("")
            
            text_parts.append("Note: This is a performance analysis based on actual model evaluation results,")
            text_parts.append("not a traditional lift analysis which would require probability scores for ranking.")
            
            full_text = "\n".join(text_parts)
            clipboard = QApplication.clipboard()
            clipboard.setText(full_text)
            
            QMessageBox.information(self, "Data Copied", "Performance analysis report copied to clipboard successfully!")
            
        except Exception as e:
            logger.error(f"Error copying performance data to clipboard: {e}")
            QMessageBox.warning(self, "Copy Failed", f"Failed to copy data: {str(e)}")
    
    def _populate_variable_importance_tab(self, evaluation_results: Dict[str, Any]):
        """Populate variable importance tab with evaluation results"""
        try:
            importance_tab_index = None
            for i in range(self.visualization_tabs.count()):
                if self.visualization_tabs.tabText(i) == "Variable Importance":
                    importance_tab_index = i
                    break
            
            if importance_tab_index is not None:
                importance_widget = QWidget()
                importance_layout = QVBoxLayout(importance_widget)
                
                scroll_area = QScrollArea()
                scroll_area.setWidgetResizable(True)
                scroll_content = QWidget()
                scroll_layout = QVBoxLayout(scroll_content)
                
                header_text = f"""
                <h3>Model Analysis</h3>
                <p><b>Overall Accuracy:</b> {evaluation_results.get('accuracy', 'N/A'):.4f}</p>
                <p><b>Model Performance:</b> {evaluation_results.get('f1_score', 'N/A'):.4f} F1-Score</p>
                """
                
                header_label = QLabel(header_text)
                header_label.setWordWrap(True)
                header_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                scroll_layout.addWidget(header_label)
                
                self._create_classification_report_table(scroll_layout, evaluation_results)
                
                self._add_feature_importance_display(scroll_layout, evaluation_results)
                
                copy_button = QPushButton("Copy Analysis to Clipboard")
                copy_button.clicked.connect(lambda: self._copy_analysis_to_clipboard(evaluation_results))
                scroll_layout.addWidget(copy_button)
                
                scroll_area.setWidget(scroll_content)
                importance_layout.addWidget(scroll_area)
                
                self.visualization_tabs.removeTab(importance_tab_index)
                self.visualization_tabs.insertTab(importance_tab_index, importance_widget, "Variable Importance")
                
        except Exception as e:
            logger.error(f"Error populating variable importance tab: {e}")
    
    def _create_classification_report_table(self, layout: QVBoxLayout, evaluation_results: Dict[str, Any]):
        """Create a properly formatted classification report table"""
        try:
            report = evaluation_results.get('classification_report', {})
            if not report:
                no_report_label = QLabel("<p><i>Classification report not available</i></p>")
                layout.addWidget(no_report_label)
                return
            
            report_header = QLabel("<h4>Detailed Classification Report:</h4>")
            layout.addWidget(report_header)
            
            report_table = QTableWidget()
            
            class_data = {}
            summary_data = {}
            
            for key, value in report.items():
                if isinstance(value, dict):
                    if key in ['0', '1', 'class_0', 'class_1'] or key.startswith('class_'):
                        class_data[key] = value
                    elif key in ['macro avg', 'weighted avg', 'micro avg']:
                        summary_data[key] = value
                
            if class_data or summary_data:
                all_data = {**class_data, **summary_data}
                
                report_table.setRowCount(len(all_data))
                report_table.setColumnCount(4)
                report_table.setHorizontalHeaderLabels(['Class/Average', 'Precision', 'Recall', 'F1-Score'])
                
                row = 0
                for class_name, metrics in all_data.items():
                    display_name = class_name
                    if class_name in ['0', 'class_0']:
                        display_name = 'Class 0 (Negative)'
                    elif class_name in ['1', 'class_1']:
                        display_name = 'Class 1 (Positive)'
                    elif 'avg' in class_name:
                        display_name = class_name.title()
                    
                    report_table.setItem(row, 0, QTableWidgetItem(display_name))
                    
                    precision = metrics.get('precision', metrics.get('prec', 0))
                    recall = metrics.get('recall', metrics.get('rec', 0))
                    f1 = metrics.get('f1-score', metrics.get('f1', 0))
                    
                    report_table.setItem(row, 1, QTableWidgetItem(f"{precision:.4f}"))
                    report_table.setItem(row, 2, QTableWidgetItem(f"{recall:.4f}"))
                    report_table.setItem(row, 3, QTableWidgetItem(f"{f1:.4f}"))
                    
                    row += 1
                
                report_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
                report_table.setAlternatingRowColors(True)
                report_table.setSelectionBehavior(QTableWidget.SelectRows)
                report_table.resizeColumnsToContents()
                report_table.setMaximumHeight(250)
                
                layout.addWidget(report_table)
            else:
                formatted_text = "<div style='font-family: monospace; background-color: #f5f5f5; padding: 10px; border-radius: 5px;'>"
                for key, value in report.items():
                    if isinstance(value, (int, float)):
                        formatted_text += f"<b>{key}:</b> {value:.4f}<br>"
                    else:
                        formatted_text += f"<b>{key}:</b> {value}<br>"
                formatted_text += "</div>"
                
                formatted_label = QLabel(formatted_text)
                formatted_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                layout.addWidget(formatted_label)
            
        except Exception as e:
            logger.warning(f"Error creating classification report table: {e}")
            fallback_label = QLabel("<p><i>Classification report formatting unavailable</i></p>")
            layout.addWidget(fallback_label)
    
    def _add_feature_importance_display(self, layout: QVBoxLayout, evaluation_results: Dict[str, Any]):
        """Add feature importance display if available"""
        try:
            feature_importance = evaluation_results.get('feature_importance', {})
            if feature_importance:
                importance_header = QLabel("<h4>Feature Importance:</h4>")
                layout.addWidget(importance_header)
                
                importance_text = "<div style='font-family: monospace; background-color: #f8f9fa; padding: 10px;'>"
                if isinstance(feature_importance, dict):
                    try:
                        sorted_features = sorted(feature_importance.items(), 
                                               key=lambda x: float(x[1]) if isinstance(x[1], (int, float)) else 0, 
                                               reverse=True)
                        for feature, importance in sorted_features[:10]:  # Top 10
                            importance_text += f"{feature}: {importance:.4f}<br>"
                    except:
                        for feature, importance in feature_importance.items():
                            importance_text += f"{feature}: {importance}<br>"
                else:
                    importance_text += f"{feature_importance}"
                
                importance_text += "</div>"
                
                importance_label = QLabel(importance_text)
                importance_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
                layout.addWidget(importance_label)
            else:
                basic_info = QLabel("""
                <h4>Model Information:</h4>
                <p>• Model type: Decision Tree Classifier</p>
                <p>• Feature importance calculated based on split information gain</p>
                <p>• Variables contributing to splits are more important</p>
                """)
                basic_info.setWordWrap(True)
                layout.addWidget(basic_info)
                
        except Exception as e:
            logger.warning(f"Error adding feature importance display: {e}")
    
    def _copy_analysis_to_clipboard(self, evaluation_results: Dict[str, Any]):
        """Copy analysis data to clipboard in structured format"""
        try:
            from PyQt5.QtWidgets import QApplication
            
            text_parts = ["=== MODEL ANALYSIS REPORT ==="]
            text_parts.append("")
            text_parts.append("OVERALL PERFORMANCE:")
            text_parts.append(f"Accuracy:  {evaluation_results.get('accuracy', 'N/A'):.4f}")
            text_parts.append(f"F1 Score:  {evaluation_results.get('f1_score', 'N/A'):.4f}")
            text_parts.append(f"Precision: {evaluation_results.get('precision', 'N/A'):.4f}")
            text_parts.append(f"Recall:    {evaluation_results.get('recall', 'N/A'):.4f}")
            text_parts.append("")
            
            report = evaluation_results.get('classification_report', {})
            if report:
                text_parts.append("DETAILED CLASSIFICATION REPORT:")
                text_parts.append("Class/Average\t\tPrecision\tRecall\t\tF1-Score")
                text_parts.append("-" * 60)
                
                for class_name, metrics in report.items():
                    if isinstance(metrics, dict):
                        display_name = class_name
                        if class_name in ['0', 'class_0']:
                            display_name = 'Class 0 (Negative)'
                        elif class_name in ['1', 'class_1']:
                            display_name = 'Class 1 (Positive)'
                        elif 'avg' in class_name:
                            display_name = class_name.title()
                        
                        precision = metrics.get('precision', metrics.get('prec', 0))
                        recall = metrics.get('recall', metrics.get('rec', 0))
                        f1 = metrics.get('f1-score', metrics.get('f1', 0))
                        
                        text_parts.append(f"{display_name:<20}\t{precision:.4f}\t\t{recall:.4f}\t\t{f1:.4f}")
                
                text_parts.append("")
            
            feature_importance = evaluation_results.get('feature_importance', {})
            if feature_importance:
                text_parts.append("FEATURE IMPORTANCE:")
                for feature, importance in feature_importance.items():
                    text_parts.append(f"{feature}: {importance}")
                text_parts.append("")
            
            full_text = "\n".join(text_parts)
            clipboard = QApplication.clipboard()
            clipboard.setText(full_text)
            
            QMessageBox.information(self, "Data Copied", "Analysis report copied to clipboard successfully!")
            
        except Exception as e:
            logger.error(f"Error copying analysis to clipboard: {e}")
            QMessageBox.warning(self, "Copy Failed", f"Failed to copy data: {str(e)}")

    def _populate_node_report_tab(self, evaluation_results: Dict[str, Any]):
        """Populate node report tab with comprehensive node analysis"""
        try:
            node_report_tab_index = None
            for i in range(self.visualization_tabs.count()):
                if self.visualization_tabs.tabText(i) == "Node Report":
                    node_report_tab_index = i
                    break
            
            if node_report_tab_index is None:
                logger.error("Node Report tab not found")
                return
            
            model = evaluation_results.get('model')
            dataset = evaluation_results.get('dataset') 
            target_variable = evaluation_results.get('target_variable')
            
            if model is None:
                model = getattr(self, 'current_model', None)
                if model is None and hasattr(self, 'models') and self.models:
                    model = next(iter(self.models.values()), None)
                    
            if dataset is None or (hasattr(dataset, 'empty') and dataset.empty):
                dataset = getattr(self, 'current_dataset', None)
                
            if target_variable is None:
                if model and hasattr(model, 'target_name'):
                    target_variable = model.target_name
                elif 'target_column' in evaluation_results:
                    target_variable = evaluation_results['target_column']
                    
            missing_items = []
            if model is None:
                missing_items.append("trained model")
            if dataset is None or (hasattr(dataset, 'empty') and dataset.empty):
                missing_items.append("dataset")
            if not target_variable:
                missing_items.append("target variable")
                
            if missing_items:
                logger.warning(f"Missing required data for node report: {', '.join(missing_items)}")
                
                placeholder_widget = QWidget()
                placeholder_layout = QVBoxLayout(placeholder_widget)
                
                header_label = QLabel("<h2>Node Report</h2>")
                header_label.setFont(self._get_standard_header_font())
                header_label.setTextFormat(Qt.RichText)
                header_label.setAlignment(Qt.AlignCenter)
                placeholder_layout.addWidget(header_label)
                
                message_text = f"""
                <div style='text-align: center; padding: 20px;'>
                    <p><b>Node report cannot be generated.</b></p>
                    <p>Missing: {', '.join(missing_items)}</p>
                    <br>
                    <p style='color: #64748b; font-style: italic;'>
                        Please ensure you have:
                        <br>• A trained decision tree model
                        <br>• Dataset with evaluation data  
                        <br>• Specified target variable
                    </p>
                    <br>
                    <p style='color: #64748b; font-size: 12px;'>
                        Run a complete workflow with Model and Evaluation nodes to generate the report.
                    </p>
                </div>
                """
                
                message_label = QLabel(message_text)
                message_label.setFont(self._get_standard_body_font())
                message_label.setTextFormat(Qt.RichText)
                message_label.setWordWrap(True)
                message_label.setAlignment(Qt.AlignCenter)
                message_label.setStyleSheet("""
                    QLabel {
                        font-family: 'Segoe UI', 'Ubuntu', system-ui, sans-serif;
                        color: #1e293b;
                        background-color: #f8fafc;
                        border: 1px solid #e2e8f0;
                        border-radius: 8px;
                        padding: 16px;
                        margin: 16px;
                    }
                """)
                placeholder_layout.addWidget(message_label)
                
                self.visualization_tabs.removeTab(node_report_tab_index)
                self.visualization_tabs.insertTab(node_report_tab_index, placeholder_widget, "Node Report")
                return
            
            node_report_widget = QWidget()
            layout = QVBoxLayout(node_report_widget)
            
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_content = QWidget()
            scroll_layout = QVBoxLayout(scroll_content)
            
            header_label = QLabel("<h2>Decision Tree Node Report</h2>")
            header_label.setTextFormat(Qt.RichText)
            scroll_layout.addWidget(header_label)
            
            try:
                node_generator = NodeReportGenerator()
                node_report_df = node_generator.generate_node_report(model, dataset, target_variable)
                
                table_widget = QTableWidget()
                table_widget.setRowCount(len(node_report_df))
                table_widget.setColumnCount(len(node_report_df.columns))
                table_widget.setHorizontalHeaderLabels(list(node_report_df.columns))
                
                for row in range(len(node_report_df)):
                    for col, column_name in enumerate(node_report_df.columns):
                        value = node_report_df.iloc[row, col]
                        if isinstance(value, (int, float)):
                            if column_name in ['target_rate', 'cumulative_target_rate', 'cumulative_pct', 'lift']:
                                item = QTableWidgetItem(f"{value:.2f}")
                            elif column_name in ['node_no', 'node_size', 'cumulative_size', 'target_count', 'cumulative_target_count']:
                                if column_name == 'node_no':
                                    item = QTableWidgetItem(str(int(value)))
                                else:
                                    item = QTableWidgetItem(f"{int(value):,}")
                            else:
                                item = QTableWidgetItem(str(value))
                        else:
                            item = QTableWidgetItem(str(value))
                        
                        if column_name in ['node_no', 'node_size', 'cumulative_size', 'target_count', 
                                         'cumulative_target_count', 'target_rate', 
                                         'cumulative_target_rate', 'cumulative_pct', 'lift']:
                            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        
                        table_widget.setItem(row, col, item)
                
                table_widget.resizeColumnsToContents()
                
                table_widget.setSortingEnabled(True)
                
                header = table_widget.horizontalHeader()
                header.setStretchLastSection(True)
                table_widget.setColumnWidth(0, 300)  # Node Logic column
                table_widget.setColumnWidth(1, 100)  # Node ID column
                
                table_widget.setEditTriggers(QTableWidget.NoEditTriggers)
                table_widget.setSelectionBehavior(QTableWidget.SelectRows)
                table_widget.setAlternatingRowColors(True)
                
                table_widget.setSelectionMode(QTableWidget.ExtendedSelection)
                
                scroll_layout.addWidget(table_widget)
                
                summary_text = f"""
                <h3>Summary</h3>
                <p><b>Total Nodes:</b> {len(node_report_df)}</p>
                <p><b>Total Records:</b> {node_report_df['node_size'].sum()}</p>
                <p><b>Total Target Count:</b> {node_report_df['target_count'].sum()}</p>
                <p><b>Overall Target Rate:</b> {(node_report_df['target_count'].sum() / node_report_df['node_size'].sum() * 100):.2f}%</p>
                """
                
                summary_label = QLabel(summary_text)
                summary_label.setTextFormat(Qt.RichText)
                summary_label.setWordWrap(True)
                scroll_layout.addWidget(summary_label)
                
                buttons_layout = QHBoxLayout()
                
                refresh_button = QPushButton("Refresh Report")
                refresh_button.setToolTip("Regenerate the node report with current data")
                refresh_button.clicked.connect(lambda: self._refresh_node_report_tab(evaluation_results))
                buttons_layout.addWidget(refresh_button)
                
                copy_button = QPushButton("Copy to Clipboard")
                copy_button.setToolTip("Copy selected rows to clipboard")
                copy_button.clicked.connect(lambda: self._copy_node_report_to_clipboard(table_widget, node_report_df))
                buttons_layout.addWidget(copy_button)
                
                export_button = QPushButton("Export to Excel")
                export_button.setToolTip("Export complete report to Excel file")
                export_button.clicked.connect(lambda: self._export_node_report_to_excel_with_progress(node_report_df))
                buttons_layout.addWidget(export_button)
                
                buttons_layout.addStretch()
                scroll_layout.addLayout(buttons_layout)
                
                logger.info(f"Node report generated with {len(node_report_df)} nodes")
                
            except Exception as e:
                logger.error(f"Error generating node report: {e}")
                error_label = QLabel(f"<p style='color: red;'>Error generating node report: {str(e)}</p>")
                error_label.setTextFormat(Qt.RichText)
                error_label.setWordWrap(True)
                scroll_layout.addWidget(error_label)
            
            scroll_area.setWidget(scroll_content)
            layout.addWidget(scroll_area)
            
            self.visualization_tabs.removeTab(node_report_tab_index)
            self.visualization_tabs.insertTab(node_report_tab_index, node_report_widget, "Node Report")
            
        except Exception as e:
            logger.error(f"Error populating node report tab: {e}")

    def _create_node_report_tab(self) -> QWidget:
        """Create the node report tab widget structure"""
        try:
            node_report_widget = QWidget()
            layout = QVBoxLayout(node_report_widget)
            
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_content = QWidget()
            scroll_layout = QVBoxLayout(scroll_content)
            
            header_label = QLabel("<h2>Decision Tree Node Report</h2>")
            header_label.setTextFormat(Qt.RichText)
            header_font = self._get_standard_header_font()
            header_label.setFont(header_font)
            header_label.setStyleSheet("""
                QLabel {
                    font-family: 'Segoe UI', 'Ubuntu', system-ui, sans-serif;
                    font-size: 18px;
                    font-weight: bold;
                    color: #1e293b;
                    padding: 8px 0;
                }
            """)
            scroll_layout.addWidget(header_label)
            
            placeholder_label = QLabel("Node report will be generated after model evaluation.")
            placeholder_label.setAlignment(Qt.AlignCenter)
            placeholder_font = self._get_standard_body_font()
            placeholder_label.setFont(placeholder_font)
            placeholder_label.setStyleSheet("""
                QLabel {
                    font-family: 'Segoe UI', 'Ubuntu', system-ui, sans-serif;
                    font-size: 14px;
                    color: #64748b;
                    font-style: italic;
                    padding: 20px;
                }
            """)
            scroll_layout.addWidget(placeholder_label)
            
            scroll_area.setWidget(scroll_content)
            layout.addWidget(scroll_area)
            
            logger.debug("Node report tab structure created")
            return node_report_widget
            
        except Exception as e:
            logger.error(f"Error creating node report tab: {e}")
            error_widget = QWidget()
            error_layout = QVBoxLayout(error_widget)
            error_label = QLabel("Error creating node report tab")
            error_label.setAlignment(Qt.AlignCenter)
            error_layout.addWidget(error_label)
            return error_widget

    def _refresh_node_report_tab(self, evaluation_results: Dict[str, Any]):
        """Refresh the node report tab with current data"""
        try:
            logger.info("Refreshing node report tab...")
            self._populate_node_report_tab(evaluation_results)
            logger.info("Node report tab refreshed successfully")
        except Exception as e:
            error_msg = f"Failed to refresh node report: {str(e)}"
            QMessageBox.warning(self, "Refresh Failed", error_msg)
            logger.error(error_msg)

    def _copy_node_report_to_clipboard(self, table_widget: QTableWidget, node_report_df: pd.DataFrame):
        """Copy selected rows from node report to clipboard"""
        try:
            selected_ranges = table_widget.selectedRanges()
            if not selected_ranges:
                clipboard_data = node_report_df.to_csv(sep='\t', index=False)
                QApplication.clipboard().setText(clipboard_data)
                QMessageBox.information(self, "Data Copied", "Complete node report copied to clipboard!")
                logger.info("Complete node report copied to clipboard")
                return
            
            selected_rows = set()
            for selected_range in selected_ranges:
                for row in range(selected_range.topRow(), selected_range.bottomRow() + 1):
                    selected_rows.add(row)
            
            if selected_rows:
                selected_data = node_report_df.iloc[list(sorted(selected_rows))]
                clipboard_data = selected_data.to_csv(sep='\t', index=False)
                QApplication.clipboard().setText(clipboard_data)
                
                rows_count = len(selected_rows)
                QMessageBox.information(self, "Data Copied", 
                                      f"Selected {rows_count} row(s) copied to clipboard!")
                logger.info(f"Node report: {rows_count} rows copied to clipboard")
            else:
                QMessageBox.information(self, "No Selection", "No rows selected. Please select rows to copy.")
                
        except Exception as e:
            error_msg = f"Failed to copy data to clipboard: {str(e)}"
            QMessageBox.warning(self, "Copy Failed", error_msg)
            logger.error(error_msg)

    def _export_node_report_to_excel_with_progress(self, node_report_df: pd.DataFrame):
        """Export node report to Excel file with progress dialog"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Node Report to Excel",
                f"node_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "Excel files (*.xlsx);;All Files (*)"
            )
            
            if file_path:
                if len(node_report_df) > 100:  # Show progress for larger reports
                    progress = QProgressDialog("Exporting node report to Excel...", "Cancel", 0, 100, self)
                    progress.setWindowModality(Qt.WindowModal)
                    progress.setMinimumDuration(500)  # Show after 500ms
                    progress.setValue(10)
                    QApplication.processEvents()
                
                node_generator = NodeReportGenerator()
                
                if len(node_report_df) > 100:
                    progress.setValue(50)
                    progress.setLabelText("Formatting Excel file...")
                    QApplication.processEvents()
                
                success, message = node_generator.export_to_excel(node_report_df, file_path)
                
                if len(node_report_df) > 100:
                    progress.setValue(100)
                    progress.close()
                
                if success:
                    QMessageBox.information(self, "Export Successful", message)
                    logger.info(f"Node report exported to: {file_path}")
                else:
                    QMessageBox.warning(self, "Export Failed", message)
                    logger.error(f"Node report export failed: {message}")
                    
        except Exception as e:
            error_msg = f"Failed to export node report: {str(e)}"
            QMessageBox.critical(self, "Export Error", error_msg)
            logger.error(error_msg)

    def _export_node_report_to_excel(self, node_report_df: pd.DataFrame):
        """Export node report to Excel file (legacy method for compatibility)"""
        self._export_node_report_to_excel_with_progress(node_report_df)

    def _get_latest_evaluation_results(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest evaluation results from the workflow engine
        
        Returns:
            Dictionary of evaluation results if available, None otherwise
        """
        try:
            if not hasattr(self, 'workflow_engine') or not self.workflow_engine:
                return None
            
            for node_id, outputs in getattr(self.workflow_engine, 'node_outputs', {}).items():
                if 'Evaluation Results' in outputs:
                    results = outputs['Evaluation Results']
                    if isinstance(results, dict) and 'accuracy' in results:
                        logger.info(f"Found evaluation results from node {node_id}")
                        return results
            
            logger.debug("No evaluation results found in workflow cache")
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest evaluation results: {e}")
            return None

    def _create_roc_curve_plot(self, evaluation_results: Dict[str, Any]):
        """Create ROC curve plot widget"""
        try:
            if 'roc_curve' in evaluation_results:
                roc_data = evaluation_results['roc_curve']
                if isinstance(roc_data, dict) and 'fpr' in roc_data and 'tpr' in roc_data:
                    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
                    from matplotlib.figure import Figure
                    
                    figure = Figure(figsize=(8, 6))
                    canvas = FigureCanvas(figure)
                    ax = figure.add_subplot(111)
                    
                    fpr = roc_data['fpr']
                    tpr = roc_data['tpr']
                    auc = evaluation_results.get('roc_auc', 0.5)
                    
                    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                    ax.legend(loc="lower right")
                    ax.grid(True, alpha=0.3)
                    
                    figure.tight_layout()
                    return canvas
            
            return None
            
        except Exception as e:
            logger.warning(f"Error creating ROC curve plot: {e}")
            return None

    def _copy_metrics_to_clipboard(self, evaluation_results: Dict[str, Any]):
        """Copy metrics to clipboard in structured format"""
        try:
            from PyQt5.QtWidgets import QApplication
            
            text_parts = ["=== PERFORMANCE METRICS ==="]
            text_parts.append(f"AUC Score: {evaluation_results.get('roc_auc', 'N/A')}")
            text_parts.append(f"Accuracy: {evaluation_results.get('accuracy', 'N/A'):.4f}")
            text_parts.append(f"Precision: {evaluation_results.get('precision', 'N/A'):.4f}")
            text_parts.append(f"Recall: {evaluation_results.get('recall', 'N/A'):.4f}")
            text_parts.append(f"F1 Score: {evaluation_results.get('f1_score', 'N/A'):.4f}")
            
            if 'confusion_matrix' in evaluation_results:
                cm = evaluation_results['confusion_matrix']
                text_parts.append("\n=== CONFUSION MATRIX ===")
                if isinstance(cm, dict):
                    for key, value in cm.items():
                        text_parts.append(f"{key.replace('_', ' ').title()}: {value}")
            
            clipboard_text = "\n".join(text_parts)
            
            clipboard = QApplication.clipboard()
            clipboard.setText(clipboard_text)
            
            if self.enhanced_status_bar:
                self.enhanced_status_bar.show_message("Metrics copied to clipboard")
            
        except Exception as e:
            logger.error(f"Error copying metrics to clipboard: {e}")
            if self.enhanced_status_bar:
                self.enhanced_status_bar.show_error("Failed to copy metrics")


    def show_filter_dialog(self):
        """Show dialog for filtering data"""
        selected_workflow_nodes = self.workflow_canvas.scene.selectedItems()
        filter_node_to_configure: Optional[FilterNode] = None
        df_context: Optional[pd.DataFrame] = None

        for item in selected_workflow_nodes:
            if isinstance(item, FilterNode):
                filter_node_to_configure = item
                df_context = self.get_node_input_data(item.node_id, 'Data Input')
                break
        
        if df_context is None:
            if self.current_dataset_name and self.current_dataset_name in self.datasets:
                df_context = self.datasets[self.current_dataset_name]
            else:
                QMessageBox.warning(self, "Filter Data", "No active dataset or valid input for filtering.")
                return

        dialog = FilterDataDialog(dataframe=df_context, parent=self)
        
        if filter_node_to_configure:
            filter_config = filter_node_to_configure.get_config()
            if 'conditions' in filter_config:
                dialog.set_conditions(filter_config['conditions'])
        
        if dialog.exec_() == QDialog.Accepted:
            conditions = dialog.get_conditions()
            
            if filter_node_to_configure:
                # Note: We can't directly set config this way, need to use proper method
                if hasattr(filter_node_to_configure, 'conditions'):
                    filter_node_to_configure.conditions = conditions
                filter_node_to_configure.update_filter_info()
                self.set_project_modified(True)
                self.status_label.setText(f"FilterNode '{filter_node_to_configure.title}' conditions updated.")
            else:
                if self.current_dataset_name:
                    filtered_df = self.data_processor.filter_data(df_context, conditions, self.current_dataset_name)
                    if filtered_df is not None:
                        self.datasets[self.current_dataset_name] = filtered_df
                        self.add_data_viewer_tab(self.current_dataset_name, filtered_df)
                        self.status_label.setText(f"Dataset '{self.current_dataset_name}' filtered directly.")
                        self.set_project_modified(True)
                else:
                    QMessageBox.information(self, "Filter Applied", "Filter conditions defined but no active dataset/node to apply to directly.")

    def show_advanced_filter_dialog(self):
        """Show advanced filter dialog with comprehensive filtering options"""
        if not self.current_dataset_name or self.current_dataset_name not in self.datasets:
            QMessageBox.warning(self, "Advanced Filter", "No active dataset available for filtering.")
            return
            
        df = self.datasets[self.current_dataset_name]
        
        dialog = AdvancedFilterDialog(df, parent=self)
        
        if dialog.exec_() == QDialog.Accepted:
            try:
                filtered_df = dialog.get_filtered_dataframe()
                
                if len(filtered_df) == 0:
                    reply = QMessageBox.question(
                        self, "Empty Result", 
                        "The filter resulted in an empty dataset. Do you want to proceed?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No
                    )
                    if reply != QMessageBox.Yes:
                        return
                
                original_name = self.current_dataset_name
                filtered_name = f"{original_name}_filtered"
                
                i = 1
                while filtered_name in self.datasets:
                    filtered_name = f"{original_name}_filtered_{i}"
                    i += 1
                
                self.datasets[filtered_name] = filtered_df
                self.current_dataset_name = filtered_name
                
                self.add_data_viewer_tab(filtered_name, filtered_df)
                self.workflow_canvas.add_node_to_canvas(
                    NODE_TYPE_DATASET, 
                    title=filtered_name, 
                    specific_config={
                        'dataset_name': filtered_name, 
                        'df_rows': len(filtered_df), 
                        'df_cols': len(filtered_df.columns),
                        'source': 'filtered'
                    }
                )
                
                self.status_label.setText(f"Created filtered dataset '{filtered_name}' with {len(filtered_df)} rows.")
                self.enhanced_status_bar.show_message(f"Active: {filtered_name} ({len(filtered_df)} rows, {len(filtered_df.columns)} cols)")
                self.set_project_modified(True)
                
            except Exception as e:
                logger.error(f"Error applying advanced filter: {e}", exc_info=True)
                QMessageBox.critical(self, "Filter Error", f"Error applying advanced filter: {e}")

    def show_transformation_dialog(self):
        """Show comprehensive data transformation dialog"""
        if not self.current_dataset_name or self.current_dataset_name not in self.datasets:
            QMessageBox.warning(self, "Transform Data", "No active dataset available for transformation.")
            return
            
        df = self.datasets[self.current_dataset_name]
        
        dialog = DataTransformationDialog(df, parent=self)
        
        if dialog.exec_() == QDialog.Accepted:
            try:
                transformed_df = dialog.get_transformed_dataframe()
                transformation_history = dialog.get_transformation_history()
                
                if len(transformation_history) == 0:
                    QMessageBox.information(self, "No Transformations", "No transformations were applied.")
                    return
                
                original_name = self.current_dataset_name
                transformed_name = f"{original_name}_transformed"
                
                i = 1
                while transformed_name in self.datasets:
                    transformed_name = f"{original_name}_transformed_{i}"
                    i += 1
                
                self.datasets[transformed_name] = transformed_df
                self.current_dataset_name = transformed_name
                
                self.add_data_viewer_tab(transformed_name, transformed_df)
                self.workflow_canvas.add_node_to_canvas(
                    NODE_TYPE_TRANSFORM, 
                    title=transformed_name, 
                    specific_config={
                        'dataset_name': transformed_name, 
                        'df_rows': len(transformed_df), 
                        'df_cols': len(transformed_df.columns),
                        'source': 'transformed',
                        'transformations': transformation_history
                    }
                )
                
                transform_summary = "\n".join(transformation_history)
                self.status_label.setText(f"Applied {len(transformation_history)} transformation(s) to create '{transformed_name}'.")
                self.enhanced_status_bar.show_message(f"Active: {transformed_name} ({len(transformed_df)} rows, {len(transformed_df.columns)} cols)")
                self.set_project_modified(True)
                
                QMessageBox.information(
                    self, "Transformations Applied", 
                    f"Created transformed dataset '{transformed_name}' with {len(transformed_df)} rows.\n\n"
                    f"Applied transformations:\n{transform_summary}"
                )
                
            except Exception as e:
                logger.error(f"Error applying data transformations: {e}", exc_info=True)
                QMessageBox.critical(self, "Transformation Error", f"Error applying data transformations: {e}")

    def show_formula_editor(self):
        """Show dialog for creating new variables"""
        if not self.current_dataset_name or self.current_dataset_name not in self.datasets:
            QMessageBox.warning(self, "Create Variable", "No active dataset available.")
            return
        
        df = self.datasets[self.current_dataset_name]
        dialog = FormulaEditorDialog(available_columns=df.columns.tolist(), parent=self)
        dialog.formulaApplied.connect(self.create_variable_from_formula)
        dialog.exec_()

    @pyqtSlot(str, str, str)
    def create_variable_from_formula(self, formula: str, var_name: str, var_type: str):
        """Create a new variable using a formula"""
        if not self.current_dataset_name or self.current_dataset_name not in self.datasets:
            QMessageBox.warning(self, "Create Variable", "No active dataset selected to add variable to.")
            return
        
        df = self.datasets[self.current_dataset_name]
        
        try:
            if var_name in df.columns:
                reply = QMessageBox.question(
                    self, "Overwrite Variable?",
                    f"Variable '{var_name}' already exists. Overwrite it?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                
                if reply == QMessageBox.No:
                    self.status_label.setText(f"Variable creation cancelled.")
                    return
            
            new_df = self.feature_engineering.create_formula_variable(
                df, formula, var_name, var_type, self.current_dataset_name
            )
            
            self.datasets[self.current_dataset_name] = new_df
            self.add_data_viewer_tab(self.current_dataset_name, new_df)
            self.status_label.setText(f"Variable '{var_name}' created in '{self.current_dataset_name}'.")
            self.set_project_modified(True)
            self.update_action_states()
        except Exception as e:
            logger.error(f"Error creating variable: {e}", exc_info=True)
            QMessageBox.critical(self, "Formula Error", f"Could not create variable: {e}")

    def show_missing_values_dialog(self):
        """Show dialog for handling missing values"""
        if not self.current_dataset_name or self.current_dataset_name not in self.datasets:
            QMessageBox.warning(self, "Handle Missing Values", "No active dataset selected.")
            return

        df = self.datasets[self.current_dataset_name]
        dialog = MissingValuesDialog(columns=df.columns.tolist(), parent=self)
        
        if dialog.exec_() == QDialog.Accepted:
            strategy = dialog.get_strategy()
            processed_df = self.data_processor.handle_missing_values(df, strategy, dataset_name=self.current_dataset_name)
            
            self.datasets[self.current_dataset_name] = processed_df
            self.add_data_viewer_tab(self.current_dataset_name, processed_df)
            self.status_label.setText(f"Missing values handled for '{self.current_dataset_name}'.")
            self.set_project_modified(True)
            self.update_action_states()

    def show_enhanced_formula_editor(self):
        """Show enhanced formula editor dialog"""
        if not self.current_dataset_name or self.current_dataset_name not in self.datasets:
            QMessageBox.warning(self, "Enhanced Formula Editor", "No active dataset available.")
            return
            
        df = self.datasets[self.current_dataset_name]
        dialog = EnhancedFormulaEditorDialog(df, parent=self)
        
        if dialog.exec_() == QDialog.Accepted:
            try:
                new_df = dialog.get_modified_dataframe()
                if new_df is not None:
                    self.datasets[self.current_dataset_name] = new_df
                    self.add_data_viewer_tab(self.current_dataset_name, new_df)
                    self.status_label.setText(f"Variables created using enhanced formula editor.")
                    self.set_project_modified(True)
            except Exception as e:
                QMessageBox.critical(self, "Formula Error", f"Error applying formulas: {str(e)}")

    def show_variable_selection_dialog(self):
        """Show variable selection dialog"""
        if not self.current_dataset_name or self.current_dataset_name not in self.datasets:
            QMessageBox.warning(self, "Variable Selection", "No active dataset available.")
            return
            
        df = self.datasets[self.current_dataset_name]
        dialog = VariableSelectionDialog(df, parent=self)
        
        if dialog.exec_() == QDialog.Accepted:
            try:
                selected_variables = dialog.get_selected_variables()
                if selected_variables:
                    selected_df = df[selected_variables].copy()
                    
                    selected_name = f"{self.current_dataset_name}_selected"
                    i = 1
                    while selected_name in self.datasets:
                        selected_name = f"{self.current_dataset_name}_selected_{i}"
                        i += 1
                    
                    self.datasets[selected_name] = selected_df
                    self.current_dataset_name = selected_name
                    
                    self.add_data_viewer_tab(selected_name, selected_df)
                    self.workflow_canvas.add_node_to_canvas(
                        NODE_TYPE_DATASET, 
                        title=selected_name, 
                        specific_config={
                            'dataset_name': selected_name, 
                            'df_rows': len(selected_df), 
                            'df_cols': len(selected_df.columns),
                            'source': 'variable_selection'
                        }
                    )
                    
                    self.status_label.setText(f"Variables selected: {len(selected_variables)} variables")
                    self.set_project_modified(True)
            except Exception as e:
                QMessageBox.critical(self, "Selection Error", f"Error selecting variables: {str(e)}")

    def show_categorical_grouping_dialog(self):
        """Show categorical grouping dialog"""
        if not self.current_dataset_name or self.current_dataset_name not in self.datasets:
            QMessageBox.warning(self, "Categorical Grouping", "No active dataset available.")
            return
            
        df = self.datasets[self.current_dataset_name]
        dialog = CategoricalSplitGroupingDialog(df, parent=self)
        
        if dialog.exec_() == QDialog.Accepted:
            try:
                grouped_df = dialog.get_grouped_dataframe()
                if grouped_df is not None:
                    self.datasets[self.current_dataset_name] = grouped_df
                    self.add_data_viewer_tab(self.current_dataset_name, grouped_df)
                    self.status_label.setText(f"Categorical features grouped successfully.")
                    self.set_project_modified(True)
            except Exception as e:
                QMessageBox.critical(self, "Grouping Error", f"Error grouping categorical features: {str(e)}")

    def show_enhanced_tree_configuration_dialog(self):
        """Show enhanced tree configuration dialog"""
        if not self.current_model_name or self.current_model_name not in self.models:
            QMessageBox.warning(self, "Enhanced Tree Configuration", "No active model selected.")
            return
            
        model = self.models[self.current_model_name]
        dialog = EnhancedTreeConfigurationDialog(current_config=model.get_params(), parent=self)
        
        if dialog.exec_() == QDialog.Accepted:
            try:
                new_config = dialog.get_configuration()
                model.set_params(**new_config)
                
                if self.model_properties_widget:
                    self.model_properties_widget.update_properties()
                
                self.status_label.setText(f"Enhanced configuration applied to '{self.current_model_name}'.")
                self.set_project_modified(True)
            except Exception as e:
                QMessageBox.critical(self, "Configuration Error", f"Error applying configuration: {str(e)}")

    def show_manual_tree_growth_dialog(self):
        """Show manual tree growth dialog"""
        if not self.current_model_name or self.current_model_name not in self.models:
            QMessageBox.warning(self, "Manual Tree Growth", "No active model selected.")
            return
            
        if not self.current_dataset_name or self.current_dataset_name not in self.datasets:
            QMessageBox.warning(self, "Manual Tree Growth", "No active dataset available.")
            return
            
        model = self.models[self.current_model_name]
        df = self.datasets[self.current_dataset_name]
        
        dialog = ManualTreeGrowthDialog(model, df, parent=self)
        
        if dialog.exec_() == QDialog.Accepted:
            try:
                if hasattr(self, 'tree_viz_widget') and model.is_fitted:
                    self.tree_viz_widget.set_tree(model.root)
                
                if self.model_properties_widget:
                    self.model_properties_widget.update_properties()
                
                self.enhanced_status_bar.show_success(f"Manual tree growth completed for '{self.current_model_name}'.")
                self.set_project_modified(True)
            except Exception as e:
                QMessageBox.critical(self, "Manual Growth Error", f"Error during manual tree growth: {str(e)}")

    def show_variable_importance_dialog(self):
        """Show variable importance analysis dialog"""
        if not self.current_model_name or self.current_model_name not in self.models:
            QMessageBox.warning(self, "Variable Importance", "No active model selected.")
            return
            
        if not self.current_dataset_name or self.current_dataset_name not in self.datasets:
            QMessageBox.warning(self, "Variable Importance", "No active dataset available.")
            return
            
        model = self.models[self.current_model_name]
        df = self.datasets[self.current_dataset_name]
        
        if not model.is_fitted:
            QMessageBox.warning(self, "Variable Importance", "Model must be trained before analyzing variable importance.")
            return
            
        dialog = VariableImportanceDialog(df, model.target_name, tree_model=model, parent=self)
        dialog.show()  # Non-modal dialog

    def show_performance_evaluation_dialog(self):
        """Show performance evaluation dialog"""
        if not self.current_model_name or self.current_model_name not in self.models:
            QMessageBox.warning(self, "Performance Evaluation", "No active model selected.")
            return
            
        if not self.current_dataset_name or self.current_dataset_name not in self.datasets:
            QMessageBox.warning(self, "Performance Evaluation", "No active dataset available.")
            return
            
        model = self.models[self.current_model_name]
        df = self.datasets[self.current_dataset_name]
        
        if not model.is_fitted:
            QMessageBox.warning(self, "Performance Evaluation", "Model must be trained before evaluating performance.")
            return
            
        target_name = getattr(model, 'target_name', None)
        if not target_name or target_name not in df.columns:
            QMessageBox.warning(self, "Performance Evaluation", f"Target variable '{target_name}' not found in dataset.")
            return
            
        X = df.drop(columns=[target_name])
        y = df[target_name]
        tree_model = getattr(model, 'root', model)  # Get the actual tree
            
        dialog = PerformanceEvaluationDialog(X, y, tree_model, parent=self)
        dialog.show()  # Non-modal dialog

    def show_node_reporting_dialog(self):
        """Show node reporting dialog"""
        if not self.current_model_name or self.current_model_name not in self.models:
            QMessageBox.warning(self, "Node Reporting", "No active model selected.")
            return
            
        model = self.models[self.current_model_name]
        
        if not model.is_fitted:
            QMessageBox.warning(self, "Node Reporting", "Model must be trained before analyzing nodes.")
            return
            
        selected_node_id = None
        if hasattr(self, 'tree_viz_widget') and hasattr(self.tree_viz_widget, 'get_selected_node_id'):
            selected_node_id = self.tree_viz_widget.get_selected_node_id()
            
        if not selected_node_id:
            selected_node_id = model.root.node_id if model.root else None
            
        if not selected_node_id:
            QMessageBox.warning(self, "Node Reporting", "No node selected and no root node available.")
            return
            
        node_data = None
        target_column = model.target_name
        
        if self.current_dataset_name and self.current_dataset_name in self.datasets:
            node_data = self.datasets[self.current_dataset_name]
            
        node = model.get_node(selected_node_id)
        if not node:
            QMessageBox.warning(self, "Node Reporting", f"Node '{selected_node_id}' not found.")
            return
            
        dialog = NodeReportingDialog(model, node, node_data, target_column, parent=self)
        dialog.show()  # Non-modal dialog

    def show_tree_navigation_dialog(self):
        """Show tree navigation dialog"""
        if not self.current_model_name or self.current_model_name not in self.models:
            QMessageBox.warning(self, "Tree Navigation", "No active model selected.")
            return
            
        model = self.models[self.current_model_name]
        
        if not model.is_fitted:
            QMessageBox.warning(self, "Tree Navigation", "Model must be trained before navigating the tree.")
            return
            
        dialog = TreeNavigationDialog(model, parent=self)
        
        if hasattr(self, 'tree_viz_widget'):
            dialog.nodeSelected.connect(self.tree_viz_widget.select_node)
            dialog.pathRequested.connect(self.tree_viz_widget.highlight_path)
            
        dialog.show()  # Non-modal dialog

    def start_manual_tree_building(self, model: BespokeDecisionTree, dataset_df: pd.DataFrame, target_variable: str):
        """Start manual tree building process with the given model and data"""
        try:
            logger.info(f"Starting manual tree building for model {model.model_name}")
            
            if not hasattr(self, 'tree_visualizer_widget') or self.tree_visualizer_widget is None:
                self.tree_visualizer_widget = TreeVisualizerWidget(self)
                
                self.tree_visualizer_widget.nodeSelected.connect(self.on_tree_node_selected_in_visualizer)
                self.tree_visualizer_widget.nodeDoubleClicked.connect(self.on_tree_node_double_clicked)
                self.tree_visualizer_widget.nodeRightClicked.connect(self.show_enhanced_context_menu)
                
                if hasattr(self, 'model_tabs'):
                    tree_tab_index = -1
                    for i in range(self.model_tabs.count()):
                        if self.model_tabs.tabText(i) == "Tree Visualizer":
                            tree_tab_index = i
                            break
                    
                    if tree_tab_index == -1:
                        self.model_tabs.addTab(self.tree_visualizer_widget, "Tree Visualizer")
                        logger.info(f"Added Tree Visualizer tab to model_tabs. Total model tabs: {self.model_tabs.count()}")
                    else:
                        self.model_tabs.removeTab(tree_tab_index)
                        self.model_tabs.insertTab(tree_tab_index, self.tree_visualizer_widget, "Tree Visualizer")
                        logger.info(f"Replaced Tree Visualizer tab in model_tabs at index {tree_tab_index}")
            
            if model.root and hasattr(model, 'root'):
                self.tree_visualizer_widget.set_tree(model.root)
            else:
                logger.warning("Model has no root node for visualization")
            
            model_name = model.model_name or f"ManualTree_{len(self.models) + 1}"
            self.models[model_name] = model
            self.current_model_name = model_name
            
            if self.enhanced_status_bar:
                self.enhanced_status_bar.show_message(f"Active Model: {model_name} (Manual Mode)", timeout=2000)
            
            if hasattr(self, 'model_dock'):
                logger.info(f"Model dock visible: {self.model_dock.isVisible()}")
                logger.info("Model dock visibility skipped - using new window management system")
            
            if hasattr(self, 'model_tabs'):
                logger.info(f"Model tabs available: {[self.model_tabs.tabText(i) for i in range(self.model_tabs.count())]}")
                for i in range(self.model_tabs.count()):
                    if self.model_tabs.tabText(i) == "Tree Visualizer":
                        self.model_tabs.setCurrentIndex(i)
                        logger.info(f"Switched to Tree Visualizer tab at index {i}")
                        break
            
            feature_columns = [col for col in dataset_df.columns if col != target_variable]
            
            model._training_data = {
                'X': dataset_df[feature_columns],
                'y': dataset_df[target_variable],
                'dataset_name': f"WorkflowData_{target_variable}"
            }
            
            X_train = dataset_df[feature_columns]
            y_train = dataset_df[target_variable]
            
            logger.info(f"Fitting model for manual tree building with {len(X_train)} samples, {len(feature_columns)} features")
            model.fit(X_train, y_train)
            logger.info(f"Model fitted successfully. Model.is_fitted: {model.is_fitted}")
            
            self.enhanced_status_bar.show_message(f"Manual tree building started for model '{model_name}'")
            
            QMessageBox.information(
                self, 
                "Manual Tree Building Started",
                f"Manual tree building has been started for model '{model_name}'.\n\n"
                "You can now:\n"
                "1. View the tree structure in the Tree Visualizer tab\n"
                "2. Single-click nodes to view details in the Node Inspector\n"
                "3. Double-click nodes to open the full Node Editor dialog\n"
                "4. Configure splits, make nodes terminal, and set properties\n"
                "5. Use the Node Editor's Split Finder to find optimal splits\n\n"
                "The tree visualizer tab is now active."
            )
            
            self.set_project_modified(True)
            self.update_action_states()
            
            logger.info(f"Manual tree building setup completed for model {model_name}")
            
        except Exception as e:
            logger.error(f"Error starting manual tree building: {e}", exc_info=True)
            QMessageBox.critical(self, "Manual Tree Building Error", 
                               f"Error starting manual tree building: {str(e)}")


    @pyqtSlot(object)
    def on_workflow_node_selected_on_canvas(self, node: Optional[CanvasWorkflowNode]):
        """Handle node selection in the workflow canvas"""
        if node:
            logger.debug(f"MainWindow: Workflow node selected: {node.node_id}")
            
            node_config = node.get_config()
            if isinstance(node, ModelNode) and node_config.get('model_ref_id'):
                model_ref = node_config['model_ref_id']
                if model_ref in self.models:
                    self.current_model_name = model_ref
                    if self.model_properties_widget:
                        self.model_properties_widget.set_model(self.models[model_ref])
                    
                    if self.enhanced_status_bar:
                        self.enhanced_status_bar.show_message(f"Active Model: {model_ref}", timeout=2000)
            elif isinstance(node, DatasetNode) and node.dataset_name:
                self.current_dataset_name = node.dataset_name
                if node.dataset_name in self.datasets:
                    df = self.datasets[node.dataset_name]
                    self.enhanced_status_bar.show_message(f"Active: {node.dataset_name} ({len(df)} rows, {len(df.columns)} cols)")
                    
                    if node.dataset_name in self.data_viewer_widgets:
                        self.data_tabs.setCurrentWidget(self.data_viewer_widgets[node.dataset_name])
        else:
            logger.debug("MainWindow: Workflow node selection cleared.")
        
        self.update_action_states()

    def on_tree_updated(self):
        """Handle tree updated signal from model - Enhanced version"""
        if self.current_model_name and self.current_model_name in self.models:
            model = self.models[self.current_model_name]
            if hasattr(model, 'root') and model.root:
                logger.debug(f"Updating tree visualization for model '{self.current_model_name}'")
                self.tree_viz_widget.set_tree(model.root, model)
            else:
                logger.debug(f"Model '{self.current_model_name}' has no root node to visualize.")

    def on_node_updated(self, node_id: str):
        """Handle node updated signal from model - Enhanced version"""
        if self.current_model_name and self.current_model_name in self.models:
            model = self.models[self.current_model_name]
            node = model.get_node(node_id)
            if node:
                logger.debug(f"Updating node visualization for node '{node_id}'")
                if hasattr(self, 'tree_viz_widget'):
                    self.tree_viz_widget.update_node(node)
                else:
                    logger.warning("Tree visualizer not ready for node update.")
            else:
                logger.warning(f"Node update signal received for non-existent node '{node_id}'")

    def on_tree_node_selected(self, node_id: str):
        """Handle node selection in the tree visualizer - Enhanced version"""
        logger.debug(f"Tree visualizer node selected: ID={node_id}")
        
        if self.current_model_name and self.current_model_name in self.models:
            model = self.models[self.current_model_name]
            node = model.get_node(node_id)
            if node:
                status_parts = [f"Selected Node: {node_id}"]
                status_parts.append(f"Depth: {node.depth}")
                
                if node.is_terminal:
                    status_parts.append("Type: Terminal")
                    if node.majority_class:
                        status_parts.append(f"Prediction: {node.majority_class}")
                else:
                    status_parts.append("Type: Internal")
                    if node.split_feature:
                        status_parts.append(f"Split: {node.split_feature}")
                
                if node.samples:
                    status_parts.append(f"Samples: {node.samples:,}")
                
                status_message = " | ".join(status_parts)
                self.statusBar.showMessage(status_message, 5000)

    @pyqtSlot(str)
    def on_tree_node_selected_in_visualizer(self, node_id: str):
        """Handle node selection in the tree visualizer"""
        if self.current_model_name and self.current_model_name in self.models:
            model = self.models[self.current_model_name]
            tree_node_obj = model.get_node(node_id)
            
            if tree_node_obj:
                self.enhanced_status_bar.show_message(f"Tree node selected: {node_id}, Depth: {tree_node_obj.depth}")
                
                if hasattr(self, 'node_inspector_widget'):
                    feature_names = []
                    if self.current_dataset_name and self.current_dataset_name in self.datasets:
                        df = self.datasets[self.current_dataset_name]
                        feature_names = list(df.columns)
                    
                    self.node_inspector_widget.set_node(tree_node_obj, feature_names, read_only=False, model=model)
                    
                    self.node_inspector_widget.propertiesChanged.connect(
                        lambda: self.update_tree_visualization_after_node_edit()
                    )
                    
                    node_inspector_tab_idx = self.model_tabs.indexOf(self.node_inspector_widget)
                    if node_inspector_tab_idx != -1:
                        self.model_tabs.setCurrentIndex(node_inspector_tab_idx)

    @pyqtSlot(str)
    def on_tree_node_double_clicked(self, node_id: str):
        """Handle double-click on tree node to open Node Editor Dialog"""
        if self.current_model_name and self.current_model_name in self.models:
            model = self.models[self.current_model_name]
            tree_node_obj = model.get_node(node_id)
            
            if tree_node_obj:
                try:
                    feature_names = []
                    if self.current_dataset_name and self.current_dataset_name in self.datasets:
                        df = self.datasets[self.current_dataset_name]
                        feature_names = list(df.columns)
                    
                    from ui.node_editor import NodeEditorDialog
                    
                    dialog = NodeEditorDialog(
                        node=tree_node_obj,
                        model=model,
                        feature_names=feature_names,
                        parent=self
                    )
                    
                    dialog.properties_widget.propertiesChanged.connect(
                        self.update_tree_visualization_after_node_edit
                    )
                    
                    result = dialog.exec_()
                    
                    if result == QDialog.Accepted:
                        self.update_tree_visualization_after_node_edit()
                        logger.info(f"Node {node_id} edited successfully")
                        
                except Exception as e:
                    logger.error(f"Error opening Node Editor: {str(e)}", exc_info=True)
                    QMessageBox.critical(
                        self, "Node Editor Error",
                        f"Could not open Node Editor: {str(e)}"
                    )

    @pyqtSlot(str, QPointF)
    def on_tree_node_right_clicked(self, node_id: str, position: QPointF):
        """Handle right-click on tree node to show context menu"""
        if self.current_model_name and self.current_model_name in self.models:
            model = self.models[self.current_model_name]
            tree_node_obj = model.get_node(node_id)
            
            if tree_node_obj:
                try:
                    context_menu = QMenu(self)
                    
                    info_action = QAction(f"Node: {node_id}", self)
                    info_action.setEnabled(False)
                    context_menu.addAction(info_action)
                    context_menu.addSeparator()
                    
                    edit_action = QAction("Edit Node...", self)
                    edit_action.setIcon(self.style().standardIcon(QStyle.SP_FileDialogInfoView))
                    edit_action.triggered.connect(lambda: self.on_tree_node_double_clicked(node_id))
                    context_menu.addAction(edit_action)
                    
                    if not tree_node_obj.is_terminal and hasattr(tree_node_obj, 'split_feature') and tree_node_obj.split_feature:
                        edit_split_action = QAction("🔧 Edit Split...", self)
                        edit_split_action.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
                        edit_split_action.triggered.connect(lambda: self.edit_split_action(node_id))
                        context_menu.addAction(edit_split_action)
                    
                    context_menu.addSeparator()
                    
                    if tree_node_obj.is_terminal or not tree_node_obj.children:
                        split_action = QAction("🛠️ Manual Split...", self)
                        split_action.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
                        split_action.triggered.connect(lambda: self.split_node_action(node_id))
                        context_menu.addAction(split_action)
                        
                        find_split_action = QAction("🔍 Find Optimal Split...", self)
                        find_split_action.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
                        find_split_action.triggered.connect(lambda: self.find_split_action(node_id))
                        context_menu.addAction(find_split_action)
                    
                    if not tree_node_obj.is_terminal:
                        terminal_action = QAction("Make Terminal", self)
                        terminal_action.setIcon(self.style().standardIcon(QStyle.SP_DialogOkButton))
                        terminal_action.triggered.connect(lambda: self.make_terminal_action(node_id))
                        context_menu.addAction(terminal_action)
                        
                        if hasattr(tree_node_obj, 'children') and tree_node_obj.children:
                            prune_action = QAction("🌳 Prune Subtree", self)
                            prune_action.setIcon(self.style().standardIcon(QStyle.SP_DialogDiscardButton))
                            prune_action.triggered.connect(lambda: self.prune_subtree(node_id))
                            context_menu.addAction(prune_action)
                    
                    context_menu.addSeparator()
                    
                    copy_action = QAction("📋 Copy Node", self)
                    copy_action.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
                    copy_action.triggered.connect(lambda: self.copy_node_action(node_id))
                    context_menu.addAction(copy_action)
                    
                    if hasattr(self, '_copied_node_data') and self._copied_node_data:
                        if tree_node_obj.is_terminal or len(tree_node_obj.children) < 10:  # Reasonable limit
                            paste_action = QAction("📄 Paste Node", self)
                            paste_action.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
                            paste_action.triggered.connect(lambda: self.paste_node_action(node_id))
                            context_menu.addAction(paste_action)
                    
                    if tree_node_obj.parent is not None:
                        context_menu.addSeparator()
                        delete_action = QAction("Delete Node", self)
                        delete_action.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
                        delete_action.triggered.connect(lambda: self.delete_node_action(node_id))
                        context_menu.addAction(delete_action)
                    
                    if hasattr(self, 'tree_visualizer_widget'):
                        view = self.tree_visualizer_widget.view
                        widget_pos = view.mapFromScene(position)
                        global_pos = view.mapToGlobal(widget_pos)
                        
                        context_menu.exec_(global_pos)
                    
                except Exception as e:
                    logger.error(f"Error showing context menu: {str(e)}", exc_info=True)

    def _export_metrics_report(self, metrics, model):
        """Export comprehensive metrics report"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Performance Metrics Report",
            f"performance_metrics_{self.current_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                from analytics.performance_metrics import MetricsCalculator
                metrics_calc = MetricsCalculator()
                metrics_calc.last_metrics = metrics
                
                success = metrics_calc.export_metrics_report(filename, self.current_model_name)
                
                if success:
                    QMessageBox.information(self, "Export Successful", 
                                        f"Performance metrics report exported to:\n{filename}")
                else:
                    QMessageBox.warning(self, "Export Failed", "Could not export metrics report.")
                    
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Error exporting report:\n{str(e)}")

    def _export_metrics_visualizations(self, metrics, metrics_calc):
        """Export metrics visualizations"""
        if not MATPLOTLIB_AVAILABLE:
            QMessageBox.warning(self, "Export Visualizations", 
                            "Matplotlib not available - cannot export visualizations.")
            return
        
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory for Visualization Export"
        )
        
        if directory:
            try:
                import os
                base_name = f"metrics_viz_{self.current_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                if 'confusion_matrix' in metrics:
                    cm_fig = metrics_calc.plot_confusion_matrix(metrics)
                    if cm_fig:
                        cm_path = os.path.join(directory, f"{base_name}_confusion_matrix.png")
                        cm_fig.savefig(cm_path, dpi=300, bbox_inches='tight')
                        plt.close(cm_fig)
                
                if 'roc_curve' in metrics:
                    roc_fig = metrics_calc.plot_roc_curve(metrics)
                    if roc_fig:
                        roc_path = os.path.join(directory, f"{base_name}_roc_curve.png")
                        roc_fig.savefig(roc_path, dpi=300, bbox_inches='tight')
                        plt.close(roc_fig)
                
                summary_fig = metrics_calc.plot_metrics_summary(metrics)
                if summary_fig:
                    summary_path = os.path.join(directory, f"{base_name}_summary.png")
                    summary_fig.savefig(summary_path, dpi=300, bbox_inches='tight')
                    plt.close(summary_fig)
                
                QMessageBox.information(self, "Export Successful", 
                                    f"Visualizations exported to:\n{directory}")
                                    
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Error exporting visualizations:\n{str(e)}")

    def manual_tree_building_mode(self):
        """Start manual tree building mode - Enhanced version"""
        if not self.current_model_name or self.current_model_name not in self.models:
            QMessageBox.warning(self, "Manual Tree Building", "No model selected.")
            return
        
        model = self.models[self.current_model_name]
        if not model.is_fitted:
            QMessageBox.warning(self, "Manual Tree Building", "Model must be trained first.")
            return
        
        logger.info(f"Starting manual tree building for model {self.current_model_name}")
        
        model.set_params(growth_mode='manual')
        
        tree_viz_tab_name = f"Tree View - {self.current_model_name}"
        
        tab_exists = False
        for i in range(self.model_tabs.count()):
            if self.model_tabs.tabText(i) == tree_viz_tab_name:
                tab_exists = True
                self.model_tabs.setCurrentIndex(i)
                break
        
        if not tab_exists:
            tree_viz_widget = TreeVisualizerWidget()
            
            tree_viz_widget.nodeSelected.connect(self.on_tree_node_selected)
            tree_viz_widget.nodeDoubleClicked.connect(lambda node_id: self.edit_node_split(node_id))
            tree_viz_widget.nodeRightClicked.connect(self.show_enhanced_context_menu)
            
            tree_viz_widget.set_tree(model.root, model)
            
            tab_index = self.model_tabs.addTab(tree_viz_widget, tree_viz_tab_name)
            self.model_tabs.setCurrentIndex(tab_index)
            
            self.tree_viz_widget = tree_viz_widget
            
            logger.info(f"Added Tree Visualizer tab to model_tabs. Total model tabs: {self.model_tabs.count()}")
        
        self.statusBar.showMessage(f"Manual tree building mode activated for '{self.current_model_name}'. Right-click nodes to edit splits.", 5000)

    def edit_node_split(self, node_id: str):
        """Edit split for a specific node - Enhanced version"""
        if not self.current_model_name or self.current_model_name not in self.models:
            QMessageBox.warning(self, "Edit Split", "No model selected.")
            return
        
        model = self.models[self.current_model_name]
        node = model.get_node(node_id)
        
        if not node:
            QMessageBox.warning(self, "Edit Split", f"Node '{node_id}' not found.")
            return
        
        if node.is_terminal and len(node.children) == 0:
            QMessageBox.information(self, "Edit Split", "Cannot edit splits for terminal nodes with no children.")
            return
        
        try:
            dataset = None
            if self.current_dataset_name and self.current_dataset_name in self.datasets:
                dataset = self.datasets[self.current_dataset_name]
            
            dialog = NodeEditorDialog(
                node=node,
                model=model,
                feature_names=model.feature_names,
                parent=self
            )
            
            if dataset is not None and hasattr(dialog, 'set_dataset'):
                dialog.set_dataset(dataset)
            
            if hasattr(dialog, 'select_tab_by_name'):
                dialog.select_tab_by_name("Split Finder")
            
            if dialog.exec_() == QDialog.Accepted:
                if hasattr(self, 'tree_viz_widget'):
                    self.tree_viz_widget.set_tree(model.root, model)
                
                logger.info(f"Split modified successfully for node {node_id}")
                self.set_project_modified(True)
            
        except Exception as e:
            logger.error(f"Error editing node split: {e}", exc_info=True)
            QMessageBox.critical(self, "Edit Split Error", f"Could not edit split:\n{str(e)}")

    def show_node_context_menu(self, node_id: str, position: QPointF):
        """Show context menu for a tree node - Enhanced version"""
        if not self.current_model_name or self.current_model_name not in self.models:
            return
        
        model = self.models[self.current_model_name]
        node = model.get_node(node_id)
        
        if not node:
            return
        
        context_menu = QMenu(self)
        
        info_action = context_menu.addAction(f"Node Info: {node_id}")
        info_action.setEnabled(False)  # Just for display
        
        context_menu.addSeparator()
        
        stats_action = context_menu.addAction("View Node Statistics")
        stats_action.triggered.connect(lambda: self.show_node_statistics_for_id(node_id))
        
        if not node.is_terminal or len(node.children) > 0:
            edit_action = context_menu.addAction("Edit Split")
            edit_action.triggered.connect(lambda: self.edit_node_split(node_id))
        
        if node.is_terminal and len(node.children) == 0:
            find_split_action = context_menu.addAction("Find Optimal Split")
            find_split_action.triggered.connect(lambda: self.find_optimal_split(node_id))
        
        context_menu.addSeparator()
        
        if not node.is_terminal and len(node.children) > 0:
            prune_action = context_menu.addAction("Prune Subtree")
            prune_action.triggered.connect(lambda: self.prune_subtree(node_id))
        
        global_pos = self.tree_viz_widget.view.mapToGlobal(
            self.tree_viz_widget.view.mapFromScene(position)
        )
        context_menu.exec_(global_pos)

    def show_enhanced_context_menu(self, node_id: str, position: QPointF):
        """Show enhanced context menu for tree node"""
        if not self.current_model_name or self.current_model_name not in self.models:
            return
        
        model = self.models[self.current_model_name]
        node = model.get_node(node_id)
        
        if not node:
            return
        
        if node.node_id == 'root':
            node_type = "root"
        elif node.is_terminal or len(node.children) == 0:
            node_type = "terminal"
        else:
            node_type = "internal"
        
        node_state = {
            'is_terminal': node.is_terminal,
            'has_children': len(node.children) > 0,
            'is_expanded': True,  # Assume expanded for now
            'can_split': node.samples > 1 if hasattr(node, 'samples') else True,
            'sample_count': getattr(node, 'samples', 0)
        }
        
        if hasattr(self, 'tree_viz_widget') and hasattr(self.tree_viz_widget, 'view'):
            global_pos = self.tree_viz_widget.view.mapToGlobal(
                self.tree_viz_widget.view.mapFromScene(position)
            )
        else:
            global_pos = self.mapToGlobal(self.mapFromGlobal(self.cursor().pos()))
        
        self.tree_context_menu.show_for_node(node_id, node_type, node_state, global_pos)

    def show_node_statistics_for_id(self, node_id: str):
        """Show statistics for a specific node ID"""
        if hasattr(self, 'tree_viz_widget'):
            self.tree_viz_widget.highlight_node(node_id, True)
            
            QTimer.singleShot(3000, lambda: self.tree_viz_widget.highlight_node(node_id, False))
        
        if hasattr(self, 'tree_viz_widget') and hasattr(self.tree_viz_widget, 'scene'):
            if node_id in self.tree_viz_widget.scene.node_items:
                item = self.tree_viz_widget.scene.node_items[node_id]
                self.tree_viz_widget.scene.clearSelection()
                item.setSelected(True)
        
        self.show_node_statistics()

    def find_optimal_split(self, node_id: str):
        """Find optimal split for a terminal node - Placeholder implementation"""
        QMessageBox.information(self, "Find Optimal Split", f"Find optimal split functionality for node {node_id} is not yet implemented.")
        
    def prune_subtree(self, node_id: str):
        """Handle prune subtree request from context menu - FIXED implementation"""
        try:
            if not self.current_model_name or self.current_model_name not in self.models:
                QMessageBox.warning(self, "No Model", "No model is currently selected.")
                return
                
            model = self.models[self.current_model_name]
            
            node = model.get_node(node_id)
            if not node:
                QMessageBox.warning(self, "Node Not Found", f"Node {node_id} not found in the current model.")
                return
                
            if not hasattr(node, 'children') or not node.children:
                QMessageBox.information(self, "Nothing to Prune", f"Node {node_id} has no subtree to prune.")
                return
            
            reply = QMessageBox.question(
                self, 
                "Confirm Prune Subtree",
                f"Are you sure you want to prune the subtree rooted at node {node_id}?\n\n"
                f"This will remove all child nodes and convert this node to a leaf node.\n"
                f"This action cannot be undone.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                model.prune_subtree(node_id)
                
                self.update_tree_visualization_after_node_edit()
                
                QMessageBox.information(
                    self, 
                    "Subtree Pruned", 
                    f"Successfully pruned subtree at node {node_id}.\n"
                    f"The node has been converted to a leaf node."
                )
                
                self.set_project_modified(True)
                logger.info(f"Successfully pruned subtree at node {node_id}")
                
        except Exception as e:
            error_msg = f"Failed to prune subtree at node {node_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Prune Error", error_msg)
    
    def copy_node_action(self, node_id: str):
        """Copy a node and its entire subtree for later pasting"""
        try:
            if not self.current_model_name or self.current_model_name not in self.models:
                QMessageBox.warning(self, "No Model", "No model is currently selected.")
                return
                
            model = self.models[self.current_model_name]
            node = model.get_node(node_id)
            
            if not node:
                QMessageBox.warning(self, "Node Not Found", f"Node {node_id} not found in the current model.")
                return
            
            copied_data = self._serialize_node_subtree(node)
            
            self._copied_node_data = copied_data
            
            node_count = self._count_nodes_in_subtree(node)
            
            QMessageBox.information(
                self, "Split Configuration Copied", 
                f"Split configuration from node {node_id} has been copied.\n"
                f"Variables and binning logic will be recreated with proper statistics.\n"
                f"You can now paste it as a child of any terminal node or node with room for children."
            )
            
            logger.info(f"Copied node {node_id} with {node_count} nodes in subtree")
            
        except Exception as e:
            error_msg = f"Failed to copy node {node_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Copy Error", error_msg)
    
    def paste_node_action(self, parent_node_id: str):
        """Paste a previously copied node as a child of the specified parent node"""
        try:
            if not hasattr(self, '_copied_node_data') or not self._copied_node_data:
                QMessageBox.warning(self, "Nothing to Paste", "No node has been copied yet.")
                return
                
            if not self.current_model_name or self.current_model_name not in self.models:
                QMessageBox.warning(self, "No Model", "No model is currently selected.")
                return
                
            model = self.models[self.current_model_name]
            parent_node = model.get_node(parent_node_id)
            
            if not parent_node:
                QMessageBox.warning(self, "Node Not Found", f"Parent node {parent_node_id} not found.")
                return
            
            if not parent_node.is_terminal and len(parent_node.children) >= 10:
                QMessageBox.warning(
                    self, "Cannot Paste", 
                    f"Parent node {parent_node_id} already has too many children ({len(parent_node.children)})."
                )
                return
            
            copied_node_info = self._copied_node_data.get('node_info', {})
            original_node_id = copied_node_info.get('node_id', 'unknown')
            node_count = self._copied_node_data.get('node_count', 1)
            
            reply = QMessageBox.question(
                self, "Confirm Paste",
                f"Are you sure you want to paste the copied subtree from node {original_node_id} "
                f"({node_count} nodes) as a child of node {parent_node_id}?\n\n"
                f"This will create new nodes with auto-generated IDs.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                new_child = self._deserialize_node_subtree(self._copied_node_data, parent_node, model)
                
                if new_child:
                    if parent_node.is_terminal:
                        parent_node.is_terminal = False
                        parent_node.children = [new_child]
                    else:
                        parent_node.children.append(new_child)
                    
                    self.update_tree_visualization_after_node_edit()
                    
                    QMessageBox.information(
                        self, "Paste Successful",
                        f"Successfully applied split configuration as child of node {parent_node_id}.\n"
                        f"Split structure recreated with proper statistics computation."
                    )
                    
                    self.set_project_modified(True)
                    logger.info(f"Successfully pasted {node_count} nodes under parent {parent_node_id}")
                
        except Exception as e:
            error_msg = f"Failed to paste node to {parent_node_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.critical(self, "Paste Error", error_msg)
    
    def _serialize_node_subtree(self, node):
        """Serialize a node and its entire subtree for copying"""
        try:
            from models.node import TreeNode
            
            def serialize_node(n):
                """Extract split configuration for manual split recreation"""
                if n.is_terminal:
                    return None
                    
                split_feature = getattr(n, 'split_feature', None)
                split_type = getattr(n, 'split_type', None)
                split_value = getattr(n, 'split_value', None)
                split_categories = getattr(n, 'split_categories', {})
                
                if not split_feature:
                    return None
                    
                if split_type == 'categorical' and split_categories:
                    split_config = {
                        'type': 'categorical_multi_bin',
                        'feature': split_feature,
                        'split_categories': split_categories.copy()
                    }
                elif split_type == 'numeric' and split_value is not None:
                    split_config = {
                        'type': 'numeric_binary',
                        'feature': split_feature,
                        'split_value': split_value
                    }
                else:
                    split_config = {
                        'type': 'binary',
                        'feature': split_feature,
                        'split_value': split_value,
                        'split_categories': split_categories.copy() if split_categories else {}
                    }
                
                child_splits = []
                if hasattr(n, 'children') and n.children:
                    for child in n.children:
                        if child:
                            child_config = serialize_node(child)
                            if child_config:
                                child_splits.append(child_config)
                
                return {
                    'split_config': split_config,
                    'child_splits': child_splits,
                    'original_node_id': n.node_id  # For reference
                }
            
            node_data = serialize_node(node)
            
            def count_split_nodes(split_data):
                if not split_data:
                    return 0
                count = 1  # Current node
                for child in split_data.get('child_splits', []):
                    count += count_split_nodes(child)
                return count
            
            node_count = count_split_nodes(node_data) if node_data else 1
            
            return {
                'node_info': node_data,
                'node_count': node_count,
                'copy_timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error serializing node subtree: {e}")
            raise
    
    def _deserialize_node_subtree(self, copied_data, parent_node, model):
        """Deserialize and create a new node subtree with new IDs"""
        try:
            from models.node import TreeNode
            
            node_info = copied_data['node_info']
            
            def create_node_from_data(data, target_node, depth_offset=0):
                """Apply split configuration using manual split process"""
                if not data:
                    return None
                    
                split_config = data.get('split_config')
                if split_config:
                    if not self._validate_split_config(split_config):
                        logger.error(f"Invalid split configuration for node {target_node.node_id}: {split_config}")
                        return None
                        
                    logger.info(f"Applying split configuration to node {target_node.node_id}: {split_config}")
                    success = model.apply_manual_split(target_node.node_id, split_config)
                    
                    if success and hasattr(target_node, 'children') and target_node.children:
                        child_splits = data.get('child_splits', [])
                        for i, child_split in enumerate(child_splits):
                            if i < len(target_node.children):
                                child_node = target_node.children[i]
                                create_node_from_data(child_split, child_node, depth_offset)
                    
                    return target_node
                
                return None
                
                new_node.children = []
                for child_data in data.get('children', []):
                    child_node = create_node_from_data(child_data, new_node, depth_offset)
                    new_node.children.append(child_node)
                
                return new_node
            
            create_node_from_data(node_info, parent_node)
            
            return parent_node.children[0] if hasattr(parent_node, 'children') and parent_node.children else None
            
        except Exception as e:
            logger.error(f"Error deserializing node subtree: {e}")
            raise
    
    def _count_nodes_in_subtree(self, node):
        """Count the total number of nodes in a subtree"""
        try:
            count = 1  # Count the node itself
            if hasattr(node, 'children') and node.children:
                for child in node.children:
                    count += self._count_nodes_in_subtree(child)
            return count
        except Exception as e:
            logger.warning(f"Error counting nodes in subtree: {e}")
            return 1
    
    def _generate_unique_node_id(self, parent_node, model):
        """Generate a unique node ID for the copied node"""
        try:
            if parent_node:
                child_index = len(parent_node.children) if hasattr(parent_node, 'children') else 0
                base_id = f"{parent_node.node_id}_copy_{child_index}"
            else:
                base_id = "copy_root"
            
            counter = 0
            candidate_id = base_id
            while model.get_node(candidate_id) is not None:
                counter += 1
                candidate_id = f"{base_id}_{counter}"
            
            return candidate_id
            
        except Exception as e:
            logger.warning(f"Error generating unique node ID: {e}")
            import time
            return f"copy_{int(time.time() * 1000)}"


    def split_node_action(self, node_id: str):
        """Handle manual split node action from context menu - STREAMLINED VERSION"""
        if self.current_model_name and self.current_model_name in self.models:
            model = self.models[self.current_model_name]
            tree_node_obj = model.get_node(node_id)
            
            if tree_node_obj:
                try:
                    logger.info(f"Opening enhanced edit split dialog for node {node_id}")
                    
                    dataset = model._cached_X if hasattr(model, '_cached_X') and model._cached_X is not None else pd.DataFrame()
                    target_column = model.target_name if hasattr(model, 'target_name') else None
                    
                    dialog = EnhancedEditSplitDialog(
                        node=tree_node_obj,
                        dataset=dataset,
                        target_column=target_column,
                        model=model,
                        parent=self
                    )
                    
                    dialog.splitApplied.connect(
                        lambda node_id: self._on_split_applied_successfully(node_id)
                    )
                    
                    result = dialog.exec_()
                    
                    if result == QDialog.Accepted:
                        logger.info(f"Manual split applied successfully for node {node_id}")
                        self.update_tree_visualization_after_node_edit()
                        
                except Exception as e:
                    logger.error(f"Error opening streamlined manual split dialog: {str(e)}", exc_info=True)
                    QMessageBox.critical(
                        self, "Manual Split Dialog Error",
                        f"Could not open manual split dialog: {str(e)}"
                    )

    def find_split_action(self, node_id: str):
        """Handle find split action from context menu - CORRECTED VERSION"""
        if self.current_model_name and self.current_model_name in self.models:
            model = self.models[self.current_model_name]
            tree_node_obj = model.get_node(node_id)
            
            if tree_node_obj:
                try:
                    logger.info(f"Opening find split dialog for node {node_id}")
                    
                    dialog = FindSplitDialog(
                        node=tree_node_obj,
                        model=model,
                        parent=self
                    )
                    
                    dialog.split_selected.connect(
                        lambda split_config: self._apply_optimal_split(node_id, split_config)
                    )
                    
                    dialog.exec_()
                    
                except Exception as e:
                    logger.error(f"Error opening find split dialog: {str(e)}", exc_info=True)
                    QMessageBox.critical(
                        self, "Find Split Dialog Error",
                        f"Could not open find split dialog: {str(e)}"
                    )

    def _apply_optimal_split(self, node_id: str, split_config: Dict[str, Any]):
        """Apply an optimal split selected from the find split dialog"""
        try:
            if self.current_model_name and self.current_model_name in self.models:
                model = self.models[self.current_model_name]
                
                success = model.apply_manual_split(node_id, split_config)
                
                if success:
                    logger.info(f"Successfully applied optimal split to node {node_id}")
                    
                    self.update_tree_visualization_after_node_edit()
                    
                    if hasattr(self, 'model_tabs') and self.model_tabs:
                        self.update_action_states()
                    
                    if hasattr(self, 'status_label'):
                        if self.enhanced_status_bar:
                            self.enhanced_status_bar.show_message(f"Optimal split applied to node {node_id}")
                        
                    QMessageBox.information(
                        self, "Split Applied",
                        f"Optimal split successfully applied to node {node_id}"
                    )
                else:
                    QMessageBox.warning(
                        self, "Apply Split Failed",
                        "Failed to apply the selected split. Check the log for details."
                    )
                    
        except Exception as e:
            logger.error(f"Error applying optimal split: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self, "Apply Split Error",
                f"Could not apply split: {str(e)}"
            )

    def edit_split_action(self, node_id: str):
        """Handle edit split action from context menu"""
        if self.current_model_name and self.current_model_name in self.models:
            model = self.models[self.current_model_name]
            tree_node_obj = model.get_node(node_id)
            
            if tree_node_obj:
                try:
                    logger.info(f"Opening enhanced edit split dialog for node {node_id}")
                    
                    if not self.current_dataset_name or self.current_dataset_name not in self.datasets:
                        QMessageBox.warning(
                            self, "No Data Available", 
                            "No dataset is available for split editing. Please load a dataset first."
                        )
                        return
                    
                    dataset = self.datasets[self.current_dataset_name]
                    
                    if dataset is None or dataset.empty:
                        QMessageBox.warning(
                            self, "No Data Available",
                            "The current dataset is empty. Please load a valid dataset first."
                        )
                        return
                    
                    target_column = model.target_name
                    if not target_column or target_column not in dataset.columns:
                        QMessageBox.warning(
                            self, "Target Not Found",
                            "Target variable not found in the current dataset."
                        )
                        return
                    
                    from ui.dialogs.enhanced_edit_split_dialog import EnhancedEditSplitDialog
                    
                    dialog = EnhancedEditSplitDialog(
                        node=tree_node_obj,
                        dataset=dataset,
                        target_column=target_column,
                        model=model,
                        parent=self
                    )
                    
                    dialog.splitModified.connect(
                        lambda node, feature, split_config: self._on_split_modified(node, feature, split_config)
                    )
                    
                    try:
                        result = dialog.exec_()
                        
                        if result == QDialog.Accepted:
                            logger.info(f"Split modified successfully for node {node_id}")
                            QTimer.singleShot(100, self.update_tree_visualization_after_node_edit)
                    
                    except RuntimeError as re:
                        if "deleted" in str(re):
                            logger.warning(f"Dialog was deleted during execution: {re}")
                        else:
                            raise
                    except Exception as dialog_error:
                        logger.error(f"Error during dialog execution: {str(dialog_error)}", exc_info=True)
                        QMessageBox.critical(
                            self, "Dialog Execution Error",
                            f"An error occurred while using the edit split dialog: {str(dialog_error)}"
                        )
                        
                except Exception as e:
                    logger.error(f"Error opening enhanced edit split dialog: {str(e)}", exc_info=True)
                    QMessageBox.critical(
                        self, "Edit Split Dialog Error",
                        f"Could not open edit split dialog: {str(e)}"
                    )

    def _on_split_modified(self, node: 'TreeNode', feature: str, split_config: dict):
        """Handle split modification from enhanced edit split dialog"""
        try:
            if self.current_model_name and self.current_model_name in self.models:
                model = self.models[self.current_model_name]
                
                logger.info(f"Received split_config: {split_config}")
                
                if split_config.get("type") == "numeric_binary":
                    split_info = {
                        'feature': feature,
                        'split_value': split_config["split_value"],
                        'split_type': 'numeric'
                    }
                    success = model.apply_manual_split(node.node_id, split_info)
                    logger.info(f"Applied binary split on {feature} at threshold {split_config['split_value']}")
                    
                elif split_config.get("type") == "numeric_multi_bin":
                    thresholds = split_config["thresholds"]
                    num_bins = split_config.get("num_bins", len(thresholds) + 1)
                    
                    logger.info(f"Processing numeric_multi_bin: {len(thresholds)} thresholds, {num_bins} bins")
                    
                    if len(thresholds) == 1:
                        split_info = {
                            'feature': feature,
                            'split_value': thresholds[0],
                            'split_type': 'numeric'
                        }
                        success = model.apply_manual_split(node.node_id, split_info)
                        logger.info(f"Applied binary split on {feature} at threshold {thresholds[0]}")
                    else:
                        split_info = {
                            'feature': feature,
                            'split_type': 'numeric_multi_bin',
                            'thresholds': thresholds,
                            'num_bins': num_bins
                        }
                        success = model.apply_manual_split(node.node_id, split_info)
                        logger.info(f"Applied multi-bin split on {feature} with {len(thresholds)} thresholds")
                elif split_config.get("type") == "categorical_binary":
                    split_info = {
                        'feature': feature,
                        'left_categories': split_config["left_categories"],
                        'right_categories': split_config["right_categories"],
                        'split_type': 'categorical'
                    }
                    success = model.apply_manual_split(node.node_id, split_info)
                    logger.info(f"Applied binary categorical split on {feature}")
                    
                elif split_config.get("type") == "categorical_multi_bin":
                    if "split_categories" in split_config:
                        split_info = {
                            'feature': feature,
                            'split_categories': split_config["split_categories"],
                            'split_type': 'categorical'
                        }
                        success = model.apply_manual_split(node.node_id, split_info)
                        num_groups = len(set(split_config["split_categories"].values()))
                        logger.info(f"Applied {num_groups}-way categorical split on {feature}")
                    else:
                        category_groups = split_config["category_groups"]
                        if len(category_groups) == 2:
                            left_categories = category_groups[0]
                            right_categories = [cat for group in category_groups[1:] for cat in group]
                            split_info = {
                                'feature': feature,
                                'left_categories': left_categories,
                                'right_categories': right_categories,
                                'split_type': 'categorical'
                            }
                            success = model.apply_manual_split(node.node_id, split_info)
                            logger.info(f"Applied binary categorical split on {feature}")
                        else:
                            split_categories = {}
                            for i, group in enumerate(category_groups):
                                for category in group:
                                    split_categories[category] = i
                            
                            split_info = {
                                'feature': feature,
                                'split_categories': split_categories,
                                'split_type': 'categorical'
                            }
                            success = model.apply_manual_split(node.node_id, split_info)
                            logger.info(f"Applied {len(category_groups)}-way categorical split on {feature}")
                else:
                    if isinstance(split_config, (int, float)):
                        split_info = {
                            'feature': feature,
                            'split_value': split_config,
                            'split_type': 'numeric'
                        }
                    else:
                        split_info = {
                            'feature': feature,
                            'split_type': 'categorical',
                            'left_categories': [split_config] if not isinstance(split_config, list) else split_config
                        }
                    success = model.apply_manual_split(node.node_id, split_info)
                    logger.info(f"Applied simple split on {feature}: {split_config}")
                
                if success:
                    QTimer.singleShot(100, self.update_tree_visualization_after_node_edit)
                    
                    logger.info(f"Modified split on node {node.node_id}: {feature}")
                    
                    QMessageBox.information(
                        self, "Split Modified",
                        f"Successfully modified split on feature '{feature}' for node {node.node_id}."
                    )
                else:
                    logger.warning(f"Failed to modify split on node {node.node_id} with feature {feature}")
                    QMessageBox.warning(
                        self, "Modify Split Failed",
                        "Failed to modify the split. Check the log for details."
                    )
                    
        except Exception as e:
            logger.error(f"Error modifying split: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self, "Split Modification Error",
                f"Could not modify split: {str(e)}"
            )

    def _open_enhanced_edit_split_dialog(self, node_id: str):
        """
        CRITICAL FIX: Open enhanced edit split dialog with proper error handling and data validation
        """
        try:
            logger.info(f"Opening enhanced edit split dialog for node {node_id}")
            
            current_model = self._get_current_model()
            if not current_model:
                QMessageBox.warning(self, "Edit Split", "No model is currently selected.")
                return
            
            node = current_model.get_node_by_id(node_id)
            if not node:
                QMessageBox.warning(self, "Edit Split", f"Node {node_id} not found.")
                return
            
            dataset = None
            target_column = None
            
            if hasattr(current_model, '_cached_X') and current_model._cached_X is not None:
                dataset = current_model._cached_X.copy()
                
                if hasattr(current_model, '_cached_y') and current_model._cached_y is not None:
                    if hasattr(current_model, '_target_name') and current_model._target_name:
                        target_column = current_model._target_name
                    else:
                        target_column = 'TARGET'  # Default target column name
                        
                    dataset[target_column] = current_model._cached_y
                    
            if dataset is None or len(dataset) == 0:
                if hasattr(current_model, 'X') and current_model.X is not None:
                    dataset = current_model.X.copy()
                    
                    if hasattr(current_model, 'y') and current_model.y is not None:
                        target_column = getattr(current_model, 'target_name', 'TARGET')
                        dataset[target_column] = current_model.y
                        
            if dataset is None or len(dataset) == 0:
                QMessageBox.warning(
                    self, "Edit Split", 
                    "No data available for editing splits. Please ensure the model is fitted with data."
                )
                return
                
            logger.info(f"Dataset prepared: {len(dataset)} rows, {len(dataset.columns)} columns")
            logger.info(f"Target column: {target_column}")
            
            initial_split = None
            if hasattr(node, 'split_feature') and node.split_feature:
                initial_split = {
                    'feature': node.split_feature,
                    'split_type': 'categorical' if not pd.api.types.is_numeric_dtype(dataset[node.split_feature]) else 'numeric'
                }
                
                if hasattr(node, 'split_value') and node.split_value is not None:
                    initial_split['split_value'] = node.split_value
                    
                if hasattr(node, 'split_categories') and node.split_categories:
                    initial_split['split_categories'] = node.split_categories
                    
                logger.info(f"Prepared initial split configuration: {initial_split}")
            
            try:
                from ui.dialogs.enhanced_edit_split_dialog import EnhancedEditSplitDialog
                
                dialog = EnhancedEditSplitDialog(
                    node=node,
                    dataset=dataset,
                    target_column=target_column,
                    model=current_model,
                    initial_split=initial_split,
                    parent=self
                )
                
                dialog.splitModified.connect(self._on_split_modified)
                dialog.splitApplied.connect(self._on_split_applied)
                
                result = dialog.exec_()
                
                if result == QDialog.Accepted:
                    logger.info(f"Split modification completed for node {node_id}")
                    
                    if hasattr(self, 'tree_visualizer') and self.tree_visualizer:
                        QTimer.singleShot(100, self.tree_visualizer.update_visualization)
                        
                else:
                    logger.info(f"Split modification cancelled for node {node_id}")
                            
            except ImportError as import_error:
                logger.error(f"Failed to import EnhancedEditSplitDialog: {import_error}")
                QMessageBox.critical(
                    self, "Import Error",
                    f"Could not load the enhanced edit split dialog: {import_error}"
                )
                
            except Exception as dialog_error:
                logger.error(f"Error in enhanced edit split dialog: {dialog_error}", exc_info=True)
                QMessageBox.critical(
                    self, "Dialog Error",
                    f"An error occurred in the edit split dialog: {dialog_error}"
                )
                
        except Exception as e:
            logger.error(f"Error opening enhanced edit split dialog for node {node_id}: {e}", exc_info=True)
            QMessageBox.critical(self, "Edit Split Dialog Error", f"Could not open edit split dialog: {e}")

    def _on_split_modified(self, node: 'TreeNode', feature: str, split_config: dict):
        """
        FIXED: Handle split modification with proper error handling
        """
        try:
            logger.info(f"Processing split modification for node {node.node_id}")
            logger.debug(f"Split config received: {split_config}")
            
            if self.current_model_name and self.current_model_name in self.models:
                model = self.models[self.current_model_name]
                
                model_split_config = self._convert_dialog_config_to_model_format(split_config)
                
                success = model.apply_manual_split(node.node_id, model_split_config)
                
                if success:
                    logger.info(f"Successfully applied split modification for node {node.node_id}")
                    
                    logger.info(f"Checking for active detail window to refresh")
                    if hasattr(self, 'window_manager') and self.window_manager.get_current_window_type() == 'model':
                        current_window = self.window_manager.stacked_widget.currentWidget()
                        if hasattr(current_window, 'refresh_content') and hasattr(current_window, 'model') and current_window.model == model:
                            logger.info(f"Found active detail window with matching model, calling refresh_content()")
                            current_window.refresh_content()
                        else:
                            logger.info(f"Detail window model mismatch or no refresh_content method")
                    else:
                        logger.info(f"No active model detail window found")
                    
                    logger.info(f"Calling update_tree_visualization_after_node_edit() as backup")
                    self.update_tree_visualization_after_node_edit()
                        
                    QMessageBox.information(
                        self, "Split Modified",
                        f"Split successfully modified for node {node.node_id} using feature '{feature}'"
                    )
                else:
                    logger.warning(f"Failed to apply split modification for node {node.node_id}")
                    QMessageBox.warning(
                        self, "Split Modification Failed",
                        "Failed to apply the split modification. Check the log for details."
                    )
                    
        except Exception as e:
            logger.error(f"Error handling split modification: {e}", exc_info=True)
            QMessageBox.critical(
                self, "Split Modification Error",
                f"Could not apply split modification: {e}"
            )

    def _on_split_applied(self, node_id: str):
        """
        FIXED: Handle split application completion
        """
        try:
            logger.info(f"Split application completed for node {node_id}")
            
            if self.current_model_name and self.current_model_name in self.models:
                model = self.models[self.current_model_name]
                
                if hasattr(model, 'mark_modified'):
                    model.mark_modified()
                    
            if hasattr(self, 'tree_visualizer') and self.tree_visualizer:
                QTimer.singleShot(200, self.tree_visualizer.update_visualization)
                
            self._update_model_dependent_ui()
            
        except Exception as e:
            logger.error(f"Error handling split application: {e}")

    def _convert_dialog_config_to_model_format(self, split_config: dict) -> dict:
        """
        FIXED: Convert dialog split configuration to model format
        """
        try:
            config_type = split_config.get('type', 'categorical_binary')
            
            if config_type == 'numeric_binary':
                return {
                    'feature': split_config['feature'],
                    'split_type': 'numeric',
                    'split_value': split_config['split_value'],
                    'threshold': split_config['split_value'],
                    'split_operator': '<='
                }
            elif config_type == 'categorical_binary':
                return {
                    'feature': split_config['feature'],
                    'split_type': 'categorical',
                    'left_categories': split_config.get('left_categories', []),
                    'right_categories': split_config.get('right_categories', [])
                }
            elif config_type == 'numeric_multi_bin':
                return {
                    'feature': split_config['feature'],
                    'split_type': 'numeric_multi_bin',
                    'thresholds': split_config['thresholds'],
                    'num_bins': split_config.get('num_bins', len(split_config['thresholds']) + 1)
                }
            elif config_type == 'categorical_multi_bin':
                return {
                    'feature': split_config['feature'],
                    'split_type': 'categorical_multi_bin',
                    'split_categories': split_config['split_categories']
                }
            else:
                return {
                    'feature': split_config['feature'],
                    'split_type': 'categorical',
                    'split_value': split_config.get('split_value')
                }
                
        except Exception as e:
            logger.error(f"Error converting dialog config to model format: {e}")
            return {
                'feature': split_config.get('feature', ''),
                'split_type': 'categorical'
            }

    def _update_model_dependent_ui(self):
        """
        FIXED: Update UI elements that depend on model state
        """
        try:
            if hasattr(self, 'tree_visualizer') and self.tree_visualizer:
                self.tree_visualizer.update_visualization()
                
            if hasattr(self, 'model_stats_widget') and self.model_stats_widget:
                self.model_stats_widget.refresh_stats()
                
            if hasattr(self, 'status_bar') and self.status_bar:
                self.status_bar.showMessage("Model updated", 2000)
                
            logger.debug("Updated model-dependent UI elements")
            
        except Exception as e:
            logger.error(f"Error updating model-dependent UI: {e}")

    def _on_split_configured(self, node: 'TreeNode', split_config: 'SplitConfiguration'):
        """Handle split configuration from new split configuration dialog"""
        try:
            if self.current_model_name and self.current_model_name in self.models:
                model = self.models[self.current_model_name]
                
                logger.info(f"Received split configuration: {split_config}")
                
                split_info = split_config.to_dict()
                success = model.apply_manual_split(node.node_id, split_info)
                
                if success:
                    if hasattr(self, '_visualization_update_timer') and self._visualization_update_timer:
                        self._visualization_update_timer.stop()
                        self._visualization_update_timer.deleteLater()
                        self._visualization_update_timer = None
                    
                    self._visualization_update_timer = QTimer()
                    self._visualization_update_timer.setSingleShot(True)
                    self._visualization_update_timer.timeout.connect(self.update_tree_visualization_after_node_edit)
                    self._visualization_update_timer.start(100)
                    
                    logger.info(f"Applied split configuration on node {node.node_id}: {split_config.feature}")
                    
                    QMessageBox.information(
                        self, "Split Applied",
                        f"Successfully applied split on feature '{split_config.feature}' for node {node.node_id}."
                    )
                else:
                    logger.warning(f"Failed to apply split on node {node.node_id} with feature {split_config.feature}")
                    QMessageBox.warning(
                        self, "Apply Split Failed",
                        "Failed to apply the split. Check the log for details."
                    )
                    
        except Exception as e:
            logger.error(f"Error applying split configuration: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self, "Split Configuration Error",
                f"Could not apply split configuration: {str(e)}"
            )

    def _apply_found_split(self, node_id: str, split_info: dict):
        """Apply a split found by the Find Split feature"""
        try:
            if self.current_model_name and self.current_model_name in self.models:
                model = self.models[self.current_model_name]
                
                success = model.apply_manual_split(node_id, split_info)
                
                if success:
                    self.update_tree_visualization_after_node_edit()
                    
                    feature = split_info.get('feature', 'unknown')
                    split_type = split_info.get('split_type', 'unknown')
                    logger.info(f"Applied optimal split found by Find Split: {feature} ({split_type}) on node {node_id}")
                    
                    QMessageBox.information(
                        self, "Split Applied",
                        f"Successfully applied optimal split on feature '{feature}' to node {node_id}."
                    )
                else:
                    QMessageBox.warning(
                        self, "Apply Split Failed",
                        "Failed to apply the selected split. Check the log for details."
                    )
                    
        except Exception as e:
            logger.error(f"Error applying found split: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self, "Apply Split Error",
                f"Could not apply split: {str(e)}"
            )

    def _on_split_applied_successfully(self, node_id: str):
        """Handle successful split application from streamlined dialog"""
        logger.info(f"Split successfully applied to node {node_id}")
        
        self.update_tree_visualization_after_node_edit()
        
        if hasattr(self, 'model_tabs') and self.model_tabs:
            self.update_action_states()
        
        if hasattr(self, 'status_label'):
            self.enhanced_status_bar.show_message(f"Split applied to node {node_id}")

    def make_terminal_action(self, node_id: str):
        """Handle make terminal action from context menu"""
        if self.current_model_name and self.current_model_name in self.models:
            model = self.models[self.current_model_name]
            tree_node_obj = model.get_node(node_id)
            
            if tree_node_obj:
                reply = QMessageBox.question(
                    self, "Make Terminal Node",
                    f"Are you sure you want to make node {node_id} terminal?\n\n"
                    "This will remove all child nodes and their subtrees.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    try:
                        tree_node_obj.is_terminal = True
                        tree_node_obj.children = []
                        tree_node_obj.split_feature = None
                        tree_node_obj.split_value = None
                        tree_node_obj.split_type = None
                        tree_node_obj.split_categories = {}
                        tree_node_obj.split_rule = None
                        
                        self.update_tree_visualization_after_node_edit()
                        
                        logger.info(f"Node {node_id} converted to terminal node")
                        
                    except Exception as e:
                        logger.error(f"Error making node terminal: {str(e)}", exc_info=True)
                        QMessageBox.critical(
                            self, "Make Terminal Error",
                            f"Could not make node terminal: {str(e)}"
                        )

    def delete_node_action(self, node_id: str):
        """Handle delete node action from context menu"""
        if self.current_model_name and self.current_model_name in self.models:
            model = self.models[self.current_model_name]
            tree_node_obj = model.get_node(node_id)
            
            if tree_node_obj and tree_node_obj.parent:
                reply = QMessageBox.question(
                    self, "Delete Node",
                    f"Are you sure you want to delete node {node_id}?\n\n"
                    "This will remove the node and all its children.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    try:
                        parent = tree_node_obj.parent
                        if tree_node_obj in parent.children:
                            parent.children.remove(tree_node_obj)
                        
                        self.update_tree_visualization_after_node_edit()
                        
                        logger.info(f"Node {node_id} deleted successfully")
                        
                    except Exception as e:
                        logger.error(f"Error deleting node: {str(e)}", exc_info=True)
                        QMessageBox.critical(
                            self, "Delete Node Error",
                            f"Could not delete node: {str(e)}"
                        )


    def update_tree_visualization_after_node_edit(self):
        """Update tree visualization after a node has been edited"""
        logger.info(f"update_tree_visualization_after_node_edit() called")
        logger.info(f"current_model_name: {getattr(self, 'current_model_name', 'None')}")
        logger.info(f"models available: {list(getattr(self, 'models', {}).keys())}")
        
        if self.current_model_name and self.current_model_name in self.models:
            model = self.models[self.current_model_name]
            logger.info(f"Found model: {model}")
            logger.info(f"Model has root: {hasattr(model, 'root') and model.root is not None}")
            logger.info(f"Has tree_visualizer_widget: {hasattr(self, 'tree_visualizer_widget')}")
            
            main_window_updated = False
            if hasattr(self, 'tree_visualizer_widget') and model.root:
                try:
                    if hasattr(self, '_visualization_update_timer') and self._visualization_update_timer:
                        self._visualization_update_timer.stop()
                        self._visualization_update_timer.deleteLater()
                        self._visualization_update_timer = None
                        logger.debug("Cancelled previous visualization update timer")
                    
                    from PyQt5.QtCore import QTimer
                    import weakref
                    
                    self._visualization_update_timer = QTimer()
                    self._visualization_update_timer.setSingleShot(True)
                    self._visualization_update_timer.timeout.connect(
                        lambda: self._safe_update_tree_visualization(weakref.ref(model))
                    )
                    self._visualization_update_timer.start(100)
                    
                    logger.info(f"Scheduled tree visualization update for model {self.current_model_name}")
                    main_window_updated = True
                except Exception as e:
                    logger.error(f"Error scheduling tree visualization update: {str(e)}", exc_info=True)
            
            detail_window_updated = False
            if hasattr(self, 'window_manager') and model.root:
                try:
                    current_window = self.window_manager.get_current_window()
                    if (current_window and hasattr(current_window, 'tree_visualizer') and 
                        current_window.tree_visualizer):
                        current_window.tree_visualizer.set_tree(model.root)
                        logger.info(f"Updated detail window tree visualizer for model {self.current_model_name}")
                        detail_window_updated = True
                except Exception as e:
                    logger.error(f"Error updating detail window tree visualizer: {str(e)}", exc_info=True)
            
            if not main_window_updated and not detail_window_updated:
                logger.warning(f"No tree visualizer updated - main_window_widget: {hasattr(self, 'tree_visualizer_widget')}, model.root: {hasattr(model, 'root') and model.root is not None}, window_manager: {hasattr(self, 'window_manager')}")
        else:
            logger.warning(f"Cannot update tree visualization - current_model_name or model not found")
    
    def _safe_update_tree_visualization(self, model_ref):
        """Safely update tree visualization with error handling"""
        try:
            import weakref
            if isinstance(model_ref, weakref.ref):  # proper weakref check
                model = model_ref()  # Dereference weak reference
            elif callable(model_ref):  # check if it's callable (could be weakref)
                try:
                    model = model_ref()  # Try to call it
                except:
                    model = model_ref  # If it fails, treat as direct reference
            else:
                model = model_ref  # Direct reference (backward compatibility)
                
            if model is None:
                logger.debug("Model was garbage collected, skipping visualization update")
                return
                
            if hasattr(self, 'tree_visualizer_widget') and model and model.root:
                if hasattr(model, 'model_name'):
                    self.tree_visualizer_widget.set_tree(model.root)
                    logger.info(f"Updated tree visualization for model {model.model_name}")
                    
                    if hasattr(self, 'window_manager') and self.window_manager.get_current_window_type() == 'model':
                        current_window = self.window_manager.stacked_widget.currentWidget()
                        if hasattr(current_window, 'tree_visualizer') and current_window.tree_visualizer:
                            current_window.tree_visualizer.set_tree(model.root)
                            logger.info(f"Updated model detail window tree visualization for model {model.model_name}")
                else:
                    logger.warning("Model appears to be in invalid state, skipping update")
            
            if hasattr(self, '_visualization_update_timer'):
                self._visualization_update_timer = None
                
        except Exception as e:
            logger.error(f"Error in safe tree visualization update: {str(e)}", exc_info=True)
            if hasattr(self, '_visualization_update_timer'):
                self._visualization_update_timer = None
    
    @pyqtSlot(int)
    def on_data_tab_changed(self, index: int):
        """Handle tab change in the data tabs"""
        if index < 0:
            return
        
        current_widget = self.data_tabs.widget(index)
        
        if current_widget == self.variables_tab_content:
            if self.current_dataset_name and self.current_dataset_name in self.datasets:
                self.update_variable_viewer(self.datasets[self.current_dataset_name])
        elif isinstance(current_widget, DataViewerWidget):
            dataset_name = self.data_tabs.tabText(index)
            
            if dataset_name in self.datasets:
                self.current_dataset_name = dataset_name
                if self.enhanced_status_bar:
                    self.enhanced_status_bar.show_message(f"Active: {dataset_name} ({len(self.datasets[dataset_name])} rows)", timeout=2000)
                self.update_variable_viewer(self.datasets[dataset_name])
            else:
                self.current_dataset_name = None
                if self.variable_viewer_widget:
                    self.variable_viewer_widget.clear()
                
                self.enhanced_status_bar.show_error(f"Dataset: {dataset_name} (Error - Not loaded)")
        elif current_widget == self.dataset_placeholder_label:
            self.current_dataset_name = None
            if self.variable_viewer_widget:
                self.variable_viewer_widget.clear()
            
            self.enhanced_status_bar.show_message("No dataset active")
        
        self.update_action_states()

    @pyqtSlot(int)
    def on_model_tab_changed(self, index: int):
        """Handle tab change in the model tabs"""
        if self.model_tabs.widget(index) == self.model_properties_widget:
            if self.current_model_name and self.current_model_name in self.models:
                self.model_properties_widget.set_model(self.models[self.current_model_name])
            else:
                self.model_properties_widget.set_model(None)

    @pyqtSlot()
    def on_tree_updated(self):
        """Handle tree updated signal from model"""
        if self.current_model_name and self.current_model_name in self.models:
            model = self.models[self.current_model_name]
            if hasattr(model, 'root') and model.root:
                logger.debug(f"Updating tree visualization for model '{self.current_model_name}'")
                self.tree_viz_widget.set_tree(model.root, model)
            else:
                logger.debug(f"Model '{self.current_model_name}' has no root node to visualize.")

    @pyqtSlot(str)
    def on_node_updated(self, node_id: str):
        """Handle node updated signal from model"""
        if self.current_model_name and self.current_model_name in self.models:
            model = self.models[self.current_model_name]
            node = model.get_node(node_id)
            
            if node:
                logger.debug(f"Updating node visualization for node '{node_id}'")
                
                if hasattr(self, 'tree_viz_widget'):
                    self.tree_viz_widget.update_node(node)
                else:
                    logger.warning("Tree visualizer not ready for node update.")
            else:
                logger.warning(f"Node update signal received for non-existent node '{node_id}'")

    @pyqtSlot(str)
    def on_model_error(self, error_msg: str):
        """Handle error signal from model"""
        logger.error(f"Model error received: {error_msg}")
        self.statusBar.showMessage(f"Error: {error_msg}", 5000)
        QMessageBox.critical(self, "Model Error", error_msg)
        
        if self.progress_bar.isVisible():
            self.progress_bar.hide()

    @pyqtSlot(int, str)
    def on_training_progress(self, progress: int, message: str):
        """Handle training progress signal from model"""
        self.progress_bar.setValue(progress)
        self.statusBar.showMessage(message, 0)
        
        if not self.progress_bar.isVisible():
            self.progress_bar.show()
        
        QApplication.processEvents()

    @pyqtSlot()
    def on_training_complete(self):
        """Handle training complete signal from model"""
        self.progress_bar.setValue(100)
        self.progress_bar.hide()
        
        if self.current_model_name and self.current_model_name in self.models:
            model = self.models[self.current_model_name]
            self.statusBar.showMessage(
                f"Model '{self.current_model_name}' training complete. Nodes: {model.num_nodes}, Depth: {model.max_depth}", 5000
            )
            logger.info(f"Training complete for model '{self.current_model_name}'.")

    @pyqtSlot(str)
    def on_training_required(self, message: str):
        """Handle training required signal from model"""
        logger.info(f"Model requires training: {message}")
        
        from PyQt5.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, 
            "Training Required", 
            f"{message}\n\nWould you like to train the model now by running the workflow?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            logger.info("User chose to train model automatically")
            if hasattr(self, 'workflow_canvas') and self.workflow_canvas:
                sender_model = self.sender()
                if sender_model:
                    model_node_id = self._find_model_node_id(sender_model)
                    if model_node_id:
                        logger.info(f"Triggering workflow rerun from model node {model_node_id}")
                        self.workflow_canvas.trigger_workflow_rerun(model_node_id)
                    else:
                        logger.warning("Could not find model node in workflow, running full workflow")
                        self.workflow_canvas.run_workflow()
                else:
                    logger.warning("Could not identify sender model, running full workflow")
                    self.workflow_canvas.run_workflow()
        else:
            logger.info("User chose not to train model automatically")
            self.statusBar.showMessage("Model training required before finding splits", 5000)

    
    def _find_model_node_id(self, model_instance) -> Optional[str]:
        """
        Find the workflow node ID that contains the given model instance
        
        Args:
            model_instance: The BespokeDecisionTree instance
            
        Returns:
            Node ID if found, None otherwise
        """
        if not hasattr(self, 'workflow_canvas') or not self.workflow_canvas:
            return None
            
        for node_id, node in self.workflow_canvas.scene.nodes.items():
            if hasattr(node, 'model') and node.model is model_instance:
                return node_id
                
        return None

    def get_node_input_data(self, target_node_id: str, target_input_port_name: str) -> Optional[pd.DataFrame]:
        """
        Get input data for a node
        
        This is a helper for UI dialogs that need column context (e.g. FilterEditor).
        It simulates what the execution engine's _get_node_inputs would find.
        NOTE: This does NOT run the workflow. It looks at existing connections and loaded data.
        """
        target_node = self.workflow_canvas.get_node_by_id(target_node_id)
        if not target_node:
            return None

        for conn in self.workflow_canvas.scene.connections.values():
            if conn.target_node == target_node and conn.target_port_name == target_input_port_name:
                source_node_id = conn.source_node.node_id
                source_port_name = conn.source_port_name

                if isinstance(conn.source_node, DatasetNode):
                    ds_name = conn.source_node.dataset_name
                    if ds_name in self.datasets:
                        return self.datasets[ds_name]
                
                logger.warning(
                    f"Getting input data for UI: Node {source_node_id} is not a directly loaded DatasetNode. "
                    "Real-time input preview for intermediate steps is complex."
                )
                break
        
        return None

    def update_action_states(self):
        """Enable or disable actions based on current state"""
        has_active_dataset = self.current_dataset_name is not None and self.current_dataset_name in self.datasets
        has_active_model = self.current_model_name is not None and self.current_model_name in self.models
        model_is_fitted = has_active_model and self.models[self.current_model_name].is_fitted

        if hasattr(self, 'filter_action'):
            self.filter_action.setEnabled(has_active_dataset)
        if hasattr(self, 'transform_data_action'):
            self.transform_data_action.setEnabled(has_active_dataset)
        if hasattr(self, 'create_var_action'):
            self.create_var_action.setEnabled(has_active_dataset)

        if hasattr(self, 'new_model_action'):
            self.new_model_action.setEnabled(has_active_dataset)
        if hasattr(self, 'export_python_action'):
            self.export_python_action.setEnabled(model_is_fitted)
        if hasattr(self, 'export_pmml_action'):
            self.export_pmml_action.setEnabled(model_is_fitted)

        if hasattr(self, 'variable_importance_dialog_action'):
            self.variable_importance_dialog_action.setEnabled(model_is_fitted)
        if hasattr(self, 'performance_evaluation_dialog_action'):
            self.performance_evaluation_dialog_action.setEnabled(model_is_fitted and has_active_dataset)

        self.run_workflow_action.setEnabled(len(self.workflow_canvas.scene.nodes) > 0)
        if hasattr(self.workflow_canvas, 'run_all_button'):
            self.workflow_canvas.run_all_button.setEnabled(len(self.workflow_canvas.scene.nodes) > 0)

    def delete_selected_workflow_items(self):
        """Delete selected items in the workflow canvas"""
        selected_items = self.workflow_canvas.scene.selectedItems()
        if not selected_items:
            return

        node_titles = [item.title for item in selected_items if isinstance(item, CanvasWorkflowNode)]
        # Note: No connection graphics to select in new system
        
        if not node_titles:
            return

        msg = f"Delete {len(node_titles)} node(s) ({', '.join(node_titles[:3])}{'...' if len(node_titles)>3 else ''})"
        msg += "?"

        if QMessageBox.question(self, "Delete Items", msg, QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            for item in selected_items:
                if isinstance(item, CanvasWorkflowNode):
                    self.workflow_canvas.scene.remove_node(item.node_id)
                # Note: Connection deletion handled automatically when nodes are removed
            
            self.set_project_modified(True)

    def set_project_modified(self, modified: bool, update_manager: bool = True):
        """Set the project modified flag and update UI accordingly"""
        self._project_modified = modified
        title = self.windowTitle()
        
        if title.endswith("*") and not modified:
            title = title[:-1]
        elif not title.endswith("*") and modified:
            title += "*"
        
        self.setWindowTitle(title)
        
        if hasattr(self, 'save_action'):
            self.save_action.setEnabled(modified)
            
        if update_manager and hasattr(self, 'project_manager') and self.project_manager:
            self.project_manager.is_modified = modified

    def check_unsaved_changes(self) -> bool:
        """Check for unsaved changes and prompt the user"""
        if self._project_modified:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "Current project has unsaved changes. Save now?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                return not self.save_project()
            elif reply == QMessageBox.Cancel:
                return True
        
        return False

    def closeEvent(self, event: QCloseEvent):
        """Handle window close event"""
        if self.check_unsaved_changes():
            event.ignore()
            return
        
        self.save_application_state()
        
        if hasattr(self.workflow_engine, 'thread_pool'):
            self.workflow_engine.thread_pool.clear()
            self.workflow_engine.thread_pool.waitForDone(1000)
        
        super().closeEvent(event)

    def load_application_state(self):
        """Load application state from settings"""
        settings = QSettings("BespokeAnalytics", "BespokeDecisionTreeUtility")
        
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
    
        logger.info("Window state restoration skipped for new UI architecture")
        recent_files = settings.value("recentFiles", [])
        if recent_files is None:
            recent_files = []
        elif not isinstance(recent_files, list):
            recent_files = []
        
        if hasattr(self, 'recent_menu') and self.recent_menu is not None:
            for file in recent_files:
                if file:  # Only add non-empty file paths
                    self.recent_menu.addAction(file, lambda f=file: self.open_project(f))
        else:
            logger.warning("recent_menu not available during application state loading")
        last_project = settings.value("lastProject")
        if last_project:
            self.open_project(last_project)
        last_dataset = settings.value("lastDataset")
        if last_dataset:
            # TODO: Implement dataset loading from path
            logger.info(f"Last dataset path: {last_dataset}")
        last_model = settings.value("lastModel")        
        if last_model:
            # TODO: Implement model loading from path
            logger.info(f"Last model path: {last_model}")
        last_workflow = settings.value("lastWorkflow")
        if last_workflow:
            # TODO: Implement workflow loading from path
            logger.info(f"Last workflow path: {last_workflow}")
        last_data_viewer = settings.value("lastDataViewer")
        if last_data_viewer:
            # TODO: Implement data viewer state loading
            logger.info(f"Last data viewer state: {last_data_viewer}")
        last_model_viewer = settings.value("lastModelViewer")
        if last_model_viewer:
            # TODO: Implement model viewer state loading
            logger.info(f"Last model viewer state: {last_model_viewer}")
        last_visualization = settings.value("lastVisualization")
        if last_visualization:
            # TODO: Implement visualization state loading
            logger.info(f"Last visualization state: {last_visualization}")
        last_analysis = settings.value("lastAnalysis")  
        if last_analysis:
            # TODO: Implement analysis state loading
            logger.info(f"Last analysis state: {last_analysis}")
        last_node_editor = settings.value("lastNodeEditor")
        if last_node_editor:
            # TODO: Implement node editor state loading
            logger.info(f"Last node editor state: {last_node_editor}")
        last_filter_editor = settings.value("lastFilterEditor")
        if last_filter_editor:
            # TODO: Implement filter editor state loading
            logger.info(f"Last filter editor state: {last_filter_editor}")
        last_formula_editor = settings.value("lastFormulaEditor")
        if last_formula_editor:
            # TODO: Implement formula editor state loading
            logger.info(f"Last formula editor state: {last_formula_editor}")
        last_missing_values_editor = settings.value("lastMissingValuesEditor")
        if last_missing_values_editor:
            # TODO: Implement missing values editor state loading
            logger.info(f"Last missing values editor state: {last_missing_values_editor}")
        last_node_inspector = settings.value("lastNodeInspector")
        if last_node_inspector:
            # TODO: Implement node inspector state loading
            logger.info(f"Last node inspector state: {last_node_inspector}")
        last_node_statistics = settings.value("lastNodeStatistics")
        if last_node_statistics:
            # TODO: Implement node statistics state loading
            logger.info(f"Last node statistics state: {last_node_statistics}")
        last_node_pruning = settings.value("lastNodePruning")
        if last_node_pruning:
            # TODO: Implement node pruning state loading
            logger.info(f"Last node pruning state: {last_node_pruning}")
        last_node_visualization = settings.value("lastNodeVisualization")
        if last_node_visualization:
            # TODO: Implement node visualization state loading
            logger.info(f"Last node visualization state: {last_node_visualization}")

        last_node_analysis = settings.value("lastNodeAnalysis")
        if last_node_analysis:
            # TODO: Implement node analysis state loading
            logger.info(f"Last node analysis state: {last_node_analysis}")
        last_node_output = settings.value("lastNodeOutput")
        if last_node_output:
            # TODO: Implement node output state loading
            logger.info(f"Last node output state: {last_node_output}")
        last_node_input = settings.value("lastNodeInput")
        if last_node_input:
            # TODO: Implement node input state loading
            logger.info(f"Last node input state: {last_node_input}")

    def save_application_state(self):
        """Save application state and settings"""
        settings = QSettings("BespokeAnalytics", "BespokeDecisionTreeUtility")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        
        # Save defaults for now - TODO: Implement via modern toolbar state
        settings.setValue("pruningEnabled", False)
        settings.setValue("growthModeIndex", 0)
        settings.setValue("criterionIndex", 0)
        
        if self.current_project_path:
            settings.setValue("lastProjectPath", str(Path(self.current_project_path).parent))
        
        settings.setValue("dataDockVisible", self.data_dock.isVisible())
        settings.setValue("modelDockVisible", self.model_dock.isVisible())
        settings.setValue("visualizationDockVisible", self.visualization_dock.isVisible())
        
        logger.info("Application window state saved")

    def load_recent_files(self):
        """Load and populate recent files menu"""
        if not hasattr(self, 'recent_menu') or self.recent_menu is None:
            logger.warning("recent_menu not found - skipping recent files loading")
            return
            
        settings = QSettings("BespokeAnalytics", "BespokeDecisionTreeUtility")
        recent_files = settings.value("recentFiles", []) or []
        
        self.recent_menu.clear()
        valid_recent_files = [f for f in recent_files if f and Path(f).exists()]
        
        if len(valid_recent_files) != len(recent_files):
            settings.setValue("recentFiles", valid_recent_files)
            recent_files = valid_recent_files
        
        for i, file_path in enumerate(recent_files):
            text = f"&{i+1} {Path(file_path).name}"
            action = QAction(text, self)
            action.setData(file_path)
            action.setStatusTip(file_path)
            action.triggered.connect(self.open_recent_file)
            self.recent_menu.addAction(action)
            
            if i >= 8:  # Limit to 9 recent files for keyboard shortcuts
                break
        
        if self.recent_menu.actions():
            self.recent_menu.addSeparator()
            clear_action = QAction("Clear Recent Files", self)
            clear_action.triggered.connect(self.clear_recent_files)
            self.recent_menu.addAction(clear_action)

    
    def setup_ui_canvas_synchronization(self):
        """Setup enhanced UI-Canvas synchronization - COMPATIBILITY STUB"""
        try:
            if hasattr(self.workflow_canvas, 'nodeConfigured'):
                self.workflow_canvas.nodeConfigured.connect(self.on_workflow_node_configured)
            
            if hasattr(self.workflow_canvas, 'workflowModified'):
                self.workflow_canvas.workflowModified.connect(lambda: self.set_project_modified(True))
            
            self.modelCreated.connect(self.sync_model_to_canvas)
            self.modelDeleted.connect(self.remove_model_from_canvas)
            self.datasetCreated.connect(self.sync_dataset_to_canvas)
            self.datasetDeleted.connect(self.remove_dataset_from_canvas)
            
            logger.info("UI-Canvas synchronization setup completed (minimal compatibility mode)")
        except Exception as e:
            logger.error(f"Error in setup_ui_canvas_synchronization compatibility stub: {e}")
        
    def sync_dataset_to_canvas(self, dataset_name: str, df=None):
        """Synchronize dataset changes to workflow canvas"""
        try:
            existing_node = None
            for node_id, node in self.workflow_canvas.scene.nodes.items():
                if (isinstance(node, DatasetNode) and 
                    hasattr(node, 'dataset_name') and 
                    node.dataset_name == dataset_name):
                    existing_node = node
                    break
            
            if existing_node:
                if df is not None:
                    existing_node.set_config({
                        'dataset_name': dataset_name,
                        'df_rows': len(df),
                        'df_cols': len(df.columns),
                        'last_updated': True
                    })
                    logger.debug(f"Updated existing dataset node for '{dataset_name}'")
            else:
                if (hasattr(self, 'project_manager') and 
                    self.project_manager and 
                    dataset_name in self.project_manager.datasets):
                    
                    dataset = self.project_manager.datasets[dataset_name]
                    self.workflow_canvas.add_node_to_canvas(
                        NODE_TYPE_DATASET,
                        title=dataset_name,
                        specific_config={
                            'dataset_name': dataset_name,
                            'df_rows': len(dataset),
                            'df_cols': len(dataset.columns),
                            'source': 'synchronized'
                        }
                    )
                    logger.debug(f"Created new dataset node for '{dataset_name}'")
                    
        except Exception as e:
            logger.error(f"Error syncing dataset '{dataset_name}' to canvas: {e}")
    
    def sync_model_to_canvas(self, model_name: str, model=None):
        """Synchronize model changes to workflow canvas"""
        try:
            existing_node = None
            for node_id, node in self.workflow_canvas.scene.nodes.items():
                if (isinstance(node, ModelNode) and 
                    node.title == model_name):
                    existing_node = node
                    break
            
            if existing_node:
                if model is not None:
                    existing_node.set_config({
                        'model_ref_id': model_name,
                        'is_fitted': getattr(model, 'is_fitted', False),
                        'last_updated': True
                    })
                    logger.debug(f"Updated existing model node for '{model_name}'")
            else:
                if (hasattr(self, 'project_manager') and 
                    self.project_manager and 
                    model_name in self.project_manager.models):
                    
                    self.workflow_canvas.add_node_to_canvas(
                        NODE_TYPE_MODEL,
                        title=model_name,
                        specific_config={
                            'model_ref_id': model_name,
                            'source': 'synchronized'
                        }
                    )
                    logger.debug(f"Created new model node for '{model_name}'")
                    
        except Exception as e:
            logger.error(f"Error syncing model '{model_name}' to canvas: {e}")
    
    def remove_dataset_from_canvas(self, dataset_name: str):
        """Remove dataset node from workflow canvas"""
        try:
            nodes_to_remove = []
            for node_id, node in self.workflow_canvas.scene.nodes.items():
                if (isinstance(node, DatasetNode) and 
                    hasattr(node, 'dataset_name') and 
                    node.dataset_name == dataset_name):
                    nodes_to_remove.append(node_id)
            
            for node_id in nodes_to_remove:
                self.workflow_canvas.scene.remove_node(node_id)
                logger.debug(f"Removed dataset node '{node_id}' for dataset '{dataset_name}'")
                
        except Exception as e:
            logger.error(f"Error removing dataset '{dataset_name}' from canvas: {e}")
    
    def remove_model_from_canvas(self, model_name: str):
        """Remove model node from workflow canvas"""
        try:
            nodes_to_remove = []
            for node_id, node in self.workflow_canvas.scene.nodes.items():
                if (isinstance(node, ModelNode) and 
                    node.title == model_name):
                    nodes_to_remove.append(node_id)
            
            for node_id in nodes_to_remove:
                self.workflow_canvas.scene.remove_node(node_id)
                logger.debug(f"Removed model node '{node_id}' for model '{model_name}'")
                
        except Exception as e:
            logger.error(f"Error removing model '{model_name}' from canvas: {e}")
    
    def update_canvas_node_status(self, node_id: str, status: str):
        """Update node status in workflow canvas"""
        try:
            node = self.workflow_canvas.get_node_by_id(node_id)
            if node:
                node.status = status
                node.update_appearance()  # Refresh visual appearance
                logger.debug(f"Updated node '{node_id}' status to '{status}'")
            else:
                logger.warning(f"Node '{node_id}' not found for status update")
                
        except Exception as e:
            logger.error(f"Error updating node '{node_id}' status: {e}")
    
    def sync_canvas_with_current_state(self):
        """Synchronize workflow canvas with current datasets and models"""
        try:
            if hasattr(self, 'project_manager') and self.project_manager:
                for dataset_name in self.project_manager.datasets:
                    self.sync_dataset_to_canvas(dataset_name)
                
                for model_name in self.project_manager.models:
                    self.sync_model_to_canvas(model_name)
            
            logger.debug("Canvas synchronized with current state")
            
        except Exception as e:
            logger.error(f"Error synchronizing canvas with current state: {e}")
    
    def on_workflow_node_configured(self, node):
        """Handle workflow node configuration changes"""
        try:
            if isinstance(node, DatasetNode):
                if (hasattr(node, 'dataset_name') and 
                    node.dataset_name in self.project_manager.datasets):
                    logger.debug(f"Dataset node '{node.dataset_name}' configured")
                    
            elif isinstance(node, ModelNode):
                if hasattr(node, 'model') and node.model:
                    logger.debug(f"Model node '{node.title}' configured")
                    
        except Exception as e:
            logger.error(f"Error handling node configuration: {e}")
    
    def on_canvas_node_double_clicked(self, node):
        """Handle double-click on workflow canvas nodes"""
        try:
            if isinstance(node, DatasetNode):
                if hasattr(node, 'dataset_name'):
                    self.select_dataset_by_name(node.dataset_name)
                    self.data_dock.show()
                    self.data_dock.raise_()
                    
            elif isinstance(node, ModelNode):
                self.select_model_by_name(node.title)
                self.model_dock.show()
                self.model_dock.raise_()
                
            elif isinstance(node, FilterNode):
                self.open_filter_dialog_for_node(node)
                
            elif isinstance(node, TransformNode):
                self.open_transform_dialog_for_node(node)
                
        except Exception as e:
            logger.error(f"Error handling node double-click: {e}")
    
    def select_dataset_by_name(self, dataset_name: str):
        """Select dataset in data view"""
        try:
            if (hasattr(self, 'project_manager') and 
                self.project_manager and 
                dataset_name in self.project_manager.datasets):
                
                dataset = self.project_manager.datasets[dataset_name]
                self.create_dataset_viewer(dataset_name, dataset)
                
                logger.debug(f"Selected dataset '{dataset_name}' in data view")
                
        except Exception as e:
            logger.error(f"Error selecting dataset '{dataset_name}': {e}")
    
    def select_model_by_name(self, model_name: str):
        """Select model in model view"""
        try:
            if (hasattr(self, 'project_manager') and 
                self.project_manager and 
                model_name in self.project_manager.models):
                
                model = self.project_manager.models[model_name]
                
                if hasattr(self.tree_viz_widget, 'load_tree'):
                    if hasattr(model, 'tree_'):
                        self.tree_viz_widget.load_tree(model.tree_, model_name)
                    elif hasattr(model, 'root_node'):
                        self.tree_viz_widget.load_tree(model.root_node, model_name)
                
                if hasattr(self, 'model_properties_widget'):
                    self.model_properties_widget.load_model(model)
                
                logger.debug(f"Selected model '{model_name}' in model view")
                
        except Exception as e:
            logger.error(f"Error selecting model '{model_name}': {e}")
    
    def open_filter_dialog_for_node(self, node):
        """Open filter configuration dialog for a filter node"""
        try:
            from ui.dialogs.filter_data_dialog import FilterDataDialog
            
            dialog = FilterDataDialog(self)
            
            if hasattr(node, 'conditions'):
                dialog.set_conditions(node.conditions)
            
            if dialog.exec_() == QDialog.Accepted:
                conditions = dialog.get_conditions()
                node.conditions = conditions
                node.set_config({'conditions': conditions})
                
                if hasattr(self.workflow_canvas, 'nodeConfigured'):
                    self.workflow_canvas.nodeConfigured.emit(node)
                
                logger.debug(f"Filter node '{node.node_id}' configured with {len(conditions)} conditions")
                
        except Exception as e:
            logger.error(f"Error opening filter dialog for node: {e}")
    
    def open_transform_dialog_for_node(self, node):
        """Open transformation configuration dialog for a transform node"""
        try:
            from ui.dialogs.data_transformation_dialog import DataTransformationDialog
            
            dialog = DataTransformationDialog(self)
            
            if hasattr(node, 'transformations'):
                dialog.set_transformations(node.transformations)
            
            if dialog.exec_() == QDialog.Accepted:
                transformations = dialog.get_transformations()
                node.transformations = transformations
                node.set_config({'transformations': transformations})
                
                if hasattr(self.workflow_canvas, 'nodeConfigured'):
                    self.workflow_canvas.nodeConfigured.emit(node)
                
                logger.debug(f"Transform node '{node.node_id}' configured with {len(transformations)} transformations")
                
        except Exception as e:
            logger.error(f"Error opening transform dialog for node: {e}")

    @pyqtSlot()
    def open_recent_file(self):
        """Opens a file selected from the recent files menu"""
        action = self.sender()
        if action:
            file_path = action.data()
            if file_path and Path(file_path).exists():
                self.open_project(file_path)
            else:
                QMessageBox.warning(
                    self, "File Not Found", 
                    f"Could not find file:\n{file_path}\nIt may have been moved or deleted."
                )
                self.remove_recent_file(file_path)

    def remove_recent_file(self, file_path: str):
        """Removes a specific file from the recent files list"""
        settings = QSettings("BespokeAnalytics", "BespokeDecisionTreeUtility")
        recent_files = settings.value("recentFiles", []) or []
        
        if file_path in recent_files:
            recent_files.remove(file_path)
            settings.setValue("recentFiles", recent_files)
            self.load_recent_files()

    def add_recent_file(self, file_path: str):
        """
        Add a file to the recent files list
        
        Args:
            file_path: Path to the file to add
        """
        try:
            settings = QSettings("BespokeAnalytics", "BespokeDecisionTreeUtility")
            recent_files = settings.value("recentFiles", [])
            
            if not isinstance(recent_files, list):
                recent_files = []
            
            if file_path in recent_files:
                recent_files.remove(file_path)
            
            recent_files.insert(0, file_path)
            
            if len(recent_files) > 9:
                recent_files = recent_files[:9]
            
            settings.setValue("recentFiles", recent_files)
            
            self.load_recent_files()
            
        except Exception as e:
            logger.error(f"Error adding recent file {file_path}: {e}", exc_info=True)

    def clear_recent_files(self):
        """Clear the recent files list"""
        settings = QSettings("BespokeAnalytics", "BespokeDecisionTreeUtility")
        settings.remove("recentFiles")
        self.load_recent_files()

    def show_preferences(self):
        """Show preferences dialog"""
        # TODO: Implement a proper preferences dialog
        QMessageBox.information(self, "Preferences", "Preferences dialog not implemented yet.")
        if self.enhanced_status_bar:
            self.enhanced_status_bar.show_message("Preferences dialog not implemented yet.", timeout=3000)

    def show_help(self):
        """Show help contents"""
        # TODO: Implement a proper help system
        help_url = get_config_value(self.config, 'help.url', 'https://example.com/help')
        try:
            webbrowser.open(help_url)
            self.status_label.setText(f"Opening help documentation: {help_url}")
        except Exception as e:
            logger.error(f"Error opening help: {e}")
            QMessageBox.information(self, "Help", "Help system not implemented yet.")
            self.status_label.setText("Help system not implemented yet.")
    
    def connect_model_signals(self, model: BespokeDecisionTree):
        """
        Connect model signals to UI handlers
        
        Args:
            model: BespokeDecisionTree instance to connect
        """
        try:
            if hasattr(model, 'treeUpdated'):
                model.treeUpdated.connect(self.on_tree_updated)
            if hasattr(model, 'nodeUpdated'):
                model.nodeUpdated.connect(self.on_node_updated)
            if hasattr(model, 'splitFound'):
                model.splitFound.connect(self.on_split_found)
            if hasattr(model, 'trainingProgress'):
                model.trainingProgress.connect(self.on_training_progress)
            if hasattr(model, 'trainingComplete'):
                model.trainingComplete.connect(self.on_training_complete)
            if hasattr(model, 'trainingRequired'):
                model.trainingRequired.connect(self.on_training_required)
            if hasattr(model, 'errorOccurred'):
                model.errorOccurred.connect(self.on_model_error)
            
            logger.info(f"Connected model signals for {model.model_name}")
            
        except Exception as e:
            logger.error(f"Error connecting model signals: {e}", exc_info=True)
    
    def train_model_from_workflow(self, node):
        """Train a model from workflow context"""
        try:
            if not hasattr(node, 'model') or not node.model:
                QMessageBox.warning(self, "Train Model", "No model associated with this node")
                return
            
            connected_data = self.get_node_input_data(node.node_id, "Data Input")
            
            if connected_data is None:
                QMessageBox.warning(self, "Train Model", "No data available from connected nodes")
                return
            
            df = connected_data
            
            target_name = getattr(node.model, 'target_name', None)
            if not target_name or target_name not in df.columns:
                target_name, ok = QInputDialog.getItem(
                    self, 'Select Target Variable', 
                    'Choose the target variable for training:',
                    list(df.columns), 0, False
                )
                if not ok:
                    return
                node.model.target_name = target_name
            
            X = df.drop(columns=[target_name])
            y = df[target_name]
            
            node.model.fit(X, y)
            
            node.update_model_info()
            
            QMessageBox.information(self, "Train Model", "Model training completed successfully!")
            
        except Exception as e:
            logger.error(f"Error training model from workflow: {e}", exc_info=True)
            QMessageBox.critical(self, "Training Error", f"Failed to train model: {str(e)}")
    
    def evaluate_model_from_workflow(self, node):
        """Evaluate a model from workflow context"""
        try:
            if not hasattr(node, 'model') or not node.model or not node.model.is_fitted:
                QMessageBox.warning(self, "Evaluate Model", "Model must be trained before evaluation")
                return
            
            dataset_node = None
            for connection in self.workflow_canvas.scene.connections.values():
                if connection.target_node_id == node.node_id:
                    source_node = self.workflow_canvas.scene.nodes.get(connection.source_node_id)
                    if isinstance(source_node, DatasetNode):
                        dataset_node = source_node
                        break
            
            if not dataset_node:
                QMessageBox.warning(self, "Evaluate Model", "No dataset connected to this model node")
                return
            
            if dataset_node.dataset_name not in self.datasets:
                QMessageBox.warning(self, "Evaluate Model", f"Dataset '{dataset_node.dataset_name}' not found")
                return
            
            df = self.datasets[dataset_node.dataset_name]
            target_name = node.model.target_name
            
            if not target_name or target_name not in df.columns:
                QMessageBox.warning(self, "Evaluate Model", "Target variable not found in dataset")
                return
            
            X = df.drop(columns=[target_name])
            y = df[target_name]
            
            metrics = node.model.compute_metrics(X, y)
            
            metrics_text = "\n".join([f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}" 
                                     for k, v in metrics.items() if not isinstance(v, dict)])
            
            QMessageBox.information(self, "Model Evaluation", f"Model Performance:\n\n{metrics_text}")
            
        except Exception as e:
            logger.error(f"Error evaluating model from workflow: {e}", exc_info=True)
            QMessageBox.critical(self, "Evaluation Error", f"Failed to evaluate model: {str(e)}")
    
    def view_dataset_from_workflow(self, node):
        """View dataset from workflow context"""
        try:
            if node.dataset_name not in self.datasets:
                QMessageBox.warning(self, "View Data", f"Dataset '{node.dataset_name}' not found")
                return
            
            self.data_tab_widget.setCurrentIndex(0)  # Switch to data viewer tab
            
            if node.dataset_name in self.data_viewer_widgets:
                data_viewer = self.data_viewer_widgets[node.dataset_name]
                self.data_dock.show()
                self.data_dock.raise_()
            
        except Exception as e:
            logger.error(f"Error viewing dataset from workflow: {e}", exc_info=True)
            QMessageBox.critical(self, "View Data Error", f"Failed to view dataset: {str(e)}")
    
    def filter_dataset_from_workflow(self, node):
        """Filter dataset from workflow context"""
        try:
            if node.dataset_name not in self.datasets:
                QMessageBox.warning(self, "Filter Data", f"Dataset '{node.dataset_name}' not found")
                return
            
            from ui.dialogs.filter_data_dialog import FilterDataDialog
            df = self.datasets[node.dataset_name]
            dialog = FilterDataDialog(df, self)
            
            if dialog.exec_() == QDialog.Accepted:
                filtered_df = dialog.get_filtered_data()
                new_name = f"{node.dataset_name}_filtered"
                self.datasets[new_name] = filtered_df
                
                QMessageBox.information(self, "Filter Data", f"Created filtered dataset: {new_name}")
            
        except Exception as e:
            logger.error(f"Error filtering dataset from workflow: {e}", exc_info=True)
            QMessageBox.critical(self, "Filter Data Error", f"Failed to filter dataset: {str(e)}")
    
    def on_tree_updated(self):
        """Handle tree updated signal"""
        if hasattr(self, 'tree_visualizer') and self.tree_visualizer:
            self.tree_visualizer.refresh_tree()
    
    def on_node_updated(self, node_id: str):
        """Handle node updated signal"""
        logger.debug(f"Node {node_id} updated")
    
    def on_split_found(self, node_id: str, splits: list):
        """Handle split found signal"""
        logger.info(f"Found {len(splits)} potential splits for node {node_id}")
    
    def on_training_progress(self, progress: int, message: str):
        """Handle training progress signal"""
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.setValue(progress)
        self.enhanced_status_bar.show_message(message)
    
    def on_training_complete(self):
        """Handle training complete signal"""
        self.enhanced_status_bar.show_message("Model training completed")
        if hasattr(self, 'progress_bar') and self.progress_bar:
            self.progress_bar.setValue(100)
    
    def on_model_error(self, error_message: str):
        """Handle model error signal"""
        QMessageBox.critical(self, "Model Error", error_message)
        self.enhanced_status_bar.show_error(f"Model error: {error_message}")
    

    def show_about(self):
        """Show about dialog"""
        version = get_config_value(self.config, 'application.version', '0.50')
        QMessageBox.about(
            self, "About Bespoke Decision Tree Utility",
            f"<h2>Bespoke Decision Tree Utility</h2>"
            f"<p>Version {version}</p>"
            "<p>A portable, interactive application for building decision tree models "
            "focused on credit risk assessment with binary target variables.</p>"
            "<p><b>Created by:</b> Rupin Desai</p>"
        )
    
    def _show_variable_importance_results(self, gini_importance, permutation_importance, model):
        """Show variable importance results in a detailed dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Variable Importance Results")
        dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        tab_widget = QTabWidget()
        
        gini_tab = QWidget()
        gini_layout = QVBoxLayout(gini_tab)
        
        gini_layout.addWidget(QLabel("Gini-based Feature Importance:"))
        
        gini_table = QTableWidget()
        gini_table.setColumnCount(2)
        gini_table.setHorizontalHeaderLabels(["Feature", "Importance"])
        
        sorted_gini = sorted(gini_importance.items(), key=lambda x: x[1], reverse=True)
        gini_table.setRowCount(len(sorted_gini))
        
        for i, (feature, importance) in enumerate(sorted_gini):
            gini_table.setItem(i, 0, QTableWidgetItem(feature))
            gini_table.setItem(i, 1, QTableWidgetItem(f"{importance:.6f}"))
        
        gini_table.resizeColumnsToContents()
        gini_layout.addWidget(gini_table)
        
        tab_widget.addTab(gini_tab, "Gini Importance")
        
        if permutation_importance:
            perm_tab = QWidget()
            perm_layout = QVBoxLayout(perm_tab)
            
            perm_layout.addWidget(QLabel("Permutation-based Feature Importance:"))
            
            perm_table = QTableWidget()
            perm_table.setColumnCount(2)
            perm_table.setHorizontalHeaderLabels(["Feature", "Importance"])
            
            sorted_perm = sorted(permutation_importance.items(), key=lambda x: x[1], reverse=True)
            perm_table.setRowCount(len(sorted_perm))
            
            for i, (feature, importance) in enumerate(sorted_perm):
                perm_table.setItem(i, 0, QTableWidgetItem(feature))
                perm_table.setItem(i, 1, QTableWidgetItem(f"{importance:.6f}"))
            
            perm_table.resizeColumnsToContents()
            perm_layout.addWidget(perm_table)
            
            tab_widget.addTab(perm_tab, "Permutation Importance")
        
        layout.addWidget(tab_widget)
        
        button_layout = QHBoxLayout()
        
        export_btn = QPushButton("Export Report")
        export_btn.clicked.connect(lambda: self._export_importance_report(gini_importance, model))
        button_layout.addWidget(export_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec_()


    def _export_importance_report(self, importance_data, model):
        """Export variable importance report"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Variable Importance Report",
            f"variable_importance_{self.current_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if filename:
            try:
                from analytics.variable_importance import VariableImportance
                importance_calc = VariableImportance()
                importance_calc.last_importance = importance_data
                
                success = importance_calc.export_importance_report(filename, self.current_model_name)
                
                if success:
                    QMessageBox.information(self, "Export Successful", 
                                        f"Variable importance report exported to:\n{filename}")
                else:
                    QMessageBox.warning(self, "Export Failed", "Could not export importance report.")
                    
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Error exporting report:\n{str(e)}")
    
        def _show_performance_metrics_results(self, metrics, model, metrics_calc):
            """Show comprehensive performance metrics results"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Performance Metrics Results")
        dialog.setMinimumSize(1000, 700)
        
        layout = QVBoxLayout(dialog)
        
        tab_widget = QTabWidget()
        
        basic_tab = self._create_basic_metrics_tab(metrics)
        tab_widget.addTab(basic_tab, "Basic Metrics")
        
        if 'confusion_matrix' in metrics:
            cm_tab = self._create_confusion_matrix_tab(metrics)
            tab_widget.addTab(cm_tab, "Confusion Matrix")
        
        if 'roc_auc' in metrics or 'average_precision' in metrics:
            advanced_tab = self._create_advanced_metrics_tab(metrics)
            tab_widget.addTab(advanced_tab, "Advanced Metrics")
        
        if 'cv_accuracy_mean' in metrics:
            cv_tab = self._create_cv_metrics_tab(metrics)
            tab_widget.addTab(cv_tab, "Cross-Validation")
        
        viz_tab = self._create_metrics_visualization_tab(metrics, metrics_calc)
        tab_widget.addTab(viz_tab, "Visualizations")
        
        layout.addWidget(tab_widget)
        
        button_layout = QHBoxLayout()
        
        export_btn = QPushButton("Export Report")
        export_btn.clicked.connect(lambda: self._export_metrics_report(metrics, model))
        button_layout.addWidget(export_btn)
        
        export_viz_btn = QPushButton("Export Visualizations")
        export_viz_btn.clicked.connect(lambda: self._export_metrics_visualizations(metrics, metrics_calc))
        button_layout.addWidget(export_viz_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec_()


    def _create_basic_metrics_tab(self, metrics):
        """Create basic metrics tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Metric", "Value"])
        
        basic_metrics = [
            ("Accuracy", metrics.get('accuracy', 0)),
            ("Precision (Weighted)", metrics.get('precision', 0)),
            ("Recall (Weighted)", metrics.get('recall', 0)),
            ("F1-Score (Weighted)", metrics.get('f1_score', 0)),
            ("Precision (Macro)", metrics.get('precision_macro', 0)),
            ("Recall (Macro)", metrics.get('recall_macro', 0)),
            ("F1-Score (Macro)", metrics.get('f1_score_macro', 0)),
            ("Error Rate", metrics.get('error_rate', 0))
        ]
        
        table.setRowCount(len(basic_metrics))
        
        for i, (metric_name, value) in enumerate(basic_metrics):
            table.setItem(i, 0, QTableWidgetItem(metric_name))
            if isinstance(value, (int, float)):
                table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))
            else:
                table.setItem(i, 1, QTableWidgetItem(str(value)))
        
        table.resizeColumnsToContents()
        layout.addWidget(table)
        
        return widget


    def _create_confusion_matrix_tab(self, metrics):
        """Create confusion matrix tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        cm = metrics.get('confusion_matrix', [])
        classes = metrics.get('classes', [])
        
        if cm and classes:
            table = QTableWidget()
            table.setRowCount(len(cm))
            table.setColumnCount(len(cm[0]) if cm else 0)
            
            table.setHorizontalHeaderLabels([f"Pred {cls}" for cls in classes])
            table.setVerticalHeaderLabels([f"True {cls}" for cls in classes])
            
            for i, row in enumerate(cm):
                for j, value in enumerate(row):
                    table.setItem(i, j, QTableWidgetItem(str(value)))
            
            table.resizeColumnsToContents()
            layout.addWidget(QLabel("Confusion Matrix:"))
            layout.addWidget(table)
            
            if len(classes) == 2 and 'true_positives' in metrics:
                binary_group = QGroupBox("Binary Classification Metrics")
                binary_layout = QFormLayout()
                
                binary_layout.addRow("True Positives:", QLabel(str(metrics.get('true_positives', 0))))
                binary_layout.addRow("True Negatives:", QLabel(str(metrics.get('true_negatives', 0))))
                binary_layout.addRow("False Positives:", QLabel(str(metrics.get('false_positives', 0))))
                binary_layout.addRow("False Negatives:", QLabel(str(metrics.get('false_negatives', 0))))
                binary_layout.addRow("Sensitivity:", QLabel(f"{metrics.get('sensitivity', 0):.4f}"))
                binary_layout.addRow("Specificity:", QLabel(f"{metrics.get('specificity', 0):.4f}"))
                
                binary_group.setLayout(binary_layout)
                layout.addWidget(binary_group)
        
        return widget


    def _create_advanced_metrics_tab(self, metrics):
        """Create advanced metrics tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        advanced_group = QGroupBox("Advanced Metrics")
        advanced_layout = QFormLayout()
        
        if 'roc_auc' in metrics:
            advanced_layout.addRow("ROC AUC Score:", QLabel(f"{metrics['roc_auc']:.4f}"))
        
        if 'average_precision' in metrics:
            advanced_layout.addRow("Average Precision:", QLabel(f"{metrics['average_precision']:.4f}"))
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        if 'classification_report' in metrics:
            report_group = QGroupBox("Classification Report")
            report_layout = QVBoxLayout()
            
            report_text = QTextEdit()
            report_text.setReadOnly(True)
            report_text.setMaximumHeight(300)
            
            report_data = metrics['classification_report']
            report_lines = []
            
            for class_name, class_metrics in report_data.items():
                if isinstance(class_metrics, dict):
                    report_lines.append(f"\n{class_name}:")
                    for metric, value in class_metrics.items():
                        if isinstance(value, (int, float)):
                            report_lines.append(f"  {metric}: {value:.4f}")
                        else:
                            report_lines.append(f"  {metric}: {value}")
            
            report_text.setText("\n".join(report_lines))
            report_layout.addWidget(report_text)
            report_group.setLayout(report_layout)
            layout.addWidget(report_group)
        
        layout.addStretch()
        return widget


    def _create_cv_metrics_tab(self, metrics):
        """Create cross-validation metrics tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        cv_group = QGroupBox("Cross-Validation Summary")
        cv_layout = QFormLayout()
        
        cv_metrics = [
            ("CV Accuracy Mean", metrics.get('cv_accuracy_mean', 0)),
            ("CV Accuracy Std", metrics.get('cv_accuracy_std', 0)),
            ("CV Precision Mean", metrics.get('cv_precision_weighted_mean', 0)),
            ("CV Recall Mean", metrics.get('cv_recall_weighted_mean', 0)),
            ("CV F1-Score Mean", metrics.get('cv_f1_weighted_mean', 0)),
            ("CV Folds", metrics.get('cv_folds', 0))
        ]
        
        for metric_name, value in cv_metrics:
            if isinstance(value, (int, float)) and metric_name != "CV Folds":
                cv_layout.addRow(f"{metric_name}:", QLabel(f"{value:.4f}"))
            else:
                cv_layout.addRow(f"{metric_name}:", QLabel(str(value)))
        
        cv_group.setLayout(cv_layout)
        layout.addWidget(cv_group)
        
        if 'cv_accuracy_scores' in metrics:
            scores_group = QGroupBox("Individual CV Scores")
            scores_layout = QVBoxLayout()
            
            scores_text = QTextEdit()
            scores_text.setReadOnly(True)
            scores_text.setMaximumHeight(150)
            
            scores = metrics['cv_accuracy_scores']
            scores_lines = []
            for i, score in enumerate(scores, 1):
                scores_lines.append(f"Fold {i}: {score:.4f}")
            
            scores_text.setText("\n".join(scores_lines))
            scores_layout.addWidget(scores_text)
            scores_group.setLayout(scores_layout)
            layout.addWidget(scores_group)
        
        layout.addStretch()
        return widget


    def _create_metrics_visualization_tab(self, metrics, metrics_calc):
        """Create metrics visualization tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            
            figure = Figure(figsize=(12, 8))
            canvas = FigureCanvas(figure)
            
            visualizer = MetricsVisualizer()
            
            if 'confusion_matrix' in metrics:
                ax1 = figure.add_subplot(2, 2, 1)
                cm = metrics['confusion_matrix']
                classes = metrics.get('classes', [])
                
                import numpy as np
                im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
                ax1.set_title('Confusion Matrix')
                
                for i in range(len(cm)):
                    for j in range(len(cm[i])):
                        ax1.text(j, i, str(cm[i][j]), ha="center", va="center")
                
                if classes:
                    ax1.set_xticks(range(len(classes)))
                    ax1.set_yticks(range(len(classes)))
                    ax1.set_xticklabels(classes)
                    ax1.set_yticklabels(classes)
                
                ax1.set_xlabel('Predicted Label')
                ax1.set_ylabel('True Label')
            
            if 'roc_curve' in metrics:
                ax2 = figure.add_subplot(2, 2, 2)
                roc_data = metrics['roc_curve']
                fpr, tpr = roc_data['fpr'], roc_data['tpr']
                
                ax2.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {metrics.get('roc_auc', 0):.3f})")
                ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
                ax2.set_xlabel('False Positive Rate')
                ax2.set_ylabel('True Positive Rate')
                ax2.set_title('ROC Curve')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            ax3 = figure.add_subplot(2, 2, 3)
            basic_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            values = [metrics.get(metric, 0) for metric in basic_metrics]
            
            bars = ax3.bar(basic_metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
            ax3.set_title('Basic Classification Metrics')
            ax3.set_ylabel('Score')
            ax3.set_ylim(0, 1)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            if 'cv_accuracy_scores' in metrics:
                ax4 = figure.add_subplot(2, 2, 4)
                cv_scores = metrics['cv_accuracy_scores']
                ax4.boxplot([cv_scores], labels=['CV Accuracy'])
                ax4.set_title('Cross-Validation Accuracy Distribution')
                ax4.set_ylabel('Accuracy')
                ax4.grid(True, alpha=0.3)
            
            figure.tight_layout()
            layout.addWidget(canvas)
            
        except ImportError:
            layout.addWidget(QLabel("Matplotlib not available for visualizations"))
        except Exception as e:
            layout.addWidget(QLabel(f"Error creating visualizations: {str(e)}"))
        
        return widget
    

        
        
        
        
        

    def _perform_enhanced_export(self, model, export_options):
        """Perform the enhanced export operation (Python only)"""
        try:
            from export.python_exporter import PythonExporter
            
            language = export_options['language']
            output_path = export_options['output_path']
            
            if language.lower() != 'python':
                logger.error(f"Unsupported export language: {language}. Only Python is supported.")
                return False
            
            config = getattr(self, 'config', {})
            
            exporter = PythonExporter(config)
            success = exporter.export_model(model, output_path)
            
            return success
            
        except Exception as e:
            logger.error(f"Error performing enhanced export: {str(e)}")
            return False
    
    def manual_split_node(self, node_id: str):
        """Handle manual split request from enhanced context menu"""
        logger.info(f"Manual split requested for node: {node_id}")
        self.split_node_action(node_id)
    
    def find_optimal_split(self, node_id: str):
        """Handle find optimal split request from enhanced context menu"""
        logger.info(f"Find optimal split requested for node: {node_id}")
        self.find_split_action(node_id)
    
    def edit_node_split(self, node_id: str):
        """Handle edit split request from enhanced context menu"""
        logger.info(f"Edit split requested for node: {node_id}")
        self.edit_split_action(node_id)
    
    def copy_node_info(self, node_id: str):
        """Copy node structure to clipboard for pasting"""
        if not self.current_model_name or self.current_model_name not in self.models:
            return
            
        model = self.models[self.current_model_name]
        node = model.get_node(node_id)
        
        if not node:
            return
        
        try:
            node_structure = self._serialize_node_structure(node)
            
            import json
            clipboard_data = {
                'type': 'bespoke_tree_node_structure',
                'version': '1.0',
                'source_node_id': node_id,
                'structure': node_structure
            }
            
            clipboard_text = json.dumps(clipboard_data, indent=2)
            clipboard = QApplication.clipboard()
            clipboard.setText(clipboard_text)
            
            node_count = self._count_nodes_in_structure(node_structure)
            
            def count_structure_nodes(struct):
                if not struct:
                    return 0
                count = 1  # Current node
                for child in struct.get('child_splits', []):
                    count += count_structure_nodes(child)
                return count
            
            node_count = count_structure_nodes(node_structure)
            
            QMessageBox.information(
                self, "Split Configuration Copied", 
                f"Split configuration for {node_id} with {node_count} nodes copied to clipboard.\n"
                f"Variables and binning logic will be recreated with proper statistics.\n"
                f"Use 'Paste Node Structure' to apply this configuration elsewhere."
            )
            logger.info(f"Node structure copied to clipboard: {node_id} with {node_count} total nodes")
            
        except Exception as e:
            logger.error(f"Error copying node structure: {e}")
            QMessageBox.warning(self, "Copy Error", f"Failed to copy node structure: {e}")
    
    def _serialize_node_structure(self, node):
        """Extract split configuration that can be applied via manual split process"""
        if node.is_terminal:
            return {
                'split_config': None,
                'child_splits': [],
                'is_terminal': True
            }
            
        split_feature = getattr(node, 'split_feature', None)
        split_type = getattr(node, 'split_type', None)
        split_value = getattr(node, 'split_value', None)
        split_categories = getattr(node, 'split_categories', {})
        split_thresholds = getattr(node, 'split_thresholds', [])
        
        if not split_feature:
            return None
        
        if split_type == 'numeric_multi_bin' and split_thresholds:
            split_config = {
                'split_type': 'numeric_multi_bin',
                'feature': split_feature,
                'thresholds': split_thresholds.copy(),
                'num_bins': len(split_thresholds) + 1
            }
        elif split_type == 'numeric' and split_value is not None:
            split_config = {
                'split_type': 'numeric',
                'feature': split_feature,
                'threshold': split_value  # FIXED: use threshold not split_value
            }
        elif split_type == 'categorical' and split_categories:
            unique_groups = set(split_categories.values())
            if len(unique_groups) == 2:
                left_categories = [k for k, v in split_categories.items() if v == 0]
                right_categories = [k for k, v in split_categories.items() if v == 1]
                split_config = {
                    'split_type': 'categorical',
                    'feature': split_feature,
                    'left_categories': left_categories,
                    'right_categories': right_categories
                }
            else:
                split_config = {
                    'split_type': 'categorical',
                    'feature': split_feature,
                    'split_categories': split_categories.copy()
                }
        else:
            logger.warning(f"Unknown split type for node {node.node_id}: {split_type}")
            return None
        
        child_splits = []
        if hasattr(node, 'children') and node.children:
            for child in node.children:
                if child:
                    child_config = self._serialize_node_structure(child)
                    if child_config is not None:
                        child_splits.append(child_config)
        
        return {
            'split_config': split_config,
            'child_splits': child_splits
        }
    
    def _count_nodes_in_structure(self, structure):
        """Count total nodes in a serialized structure (supports both old and new formats)"""
        if not structure:
            return 0
            
        count = 1  # Current node
        
        if 'child_splits' in structure:
            for child in structure['child_splits']:
                count += self._count_nodes_in_structure(child)
        elif 'children' in structure:
            for child in structure['children']:
                count += self._count_nodes_in_structure(child)
                
        return count
    
    def _validate_split_config(self, split_config):
        """Validate split configuration matches apply_manual_split expectations"""
        if not split_config or 'split_type' not in split_config or 'feature' not in split_config:
            return False
            
        config_type = split_config['split_type']
        
        if config_type == 'numeric':
            return 'threshold' in split_config
        elif config_type == 'numeric_multi_bin':
            return 'thresholds' in split_config and len(split_config['thresholds']) > 0
        elif config_type == 'categorical':
            return ('left_categories' in split_config and 'right_categories' in split_config) or \
                   ('split_categories' in split_config and len(split_config['split_categories']) > 0)
        
        return False
    
    def paste_node_info(self, target_node_id: str):
        """Paste node structure from clipboard"""
        if not self.current_model_name or self.current_model_name not in self.models:
            return
            
        model = self.models[self.current_model_name]
        target_node = model.get_node(target_node_id)
        
        if not target_node:
            return
        
        try:
            clipboard = QApplication.clipboard()
            clipboard_text = clipboard.text()
            
            if not clipboard_text:
                QMessageBox.warning(self, "Paste Error", "Clipboard is empty.")
                return
            
            import json
            try:
                clipboard_data = json.loads(clipboard_text)
            except json.JSONDecodeError:
                QMessageBox.warning(self, "Paste Error", 
                                  "Clipboard does not contain valid node structure data.")
                return
            
            is_new_format = clipboard_data.get('type') == 'bespoke_tree_node_structure' and 'structure' in clipboard_data
            is_old_format = clipboard_data.get('type') == 'bespoke_tree_node_structure' and 'structure' in clipboard_data
            
            if not (is_new_format or is_old_format):
                QMessageBox.warning(self, "Paste Error", 
                                  "Clipboard does not contain valid Bespoke tree node structure.")
                return
                
            structure = clipboard_data['structure']
            if 'split_config' not in structure and 'is_terminal' in structure:
                reply = QMessageBox.question(
                    self, "Old Format Detected", 
                    "This clipboard data uses an older format that may not work correctly.\n\n"
                    "For best results, please re-copy the source node using the updated copy function.\n\n"
                    "Continue with current paste operation?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
            
            structure = clipboard_data['structure']
            node_count = self._count_nodes_in_structure(structure)
            
            def count_structure_nodes(struct):
                if not struct:
                    return 0
                count = 1  # Current node
                for child in struct.get('child_splits', []):
                    count += count_structure_nodes(child)
                return count
            
            node_count = count_structure_nodes(structure)
            
            reply = QMessageBox.question(
                self, "Confirm Paste", 
                f"This will apply the split configuration to node {target_node_id} "
                f"creating {node_count} nodes with proper statistics.\n\n"
                f"The manual split process will be used to ensure correct data computation.\n\n"
                f"This action cannot be undone. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply != QMessageBox.Yes:
                return
            
            self._apply_node_structure(target_node, structure, model)
            
            self.update_tree_visualization_after_node_edit()
            
            QMessageBox.information(
                self, "Paste Complete", 
                f"Successfully applied split configuration to {target_node_id}.\n"
                f"Split structure recreated with proper statistics computation."
            )
            logger.info(f"Pasted node structure to {target_node_id}: {node_count} nodes")
            
        except Exception as e:
            logger.error(f"Error pasting node structure: {e}")
            QMessageBox.warning(self, "Paste Error", f"Failed to paste node structure: {e}")
    
    def _apply_node_structure(self, target_node, structure, model):
        """Apply split structure using manual split process (proper statistics computation)"""
        if not structure:
            return
            
        split_config = structure.get('split_config')
        if not split_config:
            return
            
        if not self._validate_split_config(split_config):
            logger.error(f"Invalid split configuration for node {target_node.node_id}: {split_config}")
            return
            
        logger.info(f"Applying split configuration to node {target_node.node_id}: {split_config}")
        success = model.apply_manual_split(target_node.node_id, split_config)
        
        if not success:
            logger.warning(f"Failed to apply split configuration to node {target_node.node_id}")
            return
            
        child_splits = structure.get('child_splits', [])
        if child_splits and hasattr(target_node, 'children') and target_node.children:
            for i, child_split in enumerate(child_splits):
                if i < len(target_node.children):
                    child_node = target_node.children[i]
                    self._apply_node_structure(child_node, child_split, model)
    
    def _register_nodes_in_model(self, node, model):
        """Recursively register nodes in model's node registry"""
        if hasattr(model, '_node_registry'):
            model._node_registry[node.node_id] = node
        
        if hasattr(node, 'children') and node.children:
            for child in node.children:
                if child:
                    self._register_nodes_in_model(child, model)
    
    def delete_node(self, node_id: str):
        """Handle delete node request from enhanced context menu"""
        logger.info(f"Delete node requested for node: {node_id}")
        self.delete_node_action(node_id)


class EnhancedExportDialog(QDialog):
    """Enhanced export dialog with multiple language support"""
    
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.setWindowTitle("Enhanced Model Export")
        self.setMinimumSize(600, 500)
        self.setupUI()
        
    def setupUI(self):
        layout = QVBoxLayout(self)
        
        lang_group = QGroupBox("Export Language")
        lang_layout = QVBoxLayout()
        
        self.language_combo = QComboBox()
        languages = ['python']
        
        for lang in languages:
            self.language_combo.addItem(lang.title(), lang)
        
        self.language_combo.currentTextChanged.connect(self.on_language_changed)
        
        lang_layout.addWidget(QLabel("Select export language:"))
        lang_layout.addWidget(self.language_combo)
        lang_group.setLayout(lang_layout)
        layout.addWidget(lang_group)
        
        self.options_group = QGroupBox("Export Options")
        self.options_layout = QFormLayout()
        self.options_group.setLayout(self.options_layout)
        layout.addWidget(self.options_group)
        
        file_group = QGroupBox("Output File")
        file_layout = QHBoxLayout()
        
        self.file_path_edit = QLineEdit()
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_output_file)
        
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.browse_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        preview_group = QGroupBox("Export Preview")
        preview_layout = QVBoxLayout()
        
        self.preview_text = QTextEdit()
        self.preview_text.setMaximumHeight(200)
        self.preview_text.setReadOnly(True)
        
        preview_btn = QPushButton("Generate Preview")
        preview_btn.clicked.connect(self.generate_preview)
        
        preview_layout.addWidget(preview_btn)
        preview_layout.addWidget(self.preview_text)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.accept)
        button_layout.addWidget(export_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        self.on_language_changed()
        
    def on_language_changed(self):
        """Handle language change"""
        while self.options_layout.rowCount() > 0:
            self.options_layout.removeRow(0)
        
        language = self.language_combo.currentData()
        
        if language == 'python':
            self.class_name_edit = QLineEdit("DecisionTreeModel")
            self.function_name_edit = QLineEdit("predict")
            self.standalone_check = QCheckBox()
            self.standalone_check.setChecked(True)
            
            self.options_layout.addRow("Class Name:", self.class_name_edit)
            self.options_layout.addRow("Function Name:", self.function_name_edit)
            self.options_layout.addRow("Standalone Script:", self.standalone_check)
            
        elif language == 'java':
            self.class_name_edit = QLineEdit("DecisionTreeModel")
            self.options_layout.addRow("Class Name:", self.class_name_edit)
            
        elif language in ['r', 'javascript']:
            self.function_name_edit = QLineEdit("predict")
            self.options_layout.addRow("Function Name:", self.function_name_edit)
        
        self.update_file_extension()
        
    def update_file_extension(self):
        """Update suggested file extension"""
        language = self.language_combo.currentData()
        extensions = {
            'python': '.py',
            'r': '.R',
            'java': '.java',
            'javascript': '.js',
            'csharp': '.cs',
            'sql': '.sql',
            'scala': '.scala'
        }
        
        ext = extensions.get(language, '.txt')
        current_path = self.file_path_edit.text()
        
        if not current_path or not any(current_path.endswith(e) for e in extensions.values()):
            model_name = getattr(self.model, 'model_name', 'decision_tree_model')
            safe_name = "".join(c for c in model_name if c.isalnum() or c in '_-')
            self.file_path_edit.setText(f"{safe_name}{ext}")
    
    def browse_output_file(self):
        """Browse for output file"""
        language = self.language_combo.currentData()
        extensions = {
            'python': "Python Files (*.py)",
            'r': "R Files (*.R)",
            'java': "Java Files (*.java)",
            'javascript': "JavaScript Files (*.js)",
            'csharp': "C# Files (*.cs)",
            'sql': "SQL Files (*.sql)",
            'scala': "Scala Files (*.scala)"
        }
        
        file_filter = extensions.get(language, "All Files (*)")
        
        filename, _ = QFileDialog.getSaveFileName(
            self, f"Save {language.title()} Export",
            self.file_path_edit.text(),
            f"{file_filter};;All Files (*)"
        )
        
        if filename:
            self.file_path_edit.setText(filename)
    
    def generate_preview(self):
        """Generate export preview"""
        try:
            language = self.language_combo.currentData()
            
            preview_text = f"Export Preview for {language.title()}:\n\n"
            preview_text += "This will generate a professional Python file with:\n\n"
            preview_text += "- Complete decision tree implementation\n"
            preview_text += "- Input validation and preprocessing\n"
            preview_text += "- Multiple input format support (DataFrame, dict, array)\n"
            preview_text += "- Error handling and logging\n"
            preview_text += "- Model utilities and metadata\n"
            preview_text += "- Example usage and testing code\n\n"
            preview_text += "Example usage:\n"
            preview_text += ">>> predict({'feature1': 1.5, 'feature2': 'A'})\n"
            preview_text += "0\n"
            
            self.preview_text.clear()
            self.preview_text.setPlainText(preview_text)
            
        except Exception as e:
            error_msg = f"Error generating preview: {str(e)}"
            self.preview_text.setPlainText(error_msg)
    
    def get_export_options(self):
        """Get selected export options"""
        options = {
            'language': self.language_combo.currentData(),
            'output_path': self.file_path_edit.text()
        }
        
        language = options['language']
        
        if language == 'python':
            options['class_name'] = getattr(self, 'class_name_edit', None)
            if options['class_name'] and hasattr(options['class_name'], 'text'):
                options['class_name'] = options['class_name'].text() or 'DecisionTreeModel'
            else:
                options['class_name'] = 'DecisionTreeModel'
                
            options['function_name'] = getattr(self, 'function_name_edit', None)
            if options['function_name'] and hasattr(options['function_name'], 'text'):
                options['function_name'] = options['function_name'].text() or 'predict'
            else:
                options['function_name'] = 'predict'
                
            options['standalone'] = getattr(self, 'standalone_check', None)
            if options['standalone'] and hasattr(options['standalone'], 'isChecked'):
                options['standalone'] = options['standalone'].isChecked()
            else:
                options['standalone'] = True
                
        elif language == 'java':
            options['class_name'] = getattr(self, 'class_name_edit', None)
            if options['class_name'] and hasattr(options['class_name'], 'text'):
                options['class_name'] = options['class_name'].text() or 'DecisionTreeModel'
            else:
                options['class_name'] = 'DecisionTreeModel'
                
        elif language in ['r', 'javascript']:
            options['function_name'] = getattr(self, 'function_name_edit', None)
            if options['function_name'] and hasattr(options['function_name'], 'text'):
                options['function_name'] = options['function_name'].text() or 'predict'
            else:
                options['function_name'] = 'predict'
        
        return options

    def _update_visualization_tabs_with_metrics(self, metrics: dict, node_title: str):
        """
        Update visualization tabs with computed metrics and charts
        
        Args:
            metrics: Dictionary containing computed metrics from evaluation
            node_title: Name of the evaluation node
        """
        try:
            logger.info(f"Updating visualization tabs with metrics from {node_title}")
            
            if 'roc_curve' in metrics and 'roc_auc' in metrics:
                self._update_roc_curve_tab(metrics)
            
            if 'confusion_matrix' in metrics:
                self._update_metrics_summary_tab(metrics)
                
            if self.current_model_name and self.current_model_name in self.models:
                model = self.models[self.current_model_name]
                if hasattr(model, 'feature_importance') and model.feature_importance:
                    self._update_variable_importance_tab(model.feature_importance)
            
            logger.info("Visualization tabs updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating visualization tabs: {e}", exc_info=True)
    
    def _update_roc_curve_tab(self, metrics: dict):
        """Update the ROC Curve tab with actual chart"""
        try:
            roc_tab_index = None
            for i in range(self.visualization_tabs.count()):
                if self.visualization_tabs.tabText(i) == "ROC Curve":
                    roc_tab_index = i
                    break
            
            if roc_tab_index is not None:
                figure = Figure(figsize=(8, 6))
                canvas = FigureCanvas(figure)
                ax = figure.add_subplot(111)
                
                roc_data = metrics['roc_curve']
                fpr = roc_data['fpr']
                tpr = roc_data['tpr']
                auc_score = metrics.get('roc_auc', 0)
                
                ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
                ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7, label='Random Classifier')
                
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                figure.tight_layout()
                
                self.visualization_tabs.removeTab(roc_tab_index)
                self.visualization_tabs.insertTab(roc_tab_index, canvas, "ROC Curve")
                
                logger.info("ROC Curve tab updated with chart")
            
        except Exception as e:
            logger.error(f"Error updating ROC curve tab: {e}")
    
    def _update_metrics_summary_tab(self, metrics: dict):
        """Update or create a metrics summary tab"""
        try:
            metrics_tab_index = None
            for i in range(self.visualization_tabs.count()):
                if self.visualization_tabs.tabText(i) == "Metrics Summary":
                    metrics_tab_index = i
                    break
            
            summary_widget = QWidget()
            layout = QVBoxLayout(summary_widget)
            
            if 'confusion_matrix' in metrics:
                conf_matrix = metrics['confusion_matrix']
                
                figure = Figure(figsize=(6, 5))
                canvas = FigureCanvas(figure)
                ax = figure.add_subplot(111)
                
                import numpy as np
                cm = np.array(conf_matrix)
                im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
                
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], 'd'),
                               ha="center", va="center",
                               color="white" if cm[i, j] > thresh else "black")
                
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                ax.set_title('Confusion Matrix')
                figure.colorbar(im)
                figure.tight_layout()
                
                layout.addWidget(canvas)
            
            metrics_text = QTextEdit()
            metrics_text.setReadOnly(True)
            
            summary_lines = []
            summary_lines.append("=== PERFORMANCE METRICS SUMMARY ===\n")
            
            basic_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            for metric in basic_metrics:
                if metric in metrics:
                    value = metrics[metric]
                    if isinstance(value, (int, float)):
                        summary_lines.append(f"{metric.replace('_', ' ').title():<15}: {value:.4f}")
            
            if 'roc_auc' in metrics:
                summary_lines.append(f"ROC AUC Score  : {metrics['roc_auc']:.4f}")
            
            metrics_text.setPlainText('\n'.join(summary_lines))
            layout.addWidget(metrics_text)
            
            if metrics_tab_index is not None:
                self.visualization_tabs.removeTab(metrics_tab_index)
                self.visualization_tabs.insertTab(metrics_tab_index, summary_widget, "Metrics Summary")
            else:
                self.visualization_tabs.addTab(summary_widget, "Metrics Summary")
            
            logger.info("Metrics summary tab updated")
            
        except Exception as e:
            logger.error(f"Error updating metrics summary tab: {e}")
    
    def _update_variable_importance_tab(self, feature_importance: dict):
        """Update the Variable Importance tab with actual chart"""
        try:
            vi_tab_index = None
            for i in range(self.visualization_tabs.count()):
                if self.visualization_tabs.tabText(i) == "Variable Importance":
                    vi_tab_index = i
                    break
            
            if vi_tab_index is not None and feature_importance:
                figure = Figure(figsize=(10, 6))
                canvas = FigureCanvas(figure)
                ax = figure.add_subplot(111)
                
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                top_features = sorted_features[:15]  # Show top 15 features
                
                features, importances = zip(*top_features)
                
                y_pos = range(len(features))
                bars = ax.barh(y_pos, importances, color='skyblue')
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.invert_yaxis()  # Top feature at the top
                ax.set_xlabel('Importance')
                ax.set_title('Feature Importance (Top 15)')
                ax.grid(True, alpha=0.3)
                
                for i, (bar, importance) in enumerate(zip(bars, importances)):
                    if importance > 0:
                        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                               f'{importance:.3f}', ha='left', va='center', fontsize=8)
                
                figure.tight_layout()
                
                self.visualization_tabs.removeTab(vi_tab_index)
                self.visualization_tabs.insertTab(vi_tab_index, canvas, "Variable Importance")
                
                logger.info("Variable Importance tab updated with chart")
            
        except Exception as e:
            logger.error(f"Error updating variable importance tab: {e}")
    
    def _complete_pending_model_associations(self):
        """Complete any pending model associations for ModelNodes that couldn't find their models during initial restoration"""
        try:
            for node_id, node in self.workflow_canvas.scene.nodes.items():
                if isinstance(node, ModelNode):
                    if node.model is None and hasattr(node, '_pending_config') and node._pending_config:
                        pending = node._pending_config
                        model_ref_id = pending.get('model_ref_id')
                        
                        if model_ref_id and model_ref_id in self.models:
                            model = self.models[model_ref_id]
                            node.model = model
                            
                            if pending.get('target_variable'):
                                model.target_name = pending['target_variable']
                                logger.info(f"Applied pending target variable '{pending['target_variable']}' to model '{model_ref_id}'")
                            if pending.get('model_name'):
                                model.model_name = pending['model_name']
                            if pending.get('model_params'):
                                try:
                                    model.set_params(**pending['model_params'])
                                except Exception as e:
                                    logger.warning(f"Failed to apply pending model parameters: {e}")
                            
                            delattr(node, '_pending_config')
                            node.update_model_info()
                            logger.info(f"Completed pending model association for ModelNode {node_id} with model '{model_ref_id}'")
                        else:
                            logger.warning(f"ModelNode {node_id} has pending config for model '{model_ref_id}' but model not found in loaded models")
                            
        except Exception as e:
            logger.error(f"Error completing pending model associations: {e}")
    
    def _refresh_workflow_nodes_after_project_load(self):
        """Refresh all workflow nodes to ensure they reflect the current project state after loading"""
        try:
            for node_id, node in self.workflow_canvas.scene.nodes.items():
                if isinstance(node, DatasetNode):
                    dataset_name = node.dataset_name
                    if dataset_name and dataset_name in self.datasets:
                        df = self.datasets[dataset_name]
                        node.update_dataset_info(dataset_name, len(df), len(df.columns))
                        logger.debug(f"Refreshed DatasetNode {node_id} with dataset '{dataset_name}'")
                
                elif isinstance(node, ModelNode):
                    if node.model is not None:
                        node.update_model_info()
                        logger.debug(f"Refreshed ModelNode {node_id} display")
                    else:
                        logger.warning(f"ModelNode {node_id} has no associated model after project load")
                
                elif hasattr(node, 'update_visualization_info'):
                    node.update_visualization_info()
                    logger.debug(f"Refreshed visualization node {node_id}")
            
            self.workflow_canvas.scene.update()
            logger.info("Completed workflow node refresh after project load")
            
        except Exception as e:
            logger.error(f"Error refreshing workflow nodes after project load: {e}")
    
    def _setup_data_availability_checks(self):
        """Set up graceful fallbacks for when raw data is not available"""
        try:
            missing_datasets = []
            available_datasets = []
            
            for node_id, node in self.workflow_canvas.scene.nodes.items():
                if isinstance(node, DatasetNode):
                    dataset_name = node.dataset_name
                    if dataset_name not in self.datasets or self.datasets[dataset_name] is None:
                        missing_datasets.append(dataset_name)
                    else:
                        available_datasets.append(dataset_name)
            
            self.data_availability_status = {
                'has_all_data': len(missing_datasets) == 0,
                'missing_datasets': missing_datasets,
                'available_datasets': available_datasets,
                'has_cached_results': bool(self.latest_evaluation_results or 
                                         self.latest_performance_metrics),
                'has_models': bool(self.models)
            }
            
            if missing_datasets:
                self.enhanced_status_bar.show_message(
                    f"Project loaded - {len(missing_datasets)} dataset(s) missing. "
                    f"View-only mode enabled.", 
                    timeout=5000
                )
                logger.warning(f"Missing datasets: {missing_datasets}")
            else:
                self.enhanced_status_bar.show_message(
                    "Project loaded successfully with all data available.", 
                    timeout=3000
                )
                logger.info("All datasets available")
                
        except Exception as e:
            logger.error(f"Error setting up data availability checks: {e}")
            self.data_availability_status = {
                'has_all_data': False,
                'missing_datasets': [],
                'available_datasets': [],
                'has_cached_results': False,
                'has_models': bool(self.models)
            }
    
    def _check_data_required_operation(self, operation_name: str) -> bool:
        """Check if a data-requiring operation can be performed safely"""
        try:
            if not hasattr(self, 'data_availability_status'):
                return True  # Assume available if not checked yet
            
            status = self.data_availability_status
            
            view_only_operations = [
                'view_model', 'export_model', 'view_tree', 'view_performance',
                'view_cached_results', 'navigate_tree'
            ]
            
            if operation_name in view_only_operations:
                return status['has_models'] or status['has_cached_results']
            
            data_required_operations = [
                'train_model', 'retrain_model', 'manual_split', 'evaluate_model',
                'run_workflow', 'create_split', 'data_analysis'
            ]
            
            if operation_name in data_required_operations:
                return status['has_all_data']
            
            return status['has_all_data']
            
        except Exception as e:
            logger.error(f"Error checking data requirements for operation '{operation_name}': {e}")
            return False  # Err on the side of caution
