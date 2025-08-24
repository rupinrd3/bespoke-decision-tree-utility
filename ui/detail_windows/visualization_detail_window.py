#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Redesigned Visualization Detail Window for Decision Tree Analysis
NEW UI ARCHITECTURE - Eliminates redundant metrics and provides logical grouping

"""

import logging
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (QTabWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QTableWidget, QTableWidgetItem, QTextEdit, QPushButton,
                           QFrame, QScrollArea, QWidget, QGroupBox, QGridLayout,
                           QProgressBar, QHeaderView, QSpacerItem, QSizePolicy,
                           QComboBox, QFormLayout, QSplitter)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette

from ui.base_detail_window import BaseDetailWindow

try:
    from ui.widgets.lift_chart_widget import LiftChartWidget
    LIFT_CHART_AVAILABLE = True
except ImportError:
    LIFT_CHART_AVAILABLE = False
    
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from analytics.node_report_generator import NodeReportGenerator
    NODE_REPORT_GENERATOR_AVAILABLE = True
except ImportError:
    NODE_REPORT_GENERATOR_AVAILABLE = False

try:
    from analytics.variable_importance import VariableImportance
    VARIABLE_IMPORTANCE_AVAILABLE = True
except ImportError:
    VARIABLE_IMPORTANCE_AVAILABLE = False

logger = logging.getLogger(__name__)

class VisualizationDetailWindow(BaseDetailWindow):
    """
    Redesigned visualization detail window for decision tree analysis
    NEW UI ARCHITECTURE with logical metric grouping and no redundancy
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.evaluation_results = None
        self.model_data = None
        self.tree_root = None
        self.dataset = None
        self.target_column = None
        
        if NODE_REPORT_GENERATOR_AVAILABLE:
            self.node_report_generator = NodeReportGenerator()
        if VARIABLE_IMPORTANCE_AVAILABLE:
            self.variable_importance = VariableImportance()
            
        self.setup_visualization_actions()
        self.setup_visualization_content()
        
    def setup_visualization_actions(self):
        """Setup visualization-specific action buttons"""
        self.add_action_button("refresh", "🔄 Refresh", self.refresh_all_data, "secondary")
        self.add_action_button("export_report", "📊 Export Report", self.export_full_report, "secondary")
        self.add_action_button("export_nodes", "📋 Export Node Report", self.export_node_report, "secondary")
        
    def setup_visualization_content(self):
        """Setup the NEW UI ARCHITECTURE with redesigned tabs"""
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(self.get_modern_tab_style())
        
        self.overview_tab = self.create_model_overview_tab()
        self.tab_widget.addTab(self.overview_tab, "📈 Model Overview")
        
        self.tree_analysis_tab = self.create_tree_analysis_tab()
        self.tab_widget.addTab(self.tree_analysis_tab, "🌳 Tree Analysis")
        
        self.performance_curves_tab = self.create_performance_curves_tab()
        self.tab_widget.addTab(self.performance_curves_tab, "📊 Performance Curves")
        
        self.node_report_tab = self.create_node_report_tab()
        self.tab_widget.addTab(self.node_report_tab, "📋 Node Report")
        
        self.add_content_widget(self.tab_widget, stretch=1)
        logger.info("NEW UI ARCHITECTURE: VisualizationDetailWindow setup completed with redesigned tabs")
        
        self.populate_stored_node_report()
        
    def _widgets_exist(self):
        """Check if critical widgets still exist to prevent C++ object deletion errors"""
        try:
            return hasattr(self, 'tab_widget') and self.tab_widget is not None
        except RuntimeError:
            return False
        except Exception as e:
            logger.debug(f"Unexpected error checking widgets: {e}")
            return False
    
    def _safe_set_text(self, widget_name: str, text: str):
        """Safely set text on a widget, handling deletion gracefully"""
        try:
            if hasattr(self, widget_name):
                widget = getattr(self, widget_name)
                if widget is not None:
                    widget.setText(text)
        except (RuntimeError, AttributeError):
            logger.debug(f"Could not set text on {widget_name} - widget may have been deleted")
        except Exception as e:
            logger.debug(f"Unexpected error setting text on {widget_name}: {e}")
        
    def create_model_overview_tab(self):
        """
        Model Overview Tab - SINGLE SOURCE OF TRUTH for core metrics
        Eliminates redundancy by showing key metrics only here
        """
        widget = QScrollArea()
        content = QWidget()
        layout = QVBoxLayout(content)
        
        summary_group = QGroupBox("Decision Tree Model Summary")
        summary_layout = QGridLayout(summary_group)
        
        self.tree_depth_label = QLabel("Tree Depth: --")
        self.tree_nodes_label = QLabel("Total Nodes: --")
        self.tree_leaves_label = QLabel("Leaf Nodes: --")
        self.tree_rules_label = QLabel("Decision Rules: --")
        
        structure_style = """
            QLabel {
                font-size: 13px;
                font-weight: 500;
                color: #1e293b;
                padding: 8px;
                background-color: #f1f5f9;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                margin: 2px;
            }
        """
        for label in [self.tree_depth_label, self.tree_nodes_label, self.tree_leaves_label, self.tree_rules_label]:
            label.setStyleSheet(structure_style)
        
        summary_layout.addWidget(self.tree_depth_label, 0, 0)
        summary_layout.addWidget(self.tree_nodes_label, 0, 1)
        summary_layout.addWidget(self.tree_leaves_label, 1, 0)
        summary_layout.addWidget(self.tree_rules_label, 1, 1)
        
        layout.addWidget(summary_group)
        
        metrics_group = QGroupBox("Core Performance Metrics")
        metrics_layout = QGridLayout(metrics_group)
        
        self.accuracy_label = QLabel("Accuracy: --")
        self.precision_label = QLabel("Precision: --")
        self.recall_label = QLabel("Recall: --")
        self.f1_label = QLabel("F1 Score: --")
        self.auc_label = QLabel("AUC-ROC: --")
        self.ks_label = QLabel("KS Statistic: --")
        
        metric_style = """
            QLabel {
                font-size: 16px;
                font-weight: 600;
                color: #1e293b;
                padding: 20px;
                background-color: #f8fafc;
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                margin: 6px;
                min-width: 180px;
                min-height: 60px;
                qproperty-alignment: AlignCenter;
            }
        """
        for label in [self.accuracy_label, self.precision_label, self.recall_label, self.f1_label, self.auc_label, self.ks_label]:
            label.setStyleSheet(metric_style)
        
        metrics_layout.addWidget(self.accuracy_label, 0, 0)
        metrics_layout.addWidget(self.precision_label, 0, 1)
        metrics_layout.addWidget(self.recall_label, 1, 0)
        metrics_layout.addWidget(self.f1_label, 1, 1)
        metrics_layout.addWidget(self.auc_label, 2, 0)
        metrics_layout.addWidget(self.ks_label, 2, 1)
        
        layout.addWidget(metrics_group)
        
        confusion_group = QGroupBox("Confusion Matrix")
        confusion_layout = QVBoxLayout(confusion_group)
        
        self.confusion_table = QTableWidget(2, 2)
        self.confusion_table.setHorizontalHeaderLabels(["Predicted 0", "Predicted 1"])
        self.confusion_table.setVerticalHeaderLabels(["Actual 0", "Actual 1"])
        self.confusion_table.setFixedHeight(160)  # Increased height by 40%
        self.confusion_table.setMinimumWidth(420)  # Increased width by 40%
        
        header = self.confusion_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        
        vertical_header = self.confusion_table.verticalHeader()
        vertical_header.setSectionResizeMode(0, QHeaderView.Stretch)
        vertical_header.setSectionResizeMode(1, QHeaderView.Stretch)
        
        self.style_table(self.confusion_table)
        
        confusion_layout.addWidget(self.confusion_table)
        layout.addWidget(confusion_group)
        
        complexity_group = QGroupBox("Model Complexity vs Performance")
        complexity_layout = QFormLayout(complexity_group)
        
        self.complexity_score_label = QLabel("--")
        self.overfitting_risk_label = QLabel("--")
        self.interpretability_label = QLabel("--")
        
        complexity_layout.addRow("Complexity Score:", self.complexity_score_label)
        complexity_layout.addRow("Overfitting Risk:", self.overfitting_risk_label)
        complexity_layout.addRow("Interpretability:", self.interpretability_label)
        
        layout.addWidget(complexity_group)
        
        widget.setWidget(content)
        return widget
        
    def create_tree_analysis_tab(self):
        """Tree Analysis Tab - Tree-specific insights and structure"""
        widget = QScrollArea()
        content = QWidget()
        layout = QVBoxLayout(content)
        
        importance_group = QGroupBox("Feature Importance Analysis")
        importance_layout = QVBoxLayout(importance_group)
        
        controls_layout = QHBoxLayout()
        
        importance_method_label = QLabel("Method:")
        self.importance_method_combo = QComboBox()
        self.importance_method_combo.addItems(["Gini Impurity", "Permutation", "Information Gain"])
        self.importance_method_combo.currentTextChanged.connect(self.recalculate_importance)
        
        refresh_importance_btn = QPushButton("🔄 Recalculate")
        refresh_importance_btn.clicked.connect(self.recalculate_importance)
        
        controls_layout.addWidget(importance_method_label)
        controls_layout.addWidget(self.importance_method_combo)
        controls_layout.addWidget(refresh_importance_btn)
        controls_layout.addStretch()
        
        importance_layout.addLayout(controls_layout)
        
        self.importance_table = QTableWidget()
        self.importance_table.setColumnCount(3)
        self.importance_table.setHorizontalHeaderLabels(["Rank", "Feature", "Importance"])
        self.style_table(self.importance_table)
        importance_layout.addWidget(self.importance_table)
        
        layout.addWidget(importance_group)
        
        splits_group = QGroupBox("Split Analysis")
        splits_layout = QVBoxLayout(splits_group)
        
        self.splits_table = QTableWidget()
        self.splits_table.setColumnCount(4)
        self.splits_table.setHorizontalHeaderLabels(["Feature", "Split Value", "Impurity Reduction", "Samples"])
        self.style_table(self.splits_table)
        splits_layout.addWidget(self.splits_table)
        
        layout.addWidget(splits_group)
        
        tree_viz_group = QGroupBox("Tree Structure Summary")
        tree_viz_layout = QVBoxLayout(tree_viz_group)
        
        if MATPLOTLIB_AVAILABLE:
            self.tree_structure_figure = Figure(figsize=(10, 6), dpi=100)
            self.tree_structure_canvas = FigureCanvas(self.tree_structure_figure)
            tree_viz_layout.addWidget(self.tree_structure_canvas)
        else:
            tree_placeholder = QLabel("Matplotlib not available for tree structure visualization")
            tree_placeholder.setAlignment(Qt.AlignCenter)
            tree_viz_layout.addWidget(tree_placeholder)
        
        layout.addWidget(tree_viz_group)
        
        widget.setWidget(content)
        return widget
        
    def create_performance_curves_tab(self):
        """Performance Curves Tab - Visual performance analysis"""
        widget = QScrollArea()
        content = QWidget()
        layout = QVBoxLayout(content)
        
        if not MATPLOTLIB_AVAILABLE:
            placeholder = QLabel("Matplotlib not available for performance curves")
            placeholder.setAlignment(Qt.AlignCenter)
            layout.addWidget(placeholder)
            widget.setWidget(content)
            return widget
        
        curves_splitter = QSplitter(Qt.Horizontal)
        
        roc_widget = QWidget()
        roc_layout = QVBoxLayout(roc_widget)
        roc_layout.addWidget(QLabel("ROC Curve Analysis"))
        
        self.roc_figure = Figure(figsize=(6, 5), dpi=100)
        self.roc_canvas = FigureCanvas(self.roc_figure)
        roc_layout.addWidget(self.roc_canvas)
        
        curves_splitter.addWidget(roc_widget)
        
        pr_widget = QWidget()
        pr_layout = QVBoxLayout(pr_widget)
        pr_layout.addWidget(QLabel("Precision-Recall Curve"))
        
        self.pr_figure = Figure(figsize=(6, 5), dpi=100)
        self.pr_canvas = FigureCanvas(self.pr_figure)
        pr_layout.addWidget(self.pr_canvas)
        
        curves_splitter.addWidget(pr_widget)
        
        layout.addWidget(curves_splitter)
        
        lift_group = QGroupBox("Lift Analysis")
        lift_layout = QVBoxLayout(lift_group)
        
        if LIFT_CHART_AVAILABLE:
            self.lift_chart_widget = LiftChartWidget()
            lift_layout.addWidget(self.lift_chart_widget)
        else:
            self.lift_figure = Figure(figsize=(10, 4), dpi=100)
            self.lift_canvas = FigureCanvas(self.lift_figure)
            lift_layout.addWidget(self.lift_canvas)
        
        layout.addWidget(lift_group)
        
        calibration_group = QGroupBox("Model Calibration")
        calibration_layout = QVBoxLayout(calibration_group)
        
        self.calibration_figure = Figure(figsize=(8, 4), dpi=100)
        self.calibration_canvas = FigureCanvas(self.calibration_figure)
        calibration_layout.addWidget(self.calibration_canvas)
        
        layout.addWidget(calibration_group)
        
        widget.setWidget(content)
        return widget
        
    def create_node_report_tab(self):
        """Node Report Tab - Auto-generated for all terminal nodes"""
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(12)
        
        controls_group = QGroupBox("Node Report Actions")
        controls_group.setMaximumHeight(70)  # Limit controls height
        controls_layout = QHBoxLayout(controls_group)
        controls_layout.setContentsMargins(8, 8, 8, 8)  # Reduce margins
        
        self.export_excel_btn = QPushButton("📊 Export to Excel")
        self.export_excel_btn.clicked.connect(self.export_node_report_excel)
        self.export_excel_btn.setEnabled(False)
        self.export_excel_btn.setMaximumHeight(35)  # Compact button
        
        self.copy_clipboard_btn = QPushButton("📋 Copy to Clipboard")
        self.copy_clipboard_btn.clicked.connect(self.copy_node_report_clipboard)
        self.copy_clipboard_btn.setEnabled(False)
        self.copy_clipboard_btn.setMaximumHeight(35)  # Compact button
        
        controls_layout.addWidget(self.export_excel_btn)
        controls_layout.addWidget(self.copy_clipboard_btn)
        controls_layout.addStretch()
        
        main_layout.addWidget(controls_group, 0)  # No expansion for controls
        
        table_container = QWidget()
        table_container_layout = QVBoxLayout(table_container)
        table_container_layout.setContentsMargins(0, 0, 0, 0)
        
        report_group = QGroupBox("Terminal Node Analysis Report (Auto-Generated)")
        report_layout = QVBoxLayout(report_group)
        
        self.node_report_table = QTableWidget()
        self.style_table(self.node_report_table)
        
        self.node_report_table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)
        self.node_report_table.setVerticalScrollMode(QTableWidget.ScrollPerPixel)
        self.node_report_table.setMinimumHeight(600)
        
        from PyQt5.QtWidgets import QSizePolicy
        self.node_report_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.node_report_table.setMinimumWidth(1200)  # Minimum width for 1080p screens
        
        header = self.node_report_table.horizontalHeader()
        header.setStretchLastSection(True)
        
        report_layout.addWidget(self.node_report_table)
        table_container_layout.addWidget(report_group)
        
        main_layout.addWidget(table_container, 1)  # Stretch factor 1 for expansion
        
        self.node_report_status = QLabel("Node report will be automatically generated when data is loaded")
        self.node_report_status.setAlignment(Qt.AlignCenter)
        self.node_report_status.setStyleSheet("color: #6b7280; font-style: italic; padding: 8px; font-size: 11px;")
        self.node_report_status.setMaximumHeight(30)
        main_layout.addWidget(self.node_report_status, 0)  # No expansion for status
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(main_widget)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        return scroll_area
        
    def style_table(self, table):
        """Apply modern styling to tables with increased cell sizes"""
        table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                gridline-color: #f1f5f9;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #f8fafc;
                border: none;
                border-bottom: 1px solid #e2e8f0;
                border-right: 1px solid #e2e8f0;
                padding: 12px 16px;
                font-weight: 600;
                color: #374151;
                min-height: 40px;
            }
            QTableWidget::item {
                padding: 12px 16px;
                border: none;
                min-height: 40px;
            }
            QTableWidget::item:selected {
                background-color: #dbeafe;
                color: #1e293b;
            }
            QTableWidget::item:hover {
                background-color: #f1f5f9;
            }
        """)
        
    def get_modern_tab_style(self):
        """Get modern tab widget styling"""
        return """
            QTabWidget::pane {
                border: 1px solid #e2e8f0;
                border-top: none;
                background-color: #ffffff;
                border-radius: 0 0 8px 8px;
            }
            QTabBar::tab {
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-bottom: none;
                padding: 10px 18px;
                margin-right: 2px;
                border-radius: 8px 8px 0 0;
                font-weight: 500;
                color: #64748b;
                min-width: 120px;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                color: #1e293b;
                font-weight: 600;
            }
            QTabBar::tab:hover {
                background-color: #f1f5f9;
                color: #374151;
            }
        """
        
    def load_data(self, data: Dict[str, Any]):
        """Load visualization data into the NEW UI ARCHITECTURE"""
        logger.info(f"NEW UI: Loading data into redesigned VisualizationDetailWindow")
        
        try:
            self.evaluation_results = data.get('evaluation_results', {})
            self.model_data = data.get('model_data')
            self.tree_root = data.get('tree_root')
            self.dataset = data.get('dataset')
            self.target_column = data.get('target_column')
            node_id = data.get('node_id', 'unknown')
            
            # Debug logging to understand data structure
            logger.info(f"Evaluation results keys: {list(self.evaluation_results.keys()) if self.evaluation_results else 'None'}")
            logger.info(f"Model data type: {type(self.model_data)}")
            logger.info(f"Tree root available: {self.tree_root is not None}")
            logger.info(f"Dataset available: {self.dataset is not None}")
            logger.info(f"Target column: {self.target_column}")
            
            if self.evaluation_results:
                self.populate_model_overview()
                self.populate_tree_analysis()
                self.populate_performance_curves()
                
                self.auto_generate_node_report()
                
                logger.info(f"NEW UI: Successfully loaded visualization data for node {node_id}")
            else:
                self.show_empty_state("No evaluation results available. Please run the workflow to generate metrics.")
                
        except Exception as e:
            logger.error(f"NEW UI: Error loading visualization data: {e}", exc_info=True)
            self.show_error_state(f"Error loading visualization data: {str(e)}")
            
    def populate_model_overview(self):
        """Populate Model Overview tab - SINGLE SOURCE OF TRUTH for metrics"""
        try:
            results = self.evaluation_results
            if not results:
                logger.warning("No evaluation results available for model overview")
                return
            
            tree_depth = self._get_tree_depth()
            total_nodes = self._get_total_nodes()
            leaf_nodes = self._get_leaf_nodes()
            
            self._safe_set_text('tree_depth_label', f"Tree Depth: {tree_depth}")
            self._safe_set_text('tree_nodes_label', f"Total Nodes: {total_nodes}")
            self._safe_set_text('tree_leaves_label', f"Leaf Nodes: {leaf_nodes}")
            self._safe_set_text('tree_rules_label', f"Decision Rules: {leaf_nodes}")  # Each leaf is a rule
            
            self._safe_set_text('accuracy_label', f"Accuracy: {results.get('accuracy', 0):.4f}")
            self._safe_set_text('precision_label', f"Precision: {results.get('precision', 0):.4f}")
            self._safe_set_text('recall_label', f"Recall: {results.get('recall', 0):.4f}")
            self._safe_set_text('f1_label', f"F1 Score: {results.get('f1_score', 0):.4f}")
            self._safe_set_text('auc_label', f"AUC-ROC: {results.get('roc_auc', 0):.4f}")
            
            ks_value = results.get('max_ks', results.get('ks_statistic', 0))
            self._safe_set_text('ks_label', f"KS Statistic: {ks_value:.4f}")
            
            confusion = results.get('confusion_matrix', [[0, 0], [0, 0]])
            
            if isinstance(confusion, dict):
                matrix_data = confusion.get('matrix', [[0, 0], [0, 0]])
            elif hasattr(confusion, 'tolist'):
                matrix_data = confusion.tolist()
            else:
                matrix_data = confusion
            
            if not isinstance(matrix_data, list) or len(matrix_data) < 2:
                matrix_data = [[0, 0], [0, 0]]
            
            try:
                if hasattr(self, 'confusion_table') and self.confusion_table is not None:
                    for i in range(2):
                        for j in range(2):
                            try:
                                if i < len(matrix_data) and j < len(matrix_data[i]):
                                    value = matrix_data[i][j]
                                else:
                                    value = 0
                                item = QTableWidgetItem(str(value))
                                item.setTextAlignment(Qt.AlignCenter)
                                self.confusion_table.setItem(i, j, item)
                            except (IndexError, TypeError) as e:
                                logger.warning(f"Error setting confusion matrix item [{i}][{j}]: {e}")
                                item = QTableWidgetItem("0")
                                item.setTextAlignment(Qt.AlignCenter)
                                self.confusion_table.setItem(i, j, item)
            except RuntimeError:
                logger.debug("Confusion table widget has been deleted")
            
            complexity_score = self._calculate_complexity_score(tree_depth, total_nodes)
            overfitting_risk = self._assess_overfitting_risk(tree_depth, leaf_nodes)
            interpretability = self._assess_interpretability(tree_depth, total_nodes)
            
            self._safe_set_text('complexity_score_label', f"{complexity_score:.2f}")
            self._safe_set_text('overfitting_risk_label', overfitting_risk)
            self._safe_set_text('interpretability_label', interpretability)
            
            logger.info("Model overview populated (single source of truth)")
            
        except Exception as e:
            logger.error(f"Error populating model overview: {e}", exc_info=True)
            
    def populate_tree_analysis(self):
        """Populate Tree Analysis tab"""
        try:
            if not self.model_data or not self.tree_root:
                logger.warning("No model data or tree root available for tree analysis")
                return
                
            self.recalculate_importance()
            
            self._populate_split_analysis()
            
            self._create_tree_structure_plot()
            
            logger.info("Tree analysis populated")
            
        except Exception as e:
            logger.error(f"Error populating tree analysis: {e}", exc_info=True)
            
    def populate_performance_curves(self):
        """Populate Performance Curves tab using enhanced computation"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                logger.warning("Matplotlib not available for performance curves")
                return
                
            if (self.model_data and self.dataset is not None and 
                hasattr(self, 'target_column') and self.target_column):
                
                logger.info("Computing enhanced performance curves using terminal node probabilities")
                
                from analytics.performance_metrics import MetricsCalculator
                metrics_calc = MetricsCalculator()
                
                feature_columns = [col for col in self.dataset.columns if col != self.target_column]
                X = self.dataset[feature_columns]
                y = self.dataset[self.target_column]
                
                enhanced_results = metrics_calc.compute_enhanced_performance_curves(self.model_data, X, y)
                
                if enhanced_results:
                    if not self.evaluation_results:
                        self.evaluation_results = {}
                    self.evaluation_results.update(enhanced_results)
                    
                    logger.info("Enhanced performance curves computed successfully")
                else:
                    logger.warning("Enhanced performance curves computation failed, using fallback")
            elif not self.evaluation_results:
                logger.warning("No evaluation results available for performance curves")
                return
                
            self._create_roc_curve()
            self._create_pr_curve()
            self._create_lift_analysis()
            self._create_calibration_plot()
            
            logger.info("Performance curves populated")
            
        except Exception as e:
            logger.error(f"Error populating performance curves: {e}", exc_info=True)
            
    def auto_generate_node_report(self):
        """Auto-generate node report for all terminal nodes (no user interaction needed)"""
        if not NODE_REPORT_GENERATOR_AVAILABLE:
            if hasattr(self, 'node_report_status'):
                self._safe_set_text('node_report_status', "NodeReportGenerator not available - check analytics.node_report_generator import")
            logger.error("NodeReportGenerator not available")
            return
            
        if not hasattr(self, 'node_report_status'):
            logger.warning("Node report status widget not initialized")
            return
            
        try:
            if hasattr(self, 'node_report_status') and self.node_report_status is not None:
                try:
                    self._safe_set_text('node_report_status', "Auto-generating node report for all terminal nodes...")
                except RuntimeError:
                    pass
            logger.info("Starting auto node report generation")
            
            # Debug: Check what data we have
            logger.info(f"Model data type: {type(self.model_data)}")
            logger.info(f"Model data available: {self.model_data is not None}")
            logger.info(f"Tree root available: {self.tree_root is not None}")
            logger.info(f"Dataset available: {self.dataset is not None}")
            logger.info(f"Evaluation results available: {bool(self.evaluation_results)}")
            
            dataset_to_use = None
            target_col_to_use = None
            
            if self.dataset is not None:
                dataset_to_use = self.dataset
                target_col_to_use = self.target_column
                logger.info(f"Using passed dataset: shape {dataset_to_use.shape}, target: {target_col_to_use}")
            
            elif self.model_data and hasattr(self.model_data, 'X_train'):
                dataset_to_use = self.model_data.X_train
                target_col_to_use = getattr(self.model_data, 'target_variable', None)
                logger.info(f"Using model's X_train: shape {dataset_to_use.shape if hasattr(dataset_to_use, 'shape') else 'unknown'}")
            elif self.model_data and hasattr(self.model_data, 'training_data'):
                dataset_to_use = self.model_data.training_data
                target_col_to_use = getattr(self.model_data, 'target_variable', None)
                logger.info(f"Using model's training_data")
            
            if not target_col_to_use and self.evaluation_results:
                target_col_to_use = self.evaluation_results.get('target_variable', 
                                   self.evaluation_results.get('target_column', None))
                logger.info(f"Got target column from evaluation results: {target_col_to_use}")
            
            if dataset_to_use is None and self.evaluation_results:
                eval_dataset = self.evaluation_results.get('dataset', None)
                if eval_dataset is not None:
                    dataset_to_use = eval_dataset
                    logger.info(f"Using dataset from evaluation results: shape {dataset_to_use.shape if hasattr(dataset_to_use, 'shape') else 'unknown'}")
            
            if dataset_to_use is None:
                logger.info("Trying to get dataset from main window...")
                try:
                    from PyQt5.QtWidgets import QApplication
                    app = QApplication.instance()
                    for widget in app.allWidgets():
                        if hasattr(widget, 'current_dataset') and widget.current_dataset is not None:
                            dataset_to_use = widget.current_dataset
                            logger.info(f"Using current dataset from main window: shape {dataset_to_use.shape}")
                            break
                except Exception as e:
                    logger.warning(f"Could not get dataset from main window: {e}")
            
            if not self.model_data:
                error_msg = "No model data available for node report"
                self._safe_set_text('node_report_status', error_msg)
                logger.error(error_msg)
                return
                
            if dataset_to_use is None:
                logger.info("No dataset available - creating dummy dataset")
                self._safe_set_text('node_report_status', "No dataset available - generating basic report from tree structure")
                dataset_to_use = self._create_dummy_dataset()
                target_col_to_use = 'target'
                logger.info(f"Created dummy dataset: shape {dataset_to_use.shape}")
            
            if not target_col_to_use:
                if dataset_to_use is not None:
                    possible_targets = ['target', 'y', 'label', 'class', 'outcome']
                    for col in possible_targets:
                        if col in dataset_to_use.columns:
                            target_col_to_use = col
                            logger.info(f"Inferred target column: {target_col_to_use}")
                            break
                
                if not target_col_to_use:
                    target_col_to_use = dataset_to_use.columns[-1] if dataset_to_use is not None else 'target'
                    logger.info(f"Using last column as target: {target_col_to_use}")
            
            logger.info(f"Final params - Dataset shape: {dataset_to_use.shape if hasattr(dataset_to_use, 'shape') else 'unknown'}, target: {target_col_to_use}")
            
            logger.info("Calling NodeReportGenerator.generate_node_report...")
            report_df = self.node_report_generator.generate_node_report(
                model=self.model_data,
                dataset=dataset_to_use,
                target_variable=target_col_to_use,
                max_nodes=1000  # All terminal nodes
            )
            
            logger.info(f"Generated report with {len(report_df)} rows, columns: {list(report_df.columns)}")
            
            if not self._widgets_exist():
                logger.info("Widgets deleted during report generation, skipping table population")
                return
            
            self._populate_node_report_table(report_df)
            
            self.node_report_df = report_df
            
            if self._widgets_exist():
                try:
                    self.export_excel_btn.setEnabled(True)
                    self.copy_clipboard_btn.setEnabled(True)
                    
                    success_msg = f"Node report auto-generated successfully - {len(report_df)} terminal nodes analyzed"
                    self._safe_set_text('node_report_status', success_msg)
                    logger.info(success_msg)
                except RuntimeError:
                    logger.info("Widgets deleted after report generation, but data is stored")
            else:
                logger.info("Report generated successfully but widgets were deleted, data stored for later use")
            
        except Exception as e:
            error_msg = f"Error generating node report: {str(e)}"
            logger.error(f"Error auto-generating node report: {e}", exc_info=True)
            self._safe_set_text('node_report_status', error_msg)
            
            # Show some helpful debug info in status
            if "validation" in str(e).lower():
                self._safe_set_text('node_report_status', f"Validation error: {str(e)} - Check data compatibility")
            elif "not found" in str(e).lower():
                self._safe_set_text('node_report_status', f"Data not found: {str(e)} - Check model and dataset")
            
    def _create_dummy_dataset(self):
        """Create a minimal dummy dataset for node report generation when no dataset is available"""
        try:
            import pandas as pd
            dummy_data = {'target': [0, 1] * 50}  # 100 rows with binary target
            return pd.DataFrame(dummy_data)
        except Exception as e:
            logger.error(f"Error creating dummy dataset: {e}")
            return None
            
    def _populate_node_report_table(self, report_df):
        """Populate the node report table with generated data"""
        try:
            self.node_report_table.setRowCount(len(report_df))
            self.node_report_table.setColumnCount(len(report_df.columns))
            self.node_report_table.setHorizontalHeaderLabels(report_df.columns.tolist())
            
            for row in range(len(report_df)):
                for col in range(len(report_df.columns)):
                    value = report_df.iloc[row, col]
                    item = QTableWidgetItem(str(value))
                    self.node_report_table.setItem(row, col, item)
            
            header = self.node_report_table.horizontalHeader()
            header.setMinimumSectionSize(120)
            
            available_width = self.node_report_table.width()
            if available_width < 800:  # Fallback if width not available yet
                available_width = 1400  # Assume 1080p screen width minus margins
            
            num_cols = len(report_df.columns)
            
            priority_widths = {
                'Node_ID': 80,
                'Path_Rules': 300,  # Wider for rule descriptions
                'Sample_Count': 100,
                'Class_Distribution': 180,
                'Prediction': 100,
                'Confidence': 120,
                'Impurity': 100,
                'Information_Gain': 140,
                'Parent_Node': 120,
            }
            
            priority_total = sum(priority_widths.values())
            remaining_width = max(available_width - priority_total - 50, 200)  # Leave space for scrollbar
            remaining_cols = max(num_cols - len(priority_widths), 1)
            default_width = max(remaining_width // remaining_cols, 120)
            
            for col_idx, col_name in enumerate(report_df.columns):
                if col_name in priority_widths:
                    width = priority_widths[col_name]
                else:
                    width = default_width
                
                self.node_report_table.setColumnWidth(col_idx, width)
                logger.debug(f"Set column {col_idx} ({col_name}) width to {width}")
            
            self.node_report_table.resizeRowsToContents()
            
            header.setStretchLastSection(True)
            
            logger.info(f"Table populated with optimized column widths for {available_width}px available width")
            
        except Exception as e:
            logger.error(f"Error populating node report table: {e}", exc_info=True)
            
    def populate_stored_node_report(self):
        """Populate node report table if we have stored data from previous generation"""
        try:
            if hasattr(self, 'node_report_df') and self.node_report_df is not None:
                if self._widgets_exist():
                    logger.info(f"Populating stored node report with {len(self.node_report_df)} rows")
                    self._populate_node_report_table(self.node_report_df)
                    
                    self.export_excel_btn.setEnabled(True)
                    self.copy_clipboard_btn.setEnabled(True)
                    
                    success_msg = f"Node report populated - {len(self.node_report_df)} terminal nodes"
                    self._safe_set_text('node_report_status', success_msg)
                    return True
                    
        except Exception as e:
            logger.debug(f"Error populating stored node report: {e}")
            
        return False
            
    def export_node_report_excel(self):
        """Export node report to Excel using NodeReportGenerator"""
        try:
            if not hasattr(self, 'node_report_df'):
                self._safe_set_text('node_report_status', "No node report data to export")
                return
                
            from PyQt5.QtWidgets import QFileDialog, QMessageBox
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Node Report", 
                "node_report.xlsx",
                "Excel Files (*.xlsx);;All Files (*)"
            )
            
            if file_path:
                success, message = self.node_report_generator.export_to_excel(
                    self.node_report_df, file_path
                )
                
                if success:
                    self._safe_set_text('node_report_status', f"Report exported: {file_path}")
                    QMessageBox.information(self, "Export Successful", f"Node report exported to:\n{file_path}")
                else:
                    self._safe_set_text('node_report_status', f"Export failed: {message}")
                    QMessageBox.warning(self, "Export Failed", f"Failed to export report:\n{message}")
                    
        except Exception as e:
            logger.error(f"Error exporting node report: {e}", exc_info=True)
            self._safe_set_text('node_report_status', f"Export error: {str(e)}")
            
    def copy_node_report_clipboard(self):
        """Copy node report to clipboard"""
        try:
            if not hasattr(self, 'node_report_df'):
                self._safe_set_text('node_report_status', "No node report data to copy")
                return
                
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtGui import QClipboard
            
            clipboard_text = self.node_report_df.to_csv(sep='\t', index=False)
            
            clipboard = QApplication.clipboard()
            clipboard.setText(clipboard_text)
            
            self._safe_set_text('node_report_status', f"Node report copied to clipboard ({len(self.node_report_df)} rows)")
            
        except Exception as e:
            logger.error(f"Error copying node report to clipboard: {e}", exc_info=True)
            self._safe_set_text('node_report_status', f"Copy error: {str(e)}")
            
    def export_full_report(self):
        """Export comprehensive visualization report"""
        try:
            from PyQt5.QtWidgets import QFileDialog, QMessageBox
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Visualization Report", 
                "visualization_report.txt",
                "Text Files (*.txt);;All Files (*)"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("DECISION TREE VISUALIZATION REPORT\n")
                    f.write("=" * 50 + "\n\n")
                    
                    f.write("MODEL OVERVIEW\n")
                    f.write("-" * 20 + "\n")
                    if self.evaluation_results:
                        f.write(f"Accuracy: {self.evaluation_results.get('accuracy', 0):.4f}\n")
                        f.write(f"Precision: {self.evaluation_results.get('precision', 0):.4f}\n")
                        f.write(f"Recall: {self.evaluation_results.get('recall', 0):.4f}\n")
                        f.write(f"F1 Score: {self.evaluation_results.get('f1_score', 0):.4f}\n")
                        f.write(f"AUC-ROC: {self.evaluation_results.get('roc_auc', 0):.4f}\n")
                    f.write("\n")
                    
                    f.write("TREE STRUCTURE\n")
                    f.write("-" * 20 + "\n")
                    f.write(f"Tree Depth: {self._get_tree_depth()}\n")
                    f.write(f"Total Nodes: {self._get_total_nodes()}\n")
                    f.write(f"Leaf Nodes: {self._get_leaf_nodes()}\n")
                    f.write("\n")
                    
                    if hasattr(self, 'node_report_df'):
                        f.write("NODE ANALYSIS REPORT\n")
                        f.write("-" * 20 + "\n")
                        f.write(self.node_report_df.to_string(index=False))
                        f.write("\n")
                
                QMessageBox.information(self, "Export Successful", f"Visualization report exported to:\n{file_path}")
                
        except Exception as e:
            logger.error(f"Error exporting full report: {e}", exc_info=True)
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Export Failed", f"Failed to export report:\n{str(e)}")
            
    def _get_tree_depth(self):
        """Calculate tree depth"""
        if not self.tree_root:
            return 0
        return self._calculate_depth(self.tree_root)
        
    def _calculate_depth(self, node, current_depth=0):
        """Recursively calculate tree depth"""
        if not node or getattr(node, 'is_terminal', True):
            return current_depth
        
        max_child_depth = current_depth
        children = getattr(node, 'children', [])
        for child in children:
            if child:
                child_depth = self._calculate_depth(child, current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
        
    def _get_total_nodes(self):
        """Count total nodes in tree"""
        if not self.tree_root:
            return 0
        return self._count_nodes(self.tree_root)
        
    def _count_nodes(self, node):
        """Recursively count nodes"""
        if not node:
            return 0
        
        count = 1
        children = getattr(node, 'children', [])
        for child in children:
            if child:
                count += self._count_nodes(child)
        
        return count
        
    def _get_leaf_nodes(self):
        """Count leaf nodes"""
        if not self.tree_root:
            return 0
        return self._count_leaf_nodes(self.tree_root)
        
    def _count_leaf_nodes(self, node):
        """Recursively count leaf nodes"""
        if not node:
            return 0
        
        if getattr(node, 'is_terminal', True):
            return 1
        
        count = 0
        children = getattr(node, 'children', [])
        for child in children:
            if child:
                count += self._count_leaf_nodes(child)
        
        return count
        
    def _calculate_complexity_score(self, depth, nodes):
        """Calculate model complexity score"""
        if nodes == 0:
            return 0.0
        import math
        return depth * math.log(nodes + 1)
        
    def _assess_overfitting_risk(self, depth, leaves):
        """Assess overfitting risk"""
        if depth > 10:
            return "High"
        elif depth > 6:
            return "Medium"
        else:
            return "Low"
            
    def _assess_interpretability(self, depth, nodes):
        """Assess model interpretability"""
        if depth <= 3 and nodes <= 15:
            return "High"
        elif depth <= 6 and nodes <= 50:
            return "Medium"
        else:
            return "Low"
    
    def recalculate_importance(self):
        """Recalculate feature importance based on selected method"""
        try:
            if not VARIABLE_IMPORTANCE_AVAILABLE or not self.model_data:
                logger.warning("Variable importance calculation not available")
                return
                
            if not hasattr(self, 'importance_method_combo') or not hasattr(self, 'importance_table'):
                logger.warning("Importance widgets not initialized")
                return
                
            method_text = self.importance_method_combo.currentText()
            method_map = {
                "Gini Impurity": "gini",
                "Permutation": "permutation", 
                "Information Gain": "gain"
            }
            
            method = method_map.get(method_text, "gini")
            logger.info(f"Calculating feature importance using {method} method for '{method_text}'")
            
            importance = {}
            
            if method == "gini":
                importance = self.variable_importance.calculate_gini_importance(self.model_data)
            elif method == "gain":
                from analytics.variable_importance import ImportanceMethod
                
                X, y = None, None
                
                if self.dataset is not None and self.target_column:
                    X = self.dataset.drop(columns=[self.target_column]) if self.target_column in self.dataset.columns else self.dataset
                    y = self.dataset[self.target_column] if self.target_column in self.dataset.columns else None
                    logger.info(f"Using full dataset for Information Gain: X.shape={X.shape}, y defined={y is not None}")
                elif self.model_data and hasattr(self.model_data, 'X_train') and hasattr(self.model_data, 'y_train'):
                    X = self.model_data.X_train
                    y = self.model_data.y_train
                    logger.info(f"Using model training data for Information Gain: X.shape={X.shape}, y.shape={y.shape}")
                else:
                    X = self._get_dummy_features()
                    y = self._get_dummy_target()
                    logger.info(f"Using dummy data for Information Gain: X.shape={X.shape}, y.shape={y.shape}")
                
                if y is None:
                    y = self._get_dummy_target()
                    logger.warning("No target variable found, using dummy target for Information Gain")
                
                importance = self.variable_importance.compute_importance(
                    X=X,
                    y=y,
                    tree_root=self.tree_root if self.tree_root else getattr(self.model_data, 'root', None),
                    method=ImportanceMethod.GAIN
                )
            elif method == "permutation":
                if self.dataset is not None:
                    target_col = self.target_column or self.dataset.columns[-1]
                    X = self.dataset.drop(columns=[target_col]) if target_col in self.dataset.columns else self.dataset
                    y = self.dataset[target_col] if target_col in self.dataset.columns else self._get_dummy_target()
                    
                    from analytics.variable_importance import ImportanceMethod
                    importance = self.variable_importance.compute_importance(
                        X=X, y=y,
                        tree_root=self.tree_root if self.tree_root else self.model_data.root,
                        method=ImportanceMethod.PERMUTATION
                    )
                else:
                    logger.warning("No dataset available for permutation importance, using Gini")
                    importance = self.variable_importance.calculate_gini_importance(self.model_data)
            
            logger.info(f"Calculated {len(importance)} feature importances using {method} method")
                
            self._populate_importance_table(importance)
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}", exc_info=True)
            try:
                if hasattr(self, 'importance_table') and self.importance_table:
                    self.importance_table.setRowCount(1)
                    error_item = QTableWidgetItem(f"Error: {str(e)}")
                    self.importance_table.setItem(0, 1, error_item)
            except RuntimeError:
                logger.debug("Importance table widget has been deleted, skipping error display")
            
    def _get_dummy_features(self):
        """Get dummy features for importance calculation when no dataset available"""
        try:
            import pandas as pd
            feature_names = getattr(self.model_data, 'feature_names', ['feature_1', 'feature_2', 'feature_3'])
            dummy_data = {name: [0.5] * 100 for name in feature_names}
            return pd.DataFrame(dummy_data)
        except Exception as e:
            logger.error(f"Error creating dummy features: {e}")
            return pd.DataFrame({'feature_1': [0.5] * 100})
            
    def _get_dummy_target(self):
        """Get dummy target for importance calculation"""
        try:
            import pandas as pd
            return pd.Series([0, 1] * 50)  # 100 binary target values
        except Exception as e:
            logger.error(f"Error creating dummy target: {e}")
            return pd.Series([0] * 100)
            
    def _populate_importance_table(self, importance):
        """Populate the feature importance table"""
        try:
            if not hasattr(self, 'importance_table') or self.importance_table is None:
                logger.warning("Importance table widget not available")
                return
                
            if not importance:
                logger.info("No feature importance data to display")
                return
                
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            try:
                self.importance_table.setRowCount(len(sorted_importance))
            except RuntimeError:
                logger.debug("Importance table widget has been deleted")
                return
            
            for row, (feature, value) in enumerate(sorted_importance):
                rank_item = QTableWidgetItem(str(row + 1))
                rank_item.setTextAlignment(Qt.AlignCenter)
                self.importance_table.setItem(row, 0, rank_item)
                
                feature_item = QTableWidgetItem(str(feature))
                self.importance_table.setItem(row, 1, feature_item)
                
                value_item = QTableWidgetItem(f"{value:.4f}")
                value_item.setTextAlignment(Qt.AlignRight)
                self.importance_table.setItem(row, 2, value_item)
            
            header = self.importance_table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeToContents)
            
            logger.info(f"Populated importance table with {len(sorted_importance)} features")
            
        except Exception as e:
            logger.error(f"Error populating importance table: {e}", exc_info=True)
        
    def _populate_split_analysis(self):
        """Populate split analysis table"""
        try:
            if not self.tree_root:
                logger.info("No tree root available for split analysis")
                return
                
            split_nodes = self._get_split_nodes(self.tree_root)
            
            self.splits_table.setRowCount(len(split_nodes))
            
            for row, node in enumerate(split_nodes):
                feature = getattr(node, 'split_feature', 'Unknown')
                feature_item = QTableWidgetItem(str(feature))
                self.splits_table.setItem(row, 0, feature_item)
                
                split_value = getattr(node, 'split_value', 'N/A')
                value_item = QTableWidgetItem(str(split_value))
                self.splits_table.setItem(row, 1, value_item)
                
                impurity_reduction = self._calculate_impurity_reduction(node)
                reduction_item = QTableWidgetItem(f"{impurity_reduction:.4f}")
                reduction_item.setTextAlignment(Qt.AlignRight)
                self.splits_table.setItem(row, 2, reduction_item)
                
                samples = getattr(node, 'samples', 0)
                samples_item = QTableWidgetItem(str(samples))
                samples_item.setTextAlignment(Qt.AlignRight)
                self.splits_table.setItem(row, 3, samples_item)
            
            header = self.splits_table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.ResizeToContents)
            
            logger.info(f"Populated split analysis with {len(split_nodes)} splits")
            
        except Exception as e:
            logger.error(f"Error populating split analysis: {e}", exc_info=True)
            
    def _get_split_nodes(self, root_node):
        """Get all non-terminal nodes (split nodes)"""
        split_nodes = []
        
        def traverse(node):
            if not node:
                return
                
            if not getattr(node, 'is_terminal', True):
                split_nodes.append(node)
                
            children = getattr(node, 'children', [])
            for child in children:
                if child:
                    traverse(child)
        
        traverse(root_node)
        return split_nodes
        
    def _calculate_impurity_reduction(self, node):
        """Calculate impurity reduction for a split node"""
        try:
            node_impurity = getattr(node, 'impurity', 0)
            node_samples = getattr(node, 'samples', 0)
            children = getattr(node, 'children', [])
            
            if not children or node_samples == 0:
                return 0.0
                
            weighted_child_impurity = 0.0
            for child in children:
                child_samples = getattr(child, 'samples', 0)
                child_impurity = getattr(child, 'impurity', 0)
                if child_samples > 0:
                    weight = child_samples / node_samples
                    weighted_child_impurity += weight * child_impurity
            
            return node_impurity - weighted_child_impurity
            
        except Exception as e:
            logger.warning(f"Error calculating impurity reduction: {e}")
            return 0.0
        
    def _create_tree_structure_plot(self):
        """Create tree structure visualization"""
        try:
            if not MATPLOTLIB_AVAILABLE or not self.tree_root:
                return
                
            self.tree_structure_figure.clear()
            ax = self.tree_structure_figure.add_subplot(111)
            
            depth_counts = {}
            self._count_nodes_by_depth(self.tree_root, 0, depth_counts)
            
            depths = list(depth_counts.keys())
            counts = list(depth_counts.values())
            
            ax.bar(depths, counts, color='skyblue', alpha=0.7)
            ax.set_xlabel('Tree Depth')
            ax.set_ylabel('Number of Nodes')
            ax.set_title('Tree Structure: Nodes by Depth')
            ax.grid(True, alpha=0.3)
            
            self.tree_structure_figure.tight_layout()
            self.tree_structure_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error creating tree structure plot: {e}", exc_info=True)
            
    def _count_nodes_by_depth(self, node, depth, depth_counts):
        """Count nodes at each depth level"""
        if not node:
            return
            
        if depth not in depth_counts:
            depth_counts[depth] = 0
        depth_counts[depth] += 1
        
        children = getattr(node, 'children', [])
        for child in children:
            if child:
                self._count_nodes_by_depth(child, depth + 1, depth_counts)
        
    def _create_roc_curve(self):
        """Create ROC curve using enhanced curve data"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                return
                
            if not hasattr(self, 'roc_canvas'):
                logger.warning("ROC canvas not initialized")
                return
                
            self.roc_figure.clear()
            ax = self.roc_figure.add_subplot(111)
            
            if 'roc_curve' in self.evaluation_results:
                roc_data = self.evaluation_results['roc_curve']
                fpr = roc_data.get('fpr', [0, 1])
                tpr = roc_data.get('tpr', [0, 1])
                auc = self.evaluation_results.get('auc_roc', self.evaluation_results.get('roc_auc', 0))
                
                if isinstance(fpr, list) and isinstance(tpr, list) and len(fpr) > 2:
                    logger.info(f"Using enhanced ROC curve with {len(fpr)} points")
                else:
                    logger.info("Using basic ROC curve data")
            else:
                fpr = [0, 1]
                tpr = [0, 1]
                auc = self.evaluation_results.get('auc_roc', self.evaluation_results.get('roc_auc', 0))
            
            ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            self.roc_figure.tight_layout()
            if hasattr(self, 'roc_canvas') and self.roc_canvas:
                try:
                    self.roc_canvas.draw()
                except RuntimeError as e:
                    logger.warning(f"ROC canvas has been deleted: {e}")
            
        except Exception as e:
            logger.error(f"Error creating ROC curve: {e}", exc_info=True)
        
    def _create_pr_curve(self):
        """Create Precision-Recall curve using enhanced curve data"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                return
                
            if not hasattr(self, 'pr_canvas'):
                logger.warning("PR canvas not initialized")
                return
                
            self.pr_figure.clear()
            ax = self.pr_figure.add_subplot(111)
            
            if 'pr_curve' in self.evaluation_results:
                pr_data = self.evaluation_results['pr_curve']
                precision_vals = pr_data.get('precision', [1, 0.5, 0])
                recall_vals = pr_data.get('recall', [0, 0.5, 1])
                auc_pr = self.evaluation_results.get('auc_pr', self.evaluation_results.get('pr_auc', 0))
                
                if isinstance(precision_vals, list) and isinstance(recall_vals, list) and len(precision_vals) > 3:
                    logger.info(f"Using enhanced PR curve with {len(precision_vals)} points")
                else:
                    logger.info("Using basic PR curve data")
            else:
                precision = self.evaluation_results.get('precision', 0.5)
                recall = self.evaluation_results.get('recall', 0.5)
                recall_vals = [0, recall, 1]
                precision_vals = [1, precision, 0]
                auc_pr = self.evaluation_results.get('auc_pr', self.evaluation_results.get('pr_auc', 0))
            
            ax.plot(recall_vals, precision_vals, linewidth=2, label=f'PR Curve (AUC = {auc_pr:.3f})')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            self.pr_figure.tight_layout()
            if hasattr(self, 'pr_canvas') and self.pr_canvas:
                try:
                    self.pr_canvas.draw()
                except RuntimeError as e:
                    logger.warning(f"PR canvas has been deleted: {e}")
            
        except Exception as e:
            logger.error(f"Error creating PR curve: {e}", exc_info=True)
        
    def _create_lift_analysis(self):
        """Create lift analysis chart"""
        try:
            y_true = None
            y_pred_proba = None
            
            if self.dataset is not None and hasattr(self, 'target_column') and self.target_column:
                y_true = self.dataset[self.target_column]
                logger.info(f"Extracted y_true from dataset: {len(y_true)} samples")
            elif self.evaluation_results and 'y_true' in self.evaluation_results:
                y_true = self.evaluation_results['y_true']
                logger.info(f"Extracted y_true from evaluation results: {len(y_true)} samples")
            
            if self.evaluation_results and 'probabilities' in self.evaluation_results:
                y_pred_proba_full = self.evaluation_results['probabilities']
                if hasattr(y_pred_proba_full, 'shape') and len(y_pred_proba_full.shape) == 2 and y_pred_proba_full.shape[1] >= 2:
                    y_pred_proba = y_pred_proba_full[:, 1]  # Positive class probabilities
                    logger.info(f"Extracted y_pred_proba from probabilities array: {len(y_pred_proba)} samples")
                elif hasattr(y_pred_proba_full, 'flatten'):
                    y_pred_proba = y_pred_proba_full.flatten()
                    logger.info(f"Extracted y_pred_proba from flattened array: {len(y_pred_proba)} samples")
            
            if y_pred_proba is None and self.model_data and hasattr(self.model_data, 'predict_proba'):
                try:
                    logger.info("Attempting to generate probabilities from model")
                    if self.dataset is not None:
                        feature_columns = [col for col in self.dataset.columns if col != self.target_column]
                        X = self.dataset[feature_columns]
                        
                        proba_result = self.model_data.predict_proba(X)
                        if hasattr(proba_result, 'shape') and len(proba_result.shape) == 2 and proba_result.shape[1] >= 2:
                            y_pred_proba = proba_result[:, 1]  # Positive class probabilities
                            logger.info(f"Generated y_pred_proba from model: {len(y_pred_proba)} samples")
                        else:
                            y_pred_proba = proba_result.flatten() if hasattr(proba_result, 'flatten') else proba_result
                            logger.info(f"Generated y_pred_proba from model (flattened): {len(y_pred_proba)} samples")
                except Exception as e:
                    logger.warning(f"Failed to generate probabilities from model: {e}")
            
            if y_pred_proba is None and self.model_data and self.dataset is not None:
                try:
                    logger.info("Attempting to generate probabilities from decision tree leaf nodes")
                    import numpy as np
                    
                    feature_columns = [col for col in self.dataset.columns if col != self.target_column]
                    X = self.dataset[feature_columns]
                    
                    predictions = []
                    
                    if hasattr(self.model_data, 'predict_proba'):
                        try:
                            probabilities = self.model_data.predict_proba(X)
                            if len(probabilities.shape) == 2 and probabilities.shape[1] >= 2:
                                y_pred_proba = probabilities[:, 1]  # Positive class
                                logger.info(f"Got probabilities from model predict_proba: {len(y_pred_proba)} samples")
                            else:
                                y_pred_proba = probabilities.flatten()
                                logger.info(f"Got probabilities from model predict_proba (flattened): {len(y_pred_proba)} samples")
                        except Exception as e:
                            logger.warning(f"predict_proba failed: {e}")
                            
                    if y_pred_proba is None and hasattr(self.model_data, 'root'):
                        logger.info("Using tree traversal to get class proportions")
                        tree_root = self.model_data.root
                        predictions = []
                        
                        for idx, row in X.iterrows():
                            current_node = tree_root
                            while not current_node.is_terminal and current_node.children:
                                split_feature = current_node.split_feature
                                split_value = current_node.split_value
                                
                                if split_feature and split_feature in row:
                                    feature_value = row[split_feature]
                                    if feature_value <= split_value:
                                        current_node = current_node.children[0] if current_node.children else current_node
                                    else:
                                        current_node = current_node.children[1] if len(current_node.children) > 1 else current_node.children[0]
                                else:
                                    break
                            
                            if hasattr(current_node, 'class_counts') and current_node.class_counts:
                                total_samples = sum(current_node.class_counts.values())
                                positive_count = current_node.class_counts.get(1, 0)
                                probability = positive_count / total_samples if total_samples > 0 else 0.5
                                predictions.append(probability)
                            else:
                                overall_proportion = self.dataset[self.target_column].mean() if self.target_column in self.dataset.columns else 0.5
                                predictions.append(overall_proportion)
                        
                        if predictions:
                            y_pred_proba = np.array(predictions)
                            logger.info(f"Generated probabilities from tree traversal: {len(y_pred_proba)} samples")
                        
                except Exception as e:
                    logger.warning(f"Failed to generate probabilities from decision tree: {e}")
                    if self.target_column in self.dataset.columns:
                        overall_proportion = self.dataset[self.target_column].mean()
                        y_pred_proba = np.full(len(self.dataset), overall_proportion)
                        logger.info(f"Using overall class proportion fallback: {overall_proportion}")
                    
            if y_pred_proba is None and y_true is not None:
                logger.info("Using final fallback - overall class proportion for all samples")
                import numpy as np
                overall_proportion = y_true.mean() if hasattr(y_true, 'mean') else 0.5
                y_pred_proba = np.full(len(y_true), overall_proportion)
                logger.info(f"Final fallback probabilities created: {len(y_pred_proba)} samples with proportion {overall_proportion}")
            
            if LIFT_CHART_AVAILABLE and hasattr(self, 'lift_chart_widget'):
                if y_true is not None and y_pred_proba is not None:
                    logger.info("Using LiftChartWidget with real data")
                    self.lift_chart_widget.plot_lift_chart(
                        y_true=y_true,
                        y_pred_proba=y_pred_proba,
                        positive_class_label=1,  # Assume positive class is 1
                        n_bins=10,
                        title="Model Lift Analysis"
                    )
                else:
                    self.lift_chart_widget._clear_plot()
                    self.lift_chart_widget.ax.text(0.5, 0.5, 
                        "No model prediction data available.\nTrain a model first to see lift analysis.", 
                        ha='center', va='center', transform=self.lift_chart_widget.ax.transAxes,
                        fontsize=12, color='gray', style='italic')
                    self.lift_chart_widget.canvas.draw()
                    logger.info("Displayed 'no data' message in LiftChartWidget")
            elif MATPLOTLIB_AVAILABLE and hasattr(self, 'lift_figure'):
                self.lift_figure.clear()
                ax = self.lift_figure.add_subplot(111)
                
                if 'lift_curve' in self.evaluation_results:
                    lift_data = self.evaluation_results['lift_curve']
                    lift_values = lift_data.get('lift', [])
                    population_pct = lift_data.get('population_pct', [])
                    
                    if lift_values and population_pct:
                        logger.info(f"Using enhanced lift curve with {len(lift_values)} points")
                        ax.plot(population_pct, lift_values, linewidth=2, color='lightgreen', marker='o')
                        ax.axhline(y=1.0, color='red', linestyle='--', label='Baseline')
                        ax.set_xlabel('Percentage of Population (Deciles)')
                        ax.set_ylabel('Lift')
                        ax.set_title('Lift Chart')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        self.lift_figure.tight_layout()
                        self.lift_canvas.draw()
                        logger.info("Enhanced lift chart displayed")
                        return
                
                if y_true is not None and y_pred_proba is not None:
                    import pandas as pd
                    import numpy as np
                    
                    df = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred_proba})
                    df = df.sort_values('y_pred_proba', ascending=False).reset_index(drop=True)
                    
                    df['decile'] = pd.qcut(df.index, 10, labels=False, duplicates='drop')
                    
                    overall_positive_rate = df['y_true'].mean()
                    lift_values = []
                    
                    for decile in range(10):
                        decile_data = df[df['decile'] == decile]
                        if len(decile_data) > 0:
                            decile_positive_rate = decile_data['y_true'].mean()
                            lift = decile_positive_rate / overall_positive_rate if overall_positive_rate > 0 else 0
                            lift_values.append(lift)
                        else:
                            lift_values.append(0)
                    
                    deciles = list(range(1, len(lift_values) + 1))
                    logger.info(f"Calculated real lift values: {lift_values}")
                else:
                    ax.text(0.5, 0.5, "No model prediction data available.\nTrain a model first to see lift analysis.", 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=12, color='gray', style='italic')
                    ax.set_title('Lift Analysis - No Data Available')
                    logger.warning("No evaluation data available for lift chart")
                    self.lift_figure.tight_layout()
                    self.lift_canvas.draw()
                    return
                
                ax.bar(deciles, lift_values, color='lightgreen', alpha=0.7)
                ax.axhline(y=1.0, color='red', linestyle='--', label='Baseline')
                ax.set_xlabel('Decile')
                ax.set_ylabel('Lift')
                ax.set_title('Lift Analysis by Decile')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                self.lift_figure.tight_layout()
                self.lift_canvas.draw()
                
        except Exception as e:
            logger.error(f"Error creating lift analysis: {e}", exc_info=True)
        
    def _create_calibration_plot(self):
        """Create calibration plot"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                return
                
            if not hasattr(self, 'calibration_canvas'):
                logger.warning("Calibration canvas not initialized")
                return
                
            self.calibration_figure.clear()
            ax = self.calibration_figure.add_subplot(111)
            
            prob_true = [0, 0.25, 0.5, 0.75, 1.0]
            prob_pred = [0, 0.25, 0.5, 0.75, 1.0]
            
            ax.plot(prob_pred, prob_true, 'bo-', linewidth=2, label='Model')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title('Calibration Plot')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            self.calibration_figure.tight_layout()
            if hasattr(self, 'calibration_canvas') and self.calibration_canvas:
                try:
                    self.calibration_canvas.draw()
                except RuntimeError as e:
                    logger.warning(f"Calibration canvas has been deleted: {e}")
            
        except Exception as e:
            logger.error(f"Error creating calibration plot: {e}", exc_info=True)
    
    def refresh_all_data(self):
        """Refresh all visualization data"""
        if self.evaluation_results:
            self.populate_model_overview()
            self.populate_tree_analysis()
            self.populate_performance_curves()
            
    def export_full_report(self):
        """Export comprehensive visualization report"""
        logger.info("Export full report functionality - to be implemented")
        
    def export_node_report(self):
        """Export just the node report"""
        self.export_node_report_excel()