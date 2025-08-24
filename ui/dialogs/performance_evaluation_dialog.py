#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance Evaluation Dialog for Bespoke Utility
Comprehensive interface for model performance evaluation
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont, QPixmap, QIcon
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QTabWidget, QWidget, QListWidget,
    QSplitter, QProgressBar, QMessageBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QFrame, QSizePolicy, QButtonGroup, QRadioButton
)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from analytics.performance_metrics import MetricsCalculator, MetricsVisualizer

logger = logging.getLogger(__name__)


class PerformanceEvaluationWorker(QThread):
    """Worker thread for computing performance metrics"""
    
    progressUpdate = pyqtSignal(int, str)
    resultReady = pyqtSignal(dict)
    errorOccurred = pyqtSignal(str)
    
    def __init__(self, X, y, tree_model, positive_class=None):
        super().__init__()
        self.X = X
        self.y = y
        self.tree_model = tree_model
        self.positive_class = positive_class
        
    def run(self):
        """Run performance evaluation"""
        try:
            self.progressUpdate.emit(10, "Checking for stored metrics...")
            
            if hasattr(self.tree_model, 'root'):
                tree_root = self.tree_model.root
                model_obj = self.tree_model
            else:
                tree_root = self.tree_model
                model_obj = None
            
            if model_obj and hasattr(model_obj, 'metrics') and model_obj.metrics:
                self.progressUpdate.emit(50, "Using stored performance metrics...")
                metrics = model_obj.metrics.copy()
                logger.info("Using stored metrics from model")
            else:
                self.progressUpdate.emit(30, "Computing fresh performance metrics...")
                calculator = MetricsCalculator()
                metrics = calculator.compute_metrics(
                    self.X, self.y, tree_root,
                    positive_class=self.positive_class
                )
                logger.info("Computed fresh metrics")
            
            self.progressUpdate.emit(70, "Generating predictions for curves...")
            
            try:
                y_pred = tree_root.predict(self.X)
                y_proba = tree_root.predict_proba(self.X)
                
                metrics['predictions'] = y_pred
                metrics['probabilities'] = y_proba
                
            except Exception as e:
                logger.warning(f"Could not get probabilities: {e}")
                metrics['predictions'] = self.tree_model.root.predict(self.X)
                metrics['probabilities'] = None
                
            self.progressUpdate.emit(100, "Complete!")
            
            self.resultReady.emit(metrics)
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}", exc_info=True)
            self.errorOccurred.emit(str(e))


class PerformanceEvaluationDialog(QDialog):
    """Dialog for comprehensive performance evaluation"""
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, tree_model, parent=None):
        super().__init__(parent)
        
        self.X = X
        self.y = y
        self.tree_model = tree_model
        
        self.metrics_results = {}
        self.visualizer = MetricsVisualizer()
        
        self.unique_classes = sorted(y.unique())
        self.is_binary = len(self.unique_classes) == 2
        
        self.setWindowTitle("Performance Evaluation")
        self.setModal(True)
        self.resize(1200, 800)
        
        self.setupUI()
        
    def setupUI(self):
        """Setup the simplified UI focused on essential metrics"""
        layout = QVBoxLayout()
        
        header_label = QLabel("📊 Model Performance")
        header_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header_label.setStyleSheet("color: #1e293b; padding: 10px;")
        layout.addWidget(header_label)
        
        self.main_content = self.createSimplifiedContent()
        layout.addWidget(self.main_content)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #e2e8f0;
                border-radius: 6px;
                text-align: center;
                font-weight: 600;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        self.progress_label.setStyleSheet("color: #6b7280; font-size: 12px; padding: 5px;")
        layout.addWidget(self.progress_label)
        
        button_layout = QHBoxLayout()
        
        self.evaluate_button = QPushButton("🔍 Evaluate Model")
        self.evaluate_button.clicked.connect(self.evaluateModel)
        self.evaluate_button.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #f1f5f9;
                color: #475569;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 600;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #e2e8f0;
            }
        """)
        
        button_layout.addWidget(self.evaluate_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def createSimplifiedContent(self) -> QWidget:
        """Create simplified content focused on essential metrics only"""
        widget = QScrollArea()
        content = QWidget()
        layout = QVBoxLayout(content)
        
        self.setStyleSheet("""
            QDialog {
                background-color: #f8fafc;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                font-weight: 600;
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #1e293b;
            }
            QLabel {
                color: #374151;
                font-size: 13px;
            }
        """)
        
        summary_group = QGroupBox("📋 Dataset Summary")
        summary_layout = QGridLayout()
        
        dataset_info = f"• {len(self.X):,} samples"
        features_info = f"• {len(self.X.columns)} features"  
        classes_info = f"• {len(self.unique_classes)} classes: {', '.join(map(str, self.unique_classes))}"
        
        summary_layout.addWidget(QLabel(dataset_info), 0, 0)
        summary_layout.addWidget(QLabel(features_info), 0, 1)
        summary_layout.addWidget(QLabel(classes_info), 1, 0, 1, 2)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        metrics_group = QGroupBox("🎯 Key Performance Metrics")
        metrics_layout = QGridLayout()
        
        self.accuracy_label = QLabel("--")
        self.accuracy_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.accuracy_label.setStyleSheet("color: #059669; background-color: #ecfdf5; padding: 8px; border-radius: 6px;")
        
        accuracy_desc = QLabel("Overall Accuracy")
        accuracy_desc.setStyleSheet("font-weight: 600; color: #374151;")
        
        metrics_layout.addWidget(accuracy_desc, 0, 0)
        metrics_layout.addWidget(self.accuracy_label, 0, 1)
        
        if self.is_binary:
            self.precision_label = QLabel("--")
            self.precision_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
            self.precision_label.setStyleSheet("color: #2563eb; background-color: #eff6ff; padding: 6px; border-radius: 4px;")
            
            precision_desc = QLabel("Precision")
            precision_desc.setStyleSheet("font-weight: 600; color: #374151;")
            
            metrics_layout.addWidget(precision_desc, 1, 0)
            metrics_layout.addWidget(self.precision_label, 1, 1)
            
            self.recall_label = QLabel("--")
            self.recall_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
            self.recall_label.setStyleSheet("color: #7c3aed; background-color: #f3e8ff; padding: 6px; border-radius: 4px;")
            
            recall_desc = QLabel("Recall")
            recall_desc.setStyleSheet("font-weight: 600; color: #374151;")
            
            metrics_layout.addWidget(recall_desc, 1, 2)
            metrics_layout.addWidget(self.recall_label, 1, 3)
            
            self.f1_label = QLabel("--")
            self.f1_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
            self.f1_label.setStyleSheet("color: #dc2626; background-color: #fef2f2; padding: 6px; border-radius: 4px;")
            
            f1_desc = QLabel("F1-Score")
            f1_desc.setStyleSheet("font-weight: 600; color: #374151;")
            
            metrics_layout.addWidget(f1_desc, 2, 0)
            metrics_layout.addWidget(self.f1_label, 2, 1)
        
        self.status_label = QLabel("Click 'Evaluate Model' to calculate performance metrics")
        self.status_label.setStyleSheet("color: #6b7280; font-style: italic; padding: 10px; text-align: center;")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        metrics_layout.addWidget(self.status_label, 3, 0, 1, 4)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        insight_group = QGroupBox("💡 Quick Insights")
        insight_layout = QVBoxLayout()
        
        self.insight_text = QLabel("Insights will appear here after evaluation...")
        self.insight_text.setWordWrap(True)
        self.insight_text.setStyleSheet("""
            background-color: #fef3c7;
            border: 1px solid #fbbf24;
            border-radius: 6px;
            padding: 12px;
            color: #92400e;
        """)
        
        insight_layout.addWidget(self.insight_text)
        insight_group.setLayout(insight_layout)
        layout.addWidget(insight_group)
        
        widget.setWidget(content)
        return widget
        
    def createConfigurationPanel(self) -> QWidget:
        """Create the configuration panel"""
        config_group = QGroupBox("Evaluation Configuration")
        layout = QHBoxLayout()
        
        if self.is_binary:
            pos_class_layout = QFormLayout()
            
            self.positive_class_combo = QComboBox()
            self.positive_class_combo.addItems([str(cls) for cls in self.unique_classes])
            self.positive_class_combo.setCurrentIndex(1)  # Default to second class
            
            pos_class_layout.addRow("Positive Class:", self.positive_class_combo)
            layout.addLayout(pos_class_layout)
            
        weights_layout = QFormLayout()
        
        self.use_weights_checkbox = QCheckBox("Use sample weights")
        self.use_weights_checkbox.setToolTip("Use class-balanced sample weights")
        weights_layout.addRow("", self.use_weights_checkbox)
        
        layout.addLayout(weights_layout)
        
        layout.addStretch()
        
        config_group.setLayout(layout)
        return config_group
        
    def createOverviewTab(self) -> QWidget:
        """Create the overview tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        metrics_group = QGroupBox("Key Performance Metrics")
        metrics_layout = QGridLayout()
        
        self.accuracy_label = QLabel("-")
        self.accuracy_label.setFont(QFont("Arial", 12, QFont.Bold))
        metrics_layout.addWidget(QLabel("Accuracy:"), 0, 0)
        metrics_layout.addWidget(self.accuracy_label, 0, 1)
        
        if self.is_binary:
            self.precision_label = QLabel("-")
            self.precision_label.setFont(QFont("Arial", 12, QFont.Bold))
            metrics_layout.addWidget(QLabel("Precision:"), 0, 2)
            metrics_layout.addWidget(self.precision_label, 0, 3)
            
            self.recall_label = QLabel("-")
            self.recall_label.setFont(QFont("Arial", 12, QFont.Bold))
            metrics_layout.addWidget(QLabel("Recall:"), 1, 0)
            metrics_layout.addWidget(self.recall_label, 1, 1)
            
            self.f1_label = QLabel("-")
            self.f1_label.setFont(QFont("Arial", 12, QFont.Bold))
            metrics_layout.addWidget(QLabel("F1-Score:"), 1, 2)
            metrics_layout.addWidget(self.f1_label, 1, 3)
            
            self.auc_label = QLabel("-")
            self.auc_label.setFont(QFont("Arial", 12, QFont.Bold))
            metrics_layout.addWidget(QLabel("AUC-ROC:"), 2, 0)
            metrics_layout.addWidget(self.auc_label, 2, 1)
            
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        viz_group = QGroupBox("Performance Summary")
        viz_layout = QVBoxLayout()
        
        self.summary_figure = Figure(figsize=(10, 4))
        self.summary_canvas = FigureCanvas(self.summary_figure)
        viz_layout.addWidget(self.summary_canvas)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        widget.setLayout(layout)
        return widget
        
    def createConfusionMatrixTab(self) -> QWidget:
        """Create the confusion matrix tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        controls_layout = QHBoxLayout()
        
        self.normalize_checkbox = QCheckBox("Normalize")
        self.normalize_checkbox.stateChanged.connect(self.updateConfusionMatrix)
        controls_layout.addWidget(self.normalize_checkbox)
        
        controls_layout.addStretch()
        
        refresh_cm_button = QPushButton("Refresh")
        refresh_cm_button.clicked.connect(self.updateConfusionMatrix)
        controls_layout.addWidget(refresh_cm_button)
        
        layout.addLayout(controls_layout)
        
        self.cm_figure = Figure(figsize=(8, 6))
        self.cm_canvas = FigureCanvas(self.cm_figure)
        layout.addWidget(self.cm_canvas)
        
        stats_group = QGroupBox("Confusion Matrix Statistics")
        stats_layout = QFormLayout()
        
        if self.is_binary:
            self.tp_label = QLabel("-")
            self.tn_label = QLabel("-")
            self.fp_label = QLabel("-")
            self.fn_label = QLabel("-")
            
            stats_layout.addRow("True Positives:", self.tp_label)
            stats_layout.addRow("True Negatives:", self.tn_label)
            stats_layout.addRow("False Positives:", self.fp_label)
            stats_layout.addRow("False Negatives:", self.fn_label)
            
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        widget.setLayout(layout)
        return widget
        
    def createCurvesTab(self) -> QWidget:
        """Create the ROC/PR curves tab (binary only)"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        splitter = QSplitter(Qt.Horizontal)
        
        roc_widget = QWidget()
        roc_layout = QVBoxLayout()
        roc_layout.addWidget(QLabel("ROC Curve"))
        
        self.roc_figure = Figure(figsize=(6, 5))
        self.roc_canvas = FigureCanvas(self.roc_figure)
        roc_layout.addWidget(self.roc_canvas)
        
        roc_widget.setLayout(roc_layout)
        splitter.addWidget(roc_widget)
        
        pr_widget = QWidget()
        pr_layout = QVBoxLayout()
        pr_layout.addWidget(QLabel("Precision-Recall Curve"))
        
        self.pr_figure = Figure(figsize=(6, 5))
        self.pr_canvas = FigureCanvas(self.pr_figure)
        pr_layout.addWidget(self.pr_canvas)
        
        pr_widget.setLayout(pr_layout)
        splitter.addWidget(pr_widget)
        
        layout.addWidget(splitter)
        
        widget.setLayout(layout)
        return widget
        
    def createClassificationReportTab(self) -> QWidget:
        """Create the classification report tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.report_figure = Figure(figsize=(10, 6))
        self.report_canvas = FigureCanvas(self.report_figure)
        layout.addWidget(self.report_canvas)
        
        report_group = QGroupBox("Detailed Classification Report")
        report_layout = QVBoxLayout()
        
        self.report_text = QTextEdit()
        self.report_text.setFont(QFont("Courier", 10))
        self.report_text.setReadOnly(True)
        report_layout.addWidget(self.report_text)
        
        report_group.setLayout(report_layout)
        layout.addWidget(report_group)
        
        widget.setLayout(layout)
        return widget
        
    def createDetailedMetricsTab(self) -> QWidget:
        """Create the detailed metrics tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(['Metric', 'Value'])
        
        header = self.metrics_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        
        self.metrics_table.setAlternatingRowColors(True)
        layout.addWidget(self.metrics_table)
        
        widget.setLayout(layout)
        return widget
        
    def evaluateModel(self):
        """Evaluate the model performance"""
        positive_class = None
        if self.is_binary and hasattr(self, 'positive_class_combo'):
            positive_class = self.positive_class_combo.currentText()
            if positive_class.isdigit():
                positive_class = int(positive_class)
            elif positive_class.replace('.', '').isdigit():
                positive_class = float(positive_class)
                
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.evaluate_button.setEnabled(False)
        
        self.worker = PerformanceEvaluationWorker(
            self.X, self.y, self.tree_model, positive_class
        )
        self.worker.progressUpdate.connect(self.updateProgress)
        self.worker.resultReady.connect(self.onEvaluationReady)
        self.worker.errorOccurred.connect(self.onEvaluationError)
        self.worker.start()
        
    def updateProgress(self, progress: int, message: str):
        """Update progress bar and message"""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(message)
        
    def onEvaluationReady(self, metrics: Dict[str, Any]):
        """Handle evaluation completion for simplified interface"""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.evaluate_button.setEnabled(True)
        
        self.metrics_results = metrics
        
        self.updateSimplifiedDisplay()
        
    def updateSimplifiedDisplay(self):
        """Update the simplified interface with evaluation results"""
        try:
            metrics = self.metrics_results
            
            accuracy = metrics.get('accuracy', 0)
            self.accuracy_label.setText(f"{accuracy:.1%}")
            
            if self.is_binary:
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1 = metrics.get('f1_score', 0)
                
                self.precision_label.setText(f"{precision:.1%}")
                self.recall_label.setText(f"{recall:.1%}")
                self.f1_label.setText(f"{f1:.1%}")
            
            num_samples = len(self.X)
            self.status_label.setText(f"✅ Evaluation completed on {num_samples:,} samples")
            
            insights = self.generateSimpleInsights(metrics)
            self.insight_text.setText(insights)
            
        except Exception as e:
            logger.error(f"Error updating simplified display: {e}")
            self.status_label.setText(f"❌ Error displaying results: {str(e)}")
            
    def generateSimpleInsights(self, metrics: Dict[str, Any]) -> str:
        """Generate simple, actionable insights from metrics"""
        insights = []
        
        accuracy = metrics.get('accuracy', 0)
        
        if accuracy >= 0.9:
            insights.append("🎯 Excellent accuracy! Your model performs very well.")
        elif accuracy >= 0.8:
            insights.append("👍 Good accuracy. Consider if further improvement is needed.")
        elif accuracy >= 0.7:
            insights.append("⚠️ Fair accuracy. Model may benefit from more training data or feature engineering.")
        else:
            insights.append("🔴 Low accuracy. Model needs significant improvement.")
        
        if self.is_binary:
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            
            if precision > recall + 0.1:
                insights.append("🎯 Higher precision than recall - good at avoiding false positives.")
            elif recall > precision + 0.1:
                insights.append("🔍 Higher recall than precision - good at finding all positive cases.")
            else:
                insights.append("⚖️ Balanced precision and recall.")
        
        num_samples = len(self.X)
        if num_samples < 1000:
            insights.append("📊 Small dataset - consider gathering more data for better reliability.")
        elif num_samples > 50000:
            insights.append("📈 Large dataset provides robust results.")
            
        return " ".join(insights)
        
    def onEvaluationError(self, error_message: str):
        """Handle evaluation error"""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.evaluate_button.setEnabled(True)
        
        QMessageBox.critical(self, "Evaluation Error", 
                           f"Error evaluating model: {error_message}")
        
    def updateOverviewTab(self):
        """Update the overview tab"""
        metrics = self.metrics_results
        
        self.accuracy_label.setText(f"{metrics.get('accuracy', 0):.3f}")
        
        if self.is_binary:
            self.precision_label.setText(f"{metrics.get('precision', 0):.3f}")
            self.recall_label.setText(f"{metrics.get('recall', 0):.3f}")
            self.f1_label.setText(f"{metrics.get('f1_score', 0):.3f}")
            self.auc_label.setText(f"{metrics.get('roc_auc', 0):.3f}")
            
        self.summary_figure.clear()
        
        try:
            fig = self.visualizer.plot_metrics_summary(
                metrics, title="Performance Summary", figsize=(10, 4)
            )
            
            ax_source = fig.get_axes()[0]
            ax_target = self.summary_figure.add_subplot(111)
            
            for patch in ax_source.patches:
                ax_target.add_patch(patch)
                
            ax_target.set_xlim(ax_source.get_xlim())
            ax_target.set_ylim(ax_source.get_ylim())
            ax_target.set_xlabel(ax_source.get_xlabel())
            ax_target.set_ylabel(ax_source.get_ylabel())
            ax_target.set_title(ax_source.get_title())
            ax_target.grid(True, alpha=0.3)
            
            plt.close(fig)  # Close the temporary figure
            
        except Exception as e:
            logger.warning(f"Could not create summary plot: {e}")
            
        self.summary_figure.tight_layout()
        self.summary_canvas.draw()
        
    def updateConfusionMatrix(self):
        """Update the confusion matrix"""
        if not self.metrics_results:
            return
            
        metrics = self.metrics_results
        cm_data = metrics.get('confusion_matrix')
        
        if not cm_data:
            return
            
        self.cm_figure.clear()
        
        try:
            normalize = self.normalize_checkbox.isChecked()
            
            fig = self.visualizer.plot_confusion_matrix(
                cm_data, title="Confusion Matrix", 
                normalize=normalize, figsize=(6, 5)
            )
            
            ax_source = fig.get_axes()[0]
            ax_target = self.cm_figure.add_subplot(111)
            
            matrix = np.array(cm_data['normalized'] if normalize else cm_data['matrix'])
            classes = cm_data['classes']
            
            im = ax_target.imshow(matrix, interpolation='nearest', cmap='Blues')
            self.cm_figure.colorbar(im, ax=ax_target)
            
            ax_target.set(xticks=np.arange(matrix.shape[1]),
                         yticks=np.arange(matrix.shape[0]),
                         xticklabels=classes, yticklabels=classes,
                         title="Confusion Matrix",
                         ylabel='True label',
                         xlabel='Predicted label')
                         
            fmt = '.2f' if normalize else 'd'
            thresh = matrix.max() / 2.
            
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax_target.text(j, i, format(matrix[i, j], fmt),
                                  ha="center", va="center",
                                  color="white" if matrix[i, j] > thresh else "black")
                                  
            plt.close(fig)  # Close the temporary figure
            
        except Exception as e:
            logger.warning(f"Could not create confusion matrix plot: {e}")
            
        if self.is_binary and hasattr(self, 'tp_label'):
            self.tp_label.setText(str(metrics.get('true_positives', 0)))
            self.tn_label.setText(str(metrics.get('true_negatives', 0)))
            self.fp_label.setText(str(metrics.get('false_positives', 0)))
            self.fn_label.setText(str(metrics.get('false_negatives', 0)))
            
        self.cm_figure.tight_layout()
        self.cm_canvas.draw()
        
    def updateCurves(self):
        """Update ROC and PR curves (binary only)"""
        if not self.is_binary or not self.metrics_results:
            return
            
        metrics = self.metrics_results
        y_proba = metrics.get('probabilities')
        
        if y_proba is None:
            return
            
        try:
            positive_class = self.positive_class_combo.currentText()
            pos_idx = list(self.unique_classes).index(positive_class)
            y_score = y_proba[:, pos_idx]
            
            self.roc_figure.clear()
            roc_fig = self.visualizer.plot_roc_curve(
                self.y, y_score, positive_class=positive_class, 
                title="ROC Curve", figsize=(6, 5)
            )
            
            ax_source = roc_fig.get_axes()[0]
            ax_target = self.roc_figure.add_subplot(111)
            
            for line in ax_source.get_lines():
                ax_target.plot(line.get_xdata(), line.get_ydata(), 
                              color=line.get_color(), linestyle=line.get_linestyle(),
                              linewidth=line.get_linewidth(), label=line.get_label())
                              
            ax_target.set_xlim(ax_source.get_xlim())
            ax_target.set_ylim(ax_source.get_ylim())
            ax_target.set_xlabel(ax_source.get_xlabel())
            ax_target.set_ylabel(ax_source.get_ylabel())
            ax_target.set_title(ax_source.get_title())
            ax_target.legend()
            ax_target.grid(True, alpha=0.3)
            
            plt.close(roc_fig)
            
            self.pr_figure.clear()
            pr_fig = self.visualizer.plot_precision_recall_curve(
                self.y, y_score, positive_class=positive_class,
                title="Precision-Recall Curve", figsize=(6, 5)
            )
            
            ax_source = pr_fig.get_axes()[0]
            ax_target = self.pr_figure.add_subplot(111)
            
            for line in ax_source.get_lines():
                ax_target.plot(line.get_xdata(), line.get_ydata(),
                              color=line.get_color(), linestyle=line.get_linestyle(),
                              linewidth=line.get_linewidth(), label=line.get_label())
                              
            for collection in ax_source.collections:
                ax_target.add_collection(collection)
                
            ax_target.set_xlim(ax_source.get_xlim())
            ax_target.set_ylim(ax_source.get_ylim())
            ax_target.set_xlabel(ax_source.get_xlabel())
            ax_target.set_ylabel(ax_source.get_ylabel())
            ax_target.set_title(ax_source.get_title())
            ax_target.legend()
            ax_target.grid(True, alpha=0.3)
            
            plt.close(pr_fig)
            
        except Exception as e:
            logger.warning(f"Could not create curves: {e}")
            
        self.roc_figure.tight_layout()
        self.roc_canvas.draw()
        
        self.pr_figure.tight_layout()
        self.pr_canvas.draw()
        
    def updateClassificationReport(self):
        """Update the classification report"""
        if not self.metrics_results:
            return
            
        metrics = self.metrics_results
        report = metrics.get('classification_report', {})
        
        if not report:
            return
            
        self.report_figure.clear()
        
        try:
            fig = self.visualizer.plot_classification_report(
                report, title="Classification Report", figsize=(10, 6)
            )
            
            ax_source = fig.get_axes()[0]
            ax_target = self.report_figure.add_subplot(111)
            
            for image in ax_source.get_images():
                ax_target.add_image(image)
                
            ax_target.set_xlim(ax_source.get_xlim())
            ax_target.set_ylim(ax_source.get_ylim())
            ax_target.set_xlabel(ax_source.get_xlabel())
            ax_target.set_ylabel(ax_source.get_ylabel())
            ax_target.set_title(ax_source.get_title())
            
            plt.close(fig)
            
        except Exception as e:
            logger.warning(f"Could not create report visualization: {e}")
            
        self.report_figure.tight_layout()
        self.report_canvas.draw()
        
        report_text = "Classification Report:\\n\\n"
        
        for class_name, class_metrics in report.items():
            if isinstance(class_metrics, dict):
                report_text += f"{class_name}:\\n"
                for metric, value in class_metrics.items():
                    if isinstance(value, (int, float)):
                        report_text += f"  {metric}: {value:.3f}\\n"
                    else:
                        report_text += f"  {metric}: {value}\\n"
                report_text += "\\n"
                
        self.report_text.setText(report_text)
        
    def updateDetailedMetrics(self):
        """Update the detailed metrics table"""
        if not self.metrics_results:
            return
            
        metrics = self.metrics_results
        
        display_metrics = {}
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key not in ['predictions', 'probabilities']:
                display_metrics[key] = value
                
        self.metrics_table.setRowCount(len(display_metrics))
        
        for i, (metric, value) in enumerate(display_metrics.items()):
            metric_item = QTableWidgetItem(metric.replace('_', ' ').title())
            self.metrics_table.setItem(i, 0, metric_item)
            
            if isinstance(value, float):
                value_item = QTableWidgetItem(f"{value:.6f}")
            else:
                value_item = QTableWidgetItem(str(value))
            value_item.setTextAlignment(Qt.AlignRight)
            self.metrics_table.setItem(i, 1, value_item)
            
    def exportResults(self):
        """Export evaluation results"""
        if not self.metrics_results:
            QMessageBox.warning(self, "No Results", "No evaluation results to export.")
            return
            
        from PyQt5.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Evaluation Results", 
            "performance_evaluation.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            export_data = []
            
            for metric, value in self.metrics_results.items():
                if isinstance(value, (int, float)) and metric not in ['predictions', 'probabilities']:
                    export_data.append({
                        'Metric': metric.replace('_', ' ').title(),
                        'Value': value
                    })
                    
            export_df = pd.DataFrame(export_data)
            export_df.to_csv(file_path, index=False)
            
            QMessageBox.information(self, "Export Successful", 
                                  f"Results exported to: {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", 
                               f"Error exporting results: {str(e)}")