#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Binary Classification Performance Dashboard for Bespoke Utility
Comprehensive dashboard for binary classification metrics and visualizations
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QTabWidget, QWidget, QListWidget, QListWidgetItem,
    QSplitter, QProgressBar, QMessageBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QFrame, QSizePolicy
)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate comprehensive binary classification metrics"""
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                            y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate all binary classification metrics"""
        metrics = {}
        
        try:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='binary')
            metrics['recall'] = recall_score(y_true, y_pred, average='binary')
            metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')
            
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm
            
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics['true_negative'] = int(tn)
                metrics['false_positive'] = int(fp)
                metrics['false_negative'] = int(fn)
                metrics['true_positive'] = int(tp)
                
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
                metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
                metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
                
            if y_prob is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
                    
                    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
                    metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
                    
                    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
                    metrics['pr_curve'] = {
                        'precision': precision, 
                        'recall': recall, 
                        'thresholds': pr_thresholds
                    }
                    
                except Exception as e:
                    logger.warning(f"Error calculating probability-based metrics: {e}")
                    
            report = classification_report(y_true, y_pred, output_dict=True)
            metrics['classification_report'] = report
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            metrics['error'] = str(e)
            
        return metrics


class MetricsVisualizationWidget(QWidget):
    """Widget for displaying binary classification visualizations"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str] = None):
        """Plot confusion matrix"""
        self.figure.clear()
        ax = self.figure.add_subplot(221)
        
        if class_names is None:
            class_names = ['Negative', 'Positive']
            
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, auc_score: float):
        """Plot ROC curve"""
        ax = self.figure.add_subplot(222)
        
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_precision_recall_curve(self, precision: np.ndarray, recall: np.ndarray):
        """Plot Precision-Recall curve"""
        ax = self.figure.add_subplot(223)
        
        ax.plot(recall, precision, linewidth=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.grid(True, alpha=0.3)
        
    def plot_metrics_comparison(self, metrics: Dict[str, float]):
        """Plot metrics comparison bar chart"""
        ax = self.figure.add_subplot(224)
        
        key_metrics = {
            'Accuracy': metrics.get('accuracy', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1-Score': metrics.get('f1_score', 0),
            'Specificity': metrics.get('specificity', 0)
        }
        
        names = list(key_metrics.keys())
        values = list(key_metrics.values())
        
        bars = ax.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax.set_ylim([0, 1])
        ax.set_ylabel('Score')
        ax.set_title('Metrics Comparison')
        ax.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
                   
        self.figure.tight_layout()
        self.canvas.draw()
        
    def update_all_plots(self, metrics: Dict[str, Any]):
        """Update all plots with new metrics"""
        self.figure.clear()
        
        if 'confusion_matrix' in metrics:
            self.plot_confusion_matrix(metrics['confusion_matrix'])
            
        if 'roc_curve' in metrics and 'roc_auc' in metrics:
            roc_data = metrics['roc_curve']
            self.plot_roc_curve(roc_data['fpr'], roc_data['tpr'], metrics['roc_auc'])
            
        if 'pr_curve' in metrics:
            pr_data = metrics['pr_curve']
            self.plot_precision_recall_curve(pr_data['precision'], pr_data['recall'])
            
        self.plot_metrics_comparison(metrics)


class BinaryClassificationDashboard(QDialog):
    """Comprehensive dashboard for binary classification performance"""
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_prob: Optional[np.ndarray] = None, 
                 class_names: List[str] = None, parent=None):
        super().__init__(parent)
        
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.class_names = class_names or ['Negative', 'Positive']
        
        self.metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred, y_prob)
        
        self.setWindowTitle("Binary Classification Performance Dashboard")
        self.setMinimumSize(1200, 800)
        
        self.setup_ui()
        self.populate_dashboard()
        
    def setup_ui(self):
        """Set up the user interface"""
        layout = QVBoxLayout()
        
        header_label = QLabel("Binary Classification Performance Dashboard")
        header_label.setFont(QFont("Arial", 16, QFont.Bold))
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)
        
        self.tab_widget = QTabWidget()
        
        overview_tab = self.create_overview_tab()
        self.tab_widget.addTab(overview_tab, "Overview")
        
        metrics_tab = self.create_detailed_metrics_tab()
        self.tab_widget.addTab(metrics_tab, "Detailed Metrics")
        
        viz_tab = self.create_visualizations_tab()
        self.tab_widget.addTab(viz_tab, "Visualizations")
        
        report_tab = self.create_classification_report_tab()
        self.tab_widget.addTab(report_tab, "Classification Report")
        
        layout.addWidget(self.tab_widget)
        
        button_layout = QHBoxLayout()
        
        export_btn = QPushButton("Export Report")
        export_btn.clicked.connect(self.export_report)
        button_layout.addWidget(export_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
    def create_overview_tab(self) -> QWidget:
        """Create the overview tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        summary_group = QGroupBox("Performance Summary")
        summary_layout = QGridLayout()
        
        metric_cards = [
            ("Accuracy", self.metrics.get('accuracy', 0), "Overall correctness"),
            ("Precision", self.metrics.get('precision', 0), "Positive prediction accuracy"),
            ("Recall", self.metrics.get('recall', 0), "True positive capture rate"),
            ("F1-Score", self.metrics.get('f1_score', 0), "Harmonic mean of precision/recall"),
            ("Specificity", self.metrics.get('specificity', 0), "True negative rate"),
            ("AUC-ROC", self.metrics.get('roc_auc', 0), "Area under ROC curve")
        ]
        
        for i, (name, value, description) in enumerate(metric_cards):
            card = self.create_metric_card(name, value, description)
            row, col = divmod(i, 3)
            summary_layout.addWidget(card, row, col)
            
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        cm_group = QGroupBox("Confusion Matrix Summary")
        cm_layout = QFormLayout()
        
        if 'confusion_matrix' in self.metrics:
            cm = self.metrics['confusion_matrix']
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                
                cm_layout.addRow("True Positives:", QLabel(str(tp)))
                cm_layout.addRow("False Positives:", QLabel(str(fp)))
                cm_layout.addRow("True Negatives:", QLabel(str(tn)))
                cm_layout.addRow("False Negatives:", QLabel(str(fn)))
                
                total = tp + fp + tn + fn
                cm_layout.addRow("Total Samples:", QLabel(str(total)))
                
        cm_group.setLayout(cm_layout)
        layout.addWidget(cm_group)
        
        recommendations_group = QGroupBox("Recommendations")
        recommendations_layout = QVBoxLayout()
        
        recommendations_text = QTextEdit()
        recommendations_text.setReadOnly(True)
        recommendations_text.setMaximumHeight(150)
        
        recommendations = self.generate_recommendations()
        recommendations_text.setText(recommendations)
        
        recommendations_layout.addWidget(recommendations_text)
        recommendations_group.setLayout(recommendations_layout)
        layout.addWidget(recommendations_group)
        
        widget.setLayout(layout)
        return widget
        
    def create_metric_card(self, name: str, value: float, description: str) -> QWidget:
        """Create a metric display card"""
        card = QFrame()
        card.setFrameStyle(QFrame.Box)
        card.setLineWidth(1)
        
        layout = QVBoxLayout()
        
        name_label = QLabel(name)
        name_label.setFont(QFont("Arial", 10, QFont.Bold))
        name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(name_label)
        
        value_label = QLabel(f"{value:.3f}")
        value_label.setFont(QFont("Arial", 18, QFont.Bold))
        value_label.setAlignment(Qt.AlignCenter)
        
        if value >= 0.8:
            value_label.setStyleSheet("QLabel { color: green; }")
        elif value >= 0.6:
            value_label.setStyleSheet("QLabel { color: orange; }")
        else:
            value_label.setStyleSheet("QLabel { color: red; }")
            
        layout.addWidget(value_label)
        
        desc_label = QLabel(description)
        desc_label.setFont(QFont("Arial", 8))
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        card.setLayout(layout)
        return card
        
    def create_detailed_metrics_tab(self) -> QWidget:
        """Create detailed metrics tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        table_group = QGroupBox("All Metrics")
        table_layout = QVBoxLayout()
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(['Metric', 'Value'])
        
        header = self.metrics_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        
        table_layout.addWidget(self.metrics_table)
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)
        
        widget.setLayout(layout)
        return widget
        
    def create_visualizations_tab(self) -> QWidget:
        """Create visualizations tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.viz_widget = MetricsVisualizationWidget()
        layout.addWidget(self.viz_widget)
        
        widget.setLayout(layout)
        return widget
        
    def create_classification_report_tab(self) -> QWidget:
        """Create classification report tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        report_group = QGroupBox("Sklearn Classification Report")
        report_layout = QVBoxLayout()
        
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setFont(QFont("Courier", 10))
        
        report_layout.addWidget(self.report_text)
        report_group.setLayout(report_layout)
        layout.addWidget(report_group)
        
        widget.setLayout(layout)
        return widget
        
    def populate_dashboard(self):
        """Populate all dashboard components with data"""
        self.populate_metrics_table()
        
        self.viz_widget.update_all_plots(self.metrics)
        
        self.populate_classification_report()
        
    def populate_metrics_table(self):
        """Populate the detailed metrics table"""
        display_metrics = [
            ('Accuracy', self.metrics.get('accuracy', 0)),
            ('Precision', self.metrics.get('precision', 0)),
            ('Recall (Sensitivity)', self.metrics.get('recall', 0)),
            ('Specificity', self.metrics.get('specificity', 0)),
            ('F1-Score', self.metrics.get('f1_score', 0)),
            ('Positive Predictive Value', self.metrics.get('positive_predictive_value', 0)),
            ('Negative Predictive Value', self.metrics.get('negative_predictive_value', 0)),
            ('False Positive Rate', self.metrics.get('false_positive_rate', 0)),
            ('False Negative Rate', self.metrics.get('false_negative_rate', 0)),
            ('True Positives', self.metrics.get('true_positive', 0)),
            ('False Positives', self.metrics.get('false_positive', 0)),
            ('True Negatives', self.metrics.get('true_negative', 0)),
            ('False Negatives', self.metrics.get('false_negative', 0))
        ]
        
        if 'roc_auc' in self.metrics:
            display_metrics.append(('AUC-ROC', self.metrics['roc_auc']))
            
        self.metrics_table.setRowCount(len(display_metrics))
        
        for i, (metric_name, value) in enumerate(display_metrics):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric_name))
            
            if isinstance(value, (int, float)):
                if isinstance(value, int):
                    value_text = str(value)
                else:
                    value_text = f"{value:.4f}"
            else:
                value_text = str(value)
                
            self.metrics_table.setItem(i, 1, QTableWidgetItem(value_text))
            
    def populate_classification_report(self):
        """Populate the classification report"""
        if 'classification_report' in self.metrics:
            from sklearn.metrics import classification_report
            report_text = classification_report(self.y_true, self.y_pred, 
                                              target_names=self.class_names)
            self.report_text.setText(report_text)
        else:
            self.report_text.setText("Classification report not available")
            
    def generate_recommendations(self) -> str:
        """Generate performance recommendations"""
        recommendations = []
        
        accuracy = self.metrics.get('accuracy', 0)
        precision = self.metrics.get('precision', 0)
        recall = self.metrics.get('recall', 0)
        f1 = self.metrics.get('f1_score', 0)
        
        if accuracy >= 0.9:
            recommendations.append("✓ Excellent overall performance!")
        elif accuracy >= 0.8:
            recommendations.append("✓ Good overall performance")
        elif accuracy >= 0.7:
            recommendations.append("⚠ Moderate performance - consider model improvements")
        else:
            recommendations.append("⚠ Poor performance - model needs significant improvement")
            
        if precision > 0.8 and recall > 0.8:
            recommendations.append("✓ Balanced precision and recall")
        elif precision > recall + 0.2:
            recommendations.append("⚠ High precision, low recall - model is conservative")
            recommendations.append("  Consider adjusting threshold to improve recall")
        elif recall > precision + 0.2:
            recommendations.append("⚠ High recall, low precision - model predicts many false positives")
            recommendations.append("  Consider adjusting threshold to improve precision")
            
        if 'confusion_matrix' in self.metrics:
            cm = self.metrics['confusion_matrix']
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                total_positive = tp + fn
                total_negative = tn + fp
                imbalance_ratio = min(total_positive, total_negative) / max(total_positive, total_negative)
                
                if imbalance_ratio < 0.1:
                    recommendations.append("⚠ Severe class imbalance detected")
                    recommendations.append("  Consider using balanced class weights or resampling")
                elif imbalance_ratio < 0.3:
                    recommendations.append("⚠ Moderate class imbalance - monitor minority class performance")
                    
        if precision < 0.6:
            recommendations.append("⚠ Low precision - many false positives")
        if recall < 0.6:
            recommendations.append("⚠ Low recall - missing many true positives")
        if f1 < 0.6:
            recommendations.append("⚠ Low F1-score - poor overall classification performance")
            
        return "\n".join(recommendations)
        
    def export_report(self):
        """Export performance report"""
        QMessageBox.information(self, "Export", "Report export functionality coming soon!")


def show_binary_classification_dashboard(y_true: np.ndarray, y_pred: np.ndarray,
                                       y_prob: Optional[np.ndarray] = None,
                                       class_names: List[str] = None,
                                       parent=None) -> BinaryClassificationDashboard:
    """
    Convenience function to show the binary classification dashboard
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_prob: Predicted probabilities (optional)
        class_names: Names for the classes (optional)
        parent: Parent widget (optional)
        
    Returns:
        Dashboard dialog instance
    """
    dashboard = BinaryClassificationDashboard(y_true, y_pred, y_prob, class_names, parent)
    dashboard.show()
    return dashboard