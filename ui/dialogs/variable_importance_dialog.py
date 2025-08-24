#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Variable Importance Dialog for Bespoke Utility
Interface for computing and visualizing variable importance
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
    QFrame, QSizePolicy, QSlider, QButtonGroup, QRadioButton
)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from analytics.variable_importance import VariableImportance, ImportanceMethod, InteractionImportance

logger = logging.getLogger(__name__)


class VariableImportanceWorker(QThread):
    """Worker thread for computing variable importance"""
    
    progressUpdate = pyqtSignal(int, str)
    resultReady = pyqtSignal(dict)
    errorOccurred = pyqtSignal(str)
    
    def __init__(self, X, y, tree_root, method, n_repeats=10):
        super().__init__()
        self.X = X
        self.y = y
        self.tree_root = tree_root
        self.method = method
        self.n_repeats = n_repeats
        
    def run(self):
        """Run importance calculation"""
        try:
            self.progressUpdate.emit(10, "Initializing importance calculator...")
            
            calc = VariableImportance()
            
            self.progressUpdate.emit(30, f"Computing {self.method.value} importance...")
            
            importance = calc.compute_importance(
                self.X, self.y, self.tree_root, 
                method=self.method, n_repeats=self.n_repeats
            )
            
            self.progressUpdate.emit(80, "Generating summary...")
            
            summary = calc.get_importance_summary(importance)
            
            self.progressUpdate.emit(100, "Complete!")
            
            result = {
                'importance': importance,
                'summary': summary,
                'method': self.method.value
            }
            
            self.resultReady.emit(result)
            
        except Exception as e:
            logger.error(f"Error computing importance: {e}", exc_info=True)
            self.errorOccurred.emit(str(e))


class VariableImportanceDialog(QDialog):
    """Dialog for computing and visualizing variable importance"""
    
    def __init__(self, dataframe: pd.DataFrame, target_column: str, 
                 tree_model=None, parent=None):
        super().__init__(parent)
        
        self.dataframe = dataframe
        self.target_column = target_column
        self.tree_model = tree_model
        
        self.X = dataframe.drop(columns=[target_column])
        self.y = dataframe[target_column]
        
        self.importance_results = {}
        self.current_method = None
        
        self.setWindowTitle("Variable Importance Analysis")
        self.setModal(True)
        self.resize(1200, 800)
        
        self.setupUI()
        
    def setupUI(self):
        """Setup the main UI"""
        layout = QVBoxLayout()
        
        header_label = QLabel("Variable Importance Analysis")
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header_label)
        
        info_text = (f"Dataset: {len(self.dataframe)} rows, {len(self.X.columns)} features\\n"
                    f"Target: {self.target_column}")
        info_label = QLabel(info_text)
        layout.addWidget(info_label)
        
        self.tab_widget = QTabWidget()
        
        config_tab = self.createConfigurationTab()
        self.tab_widget.addTab(config_tab, "Configuration")
        
        results_tab = self.createResultsTab()
        self.tab_widget.addTab(results_tab, "Results")
        
        viz_tab = self.createVisualizationTab()
        self.tab_widget.addTab(viz_tab, "Visualization")
        
        comparison_tab = self.createComparisonTab()
        self.tab_widget.addTab(comparison_tab, "Comparison")
        
        layout.addWidget(self.tab_widget)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)
        
        button_layout = QHBoxLayout()
        
        self.compute_button = QPushButton("Compute Importance")
        self.compute_button.clicked.connect(self.computeImportance)
        
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.exportResults)
        self.export_button.setEnabled(False)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        
        button_layout.addWidget(self.compute_button)
        button_layout.addWidget(self.export_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def createConfigurationTab(self) -> QWidget:
        """Create the configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        method_group = QGroupBox("Importance Method")
        method_layout = QVBoxLayout()
        
        self.method_group = QButtonGroup()
        
        methods = [
            (ImportanceMethod.IMPURITY, "Impurity-based", 
             "Fast computation based on impurity decrease at splits"),
            (ImportanceMethod.PERMUTATION, "Permutation", 
             "Robust method based on prediction accuracy decrease"),
            (ImportanceMethod.GAIN, "Information Gain", 
             "Based on total information gain from splits"),
            (ImportanceMethod.DROP_COLUMN, "Drop Column", 
             "Importance based on accuracy loss when feature is removed")
        ]
        
        for i, (method, name, description) in enumerate(methods):
            radio = QRadioButton(name)
            radio.setToolTip(description)
            if i == 0:  # Default to first method
                radio.setChecked(True)
            
            radio.method = method
            
            self.method_group.addButton(radio, i)
            method_layout.addWidget(radio)
            
            desc_label = QLabel(f"  {description}")
            desc_label.setFont(QFont("Arial", 8))
            desc_label.setStyleSheet("color: gray;")
            method_layout.addWidget(desc_label)
            
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)
        
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()
        
        self.repeats_spin = QSpinBox()
        self.repeats_spin.setRange(1, 100)
        self.repeats_spin.setValue(10)
        self.repeats_spin.setToolTip("Number of permutation repeats (for permutation-based methods)")
        params_layout.addRow("Permutation Repeats:", self.repeats_spin)
        
        self.sample_limit_spin = QSpinBox()
        self.sample_limit_spin.setRange(100, 100000)
        self.sample_limit_spin.setValue(10000)
        self.sample_limit_spin.setToolTip("Maximum number of samples to use (for performance)")
        params_layout.addRow("Sample Limit:", self.sample_limit_spin)
        
        self.random_seed_spin = QSpinBox()
        self.random_seed_spin.setRange(0, 99999)
        self.random_seed_spin.setValue(42)
        params_layout.addRow("Random Seed:", self.random_seed_spin)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        features_group = QGroupBox("Feature Selection")
        features_layout = QVBoxLayout()
        
        self.all_features_radio = QRadioButton("Use all features")
        self.all_features_radio.setChecked(True)
        features_layout.addWidget(self.all_features_radio)
        
        self.top_features_radio = QRadioButton("Use top features only")
        features_layout.addWidget(self.top_features_radio)
        
        self.top_features_spin = QSpinBox()
        self.top_features_spin.setRange(5, len(self.X.columns))
        self.top_features_spin.setValue(min(20, len(self.X.columns)))
        self.top_features_spin.setEnabled(False)
        features_layout.addWidget(self.top_features_spin)
        
        self.top_features_radio.toggled.connect(self.top_features_spin.setEnabled)
        
        features_group.setLayout(features_layout)
        layout.addWidget(features_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def createResultsTab(self) -> QWidget:
        """Create the results tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([
            'Rank', 'Feature', 'Importance', 'Normalized %'
        ])
        
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        
        self.results_table.setSortingEnabled(True)
        self.results_table.setAlternatingRowColors(True)
        
        layout.addWidget(self.results_table)
        
        summary_group = QGroupBox("Summary Statistics")
        summary_layout = QFormLayout()
        
        self.total_features_label = QLabel("-")
        summary_layout.addRow("Total Features:", self.total_features_label)
        
        self.important_features_label = QLabel("-")
        summary_layout.addRow("Important Features (>1%):", self.important_features_label)
        
        self.features_90_label = QLabel("-")
        summary_layout.addRow("Features for 90% Importance:", self.features_90_label)
        
        self.top_feature_label = QLabel("-")
        summary_layout.addRow("Top Feature:", self.top_feature_label)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        widget.setLayout(layout)
        return widget
        
    def createVisualizationTab(self) -> QWidget:
        """Create the visualization tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        controls_layout = QHBoxLayout()
        
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(['Bar Chart', 'Horizontal Bar', 'Pie Chart'])
        self.plot_type_combo.currentTextChanged.connect(self.updateVisualization)
        controls_layout.addWidget(QLabel("Plot Type:"))
        controls_layout.addWidget(self.plot_type_combo)
        
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(5, 50)
        self.top_n_spin.setValue(15)
        self.top_n_spin.valueChanged.connect(self.updateVisualization)
        controls_layout.addWidget(QLabel("Show Top:"))
        controls_layout.addWidget(self.top_n_spin)
        
        controls_layout.addStretch()
        
        refresh_button = QPushButton("Refresh Plot")
        refresh_button.clicked.connect(self.updateVisualization)
        controls_layout.addWidget(refresh_button)
        
        layout.addLayout(controls_layout)
        
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        widget.setLayout(layout)
        return widget
        
    def createComparisonTab(self) -> QWidget:
        """Create the comparison tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        info_label = QLabel(
            "Run multiple importance methods to compare results. "
            "Features that rank highly across multiple methods are likely to be truly important."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        self.comparison_table = QTableWidget()
        self.comparison_table.setAlternatingRowColors(True)
        layout.addWidget(self.comparison_table)
        
        correlation_group = QGroupBox("Method Correlation")
        correlation_layout = QVBoxLayout()
        
        self.correlation_text = QTextEdit()
        self.correlation_text.setMaximumHeight(100)
        self.correlation_text.setReadOnly(True)
        correlation_layout.addWidget(self.correlation_text)
        
        correlation_group.setLayout(correlation_layout)
        layout.addWidget(correlation_group)
        
        widget.setLayout(layout)
        return widget
        
    def computeImportance(self):
        """Compute variable importance"""
        if self.tree_model is None:
            QMessageBox.warning(self, "No Model", 
                              "Please train a decision tree model first.")
            return
            
        selected_button = self.method_group.checkedButton()
        if selected_button is None:
            QMessageBox.warning(self, "No Method", "Please select an importance method.")
            return
            
        method = selected_button.method
        self.current_method = method
        
        n_repeats = self.repeats_spin.value()
        sample_limit = self.sample_limit_spin.value()
        
        X = self.X.copy()
        y = self.y.copy()
        
        if len(X) > sample_limit:
            sample_indices = np.random.choice(len(X), sample_limit, replace=False)
            X = X.iloc[sample_indices]
            y = y.iloc[sample_indices]
            
        if self.top_features_radio.isChecked():
            top_n = self.top_features_spin.value()
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            top_features = correlations.head(top_n).index.tolist()
            X = X[top_features]
            
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.compute_button.setEnabled(False)
        
        self.worker = VariableImportanceWorker(
            X, y, self.tree_model.root, method, n_repeats
        )
        self.worker.progressUpdate.connect(self.updateProgress)
        self.worker.resultReady.connect(self.onImportanceReady)
        self.worker.errorOccurred.connect(self.onImportanceError)
        self.worker.start()
        
    def updateProgress(self, progress: int, message: str):
        """Update progress bar and message"""
        self.progress_bar.setValue(progress)
        self.progress_label.setText(message)
        
    def onImportanceReady(self, result: Dict[str, Any]):
        """Handle importance computation completion"""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.compute_button.setEnabled(True)
        
        method_name = result['method']
        self.importance_results[method_name] = result
        
        self.updateResultsTable(result['importance'], result['summary'])
        self.updateVisualization()
        self.updateComparison()
        
        self.export_button.setEnabled(True)
        
        self.tab_widget.setCurrentIndex(1)
        
    def onImportanceError(self, error_message: str):
        """Handle importance computation error"""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.compute_button.setEnabled(True)
        
        QMessageBox.critical(self, "Computation Error", 
                           f"Error computing importance: {error_message}")
        
    def updateResultsTable(self, importance: Dict[str, float], summary: Dict[str, Any]):
        """Update the results table"""
        self.results_table.setRowCount(0)
        
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        self.results_table.setRowCount(len(sorted_importance))
        
        for i, (feature, score) in enumerate(sorted_importance):
            rank_item = QTableWidgetItem(str(i + 1))
            rank_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(i, 0, rank_item)
            
            feature_item = QTableWidgetItem(str(feature))
            self.results_table.setItem(i, 1, feature_item)
            
            score_item = QTableWidgetItem(f"{score:.6f}")
            score_item.setTextAlignment(Qt.AlignRight)
            self.results_table.setItem(i, 2, score_item)
            
            percentage = score * 100
            pct_item = QTableWidgetItem(f"{percentage:.2f}%")
            pct_item.setTextAlignment(Qt.AlignRight)
            self.results_table.setItem(i, 3, pct_item)
            
        self.total_features_label.setText(str(summary['total_features']))
        self.important_features_label.setText(str(summary['important_features']))
        self.features_90_label.setText(str(summary['features_for_90_percent']))
        
        if summary['top_5_features']:
            top_feature = summary['top_5_features'][0]
            top_score = summary['top_5_importance'][0]
            self.top_feature_label.setText(f"{top_feature} ({top_score:.3f})")
        else:
            self.top_feature_label.setText("-")
            
    def updateVisualization(self):
        """Update the visualization plot"""
        if not self.importance_results:
            return
            
        method_name = self.current_method.value if self.current_method else list(self.importance_results.keys())[0]
        if method_name not in self.importance_results:
            return
            
        importance = self.importance_results[method_name]['importance']
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        top_n = self.top_n_spin.value()
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        features = list(sorted_importance.keys())
        values = list(sorted_importance.values())
        
        plot_type = self.plot_type_combo.currentText()
        
        if plot_type == "Bar Chart":
            bars = ax.bar(features, values, color='skyblue')
            ax.set_xlabel('Features')
            ax.set_ylabel('Importance')
            ax.set_title(f'Feature Importance ({method_name})')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
        elif plot_type == "Horizontal Bar":
            idx = np.argsort(values)
            features_sorted = [features[i] for i in idx]
            values_sorted = [values[i] for i in idx]
            
            bars = ax.barh(features_sorted, values_sorted, color='lightcoral')
            ax.set_xlabel('Importance')
            ax.set_ylabel('Features')
            ax.set_title(f'Feature Importance ({method_name})')
            
        elif plot_type == "Pie Chart":
            if len(features) > 10:
                features = features[:10]
                values = values[:10]
                
            ax.pie(values, labels=features, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'Feature Importance ({method_name})')
            
        if plot_type in ["Bar Chart", "Horizontal Bar"]:
            ax.grid(True, alpha=0.3)
            
        self.figure.tight_layout()
        self.canvas.draw()
        
    def updateComparison(self):
        """Update the comparison table"""
        if len(self.importance_results) < 2:
            self.comparison_table.setRowCount(0)
            self.correlation_text.setText("Run at least two different methods to see comparison.")
            return
            
        all_features = set()
        for result in self.importance_results.values():
            all_features.update(result['importance'].keys())
            
        all_features = sorted(all_features)
        
        methods = list(self.importance_results.keys())
        self.comparison_table.setRowCount(len(all_features))
        self.comparison_table.setColumnCount(1 + len(methods))
        
        headers = ['Feature'] + methods
        self.comparison_table.setHorizontalHeaderLabels(headers)
        
        for i, feature in enumerate(all_features):
            feature_item = QTableWidgetItem(feature)
            self.comparison_table.setItem(i, 0, feature_item)
            
            for j, method in enumerate(methods):
                importance = self.importance_results[method]['importance']
                score = importance.get(feature, 0.0)
                score_item = QTableWidgetItem(f"{score:.4f}")
                score_item.setTextAlignment(Qt.AlignRight)
                self.comparison_table.setItem(i, j + 1, score_item)
                
        if len(methods) >= 2:
            correlations = {}
            
            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    imp1 = self.importance_results[method1]['importance']
                    imp2 = self.importance_results[method2]['importance']
                    
                    common_features = set(imp1.keys()) & set(imp2.keys())
                    if len(common_features) > 1:
                        values1 = [imp1[f] for f in common_features]
                        values2 = [imp2[f] for f in common_features]
                        
                        correlation = np.corrcoef(values1, values2)[0, 1]
                        correlations[f"{method1} vs {method2}"] = correlation
                        
            correlation_text = "Method Correlations:\\n"
            for pair, corr in correlations.items():
                correlation_text += f"{pair}: {corr:.3f}\\n"
                
            self.correlation_text.setText(correlation_text)
        
    def exportResults(self):
        """Export importance results"""
        if not self.importance_results:
            QMessageBox.warning(self, "No Results", "No importance results to export.")
            return
            
        from PyQt5.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Importance Results", 
            f"variable_importance_{self.current_method.value if self.current_method else 'results'}.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            export_data = []
            
            for method_name, result in self.importance_results.items():
                importance = result['importance']
                for feature, score in importance.items():
                    export_data.append({
                        'Method': method_name,
                        'Feature': feature,
                        'Importance': score,
                        'Percentage': score * 100
                    })
                    
            export_df = pd.DataFrame(export_data)
            export_df.to_csv(file_path, index=False)
            
            QMessageBox.information(self, "Export Successful", 
                                  f"Results exported to: {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", 
                               f"Error exporting results: {str(e)}")