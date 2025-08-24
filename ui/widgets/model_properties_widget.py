    # utility/ui/widgets/model_properties_widget.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Properties Widget for Bespoke Utility
Displays information and configuration of a decision tree model.
"""

import logging
from typing import Optional, Dict, Any

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel,
    QGroupBox, QTextEdit, QScrollArea
)
from PyQt5.QtCore import Qt

from models.decision_tree import BespokeDecisionTree # For type hinting

logger = logging.getLogger(__name__)

class ModelPropertiesWidget(QWidget):
    """
    Widget to display properties of a BespokeDecisionTree model.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._model: Optional[BespokeDecisionTree] = None
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5,5,5,5)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)

        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        
        form_layout = QFormLayout(content_widget)
        form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form_layout.setLabelAlignment(Qt.AlignTop)


        model_info_group = QGroupBox("Model Information")
        model_info_layout = QFormLayout(model_info_group)
        self.model_name_label = QLabel("N/A")
        model_info_layout.addRow("Model Name:", self.model_name_label)
        self.model_id_label = QLabel("N/A")
        model_info_layout.addRow("Model ID:", self.model_id_label)
        self.is_fitted_label = QLabel("N/A")
        model_info_layout.addRow("Is Fitted:", self.is_fitted_label)
        self.target_name_label = QLabel("N/A")
        model_info_layout.addRow("Target Variable:", self.target_name_label)
        self.num_features_label = QLabel("N/A")
        model_info_layout.addRow("Number of Features Used:", self.num_features_label)
        self.training_samples_label = QLabel("N/A")
        model_info_layout.addRow("Training Samples:", self.training_samples_label)
        form_layout.addWidget(model_info_group)


        tree_structure_group = QGroupBox("Tree Structure")
        tree_structure_layout = QFormLayout(tree_structure_group)
        self.num_nodes_label = QLabel("N/A")
        tree_structure_layout.addRow("Total Nodes:", self.num_nodes_label)
        self.num_leaves_label = QLabel("N/A")
        tree_structure_layout.addRow("Leaf Nodes:", self.num_leaves_label)
        self.max_depth_label = QLabel("N/A")
        tree_structure_layout.addRow("Max Depth:", self.max_depth_label)
        form_layout.addWidget(tree_structure_group)

        params_group = QGroupBox("Model Parameters")
        params_layout = QFormLayout(params_group)
        self.params_textedit = QTextEdit()
        self.params_textedit.setReadOnly(True)
        self.params_textedit.setFixedHeight(150) # Adjust as needed
        params_layout.addRow(self.params_textedit)
        form_layout.addWidget(params_group)

        metrics_group = QGroupBox("Performance Metrics (on training/last evaluation)")
        metrics_layout = QFormLayout(metrics_group)
        self.metrics_textedit = QTextEdit()
        self.metrics_textedit.setReadOnly(True)
        self.metrics_textedit.setFixedHeight(120)
        metrics_layout.addRow(self.metrics_textedit)
        form_layout.addWidget(metrics_group)
        
        importance_group = QGroupBox("Top Feature Importances")
        importance_layout = QFormLayout(importance_group)
        self.importance_textedit = QTextEdit()
        self.importance_textedit.setReadOnly(True)
        self.importance_textedit.setFixedHeight(150)
        importance_layout.addRow(self.importance_textedit)
        form_layout.addWidget(importance_group)


    def set_model(self, model: Optional[BespokeDecisionTree]):
        """
        Sets the decision tree model to display properties for.
        Args:
            model: The BespokeDecisionTree instance.
        """
        self._model = model
        self.update_properties()

    def update_properties(self):
        """Updates the displayed properties based on the current model."""
        if not self._model:
            self.model_name_label.setText("N/A")
            self.model_id_label.setText("N/A")
            self.is_fitted_label.setText("N/A")
            self.target_name_label.setText("N/A")
            self.num_features_label.setText("N/A")
            self.training_samples_label.setText("N/A")
            self.num_nodes_label.setText("N/A")
            self.num_leaves_label.setText("N/A")
            self.max_depth_label.setText("N/A")
            self.params_textedit.setPlainText("No model loaded.")
            self.metrics_textedit.setPlainText("N/A")
            self.importance_textedit.setPlainText("N/A")
            return

        self.model_name_label.setText(self._model.model_name or "N/A")
        self.model_id_label.setText(self._model.model_id or "N/A")
        self.is_fitted_label.setText(str(self._model.is_fitted))
        self.target_name_label.setText(self._model.target_name or "N/A")
        self.num_features_label.setText(str(len(self._model.feature_names)) if self._model.feature_names else "0")
        self.training_samples_label.setText(str(self.training_samples) if hasattr(self,'training_samples') else str(self._model.training_samples))


        if self._model.is_fitted:
            self.num_nodes_label.setText(str(self._model.num_nodes))
            self.num_leaves_label.setText(str(self.num_leaves) if hasattr(self,'num_leaves') else str(self._model.num_leaves)) # Fixed attribute name
            self.max_depth_label.setText(str(self.max_depth) if hasattr(self,'max_depth') else str(self._model.max_depth)) # Fixed attribute name

            params_str = []
            for key, value in self._model.get_params().items():
                params_str.append(f"{key}: {value}")
            self.params_textedit.setPlainText("\n".join(params_str))

            if self._model.metrics:
                metrics_str = []
                for key, value in self._model.metrics.items():
                    if isinstance(value, float):
                        metrics_str.append(f"{key}: {value:.4f}")
                    else: # For confusion matrix dict etc.
                        metrics_str.append(f"{key}: {value}")
                self.metrics_textedit.setPlainText("\n".join(metrics_str))
            else:
                self.metrics_textedit.setPlainText("Not yet computed or N/A.")

            if self._model.feature_importance:
                importance_str = []
                sorted_importance = sorted(self._model.feature_importance.items(), key=lambda item: item[1], reverse=True)
                for feature, imp in sorted_importance:
                    if imp > 0.0001: # Show only if somewhat important
                        importance_str.append(f"{feature}: {imp:.4f}")
                self.importance_textedit.setPlainText("\n".join(importance_str) if importance_str else "No significant features.")
            else:
                self.importance_textedit.setPlainText("Not yet computed or N/A.")

        else: # Model not fitted
            self.num_nodes_label.setText("N/A")
            self.num_leaves_label.setText("N/A")
            self.max_depth_label.setText("N/A")
            params_str = [] # Show default/set params even if not fitted
            for key, value in self._model.get_params().items():
                params_str.append(f"{key}: {value}")
            self.params_textedit.setPlainText("\n".join(params_str))
            self.metrics_textedit.setPlainText("Model not fitted.")
            self.importance_textedit.setPlainText("Model not fitted.")


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow
    from models.decision_tree import BespokeDecisionTree # For example
    app = QApplication(sys.argv)

    main_win = QMainWindow()
    props_widget = ModelPropertiesWidget()

    dummy_model_config = {
        "decision_tree": {
            "criterion": "gini", "max_depth": 5, "min_samples_leaf": 10
        }
    }
    model = BespokeDecisionTree(config=dummy_model_config)
    model.model_name = "TestCreditScoreModel"
    model.model_id = "uuid-12345"
    model.target_name = "Default"
    model.feature_names = ["Age", "Income", "DebtRatio", "CreditHistory"]


    props_widget.set_model(model) # Set the model

    main_win.setCentralWidget(props_widget)
    main_win.setWindowTitle("Model Properties Test")
    main_win.resize(500, 700)
    main_win.show()

    sys.exit(app.exec_())