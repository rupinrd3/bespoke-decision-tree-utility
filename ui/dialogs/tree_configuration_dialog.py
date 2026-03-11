# utility/ui/dialogs/tree_configuration_dialog.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Decision Tree Configuration Dialog for Bespoke Utility
Allows users to set global parameters for decision tree model building.
"""

import logging
from typing import Dict, Any, Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QSpinBox, QDoubleSpinBox, QComboBox, QPushButton, QDialogButtonBox,
    QGroupBox, QCheckBox, QMessageBox, QLabel
)
from PyQt5.QtCore import Qt

from models.decision_tree import SplitCriterion, TreeGrowthMode

logger = logging.getLogger(__name__)

class TreeConfigurationDialog(QDialog):
    """Dialog for configuring global decision tree model parameters."""

    def __init__(self, current_config: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Decision Tree Configuration")
        self.setMinimumWidth(450)

        self.initial_config = current_config or {} # Store initial config for reset
        self.config_data = self.initial_config.copy() # Working copy

        self.defaults = {
            "criterion": "gini",
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_impurity_decrease": 0.0,
            "max_features": None, # Represented as string "None" or type
            "max_leaf_nodes": None,
            "growth_mode": "automatic",
            "class_weight": None, # e.g., "balanced" or dict
            "random_state": 42,
            "pruning_enabled": True,
            "pruning_method": "cost_complexity",
            "pruning_alpha": 0.01
        }


        layout = QVBoxLayout(self)

        general_group = QGroupBox("General Tree Structure")
        general_layout = QFormLayout(general_group)

        self.criterion_combo = QComboBox()
        try:
            for criterion in SplitCriterion:
                if criterion and criterion.name:
                    self.criterion_combo.addItem(criterion.name.capitalize(), criterion.value)
        except Exception as e:
            logger.error(f"Error loading SplitCriterion enum: {e}")
            self.criterion_combo.addItem("Gini", "gini")
            self.criterion_combo.addItem("Entropy", "entropy")
        general_layout.addRow("Splitting Criterion:", self.criterion_combo)

        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(1, 100) # Depth 0 often means unlimited or single node
        self.max_depth_spin.setSpecialValueText("Unlimited") # Or handle None via checkbox
        general_layout.addRow("Max Depth:", self.max_depth_spin)

        self.growth_mode_combo = QComboBox()
        try:
            for mode in TreeGrowthMode:
                if mode and mode.name:
                    self.growth_mode_combo.addItem(mode.name.capitalize(), mode.value)
        except Exception as e:
            logger.error(f"Error loading TreeGrowthMode enum: {e}")
            self.growth_mode_combo.addItem("Automatic", "automatic")
            self.growth_mode_combo.addItem("Manual", "manual")
        general_layout.addRow("Growth Mode:", self.growth_mode_combo)
        
        manual_note = QLabel("Note: Manual tree building becomes available after model training is complete.")
        manual_note.setStyleSheet("color: #666; font-size: 9pt; font-style: italic;")
        manual_note.setWordWrap(True)
        general_layout.addRow("", manual_note)
        
        self.random_state_spin = QSpinBox()
        self.random_state_spin.setRange(0, 999999)
        general_layout.addRow("Random State (Seed):", self.random_state_spin)

        layout.addWidget(general_group)

        splitting_group = QGroupBox("Node Splitting Conditions")
        splitting_layout = QFormLayout(splitting_group)

        self.min_samples_split_spin = QSpinBox()
        self.min_samples_split_spin.setRange(2, 100000) # Min 2 samples to split
        splitting_layout.addRow("Min Samples to Split:", self.min_samples_split_spin)

        self.min_samples_leaf_spin = QSpinBox()
        self.min_samples_leaf_spin.setRange(1, 50000)
        splitting_layout.addRow("Min Samples per Leaf:", self.min_samples_leaf_spin)

        self.min_impurity_decrease_spin = QDoubleSpinBox()
        self.min_impurity_decrease_spin.setRange(0.0, 1.0)
        self.min_impurity_decrease_spin.setDecimals(5)
        self.min_impurity_decrease_spin.setSingleStep(0.0001)
        splitting_layout.addRow("Min Impurity Decrease for Split:", self.min_impurity_decrease_spin)
        
        self.max_features_edit = QLineEdit()
        self.max_features_edit.setPlaceholderText("e.g., None, auto, sqrt, log2, 0.5, 10")
        splitting_layout.addRow("Max Features per Split:", self.max_features_edit)

        self.max_leaf_nodes_spin = QSpinBox()
        self.max_leaf_nodes_spin.setRange(0, 10000) # 0 for unlimited
        self.max_leaf_nodes_spin.setSpecialValueText("Unlimited (0)")
        splitting_layout.addRow("Max Leaf Nodes:", self.max_leaf_nodes_spin)


        layout.addWidget(splitting_group)

        pruning_group = QGroupBox("Pruning")
        pruning_layout = QFormLayout(pruning_group)
        self.pruning_enabled_check = QCheckBox("Enable Pruning")
        pruning_layout.addRow(self.pruning_enabled_check)

        self.pruning_method_combo = QComboBox()
        self.pruning_method_combo.addItems(["Cost Complexity", "Reduced Error"]) # Add more if implemented
        pruning_layout.addRow("Pruning Method:", self.pruning_method_combo)

        self.pruning_alpha_spin = QDoubleSpinBox()
        self.pruning_alpha_spin.setRange(0.0, 1.0)
        self.pruning_alpha_spin.setDecimals(5)
        self.pruning_alpha_spin.setSingleStep(0.001)
        pruning_layout.addRow("Complexity Parameter (Alpha):", self.pruning_alpha_spin)
        
        self.pruning_enabled_check.toggled.connect(self._toggle_pruning_fields)

        layout.addWidget(pruning_group)
        
        class_weight_group = QGroupBox("Class Weighting")
        class_weight_layout = QFormLayout(class_weight_group)
        self.class_weight_combo = QComboBox()
        self.class_weight_combo.addItems(["None (Equal Weights)", "Balanced"])
        # TODO: Add option for custom dictionary input for class weights
        self.class_weight_combo.setToolTip("Handle imbalanced datasets by adjusting class weights.")
        class_weight_layout.addRow("Class Weight Strategy:", self.class_weight_combo)
        layout.addWidget(class_weight_group)


        self.button_box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        
        self.apply_button = QPushButton("Apply Configuration")
        self.apply_button.setDefault(True)
        self.apply_button.clicked.connect(self._apply_and_accept)
        self.button_box.addButton(self.apply_button, QDialogButtonBox.AcceptRole)
        
        self.button_box.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.load_defaults)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.load_config_to_ui()
        self._toggle_pruning_fields(self.pruning_enabled_check.isChecked())


    def _toggle_pruning_fields(self, enabled: bool):
        self.pruning_method_combo.setEnabled(enabled)
        self.pruning_alpha_spin.setEnabled(enabled)

    def load_config_to_ui(self):
        """Loads values from self.config_data into UI elements."""
        criterion = self.config_data.get("criterion", self.defaults["criterion"])
        idx = self.criterion_combo.findData(criterion)
        if idx != -1: self.criterion_combo.setCurrentIndex(idx)

        max_depth = self.config_data.get("max_depth", self.defaults["max_depth"])
        if max_depth is None or max_depth <=0: # Handle "Unlimited" case
            self.max_depth_spin.setValue(self.max_depth_spin.minimum()-1) # Triggers special value text
        else:
            self.max_depth_spin.setValue(max_depth)

        growth_mode = self.config_data.get("growth_mode", self.defaults["growth_mode"])
        idx = self.growth_mode_combo.findData(growth_mode)
        if idx != -1: self.growth_mode_combo.setCurrentIndex(idx)
        
        self.random_state_spin.setValue(self.config_data.get("random_state", self.defaults["random_state"]))

        self.min_samples_split_spin.setValue(self.config_data.get("min_samples_split", self.defaults["min_samples_split"]))
        self.min_samples_leaf_spin.setValue(self.config_data.get("min_samples_leaf", self.defaults["min_samples_leaf"]))
        self.min_impurity_decrease_spin.setValue(self.config_data.get("min_impurity_decrease", self.defaults["min_impurity_decrease"]))
        
        mf = self.config_data.get("max_features", self.defaults["max_features"])
        self.max_features_edit.setText(str(mf) if mf is not None else "None")

        mln = self.config_data.get("max_leaf_nodes", self.defaults["max_leaf_nodes"])
        self.max_leaf_nodes_spin.setValue(mln if mln is not None and mln > 0 else 0) # 0 for unlimited

        self.pruning_enabled_check.setChecked(self.config_data.get("pruning_enabled", self.defaults["pruning_enabled"]))
        
        pruning_method = self.config_data.get("pruning_method", self.defaults["pruning_method"])
        if pruning_method == "cost_complexity": self.pruning_method_combo.setCurrentText("Cost Complexity")
        elif pruning_method == "reduced_error": self.pruning_method_combo.setCurrentText("Reduced Error")
        
        self.pruning_alpha_spin.setValue(self.config_data.get("pruning_alpha", self.defaults["pruning_alpha"]))
        
        cw = self.config_data.get("class_weight", self.defaults["class_weight"])
        if cw is None: self.class_weight_combo.setCurrentText("None (Equal Weights)")
        elif cw == "balanced": self.class_weight_combo.setCurrentText("Balanced")
        # TODO: Handle custom dict for class_weight if that UI is added


    def load_defaults(self):
        """Loads default settings into the UI and self.config_data."""
        self.config_data = self.defaults.copy()
        self.load_config_to_ui()
        QMessageBox.information(self, "Defaults Loaded", "Default tree configurations have been loaded into the dialog.")


    def _apply_and_accept(self):
        """Saves UI values to self.config_data and accepts the dialog."""
        self.config_data["criterion"] = self.criterion_combo.currentData()
        self.config_data["max_depth"] = None if self.max_depth_spin.value() < self.max_depth_spin.minimum() else self.max_depth_spin.value()
        self.config_data["growth_mode"] = self.growth_mode_combo.currentData()
        self.config_data["random_state"] = self.random_state_spin.value()

        self.config_data["min_samples_split"] = self.min_samples_split_spin.value()
        self.config_data["min_samples_leaf"] = self.min_samples_leaf_spin.value()
        self.config_data["min_impurity_decrease"] = self.min_impurity_decrease_spin.value()
        
        mf_text = self.max_features_edit.text().strip()
        if mf_text.lower() in ["none", "null", ""]: self.config_data["max_features"] = None
        elif mf_text.lower() in ["auto", "sqrt"]: self.config_data["max_features"] = "sqrt" # or auto depending on sklearn interpretation
        elif mf_text.lower() == "log2": self.config_data["max_features"] = "log2"
        else:
            try: self.config_data["max_features"] = float(mf_text) if '.' in mf_text else int(mf_text)
            except ValueError:
                QMessageBox.warning(self, "Input Error", f"Invalid value for Max Features: '{mf_text}'. Using None.")
                self.config_data["max_features"] = None

        mln_val = self.max_leaf_nodes_spin.value()
        self.config_data["max_leaf_nodes"] = None if mln_val == 0 else mln_val


        self.config_data["pruning_enabled"] = self.pruning_enabled_check.isChecked()
        if self.pruning_enabled_check.isChecked():
            pruning_method_text = self.pruning_method_combo.currentText()
            if "Cost Complexity" in pruning_method_text: self.config_data["pruning_method"] = "cost_complexity"
            elif "Reduced Error" in pruning_method_text: self.config_data["pruning_method"] = "reduced_error"
            self.config_data["pruning_alpha"] = self.pruning_alpha_spin.value()
        else:
            self.config_data["pruning_method"] = None
            self.config_data["pruning_alpha"] = 0.0

        cw_text = self.class_weight_combo.currentText()
        if "None" in cw_text: self.config_data["class_weight"] = None
        elif "Balanced" in cw_text: self.config_data["class_weight"] = "balanced"
        # TODO: Handle custom dict for class_weight

        logger.debug(f"Tree configuration accepted: {self.config_data}")
        self.accept()

    def get_configuration(self) -> Dict[str, Any]:
        """Return the configured parameters."""
        return self.config_data

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)

    initial_settings = {
        "criterion": "entropy",
        "max_depth": 7,
        "min_samples_leaf": 5,
        "pruning_enabled": False,
        "random_state": 123
    }
    dialog = TreeConfigurationDialog(current_config=initial_settings)
    if dialog.exec_() == QDialog.Accepted:
        print("Final Configuration:", dialog.get_configuration())
    else:
        print("Dialog cancelled.")
    sys.exit(app.exec_())