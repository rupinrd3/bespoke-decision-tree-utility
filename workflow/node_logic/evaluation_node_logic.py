# utility/workflow/node_logic/evaluation_node_logic.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluation Node Logic for Bespoke Utility Workflow
Defines the execution behavior of an Evaluation node.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd

from models.decision_tree import BespokeDecisionTree
from analytics.performance_metrics import MetricsCalculator, MetricsVisualizer
from ui.workflow_canvas import EvaluationNode # For type hinting if needed

logger = logging.getLogger(__name__)

class EvaluationNodeLogic:
    """
    Handles the logic for an EvaluationNode within the workflow.
    This class is responsible for taking a trained model and a dataset,
    computing performance metrics, and potentially generating visualizations.
    """

    def __init__(self, node_id: str, config: Dict[str, Any]):
        self.node_id = node_id
        self.config = config
        self.metrics_calculator = MetricsCalculator()
        self.metrics_visualizer = MetricsVisualizer()
        logger.info(f"EvaluationNodeLogic for node '{node_id}' initialized.")

    def execute(self, inputs: Dict[str, Any], evaluation_node_config: Optional[EvaluationNode] = None) -> Dict[str, Any]:
        """
        Executes the evaluation process.

        Args:
            inputs: A dictionary containing the inputs for this node.
                    Expected keys:
                    - 'Model Input': The trained BespokeDecisionTree model.
                    - 'Data Input': The pandas DataFrame to evaluate the model on.
            evaluation_node_config: The EvaluationNode instance from the canvas,
                                     containing specific configuration like selected metrics.

        Returns:
            A dictionary containing the computed performance metrics.
            Example: {'accuracy': 0.85, 'precision': 0.8, ...}
        """
        logger.info(f"Executing EvaluationNodeLogic for node '{self.node_id}'")

        model_input: Optional[BespokeDecisionTree] = inputs.get('Model Input')
        data_input: Optional[pd.DataFrame] = inputs.get('Data Input')

        if not isinstance(model_input, BespokeDecisionTree):
            error_msg = f"EvaluationNode '{self.node_id}' expects a BespokeDecisionTree for 'Model Input', got {type(model_input)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not model_input.is_fitted:
            error_msg = f"Model provided to EvaluationNode '{self.node_id}' is not fitted."
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(data_input, pd.DataFrame):
            error_msg = f"EvaluationNode '{self.node_id}' expects a pandas DataFrame for 'Data Input', got {type(data_input)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not model_input.target_name or model_input.target_name not in data_input.columns:
            error_msg = (f"Target column '{model_input.target_name}' defined in the model "
                         f"was not found in the evaluation dataset for node '{self.node_id}'.")
            logger.error(error_msg)
            raise ValueError(error_msg)

        X_eval = data_input.drop(columns=[model_input.target_name])
        y_eval = data_input[model_input.target_name]

        positive_class_override = None
        selected_metrics_list = None # TODO: Allow user to select specific metrics via node config

        if evaluation_node_config:
            logger.debug(f"Using configuration from EvaluationNode object for node '{self.node_id}'.")
        else: # Fallback to model's positive class or defaults if node config not passed
            positive_class_override = model_input.positive_class


        logger.info(f"Computing performance metrics for model '{model_input.model_name}' on evaluation data.")
        
        # Note: The BespokeDecisionTree itself has a compute_metrics method.
        
        computed_metrics = self.metrics_calculator.compute_metrics(
            X=X_eval,
            y=y_eval,
            tree_root=model_input.root, # Assuming direct access to the root TreeNode
            sample_weight=None, # Add sample_weight if your workflow supports it for evaluation
            positive_class=positive_class_override
        )


        logger.info(f"Metrics computed for node '{self.node_id}': {computed_metrics.get('accuracy', 'N/A')}")
        
        computed_metrics['model'] = model_input
        computed_metrics['dataset'] = data_input
        computed_metrics['target_variable'] = model_input.target_name
        
        return computed_metrics

    def get_configurable_parameters(self) -> Dict[str, Any]:
        """
        Returns a dictionary of parameters that can be configured for this node type.
        This could be used by a UI to generate a configuration dialog.
        """
        return {
            "selected_metrics": {
                "type": "multiselect",
                "options": ["accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix"],
                "default": ["accuracy", "precision", "recall", "f1_score", "roc_auc"],
                "description": "Select metrics to compute."
            },
            "positive_class_label": {
                "type": "string", # Or a dropdown based on target variable classes
                "default": None, # Will use model's default if None
                "description": "Specify the positive class label for binary metrics (e.g., '1', 'Yes')."
            }
        }

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Testing EvaluationNodeLogic...")

    test_config = {"app_name": "TestApp"}

    from models.node import TreeNode
    dummy_model = BespokeDecisionTree(config=test_config)
    dummy_model.target_name = 'target'
    dummy_model.target_values = [0, 1]
    dummy_model.positive_class = 1
    dummy_model.class_names = ['0', '1']
    dummy_model.feature_names = ['feature1', 'feature2']
    dummy_model.root = TreeNode(node_id="root", depth=0)
    dummy_model.root.update_stats(samples=10, class_counts={0:5, 1:5}, impurity=0.5)
    dummy_model.root.split_feature = 'feature1'
    dummy_model.root.split_value = 0.5
    dummy_model.root.split_type = 'numeric'
    
    leaf1 = TreeNode(node_id="L1", parent=dummy_model.root, depth=1, is_terminal=True)
    leaf1.update_stats(samples=5, class_counts={0:4, 1:1}, impurity=0.32) # Predicts 0
    leaf1.majority_class = 0
    
    leaf2 = TreeNode(node_id="R1", parent=dummy_model.root, depth=1, is_terminal=True)
    leaf2.update_stats(samples=5, class_counts={0:1, 1:4}, impurity=0.32) # Predicts 1
    leaf2.majority_class = 1
    
    dummy_model.root.children = [leaf1, leaf2]
    dummy_model.is_fitted = True


    dummy_eval_data = pd.DataFrame({
        'feature1': [0.2, 0.8, 0.3, 0.9, 0.4],
        'feature2': [10, 20, 15, 25, 18],
        'target':   [0,   1,   0,   1,   0]
    })

    node_inputs = {
        'Model Input': dummy_model,
        'Data Input': dummy_eval_data
    }

    eval_logic = EvaluationNodeLogic(node_id="eval_node_test", config=test_config)
    
    try:
        metrics_output = eval_logic.execute(inputs=node_inputs)
        print("\nComputed Metrics:")
        for metric, value in metrics_output.items():
            if isinstance(value, dict): # e.g. confusion matrix
                print(f"  {metric}:")
                for k,v_ in value.items():
                    print(f"    {k}: {v_}")
            else:
                print(f"  {metric}: {value}")
    except Exception as e:
        print(f"Error during test execution: {e}")