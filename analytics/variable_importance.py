#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Variable Importance Module for Bespoke Utility
Calculates variable importance for decision tree models.

"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable
import pandas as pd
import numpy as np
from enum import Enum
import random

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from PyQt5.QtCore import QObject, pyqtSignal
from models.decision_tree import BespokeDecisionTree
from models.node import TreeNode

logger = logging.getLogger(__name__)


class ImportanceMethod(Enum):
    """Enumeration of feature importance methods"""
    IMPURITY = "impurity"
    PERMUTATION = "permutation"
    DROP_COLUMN = "drop_column"
    SHAP = "shap"
    GAIN = "gain"


class VariableImportance(QObject):
    """Enhanced class for calculating variable importance in decision trees"""
    
    progress_updated = pyqtSignal(int)
    calculation_finished = pyqtSignal(dict)
    
    def __init__(self, parent=None, random_state: int = 42):
        """
        Initialize VariableImportance
        
        Args:
            parent: Parent QObject
            random_state: Random seed for reproducibility
        """
        super().__init__(parent)
        self.random_state = random_state
        self.last_importance = {}
        np.random.seed(random_state)
        random.seed(random_state)
    
    def compute_importance(self, X: pd.DataFrame, y: pd.Series, 
                          tree_root: TreeNode, method: ImportanceMethod = ImportanceMethod.IMPURITY,
                          n_repeats: int = 10, sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute feature importance using specified method
        
        Args:
            X: Feature DataFrame
            y: Target variable
            tree_root: Root node of the tree
            method: Method to use for importance calculation
            n_repeats: Number of permutation repeats (for permutation importance)
            sample_weight: Sample weights
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        logger.info(f"Computing variable importance using {method.value} method")
        
        if sample_weight is None:
            sample_weight = np.ones(len(X))
        
        importance = {feature: 0.0 for feature in X.columns}
        
        try:
            if method == ImportanceMethod.IMPURITY:
                importance = self._compute_impurity_importance(X, y, tree_root)
            elif method == ImportanceMethod.PERMUTATION:
                importance = self._compute_permutation_importance(X, y, tree_root, n_repeats, sample_weight)
            elif method == ImportanceMethod.DROP_COLUMN:
                importance = self._compute_drop_column_importance(X, y, tree_root, sample_weight)
            elif method == ImportanceMethod.GAIN:
                importance = self._compute_gain_importance(tree_root)
            else:
                logger.warning(f"Unsupported importance method: {method}")
                return importance
                
        except Exception as e:
            logger.error(f"Error computing importance with {method.value}: {str(e)}", exc_info=True)
            return importance
        
        self.last_importance = importance
        
        logger.info(f"Computed importance for {len([k for k, v in importance.items() if v > 0])} features")
        return importance
    
    def calculate_gini_importance(self, model: BespokeDecisionTree) -> Dict[str, float]:
        """
        Calculate Gini-based feature importance (enhanced version)
        
        Args:
            model: Fitted decision tree model
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not model.is_fitted or not model.root:
            logger.warning("Model is not fitted or has no root node")
            return {}
        
        logger.info("Calculating Gini-based feature importance")
        
        feature_names = getattr(model, 'feature_names', [])
        if not feature_names:
            logger.warning("No feature names available in model")
            return {}
        
        importance = {feature: 0.0 for feature in feature_names}
        
        total_samples = getattr(model.root, 'samples', 1) or 1
        
        all_nodes = self._get_all_tree_nodes(model.root)
        
        for node in all_nodes:
            if getattr(node, 'is_terminal', True) or not getattr(node, 'split_feature', None):
                continue
                
            self._calculate_node_importance(node, importance, total_samples)
        
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        importance = {k: v for k, v in importance.items() if v > 0}
        
        self.last_importance = importance
        logger.info(f"Calculated importance for {len(importance)} features")
        
        return importance
    
    def _get_all_tree_nodes(self, root_node):
        """Get all nodes in the tree using iterative traversal"""
        if not root_node:
            return []
        
        nodes = []
        stack = [root_node]
        
        while stack:
            node = stack.pop()
            nodes.append(node)
            
            children = getattr(node, 'children', [])
            if children:
                stack.extend(children)
        
        return nodes
    
    def _calculate_node_importance(self, node, importance: Dict[str, float], 
                                 total_samples: int) -> None:
        """
        Calculate importance for a single node and recursively for its children
        
        Args:
            node: Current tree node
            importance: Dictionary to accumulate importance scores
            total_samples: Total samples in the dataset
        """
        if getattr(node, 'is_terminal', True) or not getattr(node, 'split_feature', None):
            return
        
        node_samples = getattr(node, 'samples', 0) or 0
        node_impurity = getattr(node, 'impurity', 0) or 0
        split_feature = getattr(node, 'split_feature', None)
        children = getattr(node, 'children', [])
        
        if not split_feature or split_feature not in importance:
            return
        
        if node_samples == 0 or not children:
            return
        
        weighted_child_impurity = 0.0
        total_child_samples = 0
        
        for child in children:
            child_samples = getattr(child, 'samples', 0) or 0
            child_impurity = getattr(child, 'impurity', 0) or 0
            
            if child_samples > 0:
                weighted_child_impurity += (child_samples / node_samples) * child_impurity
                total_child_samples += child_samples
        
        if total_child_samples > 0:
            impurity_decrease = node_impurity - weighted_child_impurity
            
            sample_weight = node_samples / total_samples
            
            importance[split_feature] += sample_weight * impurity_decrease
    
    def _compute_impurity_importance(self, X: pd.DataFrame, y: pd.Series, 
                                   tree_root: TreeNode) -> Dict[str, float]:
        """Compute impurity-based importance using tree structure"""
        importance = {feature: 0.0 for feature in X.columns}
        
        total_samples = len(y)
        
        all_nodes = self._get_all_tree_nodes(tree_root)
        
        for node in all_nodes:
            if getattr(node, 'is_terminal', True) or not getattr(node, 'split_feature', None):
                continue
                
            split_feature = getattr(node, 'split_feature', None)
            if split_feature not in importance:
                continue
                
            node_samples = getattr(node, 'samples', 0) or 0
            node_impurity = getattr(node, 'impurity', 0) or 0
            children = getattr(node, 'children', [])
            
            if node_samples > 0 and children:
                weighted_child_impurity = 0.0
                for child in children:
                    child_samples = getattr(child, 'samples', 0) or 0
                    child_impurity = getattr(child, 'impurity', 0) or 0
                    if child_samples > 0:
                        weight = child_samples / node_samples
                        weighted_child_impurity += weight * child_impurity
                
                impurity_decrease = node_impurity - weighted_child_impurity
                
                sample_weight = node_samples / total_samples
                importance[split_feature] += sample_weight * impurity_decrease
        
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    def calculate_permutation_importance(self, model: BespokeDecisionTree, 
                                       X: pd.DataFrame, y: pd.Series,
                                       n_repeats: int = 10) -> Dict[str, float]:
        """
        Calculate permutation-based feature importance (enhanced version)
        
        Args:
            model: Fitted decision tree model
            X: Feature data
            y: Target variable
            n_repeats: Number of permutation repeats
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not model.is_fitted:
            logger.warning("Model is not fitted")
            return {}
        
        logger.info(f"Calculating permutation importance with {n_repeats} repeats")
        
        baseline_score = self._calculate_accuracy(model, X, y)
        
        importance = {}
        feature_names = list(X.columns)
        
        for i, feature in enumerate(feature_names):
            progress = int((i / len(feature_names)) * 100)
            self.progress_updated.emit(progress)
            
            feature_scores = []
            
            for repeat in range(n_repeats):
                X_permuted = X.copy()
                X_permuted[feature] = np.random.permutation(X_permuted[feature].values)
                
                permuted_score = self._calculate_accuracy(model, X_permuted, y)
                
                score_decrease = baseline_score - permuted_score
                feature_scores.append(score_decrease)
            
            importance[feature] = np.mean(feature_scores)
        
        total_importance = sum(max(0, v) for v in importance.values())
        if total_importance > 0:
            importance = {k: max(0, v) / total_importance for k, v in importance.items()}
        
        importance = {k: v for k, v in importance.items() if v > 0}
        
        self.last_importance = importance
        self.progress_updated.emit(100)
        
        logger.info(f"Calculated permutation importance for {len(importance)} features")
        
        return importance
    
    def _compute_permutation_importance(self, X: pd.DataFrame, y: pd.Series, 
                                      tree_root: TreeNode, n_repeats: int,
                                      sample_weight: np.ndarray) -> Dict[str, float]:
        """Internal method for permutation importance calculation"""
        baseline_score = self._tree_score(X, y, tree_root, sample_weight)
        
        importance = {}
        
        for feature in X.columns:
            feature_importance = 0.0
            
            for _ in range(n_repeats):
                X_permuted = X.copy()
                X_permuted[feature] = np.random.permutation(X_permuted[feature].values)
                
                permuted_score = self._tree_score(X_permuted, y, tree_root, sample_weight)
                
                feature_importance += baseline_score - permuted_score
            
            importance[feature] = feature_importance / n_repeats
        
        return importance
    
    def _compute_drop_column_importance(self, X: pd.DataFrame, y: pd.Series, 
                                      tree_root: TreeNode, sample_weight: np.ndarray) -> Dict[str, float]:
        """Compute drop-column importance"""
        logger.info("Drop-column importance not fully implemented, using permutation importance")
        return self._compute_permutation_importance(X, y, tree_root, 5, sample_weight)
    
    def _compute_gain_importance(self, tree_root: TreeNode) -> Dict[str, float]:
        """Compute gain-based importance from tree structure"""
        importance = {}
        
        all_nodes = self._get_all_tree_nodes(tree_root)
        
        for node in all_nodes:
            if getattr(node, 'is_terminal', True):
                continue
                
            split_feature = getattr(node, 'split_feature', None)
            if not split_feature:
                continue
                
            node_impurity = getattr(node, 'impurity', 0) or 0
            children = getattr(node, 'children', [])
            
            if children:
                weighted_child_impurity = 0.0
                node_samples = getattr(node, 'samples', 1) or 1
                
                for child in children:
                    child_samples = getattr(child, 'samples', 0) or 0
                    child_impurity = getattr(child, 'impurity', 0) or 0
                    if child_samples > 0:
                        weight = child_samples / node_samples
                        weighted_child_impurity += weight * child_impurity
                
                gain = node_impurity - weighted_child_impurity
                
                if split_feature not in importance:
                    importance[split_feature] = 0.0
                importance[split_feature] += gain
        
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    def _calculate_accuracy(self, model: BespokeDecisionTree, X: pd.DataFrame, 
                          y: pd.Series) -> float:
        """
        Calculate model accuracy using the existing predict method
        
        Args:
            model: Decision tree model
            X: Feature data
            y: Target variable
            
        Returns:
            Accuracy score
        """
        try:
            predictions = model.predict(X)
            if predictions is None or len(predictions) == 0:
                return 0.0
            
            if hasattr(predictions, 'values'):
                predictions = predictions.values
            if hasattr(y, 'values'):
                y_true = y.values
            else:
                y_true = np.array(y)
            
            accuracy = np.mean(predictions == y_true)
            return float(accuracy)
            
        except Exception as e:
            logger.error(f"Error calculating accuracy: {str(e)}")
            return 0.0
    
    def _tree_score(self, X: pd.DataFrame, y: pd.Series, tree_root: TreeNode, 
                   sample_weight: np.ndarray) -> float:
        """
        Calculate score (accuracy) for a tree using tree prediction
        
        Args:
            X: Feature DataFrame
            y: Target variable
            tree_root: Root node of the tree
            sample_weight: Sample weights
            
        Returns:
            Weighted accuracy score
        """
        try:
            if hasattr(tree_root, 'predict'):
                predictions = tree_root.predict(X)
            else:
                predictions = []
                for _, row in X.iterrows():
                    pred = self._predict_single_sample(tree_root, row)
                    predictions.append(pred)
                predictions = np.array(predictions)
            
            if hasattr(y, 'values'):
                y_true = y.values
            else:
                y_true = np.array(y)
                
            correct = (predictions == y_true)
            return np.sum(correct * sample_weight) / np.sum(sample_weight)
            
        except Exception as e:
            logger.error(f"Error calculating tree score: {str(e)}")
            return 0.0
    
    def _predict_single_sample(self, node, sample: pd.Series):
        """Predict a single sample by traversing the tree"""
        current = node
        
        while not getattr(current, 'is_terminal', True):
            split_feature = getattr(current, 'split_feature', None)
            split_value = getattr(current, 'split_value', None)
            children = getattr(current, 'children', [])
            
            if not split_feature or split_value is None or not children:
                break
                
            if split_feature in sample:
                feature_value = sample[split_feature]
                if len(children) >= 2:
                    if feature_value <= split_value:
                        current = children[0]  # Left child
                    else:
                        current = children[1]  # Right child
                else:
                    break
            else:
                break
        
        return getattr(current, 'majority_class', None) or getattr(current, 'prediction', None)
    
    def calculate_feature_interactions(self, model: BespokeDecisionTree, 
                                     X: pd.DataFrame, y: pd.Series,
                                     top_features: List[str] = None,
                                     n_repeats: int = 5) -> Dict[Tuple[str, str], float]:
        """
        Calculate pairwise feature interactions
        
        Args:
            model: Fitted decision tree model
            X: Feature data
            y: Target variable
            top_features: List of features to analyze (None for all)
            n_repeats: Number of permutation repeats
            
        Returns:
            Dictionary mapping feature pairs to interaction scores
        """
        if not model.is_fitted:
            logger.warning("Model is not fitted")
            return {}
        
        logger.info("Computing feature interactions")
        
        if top_features is None:
            importance = self.calculate_gini_importance(model)
            top_features = list(self.get_top_features(10, 'gini').keys())
        
        if len(top_features) > 10:
            logger.warning(f"Too many features ({len(top_features)}), limiting to 10")
            top_features = top_features[:10]
        
        baseline_score = self._calculate_accuracy(model, X, y)
        
        individual_importance = {}
        for feature in top_features:
            feature_scores = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                X_permuted[feature] = np.random.permutation(X_permuted[feature].values)
                permuted_score = self._calculate_accuracy(model, X_permuted, y)
                feature_scores.append(baseline_score - permuted_score)
            individual_importance[feature] = np.mean(feature_scores)
        
        interactions = {}
        for i, feat1 in enumerate(top_features):
            for feat2 in top_features[i+1:]:
                if (individual_importance.get(feat1, 0) <= 0 or 
                    individual_importance.get(feat2, 0) <= 0):
                    continue
                
                interaction_scores = []
                for _ in range(n_repeats):
                    X_permuted = X.copy()
                    X_permuted[feat1] = np.random.permutation(X_permuted[feat1].values)
                    X_permuted[feat2] = np.random.permutation(X_permuted[feat2].values)
                    
                    both_permuted_score = self._calculate_accuracy(model, X_permuted, y)
                    
                    both_decrease = baseline_score - both_permuted_score
                    individual_sum = (individual_importance[feat1] + 
                                    individual_importance[feat2])
                    
                    interaction = both_decrease - individual_sum
                    interaction_scores.append(interaction)
                
                interactions[(feat1, feat2)] = np.mean(interaction_scores)
        
        interactions = {k: v for k, v in interactions.items() if abs(v) > 0.001}
        
        logger.info(f"Found {len(interactions)} significant feature interactions")
        
        return interactions
    
    def get_top_features(self, n: int = 10, importance_type: str = 'gini') -> Dict[str, float]:
        """
        Get top N most important features
        
        Args:
            n: Number of top features to return
            importance_type: Type of importance to use
            
        Returns:
            Dictionary of top N features and their importance scores
        """
        if not self.last_importance:
            logger.warning("No importance calculated yet")
            return {}
        
        sorted_features = sorted(self.last_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        top_features = dict(sorted_features[:n])
        
        logger.info(f"Returning top {len(top_features)} features")
        
        return top_features
    
    def analyze_importance(self, importance: Dict[str, float], 
                         threshold: float = 0.01) -> Dict[str, Any]:
        """
        Analyze feature importance distribution and provide insights
        
        Args:
            importance: Feature importance dictionary
            threshold: Minimum importance threshold for "important" features
            
        Returns:
            Dictionary with analysis results
        """
        if not importance:
            return {"error": "No importance data provided"}
        
        sorted_importance = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        important_features = {f: v for f, v in sorted_importance.items() if v >= threshold}
        
        unimportant_features = {f: v for f, v in sorted_importance.items() if v < threshold}
        
        cumulative = 0.0
        cumulative_importance = {}
        
        for feature, value in sorted_importance.items():
            cumulative += value
            cumulative_importance[feature] = cumulative
        
        features_for_90 = 0
        for i, cum_value in enumerate(cumulative_importance.values()):
            if cum_value >= 0.9:
                features_for_90 = i + 1
                break
        
        summary = {
            "total_features": len(importance),
            "important_features": len(important_features),
            "unimportant_features": len(unimportant_features),
            "features_for_90_percent": features_for_90,
            "top_5_features": list(sorted_importance.keys())[:5],
            "top_5_importance": list(sorted_importance.values())[:5],
            "importance_threshold": threshold
        }
        
        return summary
    
    def export_importance_report(self, filepath: str, model_name: str = None) -> bool:
        """
        Export importance analysis to a text report
        
        Args:
            filepath: Path to save the report
            model_name: Name of the model (optional)
            
        Returns:
            True if export successful, False otherwise
        """
        if not self.last_importance:
            logger.warning("No importance data to export")
            return False
        
        try:
            from datetime import datetime
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("VARIABLE IMPORTANCE REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                if model_name:
                    f.write(f"Model: {model_name}\n")
                
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Features: {len(self.last_importance)}\n\n")
                
                f.write("FEATURE IMPORTANCE RANKING\n")
                f.write("-" * 30 + "\n")
                
                sorted_features = sorted(self.last_importance.items(), 
                                       key=lambda x: x[1], reverse=True)
                
                for rank, (feature, importance) in enumerate(sorted_features, 1):
                    f.write(f"{rank:2d}. {feature:<30} {importance:.6f}\n")
                
                f.write("\nEND OF REPORT\n")
            
            logger.info(f"Exported importance report to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting importance report: {str(e)}")
            return False
    
    def plot_importance(self, importance: Dict[str, float] = None, 
                       top_n: int = 15, title: str = "Feature Importance",
                       figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create a feature importance plot
        
        Args:
            importance: Feature importance dictionary (uses last_importance if None)
            top_n: Number of top features to show
            title: Plot title
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure or None if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return None
        
        if importance is None:
            importance = self.last_importance
        
        if not importance:
            logger.warning("No importance data to plot")
            return None
        
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_importance) > top_n:
            sorted_importance = sorted_importance[:top_n]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        features = [item[0] for item in sorted_importance]
        values = [item[1] for item in sorted_importance]
        
        bars = ax.barh(features, values, color='skyblue', edgecolor='navy', alpha=0.7)
        
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax.text(width + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.4f}', ha='left', va='center', fontsize=9)
        
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Features')
        ax.set_title(title)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        return fig


class InteractionImportance:
    """Class for calculating feature interaction importance"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize InteractionImportance
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
    
    def compute_interactions(self, X: pd.DataFrame, y: pd.Series, tree_root: TreeNode,
                           top_features: Optional[List[str]] = None, 
                           n_repeats: int = 5,
                           sample_weight: Optional[np.ndarray] = None) -> Dict[Tuple[str, str], float]:
        """
        Compute pairwise feature interactions
        
        Args:
            X: Feature DataFrame
            y: Target variable
            tree_root: Root node of the tree
            top_features: List of features to consider (None for all)
            n_repeats: Number of permutation repeats
            sample_weight: Sample weights
            
        Returns:
            Dictionary mapping feature pairs to interaction scores
        """
        logger.info("Computing feature interactions")
        
        if sample_weight is None:
            sample_weight = np.ones(len(X))
        
        if top_features is None:
            features = list(X.columns)
        else:
            features = top_features
        
        if len(features) > 10:
            logger.warning(f"Too many features ({len(features)}) for interaction calculation, limiting to 10")
            features = features[:10]
        
        baseline_score = self._tree_score(X, y, tree_root, sample_weight)
        
        individual_importance = {}
        
        for feature in features:
            feature_importance = 0.0
            
            for _ in range(n_repeats):
                X_permuted = X.copy()
                X_permuted[feature] = np.random.permutation(X_permuted[feature].values)
                
                permuted_score = self._tree_score(X_permuted, y, tree_root, sample_weight)
                
                feature_importance += baseline_score - permuted_score
            
            individual_importance[feature] = feature_importance / n_repeats
        
        interactions = {}
        
        for i, feat1 in enumerate(features):
            for feat2 in features[i+1:]:
                if individual_importance[feat1] <= 0 or individual_importance[feat2] <= 0:
                    interactions[(feat1, feat2)] = 0.0
                    continue
                
                pair_importance = 0.0
                
                for _ in range(n_repeats):
                    X_permuted = X.copy()
                    X_permuted[feat1] = np.random.permutation(X_permuted[feat1].values)
                    X_permuted[feat2] = np.random.permutation(X_permuted[feat2].values)
                    
                    permuted_score = self._tree_score(X_permuted, y, tree_root, sample_weight)
                    
                    pair_importance += baseline_score - permuted_score
                
                pair_importance /= n_repeats
                
                interaction = pair_importance - individual_importance[feat1] - individual_importance[feat2]
                
                interactions[(feat1, feat2)] = interaction
        
        interactions = dict(sorted(interactions.items(), key=lambda x: abs(x[1]), reverse=True))
        
        logger.info(f"Computed {len(interactions)} feature interactions")
        
        return interactions
    
    def _tree_score(self, X: pd.DataFrame, y: pd.Series, tree_root: TreeNode, 
                  sample_weight: np.ndarray) -> float:
        """
        Calculate score (accuracy) for a tree
        
        Args:
            X: Feature DataFrame
            y: Target variable
            tree_root: Root node of the tree
            sample_weight: Sample weights
            
        Returns:
            Accuracy score
        """
        try:
            if hasattr(tree_root, 'predict'):
                predictions = tree_root.predict(X)
            else:
                predictions = []
                for _, row in X.iterrows():
                    pred = self._predict_sample(tree_root, row)
                    predictions.append(pred)
                predictions = np.array(predictions)
            
            if hasattr(y, 'values'):
                y_true = y.values
            else:
                y_true = np.array(y)
                
            correct = (predictions == y_true)
            return np.sum(correct * sample_weight) / np.sum(sample_weight)
            
        except Exception as e:
            logger.error(f"Error calculating tree score: {str(e)}")
            return 0.0
    
    def _predict_sample(self, node, sample: pd.Series):
        """Predict a single sample"""
        current = node
        
        while not getattr(current, 'is_terminal', True):
            split_feature = getattr(current, 'split_feature', None)
            split_value = getattr(current, 'split_value', None)
            children = getattr(current, 'children', [])
            
            if not split_feature or split_value is None or not children:
                break
                
            if split_feature in sample and len(children) >= 2:
                feature_value = sample[split_feature]
                if feature_value <= split_value:
                    current = children[0]  # Left child
                else:
                    current = children[1]  # Right child
            else:
                break
        
        return getattr(current, 'majority_class', None) or getattr(current, 'prediction', None)
    
    def plot_interactions(self, interactions: Dict[Tuple[str, str], float], 
                        top_n: int = 10, title: str = "Feature Interactions",
                        figsize: Tuple[int, int] = (12, 8), 
                        positive_color: str = "green", negative_color: str = "red") -> plt.Figure:
        """
        Create a feature interaction plot
        
        Args:
            interactions: Feature interaction dictionary
            top_n: Number of top interactions to show
            title: Plot title
            figsize: Figure size (width, height)
            positive_color: Color for positive interactions
            negative_color: Color for negative interactions
            
        Returns:
            Matplotlib figure or None if matplotlib not available
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return None
        
        if len(interactions) > top_n:
            interactions = dict(sorted(interactions.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        pairs = [f"{f1} × {f2}" for f1, f2 in interactions.keys()]
        values = list(interactions.values())
        
        idx = np.argsort(values)
        pairs = [pairs[i] for i in idx]
        values = [values[i] for i in idx]
        
        colors = [positive_color if v >= 0 else negative_color for v in values]
        
        ax.barh(pairs, values, color=colors)
        
        ax.set_xlabel("Interaction Strength")
        ax.set_ylabel("Feature Pairs")
        ax.set_title(title)
        
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        
        return fig