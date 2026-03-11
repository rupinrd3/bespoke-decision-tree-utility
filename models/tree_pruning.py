#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tree Pruning Module for Bespoke Utility
Implements methods for pruning decision trees to reduce overfitting
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable
import pandas as pd
import numpy as np
from copy import deepcopy

from models.node import TreeNode

logger = logging.getLogger(__name__)

class TreePruner:
    """Class for pruning decision trees to reduce overfitting"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize TreePruner
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
    
    def cost_complexity_pruning(self, root: TreeNode, X: pd.DataFrame, y: pd.Series, 
                              sample_weight: Optional[np.ndarray] = None) -> TreeNode:
        """
        Apply cost-complexity pruning to a decision tree
        
        Args:
            root: Root node of the tree
            X: Validation feature DataFrame
            y: Validation target
            sample_weight: Sample weights
            
        Returns:
            Pruned tree root node
        """
        logger.info("Starting cost-complexity pruning")
        
        alphas = np.logspace(-10, 0, num=20)  # log scale from 10^-10 to 10^0=1
        
        if sample_weight is None:
            sample_weight = np.ones(len(X))
        
        best_alpha = 0
        best_score = -float('inf')
        best_tree = None
        
        for alpha in alphas:
            pruned_tree = deepcopy(root)
            
            self._cost_complexity_pruning_with_alpha(pruned_tree, alpha)
            
            score = self._evaluate_tree(pruned_tree, X, y, sample_weight)
            
            logger.debug(f"Alpha: {alpha:.8f}, Score: {score:.5f}")
            
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_tree = deepcopy(pruned_tree)
        
        logger.info(f"Selected best alpha: {best_alpha:.8f} with score: {best_score:.5f}")
        
        return best_tree
    
    def _cost_complexity_pruning_with_alpha(self, root: TreeNode, alpha: float):
        """
        Apply cost-complexity pruning with a specific alpha
        
        Args:
            root: Root node of the tree
            alpha: Complexity parameter
        """
        nodes = self._get_nodes_depth_first(root)
        
        pruned_nodes = 0
        
        for node in nodes:
            if node.node_id == "root" or node.is_terminal:
                continue
            
            leaves = node.get_leaf_nodes()
            
            subtree_error = sum(leaf.impurity * leaf.samples for leaf in leaves)
            total_samples = sum(leaf.samples for leaf in leaves)
            
            if total_samples > 0:
                subtree_error /= total_samples
            
            node_error = node.impurity
            
            num_leaves = len(leaves)
            cost_complexity = (node_error - subtree_error) / max(1, num_leaves - 1)
            
            if cost_complexity < alpha:
                node.is_terminal = True
                node.children = []
                
                pruned_nodes += 1
        
        logger.debug(f"Pruned {pruned_nodes} nodes with alpha={alpha:.8f}")
    
    def _get_nodes_depth_first(self, root: TreeNode) -> List[TreeNode]:
        """
        Get all nodes in depth-first order (deepest first)
        
        Args:
            root: Root node of the tree
            
        Returns:
            List of nodes
        """
        all_nodes = root.get_subtree_nodes()
        
        all_nodes.sort(key=lambda n: -n.depth)
        
        return all_nodes
    
    def _evaluate_tree(self, root: TreeNode, X: pd.DataFrame, y: pd.Series, 
                     sample_weight: np.ndarray) -> float:
        """
        Evaluate a tree on a dataset
        
        Args:
            root: Root node of the tree
            X: Feature DataFrame
            y: Target variable
            sample_weight: Sample weights
            
        Returns:
            Accuracy score
        """
        predictions = root.predict(X)
        
        correct = (predictions == y.to_numpy())
        return np.sum(correct * sample_weight) / np.sum(sample_weight)
    
    def reduced_error_pruning(self, root: TreeNode, X: pd.DataFrame, y: pd.Series, 
                            sample_weight: Optional[np.ndarray] = None) -> TreeNode:
        """
        Apply reduced-error pruning to a decision tree
        
        Args:
            root: Root node of the tree
            X: Validation feature DataFrame
            y: Validation target
            sample_weight: Sample weights
            
        Returns:
            Pruned tree root node
        """
        logger.info("Starting reduced-error pruning")
        
        if sample_weight is None:
            sample_weight = np.ones(len(X))
        
        pruned_tree = deepcopy(root)
        
        initial_score = self._evaluate_tree(pruned_tree, X, y, sample_weight)
        
        nodes = self._get_nodes_depth_first(pruned_tree)
        
        pruned_nodes = 0
        current_score = initial_score
        
        for node in nodes:
            if node.node_id == "root" or node.is_terminal:
                continue
            
            saved_children = node.children
            saved_is_terminal = node.is_terminal
            
            node.is_terminal = True
            node.children = []
            
            pruned_score = self._evaluate_tree(pruned_tree, X, y, sample_weight)
            
            if pruned_score >= current_score:
                current_score = pruned_score
                pruned_nodes += 1
                logger.debug(f"Pruned node {node.node_id}, new score: {pruned_score:.5f}")
            else:
                node.is_terminal = saved_is_terminal
                node.children = saved_children
        
        logger.info(f"Pruned {pruned_nodes} nodes, final score: {current_score:.5f}")
        
        return pruned_tree
    
    def error_based_pruning(self, root: TreeNode, X: pd.DataFrame, y: pd.Series,
                          confidence_level: float = 0.25,
                          sample_weight: Optional[np.ndarray] = None) -> TreeNode:
        """
        Apply error-based pruning (C4.5/J48 style) to a decision tree
        
        Args:
            root: Root node of the tree
            X: Validation feature DataFrame
            y: Validation target
            confidence_level: Confidence level for error estimate (0-1)
            sample_weight: Sample weights
            
        Returns:
            Pruned tree root node
        """
        logger.info(f"Starting error-based pruning with confidence {confidence_level}")
        
        if sample_weight is None:
            sample_weight = np.ones(len(X))
        
        pruned_tree = deepcopy(root)
        
        nodes = self._get_nodes_depth_first(pruned_tree)
        
        pruned_nodes = 0
        
        for node in nodes:
            if node.node_id == "root" or node.is_terminal:
                continue
            
            subtree_error = self._calculate_subtree_error(node, X, y, sample_weight, confidence_level)
            node_error = self._calculate_node_error(node, confidence_level)
            
            if node_error <= subtree_error:
                node.is_terminal = True
                node.children = []
                pruned_nodes += 1
                logger.debug(f"Pruned node {node.node_id}, node error: {node_error:.5f}, subtree error: {subtree_error:.5f}")
        
        logger.info(f"Pruned {pruned_nodes} nodes with error-based pruning")
        
        return pruned_tree
    
    def _calculate_subtree_error(self, node: TreeNode, X: pd.DataFrame, y: pd.Series,
                              sample_weight: np.ndarray, confidence_level: float) -> float:
        """
        Calculate error estimate for a subtree
        
        Args:
            node: Node to evaluate
            X: Feature DataFrame
            y: Target variable
            sample_weight: Sample weights
            confidence_level: Confidence level for error estimate
            
        Returns:
            Error estimate
        """
        leaves = node.get_leaf_nodes()
        
        node_indices = self._get_node_sample_indices(X, node)
        
        if not node_indices:
            return 1.0  # Default to maximum error
        
        total_error = 0.0
        total_weight = 0.0
        
        for leaf in leaves:
            leaf_indices = self._get_node_sample_indices(X, leaf)
            
            if not leaf_indices:
                continue
            
            leaf_y = y.iloc[leaf_indices]
            leaf_weights = sample_weight[leaf_indices]
            
            predictions = np.full(len(leaf_y), leaf.majority_class)
            
            errors = (predictions != leaf_y.to_numpy())
            error_rate = np.sum(errors * leaf_weights) / np.sum(leaf_weights)
            
            n = len(leaf_indices)
            if n > 0:
                conf_factor = self._confidence_factor(n, confidence_level)
                error_estimate = error_rate + conf_factor
                
                leaf_weight = np.sum(leaf_weights)
                total_error += error_estimate * leaf_weight
                total_weight += leaf_weight
        
        if total_weight > 0:
            return total_error / total_weight
        else:
            return 1.0  # Default to maximum error
    
    def _calculate_node_error(self, node: TreeNode, confidence_level: float) -> float:
        """
        Calculate error estimate for a node as a leaf
        
        Args:
            node: Node to evaluate
            confidence_level: Confidence level for error estimate
            
        Returns:
            Error estimate
        """
        majority_count = max(node.class_counts.values()) if node.class_counts else 0
        total_count = sum(node.class_counts.values()) if node.class_counts else 0
        
        if total_count > 0:
            error_rate = 1.0 - (majority_count / total_count)
        else:
            error_rate = 1.0
        
        n = node.samples
        if n > 0:
            conf_factor = self._confidence_factor(n, confidence_level)
            return error_rate + conf_factor
        else:
            return 1.0  # Default to maximum error
    
    def _confidence_factor(self, n: int, confidence_level: float) -> float:
        """
        Calculate confidence factor for error estimate
        
        Args:
            n: Number of samples
            confidence_level: Confidence level (0-1)
            
        Returns:
            Confidence factor
        """
        z_value = self._z_value(confidence_level)
        
        if n > 0:
            return z_value * np.sqrt(0.25 / n)  # 0.25 = p*(1-p) for p=0.5
        else:
            return 1.0
    
    def _z_value(self, confidence_level: float) -> float:
        """
        Get z-value for confidence level
        
        Args:
            confidence_level: Confidence level (0-1)
            
        Returns:
            z-value
        """
        if confidence_level >= 0.999:
            return 3.291  # 99.9%
        elif confidence_level >= 0.99:
            return 2.576  # 99%
        elif confidence_level >= 0.95:
            return 1.96   # 95%
        elif confidence_level >= 0.90:
            return 1.645  # 90%
        elif confidence_level >= 0.80:
            return 1.282  # 80%
        elif confidence_level >= 0.50:
            return 0.674  # 50%
        else:
            return 0.0    # Below 50%
    
    def _get_node_sample_indices(self, X: pd.DataFrame, node: TreeNode) -> List[int]:
        """
        Get indices of samples that reach this node
        
        Args:
            X: Feature DataFrame
            node: Node to get samples for
            
        Returns:
            List of sample indices
        """
        if node.node_id == "root" or node.parent is None:
            return list(range(len(X)))
        
        path = []
        current = node
        while current.parent is not None:
            path.append((current.parent, current))
            current = current.parent
        
        path.reverse()
        
        indices = list(range(len(X)))
        
        for parent, child in path:
            if not parent.split_feature or parent.split_feature not in X.columns:
                continue
            
            child_idx = parent.children.index(child)
            
            feature = parent.split_feature
            
            if parent.split_type == 'numeric':
                threshold = parent.split_value
                
                if child_idx == 0:  # Left child (<=)
                    indices = [i for i in indices if X[feature].iloc[i] <= threshold]
                else:  # Right child (>)
                    indices = [i for i in indices if X[feature].iloc[i] > threshold]
                    
            elif parent.split_type == 'categorical':
                categories = [cat for cat, idx in parent.split_categories.items() if idx == child_idx]
                indices = [i for i in indices if X[feature].iloc[i] in categories]
        
        return indices
    
    def minimal_cost_complexity_pruning(self, root: TreeNode) -> List[Tuple[float, TreeNode]]:
        """
        Generate a pruning path with minimal cost-complexity pruning
        
        Args:
            root: Root node of the tree
            
        Returns:
            List of (alpha, pruned_tree) tuples in increasing order of alpha
        """
        logger.info("Generating minimal cost-complexity pruning path")
        
        tree = deepcopy(root)
        
        pruning_path = [(0.0, deepcopy(tree))]  # Start with unpruned tree
        
        while self._has_non_terminal_non_root_nodes(tree):
            min_alpha, nodes_to_prune = self._find_min_cost_complexity_nodes(tree)
            
            for node in nodes_to_prune:
                node.is_terminal = True
                node.children = []
            
            pruning_path.append((min_alpha, deepcopy(tree)))
            
            logger.debug(f"Pruned {len(nodes_to_prune)} nodes with alpha={min_alpha:.8f}")
        
        logger.info(f"Generated pruning path with {len(pruning_path)} trees")
        
        return pruning_path
    
    def _has_non_terminal_non_root_nodes(self, root: TreeNode) -> bool:
        """
        Check if tree has non-terminal nodes other than the root
        
        Args:
            root: Root node of the tree
            
        Returns:
            True if tree has non-terminal non-root nodes
        """
        nodes = root.get_subtree_nodes()
        for node in nodes:
            if node.node_id != "root" and not node.is_terminal:
                return True
        return False
    
    def _find_min_cost_complexity_nodes(self, root: TreeNode) -> Tuple[float, List[TreeNode]]:
        """
        Find the nodes with minimum cost-complexity
        
        Args:
            root: Root node of the tree
            
        Returns:
            Tuple of (minimum alpha, list of nodes to prune)
        """
        nodes = [n for n in root.get_subtree_nodes() if not n.is_terminal and n.node_id != "root"]
        
        if not nodes:
            return float('inf'), []
        
        min_alpha = float('inf')
        min_nodes = []
        
        for node in nodes:
            leaves = node.get_leaf_nodes()
            
            subtree_error = sum(leaf.impurity * leaf.samples for leaf in leaves)
            total_samples = sum(leaf.samples for leaf in leaves)
            
            if total_samples > 0:
                subtree_error /= total_samples
            
            node_error = node.impurity
            
            num_leaves = len(leaves)
            
            if num_leaves > 1:
                alpha = (node_error - subtree_error) / (num_leaves - 1)
            else:
                alpha = float('inf')
            
            if alpha < min_alpha:
                min_alpha = alpha
                min_nodes = [node]
            elif alpha == min_alpha:
                min_nodes.append(node)
        
        return min_alpha, min_nodes
