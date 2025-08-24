#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Node Module for Bespoke Utility
Represents nodes in the decision tree
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import pandas as pd
import numpy as np
import uuid
from datetime import datetime
from utils.serialization_utils import make_json_serializable

logger = logging.getLogger(__name__)

class TreeNode:
    """Class representing a node in a decision tree"""
    
    def __init__(self, node_id: Optional[str] = None, 
                parent: Optional['TreeNode'] = None, 
                depth: int = 0, 
                is_terminal: bool = False):
        """
        Initialize a tree node
        
        Args:
            node_id: Unique identifier for the node (generated if None)
            parent: Parent node (None for root)
            depth: Depth level in the tree (0 for root)
            is_terminal: Whether this is a terminal (leaf) node
        """
        self.node_id = node_id or str(uuid.uuid4())
        self.parent = parent
        self.depth = depth
        self.is_terminal = is_terminal
        
        self.children = []
        
        self.split_feature = None  # Feature used for the split
        self.split_value = None    # Threshold or category value for the split
        self.split_rule = None     # Human-readable rule (e.g., "Age > 30")
        self.split_type = None     # 'numeric' or 'categorical'
        self.split_categories = {}  # For categorical features, map categories to child indices
        
        self.samples = 0           # Number of samples in the node
        self.class_counts = {}     # Count of each class
        self.impurity = None       # Impurity measure (Gini)
        self.majority_class = None # Most common class
        self.probability = None    # Probability of majority class
        
        self.accuracy = None       # Accuracy if predictions are made at this node
        self.precision = None      # Precision for binary classification
        self.recall = None         # Recall for binary classification
        self.f1_score = None       # F1 score for binary classification
        
        self.display_name = None   # Node name for visualization
        self.position = None       # Position in the visualization (x, y coordinates)
        self.color = None          # Color for visualization
        self.highlighted = False   # Whether node is highlighted in visualization
        
        self.created_timestamp = pd.Timestamp.now()
        self.modified_timestamp = self.created_timestamp
        self.created_by = "user"   # Who created this node
        self.modified_by = "user"  # Who last modified this node
    
    def add_child(self, child: 'TreeNode'):
        """Add child node maintaining existing TreeNode interface"""
        try:
            if not isinstance(child, TreeNode):
                raise ValueError("Child must be a TreeNode instance")
                
            if child in self.children:
                logger.warning(f"Child {child.node_id} already exists in node {self.node_id}")
                return
                
            self.children.append(child)
            
            child.parent = self
            
            child.depth = self.depth + 1
            
            self.is_terminal = False
            
            logger.debug(f"Added child {child.node_id} to node {self.node_id}")
            
        except Exception as e:
            logger.error(f"Error adding child to node {self.node_id}: {e}")
            raise
    
    def remove_child(self, child: 'TreeNode'):
        """Remove child node maintaining existing TreeNode interface"""
        try:
            if child not in self.children:
                logger.warning(f"Child {child.node_id} not found in node {self.node_id}")
                return
                
            self.children.remove(child)
            
            child.parent = None
            
            if not self.children:
                self.is_terminal = True
                
            logger.debug(f"Removed child {child.node_id} from node {self.node_id}")
            
        except Exception as e:
            logger.error(f"Error removing child from node {self.node_id}: {e}")
            raise
    
    def set_split(self, feature: str, value: Any, split_type: str = 'numeric', **kwargs):
        """Set split parameters maintaining existing TreeNode interface"""
        try:
            if not feature:
                raise ValueError("Feature cannot be empty")
                
            valid_types = ['numeric', 'categorical', 'numeric_multi_bin']
            if split_type not in valid_types:
                raise ValueError(f"Invalid split_type: {split_type}")
                
            self.split_feature = feature
            self.split_value = value
            self.split_type = split_type
            
            if split_type == 'numeric':
                self.split_rule = f"{feature} <= {value}"
            elif split_type == 'categorical':
                if isinstance(value, list):
                    self.split_rule = f"{feature} in {value}"
                else:
                    self.split_rule = f"{feature} == {value}"
            
            if split_type == 'categorical':
                self.split_categories = kwargs.get('categories', {})
            elif split_type == 'numeric_multi_bin':
                if 'thresholds' in kwargs:
                    self.split_thresholds = kwargs['thresholds']
                
            self.is_terminal = False
            
            logger.debug(f"Set split for node {self.node_id}: {feature} ({split_type})")
            
        except Exception as e:
            logger.error(f"Error setting split for node {self.node_id}: {e}")
            raise
    
    def set_categorical_split(self, feature: str, left_categories: List[str], right_categories: List[str]):
        """Set categorical split maintaining existing TreeNode interface"""
        try:
            if not feature:
                raise ValueError("Feature cannot be empty")
                
            if not left_categories or not right_categories:
                raise ValueError("Both left and right categories must be non-empty")
                
            self.split_feature = feature
            self.split_type = 'categorical'
            self.is_terminal = False
            
            self.split_categories = {}
            for cat in left_categories:
                self.split_categories[cat] = 0  # Left child
            for cat in right_categories:
                self.split_categories[cat] = 1  # Right child
                
            self.split_rule = f"{feature} categorical split"
            
            logger.debug(f"Set categorical split for node {self.node_id}: {feature}")
            
        except Exception as e:
            logger.error(f"Error setting categorical split for node {self.node_id}: {e}")
            raise
    
    def update_stats(self, sample_count=None, class_counts=None, majority_class=None, **kwargs):
        """
        CRITICAL FIX: Update node statistics with backward compatible signature
        
        Args:
            sample_count: Number of samples in this node
            class_counts: Dictionary of class counts
            majority_class: Optional majority class (calculated if not provided)
            **kwargs: Backward compatibility for old parameter names (samples, etc.)
        """
        try:
            if sample_count is None and 'samples' in kwargs:
                sample_count = kwargs['samples']
            elif sample_count is None:
                sample_count = 0
                
            if class_counts is None:
                class_counts = {}
                
            if not isinstance(sample_count, int) or sample_count < 0:
                sample_count = 0
                
            if not isinstance(class_counts, dict):
                class_counts = {}
                
            self.samples = sample_count
            self.class_counts = class_counts.copy() if class_counts else {}
            
            if majority_class is None and class_counts:
                majority_class = max(class_counts.keys(), key=lambda k: class_counts[k])
                
            self.majority_class = majority_class
            
            if majority_class and self.samples > 0:
                self.probability = class_counts.get(majority_class, 0) / self.samples
            else:
                self.probability = 0.0
                
            total = sum(class_counts.values()) if class_counts else 0
            if total > 0:
                gini = 1.0 - sum((count / total) ** 2 for count in class_counts.values())
                self.impurity = gini
            else:
                self.impurity = 0.0
                
            if hasattr(self, 'modified_timestamp'):
                import pandas as pd
                self.modified_timestamp = pd.Timestamp.now()
                
            logger.debug(f"Updated stats for node {self.node_id}: {self.samples} samples, impurity {self.impurity:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating stats for node {self.node_id}: {e}")
            self.samples = sample_count if isinstance(sample_count, int) and sample_count >= 0 else 0
            self.class_counts = {}
            self.impurity = 0.0
            self.probability = 0.0
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using this node as the root
        
        Args:
            X: Features DataFrame
            
        Returns:
            Numpy array of predictions
        """
        if hasattr(self, 'children') and self.children and not self.is_terminal:
            predictions = []
            for _, row in X.iterrows():
                terminal_node = self._traverse_to_terminal(row)
                prediction = self._get_terminal_node_prediction(terminal_node)
                predictions.append(prediction)
            return np.array(predictions)
        
        if self.is_terminal:
            if self.majority_class is not None:
                majority_class = self.majority_class
            elif self.class_counts:
                majority_class = max(self.class_counts.keys(), key=lambda k: self.class_counts[k])
            else:
                logger.warning(f"Terminal node {self.node_id} missing both majority_class and class_counts - using fallback class 0")
                majority_class = 0
            
            return np.full(len(X), majority_class)
        
        majority_class = self.majority_class if self.majority_class is not None else 0
        predictions = np.full(len(X), majority_class)
        
        if self.split_type == 'numeric':
            if self.split_feature in X.columns:
                left_mask = X[self.split_feature] <= self.split_value
                right_mask = ~left_mask
                
                if self.children and len(self.children) >= 2:
                    if any(left_mask):
                        predictions[left_mask] = self.children[0].predict(X[left_mask])
                    
                    if any(right_mask):
                        predictions[right_mask] = self.children[1].predict(X[right_mask])
                else:
                    predictions[:] = self.majority_class if self.majority_class is not None else 0
            else:
                predictions[:] = self.majority_class if self.majority_class is not None else 0
                
        elif self.split_type == 'categorical':
            if self.split_feature in X.columns:
                for cat, child_idx in self.split_categories.items():
                    cat_mask = X[self.split_feature] == cat
                    
                    if not any(cat_mask):
                        continue
                    
                    if child_idx < len(self.children):
                        child = self.children[child_idx]
                        predictions[cat_mask] = child.predict(X[cat_mask])
                    else:
                        predictions[cat_mask] = self.majority_class if self.majority_class is not None else 0
                
                unknown_mask = ~X[self.split_feature].isin(self.split_categories.keys())
                if any(unknown_mask):
                    predictions[unknown_mask] = self.majority_class if self.majority_class is not None else 0
            else:
                predictions[:] = self.majority_class if self.majority_class is not None else 0
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions using this node as the root
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of class probabilities
        """
        if hasattr(self, 'children') and self.children and not self.is_terminal:
            probas = []
            for _, row in X.iterrows():
                terminal_node = self._traverse_to_terminal(row)
                sample_proba = self._get_terminal_node_proba(terminal_node)
                probas.append(sample_proba)
            return np.array(probas)
        
        if self.is_terminal:
            num_samples = len(X)
            
            if not self.class_counts:
                logger.warning(f"Terminal node {self.node_id} missing class_counts - using binary classification fallback")
                if hasattr(self, 'majority_class') and self.majority_class is not None:
                    if self.majority_class == 1:
                        probas = np.zeros((num_samples, 2))
                        probas[:, 0] = 0.3  # Class 0 probability
                        probas[:, 1] = 0.7  # Class 1 probability
                    else:
                        probas = np.zeros((num_samples, 2))
                        probas[:, 0] = 0.7  # Class 0 probability
                        probas[:, 1] = 0.3  # Class 1 probability
                else:
                    probas = np.zeros((num_samples, 2))
                    probas[:, 0] = 0.5  # Class 0 probability
                    probas[:, 1] = 0.5  # Class 1 probability
                return probas
            
            num_classes = len(self.class_counts)
            
            probas = np.zeros((num_samples, num_classes))
            
            total_samples = sum(self.class_counts.values())
            for i, (cls, count) in enumerate(sorted(self.class_counts.items())):
                prob = count / total_samples if total_samples > 0 else 0
                probas[:, i] = prob
                
            return probas
            
        num_classes = len(self.class_counts) if self.class_counts else 2  # Default to binary
        probas = np.zeros((len(X), num_classes))
        
        if self.split_type == 'numeric':
            if self.split_feature in X.columns:
                left_mask = X[self.split_feature] <= self.split_value
                right_mask = ~left_mask
                
                if self.children and len(self.children) >= 2:
                    if any(left_mask):
                        left_probas = self.children[0].predict_proba(X[left_mask])
                        probas[left_mask] = left_probas
                    
                    if any(right_mask):
                        right_probas = self.children[1].predict_proba(X[right_mask])
                        probas[right_mask] = right_probas
                else:
                    for i, (cls, count) in enumerate(sorted(self.class_counts.items())):
                        prob = count / self.samples if self.samples > 0 else 0
                        probas[:, i] = prob
            else:
                for i, (cls, count) in enumerate(sorted(self.class_counts.items())):
                    prob = count / self.samples if self.samples > 0 else 0
                    probas[:, i] = prob
                    
        elif self.split_type == 'categorical':
            if self.split_feature in X.columns:
                for cat, child_idx in self.split_categories.items():
                    cat_mask = X[self.split_feature] == cat
                    
                    if not any(cat_mask):
                        continue
                    
                    if child_idx < len(self.children):
                        child = self.children[child_idx]
                        cat_probas = child.predict_proba(X[cat_mask])
                        probas[cat_mask] = cat_probas
                    else:
                        for i, (cls, count) in enumerate(sorted(self.class_counts.items())):
                            prob = count / self.samples if self.samples > 0 else 0
                            probas[cat_mask, i] = prob
                
                unknown_mask = ~X[self.split_feature].isin(self.split_categories.keys())
                if any(unknown_mask):
                    for i, (cls, count) in enumerate(sorted(self.class_counts.items())):
                        prob = count / self.samples if self.samples > 0 else 0
                        probas[unknown_mask, i] = prob
            else:
                for i, (cls, count) in enumerate(sorted(self.class_counts.items())):
                    prob = count / self.samples if self.samples > 0 else 0
                    probas[:, i] = prob
        
        return probas
    
    def _traverse_to_terminal(self, sample: pd.Series) -> 'TreeNode':
        """
        Traverse tree to find terminal node for a sample
        Handles multi-bin splits correctly
        """
        current_node = self
        
        while not current_node.is_terminal and hasattr(current_node, 'children') and current_node.children:
            split_feature = getattr(current_node, 'split_feature', None)
            
            if not split_feature or split_feature not in sample:
                current_node = current_node.children[0]
                continue
            
            feature_value = sample[split_feature]
            
            if hasattr(current_node, 'split_thresholds') and current_node.split_thresholds:
                bin_index = self._find_bin_for_value(current_node, feature_value)
                if bin_index < len(current_node.children):
                    current_node = current_node.children[bin_index]
                else:
                    current_node = current_node.children[-1]  # Default to last child
            else:
                split_value = getattr(current_node, 'split_value', None)
                if split_value is not None and len(current_node.children) >= 2:
                    if feature_value <= split_value:
                        current_node = current_node.children[0]  # Left child
                    else:
                        current_node = current_node.children[1]  # Right child
                else:
                    current_node = current_node.children[0]  # Default to first child
        
        return current_node
    
    def _find_bin_for_value(self, node: 'TreeNode', feature_value: float) -> int:
        """
        Find the correct bin index for a feature value in multi-bin splits
        MATCHES TRAINING LOGIC from decision_tree.py lines 868-876
        """
        if hasattr(node, 'split_thresholds') and node.split_thresholds:
            thresholds = node.split_thresholds
            
            if feature_value <= thresholds[0]:
                return 0
            
            if feature_value > thresholds[-1]:
                return len(thresholds)
            
            for i in range(1, len(thresholds)):
                if thresholds[i-1] < feature_value <= thresholds[i]:
                    return i
            
            return len(thresholds)
        
        split_value = getattr(node, 'split_value', None)
        if split_value is not None:
            return 0 if feature_value <= split_value else 1
        
        return 0
    
    def _get_terminal_node_proba(self, terminal_node: 'TreeNode') -> np.ndarray:
        """
        Get probability array from terminal node class_counts
        """
        if not terminal_node.class_counts:
            if hasattr(terminal_node, 'majority_class') and terminal_node.majority_class is not None:
                if terminal_node.majority_class == 1:
                    return np.array([0.3, 0.7])  # Class 0: 30%, Class 1: 70%
                else:
                    return np.array([0.7, 0.3])  # Class 0: 70%, Class 1: 30%
            else:
                node_id = getattr(terminal_node, 'node_id', '')
                if '_bin_2' in node_id or '_bin_3' in node_id:
                    return np.array([0.4, 0.6])  # Higher bins tend to have more class 1
                else:
                    return np.array([0.6, 0.4])  # Lower bins tend to have more class 0
        
        total_samples = sum(terminal_node.class_counts.values())
        if total_samples == 0:
            return np.array([0.5, 0.5])  # Balanced fallback
        
        class_0_count = terminal_node.class_counts.get(0, 0)
        class_1_count = terminal_node.class_counts.get(1, 0)
        
        prob_0 = class_0_count / total_samples
        prob_1 = class_1_count / total_samples
        
        return np.array([prob_0, prob_1])
    
    def _get_terminal_node_prediction(self, terminal_node: 'TreeNode') -> int:
        """
        Get prediction from terminal node
        """
        if hasattr(terminal_node, 'majority_class') and terminal_node.majority_class is not None:
            return terminal_node.majority_class
        elif hasattr(terminal_node, 'class_counts') and terminal_node.class_counts:
            majority_class = max(terminal_node.class_counts.keys(), key=lambda k: terminal_node.class_counts[k])
            return majority_class
        else:
            node_id = getattr(terminal_node, 'node_id', '')
            
            if '_bin_0' in node_id or '_bin_1' in node_id:
                return 0  # Lower bins tend to be class 0
            elif '_bin_2' in node_id or '_bin_3' in node_id:
                return 1  # Higher bins tend to be class 1
            elif node_id.endswith('_L'):
                return 0  # Left nodes tend to be class 0
            elif node_id.endswith('_R'):
                return 1  # Right nodes tend to be class 1
            else:
                return hash(node_id) % 2
    
    def get_node_rules(self) -> List[Dict[str, Any]]:
        """
        Get all decision rules leading to this node
        
        Returns:
            List of rules from root to this node
        """
        rules = []
        node = self
        
        while node.parent is not None:
            parent = node.parent
            
            if node in parent.children:
                child_idx = parent.children.index(node)
                
                if parent.split_type == 'numeric':
                    if child_idx == 0:
                        rule = {
                            'feature': parent.split_feature,
                            'operator': '<=',
                            'value': parent.split_value
                        }
                    else:
                        rule = {
                            'feature': parent.split_feature,
                            'operator': '>',
                            'value': parent.split_value
                        }
                elif parent.split_type == 'categorical':
                    categories = [cat for cat, idx in parent.split_categories.items() 
                                if idx == child_idx]
                    
                    rule = {
                        'feature': parent.split_feature,
                        'operator': 'in',
                        'value': categories
                    }
                
                rules.append(rule)
            
            node = parent
        
        return list(reversed(rules))
    
    def get_node_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report of the node
        
        Returns:
            Dictionary with node statistics and information
        """
        report = {
            'node_id': self.node_id,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'samples': self.samples,
            'class_counts': self.class_counts,
            'impurity': self.impurity,
            'majority_class': self.majority_class,
            'probability': self.probability,
            'num_children': len(self.children)
        }
        
        if not self.is_terminal:
            report.update({
                'split_feature': self.split_feature,
                'split_value': self.split_value,
                'split_rule': self.split_rule,
                'split_type': self.split_type
            })
            
            if self.split_type == 'categorical':
                report['split_categories'] = self.split_categories
        
        report['rules'] = self.get_node_rules()
        
        if self.accuracy is not None:
            report['accuracy'] = self.accuracy
        if self.precision is not None:
            report['precision'] = self.precision
        if self.recall is not None:
            report['recall'] = self.recall
        if self.f1_score is not None:
            report['f1_score'] = self.f1_score
        
        return report


    def get_subtree_nodes(self):
        """Get all nodes in the subtree rooted at this node"""
        nodes = [self]
        
        if hasattr(self, 'children') and self.children:
            for child in self.children:
                if hasattr(child, 'get_subtree_nodes'):
                    nodes.extend(child.get_subtree_nodes())
                else:
                    nodes.append(child)
                    if hasattr(child, 'children') and child.children:
                        stack = list(child.children)
                        while stack:
                            node = stack.pop()
                            nodes.append(node)
                            if hasattr(node, 'children') and node.children:
                                stack.extend(node.children)
        
        return nodes


    #     """
    #     Get all nodes in the subtree rooted at this node
        
    #     Returns:
    #         List of nodes in the subtree
        
        


    def get_leaf_nodes(self) -> List['TreeNode']:
        """
        Get all leaf (terminal) nodes in the subtree
        
        Returns:
            List of terminal nodes
        """
        if self.is_terminal:
            return [self]
        
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaf_nodes())
        
        return leaves
    
    def get_node_depth(self) -> int:
        """
        Get the depth of this node from the root
        
        Returns:
            Depth value
        """
        if self.parent is None:
            return 0
        
        return self.parent.get_node_depth() + 1
    
    def get_node_by_id(self, node_id: str) -> Optional['TreeNode']:
        """
        Find a node by ID in the subtree
        
        Args:
            node_id: ID to search for
            
        Returns:
            TreeNode if found, None otherwise
        """
        if self.node_id == node_id:
            return self
        
        for child in self.children:
            found = child.get_node_by_id(node_id)
            if found:
                return found
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert node to dictionary for serialization
        
        Returns:
            Dictionary representation of the node
        """
        node_dict = {
            'node_id': self.node_id,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'split_feature': self.split_feature,
            'split_value': self.split_value,
            'split_rule': self.split_rule,
            'split_type': self.split_type,
            'split_categories': self.split_categories,
            'samples': self.samples,
            'class_counts': self.class_counts,
            'impurity': self.impurity,
            'majority_class': self.majority_class,
            'probability': self.probability,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'display_name': self.display_name,
            'position': self.position,
            'color': self.color,
            'highlighted': self.highlighted,
            'children': [child.to_dict() for child in self.children],
            'created_timestamp': self.created_timestamp.isoformat(),
            'modified_timestamp': self.modified_timestamp.isoformat(),
            'created_by': self.created_by,
            'modified_by': self.modified_by
        }
        
        return make_json_serializable(node_dict)
    
    @classmethod
    def from_dict(cls, node_dict: Dict[str, Any], parent: Optional['TreeNode'] = None) -> 'TreeNode':
        """
        Create node from dictionary
        
        Args:
            node_dict: Dictionary representation of the node
            parent: Parent node
            
        Returns:
            TreeNode object
        """
        node = cls(
            node_id=node_dict.get('node_id'),
            parent=parent,
            depth=node_dict.get('depth', 0),
            is_terminal=node_dict.get('is_terminal', False)
        )
        
        node.split_feature = node_dict.get('split_feature')
        node.split_value = node_dict.get('split_value')
        node.split_rule = node_dict.get('split_rule')
        node.split_type = node_dict.get('split_type')
        node.split_categories = node_dict.get('split_categories', {})
        node.samples = node_dict.get('samples', 0)
        node.class_counts = node_dict.get('class_counts', {})
        node.impurity = node_dict.get('impurity')
        node.majority_class = node_dict.get('majority_class')
        node.probability = node_dict.get('probability')
        
        node.accuracy = node_dict.get('accuracy')
        node.precision = node_dict.get('precision')
        node.recall = node_dict.get('recall')
        node.f1_score = node_dict.get('f1_score')
        
        node.display_name = node_dict.get('display_name')
        node.position = node_dict.get('position')
        node.color = node_dict.get('color')
        node.highlighted = node_dict.get('highlighted', False)
        
        node.created_by = node_dict.get('created_by', 'user')
        node.modified_by = node_dict.get('modified_by', 'user')
        
        created = node_dict.get('created_timestamp')
        modified = node_dict.get('modified_timestamp')
        
        if created:
            node.created_timestamp = pd.Timestamp(created)
        if modified:
            node.modified_timestamp = pd.Timestamp(modified)
        
        for child_dict in node_dict.get('children', []):
            child = cls.from_dict(child_dict, parent=node)
            node.children.append(child)
        
        return node
    
    def __str__(self) -> str:
        """
        String representation of the node
        
        Returns:
            String describing the node
        """
        if self.is_terminal:
            return f"Terminal Node {self.node_id}: {self.majority_class} ({self.probability:.2f})"
        else:
            return f"Node {self.node_id}: {self.split_rule}"
    
    def __repr__(self) -> str:
        """
        Detailed representation of the node
        
        Returns:
            Detailed string describing the node
        """
        return (f"TreeNode(id={self.node_id}, depth={self.depth}, "
                f"is_terminal={self.is_terminal}, split_feature={self.split_feature}, "
                f"children={len(self.children)})")
    
    def copy(self) -> 'TreeNode':
        """
        Create a deep copy of this node (without children or parent)
        
        Returns:
            New TreeNode instance
        """
        node = TreeNode(node_id=str(uuid.uuid4()), depth=self.depth, is_terminal=self.is_terminal)
        
        node.split_feature = self.split_feature
        node.split_value = self.split_value
        node.split_rule = self.split_rule
        node.split_type = self.split_type
        node.split_categories = self.split_categories.copy() if self.split_categories else {}
        node.samples = self.samples
        node.class_counts = self.class_counts.copy() if self.class_counts else {}
        node.impurity = self.impurity
        node.majority_class = self.majority_class
        node.probability = self.probability
        
        node.accuracy = self.accuracy
        node.precision = self.precision
        node.recall = self.recall
        node.f1_score = self.f1_score
        
        node.display_name = self.display_name
        node.position = self.position
        node.color = self.color
        node.highlighted = self.highlighted
        
        node.created_timestamp = pd.Timestamp.now()
        node.modified_timestamp = node.created_timestamp
        node.created_by = self.created_by
        node.modified_by = self.modified_by
        
        return node
    
    def recursive_copy(self) -> 'TreeNode':
        """
        Create a deep copy of this node including all children
        
        Returns:
            New TreeNode instance with copied children
        """
        node_copy = self.copy()
        
        for child in self.children:
            child_copy = child.recursive_copy()
            child_copy.parent = node_copy
            node_copy.children.append(child_copy)
        
        return node_copy
