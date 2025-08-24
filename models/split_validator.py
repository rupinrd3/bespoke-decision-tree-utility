#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Split Validator Module for Bespoke Utility
Comprehensive validation of splits and tree state consistency
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
import pandas as pd
import numpy as np

from models.node import TreeNode
from models.split_configuration import SplitConfiguration, ValidationResult

logger = logging.getLogger(__name__)


class SplitValidator:
    """Enhanced validator for split configurations maintaining existing interface"""
    
    def __init__(self):
        """Initialize validator with existing VALID_SPLIT_TYPES"""
        self.validation_cache = {}
        self.VALID_SPLIT_TYPES = {
            'numerical', 'numeric',  # Both variants supported for backward compatibility
            'categorical', 
            'numerical_multi_bin', 'numeric_multi_bin',  # Both variants supported
            'categorical_multi_bin'
        }
        
    def validate_split_config(self, config, data=None):
        """Enhanced split configuration validation maintaining existing interface"""
        result = type('ValidationResult', (), {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'add_error': lambda self, error: (self.errors.append(error), setattr(self, 'is_valid', False)),
            'add_warning': lambda self, warning: self.warnings.append(warning)
        })()
        
        try:
            if hasattr(config, 'split_type'):
                split_type = config.split_type
                feature = getattr(config, 'feature', None)
            elif isinstance(config, dict):
                split_type = config.get('split_type')
                feature = config.get('feature')
            else:
                result.add_error("Invalid configuration format")
                return result
                
            if not split_type:
                result.add_error("Split type not specified")
                return result
                
            normalized_type = split_type
            if split_type in ['numeric', 'numerical']:
                normalized_type = 'numerical'
            elif split_type in ['numeric_multi_bin', 'numerical_multi_bin']:
                normalized_type = 'numerical_multi_bin'
                
            if normalized_type not in self.VALID_SPLIT_TYPES:
                result.add_error(f"Invalid split_type: {split_type}")
                
            if not feature:
                result.add_error("Feature name not specified")
                return result
                
            if data is not None:
                if feature not in data.columns:
                    result.add_error(f"Feature '{feature}' not found in data")
                    return result
                    
                feature_data = data[feature]
                
                if feature_data.isnull().all():
                    result.add_error(f"Feature '{feature}' has all missing values")
                elif feature_data.isnull().any():
                    null_count = feature_data.isnull().sum()
                    result.add_warning(f"Feature '{feature}' has {null_count} missing values")
                    
                if split_type in ['numeric', 'numerical']:
                    if not pd.api.types.is_numeric_dtype(feature_data):
                        result.add_error(f"Numeric split on non-numeric feature '{feature}'")
                        
                    threshold = None
                    if hasattr(config, 'threshold'):
                        threshold = config.threshold
                    elif isinstance(config, dict):
                        threshold = config.get('threshold', config.get('split_value'))
                        
                    if threshold is None:
                        result.add_error("Numeric split requires threshold value")
                    else:
                        try:
                            float(threshold)
                        except (ValueError, TypeError):
                            result.add_error(f"Invalid threshold value: {threshold}")
                            
                elif split_type in ['categorical', 'categorical_multi_bin']:
                    if hasattr(config, 'left_categories'):
                        left_cats = config.left_categories
                    elif isinstance(config, dict):
                        left_cats = config.get('left_categories')
                    else:
                        left_cats = None
                        
                    has_split_categories = (hasattr(config, 'split_categories') or 
                                          (isinstance(config, dict) and 'split_categories' in config))
                    
                    has_category_mapping = (hasattr(config, 'category_mapping') or 
                                          (isinstance(config, dict) and 'category_mapping' in config))
                    
                    if split_type == 'categorical_multi_bin':
                        if not has_split_categories and not has_category_mapping:
                            result.add_error("Categorical multi-bin split requires split_categories or category_mapping")
                    else:
                        if not left_cats and not has_split_categories and not has_category_mapping:
                            result.add_error("Categorical split requires left_categories, split_categories, or category_mapping")
                        
            return result
            
        except Exception as e:
            result.add_error(f"Exception during validation: {e}")
            return result
        
        
                
                
            
                
    
    def validate_node_state(self, node: TreeNode) -> ValidationResult:
        """Validate the state of a tree node"""
        result = ValidationResult(is_valid=True)
        
        try:
            if not node.node_id:
                result.add_error("Node ID cannot be empty")
                
            if node.depth < 0:
                result.add_error(f"Node depth cannot be negative: {node.depth}")
                
            if node.is_terminal and len(node.children) > 0:
                result.add_error(f"Terminal node {node.node_id} has children")
                
            if not node.is_terminal and len(node.children) == 0:
                result.add_warning(f"Non-terminal node {node.node_id} has no children")
                
            if not node.is_terminal and len(node.children) > 0:
                if not node.split_feature:
                    result.add_warning(f"Non-terminal node {node.node_id} has no split feature")
                    
                if hasattr(node, 'split_type') and node.split_type and node.split_type not in self.VALID_SPLIT_TYPES:
                    result.add_error(f"Invalid split_type: {node.split_type}")
                    
                if hasattr(node, 'split_type') and node.split_type in ['numeric', 'numerical'] and not hasattr(node, 'split_value'):
                    result.add_warning(f"Numeric split node {node.node_id} has no split value")
                    
                if hasattr(node, 'split_type') and node.split_type == 'categorical' and not hasattr(node, 'split_categories'):
                    result.add_warning(f"Categorical split node {node.node_id} has no split categories")
                    
            for child in node.children:
                if child.parent != node:
                    result.add_error(f"Child {child.node_id} parent pointer doesn't match")
                    
                if child.depth != node.depth + 1:
                    result.add_error(f"Child {child.node_id} depth inconsistent")
                    
            if hasattr(node, 'samples') and hasattr(node, 'class_counts'):
                if node.samples != sum(node.class_counts.values()):
                    result.add_error(f"Node {node.node_id} sample count inconsistent with class counts")
                    
        except Exception as e:
            result.add_error(f"Exception validating node {node.node_id}: {e}")
            
        return result
    
    def validate_tree_consistency(self, root_node, data=None):
        """Enhanced tree consistency validation maintaining existing interface"""
        result = type('ValidationResult', (), {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'add_error': lambda self, error: (self.errors.append(error), setattr(self, 'is_valid', False)),
            'add_warning': lambda self, warning: self.warnings.append(warning)
        })()
        
        if not root_node:
            result.add_error("Root node is None")
            return result
            
        try:
            visited = set()
            
            def validate_node_recursive(node, path=""):
                if not node:
                    return
                    
                node_path = f"{path}/{node.node_id}" if path else node.node_id
                
                if node.node_id in visited:
                    result.add_error(f"Circular reference detected at {node_path}")
                    return
                    
                visited.add(node.node_id)
                
                if not hasattr(node, 'node_id') or not node.node_id:
                    result.add_error(f"Node missing node_id at {node_path}")
                    
                if not hasattr(node, 'depth') or node.depth < 0:
                    result.add_error(f"Invalid depth at {node_path}")
                    
                for i, child in enumerate(getattr(node, 'children', [])):
                    if hasattr(child, 'parent') and child.parent != node:
                        result.add_error(f"Child {i} at {node_path} has incorrect parent reference")
                        
                    if hasattr(child, 'depth') and child.depth != node.depth + 1:
                        result.add_error(f"Child {i} at {node_path} has incorrect depth")
                        
                    validate_node_recursive(child, node_path)
                    
                # Note: During enhanced workflow, split_feature and children are set after validation
                if not getattr(node, 'is_terminal', True):
                    children = getattr(node, 'children', [])
                    if len(children) > 0:
                        if not hasattr(node, 'split_feature') or not node.split_feature:
                            result.add_warning(f"Split node missing split_feature at {node_path}")
                            
                        if len(children) < 2:
                            result.add_error(f"Split node has less than 2 children at {node_path}")
                        
            validate_node_recursive(root_node)
            
            if data is not None:
                self._validate_data_consistency_safe(root_node, data, result)
                
        except Exception as e:
            result.add_error(f"Exception during tree validation: {e}")
            
        return result
    
    def validate_sample_assignment(self, node: TreeNode, data: pd.DataFrame) -> ValidationResult:
        """Validate that samples are correctly assigned to child nodes"""
        result = ValidationResult(is_valid=True)
        
        try:
            if node.is_terminal or len(node.children) == 0:
                return result
                
            if not node.split_feature or node.split_feature not in data.columns:
                result.add_error(f"Split feature '{node.split_feature}' not found in data")
                return result
                
            feature_data = data[node.split_feature]
            total_samples = len(data)
            
            if node.split_type in ['numeric', 'numerical']:
                left_mask = feature_data <= node.split_value
                right_mask = ~left_mask
                expected_assignments = [left_mask.sum(), right_mask.sum()]
                
            elif node.split_type in ['numeric_multi_bin', 'numerical_multi_bin']:
                if hasattr(node, 'split_thresholds') and node.split_thresholds:
                    thresholds = node.split_thresholds
                    expected_assignments = []
                    
                    bin_mask = feature_data <= thresholds[0]
                    expected_assignments.append(bin_mask.sum())
                    
                    for i in range(len(thresholds) - 1):
                        bin_mask = (feature_data > thresholds[i]) & (feature_data <= thresholds[i + 1])
                        expected_assignments.append(bin_mask.sum())
                    
                    bin_mask = feature_data > thresholds[-1]
                    expected_assignments.append(bin_mask.sum())
                else:
                    left_mask = feature_data <= node.split_value
                    right_mask = ~left_mask
                    expected_assignments = [left_mask.sum(), right_mask.sum()]
                
            elif node.split_type == 'categorical':
                assignments = feature_data.map(node.split_categories).fillna(-1)
                expected_assignments = []
                for i in range(len(node.children)):
                    expected_assignments.append((assignments == i).sum())
                    
            else:
                result.add_error(f"Unknown split type: {node.split_type}")
                return result
                
            if sum(expected_assignments) != total_samples:
                missing = total_samples - sum(expected_assignments)
                result.add_error(f"Sample assignment incomplete: {missing} samples unassigned")
                
            for i, count in enumerate(expected_assignments):
                if count == 0 and i < len(node.children):
                    result.add_warning(f"Child {node.children[i].node_id} would receive 0 samples")
                    
        except Exception as e:
            result.add_error(f"Exception validating sample assignment: {e}")
            
        return result
    
    def _validate_data_compatibility(self, config: SplitConfiguration, 
                                   data: pd.DataFrame, result: ValidationResult):
        """Validate that configuration is compatible with data"""
        if config.feature not in data.columns:
            result.add_error(f"Feature '{config.feature}' not found in data")
            return
            
        feature_data = data[config.feature]
        
        if feature_data.isnull().any():
            null_count = feature_data.isnull().sum()
            result.add_warning(f"Feature '{config.feature}' has {null_count} missing values")
            
        if config.split_type in ['numeric', 'numerical']:
            if not pd.api.types.is_numeric_dtype(feature_data):
                result.add_error(f"Numeric split on non-numeric feature '{config.feature}'")
        elif config.split_type == 'categorical':
            if pd.api.types.is_numeric_dtype(feature_data) and not isinstance(feature_data.dtype, pd.CategoricalDtype):
                result.add_warning(f"Categorical split on numeric feature '{config.feature}'")
                
    def _collect_all_nodes(self, root: TreeNode) -> List[TreeNode]:
        """Collect all nodes in the tree"""
        nodes = []
        
        def traverse(node):
            nodes.append(node)
            for child in node.children:
                traverse(child)
                
        traverse(root)
        return nodes
    
    def _validate_tree_structure(self, root: TreeNode, result: ValidationResult):
        """Validate tree structure properties"""
        if root.parent is not None:
            result.add_error("Root node has a parent")
            
        if root.depth != 0:
            result.add_error(f"Root node depth is {root.depth}, should be 0")
            
        if self._has_cycle(root):
            result.add_error("Tree contains cycles")
            
        self._validate_depth_consistency(root, 0, result)
        
    def _has_cycle(self, root: TreeNode, visited: Optional[Set[str]] = None) -> bool:
        """Check if tree has cycles"""
        if visited is None:
            visited = set()
            
        if root.node_id in visited:
            return True
            
        visited.add(root.node_id)
        
        for child in root.children:
            if self._has_cycle(child, visited.copy()):
                return True
                
        return False
    
    def _validate_depth_consistency(self, node: TreeNode, expected_depth: int, result: ValidationResult):
        """Validate that node depths are consistent"""
        if node.depth != expected_depth:
            result.add_error(f"Node {node.node_id} has depth {node.depth}, expected {expected_depth}")
            
        for child in node.children:
            self._validate_depth_consistency(child, expected_depth + 1, result)
            
    def _validate_tree_data_consistency(self, root: TreeNode, data: pd.DataFrame, result: ValidationResult):
        """Validate tree consistency with data"""
        def validate_node_data(node: TreeNode, node_data: pd.DataFrame):
            if len(node_data) == 0:
                result.add_warning(f"Node {node.node_id} has no data")
                return
                
            if not node.is_terminal and len(node.children) > 0:
                sample_result = self.validate_sample_assignment(node, node_data)
                result.errors.extend(sample_result.errors)
                result.warnings.extend(sample_result.warnings)
                if not sample_result.is_valid:
                    result.is_valid = False
                    
            if not node.is_terminal and node.split_feature in node_data.columns:
                try:
                    if node.split_type in ['numeric', 'numerical']:
                        left_data = node_data[node_data[node.split_feature] <= node.split_value]
                        right_data = node_data[node_data[node.split_feature] > node.split_value]
                        child_data = [left_data, right_data]
                    elif node.split_type == 'categorical':
                        child_data = []
                        for i in range(len(node.children)):
                            mask = node_data[node.split_feature].map(node.split_categories) == i
                            child_data.append(node_data[mask])
                    else:
                        return
                        
                    for i, child in enumerate(node.children):
                        if i < len(child_data):
                            validate_node_data(child, child_data[i])
                            
                except Exception as e:
                    result.add_error(f"Error validating node {node.node_id} data: {e}")
                    
        validate_node_data(root, data)

    def _validate_data_consistency_safe(self, root_node, data, result):
        """Safe data consistency validation with error handling"""
        try:
            def validate_node_data(node, node_data):
                if node_data is None or len(node_data) == 0:
                    if not getattr(node, 'is_terminal', True):
                        result.add_warning(f"Non-terminal node {node.node_id} has no data")
                    return
                    
                if hasattr(node, 'samples') and node.samples > 0:
                    if len(node_data) != node.samples:
                        result.add_warning(f"Node {node.node_id} data size mismatch: expected {node.samples}, got {len(node_data)}")
                        
                if not getattr(node, 'is_terminal', True) and hasattr(node, 'split_feature'):
                    if node.split_feature not in node_data.columns:
                        result.add_error(f"Split feature '{node.split_feature}' not found in data for node {node.node_id}")
                        return
                        
            validate_node_data(root_node, data)
            
        except Exception as e:
            result.add_error(f"Error validating data consistency: {e}")


class TreeConsistencyChecker:
    """Utility for checking tree consistency after operations"""
    
    @staticmethod
    def check_after_split(node: TreeNode, split_config: SplitConfiguration,
                         data: Optional[pd.DataFrame] = None) -> ValidationResult:
        """Check tree consistency after applying a split"""
        validator = SplitValidator()
        result = ValidationResult(is_valid=True)
        
        config_result = validator.validate_split_config(split_config, data)
        result.errors.extend(config_result.errors)
        result.warnings.extend(config_result.warnings)
        if not config_result.is_valid:
            result.is_valid = False
            
        node_result = validator.validate_node_state(node)
        result.errors.extend(node_result.errors)
        result.warnings.extend(node_result.warnings)
        if not node_result.is_valid:
            result.is_valid = False
            
        if data is not None:
            assignment_result = validator.validate_sample_assignment(node, data)
            result.errors.extend(assignment_result.errors)
            result.warnings.extend(assignment_result.warnings)
            if not assignment_result.is_valid:
                result.is_valid = False
                
        return result
    
    @staticmethod
    def check_tree_integrity(root: TreeNode, data: Optional[pd.DataFrame] = None) -> ValidationResult:
        """Comprehensive tree integrity check"""
        validator = SplitValidator()
        return validator.validate_tree_consistency(root, data)