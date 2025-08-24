#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Central Metrics Calculator Utility for Bespoke Utility
Provides standardized metric calculations using only stored tree data
Ensures consistency across all visualization components
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np

from models.node import TreeNode
from models.decision_tree import BespokeDecisionTree

logger = logging.getLogger(__name__)


class CentralMetricsCalculator:
    """
    Central utility for all metric calculations in visualization components.
    Uses only stored tree data, never external datasets for consistency.
    """
    
    @staticmethod
    def calculate_percentage(count: Union[int, float], total: Union[int, float], 
                           decimal_places: int = 2) -> float:
        """
        Standard percentage calculation with consistent scale (0-100)
        
        Args:
            count: Numerator value
            total: Denominator value  
            decimal_places: Number of decimal places to round to
            
        Returns:
            Percentage value (0-100 scale)
        """
        if total == 0 or pd.isna(total) or pd.isna(count):
            return 0.0
        try:
            percentage = (count / total) * 100
            return round(percentage, decimal_places)
        except (ZeroDivisionError, TypeError):
            return 0.0
    
    @staticmethod
    def calculate_class_distribution(node: TreeNode) -> Dict[str, Any]:
        """
        Calculate class distribution for a node using stored data
        
        Args:
            node: TreeNode with stored class_counts and samples
            
        Returns:
            Dictionary with class distribution information
        """
        class_counts = getattr(node, 'class_counts', {})
        total_samples = getattr(node, 'samples', 0)
        
        if not class_counts or total_samples == 0:
            return {
                'class_counts': {},
                'class_percentages': {},
                'total_samples': 0,
                'majority_class': None,
                'majority_percentage': 0.0
            }
        
        class_percentages = {}
        for class_label, count in class_counts.items():
            class_percentages[class_label] = CentralMetricsCalculator.calculate_percentage(
                count, total_samples, 2
            )
        
        majority_class = max(class_counts.keys(), key=lambda k: class_counts[k])
        majority_percentage = class_percentages.get(majority_class, 0.0)
        
        return {
            'class_counts': class_counts,
            'class_percentages': class_percentages,
            'total_samples': total_samples,
            'majority_class': majority_class,
            'majority_percentage': majority_percentage
        }
    
    @staticmethod
    def calculate_target_rate(node: TreeNode, positive_class: Any) -> Dict[str, Any]:
        """
        Calculate target rate for a node using stored data
        
        Args:
            node: TreeNode with stored class_counts and samples
            positive_class: The positive class for binary classification
            
        Returns:
            Dictionary with target rate information
        """
        class_counts = getattr(node, 'class_counts', {})
        total_samples = getattr(node, 'samples', 0)
        
        if not class_counts or total_samples == 0:
            return {
                'target_count': 0,
                'target_rate': 0.0,
                'non_target_count': 0,
                'non_target_rate': 0.0
            }
        
        target_count = class_counts.get(positive_class, 0)
        non_target_count = total_samples - target_count
        
        target_rate = CentralMetricsCalculator.calculate_percentage(target_count, total_samples, 2)
        non_target_rate = CentralMetricsCalculator.calculate_percentage(non_target_count, total_samples, 2)
        
        return {
            'target_count': target_count,
            'target_rate': target_rate,
            'non_target_count': non_target_count,
            'non_target_rate': non_target_rate
        }
    
    @staticmethod
    def calculate_lift(node_target_rate: float, overall_target_rate: float, 
                      as_percentage: bool = True) -> float:
        """
        Standard lift calculation with consistent units
        
        Args:
            node_target_rate: Target rate for the node (0-100 scale)
            overall_target_rate: Overall target rate (0-100 scale)
            as_percentage: If True, return lift as percentage (default), else as ratio
            
        Returns:
            Lift value (percentage or ratio depending on as_percentage)
        """
        if overall_target_rate == 0 or pd.isna(overall_target_rate) or pd.isna(node_target_rate):
            return 100.0 if as_percentage else 1.0
        
        try:
            lift_ratio = node_target_rate / overall_target_rate
            return lift_ratio * 100 if as_percentage else lift_ratio
        except (ZeroDivisionError, TypeError):
            return 100.0 if as_percentage else 1.0
    
    @staticmethod
    def get_tree_level_statistics(model: BespokeDecisionTree) -> Dict[str, Any]:
        """
        Get overall dataset statistics from tree model (no external data needed)
        
        Args:
            model: Fitted BespokeDecisionTree
            
        Returns:
            Dictionary with overall statistics
        """
        try:
            root_node = getattr(model, 'root', None)
            if not root_node:
                logger.error("Model has no root node")
                return {'total_size': 0, 'target_count': 0, 'target_rate': 0.0}
            
            total_size = getattr(root_node, 'samples', 0)
            class_counts = getattr(root_node, 'class_counts', {})
            
            positive_class = getattr(model, 'positive_class', None)
            if not positive_class and class_counts:
                positive_class = CentralMetricsCalculator._determine_positive_class(class_counts)
            
            target_rate_info = CentralMetricsCalculator.calculate_target_rate(root_node, positive_class)
            
            return {
                'total_size': total_size,
                'target_count': target_rate_info['target_count'],
                'target_rate': target_rate_info['target_rate'],
                'positive_class': positive_class,
                'class_counts': class_counts
            }
            
        except Exception as e:
            logger.error(f"Error getting tree-level statistics: {e}")
            return {'total_size': 0, 'target_count': 0, 'target_rate': 0.0}
    
    @staticmethod
    def _determine_positive_class(class_counts: Dict) -> Any:
        """Determine positive class from tree class counts"""
        if len(class_counts) == 2:
            for class_val in class_counts.keys():
                if class_val in [1, True, 'Yes', 'yes', 'Y', 'y', 'True', 'true', 'positive', 'Positive']:
                    return class_val
            return sorted(class_counts.keys())[-1]
        else:
            return max(class_counts.keys(), key=lambda k: class_counts[k])
    
    @staticmethod
    def get_node_basic_info(node: TreeNode) -> Dict[str, Any]:
        """
        Get basic node information using only stored attributes
        
        Args:
            node: TreeNode to analyze
            
        Returns:
            Dictionary with basic node information
        """
        return {
            'node_id': getattr(node, 'node_id', 'unknown'),
            'depth': getattr(node, 'depth', 0),
            'is_terminal': getattr(node, 'is_terminal', True),
            'samples': getattr(node, 'samples', 0),
            'prediction': getattr(node, 'majority_class', None) or getattr(node, 'prediction', None),
            'split_feature': getattr(node, 'split_feature', None) if not getattr(node, 'is_terminal', True) else None,
            'split_value': getattr(node, 'split_value', None) if not getattr(node, 'is_terminal', True) else None,
            'split_type': getattr(node, 'split_type', None) if not getattr(node, 'is_terminal', True) else None,
            'children_count': len(getattr(node, 'children', [])),
            'parent_id': getattr(node.parent, 'node_id', None) if getattr(node, 'parent', None) else None,
            'impurity': getattr(node, 'impurity', 0.0),
            'confidence': getattr(node, 'confidence', None),
            'class_counts': getattr(node, 'class_counts', {}),
            'accuracy': getattr(node, 'accuracy', None),
            'precision': getattr(node, 'precision', None),
            'recall': getattr(node, 'recall', None),
            'f1_score': getattr(node, 'f1_score', None)
        }
    
    @staticmethod
    def format_percentage(value: Union[int, float], decimal_places: int = 2) -> str:
        """
        Standard percentage formatting
        
        Args:
            value: Percentage value (0-100 scale)
            decimal_places: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        if pd.isna(value):
            return "0.00%"
        return f"{value:.{decimal_places}f}%"
    
    @staticmethod
    def format_count(value: Union[int, float]) -> str:
        """
        Standard count formatting with thousands separator
        
        Args:
            value: Count value
            
        Returns:
            Formatted count string
        """
        if pd.isna(value):
            return "0"
        return f"{int(value):,}"


class MetricsValidator:
    """Utility for validating metric consistency across components"""
    
    @staticmethod
    def validate_node_metrics(node: TreeNode) -> List[str]:
        """
        Validate that node metrics are consistent
        
        Args:
            node: TreeNode to validate
            
        Returns:
            List of validation warnings (empty if all valid)
        """
        warnings = []
        
        try:
            samples = getattr(node, 'samples', 0)
            class_counts = getattr(node, 'class_counts', {})
            
            if class_counts:
                calculated_samples = sum(class_counts.values())
                if samples != calculated_samples:
                    warnings.append(f"Node samples ({samples}) != sum of class_counts ({calculated_samples})")
            
            if class_counts and samples > 0:
                total_percentage = sum(
                    CentralMetricsCalculator.calculate_percentage(count, samples, 4) 
                    for count in class_counts.values()
                )
                if abs(total_percentage - 100.0) > 0.01:  # Allow small rounding errors
                    warnings.append(f"Class percentages sum to {total_percentage:.2f}%, not 100%")
            
        except Exception as e:
            warnings.append(f"Error validating node metrics: {e}")
        
        return warnings