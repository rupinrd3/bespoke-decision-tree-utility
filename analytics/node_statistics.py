#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Node Statistics Module for Bespoke Utility
Calculates and provides detailed statistics for decision tree nodes.

"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from enum import Enum

from models.node import TreeNode
from utils.metrics_calculator import CentralMetricsCalculator

logger = logging.getLogger(__name__)

class NodeAnalyzer:
    """Class for analyzing node statistics in decision trees"""
    
    def __init__(self):
        """Initialize the node analyzer"""
        pass
    
    def get_node_report(self, node: TreeNode, include_performance: bool = True, 
                       include_tree_context: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive report for a single node - Enhanced version
        
        Args:
            node: TreeNode to analyze
            include_performance: Whether to include performance metrics
            include_tree_context: Whether to include tree context information
            
        Returns:
            Dictionary with comprehensive node analysis
        """
        try:
            report = {
                'node_id': node.node_id,
                'depth': node.depth,
                'is_terminal': node.is_terminal,
                'samples': getattr(node, 'samples', 0),
                'class_counts': getattr(node, 'class_counts', {}),
                'impurity': getattr(node, 'impurity', None),
                'majority_class': getattr(node, 'majority_class', None),
                'probability': getattr(node, 'probability', None),
                'num_children': len(node.children) if hasattr(node, 'children') else 0
            }
            
            if not node.is_terminal:
                split_info = {
                    'split_feature': getattr(node, 'split_feature', None),
                    'split_value': getattr(node, 'split_value', None),
                    'split_rule': getattr(node, 'split_rule', None),
                    'split_type': getattr(node, 'split_type', None)
                }
                
                if node.split_type == 'categorical':
                    split_info['split_categories'] = getattr(node, 'split_categories', {})
                    
                    if split_info['split_categories']:
                        categories_by_child = {}
                        for cat, child_idx in split_info['split_categories'].items():
                            if child_idx not in categories_by_child:
                                categories_by_child[child_idx] = []
                            categories_by_child[child_idx].append(cat)
                        split_info['categories_by_child'] = categories_by_child
                
                report.update(split_info)
            
            try:
                if hasattr(node, 'get_node_rules'):
                    report['path_rules'] = node.get_node_rules()
                else:
                    report['path_rules'] = self._get_path_to_node(node)
            except Exception as e:
                logger.warning(f"Could not get path rules for node {node.node_id}: {e}")
                report['path_rules'] = []
            
            if include_performance:
                performance_metrics = {}
                
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    value = getattr(node, metric, None)
                    if value is not None:
                        performance_metrics[metric] = value
                
                if node.is_terminal:
                    if hasattr(node, 'probability') and node.probability is not None:
                        performance_metrics['prediction_confidence'] = node.probability
                    
                    if node.class_counts:
                        total_samples = sum(node.class_counts.values())
                        class_distribution = {}
                        for class_label, count in node.class_counts.items():
                            class_distribution[class_label] = {
                                'count': count,
                                'percentage': CentralMetricsCalculator.calculate_percentage(count, total_samples, 2)
                            }
                        performance_metrics['class_distribution'] = class_distribution
                        
                        if total_samples > 0:
                            max_class_count = max(node.class_counts.values())
                            performance_metrics['node_purity'] = max_class_count / total_samples
                            performance_metrics['node_impurity_calculated'] = 1 - (max_class_count / total_samples)
                
                if performance_metrics:
                    report['performance_metrics'] = performance_metrics
            
            if include_tree_context:
                context_info = {}
                
                if node.parent:
                    context_info['parent_id'] = node.parent.node_id
                    context_info['parent_split_feature'] = getattr(node.parent, 'split_feature', None)
                
                if hasattr(node, 'children') and node.children:
                    children_info = []
                    for child in node.children:
                        child_info = {
                            'node_id': child.node_id,
                            'is_terminal': child.is_terminal,
                            'samples': getattr(child, 'samples', 0),
                            'majority_class': getattr(child, 'majority_class', None)
                        }
                        children_info.append(child_info)
                    context_info['children'] = children_info
                
                if node.parent and hasattr(node.parent, 'children'):
                    siblings = [child for child in node.parent.children if child.node_id != node.node_id]
                    sibling_info = []
                    for sibling in siblings:
                        sibling_info.append({
                            'node_id': sibling.node_id,
                            'is_terminal': sibling.is_terminal,
                            'samples': getattr(sibling, 'samples', 0)
                        })
                    context_info['siblings'] = sibling_info
                
                if context_info:
                    report['tree_context'] = context_info
            
            if not node.is_terminal and hasattr(node, 'children') and node.children:
                split_quality = self._calculate_split_quality(node)
                if split_quality:
                    report['split_quality'] = split_quality
            
            report['analysis_timestamp'] = pd.Timestamp.now().isoformat()
            report['node_type_summary'] = 'Terminal Node' if node.is_terminal else 'Internal Node'
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating node report for {node.node_id}: {e}", exc_info=True)
            return {
                'node_id': node.node_id,
                'error': f"Could not generate report: {str(e)}",
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
    
    def _get_path_to_node(self, node: TreeNode) -> List[Dict[str, Any]]:
        """
        Get the path from the root to this node
        
        Args:
            node: TreeNode to find path to
            
        Returns:
            List of splits leading to the node
        """
        path = []
        current = node
        
        while current.parent is not None:
            parent = current.parent
            
            if current in parent.children:
                child_idx = parent.children.index(current)
                
                split = {
                    'node_id': parent.node_id,
                    'feature': parent.split_feature,
                    'type': parent.split_type
                }
                
                if parent.split_type == 'numeric':
                    if child_idx == 0:
                        split['condition'] = '<='
                        split['value'] = parent.split_value
                        split['rule'] = f"{parent.split_feature} <= {parent.split_value}"
                    else:
                        split['condition'] = '>'
                        split['value'] = parent.split_value
                        split['rule'] = f"{parent.split_feature} > {parent.split_value}"
                
                elif parent.split_type == 'categorical':
                    categories = [cat for cat, idx in parent.split_categories.items() 
                                 if idx == child_idx]
                    
                    split['condition'] = 'in'
                    split['value'] = categories
                    
                    if len(categories) <= 3:
                        cats_str = ", ".join(str(c) for c in categories)
                        split['rule'] = f"{parent.split_feature} in [{cats_str}]"
                    else:
                        split['rule'] = f"{parent.split_feature} in [... {len(categories)} values]"
                
                path.append(split)
            
            current = parent
        
        path.reverse()
        
        return path
    
    def analyze_tree(self, root: TreeNode) -> Dict[str, Any]:
        """
        Analyze the entire decision tree
        
        Args:
            root: Root node of the tree
            
        Returns:
            Dictionary with tree statistics
        """
        all_nodes = root.get_subtree_nodes()
        
        terminal_nodes = [node for node in all_nodes if node.is_terminal]
        
        stats = {
            'total_nodes': len(all_nodes),
            'terminal_nodes': len(terminal_nodes),
            'non_terminal_nodes': len(all_nodes) - len(terminal_nodes),
            'max_depth': max(node.depth for node in all_nodes),
            'avg_depth': np.mean([node.depth for node in terminal_nodes]),
            'avg_samples_per_leaf': np.mean([node.samples for node in terminal_nodes]),
            'min_samples_in_leaf': min([node.samples for node in terminal_nodes]) if terminal_nodes else 0,
            'max_samples_in_leaf': max([node.samples for node in terminal_nodes]) if terminal_nodes else 0
        }
        
        feature_usage = {}
        for node in all_nodes:
            if not node.is_terminal and node.split_feature:
                feature_usage[node.split_feature] = feature_usage.get(node.split_feature, 0) + 1
        
        stats['feature_usage'] = feature_usage
        stats['unique_features_used'] = len(feature_usage)
        
        impurities = [node.impurity for node in all_nodes if node.impurity is not None]
        if impurities:
            stats['avg_impurity'] = np.mean(impurities)
            stats['min_impurity'] = min(impurities)
            stats['max_impurity'] = max(impurities)
        
        class_counts = {}
        for node in terminal_nodes:
            for cls, count in node.class_counts.items():
                class_counts[cls] = class_counts.get(cls, 0) + count
        
        stats['class_distribution'] = class_counts
        
        stats['balance_factor'] = self._calculate_balance_factor(root)
        
        stats['important_nodes'] = self._identify_important_nodes(all_nodes)
        
        return stats
    
    def _calculate_balance_factor(self, root: TreeNode) -> float:
        """
        Calculate the balance factor of the tree
        
        Args:
            root: Root node of the tree
            
        Returns:
            Balance factor (0.0 to 1.0, where 1.0 is perfectly balanced)
        """
        terminal_nodes = root.get_leaf_nodes()
        
        depths = [node.depth for node in terminal_nodes]
        
        if not depths:
            return 1.0  # Single node tree is balanced
        
        min_depth = min(depths)
        max_depth = max(depths)
        
        if max_depth == min_depth:
            return 1.0  # Perfectly balanced
        
        depth_counts = {}
        for depth in depths:
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        
        total = len(depths)
        entropy = 0
        
        for count in depth_counts.values():
            p = count / total
            entropy -= p * np.log2(p)
        
        max_entropy = np.log2(max_depth - min_depth + 1)
        
        if max_entropy > 0:
            balance_factor = entropy / max_entropy
        else:
            balance_factor = 1.0
        
        return balance_factor
    
    def _identify_important_nodes(self, nodes: List[TreeNode]) -> List[str]:
        """
        Identify important nodes in the tree
        
        Args:
            nodes: List of tree nodes
            
        Returns:
            List of important node IDs
        """
        important_nodes = []
        
        non_terminal = [node for node in nodes if not node.is_terminal]
        
        for node in non_terminal:
            if node.children and node.impurity is not None:
                weighted_child_impurity = 0
                total_samples = 0
                
                for child in node.children:
                    if child.impurity is not None and child.samples > 0:
                        weighted_child_impurity += child.impurity * child.samples
                        total_samples += child.samples
                
                if total_samples > 0:
                    weighted_child_impurity /= total_samples
                    
                    impurity_decrease = node.impurity - weighted_child_impurity
                    
                    if impurity_decrease > 0.1:  # Threshold can be adjusted
                        important_nodes.append(node.node_id)
        
        terminal = [node for node in nodes if node.is_terminal]
        
        if terminal:
            avg_samples = np.mean([node.samples for node in terminal])
            
            for node in terminal:
                if node.samples > 2 * avg_samples:
                    important_nodes.append(node.node_id)
                
                if node.accuracy is not None and node.accuracy > 0.9:
                    important_nodes.append(node.node_id)
        
        important_nodes = list(set(important_nodes))
        
        return important_nodes
    
    def compare_nodes(self, node1: TreeNode, node2: TreeNode) -> Dict[str, Any]:
        """
        Compare two nodes in the tree
        
        Args:
            node1: First node
            node2: Second node
            
        Returns:
            Dictionary with comparison results
        """
        report1 = self.get_node_report(node1)
        report2 = self.get_node_report(node2)
        
        comparison = {
            'node1_id': node1.node_id,
            'node2_id': node2.node_id,
            'is_sibling': node1.parent == node2.parent and node1.parent is not None,
            'depth_difference': node1.depth - node2.depth,
            'samples_difference': node1.samples - node2.samples,
            'impurity_difference': (node1.impurity - node2.impurity) 
                                 if node1.impurity is not None and node2.impurity is not None else None
        }
        
        class_distribution1 = report1.get('class_distribution', {})
        class_distribution2 = report2.get('class_distribution', {})
        
        all_classes = set(list(class_distribution1.keys()) + list(class_distribution2.keys()))
        
        distribution_diff = {}
        
        for cls in all_classes:
            count1 = class_distribution1.get(cls, 0)
            count2 = class_distribution2.get(cls, 0)
            distribution_diff[cls] = count1 - count2
        
        comparison['class_distribution_diff'] = distribution_diff
        
        if node1.is_terminal and node2.is_terminal:
            comparison['same_prediction'] = node1.majority_class == node2.majority_class
                
            if not comparison['same_prediction']:
                comparison['prediction1'] = node1.majority_class
                comparison['prediction2'] = node2.majority_class
                
            comparison['confidence_difference'] = (
                node1.probability - node2.probability
                if node1.probability is not None and node2.probability is not None
                else None
            )

        
        if node1.accuracy is not None and node2.accuracy is not None:
            comparison['accuracy_difference'] = node1.accuracy - node2.accuracy
        
        if node1.precision is not None and node2.precision is not None:
            comparison['precision_difference'] = node1.precision - node2.precision
        
        if node1.recall is not None and node2.recall is not None:
            comparison['recall_difference'] = node1.recall - node2.recall
        
        if node1.f1_score is not None and node2.f1_score is not None:
            comparison['f1_score_difference'] = node1.f1_score - node2.f1_score
        
        return comparison
    
    def analyze_node_level(self, root: TreeNode, level: int) -> Dict[str, Any]:
        """
        Analyze nodes at a specific level in the tree
        
        Args:
            root: Root node of the tree
            level: Tree level to analyze
            
        Returns:
            Dictionary with level statistics
        """
        all_nodes = root.get_subtree_nodes()
        
        level_nodes = [node for node in all_nodes if node.depth == level]
        
        if not level_nodes:
            return {'level': level, 'nodes_found': 0}
        
        stats = {
            'level': level,
            'nodes_found': len(level_nodes),
            'terminal_nodes': sum(1 for node in level_nodes if node.is_terminal),
            'non_terminal_nodes': sum(1 for node in level_nodes if not node.is_terminal),
            'total_samples': sum(node.samples for node in level_nodes),
            'avg_samples_per_node': np.mean([node.samples for node in level_nodes]),
            'min_samples': min(node.samples for node in level_nodes),
            'max_samples': max(node.samples for node in level_nodes)
        }
        
        impurities = [node.impurity for node in level_nodes if node.impurity is not None]
        if impurities:
            stats['avg_impurity'] = np.mean(impurities)
            stats['min_impurity'] = min(impurities)
            stats['max_impurity'] = max(impurities)
        
        feature_usage = {}
        for node in level_nodes:
            if not node.is_terminal and node.split_feature:
                feature_usage[node.split_feature] = feature_usage.get(node.split_feature, 0) + 1
        
        stats['feature_usage'] = feature_usage
        stats['unique_features_used'] = len(feature_usage)
        
        class_counts = {}
        for node in level_nodes:
            for cls, count in (node.class_counts or {}).items():
                class_counts[cls] = class_counts.get(cls, 0) + count
        
        stats['class_distribution'] = class_counts
        
        accuracies = [node.accuracy for node in level_nodes if node.accuracy is not None]
        if accuracies:
            stats['avg_accuracy'] = np.mean(accuracies)
            stats['min_accuracy'] = min(accuracies)
            stats['max_accuracy'] = max(accuracies)
        
        return stats
    
    def get_node_path_rules(self, node: TreeNode) -> List[str]:
        """
        Get path rules leading to a node in a human-readable format
        
        Args:
            node: TreeNode to get path for
            
        Returns:
            List of human-readable rules
        """
        path = self._get_path_to_node(node)
        
        rules = [split['rule'] for split in path]
        
        return rules
    
    def plot_node_class_distribution(self, node: TreeNode, title: Optional[str] = None,
                                  figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot class distribution for a node
        
        Args:
            node: TreeNode to plot distribution for
            title: Plot title (defaults to 'Class Distribution for Node {node_id}')
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        class_counts = node.class_counts
        
        if not class_counts:
            ax.text(0.5, 0.5, "No class distribution data available", 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        labels = [str(cls) for cls in class_counts.keys()]
        values = list(class_counts.values())
        
        majority_class = node.majority_class
        colors = ['lightgray'] * len(labels)
        
        for i, label in enumerate(labels):
            if label == str(majority_class):
                colors[i] = 'skyblue'
        
        wedges, texts, autotexts = ax.pie(
            values, 
            labels=labels, 
            autopct='%1.1f%%',
            startangle=90, 
            colors=colors,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        
        for text in texts:
            text.set_fontsize(12)
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        if title is None:
            title = f"Class Distribution for Node {node.node_id}"
        ax.set_title(title)
        
        total_samples = sum(values)
        ax.text(0, -1.1, f"Total Samples: {total_samples}", ha='center', fontsize=12)
        
        ax.axis('equal')
        
        fig.tight_layout()
        
        return fig
    
    def plot_node_importance(self, root: TreeNode, metric: str = 'samples',
                          top_n: int = 10, title: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot node importance based on various metrics
        
        Args:
            root: Root node of the tree
            metric: Metric to use ('samples', 'impurity_decrease', 'accuracy')
            top_n: Number of top nodes to display
            title: Plot title
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure
        """
        all_nodes = root.get_subtree_nodes()
        
        node_importance = {}
        
        if metric == 'samples':
            terminal_nodes = [node for node in all_nodes if node.is_terminal]
            for node in terminal_nodes:
                node_importance[node.node_id] = node.samples
            
            if not title:
                title = f"Top {top_n} Terminal Nodes by Sample Count"
                
        elif metric == 'impurity_decrease':
            non_terminal = [node for node in all_nodes if not node.is_terminal]
            
            for node in non_terminal:
                if node.children and node.impurity is not None:
                    weighted_child_impurity = 0
                    total_samples = 0
                    
                    for child in node.children:
                        if child.impurity is not None and child.samples > 0:
                            weighted_child_impurity += child.impurity * child.samples
                            total_samples += child.samples
                    
                    if total_samples > 0:
                        weighted_child_impurity /= total_samples
                        impurity_decrease = node.impurity - weighted_child_impurity
                        node_importance[node.node_id] = impurity_decrease
            
            if not title:
                title = f"Top {top_n} Splits by Impurity Decrease"
                
        elif metric == 'accuracy':
            for node in all_nodes:
                if node.accuracy is not None:
                    node_importance[node.node_id] = node.accuracy
            
            if not title:
                title = f"Top {top_n} Nodes by Accuracy"
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if not node_importance:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f"No data available for metric: {metric}", 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        top_nodes = dict(sorted(node_importance.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.bar(top_nodes.keys(), top_nodes.values())
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        ax.set_title(title)
        ax.set_xlabel('Node ID')
        
        if metric == 'samples':
            ax.set_ylabel('Sample Count')
        elif metric == 'impurity_decrease':
            ax.set_ylabel('Impurity Decrease')
        elif metric == 'accuracy':
            ax.set_ylabel('Accuracy')
            ax.set_ylim([0, 1.05])
        
        plt.xticks(rotation=45, ha='right')
        
        ax.grid(True, axis='y', alpha=0.3)
        
        fig.tight_layout()
        
        return fig

    def _calculate_split_quality(self, node: TreeNode) -> Dict[str, Any]:
        """
        Calculate enhanced split quality metrics for internal nodes
        
        Args:
            node: Internal TreeNode with children
            
        Returns:
            Dictionary with split quality metrics
        """
        try:
            if node.is_terminal or not hasattr(node, 'children') or not node.children:
                return {}
            
            quality_metrics = {}
            
            parent_impurity = getattr(node, 'impurity', None)
            if parent_impurity is None:
                return {}
            
            total_samples = 0
            weighted_impurity = 0
            child_impurities = []
            child_sample_counts = []
            
            for child in node.children:
                child_samples = getattr(child, 'samples', 0)
                child_impurity = getattr(child, 'impurity', 0)
                
                if child_samples > 0:
                    total_samples += child_samples
                    weighted_impurity += child_impurity * child_samples
                    child_impurities.append(child_impurity)
                    child_sample_counts.append(child_samples)
            
            if total_samples > 0:
                weighted_impurity /= total_samples
                
                impurity_reduction = parent_impurity - weighted_impurity
                quality_metrics['impurity_reduction'] = impurity_reduction
                quality_metrics['weighted_child_impurity'] = weighted_impurity
                quality_metrics['parent_impurity'] = parent_impurity
                
                if parent_impurity > 0:
                    quality_metrics['relative_improvement'] = impurity_reduction / parent_impurity
                
                if len(child_sample_counts) > 1:
                    sample_proportions = [count / total_samples for count in child_sample_counts]
                    split_entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in sample_proportions)
                    max_entropy = np.log2(len(child_sample_counts))
                    
                    quality_metrics['split_balance'] = split_entropy / max_entropy if max_entropy > 0 else 0
                    quality_metrics['sample_distribution'] = dict(zip(
                        [f"child_{i}" for i in range(len(child_sample_counts))],
                        child_sample_counts
                    ))
                
                if child_impurities:
                    quality_metrics['child_impurity_stats'] = {
                        'min': min(child_impurities),
                        'max': max(child_impurities),
                        'mean': np.mean(child_impurities),
                        'std': np.std(child_impurities) if len(child_impurities) > 1 else 0
                    }
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error calculating split quality for node {node.node_id}: {e}")
            return {}

    def calculate_terminal_node_metrics(self, root_node: TreeNode, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Calculate performance metrics specifically for terminal nodes
        
        Args:
            root_node: Root of the decision tree
            X: Feature data
            y: Target data
            
        Returns:
            Dictionary with terminal node performance analysis
        """
        try:
            all_nodes = root_node.get_subtree_nodes()
            terminal_nodes = [node for node in all_nodes if node.is_terminal]
            
            if not terminal_nodes:
                return {'error': 'No terminal nodes found in the tree'}
            
            metrics = {
                'total_terminal_nodes': len(terminal_nodes),
                'terminal_node_details': {},
                'overall_terminal_performance': {}
            }
            
            terminal_accuracies = []
            terminal_purities = []
            terminal_sample_counts = []
            total_correct_predictions = 0
            total_predictions = 0
            
            for node in terminal_nodes:
                node_metrics = {}
                
                node_metrics['node_id'] = node.node_id
                node_metrics['samples'] = getattr(node, 'samples', 0)
                node_metrics['prediction'] = getattr(node, 'majority_class', None)
                
                if hasattr(node, 'class_counts') and node.class_counts:
                    node_metrics['class_counts'] = node.class_counts
                    
                    total_samples = sum(node.class_counts.values())
                    if total_samples > 0:
                        max_class_count = max(node.class_counts.values())
                        purity = max_class_count / total_samples
                        node_metrics['purity'] = purity
                        terminal_purities.append(purity)
                        
                        total_correct_predictions += max_class_count
                        total_predictions += total_samples
                
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    value = getattr(node, metric, None)
                    if value is not None:
                        node_metrics[metric] = value
                        if metric == 'accuracy':
                            terminal_accuracies.append(value)
                
                terminal_sample_counts.append(node_metrics['samples'])
                metrics['terminal_node_details'][node.node_id] = node_metrics
            
            overall_perf = {}
            
            if terminal_accuracies:
                overall_perf['accuracy_stats'] = {
                    'min': min(terminal_accuracies),
                    'max': max(terminal_accuracies),
                    'mean': np.mean(terminal_accuracies),
                    'std': np.std(terminal_accuracies)
                }
            
            if terminal_purities:
                overall_perf['purity_stats'] = {
                    'min': min(terminal_purities),
                    'max': max(terminal_purities),
                    'mean': np.mean(terminal_purities),
                    'std': np.std(terminal_purities)
                }
            
            if terminal_sample_counts:
                overall_perf['sample_distribution'] = {
                    'min_samples': min(terminal_sample_counts),
                    'max_samples': max(terminal_sample_counts),
                    'mean_samples': np.mean(terminal_sample_counts),
                    'total_samples': sum(terminal_sample_counts)
                }
            
            if total_predictions > 0:
                overall_perf['weighted_accuracy'] = total_correct_predictions / total_predictions
            
            metrics['overall_terminal_performance'] = overall_perf
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating terminal node metrics: {e}", exc_info=True)
            return {'error': f"Could not calculate terminal node metrics: {str(e)}"}

    def extract_visualization_data(self, root_node: TreeNode) -> Dict[str, Any]:
        """
        Extract data needed for enhanced visualization and reporting
        
        Args:
            root_node: Root of the decision tree
            
        Returns:
            Dictionary with visualization-ready data
        """
        try:
            all_nodes = root_node.get_subtree_nodes()
            
            visualization_data = {
                'nodes': {},
                'edges': [],
                'tree_stats': {},
                'node_numbering': {}
            }
            
            nodes_by_level = {}
            for node in all_nodes:
                level = node.depth
                if level not in nodes_by_level:
                    nodes_by_level[level] = []
                nodes_by_level[level].append(node)
            
            node_number = 1
            for level in sorted(nodes_by_level.keys()):
                level_nodes = sorted(nodes_by_level[level], key=lambda n: n.node_id)
                for node in level_nodes:
                    visualization_data['node_numbering'][node.node_id] = node_number
                    node_number += 1
            
            for node in all_nodes:
                node_data = {
                    'node_id': node.node_id,
                    'node_number': visualization_data['node_numbering'][node.node_id],
                    'depth': node.depth,
                    'is_terminal': node.is_terminal,
                    'samples': getattr(node, 'samples', 0),
                    'class_counts': getattr(node, 'class_counts', {}),
                    'impurity': getattr(node, 'impurity', None),
                    'majority_class': getattr(node, 'majority_class', None),
                    'split_feature': getattr(node, 'split_feature', None),
                    'split_value': getattr(node, 'split_value', None),
                    'split_type': getattr(node, 'split_type', None)
                }
                
                visualization_data['nodes'][node.node_id] = node_data
            
            for node in all_nodes:
                if node.parent:
                    edge_data = {
                        'parent_id': node.parent.node_id,
                        'child_id': node.node_id,
                        'parent_feature': getattr(node.parent, 'split_feature', None),
                        'split_condition': self._get_split_condition(node.parent, node),
                        'child_number': visualization_data['node_numbering'][node.node_id],
                        'statistical_value': getattr(node, 'impurity', None)
                    }
                    
                    visualization_data['edges'].append(edge_data)
            
            terminal_nodes = [node for node in all_nodes if node.is_terminal]
            visualization_data['tree_stats'] = {
                'total_nodes': len(all_nodes),
                'terminal_nodes': len(terminal_nodes),
                'internal_nodes': len(all_nodes) - len(terminal_nodes),
                'max_depth': max(node.depth for node in all_nodes) if all_nodes else 0,
                'min_depth': min(node.depth for node in all_nodes) if all_nodes else 0
            }
            
            return visualization_data
            
        except Exception as e:
            logger.error(f"Error extracting visualization data: {e}", exc_info=True)
            return {'error': f"Could not extract visualization data: {str(e)}"}

    def _get_split_condition(self, parent_node: TreeNode, child_node: TreeNode) -> str:
        """
        Get the split condition text for an edge
        
        Args:
            parent_node: Parent node with split information
            child_node: Child node
            
        Returns:
            String describing the split condition
        """
        try:
            if not hasattr(parent_node, 'children') or child_node not in parent_node.children:
                return "unknown"
            
            child_index = parent_node.children.index(child_node)
            
            if parent_node.split_type == 'numeric' and parent_node.split_value is not None:
                if child_index == 0:
                    return f"≤ {parent_node.split_value}"
                else:
                    return f"> {parent_node.split_value}"
            
            elif parent_node.split_type == 'categorical' and hasattr(parent_node, 'split_categories'):
                categories = []
                for cat, idx in parent_node.split_categories.items():
                    if idx == child_index:
                        categories.append(str(cat))
                
                if len(categories) <= 3:
                    return ", ".join(categories)
                elif categories:
                    return f"{len(categories)} values"
            
            return f"bin_{child_index}"
            
        except Exception as e:
            logger.error(f"Error getting split condition: {e}")
            return "unknown"
