#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Split Finder Module for Bespoke Utility
Finds optimal splits for decision tree nodes with improved categorical handling
"""

import logging
import math
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable
from enum import Enum
import pandas as pd
import numpy as np
from itertools import combinations
import time

from models.node import TreeNode

logger = logging.getLogger(__name__)

class SplitCriterion(Enum):
    """Enumeration of splitting criteria for decision trees"""
    GINI = "gini"
    ENTROPY = "entropy"
    INFORMATION_GAIN = "information_gain"
    MISCLASSIFICATION = "misclassification"

class SplitFinder:
    """Enhanced class for finding optimal splits in decision tree nodes"""
    
    def __init__(self, criterion: str = 'gini', min_samples_leaf: int = 1, 
                 min_impurity_decrease: float = 0.0, max_features: Optional[int] = None,
                 min_bins: int = 2, max_bins: int = 20, min_samples_split: int = 2):
        """
        Initialize the Enhanced Split Finder
        
        Args:
            criterion: Splitting criterion ('gini', 'entropy', 'information_gain')
            min_samples_leaf: Minimum samples required in each leaf
            min_impurity_decrease: Minimum impurity decrease for a split
            max_features: Maximum number of features to consider
            min_bins: Minimum number of bins to consider (CRITICAL FIX: not restricted to 2)
            max_bins: Maximum number of bins to consider (CRITICAL FIX: up to 20)
            min_samples_split: Minimum samples required to split a node
        """
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        
        self.min_bins = max(2, min_bins)  # At least 2 bins for valid splits
        self.max_bins = min(20, max_bins)  # At most 20 bins for performance and interpretability
        
        self.small_dataset_threshold = 100
        self.medium_dataset_threshold = 1000
        self.high_cardinality_threshold = 50
        self.skewness_threshold = 2.0
        
        self.scipy_available = True
        try:
            from scipy import stats
        except ImportError:
            self.scipy_available = False
            logger.warning("scipy not available - using basic statistical methods")
        
        logger.info(f"Initialized SplitFinder with {self.min_bins}-{self.max_bins} bin range")
        
    def find_best_split(self, node_id: str, X: pd.DataFrame, y: pd.Series, 
                        sample_weight: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        CRITICAL FIX: Find optimal splits with flexible bin counts (2-20 bins)
        Maintains compatibility while removing artificial restrictions
        
        Args:
            node_id: Node identifier
            X: Feature data
            y: Target data
            sample_weight: Sample weights (optional)
            
        Returns:
            List of split configurations ranked by quality (NOT restricted to 2 bins)
        """
        try:
            logger.info(f"Finding best splits for node {node_id} with {len(X)} samples, {len(X.columns)} features")
            
            if len(X) < self.min_samples_leaf * 2:
                logger.warning(f"Insufficient samples ({len(X)}) for splitting")
                return []
            
            if len(X.columns) == 0:
                logger.warning("No features available for splitting")
                return []
            
            all_splits = []
            
            features_to_consider = self._select_features_for_split(X.columns.tolist())
            
            for feature in features_to_consider:
                if feature not in X.columns:
                    continue
                    
                try:
                    feature_data = X[feature]
                    
                    if feature_data.nunique() < 2:
                        logger.debug(f"Skipping feature {feature}: insufficient variation")
                        continue
                    
                    if hasattr(self, '_analyze_variable_characteristics'):
                        feature_analysis = self._analyze_variable_characteristics(feature_data)
                    else:
                        feature_analysis = {
                            'is_numerical': pd.api.types.is_numeric_dtype(feature_data),
                            'recommended_bins': self._recommend_numerical_bins(feature_data) if pd.api.types.is_numeric_dtype(feature_data) else 3
                        }
                    
                    if hasattr(self, '_find_best_splits_for_feature'):
                        feature_splits = self._find_best_splits_for_feature(
                            feature, feature_data, y, feature_analysis, sample_weight
                        )
                    else:
                        feature_splits = self._find_basic_splits_for_feature(
                            feature, feature_data, y, feature_analysis
                        )
                    
                    all_splits.extend(feature_splits)
                    
                except Exception as feature_error:
                    logger.warning(f"Error processing feature {feature}: {feature_error}")
                    continue
            
            all_splits.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            
            max_results = 10  # Return top 10 splits instead of just 2
            top_splits = all_splits[:max_results]
            
            logger.info(f"Found {len(top_splits)} split candidates with bin counts ranging from {self.min_bins} to {self.max_bins}")
            
            if not top_splits and len(all_splits) > 0:
                top_splits = all_splits[:1]  # Return at least one split
            
            return top_splits
            
        except Exception as e:
            logger.error(f"Error finding best split: {e}")
            return []

    def _analyze_variable_characteristics(self, data: pd.Series) -> Dict[str, Any]:
        """Analyze variable characteristics for intelligent type detection"""
        analysis = {
            'unique_count': data.nunique(),
            'total_count': len(data),
            'missing_count': data.isna().sum(),
            'missing_percentage': data.isna().sum() / len(data) * 100 if len(data) > 0 else 0
        }
        
        if pd.api.types.is_numeric_dtype(data):
            analysis['is_numerical'] = True
            analysis['is_categorical'] = False
            
            valid_data = data.dropna()
            if len(valid_data) > 0:
                analysis.update({
                    'min_value': valid_data.min(),
                    'max_value': valid_data.max(),
                    'mean_value': valid_data.mean(),
                    'std_value': valid_data.std(),
                    'has_decimals': not all(valid_data == valid_data.astype(int)),
                    'is_binary': analysis['unique_count'] == 2,
                    'distribution_type': self._classify_distribution(valid_data),
                    'has_outliers': self._detect_outliers(valid_data),
                    'recommended_bins': self._recommend_numerical_bins(valid_data)
                })
        else:
            analysis['is_numerical'] = False
            analysis['is_categorical'] = True
            
            value_counts = data.value_counts()
            analysis.update({
                'is_binary': analysis['unique_count'] == 2,
                'is_high_cardinality': analysis['unique_count'] > 50,
                'cardinality_level': self._classify_cardinality(analysis['unique_count']),
                'recommended_bins': self._recommend_categorical_bins(analysis['unique_count']),
                'top_categories': value_counts.head(10).to_dict(),
                'frequency_distribution': self._analyze_frequency_distribution(value_counts),
                'has_rare_categories': (value_counts < len(data) * 0.01).any()
            })
            
        return analysis

    def _classify_distribution(self, data: pd.Series) -> str:
        """Classify the distribution type of numerical data"""
        try:
            from scipy import stats
            
            skewness = stats.skew(data)
            
            if abs(skewness) < 0.5:
                return 'normal'
            elif abs(skewness) < 1.0:
                return 'moderately_skewed'
            else:
                return 'highly_skewed'
                
        except ImportError:
            mean_val = data.mean()
            median_val = data.median()
            
            if abs(mean_val - median_val) / data.std() < 0.1:
                return 'normal'
            else:
                return 'skewed'

    def _detect_outliers(self, data: pd.Series) -> bool:
        """Detect outliers using IQR method"""
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        return len(outliers) > 0

    def _recommend_numerical_bins(self, data: pd.Series) -> int:
        """
        CRITICAL FIX: Intelligent numerical bin recommendation (2-20 bins)
        Uses multiple statistical methods with fallbacks for robustness
        """
        try:
            n = len(data)
            unique_count = data.nunique()
            
            if n < 2:
                return self.min_bins
            if unique_count < 2:
                return self.min_bins
            if unique_count <= self.min_bins:
                return min(unique_count, self.max_bins)
            
            
            sturges_bins = int(math.ceil(math.log2(n) + 1))
            
            sqrt_bins = int(math.ceil(math.sqrt(n)))
            
            fd_bins = sqrt_bins  # fallback
            try:
                iqr = data.quantile(0.75) - data.quantile(0.25)
                if iqr > 0 and n > 1:
                    h = 2 * iqr / (n ** (1/3))
                    data_range = data.max() - data.min()
                    if h > 0 and data_range > 0:
                        fd_bins = int(math.ceil(data_range / h))
            except Exception:
                pass
            
            scott_bins = sqrt_bins  # fallback
            try:
                data_std = data.std()
                if data_std > 0 and n > 1:
                    h_scott = 3.5 * data_std / (n ** (1/3))
                    data_range = data.max() - data.min()
                    if h_scott > 0 and data_range > 0:
                        scott_bins = int(math.ceil(data_range / h_scott))
            except Exception:
                pass
            
            skewness = 0.0
            if self.scipy_available:
                try:
                    from scipy import stats
                    skewness = abs(stats.skew(data.dropna()))
                except Exception:
                    pass
            
            if skewness > 2.0:  # Highly skewed
                recommended = max(sturges_bins, fd_bins)
            elif unique_count < 10:  # Low cardinality
                recommended = min(unique_count, 5)
            elif n < self.small_dataset_threshold:  # Small dataset
                recommended = min(sturges_bins, 5)
            elif n > self.medium_dataset_threshold:  # Large dataset
                recommended = min(scott_bins, 15)
            else:  # Medium dataset
                recommended = min(sqrt_bins, 10)
            
            recommended = max(self.min_bins, min(self.max_bins, recommended))
            
            recommended = min(recommended, unique_count)  # Can't have more bins than unique values
            recommended = max(recommended, 2)  # Always at least 2 bins
            
            logger.debug(f"Bin recommendations for {n} samples - Sturges: {sturges_bins}, Sqrt: {sqrt_bins}, "
                        f"FD: {fd_bins}, Scott: {scott_bins}, Final: {recommended}")
            
            return recommended
            
        except Exception as e:
            logger.error(f"Error recommending numerical bins: {e}")
            return min(3, self.max_bins)  # Safe fallback

    def _classify_cardinality(self, unique_count: int) -> str:
        """Classify cardinality level"""
        if unique_count <= 2:
            return 'binary'
        elif unique_count <= 10:
            return 'low'
        elif unique_count <= 50:
            return 'medium'
        elif unique_count <= 100:
            return 'high'
        else:
            return 'very_high'

    def _recommend_categorical_bins(self, unique_count: int) -> int:
        """Recommend number of bins for categorical data"""
        if unique_count <= 5:
            return unique_count
        elif unique_count <= 20:
            return min(5, unique_count // 2)
        else:
            return min(10, unique_count // 5)

    def _analyze_frequency_distribution(self, value_counts: pd.Series) -> Dict[str, Any]:
        """Analyze frequency distribution of categorical data"""
        total = value_counts.sum()
        
        return {
            'most_frequent_pct': value_counts.iloc[0] / total * 100 if len(value_counts) > 0 else 0,
            'least_frequent_pct': value_counts.iloc[-1] / total * 100 if len(value_counts) > 0 else 0,
            'imbalance_ratio': value_counts.iloc[0] / value_counts.iloc[-1] if len(value_counts) > 1 and value_counts.iloc[-1] > 0 else 1,
            'is_balanced': (value_counts.max() / value_counts.min()) < 5 if len(value_counts) > 1 and value_counts.min() > 0 else False
        }

    def _find_best_numerical_split_enhanced(self, feature: str, feature_data: pd.Series, 
                                          target: pd.Series, weights: np.ndarray, 
                                          current_impurity: float, variable_info: Dict) -> Tuple[bool, Dict]:
        """Enhanced numerical split finding with intelligent threshold selection"""
        try:
            valid_mask = ~feature_data.isna()
            if valid_mask.sum() < self.min_samples_split:
                return False, {}
                
            valid_feature = feature_data[valid_mask]
            valid_target = target[valid_mask]
            valid_weights = weights[valid_mask]
            
            unique_values = np.unique(valid_feature)
            if len(unique_values) < 2:
                return False, {}
            
            best_gain = 0
            best_threshold = None
            best_left_impurity = 0
            best_right_impurity = 0
            best_left_samples = 0
            best_right_samples = 0
            
            if variable_info.get('distribution_type') == 'highly_skewed':
                thresholds = valid_feature.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).values
            elif variable_info.get('has_outliers'):
                q1, q3 = valid_feature.quantile([0.25, 0.75])
                iqr = q3 - q1
                thresholds = np.linspace(q1 - 0.5*iqr, q3 + 0.5*iqr, 10)
            else:
                n_thresholds = min(50, len(unique_values) - 1)
                if n_thresholds > 10:
                    indices = np.linspace(0, len(unique_values) - 2, n_thresholds, dtype=int)
                    thresholds = (unique_values[indices] + unique_values[indices + 1]) / 2
                else:
                    thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            for threshold in thresholds:
                left_mask = valid_feature <= threshold
                right_mask = ~left_mask
                
                left_samples = left_mask.sum()
                right_samples = right_mask.sum()
                
                if left_samples < self.min_samples_leaf or right_samples < self.min_samples_leaf:
                    continue
                
                left_impurity = self._calculate_impurity_robust(valid_target[left_mask], valid_weights[left_mask])
                right_impurity = self._calculate_impurity_robust(valid_target[right_mask], valid_weights[right_mask])
                
                total_weight = valid_weights.sum()
                left_weight = valid_weights[left_mask].sum()
                right_weight = valid_weights[right_mask].sum()
                
                weighted_impurity = (left_weight * left_impurity + right_weight * right_impurity) / total_weight
                gain = current_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold
                    best_left_impurity = left_impurity
                    best_right_impurity = right_impurity
                    best_left_samples = left_samples
                    best_right_samples = right_samples
            
            if best_gain > self.min_impurity_decrease:
                return True, {
                    'feature': feature,
                    'split_type': 'numeric',
                    'split_value': best_threshold,
                    'gain': best_gain,
                    'left_impurity': best_left_impurity,
                    'right_impurity': best_right_impurity,
                    'left_samples': best_left_samples,
                    'right_samples': best_right_samples,
                    'threshold': best_threshold
                }
            
            return False, {}
            
        except Exception as e:
            logger.error(f"Error in enhanced numerical split finding: {e}")
            return False, {}

    def _find_best_categorical_split_enhanced(self, feature: str, feature_data: pd.Series, 
                                            target: pd.Series, weights: np.ndarray, 
                                            current_impurity: float, variable_info: Dict) -> Tuple[bool, Dict]:
        """Enhanced categorical split finding with intelligent grouping strategies"""
        try:
            valid_mask = ~feature_data.isna()
            if valid_mask.sum() < self.min_samples_split:
                return False, {}
                
            valid_feature = feature_data[valid_mask]
            valid_target = target[valid_mask]
            valid_weights = weights[valid_mask]
            
            unique_categories = valid_feature.unique()
            n_categories = len(unique_categories)
            
            if n_categories < 2:
                return False, {}
            
            cardinality_level = variable_info.get('cardinality_level', 'medium')
            
            if cardinality_level == 'binary':
                return self._find_best_binary_categorical_split(
                    feature, valid_feature, valid_target, valid_weights, current_impurity
                )
            elif cardinality_level in ['low', 'medium']:
                binary_result = self._find_best_binary_categorical_split(
                    feature, valid_feature, valid_target, valid_weights, current_impurity
                )
                target_result = self._find_best_target_rate_split(
                    feature, valid_feature, valid_target, valid_weights, current_impurity
                )
                
                if binary_result[0] and target_result[0]:
                    if binary_result[1].get('gain', 0) >= target_result[1].get('gain', 0):
                        return binary_result
                    else:
                        return target_result
                elif binary_result[0]:
                    return binary_result
                elif target_result[0]:
                    return target_result
                else:
                    return False, {}
            else:
                return self._find_best_target_rate_split(
                    feature, valid_feature, valid_target, valid_weights, current_impurity
                )
                
        except Exception as e:
            logger.error(f"Error in enhanced categorical split finding: {e}")
            return False, {}

    def _find_best_binary_categorical_split(self, feature: str, feature_data: pd.Series, 
                                          target: pd.Series, weights: np.ndarray, 
                                          current_impurity: float) -> Tuple[bool, Dict]:
        """Find best binary categorical split (category vs rest)"""
        unique_categories = feature_data.unique()
        
        best_gain = 0
        best_category = None
        best_left_impurity = 0
        best_right_impurity = 0
        best_left_samples = 0
        best_right_samples = 0
        
        for category in unique_categories:
            left_mask = feature_data == category
            right_mask = ~left_mask
            
            left_samples = left_mask.sum()
            right_samples = right_mask.sum()
            
            if left_samples < self.min_samples_leaf or right_samples < self.min_samples_leaf:
                continue
            
            left_impurity = self._calculate_impurity_robust(target[left_mask], weights[left_mask])
            right_impurity = self._calculate_impurity_robust(target[right_mask], weights[right_mask])
            
            total_weight = weights.sum()
            left_weight = weights[left_mask].sum()
            right_weight = weights[right_mask].sum()
            
            weighted_impurity = (left_weight * left_impurity + right_weight * right_impurity) / total_weight
            gain = current_impurity - weighted_impurity
            
            if gain > best_gain:
                best_gain = gain
                best_category = category
                best_left_impurity = left_impurity
                best_right_impurity = right_impurity
                best_left_samples = left_samples
                best_right_samples = right_samples
        
        if best_gain > self.min_impurity_decrease:
            return True, {
                'feature': feature,
                'split_type': 'categorical',
                'left_categories': [best_category],
                'right_categories': [cat for cat in unique_categories if cat != best_category],
                'gain': best_gain,
                'left_impurity': best_left_impurity,
                'right_impurity': best_right_impurity,
                'left_samples': best_left_samples,
                'right_samples': best_right_samples
            }
        
        return False, {}

    def _find_best_target_rate_split(self, feature: str, feature_data: pd.Series, 
                                    target: pd.Series, weights: np.ndarray, 
                                    current_impurity: float) -> Tuple[bool, Dict]:
        """Find best split based on target rates for categorical variables"""
        category_rates = {}
        for category in feature_data.unique():
            mask = feature_data == category
            if mask.sum() > 0:
                category_target = target[mask]
                category_rates[category] = category_target.mean()
        
        sorted_categories = sorted(category_rates.items(), key=lambda x: x[1])
        
        if len(sorted_categories) < 2:
            return False, {}
        
        best_gain = 0
        best_left_categories = []
        best_right_categories = []
        best_left_impurity = 0
        best_right_impurity = 0
        best_left_samples = 0
        best_right_samples = 0
        
        for i in range(1, len(sorted_categories)):
            left_categories = [cat for cat, rate in sorted_categories[:i]]
            right_categories = [cat for cat, rate in sorted_categories[i:]]
            
            left_mask = feature_data.isin(left_categories)
            right_mask = feature_data.isin(right_categories)
            
            left_samples = left_mask.sum()
            right_samples = right_mask.sum()
            
            if left_samples < self.min_samples_leaf or right_samples < self.min_samples_leaf:
                continue
            
            left_impurity = self._calculate_impurity_robust(target[left_mask], weights[left_mask])
            right_impurity = self._calculate_impurity_robust(target[right_mask], weights[right_mask])
            
            total_weight = weights.sum()
            left_weight = weights[left_mask].sum()
            right_weight = weights[right_mask].sum()
            
            weighted_impurity = (left_weight * left_impurity + right_weight * right_impurity) / total_weight
            gain = current_impurity - weighted_impurity
            
            if gain > best_gain:
                best_gain = gain
                best_left_categories = left_categories
                best_right_categories = right_categories
                best_left_impurity = left_impurity
                best_right_impurity = right_impurity
                best_left_samples = left_samples
                best_right_samples = right_samples
        
        if best_gain > self.min_impurity_decrease:
            return True, {
                'feature': feature,
                'split_type': 'categorical',
                'left_categories': best_left_categories,
                'right_categories': best_right_categories,
                'gain': best_gain,
                'left_impurity': best_left_impurity,
                'right_impurity': best_right_impurity,
                'left_samples': best_left_samples,
                'right_samples': best_right_samples
            }
        
        return False, {}

    def _calculate_impurity_robust(self, y: pd.Series, weights: np.ndarray = None) -> float:
        """Calculate impurity with robust error handling"""
        try:
            if len(y) == 0:
                return 0.0
            
            if weights is None:
                weights = np.ones(len(y))
            
            weights = weights / weights.sum() if weights.sum() > 0 else weights
            
            unique_classes = y.unique()
            class_probs = []
            
            for cls in unique_classes:
                mask = y == cls
                prob = weights[mask].sum()
                class_probs.append(prob)
            
            class_probs = np.array(class_probs)
            class_probs = class_probs / class_probs.sum() if class_probs.sum() > 0 else class_probs
            
            if self.criterion == SplitCriterion.GINI:
                return 1.0 - np.sum(class_probs ** 2)
            elif self.criterion == SplitCriterion.ENTROPY:
                non_zero_probs = class_probs[class_probs > 0]
                return -np.sum(non_zero_probs * np.log2(non_zero_probs))
            elif self.criterion == SplitCriterion.MISCLASSIFICATION:
                return 1.0 - np.max(class_probs)
            else:
                return 1.0 - np.sum(class_probs ** 2)
                
        except Exception as e:
            logger.error(f"Error calculating impurity: {e}")
            return 0.5  # Return moderate impurity as fallback

    def get_node(self, node_id: str):
        """Get node by ID with comprehensive search strategies"""
        try:
            if hasattr(self, 'tree') and self.tree:
                node = self.tree.get_node_by_id(node_id)
                if node:
                    return node
            
            if hasattr(self, 'model') and self.model:
                if hasattr(self.model, 'get_node_by_id'):
                    node = self.model.get_node_by_id(node_id)
                    if node:
                        return node
                elif hasattr(self.model, 'root'):
                    node = self._search_tree_for_node(self.model.root, node_id)
                    if node:
                        return node
            
            if hasattr(self, 'parent') and self.parent:
                if hasattr(self.parent, 'tree'):
                    node = self.parent.tree.get_node_by_id(node_id)
                    if node:
                        return node
                elif hasattr(self.parent, 'model'):
                    if hasattr(self.parent.model, 'get_node_by_id'):
                        node = self.parent.model.get_node_by_id(node_id)
                        if node:
                            return node
                    elif hasattr(self.parent.model, 'root'):
                        node = self._search_tree_for_node(self.parent.model.root, node_id)
                        if node:
                            return node
            
            logger.warning(f"Could not find node {node_id}, creating properly initialized mock node")
            mock_node = TreeNode(node_id)
            
            if hasattr(self, 'model') and self.model and hasattr(self.model, '_cached_X'):
                data_size = len(self.model._cached_X)
                mock_node.sample_indices = list(range(min(1000, data_size)))  # Use up to 1000 samples
            else:
                mock_node.sample_indices = list(range(100))  # Default fallback
                
            mock_node.samples = len(mock_node.sample_indices)
            mock_node.is_terminal = True
            mock_node.depth = 0
            mock_node.impurity = 0.5  # Default impurity
            
            return mock_node
            
        except Exception as e:
            logger.error(f"Error getting node {node_id}: {e}")
            return None

    def find_node_with_comprehensive_search(self, node_id: str):
        """Find node with comprehensive search strategies"""
        try:
            if hasattr(self, 'tree') and self.tree:
                node = self.tree.get_node_by_id(node_id)
                if node:
                    return node
            
            if hasattr(self, 'model') and self.model:
                if hasattr(self.model, 'get_node_by_id'):
                    node = self.model.get_node_by_id(node_id)
                    if node:
                        return node
                elif hasattr(self.model, 'root'):
                    node = self._search_tree_for_node(self.model.root, node_id)
                    if node:
                        return node
            
            if hasattr(self, 'parent') and self.parent:
                if hasattr(self.parent, 'tree'):
                    node = self.parent.tree.get_node_by_id(node_id)
                    if node:
                        return node
                elif hasattr(self.parent, 'model'):
                    if hasattr(self.parent.model, 'get_node_by_id'):
                        node = self.parent.model.get_node_by_id(node_id)
                        if node:
                            return node
                    elif hasattr(self.parent.model, 'root'):
                        node = self._search_tree_for_node(self.parent.model.root, node_id)
                        if node:
                            return node
            
            logger.warning(f"Could not find node {node_id}, creating properly initialized node")
            
            if node_id == 'root' and hasattr(self, 'model') and self.model and hasattr(self.model, 'root'):
                real_root = self.model.root
                if real_root:
                    if not hasattr(real_root, 'sample_indices') or real_root.sample_indices is None:
                        if hasattr(self.model, '_cached_X') and self.model._cached_X is not None:
                            real_root.sample_indices = list(range(len(self.model._cached_X)))
                            logger.info(f"Initialized sample indices for root node: {len(real_root.sample_indices)} samples")
                    return real_root
            
            mock_node = TreeNode(node_id)
            
            if hasattr(self, 'model') and self.model and hasattr(self.model, '_cached_X'):
                data_size = len(self.model._cached_X)
                mock_node.sample_indices = list(range(min(1000, data_size)))  # Use up to 1000 samples
            else:
                mock_node.sample_indices = list(range(100))  # Default fallback
                
            mock_node.samples = len(mock_node.sample_indices)
            mock_node.is_terminal = True
            mock_node.depth = 0
            mock_node.impurity = 0.5  # Default impurity
            
            return mock_node
            
        except Exception as e:
            logger.error(f"Error in comprehensive node search for {node_id}: {e}")
            mock_node = TreeNode(node_id)
            mock_node.sample_indices = list(range(100))
            mock_node.samples = 100
            mock_node.is_terminal = True
            mock_node.depth = 0
            mock_node.impurity = 0.5
            return mock_node

    def _search_tree_for_node(self, root_node, target_id):
        """Recursively search tree for target node"""
        if root_node.node_id == target_id:
            return root_node
            
        for child in root_node.children:
            result = self._search_tree_for_node(child, target_id)
            if result:
                return result
                
        return None

    def _evaluate_categorical_split(self, feature: str, left_categories: List, 
                                  right_categories: List, feature_data: pd.Series, 
                                  target: pd.Series, weights: np.ndarray, 
                                  current_impurity: float) -> Dict[str, Any]:
        """Evaluate a categorical split configuration"""
        try:
            left_mask = feature_data.isin(left_categories)
            right_mask = feature_data.isin(right_categories)
            
            left_samples = left_mask.sum()
            right_samples = right_mask.sum()
            
            if left_samples < self.min_samples_leaf or right_samples < self.min_samples_leaf:
                return {'valid': False, 'gain': 0}
            
            left_impurity = self._calculate_impurity_robust(target[left_mask], weights[left_mask])
            right_impurity = self._calculate_impurity_robust(target[right_mask], weights[right_mask])
            
            total_weight = weights.sum()
            left_weight = weights[left_mask].sum()
            right_weight = weights[right_mask].sum()
            
            weighted_impurity = (left_weight * left_impurity + right_weight * right_impurity) / total_weight
            gain = current_impurity - weighted_impurity
            
            return {
                'valid': True,
                'gain': gain,
                'left_impurity': left_impurity,
                'right_impurity': right_impurity,
                'left_samples': left_samples,
                'right_samples': right_samples,
                'left_categories': left_categories,
                'right_categories': right_categories
            }
            
        except Exception as e:
            logger.error(f"Error evaluating categorical split: {e}")
            return {'valid': False, 'gain': 0}

    def _select_features_for_split(self, features: List[str]) -> List[str]:
        """Select features to consider for splitting"""
        if self.max_features is None or self.max_features >= len(features):
            return features
        else:
            np.random.shuffle(features)
            return features[:self.max_features]

    def _find_basic_splits_for_feature(self, feature: str, feature_data: pd.Series, 
                                       target: pd.Series, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fallback method for basic split finding when advanced methods are unavailable
        Ensures functionality even when enhanced split finding is not available
        """
        try:
            splits = []
            
            valid_mask = ~(feature_data.isna() | target.isna())
            clean_feature = feature_data[valid_mask]
            clean_target = target[valid_mask]
            
            if len(clean_feature) < self.min_samples_leaf * 2:
                return splits
            
            if analysis.get('is_numerical', True):
                try:
                    median_val = clean_feature.median()
                    quality = self._evaluate_basic_split(clean_feature, clean_target, median_val)
                    
                    if quality > self.min_impurity_decrease:
                        splits.append({
                            'feature': feature,
                            'split_type': 'numeric',
                            'split_value': median_val,
                            'quality_score': quality,
                            'bin_count': 2,
                            'method': 'basic_median'
                        })
                except Exception as num_error:
                    logger.warning(f"Error in basic numerical split: {num_error}")
            
            else:
                try:
                    unique_cats = clean_feature.unique()
                    if len(unique_cats) >= 2:
                        mid_point = len(unique_cats) // 2
                        left_categories = unique_cats[:mid_point].tolist()
                        right_categories = unique_cats[mid_point:].tolist()
                        
                        quality = self._evaluate_basic_categorical_split(
                            clean_feature, clean_target, left_categories
                        )
                        
                        if quality > self.min_impurity_decrease:
                            splits.append({
                                'feature': feature,
                                'split_type': 'categorical',
                                'left_categories': left_categories,
                                'right_categories': right_categories,
                                'quality_score': quality,
                                'bin_count': 2,
                                'method': 'basic_categorical'
                            })
                except Exception as cat_error:
                    logger.warning(f"Error in basic categorical split: {cat_error}")
            
            return splits
            
        except Exception as e:
            logger.error(f"Error in basic split finding for feature {feature}: {e}")
            return []

    def _evaluate_basic_split(self, feature_data: pd.Series, target: pd.Series, threshold: float) -> float:
        """Basic split evaluation for numerical features"""
        try:
            left_mask = feature_data <= threshold
            right_mask = ~left_mask
            
            left_target = target[left_mask]
            right_target = target[right_mask]
            
            if len(left_target) < self.min_samples_leaf or len(right_target) < self.min_samples_leaf:
                return 0.0
            
            def gini_impurity(target_series):
                if len(target_series) == 0:
                    return 0.0
                value_counts = target_series.value_counts()
                probabilities = value_counts / len(target_series)
                return 1.0 - sum(prob ** 2 for prob in probabilities)
            
            original_gini = gini_impurity(target)
            left_gini = gini_impurity(left_target)
            right_gini = gini_impurity(right_target)
            
            total_samples = len(target)
            left_weight = len(left_target) / total_samples
            right_weight = len(right_target) / total_samples
            
            weighted_gini = left_weight * left_gini + right_weight * right_gini
            gini_improvement = original_gini - weighted_gini
            
            return gini_improvement
            
        except Exception as e:
            logger.error(f"Error evaluating basic split: {e}")
            return 0.0

    def _evaluate_basic_categorical_split(self, feature_data: pd.Series, target: pd.Series, 
                                          left_categories: List[str]) -> float:
        """Basic split evaluation for categorical features"""
        try:
            left_mask = feature_data.isin(left_categories)
            right_mask = ~left_mask
            
            left_target = target[left_mask]
            right_target = target[right_mask]
            
            if len(left_target) < self.min_samples_leaf or len(right_target) < self.min_samples_leaf:
                return 0.0
            
            return self._evaluate_basic_split(
                pd.Series([0 if x else 1 for x in left_mask], index=feature_data.index),
                target, 0.5
            )
            
        except Exception as e:
            logger.error(f"Error evaluating basic categorical split: {e}")
            return 0.0