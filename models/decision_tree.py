#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Decision Tree Module for Bespoke Utility
Implements the core decision tree model
"""

import logging
import json
import os
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Callable
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime
import uuid

from PyQt5.QtCore import QObject, pyqtSignal
from joblib import Parallel, delayed

from models.node import TreeNode
from utils.memory_management import optimize_dataframe
from utils.serialization_utils import make_json_serializable

try:
    from models.split_configuration import SplitConfiguration, SplitConfigurationFactory, ValidationResult
    from models.split_validator import SplitValidator
    from models.split_transaction import SplitTransaction, atomic_split_operation, safe_apply_split
    ENHANCED_VALIDATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced validation components not available: {e}")
    ENHANCED_VALIDATION_AVAILABLE = False

logger = logging.getLogger(__name__)

class SplitCriterion(Enum):
    """Enumeration of splitting criteria for decision trees"""
    GINI = "gini"
    ENTROPY = "entropy"
    INFORMATION_GAIN = "information_gain"
    MISCLASSIFICATION = "misclassification"

class TreeGrowthMode(Enum):
    """Enumeration of tree growth modes"""
    AUTOMATIC = "automatic"  # Grow tree automatically
    MANUAL = "manual"        # Grow tree manually (user-guided)
    HYBRID = "hybrid"        # Start automatic, then allow manual refinement

class BespokeDecisionTree(QObject):  # Re-enable QObject inheritance for signals
    """
    Bespoke Decision Tree model for binary classification
    Designed for interactive use in credit risk modeling
    """
    
    treeUpdated = pyqtSignal()
    nodeUpdated = pyqtSignal(str)  # node_id
    splitFound = pyqtSignal(str, list)  # node_id, splits
    errorOccurred = pyqtSignal(str)
    trainingProgress = pyqtSignal(int, str)  # progress %, message
    trainingRequired = pyqtSignal(str)  # message
    trainingComplete = pyqtSignal()
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the decision tree
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super().__init__()  # Re-enable QObject init
        self.config = config or {}
        
        self.model_id = str(uuid.uuid4())
        self.model_name = f"Decision_Tree_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.root = TreeNode(node_id="root", depth=0)
        
        self._set_default_parameters()
        
        self._target_name = None
        self.target_values = None
        self.class_names = []
        self.positive_class = None  # For binary classification
        
        self.feature_names = []
        self.categorical_features = set()
        self.numerical_features = set()
        self.training_samples = 0
        
        self.feature_importance = {}
        
        self.metrics = {}
        
        self.max_node_id = 0   # Track highest node ID assigned
        self.num_nodes = 1     # Start with root node
        self.num_leaves = 1    # Start with root as a leaf
        self.max_depth = 0     # Current max depth
        
        self.history = []
        
        self.is_fitted = False
        
        self.active_node_id = "root"
        
        self._cached_X = None
        self._cached_y = None
        self._cached_sample_weight = None
        
        if not hasattr(self, 'VALID_SPLIT_TYPES'):
            self.VALID_SPLIT_TYPES = {
                'numeric', 'numerical',  # Both variants for backward compatibility
                'categorical', 
                'numeric_multi_bin', 'numerical_multi_bin',  # Multi-bin support
                'categorical_multi_bin'  # CRITICAL: Add missing multi-bin categorical
            }
        
        try:
            from models.split_configuration import SplitConfiguration, SplitConfigurationFactory, ValidationResult
            from models.split_validator import SplitValidator
            from models.split_transaction import SplitTransaction, atomic_split_operation, safe_apply_split
            ENHANCED_VALIDATION_AVAILABLE = True
            if not hasattr(self, 'validator'):
                self.validator = SplitValidator()
            self._use_enhanced_validation = True
            logger.info("Enhanced validation enabled for decision tree")
        except ImportError as e:
            ENHANCED_VALIDATION_AVAILABLE = False
            self.validator = None
            self._use_enhanced_validation = False
            logger.warning(f"Enhanced validation components not available: {e}")
    
    @property
    def target_name(self):
        """Get the target variable name"""
        return self._target_name
    
    @target_name.setter
    def target_name(self, value):
        """Set the target variable name and clear cached data if changed"""
        if self._target_name != value:
            logger.info(f"Target variable changed from '{self._target_name}' to '{value}', clearing cache")
            self._target_name = value
            self._cached_X = None
            self._cached_y = None
            self._cached_sample_weight = None
            
            should_reset_tree = False  # Disable automatic tree resets for now
            
            if should_reset_tree:
                logger.info("Resetting tree structure due to target variable change")
                self.root = TreeNode(node_id="root", depth=0)
                self.num_nodes = 1
                self.num_leaves = 1
                self.max_depth = 0
                self.is_fitted = False
            else:
                if (self._target_name is not None and 
                    self.is_fitted and 
                    hasattr(self, 'num_nodes') and self.num_nodes > 1):
                    logger.warning(f"Target name change from '{self._target_name}' to '{value}' on fitted model with {self.num_nodes} nodes")
                elif self._target_name is not None:
                    logger.info(f"Updating target name from '{self._target_name}' to '{value}'")
        else:
            self._target_name = value
    
    def set_target_name_safe(self, value):
        """
        Set target name without triggering tree resets - for workflow compatibility
        """
        if self._target_name != value:
            logger.info(f"Safely updating target name from '{self._target_name}' to '{value}' (no reset)")
            self._target_name = value
    
    def _set_default_parameters(self):
        """Set default model parameters from config or built-in defaults"""
        params = self.config.get('decision_tree', {})
        
        self.criterion = SplitCriterion(params.get('criterion', 'entropy'))
        self.max_depth = params.get('max_depth', 10)
        self.min_samples_split = params.get('min_samples_split', 2)
        self.min_samples_leaf = params.get('min_samples_leaf', 1)
        self.min_impurity_decrease = params.get('min_impurity_decrease', 0.0)
        self.max_features = params.get('max_features', None)  # None = use all features
        self.max_leaf_nodes = params.get('max_leaf_nodes', None)  # None = unlimited
        
        self.growth_mode = TreeGrowthMode(params.get('growth_mode', 'automatic'))
        
        self.class_weight = params.get('class_weight', None)  # None = balanced
        
        self.random_state = params.get('random_state', 42)
        np.random.seed(self.random_state)
        
        self.pruning_enabled = params.get('pruning_enabled', True)
        self.pruning_method = params.get('pruning_method', 'cost_complexity')
        self.pruning_alpha = params.get('pruning_alpha', 0.01)  # Complexity parameter
        
        self.use_memory_optimization = params.get('use_memory_optimization', True)
        self.chunk_size = params.get('chunk_size', 50000)
    
    def set_params(self, **params):
        """
        Set model parameters
        
        Args:
            **params: Parameters to set (e.g., criterion='gini', max_depth=5)
        """
        for param, value in params.items():
            if param == 'criterion':
                self.criterion = SplitCriterion(value)
            elif param == 'growth_mode':
                self.growth_mode = TreeGrowthMode(value)
            elif hasattr(self, param):
                setattr(self, param, value)
            else:
                logger.warning(f"Unknown parameter: {param}")
        
        logger.info(f"Updated model parameters: {', '.join([f'{p}={v}' for p, v in params.items()])}")
        
        
        return self
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get model parameters
        
        Args:
            deep: Whether to include nested parameters (ignored)
            
        Returns:
            Dictionary of parameter names and values
        """
        params = {
            'criterion': self.criterion.value,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_impurity_decrease': self.min_impurity_decrease,
            'max_features': self.max_features,
            'max_leaf_nodes': self.max_leaf_nodes,
            'growth_mode': self.growth_mode.value,
            'class_weight': self.class_weight,
            'random_state': self.random_state,
            'pruning_enabled': self.pruning_enabled,
            'pruning_method': self.pruning_method,
            'pruning_alpha': self.pruning_alpha
        }
        
        return params
    
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None) -> 'BespokeDecisionTree':
        """
        Fit the decision tree to the data
        
        Args:
            X: Training features
            y: Target variable (binary)
            sample_weight: Sample weights (optional)
            
        Returns:
            Self (fitted model)
        """
        try:
            X, y, sample_weight = self._validate_inputs(X, y, sample_weight)
            
            self.feature_names = list(X.columns)
            
            if not self.target_name or (y.name and y.name != self.target_name):
                self.target_name = y.name  # This will use the property setter
            
            self.target_values = sorted(y.unique())
            
            self._detect_feature_types(X)
            
            if len(self.target_values) == 2:
                self.class_names = [str(val) for val in self.target_values]
                self.positive_class = self.class_names[1]  # Convention: second class is positive
                logger.info(f"Binary classification detected: {self.class_names}")
            elif len(self.target_values) > 2:
                self.class_names = [str(val) for val in self.target_values]
                self.positive_class = None  # No single positive class in multi-class
                logger.info(f"Multi-class classification detected: {len(self.target_values)} classes")
            else:
                raise ValueError(f"Target variable must have at least 2 unique values, got {len(self.target_values)}: {self.target_values}")
            
            self.training_samples = len(X)
            
            self._cached_X = X.copy()
            self._cached_y = y.copy()
            self._cached_sample_weight = sample_weight.copy()
            
            if self.is_fitted:
                self.root = TreeNode(node_id="root", depth=0)
                self.active_node_id = "root"
                self.max_node_id = 0
                self.num_nodes = 1
                self.num_leaves = 1
                self.max_depth = 0
            
            
            if self.growth_mode == TreeGrowthMode.AUTOMATIC:
                self._grow_tree_automatic(X, y, sample_weight)
            elif self.growth_mode == TreeGrowthMode.MANUAL:
                self._initialize_root_node(X, y, sample_weight)
            elif self.growth_mode == TreeGrowthMode.HYBRID:
                hybrid_depth = max(1, self.max_depth // 2)
                old_max_depth = self.max_depth
                self.max_depth = hybrid_depth
                self._grow_tree_automatic(X, y, sample_weight)
                self.max_depth = old_max_depth
            
            
            self._calculate_feature_importance()
            
            self.is_fitted = True
            
            
            self._add_to_history("fit", {"samples": len(X), "features": len(self.feature_names)})
            
            
            return self
            
        except Exception as e:
            error_msg = f"Error fitting decision tree: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise
    
    def _validate_inputs(self, X: pd.DataFrame, y: pd.Series, 
                        sample_weight: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Validate and prepare input data
        
        Args:
            X: Training features
            y: Target variable
            sample_weight: Sample weights
            
        Returns:
            Tuple of (X, y, sample_weight) with validated data
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")
        
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length, got {len(X)} and {len(y)}")
        
        if X.isna().any().any():
            logger.warning(f"X contains {X.isna().sum().sum()} missing values")
            
            missing_strategy = self.config.get('missing_values', {}).get('strategy', 'raise')
            
            if missing_strategy == 'raise':
                logger.warning("Missing values detected - switching to automatic handling instead of raising error")
                missing_strategy = 'fill'  # Automatically switch to fill strategy
            elif missing_strategy == 'drop':
                mask = ~X.isna().any(axis=1)
                X = X[mask]
                y = y[mask]
                if sample_weight is not None:
                    sample_weight = sample_weight[mask]
                
                logger.info(f"Dropped {len(mask) - sum(mask)} rows with missing values")
            elif missing_strategy == 'impute':
                impute_method = self.config.get('missing_values', {}).get('impute_method', 'mean')
                
                for col in X.columns:
                    if X[col].isna().any():
                        if pd.api.types.is_numeric_dtype(X[col]):
                            if impute_method == 'mean':
                                X[col] = X[col].fillna(X[col].mean())
                            elif impute_method == 'median':
                                X[col] = X[col].fillna(X[col].median())
                            elif impute_method == 'zero':
                                X[col] = X[col].fillna(0)
                        else:
                            X[col] = X[col].fillna(X[col].mode()[0])
                
                logger.info(f"Imputed missing values using {impute_method} method")
            else:
                raise ValueError(f"Unknown missing value strategy: {missing_strategy}")
        
        if y.isna().any():
            logger.warning(f"y contains {y.isna().sum()} missing values")
            
            mask = ~y.isna()
            X = X[mask]
            y = y[mask]
            if sample_weight is not None:
                sample_weight = sample_weight[mask]
            
            logger.info(f"Dropped {len(mask) - sum(mask)} rows with missing target")
        
        if sample_weight is None:
            if self.class_weight:
                if self.class_weight == 'balanced':
                    class_counts = y.value_counts()
                    n_samples = len(y)
                    weights = {cls: n_samples / (len(class_counts) * count) 
                              for cls, count in class_counts.items()}
                    
                    sample_weight = np.array([weights[cls] for cls in y])
                else:
                    weights = self.class_weight
                    sample_weight = np.array([weights.get(cls, 1.0) for cls in y])
            else:
                sample_weight = np.ones(len(y))
        
        if self.use_memory_optimization:
            X = optimize_dataframe(X)
        
        return X, y, sample_weight
    
    def _detect_feature_types(self, X: pd.DataFrame):
        """
        Detect categorical and numerical features
        
        Args:
            X: Feature DataFrame
        """
        self.categorical_features = set()
        self.numerical_features = set()
        
        for col in X.columns:
            if isinstance(X[col].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(X[col]):
                self.categorical_features.add(col)
            else:
                self.numerical_features.add(col)
        
        logger.info(f"Detected {len(self.numerical_features)} numerical features and "
                   f"{len(self.categorical_features)} categorical features")
    
    def _initialize_root_node(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None):
        """
        CRITICAL FIX: Initialize root node with correct TreeNode.update_stats() call
        """
        try:
            if self.root is None:
                self.root = TreeNode(node_id='root', parent=None, depth=0, is_terminal=True)
                if not hasattr(self, 'nodes'):
                    self.nodes = {}
                self.nodes['root'] = self.root
            
            class_counts = y.value_counts().to_dict()
            sample_count = len(y)
            majority_class = y.mode().iloc[0] if len(y) > 0 else None
            
            self.root.update_stats(
                sample_count=sample_count,
                class_counts=class_counts,
                majority_class=majority_class
            )
            
            if not hasattr(self, 'node_count'):
                self.node_count = 0
            if not hasattr(self, 'tree_depth'):
                self.tree_depth = 0
                
            self.node_count = 1
            self.tree_depth = 0
            
            logger.info(f"Root node initialized with {sample_count} samples, majority class: {majority_class}")
            
        except Exception as e:
            logger.error(f"Error initializing root node: {e}")
            raise
    
    def _initialize_root_node_with_data(self, X, y):
        """Initialize root node with proper sample indices and statistics"""
        try:
            self._cached_X = X.copy()
            self._cached_y = y.copy()
            
            if not self.root:
                self.root = TreeNode('root')
            
            self.root.sample_indices = list(range(len(X)))
            self.root.samples = len(X)
            self.root.depth = 0
            self.root.is_terminal = True
            self.root.parent = None
            
            if hasattr(y, 'value_counts'):
                class_counts = y.value_counts().to_dict()
            else:
                unique, counts = np.unique(y, return_counts=True)
                class_counts = dict(zip(unique, counts))
                
            self.root.class_counts = class_counts
            
            total = sum(class_counts.values())
            if total > 0:
                gini = 1.0 - sum((count/total)**2 for count in class_counts.values())
                self.root.impurity = gini
            else:
                self.root.impurity = 0.0
                
            if class_counts:
                self.root.prediction = max(class_counts.items(), key=lambda x: x[1])[0]
                
            logger.info(f"Initialized root node: {self.root.samples} samples, class counts: {class_counts}, impurity: {self.root.impurity:.4f}")
            
        except Exception as e:
            logger.error(f"Error initializing root node: {e}", exc_info=True)
            raise
    
    def _grow_tree_automatic(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray):
        """
        Automatically grow the decision tree
        
        Args:
            X: Training features
            y: Target variable
            sample_weight: Sample weights
        """
        self._initialize_root_node(X, y, sample_weight)
        
        nodes_to_split = [self.root]
        
        nodes_processed = 0
        
        while nodes_to_split:
            node = nodes_to_split.pop(0)
            
            nodes_processed += 1
            progress = min(99, int(nodes_processed / (2 * len(X) / self.min_samples_split) * 100))
            
            if self._should_stop_splitting(node):
                node.is_terminal = True
                continue
            
            split_found, split_info = self._find_best_split(X, y, sample_weight, node)
            
            if not split_found:
                node.is_terminal = True
                continue
            
            left_indices, right_indices = self._apply_split(X, split_info)
            
            self._create_children_from_split(X, y, sample_weight, node, split_info, left_indices, right_indices)
            
            for child in node.children:
                if not child.is_terminal and not self._should_stop_splitting(child):
                    nodes_to_split.append(child)
            
            self.num_nodes = self.num_nodes + len(node.children) - 1
            self.num_leaves = sum(1 for n in self._get_all_nodes() if n.is_terminal)
            self.max_depth = max(n.depth for n in self._get_all_nodes())
            
        
        logger.info(f"Automatically grew decision tree: {self.num_nodes} nodes, "
                   f"{self.num_leaves} leaves, max depth {self.max_depth}")
    
    def _should_stop_splitting(self, node: TreeNode) -> bool:
        """
        Check if node splitting should stop based on stopping criteria
        
        Args:
            node: Node to check
            
        Returns:
            True if splitting should stop, False otherwise
        """
        if self.max_depth is not None and node.depth >= self.max_depth:
            return True
        
        if node.samples < self.min_samples_split:
            return True
        
        if node.impurity <= 0.0001:  # Near-zero impurity
            return True
        
        if len(node.class_counts) <= 1:
            return True
        
        if (self.max_leaf_nodes is not None and 
            self.num_leaves >= self.max_leaf_nodes):
            return True
        
        return False
    
    def _find_best_split(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray, 
                       node: TreeNode) -> Tuple[bool, Dict[str, Any]]:
        """
        Find the best split for a node
        
        Args:
            X: Training features
            y: Target variable
            sample_weight: Sample weights
            node: Node to split
            
        Returns:
            Tuple of (split_found, split_info)
        """
        indices = self._get_node_sample_indices(X, node)
        
        if len(indices) <= 1:
            return False, {}
        
        best_gain = -float('inf')
        best_split = {}
        
        features_to_consider = self._select_features_for_split()
        
        
        for feature in features_to_consider:
            if X[feature].iloc[indices].isna().all():
                continue
            
            if feature in self.categorical_features:
                split_found, split_info = self._find_best_categorical_split(
                    X, y, sample_weight, feature, indices, node.impurity
                )
            else:
                split_found, split_info = self._find_best_numerical_split(
                    X, y, sample_weight, feature, indices, node.impurity
                )
            
            if split_found and split_info['gain'] > best_gain:
                best_gain = split_info['gain']
                best_split = split_info
        
        if best_gain > self.min_impurity_decrease and best_split:
            logger.debug(f"Found valid split with gain {best_gain:.6f} > threshold {self.min_impurity_decrease}")
            return True, best_split
        else:
            logger.debug(f"No valid split found. Best gain: {best_gain:.6f}, threshold: {self.min_impurity_decrease}, has_split: {bool(best_split)}")
            return False, {}
    
    def _select_features_for_split(self) -> List[str]:
        """
        Select features to consider for splitting
        
        Returns:
            List of feature names to consider
        """
        if self.max_features is None:
            return self.feature_names
        
        if isinstance(self.max_features, int):
            n_features = min(self.max_features, len(self.feature_names))
            return list(np.random.choice(self.feature_names, n_features, replace=False))
        
        if isinstance(self.max_features, float):
            n_features = max(1, int(self.max_features * len(self.feature_names)))
            return list(np.random.choice(self.feature_names, n_features, replace=False))
        
        return self.feature_names
    
    def _get_node_sample_indices(self, X: pd.DataFrame, node: TreeNode) -> List[int]:
        """
        Get indices of samples that reach this node with optimized data propagation
        
        Args:
            X: Feature DataFrame
            node: Node to get samples for
            
        Returns:
            List of sample indices
        """
        if node.node_id == "root" or node.parent is None:
            return list(range(len(X)))
        
        indices = np.arange(len(X))  # Use numpy array for efficiency
        
        path = []
        current = node
        while current.parent is not None:
            path.append((current.parent, current))
            current = current.parent
        
        path.reverse()
        
        for parent, child in path:
            if not parent.split_feature or parent.split_feature not in X.columns:
                continue
            
            try:
                child_idx = parent.children.index(child)
            except ValueError:
                logger.warning(f"Child {child.node_id} not found in parent {parent.node_id} children")
                continue
            
            feature_values = X[parent.split_feature].iloc[indices]
            
            if parent.split_type == 'numeric':
                valid_mask = ~pd.isna(feature_values)
                
                if not valid_mask.any():
                    if child_idx != 0:
                        indices = np.array([], dtype=int)
                    continue
                
                threshold = parent.split_value
                
                if child_idx == 0:  # Left child (<=)
                    mask = (feature_values <= threshold) | ~valid_mask
                else:  # Right child (>)
                    mask = valid_mask & (feature_values > threshold)
                
                indices = indices[mask.values]
                
            elif parent.split_type == 'numeric_multi_bin':
                valid_mask = ~pd.isna(feature_values)
                
                if not valid_mask.any():
                    if child_idx != 0:
                        indices = np.array([], dtype=int)
                    continue
                
                if hasattr(parent, 'split_thresholds') and parent.split_thresholds:
                    thresholds = parent.split_thresholds
                    
                    if child_idx == 0:
                        mask = (feature_values <= thresholds[0]) | ~valid_mask
                    elif child_idx == len(thresholds):
                        mask = valid_mask & (feature_values > thresholds[-1])
                    elif child_idx < len(thresholds):
                        mask = valid_mask & (feature_values > thresholds[child_idx - 1]) & (feature_values <= thresholds[child_idx])
                    else:
                        logger.warning(f"Invalid child index {child_idx} for multi-bin split with {len(thresholds)} thresholds")
                        mask = pd.Series(False, index=feature_values.index)
                else:
                    logger.warning(f"Multi-bin split node {parent.node_id} missing split_thresholds")
                    mask = pd.Series(child_idx == 0, index=feature_values.index)
                
                indices = indices[mask.values]
                
            elif parent.split_type == 'categorical':
                if hasattr(parent, 'split_categories') and parent.split_categories:
                    mask = feature_values.map(parent.split_categories) == child_idx
                    mask = mask.fillna(child_idx == 0)
                else:
                    if child_idx == 0 and hasattr(parent, 'left_categories'):
                        mask = feature_values.isin(parent.left_categories)
                        mask = mask | pd.isna(feature_values)
                    elif child_idx == 1 and hasattr(parent, 'right_categories'):
                        mask = feature_values.isin(parent.right_categories)
                    else:
                        mask = pd.Series(True, index=feature_values.index) if child_idx == 0 else pd.Series(False, index=feature_values.index)
                
                indices = indices[mask.values]
        
        return indices.tolist()
    
    def _find_best_numerical_split(self, X: pd.DataFrame, y: pd.Series, 
                                 sample_weight: np.ndarray, feature: str, 
                                 indices: List[int], current_impurity: float) -> Tuple[bool, Dict[str, Any]]:
        """
        Find the best split for a numerical feature
        
        Args:
            X: Training features
            y: Target variable
            sample_weight: Sample weights
            feature: Feature name to split on
            indices: Sample indices to consider
            current_impurity: Current node impurity
            
        Returns:
            Tuple of (split_found, split_info)
        """
        feature_values = X[feature].iloc[indices].to_numpy()
        target_values = y.iloc[indices].to_numpy()
        weights = sample_weight[indices]
        
        valid_mask = ~np.isnan(feature_values)
        if not any(valid_mask):
            return False, {}
        
        feature_values = feature_values[valid_mask]
        target_values = target_values[valid_mask]
        weights = weights[valid_mask]
        
        sort_idx = np.argsort(feature_values)
        feature_values = feature_values[sort_idx]
        target_values = target_values[sort_idx]
        weights = weights[sort_idx]
        
        unique_values = np.unique(feature_values)
        if len(unique_values) <= 1:
            return False, {}
        
        split_points = (unique_values[1:] + unique_values[:-1]) / 2
        
        best_gain = -float('inf')
        best_split_idx = -1
        
        n_samples = len(feature_values)
        n_classes = len(self.target_values)
        
        left_counts = {cls: 0 for cls in self.target_values}
        right_counts = {cls: sum(weights[target_values == cls]) for cls in self.target_values}
        
        sample_idx = 0  # Track current position in sorted samples
        for threshold_idx, threshold in enumerate(split_points):
            while sample_idx < n_samples and feature_values[sample_idx] <= threshold:
                cls = target_values[sample_idx]
                weight = weights[sample_idx]
                left_counts[cls] += weight
                right_counts[cls] -= weight
                sample_idx += 1
            
            left_impurity = self._calculate_impurity(left_counts)
            right_impurity = self._calculate_impurity(right_counts)
            
            left_weight = sum(left_counts.values())
            right_weight = sum(right_counts.values())
            total_weight = left_weight + right_weight
            
            weighted_impurity = (
                (left_weight / total_weight) * left_impurity + 
                (right_weight / total_weight) * right_impurity
            )
            
            gain = current_impurity - weighted_impurity
            
            left_weight = sum(left_counts.values())
            right_weight = sum(right_counts.values())
            
            if left_weight < self.min_samples_leaf or right_weight < self.min_samples_leaf:
                continue
                
            if gain > best_gain:
                best_gain = gain
                best_split_idx = threshold_idx
                best_split_info = {
                    'feature': feature,
                    'threshold': threshold,
                    'gain': gain,
                    'impurity_decrease': gain,
                    'left_impurity': left_impurity,
                    'right_impurity': right_impurity,
                    'left_counts': left_counts.copy(),
                    'right_counts': right_counts.copy(),
                    'split_type': 'numeric',
                    'left_samples': int(left_weight),
                    'right_samples': int(right_weight)
                }
        
        if best_split_idx >= 0:
            return True, best_split_info
        else:
            return False, {}
    
    def _find_best_categorical_split(self, X: pd.DataFrame, y: pd.Series, 
                                   sample_weight: np.ndarray, feature: str, 
                                   indices: List[int], current_impurity: float) -> Tuple[bool, Dict[str, Any]]:
        """
        Find the best split for a categorical feature
        
        Args:
            X: Training features
            y: Target variable
            sample_weight: Sample weights
            feature: Feature name to split on
            indices: Sample indices to consider
            current_impurity: Current node impurity
            
        Returns:
            Tuple of (split_found, split_info)
        """
        feature_values = X[feature].iloc[indices].to_numpy()
        target_values = y.iloc[indices].to_numpy()
        weights = sample_weight[indices]
        
        valid_mask = ~pd.isna(feature_values)
        if not any(valid_mask):
            return False, {}
        
        feature_values = feature_values[valid_mask]
        target_values = target_values[valid_mask]
        weights = weights[valid_mask]
        
        categories = np.unique(feature_values)
        
        if len(categories) > 10:
            return self._find_best_grouped_categorical_split(
                feature, categories, feature_values, target_values, weights, current_impurity
            )
        else:
            return self._find_best_binary_categorical_split(
                feature, categories, feature_values, target_values, weights, current_impurity
            )
    
    def _find_best_grouped_categorical_split(self, feature: str, categories: np.ndarray,
                                          feature_values: np.ndarray, target_values: np.ndarray,
                                          weights: np.ndarray, current_impurity: float) -> Tuple[bool, Dict[str, Any]]:
        """
        Find the best split for a categorical feature using a grouping heuristic
        
        Args:
            feature: Feature name
            categories: Unique categories
            feature_values: Feature values
            target_values: Target values
            weights: Sample weights
            current_impurity: Current node impurity
            
        Returns:
            Tuple of (split_found, split_info)
        """
        category_stats = {}
        
        for cat in categories:
            cat_mask = feature_values == cat
            if not any(cat_mask):
                continue
                
            cat_targets = target_values[cat_mask]
            cat_weights = weights[cat_mask]
            
            cat_counts = {}
            for cls in self.target_values:
                cls_mask = cat_targets == cls
                if any(cls_mask):
                    cat_counts[cls] = cat_weights[cls_mask].sum()
                else:
                    cat_counts[cls] = 0
            
            total_weight = sum(cat_counts.values())
            if total_weight > 0:
                positive_prop = cat_counts.get(self.positive_class, 0) / total_weight
                category_stats[cat] = {
                    'counts': cat_counts,
                    'positive_prop': positive_prop,
                    'total_weight': total_weight
                }
        
        sorted_cats = sorted(category_stats.keys(), 
                             key=lambda c: category_stats[c]['positive_prop'])
        
        if len(sorted_cats) <= 1:
            return False, {}
        
        best_gain = -float('inf')
        best_threshold_idx = -1
        
        left_counts = {cls: 0 for cls in self.target_values}
        right_counts = {cls: sum(weights[target_values == cls]) for cls in self.target_values}
        
        for i in range(1, len(sorted_cats)):
            left_cats = set(sorted_cats[:i])
            
            left_counts = {cls: 0 for cls in self.target_values}
            right_counts = {cls: 0 for cls in self.target_values}
            
            for cat, stats in category_stats.items():
                if cat in left_cats:
                    for cls, count in stats['counts'].items():
                        left_counts[cls] += count
                else:
                    for cls, count in stats['counts'].items():
                        right_counts[cls] += count
            
            left_impurity = self._calculate_impurity(left_counts)
            right_impurity = self._calculate_impurity(right_counts)
            
            left_weight = sum(left_counts.values())
            right_weight = sum(right_counts.values())
            total_weight = left_weight + right_weight
            
            if total_weight <= 0:
                continue
                
            weighted_impurity = (
                (left_weight / total_weight) * left_impurity + 
                (right_weight / total_weight) * right_impurity
            )
            
            gain = current_impurity - weighted_impurity
            
            if gain > best_gain:
                best_gain = gain
                best_threshold_idx = i
                best_split_info = {
                    'feature': feature,
                    'categories': sorted_cats,
                    'left_categories': sorted_cats[:i],
                    'right_categories': sorted_cats[i:],
                    'gain': gain,
                    'impurity_decrease': gain,
                    'left_impurity': left_impurity,
                    'right_impurity': right_impurity,
                    'left_counts': left_counts.copy(),
                    'right_counts': right_counts.copy(),
                    'split_type': 'categorical'
                }
        
        if best_threshold_idx >= 0:
            return True, best_split_info
        else:
            return False, {}
    
    def _find_best_binary_categorical_split(self, feature: str, categories: np.ndarray,
                                         feature_values: np.ndarray, target_values: np.ndarray,
                                         weights: np.ndarray, current_impurity: float) -> Tuple[bool, Dict[str, Any]]:
        """
        Find the best binary split for a categorical feature trying all possible groupings
        
        Args:
            feature: Feature name
            categories: Unique categories
            feature_values: Feature values
            target_values: Target values
            weights: Sample weights
            current_impurity: Current node impurity
            
        Returns:
            Tuple of (split_found, split_info)
        """
        n_categories = len(categories)
        max_combinations = 2 ** (n_categories - 1) - 1  # -1 to avoid empty set
        
        max_to_try = 100 if n_categories > 7 else max_combinations
        
        best_gain = -float('inf')
        best_split = None
        
        combinations_tried = 0
        
        for subset_size in range(1, n_categories):
            from itertools import combinations
            
            for subset in combinations(range(n_categories), subset_size):
                combinations_tried += 1
                if combinations_tried > max_to_try:
                    break
                
                left_cats = set(categories[list(subset)])
                
                left_counts = {cls: 0 for cls in self.target_values}
                right_counts = {cls: 0 for cls in self.target_values}
                
                for i, (cat, target, weight) in enumerate(zip(feature_values, target_values, weights)):
                    if cat in left_cats:
                        left_counts[target] += weight
                    else:
                        right_counts[target] += weight
                
                left_impurity = self._calculate_impurity(left_counts)
                right_impurity = self._calculate_impurity(right_counts)
                
                left_weight = sum(left_counts.values())
                right_weight = sum(right_counts.values())
                total_weight = left_weight + right_weight
                
                if total_weight <= 0:
                    continue
                    
                weighted_impurity = (
                    (left_weight / total_weight) * left_impurity + 
                    (right_weight / total_weight) * right_impurity
                )
                
                gain = current_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature': feature,
                        'left_categories': list(left_cats),
                        'right_categories': [cat for cat in categories if cat not in left_cats],
                        'gain': gain,
                        'impurity_decrease': gain,
                        'left_impurity': left_impurity,
                        'right_impurity': right_impurity,
                        'left_counts': left_counts.copy(),
                        'right_counts': right_counts.copy(),
                        'split_type': 'categorical'
                    }
            
            if combinations_tried > max_to_try:
                break
        
        if best_split is not None:
            return True, best_split
        else:
            return False, {}
    
    def _calculate_impurity(self, class_counts: Dict[Any, float]) -> float:
        """
        Calculate impurity based on the selected criterion
        
        Args:
            class_counts: Dictionary of class counts
            
        Returns:
            Impurity value
        """
        total_count = sum(class_counts.values())
        
        if total_count <= 0:
            return 0.0
        
        proportions = [count / total_count for count in class_counts.values()]
        
        proportions = [p for p in proportions if p > 0]
        
        if not proportions:
            return 0.0
        
        if self.criterion == SplitCriterion.GINI:
            return 1.0 - sum(p * p for p in proportions)
            
        elif self.criterion == SplitCriterion.ENTROPY:
            return -sum(p * np.log2(p) for p in proportions)
            
        elif self.criterion == SplitCriterion.INFORMATION_GAIN:
            return -sum(p * np.log2(p) for p in proportions)
            
        elif self.criterion == SplitCriterion.MISCLASSIFICATION:
            return 1.0 - max(proportions)
        
        return 1.0 - sum(p * p for p in proportions)
    
    def _apply_split(self, X: pd.DataFrame, split_info: Dict[str, Any]) -> Tuple[List[int], List[int]]:
        """
        Apply a split to the data and return indices for left and right children
        
        Args:
            X: Feature DataFrame
            split_info: Split information
            
        Returns:
            Tuple of (left_indices, right_indices)
        """
        feature = split_info['feature']
        
        if split_info['split_type'] == 'numeric':
            threshold = split_info['threshold']
            left_indices = [i for i in range(len(X)) if X[feature].iloc[i] <= threshold]
            right_indices = [i for i in range(len(X)) if X[feature].iloc[i] > threshold]
            
        elif split_info['split_type'] == 'categorical':
            left_categories = set(split_info['left_categories'])
            left_indices = [i for i in range(len(X)) if X[feature].iloc[i] in left_categories]
            right_indices = [i for i in range(len(X)) if X[feature].iloc[i] not in left_categories]
        
        return left_indices, right_indices
    
    def _create_children_from_split(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray,
                                  node: TreeNode, split_info: Dict[str, Any], 
                                  left_indices: List[int], right_indices: List[int]):
        """
        Create child nodes based on a split
        
        Args:
            X: Feature DataFrame
            y: Target variable
            sample_weight: Sample weights
            node: Parent node
            split_info: Split information
            left_indices: Indices for left child
            right_indices: Indices for right child
        """
        if split_info['split_type'] == 'numeric':
            node.set_split(
                feature=split_info['feature'],
                value=split_info['threshold'],
                split_type='numeric'
            )
        elif split_info['split_type'] == 'categorical':
            left_categories = split_info['left_categories']
            all_categories = left_categories + split_info['right_categories']
            
            categories_map = {}
            for cat in all_categories:
                categories_map[cat] = 0 if cat in left_categories else 1
            
            node.set_categorical_split(
                feature=split_info['feature'],
                categories=categories_map
            )
        
        left_node_id = f"{node.node_id}_L"
        left_node = TreeNode(node_id=left_node_id, parent=node, depth=node.depth + 1)
        left_node.update_stats(
            samples=len(left_indices),
            class_counts=split_info['left_counts'],
            impurity=split_info['left_impurity']
        )
        
        right_node_id = f"{node.node_id}_R"
        right_node = TreeNode(node_id=right_node_id, parent=node, depth=node.depth + 1)
        right_node.update_stats(
            samples=len(right_indices),
            class_counts=split_info['right_counts'],
            impurity=split_info['right_impurity']
        )
        
        node.add_child(left_node)
        node.add_child(right_node)
        
        min_samples_leaf = max(1, self.min_samples_leaf)
        
        if len(left_indices) < min_samples_leaf:
            left_node.is_terminal = True
        
        if len(right_indices) < min_samples_leaf:
            right_node.is_terminal = True
        
        self.max_node_id = max(self.max_node_id, int(node.node_id.split('_')[-1]) if '_' in node.node_id else 0)
        
        self.nodeUpdated.emit(left_node.node_id)
        self.nodeUpdated.emit(right_node.node_id)
    
    
    def _prune_tree(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray):
        """
        Prune the tree to reduce overfitting
        
        Args:
            X: Feature DataFrame
            y: Target variable
            sample_weight: Sample weights
        """
        if not self.pruning_enabled or not self.pruning_method:
            return
        
        logger.info(f"Pruning tree using {self.pruning_method} method")
        
        if self.pruning_method == 'cost_complexity':
            self._cost_complexity_pruning(X, y, sample_weight)
        elif self.pruning_method == 'reduced_error':
            self._reduced_error_pruning(X, y, sample_weight)
        else:
            logger.warning(f"Unknown pruning method: {self.pruning_method}")
    
    def _cost_complexity_pruning(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray):
        """
        Apply cost-complexity pruning
        
        Args:
            X: Feature DataFrame
            y: Target variable
            sample_weight: Sample weights
        """
        alphas = np.logspace(-10, 0, num=20)
        
        best_alpha = 0
        best_score = -float('inf')
        
        for alpha in alphas:
            pruned_tree = self.copy()
            
            pruned_tree._cost_complexity_pruning_with_alpha(alpha)
            
            score = pruned_tree._evaluate_tree(X, y, sample_weight)
            
            if score > best_score:
                best_score = score
                best_alpha = alpha
        
        self._cost_complexity_pruning_with_alpha(best_alpha)
        
        logger.info(f"Cost complexity pruning completed with alpha={best_alpha}, score={best_score:.4f}")
    
    def _cost_complexity_pruning_with_alpha(self, alpha: float):
        """
        Apply cost-complexity pruning with a specific alpha
        
        Args:
            alpha: Complexity parameter
        """
        nodes = self._get_all_nodes()
        
        nodes.sort(key=lambda n: -n.depth)
        
        for node in nodes:
            if node.is_terminal or node.node_id == "root":
                continue
                
            leaves = node.get_leaf_nodes()
            
            subtree_error = sum(leaf.impurity * leaf.samples for leaf in leaves) / sum(leaf.samples for leaf in leaves)
            
            node_error = node.impurity
            
            num_leaves = len(leaves)
            cost_complexity = (node_error - subtree_error) / (num_leaves - 1)
            
            if cost_complexity < alpha:
                node.is_terminal = True
                node.children = []  # Remove children
                
                self.num_nodes = len(self._get_all_nodes())
                self.num_leaves = sum(1 for n in self._get_all_nodes() if n.is_terminal)
                
                self.nodeUpdated.emit(node.node_id)
    
    def _reduced_error_pruning(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray):
        """
        Apply reduced-error pruning
        
        Args:
            X: Feature DataFrame
            y: Target variable
            sample_weight: Sample weights
        """
        from sklearn.model_selection import train_test_split
        
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X, y, sample_weight, test_size=0.3, random_state=self.random_state
        )
        
        nodes = self._get_all_nodes()
        
        nodes.sort(key=lambda n: -n.depth)
        
        initial_error = 1.0 - self.score(X_val, y_val, sample_weight=w_val)
        
        for node in nodes:
            if node.is_terminal or node.node_id == "root":
                continue
                
            temp_tree = self.copy()
            
            temp_node = temp_tree.root.get_node_by_id(node.node_id)
            
            if temp_node:
                temp_node.is_terminal = True
                temp_node.children = []  # Remove children
                
                pruned_error = 1.0 - temp_tree.score(X_val, y_val, sample_weight=w_val)
                
                if pruned_error <= initial_error:
                    node.is_terminal = True
                    node.children = []  # Remove children
                    
                    initial_error = pruned_error
                    
                    self.num_nodes = len(self._get_all_nodes())
                    self.num_leaves = sum(1 for n in self._get_all_nodes() if n.is_terminal)
                    
                    self.nodeUpdated.emit(node.node_id)
        
        logger.info(f"Reduced error pruning completed, final error={initial_error:.4f}")
    
    def _evaluate_tree(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray) -> float:
        """
        Evaluate the tree on a dataset
        
        Args:
            X: Feature DataFrame
            y: Target variable
            sample_weight: Sample weights
            
        Returns:
            Accuracy score
        """
        if self.root is None:
            return 0.0
        
        predictions = self.root.predict(X)
        
        correct = (predictions == y.to_numpy())
        return np.sum(correct * sample_weight) / np.sum(sample_weight)
    
    def _calculate_feature_importance(self):
        """Calculate feature importance based on impurity decrease"""
        self.feature_importance = {feature: 0.0 for feature in self.feature_names}
        
        nodes = [node for node in self._get_all_nodes() if not node.is_terminal]
        
        total_samples = self.root.samples
        
        for node in nodes:
            if node.split_feature is None:
                continue
                
            feature = node.split_feature
            
            n_samples = node.samples
            impurity = node.impurity
            
            weighted_decrease = 0.0
            
            for child in node.children:
                child_samples = child.samples
                child_impurity = child.impurity
                
                weighted_decrease += (child_samples / n_samples) * child_impurity
            
            impurity_decrease = impurity - weighted_decrease
            
            self.feature_importance[feature] += (n_samples / total_samples) * impurity_decrease
        
        total_importance = sum(self.feature_importance.values())
        if total_importance > 0:
            for feature in self.feature_importance:
                self.feature_importance[feature] /= total_importance
        
        logger.info(f"Calculated feature importance: {', '.join([f'{f}={i:.4f}' for f, i in self.feature_importance.items() if i > 0])}")
    
    def _calculate_impurity_from_values(self, y_values: np.ndarray) -> float:
        """Calculate impurity for given target values"""
        if len(y_values) == 0:
            return 0.0
        
        unique, counts = np.unique(y_values, return_counts=True)
        class_counts = dict(zip(unique, counts.astype(float)))
        
        return self._calculate_impurity(class_counts)
    
    def get_node_split_candidates(self, node_id: str, max_candidates: int = 10) -> List[Dict[str, Any]]:
        """
        Get potential split candidates for a node in manual tree building mode
        
        Args:
            node_id: ID of the node to get candidates for
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of split candidate dictionaries
        """
        try:
            if not hasattr(self, '_cached_X') or self._cached_X is None:
                logger.warning("No cached training data available for split candidates")
                return []
            
            node = self.get_node_by_id(node_id)
            if not node:
                logger.warning(f"Node {node_id} not found")
                return []
            
            sample_indices = self._get_node_sample_indices(self._cached_X, node)
            if len(sample_indices) < self.min_samples_split:
                logger.info(f"Node {node_id} has insufficient samples for splitting ({len(sample_indices)} < {self.min_samples_split})")
                return []
            
            X_node = self._cached_X.iloc[sample_indices]
            y_node = self._cached_y.iloc[sample_indices]
            sample_weight_node = self._cached_sample_weight[sample_indices] if self._cached_sample_weight is not None else None
            
            current_impurity = self._calculate_impurity_from_values(y_node.values)
            
            candidates = []
            features_to_try = self._select_features_for_split()
            
            for feature in features_to_try:
                if feature not in X_node.columns:
                    continue
                
                if X_node[feature].dtype in ['int64', 'float64']:
                    found_split, split_info = self._find_best_numerical_split(
                        X_node, y_node, sample_weight_node or np.ones(len(y_node)), 
                        feature, list(range(len(X_node))), current_impurity
                    )
                else:
                    found_split, split_info = self._find_best_categorical_split(
                        X_node, y_node, sample_weight_node or np.ones(len(y_node)), 
                        feature, list(range(len(X_node))), current_impurity
                    )
                
                if found_split and split_info.get('gain', 0) > 0:
                    split_info['node_id'] = node_id
                    split_info['n_samples'] = len(sample_indices)
                    split_info['current_impurity'] = current_impurity
                    candidates.append(split_info)
            
            candidates.sort(key=lambda x: x.get('gain', 0), reverse=True)
            
            return candidates[:max_candidates]
            
        except Exception as e:
            logger.error(f"Error getting split candidates for node {node_id}: {e}", exc_info=True)
            return []
    
    def apply_manual_split(self, node_id: str, split_config: Dict[str, Any]) -> bool:
        """Apply manual split with enhanced error handling - maintains existing functionality"""
        try:
            logger.info(f"Applying manual split to node {node_id}")
            
            if hasattr(self, '_target_name') and self._target_name:
                if hasattr(self, '_cached_X') and self._cached_X is not None:
                    if self._target_name not in self._cached_X.columns:
                        if not hasattr(self, '_cached_y') or self._cached_y is None:
                            logger.error(f"Target column '{self._target_name}' not found in dataset")
                            return False
                else:
                    logger.error("No cached dataset available for validation")
                    return False
            
            node = self.get_node_by_id(node_id)
            if not node:
                logger.error(f"Node {node_id} not found")
                return False
                
            if not hasattr(node, 'samples') or node.samples == 0:
                if node_id == 'root' and hasattr(self, '_cached_X') and self._cached_X is not None:
                    node.samples = len(self._cached_X)
                    node.class_counts = {}
                    if hasattr(self, '_cached_y') and self._cached_y is not None:
                        node.class_counts = self._cached_y.value_counts().to_dict()
                        node.majority_class = self._cached_y.mode().iloc[0] if len(self._cached_y) > 0 else None
                    logger.info(f"Initialized root node with {node.samples} samples")
                else:
                    logger.error(f"Node {node_id} has no samples and cannot be split")
                    return False
                    
            normalized_config = split_config.copy()
            
            if 'split_type' not in normalized_config:
                logger.error("Missing 'split_type' in split configuration")
                return False
                
            original_split_type = normalized_config['split_type']
            if original_split_type in ['numeric', 'numerical']:
                normalized_config['split_type'] = 'numeric'
            elif original_split_type in ['numeric_multi_bin', 'numerical_multi_bin']:
                normalized_config['split_type'] = 'numeric_multi_bin'
            elif original_split_type == 'categorical':
                normalized_config['split_type'] = 'categorical'
            elif original_split_type in ['categorical_multi_bin', 'categorical_multi_way']:
                normalized_config['split_type'] = 'categorical_multi_bin'
            else:
                logger.error(f"Invalid split_type: {original_split_type}")
                return False
                
            if normalized_config['split_type'] in ['numeric', 'numerical']:
                threshold_keys = ['threshold', 'split_value', 'value']
                threshold_value = None
                
                for key in threshold_keys:
                    if key in normalized_config:
                        threshold_value = normalized_config[key]
                        break
                        
                if threshold_value is None:
                    logger.error("Numerical split requires threshold/split_value")
                    return False
                    
                try:
                    threshold_value = float(threshold_value)
                    normalized_config['threshold'] = threshold_value
                    normalized_config['split_value'] = threshold_value  # Ensure both are set
                except (ValueError, TypeError):
                    logger.error(f"Invalid threshold value: {threshold_value}")
                    return False
                    
            elif normalized_config['split_type'] == 'categorical':
                if 'left_categories' not in normalized_config and 'split_categories' not in normalized_config:
                    logger.error("Categorical split requires left_categories or split_categories")
                    return False
                    
            feature = normalized_config.get('feature')
            if not feature:
                logger.error("Missing 'feature' in split configuration")
                return False
                
            if hasattr(self, '_cached_X') and self._cached_X is not None:
                if feature not in self._cached_X.columns:
                    logger.error(f"Feature '{feature}' not found in dataset columns")
                    return False
            else:
                logger.error("No cached dataset available for feature validation")
                return False
                
            success = self._apply_split_with_enhanced_validation(node_id, normalized_config)
            
            if success:
                if hasattr(self, 'nodeUpdated'):
                    self.nodeUpdated.emit(node_id)
                if hasattr(self, 'treeUpdated'):
                    self.treeUpdated.emit()
                    
                logger.info(f"Successfully applied manual split to node {node_id}")
                return True
            else:
                logger.error(f"Failed to apply split to node {node_id}")
                return False
                    
        except Exception as e:
            logger.error(f"Exception in apply_manual_split: {e}", exc_info=True)
            return False
            
    def _apply_split_with_validation(self, node: TreeNode, config: SplitConfiguration, 
                                   transaction: SplitTransaction) -> bool:
        """Apply split with comprehensive validation"""
        try:
            transaction.backup_node(node)
            
            if hasattr(node, 'sample_indices') and node.sample_indices is not None:
                node_X = self._cached_X.iloc[node.sample_indices]
                node_y = self._cached_y.iloc[node.sample_indices]
            else:
                if node.node_id == 'root':
                    logger.info("Root node missing sample indices, initializing with all data")
                    node.sample_indices = list(range(len(self._cached_X)))
                    node_X = self._cached_X
                    node_y = self._cached_y
                else:
                    logger.info(f"Node {node.node_id} missing sample indices, calculating from parent splits")
                    node.sample_indices = self._get_node_sample_indices(self._cached_X, node)
                    if not node.sample_indices:
                        logger.error(f"Could not calculate sample indices for node {node.node_id}")
                        return False
                    node_X = self._cached_X.iloc[node.sample_indices]
                    node_y = self._cached_y.iloc[node.sample_indices]
                
            if config.split_type == 'numeric':
                return self._apply_numerical_split(node, config, node_X, node_y, transaction)
            elif config.split_type == 'categorical':
                return self._apply_categorical_split(node, config, node_X, node_y, transaction)
            elif config.split_type == 'numeric_multi_bin':
                return self._apply_multi_bin_split(node, config, node_X, node_y, transaction)
            else:
                logger.error(f"Unsupported split type: {config.split_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error in split application: {e}")
            return False
            
    def _apply_categorical_split(self, node: TreeNode, config: SplitConfiguration,
                               node_X: pd.DataFrame, node_y: pd.Series,
                               transaction: SplitTransaction) -> bool:
        """Apply categorical split with bin configuration support"""
        try:
            feature = config.feature
            
            if feature not in node_X.columns:
                logger.error(f"Feature {feature} not found in node data")
                return False
                
            feature_data = node_X[feature]
            
            if hasattr(config, 'bin_configurations') and config.bin_configurations:
                return self._apply_categorical_multi_bin_split(
                    node, feature, config.bin_configurations, feature_data, node_y, transaction
                )
            else:
                left_categories = set(config.threshold) if isinstance(config.threshold, list) else {config.threshold}
                return self._apply_categorical_binary_split(
                    node, feature, left_categories, feature_data, node_y, transaction
                )
                
        except Exception as e:
            logger.error(f"Error applying categorical split: {e}")
            return False

    def _apply_split_with_transaction(self, node: TreeNode, config: SplitConfiguration, 
                                    transaction: SplitTransaction) -> bool:
        """Apply split within a transaction"""
        try:
            transaction.backup_node_state(node)
            
            if hasattr(node, 'sample_indices') and node.sample_indices is not None:
                node_X = self._cached_X.iloc[node.sample_indices]
                node_y = self._cached_y.iloc[node.sample_indices]
            else:
                if node.node_id == 'root':
                    logger.info("Root node missing sample indices, initializing with all data")
                    node.sample_indices = list(range(len(self._cached_X)))
                    node_X = self._cached_X
                    node_y = self._cached_y
                else:
                    logger.info(f"Node {node.node_id} missing sample indices, calculating from parent splits")
                    node.sample_indices = self._get_node_sample_indices(self._cached_X, node)
                    if not node.sample_indices:
                        logger.error(f"Could not calculate sample indices for node {node.node_id}")
                        return False
                    node_X = self._cached_X.iloc[node.sample_indices]
                    node_y = self._cached_y.iloc[node.sample_indices]
                
            if config.split_type == 'numeric':
                return self._apply_numerical_split_enhanced(node, config, node_X, node_y, transaction)
            elif config.split_type == 'categorical':
                return self._apply_categorical_split_enhanced(node, node_X, node_y, config, transaction)
            elif config.split_type == 'numeric_multi_bin':
                return self._apply_multi_bin_split(node, config, node_X, node_y, transaction)
            else:
                logger.error(f"Unsupported split type: {config.split_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error in split application: {e}")
            return False

    def _apply_numerical_split_enhanced(self, node: TreeNode, config: SplitConfiguration,
                                      node_X: pd.DataFrame, node_y: pd.Series,
                                      transaction: SplitTransaction) -> bool:
        """Apply enhanced numerical split with transaction support"""
        try:
            feature = config.feature
            threshold = config.split_value if hasattr(config, 'split_value') else config.threshold
            
            if feature not in node_X.columns:
                logger.error(f"Feature {feature} not found in node data")
                return False
                
            feature_data = node_X[feature]
            
            left_mask = feature_data <= threshold
            right_mask = ~left_mask
            
            left_indices = node_X.index[left_mask].tolist()
            right_indices = node_X.index[right_mask].tolist()
            
            if len(left_indices) == 0 or len(right_indices) == 0:
                logger.warning(f"Split would create empty child nodes")
                return False
                
            left_child = TreeNode(f"{node.node_id}_L")
            right_child = TreeNode(f"{node.node_id}_R")
            
            left_child.parent = node
            right_child.parent = node
            left_child.depth = node.depth + 1
            right_child.depth = node.depth + 1
            left_child.is_terminal = True
            right_child.is_terminal = True
            
            left_child.sample_indices = left_indices
            right_child.sample_indices = right_indices
            
            left_y = node_y[left_mask]
            right_y = node_y[right_mask]
            
            if len(left_y) > 0:
                left_child.class_counts = left_y.value_counts().to_dict()
                left_child.predicted_class = left_y.mode().iloc[0] if len(left_y.mode()) > 0 else left_y.iloc[0]
                left_child.samples = len(left_y)
                left_child.impurity = self._calculate_node_impurity(left_y)
                
            if len(right_y) > 0:
                right_child.class_counts = right_y.value_counts().to_dict()
                right_child.predicted_class = right_y.mode().iloc[0] if len(right_y.mode()) > 0 else right_y.iloc[0]
                right_child.samples = len(right_y)
                right_child.impurity = self._calculate_node_impurity(right_y)
            
            node.children = [left_child, right_child]
            node.is_terminal = False
            node.split_feature = feature
            node.split_value = threshold
            node.split_type = 'numeric'
            
            transaction.record_split_creation(node, [left_child, right_child])
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying enhanced numerical split: {e}")
            return False

    def _apply_categorical_split_enhanced(self, node, node_X, node_y, config, transaction) -> bool:
        """Apply enhanced categorical split (binary or multi-bin)"""
        try:
            feature = config.feature
            
            category_mapping = getattr(config, 'category_mapping', {})
            if category_mapping and len(set(category_mapping.values())) > 2:
                return self._apply_categorical_multi_bin_split_enhanced(node, node_X, node_y, config, transaction)
            
            left_categories = getattr(config, 'left_categories', [])
            
            if not left_categories and category_mapping:
                left_categories = [cat for cat, bin_idx in category_mapping.items() if bin_idx == 0]
            
            if not left_categories:
                logger.error("No left categories provided for categorical split")
                return False
            
            left_mask = node_X[feature].isin(left_categories)
            right_mask = ~left_mask
            
            if hasattr(node, 'sample_indices'):
                left_indices = [idx for i, idx in enumerate(node.sample_indices) if left_mask.iloc[i]]
                right_indices = [idx for i, idx in enumerate(node.sample_indices) if right_mask.iloc[i]]
            else:
                left_indices = node_X.index[left_mask].tolist()
                right_indices = node_X.index[right_mask].tolist()
            
            if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
                logger.error(f"Split would create nodes with insufficient samples: left={len(left_indices)}, right={len(right_indices)}")
                return False
            
            left_child = self._create_child_node_enhanced(node, 'left', left_indices, transaction)
            right_child = self._create_child_node_enhanced(node, 'right', right_indices, transaction)
            
            node.split_feature = feature
            right_categories = list(set(node_X[feature].unique()) - set(left_categories))
            split_categories = {}
            for cat in left_categories:
                split_categories[cat] = 0  # Left child
            for cat in right_categories:
                split_categories[cat] = 1  # Right child
            node.split_categories = split_categories
            node.split_type = 'categorical'
            node.children = [left_child, right_child]
            node.is_terminal = False
            
            transaction.record_split_creation(node, [left_child, right_child])
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying enhanced categorical split: {e}")
            return False

    def _apply_multi_bin_numerical_split_enhanced(self, node, node_X, node_y, config, transaction) -> bool:
        """Apply enhanced multi-bin numerical split"""
        try:
            feature = config.feature
            thresholds = getattr(config, 'thresholds', [])
            
            if not thresholds:
                logger.error("No thresholds provided for multi-bin numerical split")
                return False
            
            thresholds = sorted(thresholds)
            
            child_nodes = []
            child_indices_list = []
            
            for i in range(len(thresholds) + 1):
                if i == 0:
                    mask = node_X[feature] <= thresholds[0]
                elif i == len(thresholds):
                    mask = node_X[feature] > thresholds[-1]
                else:
                    mask = (node_X[feature] > thresholds[i-1]) & (node_X[feature] <= thresholds[i])
                
                if hasattr(node, 'sample_indices'):
                    bin_indices = [idx for j, idx in enumerate(node.sample_indices) if mask.iloc[j]]
                else:
                    bin_indices = node_X.index[mask].tolist()
                
                if len(bin_indices) < self.min_samples_leaf:
                    logger.error(f"Bin {i} would have insufficient samples: {len(bin_indices)}")
                    return False
                
                child_indices_list.append(bin_indices)
            
            for i, bin_indices in enumerate(child_indices_list):
                child_node = self._create_child_node_enhanced(node, f'bin_{i}', bin_indices, transaction)
                child_nodes.append(child_node)
            
            node.split_feature = feature
            node.split_thresholds = thresholds
            node.split_type = 'numeric_multi_bin'
            node.children = child_nodes
            node.is_terminal = False
            
            transaction.record_split_creation(node, child_nodes)
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying enhanced multi-bin numerical split: {e}")
            return False

    def _apply_multi_bin_categorical_split_enhanced(self, node, node_X, node_y, config, transaction) -> bool:
        """Apply enhanced multi-bin categorical split"""
        try:
            feature = config.feature
            bin_categories = getattr(config, 'bin_categories', [])
            
            if not bin_categories:
                logger.error("No bin categories provided for multi-bin categorical split")
                return False
            
            child_nodes = []
            child_indices_list = []
            
            for i, categories in enumerate(bin_categories):
                if not categories:
                    continue
                    
                mask = node_X[feature].isin(categories)
                
                if hasattr(node, 'sample_indices'):
                    bin_indices = [idx for j, idx in enumerate(node.sample_indices) if mask.iloc[j]]
                else:
                    bin_indices = node_X.index[mask].tolist()
                
                if len(bin_indices) < self.min_samples_leaf:
                    logger.error(f"Bin {i} would have insufficient samples: {len(bin_indices)}")
                    return False
                
                child_indices_list.append(bin_indices)
            
            for i, bin_indices in enumerate(child_indices_list):
                child_node = self._create_child_node_enhanced(node, f'bin_{i}', bin_indices, transaction)
                child_nodes.append(child_node)
            
            node.split_feature = feature
            node.split_bin_categories = bin_categories
            node.split_type = 'categorical_multi_bin'
            node.children = child_nodes
            node.is_terminal = False
            
            transaction.record_split_creation(node, child_nodes)
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying enhanced multi-bin categorical split: {e}")
            return False

    def _apply_categorical_multi_bin_split_enhanced(self, node, node_X, node_y, config, transaction) -> bool:
        """Apply enhanced categorical multi-bin split using category_mapping"""
        try:
            feature = config.feature
            category_mapping = getattr(config, 'category_mapping', {})
            
            if not category_mapping:
                logger.error("No category mapping provided for categorical multi-bin split")
                return False
            
            bins = {}
            for category, bin_idx in category_mapping.items():
                if bin_idx not in bins:
                    bins[bin_idx] = []
                bins[bin_idx].append(category)
            
            child_nodes = []
            child_indices_list = []
            
            for bin_idx in sorted(bins.keys()):
                categories = bins[bin_idx]
                
                mask = node_X[feature].isin(categories)
                
                if hasattr(node, 'sample_indices'):
                    bin_indices = [idx for j, idx in enumerate(node.sample_indices) if mask.iloc[j]]
                else:
                    bin_indices = node_X.index[mask].tolist()
                
                if len(bin_indices) < self.min_samples_leaf:
                    logger.error(f"Bin {bin_idx} would have insufficient samples: {len(bin_indices)}")
                    return False
                
                child_indices_list.append(bin_indices)
            
            for i, bin_indices in enumerate(child_indices_list):
                child_node = self._create_child_node_enhanced(node, f'bin_{i}', bin_indices, transaction)
                child_nodes.append(child_node)
            
            node.split_feature = feature
            node.split_bin_categories = list(bins.values())  # Store for reference
            node.split_type = 'categorical_multi_bin'
            node.children = child_nodes
            node.is_terminal = False
            
            transaction.record_split_creation(node, child_nodes)
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying categorical multi-bin split: {e}")
            return False

    def _create_child_node_enhanced(self, parent, side, sample_indices, transaction):
        """Create enhanced child node with proper initialization"""
        try:
            child_id = f"{parent.node_id}_{side}"
            child = TreeNode(child_id)
            child.parent = parent
            child.depth = parent.depth + 1
            child.is_terminal = True
            child.sample_indices = sample_indices
            
            if hasattr(self, '_cached_X') and hasattr(self, '_cached_y') and sample_indices:
                try:
                    child_y = self._cached_y.iloc[sample_indices]
                    
                    if hasattr(child_y, 'value_counts'):
                        class_counts = child_y.value_counts().to_dict()
                        child.class_counts = class_counts
                        child.samples = len(sample_indices)
                        
                        if len(class_counts) > 0:
                            total = sum(class_counts.values())
                            if total > 0:
                                gini = 1.0 - sum((count/total)**2 for count in class_counts.values())
                                child.impurity = gini
                            else:
                                child.impurity = 0.0
                        else:
                            child.impurity = 0.0
                            
                        if class_counts:
                            majority_class = max(class_counts.items(), key=lambda x: x[1])[0]
                            child.prediction = majority_class
                            child.majority_class = majority_class
                        
                        logger.debug(f"Created child {child_id}: {child.samples} samples, impurity={child.impurity:.4f}")
                        
                except Exception as e:
                    logger.warning(f"Could not calculate statistics for child {child_id}: {e}")
                    child.samples = len(sample_indices)
                    child.impurity = 0.0
                    if hasattr(parent, 'majority_class') and parent.majority_class is not None:
                        child.majority_class = parent.majority_class
                        child.prediction = parent.majority_class
                    else:
                        if hasattr(self, 'target_values') and self.target_values:
                            child.majority_class = self.target_values[0]
                            child.prediction = self.target_values[0]
            else:
                if hasattr(parent, 'majority_class') and parent.majority_class is not None:
                    child.majority_class = parent.majority_class
                    child.prediction = parent.majority_class
                elif hasattr(self, 'target_values') and self.target_values:
                    child.majority_class = self.target_values[0]
                    child.prediction = self.target_values[0]
                child.samples = len(sample_indices) if sample_indices else 0
                child.impurity = 0.0
            
            return child
            
        except Exception as e:
            logger.error(f"Error creating child node: {e}")
            return None

    def get_split_candidates(self, node_id: str) -> List[Dict[str, Any]]:
        """Get split candidates for a node using enhanced split finder"""
        try:
            if not hasattr(self, '_cached_X') or not hasattr(self, '_cached_y'):
                logger.error("No cached data available for split finding")
                return []
            
            node = self.get_node_by_id(node_id)
            if node is None:
                logger.error(f"Node {node_id} not found")
                return []
            
            sample_indices = self._get_node_sample_indices(self._cached_X, node)
            if not sample_indices:
                logger.warning(f"No samples reach node {node_id}")
                return []
            
            node_X = self._cached_X.iloc[sample_indices]
            node_y = self._cached_y.iloc[sample_indices]
            
            from models.split_finder import SplitFinder
            
            split_finder = SplitFinder(
                criterion=self.criterion,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=getattr(self, 'min_samples_split', 2),
                min_impurity_decrease=getattr(self, 'min_impurity_decrease', 0.0)
            )
            
            candidates = split_finder.find_best_split(node_id, node_X, node_y)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error getting split candidates for node {node_id}: {e}")
            return []
            
    def _apply_categorical_multi_bin_split(self, node: TreeNode, feature: str,
                                         bin_configs: List[Dict[str, Any]],
                                         feature_data: pd.Series, node_y: pd.Series,
                                         transaction: SplitTransaction) -> bool:
        """Apply multi-bin categorical split"""
        try:
            child_nodes = []
            all_assigned_indices = set()
            
            for i, bin_config in enumerate(bin_configs):
                bin_categories = set(bin_config['categories'])
                
                bin_mask = feature_data.isin(bin_categories)
                bin_indices = feature_data.index[bin_mask].tolist()
                
                if len(bin_indices) == 0:
                    logger.warning(f"Bin {bin_config.get('bin_name', f'Bin_{i}')} is empty")
                    continue
                    
                child_node = TreeNode(
                    node_id=f"{node.node_id}_bin_{i}",
                    depth=node.depth + 1,
                    parent=node
                )
                
                child_node.sample_indices = bin_indices
                child_node.split_feature = feature
                child_node.split_condition = f"in {bin_categories}"
                child_node.is_terminal = True  # Initially terminal
                
                bin_y = node_y[bin_mask]
                if len(bin_y) > 0:
                    child_node.class_counts = bin_y.value_counts().to_dict()
                    child_node.predicted_class = bin_y.mode().iloc[0] if len(bin_y.mode()) > 0 else bin_y.iloc[0]
                    child_node.samples = len(bin_y)
                    child_node.impurity = self._calculate_node_impurity(bin_y)
                
                child_nodes.append(child_node)
                all_assigned_indices.update(bin_indices)
                
                transaction.record_node_creation(child_node)
                
            total_node_samples = len(feature_data)
            assigned_samples = len(all_assigned_indices)
            
            if assigned_samples != total_node_samples:
                all_indices = set(feature_data.index)
                unassigned_indices = all_indices - all_assigned_indices
                
                if unassigned_indices:
                    logger.warning(f"Found {len(unassigned_indices)} unassigned samples, assigning to largest bin")
                    
                    if child_nodes:
                        largest_bin = max(child_nodes, key=lambda x: x.samples)
                        largest_bin.sample_indices.extend(list(unassigned_indices))
                        
                        unassigned_mask = feature_data.index.isin(unassigned_indices)
                        unassigned_y = node_y[unassigned_mask]
                        if len(unassigned_y) > 0:
                            all_bin_y = pd.concat([node_y[node_y.index.isin(largest_bin.sample_indices)]])
                            largest_bin.class_counts = all_bin_y.value_counts().to_dict()
                            largest_bin.predicted_class = all_bin_y.mode().iloc[0] if len(all_bin_y.mode()) > 0 else all_bin_y.iloc[0]
                            largest_bin.samples = len(all_bin_y)
                            largest_bin.impurity = self._calculate_node_impurity(all_bin_y)
                        
                        all_assigned_indices.update(unassigned_indices)
                        assigned_samples = len(all_assigned_indices)
                        
                if assigned_samples != total_node_samples:
                    logger.error(f"Sample assignment mismatch after fix: {assigned_samples}/{total_node_samples}")
                    return False
                
            node.children = child_nodes
            node.is_terminal = False
            node.split_feature = feature
            node.split_type = 'categorical_multi_bin'
            
            return True
            
        except Exception as e:
            logger.error(f"Error in multi-bin categorical split: {e}")
            return False
            
    def _apply_categorical_binary_split(self, node: TreeNode, feature: str,
                                      left_categories: set, feature_data: pd.Series,
                                      node_y: pd.Series, transaction: SplitTransaction) -> bool:
        """Apply binary categorical split"""
        try:
            left_mask = feature_data.isin(left_categories)
            right_mask = ~left_mask & ~feature_data.isna()
            
            left_indices = feature_data.index[left_mask].tolist()
            right_indices = feature_data.index[right_mask].tolist()
            
            if len(left_indices) == 0 or len(right_indices) == 0:
                logger.error("Cannot create binary split with empty children")
                return False
                
            left_child = TreeNode(
                node_id=f"{node.node_id}_L",
                depth=node.depth + 1,
                parent=node
            )
            
            right_child = TreeNode(
                node_id=f"{node.node_id}_R",
                depth=node.depth + 1,
                parent=node
            )
            
            left_child.sample_indices = left_indices
            left_child.split_condition = f"in {left_categories}"
            left_child.is_terminal = True
            
            right_child.sample_indices = right_indices
            right_child.split_condition = f"not in {left_categories}"
            right_child.is_terminal = True
            
            left_y = node_y[left_mask]
            right_y = node_y[right_mask]
            
            if len(left_y) > 0:
                left_child.class_counts = left_y.value_counts().to_dict()
                left_child.predicted_class = left_y.mode().iloc[0] if len(left_y.mode()) > 0 else left_y.iloc[0]
                left_child.samples = len(left_y)
                left_child.impurity = self._calculate_node_impurity(left_y)
                
            if len(right_y) > 0:
                right_child.class_counts = right_y.value_counts().to_dict()
                right_child.predicted_class = right_y.mode().iloc[0] if len(right_y.mode()) > 0 else right_y.iloc[0]
                right_child.samples = len(right_y)
                right_child.impurity = self._calculate_node_impurity(right_y)
            
            node.children = [left_child, right_child]
            node.is_terminal = False
            node.split_feature = feature
            node.split_type = 'categorical'
            node.split_categories = {cat: 0 for cat in left_categories}
            node.split_categories.update({cat: 1 for cat in set(feature_data.unique()) - left_categories})
            
            transaction.record_node_creation(left_child)
            transaction.record_node_creation(right_child)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in binary categorical split: {e}")
            return False
            
    def _calculate_node_impurity(self, y: pd.Series) -> float:
        """Calculate node impurity"""
        if len(y) == 0:
            return 0.0
            
        class_counts = y.value_counts()
        total = len(y)
        
        gini = 1.0
        for count in class_counts:
            prob = count / total
            gini -= prob * prob
            
        return gini
        
            
    def _validate_tree_integrity(self) -> bool:
        """Validate tree integrity after split"""
        try:
            def validate_node(node):
                if node is None:
                    return True
                    
                for child in node.children:
                    if child.parent != node:
                        logger.error(f"Child {child.node_id} has incorrect parent")
                        return False
                    if not validate_node(child):
                        return False
                        
                return True
                
            return validate_node(self.root)
            
        except Exception as e:
            logger.error(f"Error validating tree integrity: {e}")
            return False
            
    def _apply_split_with_enhanced_validation(self, node_id: str, split_config: Dict[str, Any]) -> bool:
        """Apply split with enhanced validation and transaction support"""
        from models.split_transaction import SplitTransaction
        from models.split_validator import SplitValidator
        from models.split_configuration import SplitConfigurationFactory
        
        node = self._find_node_by_id(node_id)
        if node is None:
            logger.error(f"Node {node_id} not found")
            return False
            
        split_type = split_config.get('split_type', split_config.get('type'))
        if split_type in ['numeric_multi_bin', 'categorical_multi_bin'] or 'type' in split_config:
            enhanced_config = split_config.copy()
            if 'split_type' in enhanced_config and 'type' not in enhanced_config:
                enhanced_config['type'] = enhanced_config['split_type']
            config = SplitConfigurationFactory.from_enhanced_dialog_format(enhanced_config)
        else:
            config = SplitConfigurationFactory.from_dict(split_config)
            
        validator = SplitValidator()
        
        if self._cached_X is not None and self._cached_y is not None:
            combined_data = self._cached_X.copy()
            combined_data[self._target_name] = self._cached_y
            validation_result = validator.validate_split_config(config, combined_data)
            if not validation_result.is_valid:
                logger.error(f"Split configuration validation failed: {validation_result.errors}")
                return False
        else:
            validation_result = validator.validate_split_config(config)
            if not validation_result.is_valid:
                logger.error(f"Split configuration validation failed: {validation_result.errors}")
                return False
            
        with SplitTransaction(self) as transaction:
            success = self._apply_split_with_transaction(node, config, transaction)
            
            if success:
                tree_validation = validator.validate_tree_consistency(self.root, combined_data if 'combined_data' in locals() else None)
                if not tree_validation.is_valid:
                    logger.error(f"Tree integrity validation failed: {tree_validation.errors}")
                    return False
                    
                if hasattr(self, '_feature_importance'):
                    self._update_feature_importance_after_split(config.feature)
                    
                logger.info(f"Successfully applied manual split to node {node_id}")
                return True
            else:
                logger.error("Split application failed within transaction")
                return False
    
    
            
            
            
            
            
            
                
                
                
                
                    
                    
                
                
                
                
                
                
                
            
            
                
                
                
                
                
            
            
    
    def _update_child_statistics(self, parent_node: TreeNode):
        """Update statistics for child nodes after a split"""
        try:
            if not hasattr(self, '_cached_X') or self._cached_X is None:
                return
            
            for child in parent_node.children:
                sample_indices = self._get_node_sample_indices(self._cached_X, child)
                y_child = self._cached_y.iloc[sample_indices]
                
                child.samples = len(sample_indices)
                child.class_counts = y_child.value_counts().to_dict()
                child.impurity = self._calculate_impurity_from_values(y_child.values)
                
                if len(child.class_counts) > 0:
                    child.majority_class = max(child.class_counts.keys(), key=lambda k: child.class_counts[k])
                
        except Exception as e:
            logger.warning(f"Error updating child statistics: {e}")
    
    def _set_basic_child_statistics(self, parent_node: TreeNode):
        """Set basic statistics for child nodes when full data isn't available"""
        try:
            for i, child in enumerate(parent_node.children):
                child.is_terminal = True
                
                child.samples = parent_node.samples // len(parent_node.children)  # Rough estimate
                child.class_counts = parent_node.class_counts.copy() if parent_node.class_counts else {}
                child.impurity = parent_node.impurity if parent_node.impurity else 0.5
                
                if parent_node.majority_class:
                    child.majority_class = parent_node.majority_class
                elif parent_node.class_counts:
                    child.majority_class = max(parent_node.class_counts.keys(), key=lambda k: parent_node.class_counts[k])
                
                logger.info(f"Set basic statistics for child node {child.node_id}: {child.samples} samples")
                
        except Exception as e:
            logger.warning(f"Error setting basic child statistics: {e}")
    
    def _find_node_by_id(self, node_id: str):
        """Find node by ID with comprehensive search"""
        if not node_id:
            return None
            
        if not self.root:
            logger.error("Tree has no root node")
            return None
            
        return self._search_node_recursive(self.root, node_id)

    def _search_node_recursive(self, node, target_id):
        """Recursively search for node with target ID"""
        if not node:
            return None
            
        if node.node_id == target_id:
            return node
            
        if hasattr(node, 'children') and node.children:
            for child in node.children:
                result = self._search_node_recursive(child, target_id)
                if result:
                    return result
                    
        return None

    def get_node_by_id(self, node_id: str):
        """Public method to get node by ID"""
        return self._find_node_by_id(node_id)
    
    def _select_features_for_split_optimized(self, X: pd.DataFrame, y: pd.Series, 
                                           indices: np.ndarray, max_features: int = 20) -> List[str]:
        """
        Select features for split finding with performance optimization
        
        Args:
            X: Feature DataFrame
            y: Target variable
            indices: Sample indices for this node
            max_features: Maximum number of features to consider
            
        Returns:
            List of selected feature names
        """
        all_features = list(X.columns)
        
        if self.target_name and self.target_name in all_features:
            all_features.remove(self.target_name)
        
        if len(all_features) <= max_features:
            return all_features
        
        if self.feature_importance:
            sorted_features = sorted(all_features, 
                                   key=lambda f: self.feature_importance.get(f, 0), 
                                   reverse=True)
            selected_features = sorted_features[:max_features//2]
        else:
            selected_features = []
        
        remaining_features = [f for f in all_features if f not in selected_features]
        
        if remaining_features:
            np.random.shuffle(remaining_features)
            additional_count = max_features - len(selected_features)
            selected_features.extend(remaining_features[:additional_count])
        
        selected_features = selected_features[:max_features]
        
        logger.info(f"Selected {len(selected_features)} features for split evaluation")
        
        return selected_features
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the decision tree
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Numpy array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.root.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get class probabilities for input samples
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of class probabilities [n_samples, n_classes]
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.root.predict_proba(X)
    
    def score(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Calculate the accuracy score
        
        Args:
            X: Feature DataFrame
            y: Target variable
            sample_weight: Sample weights
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        
        if sample_weight is None:
            return np.mean(predictions == y.to_numpy())
        else:
            correct = (predictions == y.to_numpy())
            return np.sum(correct * sample_weight) / np.sum(sample_weight)


    def get_node(self, node_id: str) -> Optional[TreeNode]:
        """
        Get a node by its ID - Enhanced version with better error handling
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            TreeNode if found, None otherwise
        """
        try:
            if not self.is_fitted or not self.root:
                logger.warning(f"Model not fitted or no root node when looking for node {node_id}")
                return None
            
            return self.root.get_node_by_id(node_id)
            
        except Exception as e:
            logger.error(f"Error retrieving node {node_id}: {e}")
            return None

    @property
    def model_name(self):
        """Get model name"""
        return getattr(self, '_model_name', f"DecisionTree_{id(self)}")

    @model_name.setter
    def model_name(self, name):
        """Set model name"""
        self._model_name = name
    
    #     """
    #     Get a node by ID
        
    #     Args:
    #         node_id: Node ID
            
    #     Returns:
    #         TreeNode if found, None otherwise
    
    def set_active_node(self, node_id: str):
        """
        Set the active node for interactive operations
        
        Args:
            node_id: Node ID
        """
        node = self.get_node(node_id)
        if node:
            self.active_node_id = node_id
            logger.info(f"Set active node to {node_id}")
        else:
            logger.warning(f"Node {node_id} not found")
    
    def find_split_for_node(self, node_id: str, X: pd.DataFrame = None, y: pd.Series = None, 
                          sample_weight: Optional[np.ndarray] = None, max_features: int = None) -> List[Dict[str, Any]]:
        """
        Find potential splits for a node with performance optimization
        
        Args:
            node_id: Node ID
            X: Feature DataFrame
            y: Target variable
            sample_weight: Sample weights
            max_features: Maximum number of features to evaluate for performance
            
        Returns:
            List of potential splits
        """
        node = self.get_node(node_id)
        if not node:
            logger.warning(f"Node {node_id} not found")
            return []
        
        if node.is_terminal:
            logger.info(f"Node {node_id} is terminal, finding potential splits")
        
        if X is None:
            X = self._cached_X
        if y is None:
            y = self._cached_y
        if sample_weight is None:
            sample_weight = self._cached_sample_weight
            
        if sample_weight is None and X is not None:
            sample_weight = np.ones(len(X))
            
        if X is None or y is None:
            if not self.is_fitted:
                error_msg = "Model is not fitted yet. Cannot find splits without training data."
                logger.error(error_msg)
                self.trainingRequired.emit("Model needs to be trained before finding splits")
            else:
                error_msg = f"No training data available for finding splits. Target variable: '{self.target_name}'. Model fitted: {self.is_fitted}"
                logger.error(error_msg)
                # Additional debugging information
                if hasattr(self, '_cached_X') and self._cached_X is not None:
                    logger.debug(f"Cached X shape: {self._cached_X.shape}")
                if hasattr(self, '_cached_y') and self._cached_y is not None:
                    logger.debug(f"Cached y shape: {self._cached_y.shape}")
                self.trainingRequired.emit(f"Training data not available. Please ensure model is properly trained with target variable '{self.target_name}'")
            return []
        
        indices = self._get_node_sample_indices(X, node)
        
        if len(indices) <= 1:
            logger.warning(f"Node {node_id} has too few samples to split")
            return []
        
        logger.info(f"Finding splits for node {node_id} with {len(indices)} samples")
        
        split_config = self.config.get('decision_tree', {}).get('split_finding', {})
        if max_features is None:
            max_features = split_config.get('max_features_to_evaluate', 30)
        
        use_threading = split_config.get('use_threading', True)
        max_threads = split_config.get('max_threads', 4)  # Keep for backward compatibility
        max_splits_returned = split_config.get('max_splits_returned', 50)
        
        n_jobs = -2 if use_threading else 1  # N-1 cores or sequential
        
        potential_splits = []
        
        features_to_consider = self._select_features_for_split_optimized(X, y, indices, max_features)
        logger.info(f"Evaluating {len(features_to_consider)} features out of {len(X.columns)} total")
        
        def process_feature(feature):
            """Process a single feature for splits - optimized for parallel execution"""
            try:
                if X[feature].iloc[indices].isna().all():
                    logger.debug(f"Skipping feature {feature}: all missing values")
                    return None
                
                logger.debug(f"Processing feature {feature} ({'categorical' if feature in self.categorical_features else 'numerical'})")
                
                if feature in self.categorical_features:
                    split_found, split_info = self._find_best_categorical_split(
                        X, y, sample_weight, feature, indices, node.impurity
                    )
                else:
                    split_found, split_info = self._find_best_numerical_split(
                        X, y, sample_weight, feature, indices, node.impurity
                    )
                
                if split_found and split_info.get('gain', 0) > 0:
                    logger.debug(f"Feature {feature}: found split with gain={split_info.get('gain', 0):.6f}")
                    return feature, split_info
                else:
                    # Log the gain for debugging
                    gain = split_info.get('gain', 0) if split_found else 0
                    logger.debug(f"Feature {feature}: split_found={split_found}, gain={gain:.6f}")
                    return None
                    
            except Exception as e:
                logger.error(f"Exception processing feature {feature}: {type(e).__name__}: {e}")
                import traceback
                logger.debug(f"Traceback for feature {feature}: {traceback.format_exc()}")
                return None
        
        logger.info(f"Starting parallel processing of {len(features_to_consider)} features using {n_jobs if n_jobs > 0 else 'N-1 cores'}")
        
        results = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(process_feature)(feature) for feature in features_to_consider
        )
        
        valid_results = [result for result in results if result is not None]
        logger.info(f"Parallel processing completed. Found {len(valid_results)} valid splits from {len(features_to_consider)} features")
        
        for result in valid_results:
            try:
                feature_name, split_info = result
                
                if split_info['split_type'] == 'numeric':
                    split_desc = f"{feature_name} <= {split_info['threshold']:.4f}"
                else:
                    left_cats = split_info.get('left_categories', [])
                    if len(left_cats) <= 3:
                        split_desc = f"{feature_name} in {left_cats}"
                    else:
                        split_desc = f"{feature_name} in {len(left_cats)} categories"
                
                enhanced_split_info = split_info.copy()
                
                left_counts = split_info.get('left_counts', {})
                right_counts = split_info.get('right_counts', {})
                
                enhanced_split_info['left_samples'] = sum(left_counts.values()) if left_counts else 0
                enhanced_split_info['right_samples'] = sum(right_counts.values()) if right_counts else 0
                
                enhanced_split_info['left_class_distribution'] = left_counts
                enhanced_split_info['right_class_distribution'] = right_counts
                
                potential_splits.append({
                    'feature': feature_name,
                    'split_type': split_info['split_type'],
                    'split_desc': split_desc,
                    'gain': split_info['gain'],
                    'split_info': enhanced_split_info
                })
                
            except Exception as e:
                logger.error(f"Error processing result for feature: {e}")
        
        potential_splits.sort(key=lambda s: -s['gain'])
        
        potential_splits = potential_splits[:max_splits_returned]
        
        logger.info(f"Found {len(potential_splits)} potential splits for node {node_id}")
        
        if len(potential_splits) == 0:
            logger.info(f"No automatic splits found for node {node_id}, providing manual split options")
            potential_splits = self._generate_manual_split_suggestions(X, y, sample_weight, indices, node_id)
        
        self.splitFound.emit(node_id, potential_splits)
        
        logger.info(f"Found {len(potential_splits)} potential splits for node {node_id} (including manual suggestions)")
        
        return potential_splits
    
    def _generate_manual_split_suggestions(self, X: pd.DataFrame, y: pd.Series, 
                                         sample_weight: np.ndarray, indices: List[int], 
                                         node_id: str) -> List[Dict[str, Any]]:
        """
        Generate manual split suggestions when automatic finding fails
        
        Args:
            X: Feature DataFrame
            y: Target variable
            sample_weight: Sample weights
            indices: Sample indices for this node
            node_id: Node ID
            
        Returns:
            List of manual split suggestions
        """
        suggestions = []
        
        try:
            node_X = X.iloc[indices]
            node_y = y.iloc[indices]
            
            if len(node_y) < 2:
                return suggestions
                
            unique_targets = node_y.unique()
            if len(unique_targets) <= 1:
                return suggestions
                
            logger.info(f"Generating manual splits for {len(node_X)} samples with {len(unique_targets)} unique target values")
            
            numeric_features = []
            categorical_features = []
            
            for col in node_X.columns:
                if col in self.categorical_features:
                    categorical_features.append(col)
                else:
                    numeric_features.append(col)
            
            for feature in numeric_features[:10]:  # Limit to top 10
                feature_data = node_X[feature].dropna()
                if len(feature_data) >= 2 and feature_data.var() > 0:
                    median_val = feature_data.median()
                    
                    left_mask = feature_data <= median_val
                    right_mask = ~left_mask
                    
                    if left_mask.sum() > 0 and right_mask.sum() > 0:
                        suggestions.append({
                            'feature': feature,
                            'split_type': 'numeric',
                            'split_desc': f"{feature} <= {median_val:.4f} (median split)",
                            'gain': 0.001,  # Small positive gain to show it's viable
                            'split_info': {
                                'feature': feature,
                                'threshold': median_val,
                                'split_type': 'numeric',
                                'gain': 0.001,
                                'left_samples': int(left_mask.sum()),
                                'right_samples': int(right_mask.sum()),
                                'manual_suggestion': True
                            }
                        })
            
            for feature in categorical_features[:5]:  # Limit to top 5
                feature_data = node_X[feature].dropna()
                unique_cats = feature_data.unique()
                
                if len(unique_cats) >= 2:
                    most_common = feature_data.mode().iloc[0] if len(feature_data.mode()) > 0 else unique_cats[0]
                    
                    left_mask = feature_data == most_common
                    right_mask = ~left_mask
                    
                    if left_mask.sum() > 0 and right_mask.sum() > 0:
                        suggestions.append({
                            'feature': feature,
                            'split_type': 'categorical',
                            'split_desc': f"{feature} == '{most_common}' vs others",
                            'gain': 0.001,  # Small positive gain
                            'split_info': {
                                'feature': feature,
                                'left_categories': [most_common],
                                'right_categories': [cat for cat in unique_cats if cat != most_common],
                                'split_type': 'categorical',
                                'gain': 0.001,
                                'left_samples': int(left_mask.sum()),
                                'right_samples': int(right_mask.sum()),
                                'manual_suggestion': True
                            }
                        })
            
            if hasattr(self, 'feature_importance') and self.feature_importance:
                suggestions.sort(key=lambda s: self.feature_importance.get(s['feature'], 0), reverse=True)
            
            logger.info(f"Generated {len(suggestions)} manual split suggestions for node {node_id}")
            
        except Exception as e:
            logger.error(f"Error generating manual split suggestions: {e}")
            
        return suggestions[:20]  # Limit to top 20 suggestions
    
    def apply_split_to_node(self, node_id: str, split_info: Dict[str, Any], 
                          X: pd.DataFrame = None, y: pd.Series = None, 
                          sample_weight: Optional[np.ndarray] = None) -> bool:
        """
        Apply a split to a node
        
        Args:
            node_id: Node ID
            split_info: Split information
            X: Feature DataFrame
            y: Target variable
            sample_weight: Sample weights
            
        Returns:
            True if split was applied, False otherwise
        """
        node = self.get_node(node_id)
        if not node:
            logger.warning(f"Node {node_id} not found")
            return False
        
        if X is None:
            X = self._cached_X
        if y is None:
            y = self._cached_y
        if sample_weight is None:
            sample_weight = self._cached_sample_weight
            
        if sample_weight is None and X is not None:
            sample_weight = np.ones(len(X))
            
        if X is None or y is None:
            logger.error("No training data available for applying splits")
            return False
        
        left_indices, right_indices = self._apply_split(X, split_info)
        
        self._create_children_from_split(X, y, sample_weight, node, split_info, left_indices, right_indices)
        
        self.num_nodes = len(self._get_all_nodes())
        self.num_leaves = sum(1 for n in self._get_all_nodes() if n.is_terminal)
        self.max_depth = max(n.depth for n in self._get_all_nodes())
        
        self.nodeUpdated.emit(node.node_id)
        
        self.treeUpdated.emit()
        
        self._add_to_history("apply_split", {
            "node_id": node_id,
            "feature": split_info['feature'],
            "split_type": split_info['split_type']
        })
        
        logger.info(f"Applied split to node {node_id} on feature {split_info['feature']}")
        
        return True
    
    def make_node_terminal(self, node_id: str):
        """
        Make a node terminal (leaf)
        
        Args:
            node_id: Node ID
        """
        node = self.get_node(node_id)
        if not node:
            logger.warning(f"Node {node_id} not found")
            return
        
        node.is_terminal = True
        node.children = []  # Remove children
        
        self.num_nodes = len(self._get_all_nodes())
        self.num_leaves = sum(1 for n in self._get_all_nodes() if n.is_terminal)
        
        self.nodeUpdated.emit(node.node_id)
        
        self.treeUpdated.emit()
        
        self._add_to_history("make_terminal", {"node_id": node_id})
        
        logger.info(f"Made node {node_id} terminal")
    
    def prune_subtree(self, node_id: str):
        """
        Prune the subtree rooted at a node
        
        Args:
            node_id: Node ID
        """
        node = self.get_node(node_id)
        if not node:
            logger.warning(f"Node {node_id} not found")
            return
        
        self.make_node_terminal(node_id)
    
    def get_node_report(self, node_id: str) -> Dict[str, Any]:
        """
        Get a report for a node
        
        Args:
            node_id: Node ID
            
        Returns:
            Dictionary with node information
        """
        node = self.get_node(node_id)
        if not node:
            logger.warning(f"Node {node_id} not found")
            return {}
        
        return node.get_node_report()
    
    
    
    def save(self, filepath: str):
        """
        Save the model to a file
        
        Args:
            filepath: Path to save the model
        """
        model_dict = self.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(model_dict, f, indent=2)
        
        logger.info(f"Saved model to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, config: Dict[str, Any] = None) -> 'BespokeDecisionTree':
        """
        Load a model from a file
        
        Args:
            filepath: Path to load the model from
            config: Configuration dictionary
            
        Returns:
            BespokeDecisionTree instance
        """
        with open(filepath, 'r') as f:
            model_dict = json.load(f)
        
        model = cls.from_dict(model_dict, config=config)
        
        logger.info(f"Loaded model from {filepath}")
        
        return model
    
    def copy(self) -> 'BespokeDecisionTree':
        """
        Create a deep copy of the model
        
        Returns:
            New BespokeDecisionTree instance
        """
        try:
            model_dict = self.to_dict()
            
            copied_model = BespokeDecisionTree.from_dict(model_dict, config=self.config)
            
            return copied_model
        except Exception as e:
            logger.error(f"Error copying model: {e}")
            return BespokeDecisionTree(config=self.config)
    
    def export_to_python(self, filepath: Optional[str] = None) -> str:
        """
        Export the decision tree as professional Python code using dedicated exporter
        
        Args:
            filepath: Path to save the Python file (optional)
            
        Returns:
            Python code as a string
        """
        from export.python_exporter import PythonExporter
        
        exporter = PythonExporter(self.config)
        
        if filepath:
            success = exporter.export_model(self, filepath)
            if not success:
                raise RuntimeError("Failed to export model to Python code")
        
        return exporter._generate_professional_python_code(self)
    
    def _add_to_history(self, action: str, details: Dict[str, Any]):
        """
        Add an action to the model history
        
        Args:
            action: Name of the action
            details: Details of the action
        """
        self.history.append({
            'action': action,
            'timestamp': datetime.now().isoformat(),
            'details': details
        })
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        return self.feature_importance
    
    def compute_metrics(self, X: pd.DataFrame, y: pd.Series, 
                       sample_weight: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute performance metrics - Enhanced version with terminal node focus
        
        Args:
            X: Feature DataFrame
            y: Target Series
            sample_weight: Optional sample weights
            
        Returns:
            Dictionary of comprehensive performance metrics
        """
        try:
            if not self.is_fitted:
                logger.error("Model must be fitted before computing metrics")
                return {}
            
            from analytics.performance_metrics import MetricsCalculator
            
            calculator = MetricsCalculator()
            
            metrics = calculator.compute_metrics(
                X=X, 
                y=y, 
                tree_root=self.root,
                sample_weight=sample_weight,
                positive_class=self.positive_class
            )
            
            terminal_metrics = self._calculate_terminal_node_performance(X, y)
            if terminal_metrics:
                metrics['terminal_node_analysis'] = terminal_metrics
            
            structure_metrics = self._calculate_tree_structure_metrics()
            if structure_metrics:
                metrics['tree_structure'] = structure_metrics
            
            self.metrics = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}", exc_info=True)
            return {}
    
    def evaluate_nodes(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None):
        """
        Evaluate all nodes in the tree
        
        Args:
            X: Feature DataFrame
            y: Target variable
            sample_weight: Sample weights
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if sample_weight is None:
            sample_weight = np.ones(len(X))
        
        nodes = self._get_all_nodes()
        
        for node in nodes:
            indices = self._get_node_sample_indices(X, node)
            
            if len(indices) == 0:
                continue
                
            y_node = y.iloc[indices]
            w_node = sample_weight[indices]
            
            y_pred = np.full(len(y_node), node.majority_class)
            
            correct = (y_pred == y_node.to_numpy())
            accuracy = np.sum(correct * w_node) / np.sum(w_node)
            
            node.accuracy = accuracy
            
            if len(self.class_names) == 2:
                pos_class = self.positive_class
                
                y_true_binary = (y_node == pos_class).astype(int)
                y_pred_binary = (y_pred == pos_class).astype(int)
                
                tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
                tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
                fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
                fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                node.precision = precision
                node.recall = recall
                node.f1_score = f1
        
        self.treeUpdated.emit()
        
        logger.info(f"Evaluated {len(nodes)} nodes")
    
    def print_tree(self, node: Optional[TreeNode] = None, depth: int = 0) -> str:
        """
        Print the decision tree structure
        
        Args:
            node: Starting node (default: root)
            depth: Current depth
            
        Returns:
            String representation of the tree
        """
        if node is None:
            node = self.root
        
        output = []
        
        indent = "  " * depth
        
        if node.is_terminal:
            class_info = ", ".join([f"{cls}: {count}" for cls, count in node.class_counts.items()])
            output.append(f"{indent}Terminal Node {node.node_id}: {node.majority_class} ({node.probability:.2f})")
            output.append(f"{indent}  Samples: {node.samples}, Impurity: {node.impurity:.4f}")
            output.append(f"{indent}  Class counts: {class_info}")
        else:
            output.append(f"{indent}Node {node.node_id}: {node.split_rule}")
            output.append(f"{indent}  Samples: {node.samples}, Impurity: {node.impurity:.4f}")
            
            for child in node.children:
                output.append(self.print_tree(child, depth + 1))
        
        return "\n".join(output)
    
    def __str__(self) -> str:
        """
        String representation of the model
        
        Returns:
            String describing the model
        """
        if not self.is_fitted:
            return f"BespokeDecisionTree(fitted=False, params={self.get_params()})"
        
        return (f"BespokeDecisionTree(fitted=True, nodes={self.num_nodes}, leaves={self.num_leaves}, "
                f"max_depth={self.max_depth}, accuracy={self.metrics.get('accuracy', 'N/A')})")
    
    def __repr__(self) -> str:
        """
        Detailed representation of the model
        
        Returns:
            Detailed string describing the model
        """
        params_str = ", ".join([f"{k}={v}" for k, v in self.get_params().items()])
        return f"BespokeDecisionTree({params_str})"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the model to a dictionary
        
        Returns:
            Dictionary representation of the model
        """
        try:
            model_dict = {
                'model_type': 'BespokeDecisionTree',
                'version': '1.0',
                'parameters': getattr(self, '_cached_params', {}),  # Use cached params to prevent recursion
                'fitted': getattr(self, 'is_fitted', False),
                'feature_names': getattr(self, 'feature_names', []),
                'class_names': getattr(self, 'class_names', []),
                'num_nodes': getattr(self, 'num_nodes', 0),
                'num_leaves': getattr(self, 'num_leaves', 0),
                'tree_depth': getattr(self, 'max_depth', 0),
                'feature_importance': getattr(self, 'feature_importance', {}),
                'metrics': getattr(self, 'metrics', {}),
                'active_node_id': getattr(self, 'active_node_id', None),
                'target_name': getattr(self, 'target_name', None),
                'timestamp': datetime.now().isoformat(),
                
                'model_id': getattr(self, 'model_id', None),
                'model_name': getattr(self, 'model_name', None),
                'target_values': getattr(self, 'target_values', None),
                'positive_class': getattr(self, 'positive_class', None),
                'categorical_features': list(getattr(self, 'categorical_features', set())),
                'numerical_features': list(getattr(self, 'numerical_features', set())),
                'training_samples': getattr(self, 'training_samples', 0)
            }
            
            try:
                self._cached_params = self.get_params()
                model_dict['parameters'] = self._cached_params
            except Exception as param_error:
                logger.warning(f"Error getting params during serialization: {param_error}")
                model_dict['parameters'] = {}
            
            if hasattr(self, 'root') and self.root is not None:
                try:
                    model_dict['root_node'] = self.root.to_dict()
                except Exception as tree_error:
                    logger.warning(f"Error serializing tree structure: {tree_error}")
                    model_dict['root_node'] = None
            else:
                model_dict['root_node'] = None
                
            if hasattr(self, '_cached_params'):
                delattr(self, '_cached_params')
                
            return make_json_serializable(model_dict)
            
        except Exception as e:
            if hasattr(self, '_cached_params'):
                delattr(self, '_cached_params')
            logger.error(f"Error serializing BespokeDecisionTree: {e}", exc_info=True)
            raise
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> 'BespokeDecisionTree':
        """
        Deserialize a model from a dictionary
        
        Args:
            data: Dictionary representation of the model
            config: Optional configuration dictionary
            
        Returns:
            BespokeDecisionTree instance
        """
        try:
            if data.get('model_type') != 'BespokeDecisionTree':
                raise ValueError(f"Invalid model type: {data.get('model_type')}")
            
            params = data.get('parameters', {})
            instance = cls(config=config or {})
            
            instance.set_params(**params)
            
            fitted_status = data.get('fitted', False)
            
            instance.feature_names = data.get('feature_names', [])
            instance.class_names = data.get('class_names', [])
            instance.num_nodes = data.get('num_nodes', 0)
            instance.num_leaves = data.get('num_leaves', 0)
            instance.max_depth = data.get('tree_depth', 0)
            instance.feature_importance = data.get('feature_importance', {})
            instance.metrics = data.get('metrics', {})
            instance.active_node_id = data.get('active_node_id')
            
            if data.get('root_node') is not None:
                instance.root = TreeNode.from_dict(data['root_node'])
            else:
                instance.root = None
            
            if 'target_name' in data:
                instance._target_name = data['target_name']
            
            if 'model_id' in data:
                instance.model_id = data['model_id']
            if 'model_name' in data:
                instance.model_name = data['model_name']
            if 'target_values' in data:
                instance.target_values = data['target_values']
            if 'positive_class' in data:
                instance.positive_class = data['positive_class']
            if 'categorical_features' in data:
                instance.categorical_features = set(data['categorical_features'])
            if 'numerical_features' in data:
                instance.numerical_features = set(data['numerical_features'])
            if 'training_samples' in data:
                instance.training_samples = data['training_samples']
            
            instance.is_fitted = fitted_status
                
            logger.info(f"Successfully deserialized BespokeDecisionTree (fitted={instance.is_fitted})")
            return instance
            
        except Exception as e:
            logger.error(f"Error deserializing BespokeDecisionTree: {e}", exc_info=True)
            raise

    def _find_node_by_id(self, node_id: str):
        """Find a node by its ID"""
        if not self.root:
            return None
            
        def search_node(node):
            if node.node_id == node_id:
                return node
            for child in node.children:
                result = search_node(child)
                if result:
                    return result
            return None
            
        return search_node(self.root)



    def _create_numeric_children(self, parent, config):
        """Create children for numeric split with proper threshold handling"""
        children = []
        
        left_child = TreeNode(f"{parent.node_id}_left")
        left_child.parent = parent
        left_child.depth = parent.depth + 1
        left_child.is_terminal = True
        left_child.split_condition = f"<= {config.threshold}"
        children.append(left_child)
        
        right_child = TreeNode(f"{parent.node_id}_right")
        right_child.parent = parent
        right_child.depth = parent.depth + 1
        right_child.is_terminal = True
        right_child.split_condition = f"> {config.threshold}"
        children.append(right_child)
        
        return children
        
    def _create_categorical_children(self, parent, config):
        """Create children for categorical split with proper grouping"""
        children = []
        
        left_categories = config.left_categories
        right_categories = config.right_categories
        
        if left_categories:
            left_child = TreeNode(f"{parent.node_id}_left")
            left_child.parent = parent
            left_child.depth = parent.depth + 1
            left_child.is_terminal = True
            left_child.split_condition = f"in {left_categories}"
            children.append(left_child)
        
        if right_categories:
            right_child = TreeNode(f"{parent.node_id}_right")
            right_child.parent = parent
            right_child.depth = parent.depth + 1
            right_child.is_terminal = True
            right_child.split_condition = f"in {right_categories}"
            children.append(right_child)
            
        return children

    def _update_feature_importance_after_split(self, feature):
        """Update feature importance after applying a split"""
        pass
        
    def _update_tree_statistics_after_split(self, node: TreeNode):
        """Update tree-level statistics after a split is applied"""
        try:
            if hasattr(self, 'num_nodes'):
                self.num_nodes = len(self._get_all_nodes())
            
            if hasattr(self, 'num_leaves'):
                self.num_leaves = len([n for n in self._get_all_nodes() if n.is_terminal])
            
            if hasattr(self, 'max_depth'):
                self.max_depth = max(n.depth for n in self._get_all_nodes())
                
            logger.debug(f"Updated tree statistics: {self.num_nodes} nodes, {self.num_leaves} leaves, depth {self.max_depth}")
            
        except Exception as e:
            logger.error(f"Error updating tree statistics: {e}")
    
    def _get_all_nodes(self) -> List[TreeNode]:
        """Get all nodes in the tree"""
        if not hasattr(self, 'root') or not self.root:
            return []
            
        nodes = []
        def traverse(node):
            nodes.append(node)
            for child in node.children:
                traverse(child)
        
        traverse(self.root)
        return nodes
    
    def _apply_numerical_split(self, node: TreeNode, config: SplitConfiguration,
                             node_X: pd.DataFrame, node_y: pd.Series,
                             transaction: SplitTransaction) -> bool:
        """Apply numerical binary split"""
        try:
            feature = config.feature
            threshold = config.threshold
            
            if feature not in node_X.columns:
                logger.error(f"Feature {feature} not found in node data")
                return False
                
            feature_data = node_X[feature]
            
            left_mask = feature_data <= threshold
            right_mask = ~left_mask & ~feature_data.isna()
            
            left_indices = feature_data.index[left_mask].tolist()
            right_indices = feature_data.index[right_mask].tolist()
            
            if len(left_indices) == 0 or len(right_indices) == 0:
                logger.error("Cannot create binary split with empty children")
                return False
                
            left_child = TreeNode(
                node_id=f"{node.node_id}_L",
                depth=node.depth + 1,
                parent=node
            )
            right_child = TreeNode(
                node_id=f"{node.node_id}_R", 
                depth=node.depth + 1,
                parent=node
            )
            
            left_child.sample_indices = left_indices
            right_child.sample_indices = right_indices
            
            left_child.split_condition = f"<= {threshold}"
            right_child.split_condition = f"> {threshold}"
            
            left_child.is_terminal = True
            right_child.is_terminal = True
            
            left_y = node_y[left_mask]
            right_y = node_y[right_mask]
            
            if len(left_y) > 0:
                left_child.class_counts = left_y.value_counts().to_dict()
                left_child.predicted_class = left_y.mode().iloc[0] if len(left_y.mode()) > 0 else left_y.iloc[0]
                left_child.samples = len(left_y)
                left_child.impurity = self._calculate_node_impurity(left_y)
                
            if len(right_y) > 0:
                right_child.class_counts = right_y.value_counts().to_dict()
                right_child.predicted_class = right_y.mode().iloc[0] if len(right_y.mode()) > 0 else right_y.iloc[0]
                right_child.samples = len(right_y)
                right_child.impurity = self._calculate_node_impurity(right_y)
            
            node.children = [left_child, right_child]
            node.is_terminal = False
            node.split_feature = feature
            node.split_value = threshold
            node.split_type = 'numeric'
            
            transaction.record_node_creation(left_child.node_id)
            transaction.record_node_creation(right_child.node_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in numerical split: {e}")
            return False
    
    def _apply_multi_bin_split(self, node: TreeNode, config: SplitConfiguration,
                             node_X: pd.DataFrame, node_y: pd.Series,
                             transaction: SplitTransaction) -> bool:
        """Apply multi-bin numerical split"""
        try:
            feature = config.feature
            
            if feature not in node_X.columns:
                logger.error(f"Feature {feature} not found in node data")
                return False
                
            feature_data = node_X[feature]
            
            if hasattr(config, 'bins') and config.bins:
                bins = config.bins  # List of (min_val, max_val) tuples
            elif hasattr(config, 'thresholds') and config.thresholds:
                thresholds = config.thresholds
                min_val = feature_data.min()
                max_val = feature_data.max()
                
                bins = []
                for i in range(len(thresholds) + 1):
                    if i == 0:
                        bins.append((min_val, thresholds[0]))
                    elif i == len(thresholds):
                        bins.append((thresholds[-1], max_val))
                    else:
                        bins.append((thresholds[i-1], thresholds[i]))
            else:
                logger.error("Multi-bin split requires bin definitions or thresholds")
                return False
            
            child_nodes = []
            all_assigned_indices = set()
            
            for i, (min_val, max_val) in enumerate(bins):
                if i == len(bins) - 1:  # Last bin includes max value
                    bin_mask = (feature_data >= min_val) & (feature_data <= max_val)
                else:
                    bin_mask = (feature_data >= min_val) & (feature_data < max_val)
                    
                bin_indices = feature_data.index[bin_mask].tolist()
                
                if len(bin_indices) == 0:
                    logger.warning(f"Bin {i+1} ({min_val}-{max_val}) is empty")
                    continue
                    
                child_node = TreeNode(
                    node_id=f"{node.node_id}_bin_{i}",
                    depth=node.depth + 1,
                    parent=node
                )
                
                child_node.sample_indices = bin_indices
                child_node.split_feature = feature
                child_node.split_condition = f"[{min_val:.3f}, {max_val:.3f}]"
                child_node.is_terminal = True
                
                bin_y = node_y[bin_mask]
                if len(bin_y) > 0:
                    child_node.class_counts = bin_y.value_counts().to_dict()
                    child_node.predicted_class = bin_y.mode().iloc[0] if len(bin_y.mode()) > 0 else bin_y.iloc[0]
                    child_node.samples = len(bin_y)
                    child_node.impurity = self._calculate_node_impurity(bin_y)
                
                child_nodes.append(child_node)
                all_assigned_indices.update(bin_indices)
                
                transaction.record_node_creation(child_node.node_id)
                
            total_node_samples = len(feature_data)
            assigned_samples = len(all_assigned_indices)
            
            if assigned_samples != total_node_samples:
                all_indices = set(feature_data.index)
                unassigned_indices = all_indices - all_assigned_indices
                
                if unassigned_indices:
                    logger.warning(f"Found {len(unassigned_indices)} unassigned samples, assigning to largest bin")
                    
                    if child_nodes:
                        largest_bin = max(child_nodes, key=lambda x: x.samples)
                        largest_bin.sample_indices.extend(list(unassigned_indices))
                        
                        unassigned_mask = feature_data.index.isin(unassigned_indices)
                        unassigned_y = node_y[unassigned_mask]
                        if len(unassigned_y) > 0:
                            all_bin_y = pd.concat([node_y[node_y.index.isin(largest_bin.sample_indices)]])
                            largest_bin.class_counts = all_bin_y.value_counts().to_dict()
                            largest_bin.predicted_class = all_bin_y.mode().iloc[0] if len(all_bin_y.mode()) > 0 else all_bin_y.iloc[0]
                            largest_bin.samples = len(all_bin_y)
                            largest_bin.impurity = self._calculate_node_impurity(all_bin_y)
                        
                        all_assigned_indices.update(unassigned_indices)
                        assigned_samples = len(all_assigned_indices)
                        
                if assigned_samples != total_node_samples:
                    logger.error(f"Sample assignment mismatch after fix: {assigned_samples}/{total_node_samples}")
                    return False
                
            node.children = child_nodes
            node.is_terminal = False
            node.split_feature = feature
            node.split_type = 'numeric_multi_bin'
            
            if hasattr(config, 'thresholds') and config.thresholds:
                node.split_thresholds = sorted(config.thresholds)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in multi-bin split: {e}")
            return False

                
            if split_type == 'numeric':
                threshold = split_info.get('threshold') or split_info.get('split_value')
                if threshold is None:
                    logger.error("Numeric split requires threshold")
                    return False
                    
                try:
                    threshold = float(threshold)
                except (ValueError, TypeError):
                    logger.error(f"Invalid threshold value: {threshold}")
                    return False
                    
                success = self._apply_numeric_split_compatible(node, feature, threshold, node_X, node_y)
                
            elif split_type == 'categorical':
                left_categories = split_info.get('left_categories', [])
                if not left_categories:
                    logger.error("Categorical split requires left_categories")
                    return False
                    
                success = self._apply_categorical_split_compatible(node, feature, left_categories, node_X, node_y)
                
            elif split_type == 'numeric_multi_bin':
                bins = split_info.get('bins', [])
                if not bins or len(bins) < 2:
                    logger.error("Multi-bin split requires at least 2 bins")
                    return False
                
                first_bin = bins[0]
                if isinstance(first_bin, dict) and 'max' in first_bin:
                    threshold = first_bin['max']
                elif isinstance(first_bin, (list, tuple)) and len(first_bin) >= 2:
                    threshold = first_bin[1]  # Use max value of first bin
                else:
                    logger.error(f"Invalid bin format: {first_bin}")
                    return False
                
                try:
                    threshold = float(threshold)
                    logger.info(f"Converting multi-bin split to binary threshold: {threshold}")
                    success = self._apply_numeric_split_compatible(node, feature, threshold, node_X, node_y)
                except (ValueError, TypeError):
                    logger.error(f"Invalid threshold from multi-bin: {threshold}")
                    return False
                
            else:
                logger.error(f"Split type '{split_type}' not implemented in legacy method")
                return False
                
            if success:
                if hasattr(self, '_add_to_history'):
                    try:
                        self._add_to_history("apply_split", {
                            "node_id": node_id,
                            "feature": feature,
                            "split_type": split_type
                        })
                    except Exception as e:
                        logger.warning(f"Failed to add to history: {e}")
                
                logger.info(f"Successfully applied legacy split to node {node_id}")
                return True
            else:
                logger.error(f"Failed to apply legacy split to node {node_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error in legacy split application: {e}", exc_info=True)
            return False

    def _apply_numeric_split_compatible(self, node: 'TreeNode', feature: str, threshold: float, 
                                       node_X: pd.DataFrame, node_y: pd.Series) -> bool:
        """Apply numeric split using existing TreeNode interface"""
        try:
            if feature not in node_X.columns:
                logger.error(f"Feature {feature} not found in node data")
                return False
                
            feature_data = node_X[feature]
            
            valid_mask = ~feature_data.isna()
            if valid_mask.sum() == 0:
                logger.error("All values are missing for feature")
                return False
                
            left_mask = (feature_data <= threshold) & valid_mask
            right_mask = (feature_data > threshold) & valid_mask
            
            left_count = left_mask.sum()
            right_count = right_mask.sum()
            
            if left_count == 0:
                logger.error(f"Left child would be empty (threshold={threshold})")
                return False
            if right_count == 0:
                logger.error(f"Right child would be empty (threshold={threshold})")
                return False
                
            left_child = TreeNode(
                node_id=f"{node.node_id}_L",
                parent=node,  # Set parent in constructor
                depth=node.depth + 1,
                is_terminal=True
            )
            right_child = TreeNode(
                node_id=f"{node.node_id}_R", 
                parent=node,  # Set parent in constructor
                depth=node.depth + 1,
                is_terminal=True
            )
            
            left_child.samples = int(left_count)
            right_child.samples = int(right_count)
            
            left_child.split_rule = f"{feature} <= {threshold}"
            right_child.split_rule = f"{feature} > {threshold}"
            
            if node_y is not None:
                left_indices = node_X.index[left_mask]
                right_indices = node_X.index[right_mask]
                
                left_y = node_y.loc[left_indices]
                right_y = node_y.loc[right_indices]
                
                left_child.class_counts = left_y.value_counts().to_dict()
                right_child.class_counts = right_y.value_counts().to_dict()
                
                left_child.majority_class = left_y.mode().iloc[0] if len(left_y) > 0 else None
                right_child.majority_class = right_y.mode().iloc[0] if len(right_y) > 0 else None
                
                if left_child.majority_class and left_child.samples > 0:
                    left_child.probability = left_child.class_counts.get(left_child.majority_class, 0) / left_child.samples
                if right_child.majority_class and right_child.samples > 0:
                    right_child.probability = right_child.class_counts.get(right_child.majority_class, 0) / right_child.samples
            
            if hasattr(node, 'set_split'):
                try:
                    node.set_split(feature, threshold, 'numeric')
                except Exception as e:
                    logger.warning(f"set_split method failed, using direct assignment: {e}")
                    node.split_feature = feature
                    node.split_value = threshold
                    node.split_type = 'numeric'
                    node.is_terminal = False
            else:
                node.split_feature = feature
                node.split_value = threshold
                node.split_type = 'numeric'
                node.split_rule = f"{feature} <= {threshold}"
                node.is_terminal = False
            
            if hasattr(node, 'add_child'):
                try:
                    node.add_child(left_child)
                    node.add_child(right_child)
                except Exception as e:
                    logger.warning(f"add_child method failed, using direct assignment: {e}")
                    node.children = [left_child, right_child]
            else:
                node.children = [left_child, right_child]
                
            left_child.parent = node
            right_child.parent = node
            
            logger.info(f"Successfully applied numeric split: {feature} <= {threshold}")
            return True
            
        except Exception as e:
            logger.error(f"Error in numeric split application: {e}", exc_info=True)
            return False

    def _apply_categorical_split_compatible(self, node: 'TreeNode', feature: str, left_categories: List[str],
                                           node_X: pd.DataFrame, node_y: pd.Series) -> bool:
        """Apply categorical split using existing TreeNode interface"""
        try:
            if feature not in node_X.columns:
                logger.error(f"Feature {feature} not found in node data")
                return False
                
            feature_data = node_X[feature]
            
            left_categories_set = set(left_categories)
            all_categories = set(feature_data.dropna().unique())
            right_categories_set = all_categories - left_categories_set
            
            if not left_categories_set or not right_categories_set:
                logger.error("Invalid categorical split: empty category sets")
                return False
                
            left_mask = feature_data.isin(left_categories_set)
            right_mask = feature_data.isin(right_categories_set)
            
            left_count = left_mask.sum()
            right_count = right_mask.sum()
            
            if left_count == 0:
                logger.error(f"Left child would be empty with categories: {left_categories}")
                return False
            if right_count == 0:
                logger.error(f"Right child would be empty with remaining categories")
                return False
                
            left_child = TreeNode(
                node_id=f"{node.node_id}_L",
                parent=node,
                depth=node.depth + 1,
                is_terminal=True
            )
            right_child = TreeNode(
                node_id=f"{node.node_id}_R",
                parent=node,
                depth=node.depth + 1,
                is_terminal=True
            )
            
            left_child.samples = int(left_count)
            right_child.samples = int(right_count)
            
            left_child.split_rule = f"{feature} in {list(left_categories_set)}"
            right_child.split_rule = f"{feature} in {list(right_categories_set)}"
            
            if node_y is not None:
                left_indices = node_X.index[left_mask]
                right_indices = node_X.index[right_mask]
                
                left_y = node_y.loc[left_indices]
                right_y = node_y.loc[right_indices]
                
                left_child.class_counts = left_y.value_counts().to_dict()
                right_child.class_counts = right_y.value_counts().to_dict()
                
                left_child.majority_class = left_y.mode().iloc[0] if len(left_y) > 0 else None
                right_child.majority_class = right_y.mode().iloc[0] if len(right_y) > 0 else None
                
                if left_child.majority_class and left_child.samples > 0:
                    left_child.probability = left_child.class_counts.get(left_child.majority_class, 0) / left_child.samples
                if right_child.majority_class and right_child.samples > 0:
                    right_child.probability = right_child.class_counts.get(right_child.majority_class, 0) / right_child.samples
            
            if hasattr(node, 'set_categorical_split'):
                try:
                    node.set_categorical_split(feature, left_categories, list(right_categories_set))
                except Exception as e:
                    logger.warning(f"set_categorical_split method failed, using direct assignment: {e}")
                    node.split_feature = feature
                    node.split_type = 'categorical'
                    node.is_terminal = False
                    node.split_categories = {}
                    for cat in left_categories_set:
                        node.split_categories[cat] = 0
                    for cat in right_categories_set:
                        node.split_categories[cat] = 1
            else:
                node.split_feature = feature
                node.split_type = 'categorical'
                node.is_terminal = False
                node.split_rule = f"{feature} categorical split"
                node.split_categories = {}
                for cat in left_categories_set:
                    node.split_categories[cat] = 0
                for cat in right_categories_set:
                    node.split_categories[cat] = 1
            
            if hasattr(node, 'add_child'):
                try:
                    node.add_child(left_child)
                    node.add_child(right_child)
                except Exception as e:
                    logger.warning(f"add_child method failed, using direct assignment: {e}")
                    node.children = [left_child, right_child]
            else:
                node.children = [left_child, right_child]
                
            left_child.parent = node
            right_child.parent = node
            
            logger.info(f"Successfully applied categorical split on {feature}")
            return True
            
        except Exception as e:
            logger.error(f"Error in categorical split application: {e}", exc_info=True)
            return False

    def _apply_categorical_multi_bin_split_legacy(self, node: TreeNode, split_info: Dict[str, Any]) -> bool:
        """
        CRITICAL FIX: Apply categorical multi-bin split (NEW FUNCTIONALITY)
        Follows existing patterns for maximum compatibility
        """
        try:
            feature = split_info['feature']
            split_categories = split_info.get('split_categories', {})
            
            if not split_categories:
                logger.error("No split_categories provided for categorical_multi_bin split")
                return False
            
            node.split_feature = feature
            node.split_type = 'categorical_multi_bin'
            node.split_categories = split_categories
            node.split_rule = f"{feature} multi-bin split"
            
            children = []
            for bin_name, categories in split_categories.items():
                child = TreeNode(
                    node_id=f"{node.node_id}_{bin_name}",
                    parent=node,
                    depth=node.depth + 1,
                    is_terminal=True
                )
                children.append(child)
                
                if hasattr(self, 'nodes'):
                    self.nodes[child.node_id] = child
            
            node.children = children
            
            if hasattr(self, '_calculate_child_statistics_legacy'):
                self._calculate_child_statistics_legacy(node, children)
            elif hasattr(self, '_calculate_child_statistics'):
                self._calculate_child_statistics(node, children)
            
            logger.info(f"Successfully applied categorical multi-bin split on {feature} with {len(children)} bins")
            return True
            
        except Exception as e:
            logger.error(f"Error applying categorical multi-bin split: {e}")
            return False

    def _split_categorical_multi_bin_data_legacy(self, parent_node: TreeNode, children: List[TreeNode],
                                                 node_data: pd.DataFrame, node_target: pd.Series):
        """
        CRITICAL FIX: Split categorical multi-bin data for child statistics
        Follows existing data splitting patterns
        """
        try:
            feature = parent_node.split_feature
            split_categories = parent_node.split_categories
            
            if feature not in node_data.columns:
                logger.error(f"Feature {feature} not found in data")
                return
            
            for i, (bin_name, categories) in enumerate(split_categories.items()):
                if i < len(children):
                    bin_mask = node_data[feature].isin(categories)
                    bin_target = node_target[bin_mask]
                    
                    if len(bin_target) > 0:
                        bin_class_counts = bin_target.value_counts().to_dict()
                        children[i].update_stats(
                            sample_count=len(bin_target),
                            class_counts=bin_class_counts
                        )
                        
        except Exception as e:
            logger.error(f"Error splitting categorical multi-bin data: {e}")

    def get_terminal_nodes(self) -> List[TreeNode]:
        """
        Get all terminal nodes in the tree
        
        Returns:
            List of terminal TreeNode objects
        """
        try:
            if not self.is_fitted or not self.root:
                return []
            
            all_nodes = self.root.get_subtree_nodes()
            return [node for node in all_nodes if node.is_terminal]
            
        except Exception as e:
            logger.error(f"Error getting terminal nodes: {e}")
            return []

    def _calculate_terminal_node_performance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Calculate performance metrics specifically for terminal nodes
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dictionary with terminal node performance analysis
        """
        try:
            terminal_nodes = self.get_terminal_nodes()
            
            if not terminal_nodes:
                return {}
            
            terminal_metrics = {
                'total_terminal_nodes': len(terminal_nodes),
                'node_performance': {},
                'aggregate_performance': {}
            }
            
            total_correct = 0
            total_samples = 0
            node_accuracies = []
            node_purities = []
            
            for node in terminal_nodes:
                node_predictions = []
                node_actuals = []
                
                node_info = {
                    'node_id': node.node_id,
                    'samples': getattr(node, 'samples', 0),
                    'prediction': getattr(node, 'majority_class', None),
                    'class_counts': getattr(node, 'class_counts', {}),
                    'impurity': getattr(node, 'impurity', None)
                }
                
                if node.class_counts:
                    total_node_samples = sum(node.class_counts.values())
                    if total_node_samples > 0:
                        max_class_count = max(node.class_counts.values())
                        purity = max_class_count / total_node_samples
                        node_info['purity'] = purity
                        node_purities.append(purity)
                        
                        total_correct += max_class_count
                        total_samples += total_node_samples
                
                for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    value = getattr(node, metric, None)
                    if value is not None:
                        node_info[metric] = value
                        if metric == 'accuracy':
                            node_accuracies.append(value)
                
                terminal_metrics['node_performance'][node.node_id] = node_info
            
            if node_accuracies:
                terminal_metrics['aggregate_performance']['accuracy_stats'] = {
                    'min': min(node_accuracies),
                    'max': max(node_accuracies),
                    'mean': np.mean(node_accuracies),
                    'std': np.std(node_accuracies)
                }
            
            if node_purities:
                terminal_metrics['aggregate_performance']['purity_stats'] = {
                    'min': min(node_purities),
                    'max': max(node_purities),
                    'mean': np.mean(node_purities),
                    'std': np.std(node_purities)
                }
            
            if total_samples > 0:
                terminal_metrics['aggregate_performance']['weighted_terminal_accuracy'] = total_correct / total_samples
            
            return terminal_metrics
            
        except Exception as e:
            logger.error(f"Error calculating terminal node performance: {e}")
            return {}

    def _calculate_tree_structure_metrics(self) -> Dict[str, Any]:
        """
        Calculate metrics about the tree structure
        
        Returns:
            Dictionary with tree structure metrics
        """
        try:
            if not self.root:
                return {}
            
            all_nodes = self.root.get_subtree_nodes()
            terminal_nodes = [node for node in all_nodes if node.is_terminal]
            internal_nodes = [node for node in all_nodes if not node.is_terminal]
            
            structure_metrics = {
                'total_nodes': len(all_nodes),
                'terminal_nodes': len(terminal_nodes),
                'internal_nodes': len(internal_nodes),
                'max_depth': self.max_depth,
                'balance_factor': self._calculate_balance_factor()
            }
            
            depth_counts = {}
            for node in all_nodes:
                depth = node.depth
                depth_counts[depth] = depth_counts.get(depth, 0) + 1
            
            structure_metrics['depth_distribution'] = depth_counts
            
            if internal_nodes:
                feature_usage = {}
                for node in internal_nodes:
                    feature = getattr(node, 'split_feature', None)
                    if feature:
                        feature_usage[feature] = feature_usage.get(feature, 0) + 1
                
                structure_metrics['feature_usage'] = feature_usage
                structure_metrics['unique_features_used'] = len(feature_usage)
            
            if terminal_nodes:
                terminal_samples = [getattr(node, 'samples', 0) for node in terminal_nodes]
                structure_metrics['terminal_sample_stats'] = {
                    'min': min(terminal_samples),
                    'max': max(terminal_samples),
                    'mean': np.mean(terminal_samples),
                    'std': np.std(terminal_samples)
                }
            
            return structure_metrics
            
        except Exception as e:
            logger.error(f"Error calculating tree structure metrics: {e}")
            return {}

    def _calculate_balance_factor(self) -> float:
        """
        Calculate the balance factor of the tree
        
        Returns:
            Balance factor (0 = perfectly balanced, 1 = completely unbalanced)
        """
        try:
            if not self.root:
                return 0.0
            
            terminal_nodes = self.get_terminal_nodes()
            
            if len(terminal_nodes) <= 1:
                return 0.0
            
            depths = [node.depth for node in terminal_nodes]
            min_depth = min(depths)
            max_depth = max(depths)
            
            if max_depth == 0:
                return 0.0
            
            balance_factor = (max_depth - min_depth) / max_depth
            return balance_factor
            
        except Exception as e:
            logger.error(f"Error calculating balance factor: {e}")
            return 0.0

    def get_visualization_config(self) -> Dict[str, Any]:
        """
        Get configuration data needed for enhanced visualization
        
        Returns:
            Dictionary with visualization configuration
        """
        config = {
            'target_variable_name': self.target_name or 'FLAG_EMP_PHONE',  # Use actual target variable name
            'criterion': self.criterion,
            'model_name': self.model_name,
            'model_id': self.model_id,
            'is_fitted': self.is_fitted,
            'class_names': getattr(self, 'class_names', []),
            'positive_class': getattr(self, 'positive_class', None),
            'feature_names': getattr(self, 'feature_names', [])
        }
        
        return config

    def update_node_metrics(self, X: pd.DataFrame, y: pd.Series):
        """
        Update performance metrics for all nodes in the tree
        
        Args:
            X: Feature DataFrame
            y: Target Series
        """
        try:
            if not self.is_fitted or not self.root:
                logger.warning("Cannot update node metrics: model not fitted or no root node")
                return
            
            all_nodes = self.root.get_subtree_nodes()
            
            predictions = self.predict(X)
            
            for node in all_nodes:
                if node.is_terminal:
                    node_accuracy = getattr(node, 'accuracy', None)
                    if node_accuracy is None and hasattr(node, 'class_counts') and node.class_counts:
                        total_samples = sum(node.class_counts.values())
                        max_class_count = max(node.class_counts.values())
                        node.accuracy = max_class_count / total_samples if total_samples > 0 else 0.0
                    
                    if hasattr(node, 'majority_class') and node.majority_class is not None:
                        if len(self.class_names) == 2:  # Binary classification
                            if node.class_counts and len(node.class_counts) == 2:
                                classes = list(node.class_counts.keys())
                                if node.majority_class in classes:
                                    tp = node.class_counts[node.majority_class]
                                    fn = sum(count for class_label, count in node.class_counts.items() 
                                            if class_label != node.majority_class)
                                    
                                    node.precision = tp / (tp + 0.1) if tp > 0 else 0.0  # Simplified
                                    node.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                                    
                                    if node.precision + node.recall > 0:
                                        node.f1_score = 2 * node.precision * node.recall / (node.precision + node.recall)
                                    else:
                                        node.f1_score = 0.0
            
            self.treeUpdated.emit()
            
            logger.info(f"Updated metrics for {len(all_nodes)} nodes")
            
        except Exception as e:
            logger.error(f"Error updating node metrics: {e}", exc_info=True)

    def get_node_data_for_visualization(self) -> Dict[str, Any]:
        """
        Get all node data formatted for visualization
        
        Returns:
            Dictionary with node data for visualization
        """
        try:
            if not self.is_fitted or not self.root:
                return {}
            
            all_nodes = self.root.get_subtree_nodes()
            
            nodes_by_level = {}
            for node in all_nodes:
                level = node.depth
                if level not in nodes_by_level:
                    nodes_by_level[level] = []
                nodes_by_level[level].append(node)
            
            node_numbering = {}
            node_number = 1
            for level in sorted(nodes_by_level.keys()):
                level_nodes = sorted(nodes_by_level[level], key=lambda n: n.node_id)
                for node in level_nodes:
                    node_numbering[node.node_id] = node_number
                    node_number += 1
            
            visualization_data = {
                'nodes': {},
                'node_numbering': node_numbering,
                'tree_stats': {
                    'total_nodes': len(all_nodes),
                    'terminal_nodes': len([n for n in all_nodes if n.is_terminal]),
                    'max_depth': max(n.depth for n in all_nodes) if all_nodes else 0
                },
                'model_config': self.get_visualization_config()
            }
            
            for node in all_nodes:
                node_data = {
                    'node_id': node.node_id,
                    'node_number': node_numbering[node.node_id],
                    'depth': node.depth,
                    'is_terminal': node.is_terminal,
                    'samples': getattr(node, 'samples', 0),
                    'class_counts': getattr(node, 'class_counts', {}),
                    'impurity': getattr(node, 'impurity', None),
                    'majority_class': getattr(node, 'majority_class', None),
                    'split_feature': getattr(node, 'split_feature', None),
                    'split_value': getattr(node, 'split_value', None),
                    'split_type': getattr(node, 'split_type', None),
                    'split_categories': getattr(node, 'split_categories', {}),
                    'performance_metrics': {
                        'accuracy': getattr(node, 'accuracy', None),
                        'precision': getattr(node, 'precision', None),
                        'recall': getattr(node, 'recall', None),
                        'f1_score': getattr(node, 'f1_score', None)
                    }
                }
                
                visualization_data['nodes'][node.node_id] = node_data
            
            return visualization_data
            
        except Exception as e:
            logger.error(f"Error getting visualization data: {e}")
            return {}
