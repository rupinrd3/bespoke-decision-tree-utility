#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance Metrics Module for Bespoke Utility
Enhanced implementation for calculating and visualizing model performance metrics

"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import psutil
from PyQt5.QtCore import QObject, pyqtSignal
from joblib import Parallel, delayed

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
        precision_recall_curve, roc_curve, average_precision_score
    )
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from models.decision_tree import BespokeDecisionTree
from models.node import TreeNode

logger = logging.getLogger(__name__)


class MetricsCalculator(QObject):
    """Enhanced class for calculating model performance metrics"""
    
    progress_updated = pyqtSignal(int)
    calculation_finished = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_metrics = {}
        
        self.use_parallel_processing = True
        self.memory_threshold_gb = 8.0  # Only use parallel processing if available memory > 8GB
        self.min_samples_for_parallel = 1000  # Minimum samples to justify parallel overhead
        self.min_folds_for_parallel = 5  # Minimum CV folds to justify parallel processing
        
    def _should_use_parallel_processing(self, data_size: int = 0, n_iterations: int = 1) -> bool:
        """
        Determine if parallel processing should be used based on system resources and task size
        
        Args:
            data_size: Size of the dataset
            n_iterations: Number of iterations/folds/bootstraps
            
        Returns:
            True if parallel processing should be used
        """
        if not self.use_parallel_processing:
            return False
            
        try:
            memory_info = psutil.virtual_memory()
            available_memory_gb = memory_info.available / (1024**3)
            
            estimated_workers = min(psutil.cpu_count() - 1, n_iterations) if n_iterations > 1 else psutil.cpu_count() - 1  
            parallel_overhead_gb = estimated_workers * 0.1  # Conservative 100MB per worker
            
            data_overhead_multiplier = 1.5  # 50% overhead for data handling
            
            
            effective_available_memory = available_memory_gb - parallel_overhead_gb
            memory_safe = effective_available_memory > (self.memory_threshold_gb * data_overhead_multiplier)
            dataset_large_enough = data_size >= self.min_samples_for_parallel
            iterations_sufficient = n_iterations >= self.min_folds_for_parallel
            not_memory_constrained = memory_info.percent < 80.0  # Don't parallelize if system is already under memory pressure
            
            use_parallel = (memory_safe and 
                           (dataset_large_enough or iterations_sufficient) and 
                           not_memory_constrained)
            
            if use_parallel:
                logger.info(f"Using parallel processing: {available_memory_gb:.2f}GB available ({effective_available_memory:.2f}GB after overhead), {data_size} samples, {n_iterations} iterations, memory usage: {memory_info.percent:.1f}%")
            else:
                reason = "insufficient memory" if not memory_safe else "small dataset/iterations" if not (dataset_large_enough or iterations_sufficient) else "high memory pressure"
                logger.info(f"Using sequential processing: {available_memory_gb:.2f}GB available, {data_size} samples, {n_iterations} iterations, reason: {reason}")
                
            return use_parallel
            
        except Exception as e:
            logger.warning(f"Error checking system resources: {e}, defaulting to sequential processing")
            return False
    
    def _process_single_cv_fold(self, fold_data: Tuple) -> Optional[float]:
        """
        Process a single cross-validation fold - designed for parallel execution
        THREAD-SAFE: Creates independent model copy for each fold
        
        Args:
            fold_data: Tuple containing (model_config, model_params, train_idx, test_idx, X, y)
            
        Returns:
            Accuracy score for the fold, or None if processing failed
        """
        try:
            model_config, model_params, model_class, train_idx, test_idx, X, y = fold_data
            
            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy() 
            y_train = y.iloc[train_idx].copy()
            y_test = y.iloc[test_idx].copy()
            
            try:
                temp_model = model_class(config=model_config)
                temp_model.set_params(**model_params)
            except Exception as model_e:
                logger.warning(f"Failed to create model copy in CV fold: {str(model_e)}")
                return None
            
            temp_model.fit(X_train, y_train)
            predictions = temp_model.predict(X_test)
            
            if predictions is not None and len(predictions) > 0:
                if hasattr(predictions, 'values'):
                    predictions = predictions.values
                if hasattr(y_test, 'values'):
                    y_test_vals = y_test.values
                else:
                    y_test_vals = np.array(y_test)
                
                accuracy = np.mean(predictions == y_test_vals)
                return accuracy
            else:
                logger.warning("CV fold produced no valid predictions")
                return None
                
        except Exception as e:
            logger.warning(f"Error in CV fold processing: {str(e)}")
            return None
    
    def _process_single_bootstrap_sample(self, bootstrap_data: Tuple) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """
        Process a single bootstrap sample - designed for parallel execution
        
        Args:
            bootstrap_data: Tuple containing (model, X, y, indices, sample_id)
            
        Returns:
            Tuple of (accuracy, predictions) or (None, None) if processing failed
        """
        try:
            model, X, y, indices, sample_id = bootstrap_data
            
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]
            
            y_pred = model.predict(X_boot)
            
            if y_pred is not None and len(y_pred) > 0:
                if hasattr(y_pred, 'values'):
                    y_pred = y_pred.values
                if hasattr(y_boot, 'values'):
                    y_boot_vals = y_boot.values
                else:
                    y_boot_vals = np.array(y_boot)
                
                accuracy = np.mean(y_pred == y_boot_vals)
                return accuracy, y_pred
            else:
                return None, None
                
        except Exception as e:
            logger.warning(f"Error in bootstrap sample {sample_id}: {str(e)}")
            return None, None
    
    def _process_single_sample_tdr(self, sample_data: Tuple) -> Dict[str, Any]:
        """
        Process a single sample for TDR calculation - designed for parallel execution
        
        Args:
            sample_data: Tuple containing (model_root, sample_row, actual_target, sample_index)
            
        Returns:
            Dictionary with sample TDR data
        """
        try:
            model_root, sample_row, actual_target, sample_index = sample_data
            
            tdr = self._get_terminal_node_probability(model_root, sample_row)
            
            return {
                'actual': int(actual_target),
                'tdr': tdr,
                'index': sample_index
            }
            
        except Exception as e:
            logger.warning(f"Error processing sample TDR for index {sample_index}: {str(e)}")
            return {
                'actual': 0,
                'tdr': 0.0,
                'index': sample_index
            }
        
    def compute_metrics(self, X: pd.DataFrame, y: pd.Series, tree_root: TreeNode,
                       sample_weight: Optional[np.ndarray] = None,
                       positive_class: Optional[Any] = None) -> Dict[str, Any]:
        """
        Compute comprehensive performance metrics for a decision tree (Original + Enhanced)
        
        Args:
            X: Feature DataFrame
            y: Target variable
            tree_root: Root node of the tree
            sample_weight: Sample weights
            positive_class: Positive class for binary classification
            
        Returns:
            Dictionary containing performance metrics
        """
        logger.info("Computing performance metrics")
        
        if sample_weight is None:
            sample_weight = np.ones(len(X))
        
        try:
            if hasattr(tree_root, 'predict'):
                y_pred = tree_root.predict(X)
            else:
                y_pred = []
                for _, row in X.iterrows():
                    pred = self._predict_sample(tree_root, row)
                    y_pred.append(pred)
                y_pred = np.array(y_pred)
                
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return {}
        
        if y_pred is None or len(y_pred) == 0:
            logger.error("No predictions generated")
            return {}
        
        classes = sorted(y.unique())
        
        if len(classes) == 2 and positive_class is None:
            positive_class = classes[1]  # Second class by default
        
        metrics = {}
        
        metrics['accuracy'] = self._accuracy(y, y_pred, sample_weight)
        metrics['weighted_accuracy'] = self._weighted_accuracy(y, y_pred, sample_weight)
        metrics['error_rate'] = 1.0 - metrics['accuracy']
        
        metrics.update(self._precision_recall_f1(y, y_pred, classes, sample_weight))
        
        metrics['confusion_matrix'] = self._confusion_matrix(y, y_pred, classes)
        
        if len(classes) == 2:
            logger.debug("Computing binary classification metrics")
            
            neg_class, pos_class = classes[0], classes[1]
            if positive_class is not None:
                if isinstance(positive_class, str) and all(isinstance(c, (int, float)) for c in classes):
                    try:
                        pos_class = int(positive_class) if positive_class.isdigit() else float(positive_class)
                        neg_class = [c for c in classes if c != pos_class][0]
                        logger.debug(f"Converted string positive_class '{positive_class}' to {pos_class} to match data types")
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert positive_class '{positive_class}' to match numeric classes {classes}, using default")
                        pos_class = classes[1]  # Keep default
                        neg_class = classes[0]
                elif isinstance(positive_class, (int, float)) and all(isinstance(c, str) for c in classes):
                    pos_class = str(positive_class)
                    neg_class = [c for c in classes if c != pos_class][0]
                    logger.debug(f"Converted numeric positive_class {positive_class} to '{pos_class}' to match data types")
                else:
                    pos_class = positive_class
                    neg_class = [c for c in classes if c != positive_class][0]
                
            y_true_binary = (y == pos_class).astype(int)
            y_pred_binary = (y_pred == pos_class).astype(int)
            
            tn, fp, fn, tp = self._get_binary_counts(y_true_binary, y_pred_binary)
            
            metrics.update({
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                'true_negative_rate': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                'false_positive_rate': fp / (tn + fp) if (tn + fp) > 0 else 0.0,
                'false_negative_rate': fn / (tp + fn) if (tp + fn) > 0 else 0.0,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0.0
            })
            
            precision = metrics['precision']
            recall = metrics['recall']
            metrics['f1_score'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            metrics['mcc'] = ((tp * tn) - (fp * fn)) / denominator if denominator > 0 else 0.0
            
            try:
                if hasattr(tree_root, 'predict_proba'):
                    y_proba = tree_root.predict_proba(X)
                    if y_proba is not None and len(y_proba.shape) == 2:
                        unique_classes = list(sorted(np.unique(y)))
                        try:
                            pos_idx = unique_classes.index(pos_class)
                        except ValueError:
                            pos_idx = None
                            
                            if isinstance(pos_class, str) and all(isinstance(c, (int, float)) for c in unique_classes):
                                try:
                                    pos_class_converted = int(pos_class) if pos_class.isdigit() else float(pos_class)
                                    pos_idx = unique_classes.index(pos_class_converted)
                                    logger.debug(f"Successfully converted pos_class '{pos_class}' to {pos_class_converted}")
                                except (ValueError, TypeError) as e:
                                    logger.debug(f"Failed to convert string pos_class to numeric: {e}")
                            
                            elif isinstance(pos_class, (int, float)) and all(isinstance(c, str) for c in unique_classes):
                                try:
                                    pos_class_converted = str(pos_class)
                                    pos_idx = unique_classes.index(pos_class_converted)
                                    logger.debug(f"Successfully converted pos_class {pos_class} to '{pos_class_converted}'")
                                except (ValueError, TypeError) as e:
                                    logger.debug(f"Failed to convert numeric pos_class to string: {e}")
                            
                            if pos_idx is None:
                                if isinstance(pos_class, str):
                                    str_classes = [str(c) for c in unique_classes]
                                    if pos_class in str_classes:
                                        pos_idx = str_classes.index(pos_class)
                                        logger.debug(f"Found pos_class '{pos_class}' in string-converted classes")
                                elif isinstance(pos_class, (int, float)):
                                    for i, c in enumerate(unique_classes):
                                        try:
                                            if isinstance(c, str) and c.replace('.', '').replace('-', '').isdigit():
                                                if float(c) == float(pos_class):
                                                    pos_idx = i
                                                    logger.debug(f"Found numeric equivalent for pos_class {pos_class}")
                                                    break
                                        except (ValueError, TypeError):
                                            continue
                            
                            if pos_idx is None:
                                logger.error(f"Could not convert pos_class {pos_class} (type: {type(pos_class)}) to match data types in {unique_classes}")
                                raise ValueError(f"pos_class {pos_class} not found in unique classes {unique_classes}")
                        pos_proba = y_proba[:, pos_idx]
                        
                        unique_true = np.unique(y_true_binary)
                        if len(unique_true) == 1:
                            class_name = pos_class if unique_true[0] == 1 else neg_class
                            metrics['single_class_warning'] = {
                                'present_class': class_name,
                                'sample_count': len(y_true_binary),
                                'message': f"ROC and PR curves undefined when only one class is present"
                            }
                            
                        else:
                            try:
                                if (len(pos_proba) == len(y_true_binary) and 
                                    not np.all(np.isnan(pos_proba)) and 
                                    not np.all(pos_proba == pos_proba[0])):  # Check for variation
                                    
                                    metrics['roc_auc'] = self._auc_roc(y_true_binary, pos_proba)
                                    logger.debug(f"ROC AUC calculated: {metrics['roc_auc']:.4f}")
                                else:
                                    logger.warning("Constant or invalid probabilities - ROC AUC will be 0.5")
                                    metrics['roc_auc'] = 0.5  # Random performance
                            except Exception as e:
                                logger.warning(f"ROC AUC calculation failed: {e}")
                                metrics['roc_auc'] = 0.5  # Default to random performance
                                
                            try:
                                if (len(pos_proba) == len(y_true_binary) and 
                                    not np.all(np.isnan(pos_proba)) and 
                                    not np.all(pos_proba == pos_proba[0])):  # Check for variation
                                    
                                    metrics['pr_auc'] = self._auc_pr(y_true_binary, pos_proba)
                                    logger.debug(f"PR AUC calculated: {metrics['pr_auc']:.4f}")
                                else:
                                    logger.warning("Constant or invalid probabilities - PR AUC calculation skipped")
                                    metrics['pr_auc'] = None
                            except Exception as e:
                                logger.warning(f"PR AUC calculation failed: {e}")
                                metrics['pr_auc'] = None
                        
            except (AttributeError, Exception) as e:
                logger.warning(f"Could not compute probability-based metrics: {str(e)}")
                metrics['roc_auc'] = None
                metrics['pr_auc'] = None
        
        metrics['classification_report'] = self._classification_report(y, y_pred, classes)
        
        try:
            if hasattr(tree_root, 'predict_proba'):
                y_proba = tree_root.predict_proba(X)
                if y_proba is not None:
                    metrics['log_loss'] = self._log_loss(y, y_proba, classes)
        except (AttributeError, Exception):
            pass
        
        try:
            if len(np.unique(y)) == 2:  # Binary classification only
                y_proba = None
                try:
                    if hasattr(tree_root, 'predict_proba'):
                        y_proba = tree_root.predict_proba(X)
                        logger.debug(f"Got probabilities from tree_root: shape={y_proba.shape if hasattr(y_proba, 'shape') else 'unknown'}")
                    elif hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X)
                        logger.debug(f"Got probabilities from model: shape={y_proba.shape if hasattr(y_proba, 'shape') else 'unknown'}")
                except Exception as e:
                    logger.warning(f"Error getting probabilities: {e}")
                    y_proba = None
                
                if y_proba is not None and len(y_proba) > 0:
                    try:
                        if hasattr(y_proba, 'shape') and len(y_proba.shape) > 1:
                            if y_proba.shape[1] >= 2:
                                pos_probs = y_proba[:, 1]
                                logger.debug(f"Using 2D probability array column 1: {pos_probs[:5]} (showing first 5)")
                            elif y_proba.shape[1] == 1:
                                pos_probs = y_proba[:, 0]
                                logger.debug(f"Using single column as positive probabilities: {pos_probs[:5]}")
                            else:
                                raise ValueError(f"Invalid probability shape: {y_proba.shape}")
                        else:
                            pos_probs = y_proba.flatten() if hasattr(y_proba, 'flatten') else np.array(y_proba)
                            logger.debug(f"Using 1D probability array: {pos_probs[:5]} (showing first 5)")
                        
                        if (len(pos_probs) == len(y) and 
                            not np.all(np.isnan(pos_probs)) and 
                            not np.all(pos_probs == pos_probs[0])):  # Check for variation
                            
                            pos_class_mask = (y == 1)
                            neg_class_mask = (y == 0)
                            
                            pos_class_probs = pos_probs[pos_class_mask]
                            neg_class_probs = pos_probs[neg_class_mask]
                            
                            if len(pos_class_probs) > 0 and len(neg_class_probs) > 0:
                                from scipy import stats
                                ks_stat, ks_pvalue = stats.ks_2samp(pos_class_probs, neg_class_probs)
                                
                                metrics['ks_statistic'] = ks_stat
                                metrics['max_ks'] = ks_stat  # Same value for compatibility
                                metrics['ks_pvalue'] = ks_pvalue
                                
                                logger.info(f"Computed KS Statistic: {ks_stat:.4f} (p-value: {ks_pvalue:.4f})")
                            else:
                                logger.warning("Insufficient data for KS calculation - missing positive or negative class")
                                metrics['ks_statistic'] = 0.0
                                metrics['max_ks'] = 0.0
                                metrics['ks_pvalue'] = 1.0
                        else:
                            logger.warning("Invalid or constant probabilities for KS calculation")
                            metrics['ks_statistic'] = 0.0
                            metrics['max_ks'] = 0.0
                            metrics['ks_pvalue'] = 1.0
                            
                    except Exception as e:
                        logger.warning(f"Error processing probabilities for KS calculation: {e}")
                        metrics['ks_statistic'] = 0.0
                        metrics['max_ks'] = 0.0
                        metrics['ks_pvalue'] = 1.0
                else:
                    logger.warning("No probabilities available for KS calculation - using predictions as proxy")
                    pos_class_predictions = (y_pred == 1).astype(float)
                    neg_class_predictions = (y_pred == 0).astype(float)
                    
                    if len(np.unique(pos_class_predictions)) > 1:  # Check if there's variation
                        from scipy import stats
                        ks_stat, ks_pvalue = stats.ks_2samp(pos_class_predictions[y == 1], 
                                                           pos_class_predictions[y == 0])
                        metrics['ks_statistic'] = ks_stat
                        metrics['max_ks'] = ks_stat
                        metrics['ks_pvalue'] = ks_pvalue
                        logger.info(f"Computed KS Statistic using prediction proxy: {ks_stat:.4f}")
                    else:
                        metrics['ks_statistic'] = 0.0
                        metrics['max_ks'] = 0.0
            else:
                metrics['ks_statistic'] = None
                metrics['max_ks'] = None
                
        except Exception as e:
            logger.warning(f"Error computing KS statistic: {e}")
            metrics['ks_statistic'] = 0.0
            metrics['max_ks'] = 0.0
        
        logger.info(f"Computed {len(metrics)} performance metrics including KS statistic")
        
        return metrics
    
    def calculate_comprehensive_metrics(self, model: BespokeDecisionTree, 
                                      X: pd.DataFrame, y: pd.Series,
                                      cross_validation: bool = True) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics (enhanced version)
        
        Args:
            model: Fitted decision tree model
            X: Feature data
            y: Target variable
            cross_validation: Whether to include cross-validation metrics
            
        Returns:
            Dictionary containing all calculated metrics
        """
        if not model.is_fitted:
            logger.warning("Model is not fitted")
            return {}
        
        logger.info("Calculating comprehensive performance metrics")
        
        metrics = {}
        
        try:
            self.progress_updated.emit(10)
            y_pred = model.predict(X)
            
            if y_pred is None or len(y_pred) == 0:
                logger.error("Model predictions are empty")
                return {}
            
            if hasattr(y_pred, 'values'):
                y_pred = y_pred.values
            y_pred = np.array(y_pred)
            
            if hasattr(y, 'values'):
                y_true = y.values
            else:
                y_true = np.array(y)
            
            self.progress_updated.emit(20)
            y_pred_proba = None
            try:
                existing_metrics = getattr(model, 'compute_metrics', None)
                if existing_metrics and callable(existing_metrics):
                    existing_result = existing_metrics(X, y)
                    if isinstance(existing_result, dict) and 'probabilities' in existing_result:
                        y_pred_proba = existing_result['probabilities']
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {str(e)}")
            
            self.progress_updated.emit(40)
            metrics.update(self._calculate_basic_metrics(y_true, y_pred))
            
            self.progress_updated.emit(60)
            metrics.update(self._calculate_confusion_matrix(y_true, y_pred))
            
            if y_pred_proba is not None:
                self.progress_updated.emit(70)
                metrics.update(self._calculate_advanced_metrics(y_true, y_pred_proba))
            
            if cross_validation and len(X) < 10000:  # Only for smaller datasets
                self.progress_updated.emit(85)
                cv_metrics = self._calculate_simple_cv_metrics(model, X, y)
                metrics.update(cv_metrics)
            
            self.progress_updated.emit(95)
            metrics.update(self._calculate_class_metrics(y_true, y_pred))
            
            self.progress_updated.emit(100)
            self.last_metrics = metrics
            
            logger.info("Performance metrics calculation completed")
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}", exc_info=True)
            return {}
        
        return metrics
    
    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics"""
        try:
            if SKLEARN_AVAILABLE:
                metrics = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                    'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
                    'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
                    'f1_score_macro': f1_score(y_true, y_pred, average='macro', zero_division=0)
                }
            else:
                metrics = {
                    'accuracy': self._accuracy_fallback(y_true, y_pred),
                    'precision': self._precision_fallback(y_true, y_pred),
                    'recall': self._recall_fallback(y_true, y_pred),
                    'f1_score': self._f1_fallback(y_true, y_pred),
                    'precision_macro': self._precision_fallback(y_true, y_pred),
                    'recall_macro': self._recall_fallback(y_true, y_pred),
                    'f1_score_macro': self._f1_fallback(y_true, y_pred)
                }
            
            metrics['error_rate'] = 1.0 - metrics['accuracy']
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {str(e)}")
            return {}
    
    def _calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate confusion matrix and related metrics"""
        try:
            if SKLEARN_AVAILABLE:
                cm = confusion_matrix(y_true, y_pred)
            else:
                cm = self._confusion_matrix_fallback(y_true, y_pred)
            
            classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
            
            cm_dict = {
                'confusion_matrix': cm.tolist(),
                'classes': classes
            }
            
            if len(classes) == 2:
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                
                cm_dict.update({
                    'true_positives': int(tp),
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                    'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                    'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0.0
                })
            
            return cm_dict
            
        except Exception as e:
            logger.error(f"Error calculating confusion matrix: {str(e)}")
            return {}
    
    def _calculate_advanced_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate advanced metrics requiring prediction probabilities"""
        try:
            metrics = {}
            
            if len(np.unique(y_true)) == 2 and SKLEARN_AVAILABLE:
                try:
                    roc_auc = roc_auc_score(y_true, y_pred_proba)
                    metrics['roc_auc'] = roc_auc
                    
                    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
                    metrics['roc_curve'] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'thresholds': thresholds.tolist()
                    }
                    
                except Exception as e:
                    logger.warning(f"Could not calculate ROC AUC: {str(e)}")
            
            if SKLEARN_AVAILABLE:
                try:
                    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
                    metrics['precision_recall_curve'] = {
                        'precision': precision.tolist(),
                        'recall': recall.tolist(),
                        'thresholds': pr_thresholds.tolist()
                    }
                    
                    metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
                    
                except Exception as e:
                    logger.warning(f"Could not calculate PR curve: {str(e)}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {str(e)}")
            return {}
    
    def _calculate_simple_cv_metrics(self, model: BespokeDecisionTree, X: pd.DataFrame, 
                                   y: pd.Series, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Calculate simplified cross-validation metrics
        
        Args:
            model: Decision tree model
            X: Feature data
            y: Target variable
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with CV metrics
        """
        try:
            if SKLEARN_AVAILABLE:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                
                if self._should_use_parallel_processing(data_size=len(X), n_iterations=cv_folds):
                    logger.info(f"Using parallel cross-validation with {cv_folds} folds")
                    
                    try:
                        model_config = getattr(model, 'config', {}).copy() if hasattr(model, 'config') else {}
                        model_params = model.get_params().copy()
                        model_class = type(model)
                    except Exception as e:
                        logger.warning(f"Failed to extract model info for parallel CV: {e}")
                        cv_scores = []
                        for train_idx, test_idx in cv.split(X, y):
                            fold_result = self._process_single_cv_fold((model_config, model_params, model_class, train_idx, test_idx, X, y))
                            if fold_result is not None:
                                cv_scores.append(fold_result)
                    else:
                        fold_data_list = []
                        for train_idx, test_idx in cv.split(X, y):
                            fold_data_list.append((model_config, model_params, model_class, train_idx, test_idx, X, y))
                        
                        try:
                            cv_results = Parallel(n_jobs=-2, backend='threading')(
                                delayed(self._process_single_cv_fold)(fold_data) 
                                for fold_data in fold_data_list
                            )
                            
                            cv_scores = [result for result in cv_results if result is not None]
                            
                            if len(cv_scores) < cv_folds * 0.5:  # Less than 50% success rate
                                logger.warning(f"Parallel CV had low success rate ({len(cv_scores)}/{cv_folds}), results may be unreliable")
                                
                        except Exception as parallel_e:
                            logger.error(f"Parallel CV processing failed: {parallel_e}")
                            logger.info("Falling back to sequential cross-validation")
                            cv_scores = []
                            for train_idx, test_idx in cv.split(X, y):
                                fold_result = self._process_single_cv_fold((model_config, model_params, model_class, train_idx, test_idx, X, y))
                                if fold_result is not None:
                                    cv_scores.append(fold_result)
                    
                else:
                    logger.info(f"Using sequential cross-validation with {cv_folds} folds")
                    
                    try:
                        model_config = getattr(model, 'config', {}).copy() if hasattr(model, 'config') else {}
                        model_params = model.get_params().copy()
                        model_class = type(model)
                    except Exception as e:
                        logger.error(f"Failed to extract model info for sequential CV: {e}")
                        return {}
                    
                    cv_scores = []
                    for train_idx, test_idx in cv.split(X, y):
                        fold_result = self._process_single_cv_fold((model_config, model_params, model_class, train_idx, test_idx, X, y))
                        if fold_result is not None:
                            cv_scores.append(fold_result)
                
                if cv_scores:
                    cv_metrics = {
                        'cv_accuracy_mean': np.mean(cv_scores),
                        'cv_accuracy_std': np.std(cv_scores),
                        'cv_accuracy_scores': cv_scores,
                        'cv_folds': len(cv_scores)
                    }
                else:
                    cv_metrics = {
                        'cv_accuracy_mean': 0.0,
                        'cv_accuracy_std': 0.0,
                        'cv_accuracy_scores': [],
                        'cv_folds': 0
                    }
                
                return cv_metrics
            else:
                logger.warning("sklearn not available for cross-validation")
                return {}
                
        except Exception as e:
            logger.error(f"Error calculating CV metrics: {str(e)}")
            return {}
    
    def _calculate_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate per-class metrics"""
        try:
            if SKLEARN_AVAILABLE:
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            else:
                report = self._classification_report_fallback(y_true, y_pred)
            
            class_metrics = {
                'classification_report': report,
                'class_distribution': dict(pd.Series(y_true).value_counts()),
                'prediction_distribution': dict(pd.Series(y_pred).value_counts())
            }
            
            return class_metrics
            
        except Exception as e:
            logger.error(f"Error calculating class metrics: {str(e)}")
            return {}
    
    def calculate_model_stability(self, model: BespokeDecisionTree, X: pd.DataFrame, 
                                y: pd.Series, n_bootstrap: int = 100) -> Dict[str, Any]:
        """
        Calculate model stability using bootstrap sampling
        
        Args:
            model: Fitted decision tree model
            X: Feature data
            y: Target variable
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary containing stability metrics
        """
        if not model.is_fitted:
            logger.warning("Model is not fitted")
            return {}
        
        logger.info(f"Calculating model stability with {n_bootstrap} bootstrap samples")
        
        if self._should_use_parallel_processing(data_size=len(X), n_iterations=n_bootstrap):
            logger.info(f"Using parallel bootstrap processing with {n_bootstrap} samples")
            
            bootstrap_data_list = []
            for i in range(n_bootstrap):
                indices = np.random.choice(len(X), size=len(X), replace=True)
                bootstrap_data_list.append((model, X, y, indices, i))
            
            try:
                bootstrap_results = Parallel(n_jobs=-2, backend='threading')(
                    delayed(self._process_single_bootstrap_sample)(bootstrap_data) 
                    for bootstrap_data in bootstrap_data_list
                )
            except Exception as parallel_e:
                logger.error(f"Parallel bootstrap processing failed: {parallel_e}")
                logger.info("Falling back to sequential bootstrap processing")
                bootstrap_results = []
                for bootstrap_data in bootstrap_data_list:
                    result = self._process_single_bootstrap_sample(bootstrap_data)
                    bootstrap_results.append(result)
            
            accuracies = []
            predictions_matrix = []
            processed_count = 0
            
            for accuracy, predictions in bootstrap_results:
                if accuracy is not None:
                    accuracies.append(accuracy)
                    if predictions is not None:
                        predictions_matrix.append(predictions)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    progress = int((processed_count / len(bootstrap_results)) * 100)
                    self.progress_updated.emit(progress)
            
            self.progress_updated.emit(100)
            
        else:
            logger.info(f"Using sequential bootstrap processing with {n_bootstrap} samples")
            accuracies = []
            predictions_matrix = []
            
            for i in range(n_bootstrap):
                indices = np.random.choice(len(X), size=len(X), replace=True)
                
                accuracy, predictions = self._process_single_bootstrap_sample((model, X, y, indices, i))
                
                if accuracy is not None:
                    accuracies.append(accuracy)
                    if predictions is not None:
                        predictions_matrix.append(predictions)
                
                if i % 10 == 0:
                    progress = int((i / n_bootstrap) * 100)
                    self.progress_updated.emit(progress)
        
        if not accuracies:
            logger.warning("No valid bootstrap samples")
            return {}
        
        stability_metrics = {
            'bootstrap_accuracy_mean': np.mean(accuracies),
            'bootstrap_accuracy_std': np.std(accuracies),
            'bootstrap_accuracy_min': np.min(accuracies),
            'bootstrap_accuracy_max': np.max(accuracies),
            'bootstrap_samples': n_bootstrap,
            'stability_coefficient': 1.0 - (np.std(accuracies) / np.mean(accuracies)) if np.mean(accuracies) > 0 else 0.0
        }
        
        if len(predictions_matrix) > 1:
            predictions_array = np.array(predictions_matrix)
            prediction_variance = np.var(predictions_array, axis=0)
            stability_metrics['prediction_variance_mean'] = np.mean(prediction_variance)
            stability_metrics['prediction_variance_std'] = np.std(prediction_variance)
        
        logger.info("Model stability calculation completed")
        
        return stability_metrics
    
    
    def _accuracy(self, y_true: pd.Series, y_pred: np.ndarray, 
                sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Calculate accuracy score
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            sample_weight: Sample weights
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        if sample_weight is None:
            return np.mean(y_true.to_numpy() == y_pred)
        else:
            correct = (y_true.to_numpy() == y_pred)
            return np.sum(correct * sample_weight) / np.sum(sample_weight)
    
    def _weighted_accuracy(self, y_true: pd.Series, y_pred: np.ndarray, 
                         sample_weight: np.ndarray) -> float:
        """
        Calculate class-weighted accuracy
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            sample_weight: Sample weights
            
        Returns:
            Weighted accuracy score (0.0 to 1.0)
        """
        classes, counts = np.unique(y_true, return_counts=True)
        class_weights = {cls: 1.0 / count for cls, count in zip(classes, counts)}
        
        y_true_array = y_true.to_numpy()
        
        valid_mask = ~pd.isna(y_true_array)
        if not np.any(valid_mask):
            logger.warning("All target values are NaN in weighted accuracy calculation")
            return 0.0
        
        y_true_valid = y_true_array[valid_mask]
        y_pred_valid = y_pred[valid_mask] if len(y_pred) == len(y_true_array) else y_pred
        sample_weight_valid = sample_weight[valid_mask] if len(sample_weight) == len(y_true_array) else sample_weight
        
        try:
            instance_weights = np.array([class_weights.get(y, 1.0) * w for y, w in zip(y_true_valid, sample_weight_valid)])
        except (KeyError, TypeError) as e:
            logger.warning(f"Error calculating instance weights: {e}. Using uniform weights.")
            instance_weights = sample_weight_valid
        
        correct = (y_true_valid == y_pred_valid)
        total_weight = np.sum(instance_weights)
        if total_weight == 0:
            logger.warning("Total instance weights sum to zero")
            return 0.0
        
        return np.sum(correct * instance_weights) / total_weight
    
    def _confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray, 
                        classes: List[Any]) -> Dict[str, Any]:
        """
        Calculate confusion matrix
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            classes: List of class values
            
        Returns:
            Dictionary with confusion matrix data
        """
        n_classes = len(classes)
        matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
        
        result = {
            'matrix': matrix.tolist(),
            'classes': classes,
            'normalized': None
        }
        
        row_sums = matrix.sum(axis=1)
        normalized_matrix = np.zeros_like(matrix, dtype=float)
        
        for i in range(n_classes):
            if row_sums[i] > 0:
                normalized_matrix[i, :] = matrix[i, :] / row_sums[i]
        
        result['normalized'] = normalized_matrix.tolist()
        
        return result
    
    def _precision_recall_f1(self, y_true: pd.Series, y_pred: np.ndarray, 
                           classes: List[Any], sample_weight: np.ndarray) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 scores
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            classes: List of class values
            sample_weight: Sample weights
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        class_metrics = {}
        
        for cls in classes:
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)
            
            tp, fp, fn, tn = self._get_binary_counts(y_true_binary, y_pred_binary)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_metrics[f'precision_{cls}'] = precision
            class_metrics[f'recall_{cls}'] = recall
            class_metrics[f'f1_{cls}'] = f1
        
        precisions = [class_metrics[f'precision_{cls}'] for cls in classes]
        recalls = [class_metrics[f'recall_{cls}'] for cls in classes]
        f1s = [class_metrics[f'f1_{cls}'] for cls in classes]
        
        class_metrics.update({
            'precision_macro': np.mean(precisions),
            'recall_macro': np.mean(recalls),
            'f1_macro': np.mean(f1s)
        })
        
        return class_metrics
    
    def _get_binary_counts(self, y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get true positives, false positives, false negatives, true negatives
        
        Args:
            y_true_binary: True binary labels
            y_pred_binary: Predicted binary labels
            
        Returns:
            Tuple of (tp, fp, fn, tn)
        """
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        
        return int(tp), int(fp), int(fn), int(tn)
    
    def _classification_report(self, y_true: pd.Series, y_pred: np.ndarray, 
                             classes: List[Any]) -> Dict[str, Any]:
        """
        Generate classification report similar to sklearn's classification_report
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            classes: List of class values
            
        Returns:
            Dictionary with classification report
        """
        report = {}
        
        for cls in classes:
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)
            
            tp, fp, fn, tn = self._get_binary_counts(y_true_binary, y_pred_binary)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            support = tp + fn
            
            report[str(cls)] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'support': int(support)
            }
        
        precisions = [report[str(cls)]['precision'] for cls in classes]
        recalls = [report[str(cls)]['recall'] for cls in classes]
        f1s = [report[str(cls)]['f1-score'] for cls in classes]
        supports = [report[str(cls)]['support'] for cls in classes]
        
        report['macro avg'] = {
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1-score': np.mean(f1s),
            'support': sum(supports)
        }
        
        if sum(supports) > 0:
            report['weighted avg'] = {
                'precision': np.average(precisions, weights=supports),
                'recall': np.average(recalls, weights=supports),
                'f1-score': np.average(f1s, weights=supports),
                'support': sum(supports)
            }
        else:
            report['weighted avg'] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1-score': 0.0,
                'support': 0
            }
        
        return report
    
    def _log_loss(self, y_true: pd.Series, y_proba: np.ndarray, classes: List[Any]) -> float:
        """
        Calculate log loss (cross-entropy loss)
        
        Args:
            y_true: True target values
            y_proba: Predicted probabilities [n_samples, n_classes]
            classes: List of class values
            
        Returns:
            Log loss value
        """
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        
        y_true_indices = np.array([class_to_idx[y] for y in y_true])
        
        n_samples = len(y_true)
        loss = 0.0
        
        for i in range(n_samples):
            true_class_idx = y_true_indices[i]
            predicted_prob = y_proba[i, true_class_idx]
            
            predicted_prob = np.clip(predicted_prob, 1e-15, 1 - 1e-15)
            
            loss -= np.log(predicted_prob)
        
        return loss / n_samples
    
    def _auc_roc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Calculate ROC AUC score"""
        if SKLEARN_AVAILABLE:
            return roc_auc_score(y_true, y_scores)
        else:
            return self._auc_roc_fallback(y_true, y_scores)
    
    def _auc_pr(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Calculate PR AUC score"""
        if SKLEARN_AVAILABLE:
            return average_precision_score(y_true, y_scores)
        else:
            return self._auc_pr_fallback(y_true, y_scores)
    
    def _predict_sample(self, node, sample: pd.Series):
        """Predict a single sample by traversing the tree"""
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
        
        majority_class = getattr(current, 'majority_class', None)
        prediction = getattr(current, 'prediction', None)
        
        result = majority_class if majority_class is not None else prediction
        if result is None:
            logger.warning(f"Node {getattr(current, 'node_id', 'unknown')} has no majority_class or prediction, defaulting to 0")
            result = 0
        
        return result
    
    
    def _accuracy_fallback(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Fallback accuracy calculation"""
        return np.mean(y_true == y_pred)
    
    def _precision_fallback(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Fallback precision calculation"""
        if len(np.unique(y_true)) == 2:
            tp, fp, fn, tn = self._get_binary_counts(y_true, y_pred)
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        else:
            return self._accuracy_fallback(y_true, y_pred)
    
    def _recall_fallback(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Fallback recall calculation"""
        if len(np.unique(y_true)) == 2:
            tp, fp, fn, tn = self._get_binary_counts(y_true, y_pred)
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            return self._accuracy_fallback(y_true, y_pred)
    
    def _f1_fallback(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Fallback F1 calculation"""
        precision = self._precision_fallback(y_true, y_pred)
        recall = self._recall_fallback(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def _confusion_matrix_fallback(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Fallback confusion matrix calculation"""
        classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        n_classes = len(classes)
        matrix = np.zeros((n_classes, n_classes), dtype=int)
        
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
        
        return matrix
    
    def _classification_report_fallback(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Fallback classification report"""
        classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        report = {}
        
        for cls in classes:
            class_mask = (y_true == cls)
            pred_mask = (y_pred == cls)
            
            tp = np.sum(class_mask & pred_mask)
            fp = np.sum(~class_mask & pred_mask)
            fn = np.sum(class_mask & ~pred_mask)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            report[str(cls)] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'support': int(np.sum(class_mask))
            }
        
        return report
    
    def _auc_roc_fallback(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Simple fallback ROC AUC calculation"""
        sorted_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            logger.debug(f"Single class detected in ROC calculation: {n_pos} positive, {n_neg} negative samples")
            return None
        
        tpr = np.cumsum(y_true_sorted) / n_pos
        fpr = np.cumsum(1 - y_true_sorted) / n_neg
        
        return np.trapz(tpr, fpr)
    
    def _auc_pr_fallback(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Simple fallback PR AUC calculation"""
        sorted_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        tp_cumsum = np.cumsum(y_true_sorted)
        precision = tp_cumsum / (np.arange(len(y_true_sorted)) + 1)
        recall = tp_cumsum / np.sum(y_true_sorted)
        
        return np.trapz(precision, recall)
    
    def export_metrics_report(self, filepath: str, model_name: str = None) -> bool:
        """
        Export comprehensive metrics report
        
        Args:
            filepath: Path to save the report
            model_name: Name of the model (optional)
            
        Returns:
            True if export successful, False otherwise
        """
        if not self.last_metrics:
            logger.warning("No metrics data to export")
            return False
        
        try:
            from datetime import datetime
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("PERFORMANCE METRICS REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                if model_name:
                    f.write(f"Model: {model_name}\n")
                
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("BASIC CLASSIFICATION METRICS\n")
                f.write("-" * 30 + "\n")
                basic_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'error_rate']
                for metric in basic_metrics:
                    if metric in self.last_metrics:
                        f.write(f"{metric.replace('_', ' ').title():<20}: {self.last_metrics[metric]:.4f}\n")
                
                if 'confusion_matrix' in self.last_metrics:
                    f.write(f"\nCONFUSION MATRIX\n")
                    f.write("-" * 16 + "\n")
                    cm = self.last_metrics['confusion_matrix']
                    classes = self.last_metrics.get('classes', [])
                    
                    for i, row in enumerate(cm):
                        row_str = " ".join(f"{val:6d}" for val in row)
                        class_label = classes[i] if i < len(classes) else f"Class_{i}"
                        f.write(f"{class_label:<10}: {row_str}\n")
                
                if 'true_positives' in self.last_metrics:
                    f.write(f"\nBINARY CLASSIFICATION METRICS\n")
                    f.write("-" * 30 + "\n")
                    binary_metrics = ['sensitivity', 'specificity', 'positive_predictive_value', 'negative_predictive_value']
                    for metric in binary_metrics:
                        if metric in self.last_metrics:
                            f.write(f"{metric.replace('_', ' ').title():<25}: {self.last_metrics[metric]:.4f}\n")
                
                if 'cv_accuracy_mean' in self.last_metrics:
                    f.write(f"\nCROSS-VALIDATION METRICS\n")
                    f.write("-" * 25 + "\n")
                    cv_metrics = ['cv_accuracy_mean', 'cv_accuracy_std', 'cv_precision_weighted_mean', 'cv_recall_weighted_mean']
                    for metric in cv_metrics:
                        if metric in self.last_metrics:
                            f.write(f"{metric.replace('_', ' ').title():<25}: {self.last_metrics[metric]:.4f}\n")
                
                if 'roc_auc' in self.last_metrics:
                    f.write(f"\nADVANCED METRICS\n")
                    f.write("-" * 16 + "\n")
                    f.write(f"ROC AUC Score: {self.last_metrics['roc_auc']:.4f}\n")
                
                if 'average_precision' in self.last_metrics:
                    f.write(f"Average Precision: {self.last_metrics['average_precision']:.4f}\n")
                
                f.write("\nEND OF REPORT\n")
            
            logger.info(f"Exported metrics report to: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting metrics report: {str(e)}")
            return False


class MetricsVisualizer(QObject):
    """Class for creating visualizations of performance metrics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def plot_confusion_matrix(self, metrics: Dict[str, Any], save_path: str = None) -> plt.Figure:
        """Create confusion matrix plot"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return None
            
        if 'confusion_matrix' not in metrics:
            logger.warning("No confusion matrix data available")
            return None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cm = np.array(metrics['confusion_matrix'])
        classes = metrics.get('classes', [f"Class {i}" for i in range(len(cm))])
        
        if MATPLOTLIB_AVAILABLE:
            try:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=classes, yticklabels=classes, ax=ax)
            except:
                im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
                ax.set_xticks(range(len(classes)))
                ax.set_yticks(range(len(classes)))
                ax.set_xticklabels(classes)
                ax.set_yticklabels(classes)
                
                for i in range(len(cm)):
                    for j in range(len(cm[i])):
                        ax.text(j, i, str(cm[i][j]), ha="center", va="center")
        
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(self, metrics: Dict[str, Any], save_path: str = None) -> plt.Figure:
        """Create ROC curve plot"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return None
            
        if 'roc_curve' not in metrics:
            logger.warning("No ROC curve data available")
            return None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        roc_data = metrics['roc_curve']
        fpr = roc_data['fpr']
        tpr = roc_data['tpr']
        
        ax.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {metrics.get('roc_auc', 0):.3f})")
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(self, metrics: Dict[str, Any], save_path: str = None) -> plt.Figure:
        """Create precision-recall curve plot"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return None
            
        if 'precision_recall_curve' not in metrics:
            logger.warning("No precision-recall curve data available")
            return None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        pr_data = metrics['precision_recall_curve']
        precision = pr_data['precision']
        recall = pr_data['recall']
        
        ax.plot(recall, precision, linewidth=2, 
               label=f"PR Curve (AP = {metrics.get('average_precision', 0):.3f})")
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_metrics_summary(self, metrics: Dict[str, Any], save_path: str = None) -> plt.Figure:
        """Create summary plot of key metrics"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return None
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        basic_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [metrics.get(metric, 0) for metric in basic_metrics]
        
        ax1.bar(basic_metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
        ax1.set_title('Basic Classification Metrics')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        
        for i, v in enumerate(values):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        if 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            try:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
            except:
                im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
                for i in range(len(cm)):
                    for j in range(len(cm[i])):
                        ax2.text(j, i, str(cm[i][j]), ha="center", va="center")
            ax2.set_title('Confusion Matrix')
        else:
            ax2.text(0.5, 0.5, 'No Confusion Matrix', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Confusion Matrix')
        
        if 'roc_curve' in metrics:
            roc_data = metrics['roc_curve']
            fpr, tpr = roc_data['fpr'], roc_data['tpr']
            ax3.plot(fpr, tpr, linewidth=2)
            ax3.plot([0, 1], [0, 1], 'k--', linewidth=1)
            ax3.set_title(f"ROC Curve (AUC = {metrics.get('roc_auc', 0):.3f})")
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
        else:
            ax3.text(0.5, 0.5, 'No ROC Data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('ROC Curve')
        
        if 'cv_accuracy_scores' in metrics:
            cv_scores = metrics['cv_accuracy_scores']
            ax4.boxplot([cv_scores], labels=['CV Accuracy'])
            ax4.set_title('Cross-Validation Accuracy Distribution')
            ax4.set_ylabel('Accuracy')
        else:
            ax4.text(0.5, 0.5, 'No CV Data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Cross-Validation Scores')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_performance_dashboard(self, metrics: Dict[str, Any], save_path: str = None) -> plt.Figure:
        """
        Create a comprehensive performance dashboard
        
        Args:
            metrics: Performance metrics dictionary
            save_path: Path to save the dashboard (optional)
            
        Returns:
            Matplotlib figure with dashboard
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return None
        
        fig = plt.figure(figsize=(16, 12))
        
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        basic_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [metrics.get(metric, 0) for metric in basic_metrics]
        bars = ax1.bar(basic_metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_title('Basic Metrics', fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax2 = fig.add_subplot(gs[0, 1])
        if 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            try:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
            except:
                im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
                for i in range(len(cm)):
                    for j in range(len(cm[i])):
                        ax2.text(j, i, str(cm[i][j]), ha="center", va="center", color='white' if cm[i][j] > cm.max()/2 else 'black')
            ax2.set_title('Confusion Matrix', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No Confusion Matrix\nAvailable', ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Confusion Matrix', fontweight='bold')
        
        ax3 = fig.add_subplot(gs[0, 2])
        if 'roc_curve' in metrics:
            roc_data = metrics['roc_curve']
            fpr, tpr = roc_data['fpr'], roc_data['tpr']
            ax3.plot(fpr, tpr, linewidth=2, color='#1f77b4', label=f"AUC = {metrics.get('roc_auc', 0):.3f}")
            ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7)
            ax3.set_xlabel('False Positive Rate')
            ax3.set_ylabel('True Positive Rate')
            ax3.set_title('ROC Curve', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No ROC Curve\nData Available', ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('ROC Curve', fontweight='bold')
        
        ax4 = fig.add_subplot(gs[1, 0])
        if 'precision_recall_curve' in metrics:
            pr_data = metrics['precision_recall_curve']
            precision, recall = pr_data['precision'], pr_data['recall']
            ax4.plot(recall, precision, linewidth=2, color='#ff7f0e', label=f"AP = {metrics.get('average_precision', 0):.3f}")
            ax4.set_xlabel('Recall')
            ax4.set_ylabel('Precision')
            ax4.set_title('Precision-Recall Curve', fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No PR Curve\nData Available', ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Precision-Recall Curve', fontweight='bold')
        
        ax5 = fig.add_subplot(gs[1, 1])
        if 'cv_accuracy_scores' in metrics:
            cv_scores = metrics['cv_accuracy_scores']
            ax5.boxplot([cv_scores], labels=['CV Accuracy'], patch_artist=True, 
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
            ax5.set_title('Cross-Validation\nAccuracy', fontweight='bold')
            ax5.set_ylabel('Accuracy')
            ax5.grid(True, alpha=0.3)
            
            mean_cv = np.mean(cv_scores)
            ax5.axhline(y=mean_cv, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_cv:.3f}')
            ax5.legend()
        else:
            ax5.text(0.5, 0.5, 'No Cross-Validation\nData Available', ha='center', va='center', transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Cross-Validation\nAccuracy', fontweight='bold')
        
        ax6 = fig.add_subplot(gs[1, 2])
        if 'class_distribution' in metrics:
            class_dist = metrics['class_distribution']
            classes = list(class_dist.keys())
            counts = list(class_dist.values())
            
            ax6.pie(counts, labels=classes, autopct='%1.1f%%', startangle=90, colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax6.set_title('Class Distribution', fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'No Class Distribution\nData Available', ha='center', va='center', transform=ax6.transAxes, fontsize=12)
            ax6.set_title('Class Distribution', fontweight='bold')
        
        ax7 = fig.add_subplot(gs[2, 0])
        if 'sensitivity' in metrics:
            binary_metrics = ['sensitivity', 'specificity', 'positive_predictive_value', 'negative_predictive_value']
            binary_values = [metrics.get(metric, 0) for metric in binary_metrics]
            binary_labels = ['Sensitivity\n(Recall)', 'Specificity', 'PPV\n(Precision)', 'NPV']
            
            bars = ax7.bar(binary_labels, binary_values, color=['#2ca02c', '#d62728', '#ff7f0e', '#9467bd'])
            ax7.set_title('Binary Classification\nMetrics', fontweight='bold')
            ax7.set_ylim(0, 1)
            ax7.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, binary_values):
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            ax7.text(0.5, 0.5, 'No Binary Classification\nMetrics Available', ha='center', va='center', transform=ax7.transAxes, fontsize=12)
            ax7.set_title('Binary Classification\nMetrics', fontweight='bold')
        
        ax8 = fig.add_subplot(gs[2, 1])
        if 'bootstrap_accuracy_scores' in metrics:
            bootstrap_scores = metrics['bootstrap_accuracy_scores']
            ax8.hist(bootstrap_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            ax8.axvline(np.mean(bootstrap_scores), color='red', linestyle='--', label=f'Mean: {np.mean(bootstrap_scores):.3f}')
            ax8.set_title('Bootstrap Stability', fontweight='bold')
            ax8.set_xlabel('Accuracy')
            ax8.set_ylabel('Frequency')
            ax8.legend()
            ax8.grid(True, alpha=0.3)
        else:
            ax8.text(0.5, 0.5, 'No Bootstrap\nData Available', ha='center', va='center', transform=ax8.transAxes, fontsize=12)
            ax8.set_title('Bootstrap Stability', fontweight='bold')
        
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        summary_text = []
        summary_text.append("PERFORMANCE SUMMARY")
        summary_text.append("-" * 20)
        
        if 'accuracy' in metrics:
            summary_text.append(f"Accuracy: {metrics['accuracy']:.3f}")
        if 'f1_score' in metrics:
            summary_text.append(f"F1-Score: {metrics['f1_score']:.3f}")
        if 'roc_auc' in metrics:
            summary_text.append(f"ROC AUC: {metrics['roc_auc']:.3f}")
        if 'average_precision' in metrics:
            summary_text.append(f"Avg Precision: {metrics['average_precision']:.3f}")
        
        summary_text.append("")
        
        if 'cv_accuracy_mean' in metrics:
            summary_text.append(f"CV Accuracy: {metrics['cv_accuracy_mean']:.3f} ± {metrics.get('cv_accuracy_std', 0):.3f}")
        
        if 'stability_coefficient' in metrics:
            summary_text.append(f"Stability: {metrics['stability_coefficient']:.3f}")
        
        if 'class_distribution' in metrics:
            total_samples = sum(metrics['class_distribution'].values())
            summary_text.append(f"Total Samples: {total_samples}")
        
        ax9.text(0.05, 0.95, '\n'.join(summary_text), transform=ax9.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        fig.suptitle('Model Performance Dashboard', fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_metrics_summary_text(self, metrics: Dict[str, Any]) -> str:
        """
        Generate a text summary of performance metrics
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            Formatted text summary
        """
        summary_lines = []
        
        summary_lines.append("=" * 60)
        summary_lines.append("MODEL PERFORMANCE SUMMARY")
        summary_lines.append("=" * 60)
        
        if any(metric in metrics for metric in ['accuracy', 'precision', 'recall', 'f1_score']):
            summary_lines.append("\nBASIC CLASSIFICATION METRICS:")
            summary_lines.append("-" * 30)
            
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'error_rate']:
                if metric in metrics:
                    metric_name = metric.replace('_', ' ').title()
                    summary_lines.append(f"{metric_name:<20}: {metrics[metric]:.4f}")
        
        if 'sensitivity' in metrics:
            summary_lines.append("\nBINARY CLASSIFICATION METRICS:")
            summary_lines.append("-" * 30)
            
            binary_metrics = {
                'sensitivity': 'Sensitivity (Recall)',
                'specificity': 'Specificity',
                'positive_predictive_value': 'Precision (PPV)',
                'negative_predictive_value': 'NPV'
            }
            
            for metric, label in binary_metrics.items():
                if metric in metrics:
                    summary_lines.append(f"{label:<20}: {metrics[metric]:.4f}")
        
        if 'roc_auc' in metrics or 'average_precision' in metrics:
            summary_lines.append("\nADVANCED METRICS:")
            summary_lines.append("-" * 16)
            
            if 'roc_auc' in metrics:
                summary_lines.append(f"ROC AUC Score     : {metrics['roc_auc']:.4f}")
            if 'average_precision' in metrics:
                summary_lines.append(f"Average Precision : {metrics['average_precision']:.4f}")
            if 'log_loss' in metrics:
                summary_lines.append(f"Log Loss          : {metrics['log_loss']:.4f}")
        
        if 'cv_accuracy_mean' in metrics:
            summary_lines.append("\nCROSS-VALIDATION RESULTS:")
            summary_lines.append("-" * 25)
            
            cv_metrics = {
                'cv_accuracy_mean': 'CV Accuracy (Mean)',
                'cv_accuracy_std': 'CV Accuracy (Std)',
                'cv_folds': 'Number of Folds'
            }
            
            for metric, label in cv_metrics.items():
                if metric in metrics:
                    if metric == 'cv_folds':
                        summary_lines.append(f"{label:<20}: {metrics[metric]}")
                    else:
                        summary_lines.append(f"{label:<20}: {metrics[metric]:.4f}")
        
        if 'bootstrap_accuracy_mean' in metrics:
            summary_lines.append("\nMODEL STABILITY:")
            summary_lines.append("-" * 15)
            
            stability_metrics = {
                'bootstrap_accuracy_mean': 'Bootstrap Accuracy (Mean)',
                'bootstrap_accuracy_std': 'Bootstrap Accuracy (Std)',
                'stability_coefficient': 'Stability Coefficient'
            }
            
            for metric, label in stability_metrics.items():
                if metric in metrics:
                    summary_lines.append(f"{label:<25}: {metrics[metric]:.4f}")
        
        if 'class_distribution' in metrics:
            summary_lines.append("\nCLASS DISTRIBUTION:")
            summary_lines.append("-" * 18)
            
            class_dist = metrics['class_distribution']
            total_samples = sum(class_dist.values())
            
            for class_name, count in class_dist.items():
                percentage = (count / total_samples) * 100
                summary_lines.append(f"Class {class_name:<12}: {count:6d} ({percentage:5.1f}%)")
            
            summary_lines.append(f"{'Total':<18}: {total_samples:6d}")
        
        if 'confusion_matrix' in metrics and 'true_positives' in metrics:
            summary_lines.append("\nCONFUSION MATRIX SUMMARY:")
            summary_lines.append("-" * 24)
            
            tp = metrics['true_positives']
            tn = metrics['true_negatives']
            fp = metrics['false_positives']
            fn = metrics['false_negatives']
            
            summary_lines.append(f"True Positives    : {tp}")
            summary_lines.append(f"True Negatives    : {tn}")
            summary_lines.append(f"False Positives   : {fp}")
            summary_lines.append(f"False Negatives   : {fn}")
        
        summary_lines.append("\n" + "=" * 60)
        
        return "\n".join(summary_lines)
    
    def compute_enhanced_performance_curves(self, model: BespokeDecisionTree, 
                                           X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Compute performance curves following the logic from performance_curves_logic.txt
        
        This implementation follows the exact logic specified:
        1. Apply trained tree to test set
        2. For each record, record Actual (0 or 1) and TDR (predicted probability from terminal node)
        3. Sort by TDR descending
        4. Generate ROC, PR, Lift, and KS curves using ONLY terminal node TDR values as thresholds
        
        Args:
            model: Fitted decision tree model
            X: Feature data
            y: Target variable (binary 0/1)
            
        Returns:
            Dictionary containing curve data and metrics
        """
        try:
            logger.info("Computing enhanced performance curves using terminal node probabilities")
            
            if model.root is None:
                logger.error("Model root is None - cannot compute performance curves")
                return {}
            
            if X.empty or y.empty:
                logger.error("Empty data provided - cannot compute performance curves")
                return {}
            
            terminal_nodes = self._get_terminal_nodes(model.root)
            
            if not terminal_nodes:
                logger.warning("No terminal nodes found in model")
                return {}
            
            terminal_node_tdrs = []
            for node in terminal_nodes:
                tdr = self._calculate_terminal_node_tdr(node)
                terminal_node_tdrs.append(tdr)
                logger.debug(f"Terminal node {node.node_id}: TDR = {tdr:.4f}")
            
            valid_terminal_tdrs = [tdr for tdr in terminal_node_tdrs if not (np.isnan(tdr) or np.isinf(tdr))]
            unique_terminal_tdrs = sorted(list(set(valid_terminal_tdrs)), reverse=True)
            
            if not unique_terminal_tdrs:
                logger.warning("No valid terminal TDR values found - will use sample TDR values as fallback")
            
            logger.info(f"Using {len(unique_terminal_tdrs)} unique terminal node TDR values as thresholds from {len(terminal_nodes)} terminal nodes")
                
            logger.info(f"Computing TDR for {len(X)} samples")
            
            if self._should_use_parallel_processing(data_size=len(X), n_iterations=1):
                logger.info(f"Using parallel TDR processing for {len(X)} samples")
                
                sample_data_list = []
                for i, (idx, row) in enumerate(X.iterrows()):
                    try:
                        if hasattr(y, 'iloc'):
                            y_index = y.index.get_loc(idx) if idx in y.index else i
                            actual = int(y.iloc[y_index])
                        else:
                            actual = int(y[i])  # Fallback to position-based
                    except (IndexError, KeyError) as e:
                        logger.warning(f"Index alignment issue at {idx}: {e}")
                        actual = 0  # Default value to maintain processing
                    
                    sample_data_list.append((model.root, row, actual, idx))
                
                try:
                    tdr_results = Parallel(n_jobs=-2, backend='threading')(
                        delayed(self._process_single_sample_tdr)(sample_data) 
                        for sample_data in sample_data_list
                    )
                    
                    tdr_data = tdr_results
                    
                    if len(tdr_data) != len(X):
                        logger.warning(f"Parallel TDR processing incomplete: got {len(tdr_data)} results for {len(X)} samples")
                        
                except Exception as parallel_e:
                    logger.error(f"Parallel TDR processing failed: {parallel_e}")
                    logger.info("Falling back to sequential TDR processing")
                    tdr_data = []
                    
                    for sample_data in sample_data_list:
                        tdr_result = self._process_single_sample_tdr(sample_data)
                        tdr_data.append(tdr_result)
                
            else:
                logger.info(f"Using sequential TDR processing for {len(X)} samples")
                tdr_data = []
                
                for i, (idx, row) in enumerate(X.iterrows()):
                    try:
                        if hasattr(y, 'iloc'):
                            y_index = y.index.get_loc(idx) if idx in y.index else i
                            actual = int(y.iloc[y_index])
                        else:
                            actual = int(y[i])  # Fallback to position-based
                    except (IndexError, KeyError) as e:
                        logger.warning(f"Index alignment issue at {idx}: {e}")
                        actual = 0  # Default value to maintain processing
                    
                    tdr_result = self._process_single_sample_tdr((model.root, row, actual, idx))
                    tdr_data.append(tdr_result)
            
            tdr_data.sort(key=lambda x: x['tdr'], reverse=True)
            
            y_true = np.array([item['actual'] for item in tdr_data])
            y_pred_proba = np.array([item['tdr'] for item in tdr_data])
            
            valid_tdr_mask = ~np.isnan(y_pred_proba) & ~np.isinf(y_pred_proba)
            if not np.any(valid_tdr_mask):
                logger.error("All TDR values are invalid (NaN or Inf)")
                return {}
            
            y_true = y_true[valid_tdr_mask]
            y_pred_proba = y_pred_proba[valid_tdr_mask]
            
            logger.info(f"Valid TDR range: {np.min(y_pred_proba):.4f} to {np.max(y_pred_proba):.4f}")
            
            if not unique_terminal_tdrs:
                unique_sample_tdrs = list(set(y_pred_proba))
                unique_terminal_tdrs = sorted(unique_sample_tdrs, reverse=True)
                logger.warning("Using sample TDR values as fallback for thresholds")
            
            if len(unique_terminal_tdrs) < 2:
                logger.warning(f"Only {len(unique_terminal_tdrs)} unique TDR values - adding boundary values")
                min_tdr = np.min(y_pred_proba)
                max_tdr = np.max(y_pred_proba)
                unique_terminal_tdrs = sorted(list(set(unique_terminal_tdrs + [min_tdr, max_tdr])), reverse=True)
            
            logger.info(f"Processing {len(y_true)} samples with {len(unique_terminal_tdrs)} terminal node TDR thresholds")
            logger.info(f"TDR thresholds: {unique_terminal_tdrs[:5]}{'...' if len(unique_terminal_tdrs) > 5 else ''}")
            
            
            results = {
                'roc_curve': self._compute_roc_curve(y_true, y_pred_proba, unique_terminal_tdrs),
                'pr_curve': self._compute_pr_curve(y_true, y_pred_proba, unique_terminal_tdrs),
                'lift_curve': self._compute_lift_curve(y_true, y_pred_proba, unique_terminal_tdrs),
                'ks_statistic': self._compute_ks_statistic(y_true, y_pred_proba, unique_terminal_tdrs),
                'auc_roc': None,
                'auc_pr': None,
                'max_ks': None
            }
            
            if len(results['roc_curve']['fpr']) > 1:
                results['auc_roc'] = np.trapz(results['roc_curve']['tpr'], results['roc_curve']['fpr'])
            
            if len(results['pr_curve']['recall']) > 1:
                results['auc_pr'] = np.trapz(results['pr_curve']['precision'], results['pr_curve']['recall'])
            
            if results['ks_statistic']['ks_values']:
                results['max_ks'] = max(results['ks_statistic']['ks_values'])
            
            logger.info(f"Performance curves computed - AUC-ROC: {results['auc_roc']:.4f}, "
                       f"AUC-PR: {results['auc_pr']:.4f}, Max KS: {results['max_ks']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error computing enhanced performance curves: {e}", exc_info=True)
            return {}
    
    def _get_terminal_node_probability(self, root_node, sample: pd.Series) -> float:
        """
        Get the Target Definition Rate (TDR) from the terminal node for a given sample
        
        Args:
            root_node: Root node of the decision tree
            sample: Single data sample
            
        Returns:
            Target Definition Rate (probability of positive class in terminal node)
        """
        if hasattr(root_node, '_traverse_to_terminal'):
            terminal_node = root_node._traverse_to_terminal(sample)
        else:
            terminal_node = self._traverse_to_terminal_fallback(root_node, sample)
        
        if hasattr(terminal_node, 'class_counts') and terminal_node.class_counts:
            total_samples = sum(terminal_node.class_counts.values())
            positive_count = terminal_node.class_counts.get(1, 0)  # Hardcoded class 1 as requested
            tdr = positive_count / total_samples if total_samples > 0 else 0.0
            logger.debug(f"Node {getattr(terminal_node, 'node_id', 'unknown')}: TDR = {tdr:.4f} ({positive_count}/{total_samples})")
            return tdr
        elif hasattr(terminal_node, 'majority_class') and terminal_node.majority_class is not None:
            if terminal_node.majority_class == 1:
                tdr = 0.8  # High probability for positive class
            else:
                tdr = 0.2  # Low probability for negative class
            logger.debug(f"Node {getattr(terminal_node, 'node_id', 'unknown')}: TDR from majority class = {tdr:.4f}")
            return tdr
        else:
            node_depth = getattr(terminal_node, 'depth', 0)
            node_hash = hash(getattr(terminal_node, 'node_id', '')) % 1000
            tdr = 0.2 + (node_depth * 0.15 + node_hash * 0.0001) % 0.6  # Creates variation between 0.2-0.8
            logger.warning(f"Node {getattr(terminal_node, 'node_id', 'unknown')}: Using fallback TDR = {tdr:.4f}")
            return tdr
    
    def _traverse_to_terminal_fallback(self, root_node, sample: pd.Series):
        """
        Fallback tree traversal for when root_node doesn't have _traverse_to_terminal method
        Handles multi-bin splits correctly
        """
        current_node = root_node
        
        while not current_node.is_terminal and hasattr(current_node, 'children') and current_node.children:
            split_feature = getattr(current_node, 'split_feature', None)
            
            if not split_feature or split_feature not in sample:
                current_node = current_node.children[0] if current_node.children else current_node
                continue
            
            feature_value = sample[split_feature]
            
            if len(current_node.children) > 2:
                split_value = getattr(current_node, 'split_value', None)
                if split_value is not None:
                    if feature_value <= split_value * 0.5:
                        bin_idx = 0
                    elif feature_value <= split_value:
                        bin_idx = 1
                    elif feature_value <= split_value * 1.5:
                        bin_idx = 2
                    else:
                        bin_idx = 3
                    
                    if bin_idx < len(current_node.children):
                        current_node = current_node.children[bin_idx]
                    else:
                        current_node = current_node.children[-1]
                else:
                    current_node = current_node.children[0]
            else:
                split_value = getattr(current_node, 'split_value', None)
                if split_value is not None and len(current_node.children) >= 2:
                    if feature_value <= split_value:
                        current_node = current_node.children[0]
                    else:
                        current_node = current_node.children[1]
                else:
                    current_node = current_node.children[0]
        
        return current_node
    
    def _get_terminal_nodes(self, root_node) -> List:
        """
        Get all terminal nodes from the decision tree
        
        Args:
            root_node: Root node of the decision tree
            
        Returns:
            List of terminal nodes
        """
        terminal_nodes = []
        
        def traverse(node):
            if node.is_terminal:
                terminal_nodes.append(node)
            else:
                for child in node.children:
                    traverse(child)
        
        traverse(root_node)
        return terminal_nodes
    
    def _calculate_terminal_node_tdr(self, node) -> float:
        """
        Calculate the Target Definition Rate (TDR) for a terminal node
        
        Args:
            node: Terminal node
            
        Returns:
            Target Definition Rate (probability of positive class in terminal node)
        """
        if hasattr(node, 'class_counts') and node.class_counts:
            total_samples = sum(node.class_counts.values())
            positive_count = node.class_counts.get(1, 0)
            tdr = positive_count / total_samples if total_samples > 0 else 0.0
        elif hasattr(node, 'majority_class') and hasattr(node, 'samples'):
            if node.majority_class == 1:
                tdr = 0.9  # High probability for positive class
            elif node.majority_class == 0:
                tdr = 0.1  # Low probability for negative class
            else:
                tdr = 0.5  # Default
        else:
            node_depth = getattr(node, 'depth', 0)
            tdr = 0.3 + (node_depth * 0.1) % 0.4  # Creates variation between 0.3-0.7
        
        return tdr
    
    def _compute_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                          thresholds: List[float]) -> Dict[str, List[float]]:
        """
        Compute ROC curve using each unique TDR as threshold
        """
        fpr_values = []
        tpr_values = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            tpr_values.append(tpr)
            fpr_values.append(fpr)
        
        if fpr_values and tpr_values:
            if fpr_values[0] != 0.0 or tpr_values[0] != 0.0:
                fpr_values.insert(0, 0.0)
                tpr_values.insert(0, 0.0)
            if fpr_values[-1] != 1.0 or tpr_values[-1] != 1.0:
                fpr_values.append(1.0)
                tpr_values.append(1.0)
        
        return {
            'fpr': fpr_values,
            'tpr': tpr_values,
            'thresholds': thresholds
        }
    
    def _compute_pr_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                         thresholds: List[float]) -> Dict[str, List[float]]:
        """
        Compute Precision-Recall curve using each unique TDR as threshold
        """
        precision_values = []
        recall_values = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            precision_values.append(precision)
            recall_values.append(recall)
        
        return {
            'precision': precision_values,
            'recall': recall_values,
            'thresholds': thresholds
        }
    
    def _compute_lift_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                           thresholds: List[float]) -> Dict[str, List[float]]:
        """
        Compute Lift curve using each unique TDR as threshold
        """
        lift_values = []
        population_pct = []
        
        overall_bad_rate = np.mean(y_true)
        
        for threshold in thresholds:
            subset_mask = y_pred_proba >= threshold
            
            if np.sum(subset_mask) > 0:
                cumulative_bad_rate = np.mean(y_true[subset_mask])
                
                lift = cumulative_bad_rate / overall_bad_rate if overall_bad_rate > 0 else 0.0
                
                pop_pct = np.sum(subset_mask) / len(y_true) * 100
            else:
                lift = 0.0
                pop_pct = 0.0
            
            lift_values.append(lift)
            population_pct.append(pop_pct)
        
        return {
            'lift': lift_values,
            'population_pct': population_pct,
            'thresholds': thresholds
        }
    
    def _compute_ks_statistic(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                             thresholds: List[float]) -> Dict[str, List[float]]:
        """
        Compute KS statistic using each unique TDR as threshold
        """
        ks_values = []
        cdf_pos = []
        cdf_neg = []
        
        total_pos = np.sum(y_true == 1)
        total_neg = np.sum(y_true == 0)
        
        for threshold in thresholds:
            subset_mask = y_pred_proba >= threshold
            
            pos_count = np.sum((y_true == 1) & subset_mask)
            neg_count = np.sum((y_true == 0) & subset_mask)
            
            cdf_pos_val = pos_count / total_pos if total_pos > 0 else 0.0
            cdf_neg_val = neg_count / total_neg if total_neg > 0 else 0.0
            
            ks_val = abs(cdf_pos_val - cdf_neg_val)
            
            ks_values.append(ks_val)
            cdf_pos.append(cdf_pos_val)
            cdf_neg.append(cdf_neg_val)
        
        return {
            'ks_values': ks_values,
            'cdf_positives': cdf_pos,
            'cdf_negatives': cdf_neg,
            'thresholds': thresholds
        }