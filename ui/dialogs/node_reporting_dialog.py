#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Node Reporting Dialog for Bespoke Utility
Comprehensive reporting and analysis for individual decision tree nodes
COMPLETE FILE REPLACEMENT - Includes all original functionality plus enhancements

"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime
import json
import csv
from io import StringIO

from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QTabWidget, QWidget, QListWidget, QListWidgetItem,
    QSplitter, QProgressBar, QMessageBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QFrame, QSizePolicy, QTreeWidget, QTreeWidgetItem, QFileDialog,
    QApplication
)

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from models.node import TreeNode
from utils.metrics_calculator import CentralMetricsCalculator, MetricsValidator

logger = logging.getLogger(__name__)


class NodeStatisticsCalculator:
    """Calculator for comprehensive node statistics"""
    
    def __init__(self, node: TreeNode, node_data: pd.DataFrame, target_column: str):
        self.node = node
        self.node_data = node_data
        self.target_column = target_column
        
    def calculate_comprehensive_stats(self) -> Dict[str, Any]:
        """Calculate comprehensive statistics for the node"""
        stats = {
            'basic_info': self._calculate_basic_info(),
            'data_distribution': self._calculate_data_distribution(),
            'target_analysis': self._calculate_target_analysis(),
            'feature_importance': self._calculate_feature_importance(),
            'split_analysis': self._calculate_split_analysis(),
            'purity_metrics': self._calculate_purity_metrics(),
            'performance_metrics': self._calculate_performance_metrics(),
            'comparison_metrics': self._calculate_comparison_metrics(),
            'advanced_analysis': self._calculate_advanced_analysis()
        }
        
        return stats
        
    def _calculate_basic_info(self) -> Dict[str, Any]:
        """Calculate basic node information"""
        return {
            'node_id': getattr(self.node, 'node_id', 'unknown'),
            'depth': getattr(self.node, 'depth', 0),
            'is_terminal': getattr(self.node, 'is_terminal', True),
            'samples': getattr(self.node, 'samples', 0),
            'prediction': getattr(self.node, 'majority_class', None) or getattr(self.node, 'prediction', None),
            'split_feature': getattr(self.node, 'split_feature', None) if not getattr(self.node, 'is_terminal', True) else None,
            'split_value': getattr(self.node, 'split_value', None) if not getattr(self.node, 'is_terminal', True) else None,
            'split_operator': getattr(self.node, 'split_operator', '<='),
            'children_count': len(getattr(self.node, 'children', [])),
            'parent_id': getattr(self.node.parent, 'node_id', None) if getattr(self.node, 'parent', None) else None,
            'impurity': getattr(self.node, 'impurity', 0.0),
            'confidence': getattr(self.node, 'confidence', None),
            'class_counts': getattr(self.node, 'class_counts', {}),
            'data_samples': len(self.node_data) if self.node_data is not None else 0
        }
        
    def _calculate_data_distribution(self) -> Dict[str, Any]:
        """Calculate data distribution for all features in the node"""
        if self.node_data is None or len(self.node_data) == 0:
            return {'error': 'No data available for this node'}
        
        distribution = {
            'total_features': len(self.node_data.columns),
            'numeric_features': 0,
            'categorical_features': 0,
            'feature_statistics': {},
            'missing_values': {},
            'data_quality': {}
        }
        
        for column in self.node_data.columns:
            if column == self.target_column:
                continue
                
            col_data = self.node_data[column]
            
            if pd.api.types.is_numeric_dtype(col_data):
                distribution['numeric_features'] += 1
                stats = col_data.describe()
                distribution['feature_statistics'][column] = {
                    'type': 'numeric',
                    'count': int(stats['count']),
                    'mean': round(float(stats['mean']), 4),
                    'std': round(float(stats['std']), 4),
                    'min': round(float(stats['min']), 4),
                    'max': round(float(stats['max']), 4),
                    'median': round(float(stats['50%']), 4),
                    'q1': round(float(stats['25%']), 4),
                    'q3': round(float(stats['75%']), 4),
                    'skewness': self._calculate_skewness(col_data),
                    'outliers': self._detect_outliers(col_data)
                }
            else:
                distribution['categorical_features'] += 1
                value_counts = col_data.value_counts()
                distribution['feature_statistics'][column] = {
                    'type': 'categorical',
                    'unique_values': len(value_counts),
                    'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'top_5_values': dict(value_counts.head().items()) if len(value_counts) > 0 else {},
                    'entropy': self._calculate_entropy(col_data)
                }
            
            missing_count = int(col_data.isna().sum())
            missing_percentage = round((missing_count / len(col_data)) * 100, 2)
            distribution['missing_values'][column] = {
                'count': missing_count,
                'percentage': missing_percentage
            }
            
            distribution['data_quality'][column] = self._assess_data_quality(col_data)
        
        if self.node_data.size > 0:
            total_missing = self.node_data.isnull().sum().sum()
            distribution['missing_data_percentage'] = round((total_missing / self.node_data.size) * 100, 2)
        else:
            distribution['missing_data_percentage'] = 0.0
                
        return distribution
        
    def _calculate_target_analysis(self) -> Dict[str, Any]:
        """Calculate target variable analysis"""
        if self.node_data is None or len(self.node_data) == 0:
            return {'error': 'No data available for this node'}
            
        if self.target_column not in self.node_data.columns:
            return {'error': f'Target column {self.target_column} not found'}
            
        target_data = self.node_data[self.target_column]
        
        analysis = {
            'target_column': self.target_column,
            'data_type': str(target_data.dtype),
            'missing_values': int(target_data.isna().sum()),
            'missing_percentage': round((target_data.isna().sum() / len(target_data)) * 100, 2),
            'unique_values': target_data.nunique(),
            'sample_size': len(target_data)
        }
        
        if pd.api.types.is_numeric_dtype(target_data):
            stats = target_data.describe()
            analysis.update({
                'type': 'numeric',
                'mean': round(float(stats['mean']), 4),
                'std': round(float(stats['std']), 4),
                'min': round(float(stats['min']), 4),
                'max': round(float(stats['max']), 4),
                'median': round(float(stats['50%']), 4),
                'range': round(float(stats['max'] - stats['min']), 4),
                'skewness': self._calculate_skewness(target_data),
                'outliers': self._detect_outliers(target_data),
                'variance': round(float(target_data.var()), 4)
            })
        else:
            value_counts = target_data.value_counts()
            total_count = len(target_data.dropna())
            
            analysis.update({
                'type': 'categorical',
                'unique_classes': len(value_counts),
                'class_distribution': {},
                'majority_class': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'majority_percentage': round((value_counts.iloc[0] / total_count) * 100, 2) if len(value_counts) > 0 else 0,
                'gini_impurity': self._calculate_gini_impurity(target_data),
                'entropy': self._calculate_entropy(target_data)
            })
            
            for class_val, count in value_counts.items():
                analysis['class_distribution'][str(class_val)] = {
                    'count': int(count),
                    'percentage': round((count / total_count) * 100, 2)
                }
                
        return analysis
        
    def _calculate_feature_importance(self) -> Dict[str, Any]:
        """Calculate feature importance within this node"""
        if self.node_data is None or len(self.node_data) == 0:
            return {'error': 'No data available for this node'}
            
        numeric_features = self.node_data.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col != self.target_column]
        
        if len(numeric_features) == 0:
            return {'error': 'No numeric features available for importance calculation'}
        
        try:
            if self.target_column in self.node_data.columns:
                target_data = self.node_data[self.target_column]
                
                if pd.api.types.is_numeric_dtype(target_data):
                    correlations = {}
                    
                    if target_data.std() > 1e-10:
                        import warnings
                        
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", "invalid value encountered")
                            
                            for feature in numeric_features:
                                if self.node_data[feature].std() > 1e-10:
                                    corr = self.node_data[feature].corr(target_data)
                                    if pd.notna(corr) and np.isfinite(corr):
                                        correlations[feature] = abs(corr)
                    else:
                        logger.debug("Target variable has insufficient variance for correlation analysis")
                    
                    sorted_correlations = dict(sorted(correlations.items(), 
                                                    key=lambda x: x[1], reverse=True))
                    
                    return {
                        'method': 'correlation',
                        'feature_importance': sorted_correlations,
                        'top_5_features': dict(list(sorted_correlations.items())[:5]),
                        'total_features_analyzed': len(sorted_correlations)
                    }
                else:
                    variances = {}
                    mutual_info = {}
                    
                    for feature in numeric_features:
                        variance = self.node_data[feature].var()
                        if not pd.isna(variance):
                            variances[feature] = variance
                            
                        mutual_info[feature] = self._calculate_mutual_info_proxy(
                            self.node_data[feature], target_data)
                    
                    max_variance = max(variances.values()) if variances else 1
                    normalized_variances = {k: v/max_variance for k, v in variances.items()}
                    
                    combined_importance = {}
                    for feature in numeric_features:
                        var_score = normalized_variances.get(feature, 0)
                        mi_score = mutual_info.get(feature, 0)
                        combined_importance[feature] = (var_score + mi_score) / 2
                    
                    sorted_importance = dict(sorted(combined_importance.items(), 
                                                 key=lambda x: x[1], reverse=True))
                    
                    return {
                        'method': 'variance_mutual_info',
                        'feature_importance': sorted_importance,
                        'top_5_features': dict(list(sorted_importance.items())[:5]),
                        'total_features_analyzed': len(sorted_importance)
                    }
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {'error': f'Error calculating feature importance: {str(e)}'}
        
        return {'error': 'Could not calculate feature importance'}
        
    def _calculate_split_analysis(self) -> Dict[str, Any]:
        """Calculate split analysis for non-terminal nodes"""
        if getattr(self.node, 'is_terminal', True):
            return {'info': 'Node is terminal - no split analysis available'}
        
        split_feature = getattr(self.node, 'split_feature', None)
        split_value = getattr(self.node, 'split_value', None)
        
        if not split_feature or split_value is None:
            return {'error': 'Split information not available'}
        
        analysis = {
            'feature': split_feature,
            'value': split_value,
            'operator': getattr(self.node, 'split_operator', '<='),
            'children_count': len(getattr(self.node, 'children', [])),
            'improvement': getattr(self.node, 'improvement', 0.0)
        }
        
        if self.node_data is not None and split_feature in self.node_data.columns:
            feature_data = self.node_data[split_feature]
            
            if pd.api.types.is_numeric_dtype(feature_data):
                left_condition = feature_data <= split_value
                right_condition = feature_data > split_value
                
                left_samples = int(left_condition.sum())
                right_samples = int(right_condition.sum())
                total_samples = len(feature_data)
                
                analysis.update({
                    'left_samples': left_samples,
                    'right_samples': right_samples,
                    'left_percentage': round((left_samples / total_samples) * 100, 2),
                    'right_percentage': round((right_samples / total_samples) * 100, 2),
                    'balance_score': min(left_samples, right_samples) / max(left_samples, right_samples) if max(left_samples, right_samples) > 0 else 0
                })
                
                if self.target_column in self.node_data.columns:
                    target_data = self.node_data[self.target_column]
                    
                    left_target = target_data[left_condition]
                    right_target = target_data[right_condition]
                    
                    if pd.api.types.is_categorical_dtype(target_data) or target_data.dtype == 'object':
                        left_dist = left_target.value_counts(normalize=True).to_dict()
                        right_dist = right_target.value_counts(normalize=True).to_dict()
                        
                        analysis.update({
                            'left_target_distribution': left_dist,
                            'right_target_distribution': right_dist,
                            'left_purity': self._calculate_purity_score(left_target),
                            'right_purity': self._calculate_purity_score(right_target)
                        })
                    else:
                        analysis.update({
                            'left_target_mean': round(float(left_target.mean()), 4) if len(left_target) > 0 else 0,
                            'right_target_mean': round(float(right_target.mean()), 4) if len(right_target) > 0 else 0,
                            'left_target_std': round(float(left_target.std()), 4) if len(left_target) > 0 else 0,
                            'right_target_std': round(float(right_target.std()), 4) if len(right_target) > 0 else 0,
                            'variance_reduction': self._calculate_variance_reduction(target_data, left_target, right_target)
                        })
                        
                analysis['split_feature_analysis'] = self._analyze_split_feature_distributions(
                    feature_data, left_condition, right_condition)
        
        return analysis
        
    def _calculate_purity_metrics(self) -> Dict[str, Any]:
        """Calculate node purity metrics"""
        purity = {
            'impurity': getattr(self.node, 'impurity', 0.0) or 0.0,
            'samples': getattr(self.node, 'samples', 0) or 0,
            'type': 'unknown'
        }
        
        class_counts = getattr(self.node, 'class_counts', {})
        if class_counts and isinstance(class_counts, dict):
            total_samples = sum(class_counts.values()) if class_counts.values() else 0
            
            if total_samples > 0:
                gini = 1.0 - sum((count/total_samples)**2 for count in class_counts.values())
                
                entropy = 0.0
                for count in class_counts.values():
                    if count > 0:
                        prob = count / total_samples
                        entropy -= prob * np.log2(prob)
                
                dominant_class = max(class_counts, key=class_counts.get) if class_counts else None
                dominant_count = max(class_counts.values()) if class_counts else 0
                
                purity.update({
                    'type': 'categorical',
                    'gini_impurity': round(gini, 4),
                    'entropy': round(entropy, 4),
                    'class_distribution': class_counts,
                    'dominant_class': dominant_class,
                    'dominant_class_percentage': round((dominant_count / total_samples) * 100, 2) if total_samples > 0 else 0,
                    'num_classes': len(class_counts),
                    'diversity_index': self._calculate_diversity_index(class_counts)
                })
        else:
            if self.node_data is not None and self.target_column in self.node_data.columns:
                target_data = self.node_data[self.target_column]
                if pd.api.types.is_numeric_dtype(target_data):
                    variance = float(target_data.var()) if len(target_data) > 1 else 0.0
                    std = float(target_data.std()) if len(target_data) > 1 else 0.0
                    
                    purity.update({
                        'type': 'numeric',
                        'variance': round(variance, 4),
                        'std': round(std, 4),
                        'coefficient_of_variation': round(std / target_data.mean(), 4) if target_data.mean() != 0 else 0
                    })
        
        return purity
        
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics for terminal nodes"""
        if not getattr(self.node, 'is_terminal', True):
            return {'info': 'Performance metrics only available for terminal nodes'}
        
        prediction = getattr(self.node, 'majority_class', None) or getattr(self.node, 'prediction', None)
        
        metrics = {
            'prediction': prediction,
            'confidence': getattr(self.node, 'confidence', None),
            'samples': getattr(self.node, 'samples', 0) or 0,
            'type': 'terminal'
        }
        
        if (self.node_data is not None and len(self.node_data) > 0 and 
            self.target_column in self.node_data.columns and prediction is not None):
            
            actual_values = self.node_data[self.target_column]
            
            if pd.api.types.is_categorical_dtype(actual_values) or actual_values.dtype == 'object':
                correct_predictions = (actual_values == prediction).sum()
                total_predictions = len(actual_values)
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                
                class_dist = actual_values.value_counts().to_dict()
                
                metrics.update({
                    'type': 'classification',
                    'accuracy': round(float(accuracy), 4),
                    'correct_predictions': int(correct_predictions),
                    'incorrect_predictions': int(total_predictions - correct_predictions),
                    'class_distribution': class_dist,
                    'error_rate': round(1.0 - accuracy, 4),
                    'prediction_confidence_score': self._calculate_prediction_confidence(actual_values, prediction)
                })
            else:
                predictions_array = np.full(len(actual_values), prediction)
                mse = np.mean((actual_values - predictions_array) ** 2)
                mae = np.mean(np.abs(actual_values - predictions_array))
                rmse = np.sqrt(mse)
                
                metrics.update({
                    'type': 'regression',
                    'mse': round(float(mse), 4),
                    'mae': round(float(mae), 4),
                    'rmse': round(float(rmse), 4),
                    'target_mean': round(float(actual_values.mean()), 4),
                    'target_std': round(float(actual_values.std()), 4)
                })
        
        return metrics
        
    def _calculate_comparison_metrics(self) -> Dict[str, Any]:
        """Calculate metrics comparing this node to its parent and siblings"""
        comparison = {
            'type': 'comparison',
            'has_parent': False,
            'has_siblings': False
        }
        
        parent = getattr(self.node, 'parent', None)
        if parent:
            comparison['has_parent'] = True
            
            parent_samples = getattr(parent, 'samples', 1) or 1
            node_samples = getattr(self.node, 'samples', 0) or 0
            comparison['sample_ratio_to_parent'] = round(node_samples / parent_samples, 4)
            
            node_impurity = getattr(self.node, 'impurity', None)
            parent_impurity = getattr(parent, 'impurity', None)
            
            if node_impurity is not None and parent_impurity is not None:
                comparison['impurity_change'] = round(node_impurity - parent_impurity, 4)
                comparison['impurity_reduction'] = round(parent_impurity - node_impurity, 4)
                comparison['relative_impurity_reduction'] = round(
                    (parent_impurity - node_impurity) / parent_impurity, 4) if parent_impurity > 0 else 0
            
            siblings = [child for child in getattr(parent, 'children', []) 
                       if getattr(child, 'node_id', None) != getattr(self.node, 'node_id', None)]
            
            if siblings:
                comparison['has_siblings'] = True
                comparison['siblings_count'] = len(siblings)
                
                sibling_samples = [getattr(sibling, 'samples', 0) or 0 for sibling in siblings]
                sibling_impurities = [getattr(sibling, 'impurity', None) for sibling in siblings 
                                    if getattr(sibling, 'impurity', None) is not None]
                
                if sibling_samples:
                    comparison['avg_sibling_samples'] = round(np.mean(sibling_samples), 2)
                    comparison['max_sibling_samples'] = max(sibling_samples)
                    comparison['min_sibling_samples'] = min(sibling_samples)
                    comparison['sample_rank_among_siblings'] = sorted(sibling_samples + [node_samples], reverse=True).index(node_samples) + 1
                
                if sibling_impurities:
                    comparison['avg_sibling_impurity'] = round(np.mean(sibling_impurities), 4)
                    if node_impurity is not None:
                        comparison['impurity_rank_among_siblings'] = sorted(sibling_impurities + [node_impurity]).index(node_impurity) + 1
        
        return comparison
        
    def _calculate_advanced_analysis(self) -> Dict[str, Any]:
        """Calculate advanced analysis metrics"""
        if self.node_data is None or len(self.node_data) == 0:
            return {'error': 'No data available for advanced analysis'}
        
        analysis = {
            'data_insights': {},
            'feature_correlations': {},
            'anomaly_detection': {},
            'statistical_tests': {}
        }
        
        analysis['data_insights'] = {
            'sample_density': len(self.node_data) / (len(self.node_data.columns) if len(self.node_data.columns) > 0 else 1),
            'missing_data_percentage': round((self.node_data.isnull().sum().sum() / self.node_data.size) * 100, 2),
            'data_completeness_score': round(1 - (self.node_data.isnull().sum().sum() / self.node_data.size), 4)
        }
        
        numeric_columns = self.node_data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 1:
            try:
                import warnings
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "invalid value encountered")
                    warnings.filterwarnings("ignore", "divide by zero")
                    
                    non_constant_features = []
                    for col in numeric_columns:
                        if self.node_data[col].std() > 1e-10:  # Not essentially constant
                            non_constant_features.append(col)
                    
                    if len(non_constant_features) > 1:
                        corr_matrix = self.node_data[non_constant_features].corr()
                        
                        high_correlations = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                corr_value = corr_matrix.iloc[i, j]
                                if pd.notna(corr_value) and abs(corr_value) > 0.7:  # High correlation threshold
                                    high_correlations.append({
                                        'feature1': corr_matrix.columns[i],
                                        'feature2': corr_matrix.columns[j],
                                        'correlation': round(float(corr_value), 4)
                                    })
                        
                        abs_corr = corr_matrix.abs()
                        abs_corr = abs_corr.where(abs_corr != 1.0)  # Remove diagonal (perfect self-correlation)
                        
                        max_corr = abs_corr.max().max()
                        avg_corr = abs_corr.mean().mean()
                        
                        analysis['feature_correlations'] = {
                            'high_correlations': high_correlations,
                            'max_correlation': round(float(max_corr), 4) if pd.notna(max_corr) else 0.0,
                            'avg_correlation': round(float(avg_corr), 4) if pd.notna(avg_corr) else 0.0
                        }
                    else:
                        analysis['feature_correlations'] = {
                            'high_correlations': [],
                            'max_correlation': 0.0,
                            'avg_correlation': 0.0,
                            'note': 'Insufficient non-constant features for correlation analysis'
                        }
            except Exception as e:
                logger.warning(f"Error in correlation analysis: {e}")
                analysis['feature_correlations'] = {
                    'error': f'Correlation analysis failed: {str(e)}'
                }
        
        anomalies = {}
        for column in numeric_columns:
            if column != self.target_column:
                col_data = self.node_data[column].dropna()
                if len(col_data) > 0:
                    q1, q3 = col_data.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    anomalies[column] = {
                        'outlier_count': len(outliers),
                        'outlier_percentage': round((len(outliers) / len(col_data)) * 100, 2)
                    }
        
        outlier_features = []
        for feature, anomaly_data in anomalies.items():
            outlier_features.append({
                'feature': feature,
                'outlier_count': anomaly_data['outlier_count'],
                'outlier_percentage': anomaly_data['outlier_percentage']
            })
        
        analysis['anomaly_detection'] = {
            'outlier_features': outlier_features,
            'total_outlier_features': len([f for f in outlier_features if f['outlier_count'] > 0])
        }
        
        return analysis
    
    def _calculate_skewness(self, data: pd.Series) -> float:
        """Calculate skewness of data with improved numerical stability"""
        try:
            from scipy.stats import skew
            import warnings
            
            data_clean = data.dropna()
            if len(data_clean) < 3:
                return 0.0
            
            if data_clean.std() < 1e-10:
                return 0.0  # Return 0 for essentially constant data
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Precision loss occurred")
                skewness = skew(data_clean, nan_policy='omit')
                
            if not np.isfinite(skewness):
                return 0.0
                
            return round(float(skewness), 4)
        except ImportError:
            data_clean = data.dropna()
            if len(data_clean) < 3:
                return 0.0
            mean = data_clean.mean()
            std = data_clean.std()
            if std == 0:
                return 0.0
            return round(float(((data_clean - mean) / std).pow(3).mean()), 4)
    
    def _detect_outliers(self, data: pd.Series) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        data_clean = data.dropna()
        if len(data_clean) < 4:
            return {'count': 0, 'percentage': 0.0}
        
        q1, q3 = data_clean.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = data_clean[(data_clean < lower_bound) | (data_clean > upper_bound)]
        
        return {
            'count': len(outliers),
            'percentage': round((len(outliers) / len(data_clean)) * 100, 2),
            'lower_bound': round(float(lower_bound), 4),
            'upper_bound': round(float(upper_bound), 4)
        }
    
    def _calculate_entropy(self, data: pd.Series) -> float:
        """Calculate entropy of categorical data"""
        value_counts = data.value_counts()
        total = len(data)
        
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in value_counts:
            if count > 0:
                prob = count / total
                entropy -= prob * np.log2(prob)
        
        return round(entropy, 4)
    
    def _calculate_gini_impurity(self, data: pd.Series) -> float:
        """Calculate Gini impurity"""
        value_counts = data.value_counts()
        total = len(data)
        
        if total == 0:
            return 0.0
        
        gini = 1.0 - sum((count / total) ** 2 for count in value_counts)
        return round(gini, 4)
    
    def _assess_data_quality(self, data: pd.Series) -> Dict[str, Any]:
        """Assess data quality for a feature"""
        quality = {
            'completeness': round(1 - (data.isnull().sum() / len(data)), 4),
            'uniqueness': round(data.nunique() / len(data), 4) if len(data) > 0 else 0,
            'consistency_score': 1.0  # Placeholder for consistency checks
        }
        
        if pd.api.types.is_numeric_dtype(data):
            data_clean = data.dropna()
            if len(data_clean) > 0:
                quality['range_reasonableness'] = self._check_numeric_reasonableness(data_clean)
        
        return quality
    
    def _check_numeric_reasonableness(self, data: pd.Series) -> float:
        """Check if numeric data is in reasonable ranges"""
        q1, q3 = data.quantile([0.25, 0.75])
        iqr = q3 - q1
        
        if iqr == 0:
            return 1.0  # All values are the same
        
        extreme_outliers = data[(data < q1 - 3 * iqr) | (data > q3 + 3 * iqr)]
        outlier_ratio = len(extreme_outliers) / len(data)
        
        return round(1.0 - outlier_ratio, 4)
    
    def _calculate_mutual_info_proxy(self, feature_data: pd.Series, target_data: pd.Series) -> float:
        """Calculate a simple proxy for mutual information"""
        try:
            grouped_variance = feature_data.groupby(target_data).var().mean()
            total_variance = feature_data.var()
            
            if total_variance == 0:
                return 0.0
            
            return round(1.0 - (grouped_variance / total_variance), 4)
        except:
            return 0.0
    
    def _calculate_purity_score(self, target_data: pd.Series) -> float:
        """Calculate purity score for categorical data"""
        if len(target_data) == 0:
            return 0.0
        
        value_counts = target_data.value_counts()
        max_count = value_counts.max()
        total_count = len(target_data)
        
        return round(max_count / total_count, 4)
    
    def _calculate_variance_reduction(self, parent_data: pd.Series, left_data: pd.Series, right_data: pd.Series) -> float:
        """Calculate variance reduction for numeric splits"""
        if len(parent_data) == 0:
            return 0.0
        
        parent_variance = parent_data.var()
        
        if parent_variance == 0:
            return 0.0
        
        left_weight = len(left_data) / len(parent_data)
        right_weight = len(right_data) / len(parent_data)
        
        left_variance = left_data.var() if len(left_data) > 1 else 0
        right_variance = right_data.var() if len(right_data) > 1 else 0
        
        weighted_child_variance = left_weight * left_variance + right_weight * right_variance
        
        variance_reduction = parent_variance - weighted_child_variance
        
        return round(variance_reduction, 4)
    
    def _analyze_split_feature_distributions(self, feature_data: pd.Series, left_condition: pd.Series, right_condition: pd.Series) -> Dict[str, Any]:
        """Analyze feature distributions in left and right splits"""
        left_data = feature_data[left_condition]
        right_data = feature_data[right_condition]
        
        analysis = {
            'left_distribution': {},
            'right_distribution': {},
            'distribution_comparison': {}
        }
        
        if pd.api.types.is_numeric_dtype(feature_data):
            left_stats = left_data.describe() if len(left_data) > 0 else pd.Series()
            right_stats = right_data.describe() if len(right_data) > 0 else pd.Series()
            
            analysis['left_distribution'] = {
                'mean': round(float(left_stats.get('mean', 0)), 4),
                'std': round(float(left_stats.get('std', 0)), 4),
                'min': round(float(left_stats.get('min', 0)), 4),
                'max': round(float(left_stats.get('max', 0)), 4)
            }
            
            analysis['right_distribution'] = {
                'mean': round(float(right_stats.get('mean', 0)), 4),
                'std': round(float(right_stats.get('std', 0)), 4),
                'min': round(float(right_stats.get('min', 0)), 4),
                'max': round(float(right_stats.get('max', 0)), 4)
            }
            
            if len(left_data) > 0 and len(right_data) > 0:
                analysis['distribution_comparison'] = {
                    'mean_difference': round(float(left_stats.get('mean', 0) - right_stats.get('mean', 0)), 4),
                    'std_ratio': round(float(left_stats.get('std', 1) / right_stats.get('std', 1)), 4) if right_stats.get('std', 0) > 0 else 0
                }
        
        return analysis
    
    def _calculate_diversity_index(self, class_counts: Dict[str, int]) -> float:
        """Calculate diversity index (Simpson's diversity index)"""
        total = sum(class_counts.values())
        if total <= 1:
            return 0.0
        
        diversity = 1.0 - sum((count / total) ** 2 for count in class_counts.values())
        return round(diversity, 4)
    
    def _calculate_prediction_confidence(self, actual_values: pd.Series, prediction: Any) -> float:
        """Calculate confidence score for prediction"""
        if len(actual_values) == 0:
            return 0.0
        
        predicted_count = (actual_values == prediction).sum()
        total_count = len(actual_values)
        
        return round(predicted_count / total_count, 4)


class NodeReportingDialog(QDialog):
    """Dialog for comprehensive node reporting and analysis"""
    
    def __init__(self, tree_model, node: TreeNode, node_data: pd.DataFrame = None, 
                 target_column: str = None, parent=None):
        super().__init__(parent)
        
        if node is None:
            raise ValueError("Node cannot be None")
        
        self.tree_model = tree_model
        self.node = node
        self.node_data = node_data
        self.target_column = target_column
        self.statistics = {}
        
        if self.node_data is not None and self.target_column is not None:
            if not isinstance(self.node_data, pd.DataFrame):
                logger.warning("node_data is not a pandas DataFrame")
                self.node_data = None
            elif self.target_column not in self.node_data.columns:
                logger.warning(f"Target column '{self.target_column}' not found in node_data")
        
        self.setWindowTitle(f"Node Report - {getattr(node, 'node_id', 'Unknown')}")
        self.setModal(False)
        self.setMinimumSize(1400, 800)
        self.resize(1650, 950)  # Fits comfortably on 1080p screens with taskbar/window chrome
        
        self.setupUI()
        self.calculateStatistics()
        
    def setupUI(self):
        """Setup the main UI"""
        layout = QVBoxLayout(self)
        
        header_layout = self._create_header_section()
        layout.addLayout(header_layout)
        
        self.tab_widget = QTabWidget()
        
        self._create_all_tabs()
        
        layout.addWidget(self.tab_widget)
        
        button_layout = self._create_button_section()
        layout.addLayout(button_layout)
        
    def _create_header_section(self) -> QHBoxLayout:
        """Create the header section with node information"""
        header_layout = QHBoxLayout()
        
        title_label = QLabel(f"Node Report: {getattr(self.node, 'node_id', 'Unknown')}")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        info_layout = QVBoxLayout()
        
        node_info_layout = QHBoxLayout()
        
        depth_label = QLabel(f"Depth: {getattr(self.node, 'depth', 'N/A')}")
        samples_label = QLabel(f"Samples: {getattr(self.node, 'samples', 'N/A')}")
        type_label = QLabel(f"Type: {'Leaf' if getattr(self.node, 'is_terminal', True) else 'Internal'}")
        
        node_info_layout.addWidget(depth_label)
        node_info_layout.addWidget(QLabel(" | "))
        node_info_layout.addWidget(samples_label)
        node_info_layout.addWidget(QLabel(" | "))
        node_info_layout.addWidget(type_label)
        
        info_layout.addLayout(node_info_layout)
        
        self.stats_summary_label = QLabel("Calculating statistics...")
        self.stats_summary_label.setWordWrap(True)
        info_layout.addWidget(self.stats_summary_label)
        
        header_layout.addLayout(info_layout)
        
        return header_layout
        
    def _create_all_tabs(self):
        """Create all tabs for the dialog"""
        overview_tab = self.createOverviewTab()
        self.tab_widget.addTab(overview_tab, "Overview")
        
        distribution_tab = self.createDistributionTab()
        self.tab_widget.addTab(distribution_tab, "Data Distribution")
        
        target_tab = self.createTargetAnalysisTab()
        self.tab_widget.addTab(target_tab, "Target Analysis")
        
        if not getattr(self.node, 'is_terminal', True):
            split_tab = self.createSplitAnalysisTab()
            self.tab_widget.addTab(split_tab, "Split Analysis")
            
        if getattr(self.node, 'is_terminal', True):
            performance_tab = self.createPerformanceTab()
            self.tab_widget.addTab(performance_tab, "Performance")
            
        importance_tab = self.createFeatureImportanceTab()
        self.tab_widget.addTab(importance_tab, "Feature Importance")
        
        advanced_tab = self.createAdvancedAnalysisTab()
        self.tab_widget.addTab(advanced_tab, "Advanced Analysis")
        
        viz_tab = self.createVisualizationTab()
        self.tab_widget.addTab(viz_tab, "Visualizations")
        
    def _create_button_section(self) -> QHBoxLayout:
        """Create the button section"""
        button_layout = QHBoxLayout()
        
        self.export_report_button = QPushButton("Export Report")
        self.export_report_button.clicked.connect(self.exportReport)
        
        self.export_data_button = QPushButton("Export Node Data")
        self.export_data_button.clicked.connect(self.exportNodeData)
        
        self.export_charts_button = QPushButton("Export Charts")
        self.export_charts_button.clicked.connect(self.exportCharts)
        
        self.refresh_button = QPushButton("Refresh Analysis")
        self.refresh_button.clicked.connect(self.calculateStatistics)
        
        self.compare_button = QPushButton("Compare with Parent")
        self.compare_button.clicked.connect(self.compareWithParent)
        
        button_layout.addWidget(self.export_report_button)
        button_layout.addWidget(self.export_data_button)
        button_layout.addWidget(self.export_charts_button)
        button_layout.addStretch()
        button_layout.addWidget(self.compare_button)
        button_layout.addWidget(self.refresh_button)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)
        
        return button_layout
        
    def createOverviewTab(self) -> QWidget:
        """Create the overview tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll_area = QScrollArea()
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        basic_group = QGroupBox("Basic Information")
        basic_layout = QFormLayout()
        
        self.node_id_label = QLabel()
        self.depth_label = QLabel()
        self.samples_label = QLabel()
        self.type_label = QLabel()
        self.impurity_label = QLabel()
        
        basic_layout.addRow("Node ID:", self.node_id_label)
        basic_layout.addRow("Depth:", self.depth_label)
        basic_layout.addRow("Samples:", self.samples_label)
        basic_layout.addRow("Type:", self.type_label)
        basic_layout.addRow("Impurity:", self.impurity_label)
        
        if getattr(self.node, 'is_terminal', True):
            self.prediction_label = QLabel()
            self.confidence_label = QLabel()
            basic_layout.addRow("Prediction:", self.prediction_label)
            basic_layout.addRow("Confidence:", self.confidence_label)
        else:
            self.split_feature_label = QLabel()
            self.split_value_label = QLabel()
            self.split_operator_label = QLabel()
            self.children_count_label = QLabel()
            
            basic_layout.addRow("Split Feature:", self.split_feature_label)
            basic_layout.addRow("Split Value:", self.split_value_label)
            basic_layout.addRow("Split Operator:", self.split_operator_label)
            basic_layout.addRow("Children Count:", self.children_count_label)
        
        basic_group.setLayout(basic_layout)
        scroll_layout.addWidget(basic_group)
        
        purity_group = QGroupBox("Purity Metrics")
        purity_layout = QFormLayout()
        
        self.gini_label = QLabel()
        self.entropy_label = QLabel()
        self.dominant_class_label = QLabel()
        
        purity_layout.addRow("Gini Impurity:", self.gini_label)
        purity_layout.addRow("Entropy:", self.entropy_label)
        purity_layout.addRow("Dominant Class:", self.dominant_class_label)
        
        purity_group.setLayout(purity_layout)
        scroll_layout.addWidget(purity_group)
        
        stats_group = QGroupBox("Statistics Summary")
        stats_layout = QVBoxLayout()
        
        self.stats_text = QTextEdit()
        self.stats_text.setMaximumHeight(200)
        self.stats_text.setReadOnly(True)
        
        stats_layout.addWidget(self.stats_text)
        stats_group.setLayout(stats_layout)
        scroll_layout.addWidget(stats_group)
        
        scroll_area.setWidget(scroll_content)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        return widget
        
    def createDistributionTab(self) -> QWidget:
        """Create the data distribution tab"""
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        summary_group = QGroupBox("Distribution Summary")
        summary_layout = QFormLayout()
        
        self.total_features_label = QLabel()
        self.numeric_features_label = QLabel()
        self.categorical_features_label = QLabel()
        self.missing_data_label = QLabel()
        
        summary_layout.addRow("Total Features:", self.total_features_label)
        summary_layout.addRow("Numeric Features:", self.numeric_features_label)
        summary_layout.addRow("Categorical Features:", self.categorical_features_label)
        summary_layout.addRow("Missing Data:", self.missing_data_label)
        
        summary_group.setLayout(summary_layout)
        summary_group.setMaximumHeight(150)
        layout.addWidget(summary_group)
        
        self.features_table = QTableWidget()
        self.features_table.setAlternatingRowColors(True)
        self.features_table.setSortingEnabled(True)
        self.features_table.setMinimumWidth(1400)  # Force table to be wide enough
        self.features_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.features_table.setMinimumHeight(400)
        
        table_label = QLabel("Feature Statistics:")
        layout.addWidget(table_label)
        layout.addWidget(self.features_table, 1)  # Stretch factor of 1 to expand
        
        return widget
        
    def createTargetAnalysisTab(self) -> QWidget:
        """Create the target analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        target_group = QGroupBox("Target Variable Information")
        target_layout = QFormLayout()
        
        self.target_column_label = QLabel()
        self.target_type_label = QLabel()
        self.target_missing_label = QLabel()
        self.target_unique_label = QLabel()
        self.target_dominant_label = QLabel()
        
        target_layout.addRow("Target Column:", self.target_column_label)
        target_layout.addRow("Data Type:", self.target_type_label)
        target_layout.addRow("Missing Values:", self.target_missing_label)
        target_layout.addRow("Unique Values:", self.target_unique_label)
        target_layout.addRow("Dominant Class/Mean:", self.target_dominant_label)
        
        target_group.setLayout(target_layout)
        layout.addWidget(target_group)
        
        distribution_group = QGroupBox("Target Distribution")
        distribution_layout = QVBoxLayout()
        
        self.target_distribution_table = QTableWidget()
        self.target_distribution_table.setColumnCount(3)
        self.target_distribution_table.setHorizontalHeaderLabels(['Value', 'Count', 'Percentage'])
        self.target_distribution_table.setMaximumHeight(400)
        self.target_distribution_table.setAlternatingRowColors(True)
        
        distribution_layout.addWidget(self.target_distribution_table)
        distribution_group.setLayout(distribution_layout)
        layout.addWidget(distribution_group)
        
        self.numeric_target_group = QGroupBox("Numeric Target Statistics")
        numeric_layout = QFormLayout()
        
        self.target_mean_label = QLabel()
        self.target_std_label = QLabel()
        self.target_min_label = QLabel()
        self.target_max_label = QLabel()
        self.target_median_label = QLabel()
        
        numeric_layout.addRow("Mean:", self.target_mean_label)
        numeric_layout.addRow("Std Deviation:", self.target_std_label)
        numeric_layout.addRow("Minimum:", self.target_min_label)
        numeric_layout.addRow("Maximum:", self.target_max_label)
        numeric_layout.addRow("Median:", self.target_median_label)
        
        self.numeric_target_group.setLayout(numeric_layout)
        layout.addWidget(self.numeric_target_group)
        
        layout.addStretch()
        return widget
        
    def createSplitAnalysisTab(self) -> QWidget:
        """Create the split analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        split_group = QGroupBox("Split Details")
        split_layout = QFormLayout()
        
        self.split_feature_detail_label = QLabel()
        self.split_value_detail_label = QLabel()
        self.split_operator_detail_label = QLabel()
        self.split_improvement_label = QLabel()
        self.left_samples_label = QLabel()
        self.right_samples_label = QLabel()
        self.balance_score_label = QLabel()
        
        split_layout.addRow("Feature:", self.split_feature_detail_label)
        split_layout.addRow("Value:", self.split_value_detail_label)
        split_layout.addRow("Operator:", self.split_operator_detail_label)
        split_layout.addRow("Improvement:", self.split_improvement_label)
        split_layout.addRow("Left Samples:", self.left_samples_label)
        split_layout.addRow("Right Samples:", self.right_samples_label)
        split_layout.addRow("Balance Score:", self.balance_score_label)
        
        split_group.setLayout(split_layout)
        layout.addWidget(split_group)
        
        quality_group = QGroupBox("Split Quality Analysis")
        quality_layout = QVBoxLayout()
        
        self.split_quality_text = QTextEdit()
        self.split_quality_text.setMaximumHeight(200)
        self.split_quality_text.setReadOnly(True)
        
        quality_layout.addWidget(self.split_quality_text)
        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)
        
        effectiveness_group = QGroupBox("Split Effectiveness")
        effectiveness_layout = QFormLayout()
        
        self.left_purity_label = QLabel()
        self.right_purity_label = QLabel()
        self.variance_reduction_label = QLabel()
        
        effectiveness_layout.addRow("Left Branch Purity:", self.left_purity_label)
        effectiveness_layout.addRow("Right Branch Purity:", self.right_purity_label)
        effectiveness_layout.addRow("Variance Reduction:", self.variance_reduction_label)
        
        effectiveness_group.setLayout(effectiveness_layout)
        layout.addWidget(effectiveness_group)
        
        layout.addStretch()
        return widget
        
    def createPerformanceTab(self) -> QWidget:
        """Create the performance tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        perf_group = QGroupBox("Performance Metrics")
        perf_layout = QFormLayout()
        
        self.accuracy_label = QLabel()
        self.error_rate_label = QLabel()
        self.prediction_confidence_label = QLabel()
        self.correct_predictions_label = QLabel()
        self.incorrect_predictions_label = QLabel()
        
        perf_layout.addRow("Accuracy:", self.accuracy_label)
        perf_layout.addRow("Error Rate:", self.error_rate_label)
        perf_layout.addRow("Confidence Score:", self.prediction_confidence_label)
        perf_layout.addRow("Correct Predictions:", self.correct_predictions_label)
        perf_layout.addRow("Incorrect Predictions:", self.incorrect_predictions_label)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        pred_group = QGroupBox("Prediction Analysis")
        pred_layout = QVBoxLayout()
        
        self.performance_text = QTextEdit()
        self.performance_text.setMaximumHeight(200)
        self.performance_text.setReadOnly(True)
        
        pred_layout.addWidget(self.performance_text)
        pred_group.setLayout(pred_layout)
        layout.addWidget(pred_group)
        
        class_group = QGroupBox("Class Distribution in Node")
        class_layout = QVBoxLayout()
        
        self.class_distribution_table = QTableWidget()
        self.class_distribution_table.setColumnCount(2)
        self.class_distribution_table.setHorizontalHeaderLabels(['Class', 'Count'])
        self.class_distribution_table.setMaximumHeight(250)
        
        class_layout.addWidget(self.class_distribution_table)
        class_group.setLayout(class_layout)
        layout.addWidget(class_group)
        
        layout.addStretch()
        return widget
        
    def createFeatureImportanceTab(self) -> QWidget:
        """Create the feature importance tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        method_group = QGroupBox("Importance Analysis")
        method_layout = QFormLayout()
        
        self.importance_method_label = QLabel()
        self.features_analyzed_label = QLabel()
        
        method_layout.addRow("Method Used:", self.importance_method_label)
        method_layout.addRow("Features Analyzed:", self.features_analyzed_label)
        
        method_group.setLayout(method_layout)
        layout.addWidget(method_group)
        
        top_group = QGroupBox("Top Important Features")
        top_layout = QVBoxLayout()
        
        self.importance_table = QTableWidget()
        self.importance_table.setColumnCount(2)
        self.importance_table.setHorizontalHeaderLabels(['Feature', 'Importance Score'])
        self.importance_table.setAlternatingRowColors(True)
        self.importance_table.setSortingEnabled(True)
        
        top_layout.addWidget(self.importance_table)
        top_group.setLayout(top_layout)
        layout.addWidget(top_group)
        
        return widget
        
    def createAdvancedAnalysisTab(self) -> QWidget:
        """Create the advanced analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll_area = QScrollArea()
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        insights_group = QGroupBox("Data Insights")
        insights_layout = QFormLayout()
        
        self.sample_density_label = QLabel()
        self.missing_percentage_label = QLabel()
        self.completeness_score_label = QLabel()
        
        insights_layout.addRow("Sample Density:", self.sample_density_label)
        insights_layout.addRow("Missing Data %:", self.missing_percentage_label)
        insights_layout.addRow("Completeness Score:", self.completeness_score_label)
        
        insights_group.setLayout(insights_layout)
        scroll_layout.addWidget(insights_group)
        
        corr_group = QGroupBox("Feature Correlations")
        corr_layout = QVBoxLayout()
        
        self.correlations_table = QTableWidget()
        self.correlations_table.setColumnCount(3)
        self.correlations_table.setHorizontalHeaderLabels(['Feature 1', 'Feature 2', 'Correlation'])
        self.correlations_table.setMaximumHeight(300)
        
        corr_layout.addWidget(self.correlations_table)
        corr_group.setLayout(corr_layout)
        scroll_layout.addWidget(corr_group)
        
        anomaly_group = QGroupBox("Anomaly Detection")
        anomaly_layout = QVBoxLayout()
        
        self.anomalies_table = QTableWidget()
        self.anomalies_table.setColumnCount(3)
        self.anomalies_table.setHorizontalHeaderLabels(['Feature', 'Outlier Count', 'Outlier %'])
        self.anomalies_table.setMaximumHeight(300)
        
        anomaly_layout.addWidget(self.anomalies_table)
        anomaly_group.setLayout(anomaly_layout)
        scroll_layout.addWidget(anomaly_group)
        
        scroll_area.setWidget(scroll_content)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        return widget
        
    def createVisualizationTab(self) -> QWidget:
        """Create the visualization tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        if not MATPLOTLIB_AVAILABLE:
            no_viz_label = QLabel("Matplotlib not available - visualizations disabled")
            no_viz_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(no_viz_label)
            return widget
        
        controls_layout = QHBoxLayout()
        
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems([
            "Target Distribution",
            "Feature Distributions", 
            "Split Visualization",
            "Node Purity",
            "Feature Correlations",
            "Data Quality Overview"
        ])
        self.viz_type_combo.currentTextChanged.connect(self.updateVisualization)
        
        self.refresh_viz_button = QPushButton("Refresh Chart")
        self.refresh_viz_button.clicked.connect(self.updateVisualization)
        
        controls_layout.addWidget(QLabel("Visualization:"))
        controls_layout.addWidget(self.viz_type_combo)
        controls_layout.addStretch()
        controls_layout.addWidget(self.refresh_viz_button)
        
        layout.addLayout(controls_layout)
        
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        return widget
        
    def calculateStatistics(self):
        """Calculate comprehensive node statistics"""
        try:
            self.stats_summary_label.setText("Calculating statistics...")
            QApplication.processEvents()
            
            calculator = NodeStatisticsCalculator(
                self.node, self.node_data, self.target_column
            )
            self.statistics = calculator.calculate_comprehensive_stats()
            
            self.updateOverviewTab()
            self.updateDistributionTab()
            self.updateTargetAnalysisTab()
            
            if not getattr(self.node, 'is_terminal', True):
                self.updateSplitAnalysisTab()
            else:
                self.updatePerformanceTab()
                
            self.updateFeatureImportanceTab()
            self.updateAdvancedAnalysisTab()
            self.updateVisualization()
            
            self._update_header_summary()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error calculating statistics: {str(e)}")
            logger.error(f"Error calculating node statistics: {str(e)}", exc_info=True)
            self.stats_summary_label.setText("Error calculating statistics")
    
    def _update_header_summary(self):
        """Update the header summary with key statistics"""
        try:
            summary_parts = []
            
            if 'basic_info' in self.statistics:
                data_samples = self.statistics['basic_info'].get('data_samples', 0)
                summary_parts.append(f"Data: {data_samples} samples")
            
            if 'data_distribution' in self.statistics and 'error' not in self.statistics['data_distribution']:
                dist = self.statistics['data_distribution']
                total_features = dist.get('total_features', 0)
                numeric_features = dist.get('numeric_features', 0)
                summary_parts.append(f"Features: {total_features} ({numeric_features} numeric)")
            
            if 'target_analysis' in self.statistics and 'error' not in self.statistics['target_analysis']:
                target = self.statistics['target_analysis']
                if target.get('type') == 'categorical':
                    classes = target.get('unique_classes', 0)
                    summary_parts.append(f"Target: {classes} classes")
                else:
                    summary_parts.append(f"Target: numeric")
            
            if getattr(self.node, 'is_terminal', True):
                if 'performance_metrics' in self.statistics and 'accuracy' in self.statistics['performance_metrics']:
                    accuracy = self.statistics['performance_metrics']['accuracy']
                    summary_parts.append(f"Accuracy: {accuracy:.3f}")
            else:
                if 'purity_metrics' in self.statistics and 'gini_impurity' in self.statistics['purity_metrics']:
                    gini = self.statistics['purity_metrics']['gini_impurity']
                    summary_parts.append(f"Gini: {gini:.3f}")
            
            summary_text = " | ".join(summary_parts)
            self.stats_summary_label.setText(summary_text)
            
        except Exception as e:
            logger.warning(f"Error updating header summary: {str(e)}")
            self.stats_summary_label.setText("Statistics calculated")
    
    def updateOverviewTab(self):
        """Update the overview tab with statistics"""
        basic_info = self.statistics.get('basic_info', {})
        purity_info = self.statistics.get('purity_metrics', {})
        
        self.node_id_label.setText(str(basic_info.get('node_id', 'N/A')))
        self.depth_label.setText(str(basic_info.get('depth', 'N/A')))
        self.samples_label.setText(str(basic_info.get('samples', 'N/A')))
        self.type_label.setText("Terminal (Leaf)" if basic_info.get('is_terminal') else "Internal (Split)")
        self.impurity_label.setText(f"{basic_info.get('impurity', 0):.4f}")
        
        if basic_info.get('is_terminal'):
            if hasattr(self, 'prediction_label'):
                self.prediction_label.setText(str(basic_info.get('prediction', 'N/A')))
            if hasattr(self, 'confidence_label'):
                self.confidence_label.setText(str(basic_info.get('confidence', 'N/A')))
        else:
            if hasattr(self, 'split_feature_label'):
                self.split_feature_label.setText(str(basic_info.get('split_feature', 'N/A')))
            if hasattr(self, 'split_value_label'):
                self.split_value_label.setText(str(basic_info.get('split_value', 'N/A')))
            if hasattr(self, 'split_operator_label'):
                self.split_operator_label.setText(str(basic_info.get('split_operator', 'N/A')))
            if hasattr(self, 'children_count_label'):
                self.children_count_label.setText(str(basic_info.get('children_count', 'N/A')))
        
        self.gini_label.setText(f"{purity_info.get('gini_impurity', 0):.4f}")
        self.entropy_label.setText(f"{purity_info.get('entropy', 0):.4f}")
        self.dominant_class_label.setText(str(purity_info.get('dominant_class', 'N/A')))
        
        summary_lines = []
        
        if 'data_distribution' in self.statistics:
            dist = self.statistics['data_distribution']
            if 'error' not in dist:
                summary_lines.append(f"Features: {dist.get('total_features', 0)} total "
                                    f"({dist.get('numeric_features', 0)} numeric, "
                                    f"{dist.get('categorical_features', 0)} categorical)")
                                    
        if 'target_analysis' in self.statistics:
            target = self.statistics['target_analysis']
            if 'error' not in target:
                if target.get('type') == 'categorical':
                    summary_lines.append(f"Target: {target.get('unique_classes', 0)} classes, "
                                        f"majority: {target.get('majority_class', 'N/A')} "
                                        f"({target.get('majority_percentage', 0):.1f}%)")
                else:
                    summary_lines.append(f"Target: mean={target.get('mean', 0):.3f}, "
                                        f"std={target.get('std', 0):.3f}")
                                        
        if 'purity_metrics' in self.statistics:
            purity = self.statistics['purity_metrics']
            if 'error' not in purity:
                if purity.get('type') == 'categorical':
                    summary_lines.append(f"Purity: Gini={purity.get('gini_impurity', 0):.3f}, "
                                        f"Entropy={purity.get('entropy', 0):.3f}")
                else:
                    summary_lines.append(f"Variance: {purity.get('variance', 0):.3f}")
                    
        self.stats_text.setPlainText('\n'.join(summary_lines))
        
    def updateDistributionTab(self):
        """Update the distribution tab"""
        if 'data_distribution' not in self.statistics:
            return
            
        dist = self.statistics['data_distribution']
        if 'error' in dist:
            return
            
        self.total_features_label.setText(str(dist.get('total_features', 0)))
        self.numeric_features_label.setText(str(dist.get('numeric_features', 0)))
        self.categorical_features_label.setText(str(dist.get('categorical_features', 0)))
        missing_data_pct = dist.get('missing_data_percentage', 0)
        if missing_data_pct == 0 and self.node_data is not None:
            total_cells = self.node_data.size
            missing_cells = self.node_data.isnull().sum().sum()
            missing_data_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0
        self.missing_data_label.setText(f"{missing_data_pct:.1f}%")
        
        feature_stats = dist.get('feature_statistics', {})
        missing_values = dist.get('missing_values', {})
        
        self.features_table.setRowCount(len(feature_stats))
        self.features_table.setColumnCount(7)
        self.features_table.setHorizontalHeaderLabels([
            'Feature', 'Type', 'Count/Unique', 'Mean/Mode', 'Std/Entropy', 'Range/Top_Value', 'Missing %'
        ])
        
        for i, (feature, stats) in enumerate(feature_stats.items()):
            self.features_table.setItem(i, 0, QTableWidgetItem(feature))
            self.features_table.setItem(i, 1, QTableWidgetItem(stats.get('type', 'unknown')))
            
            missing_info = missing_values.get(feature, {})
            missing_pct = missing_info.get('percentage', 0)
            
            if stats.get('type') == 'numeric':
                self.features_table.setItem(i, 2, QTableWidgetItem(str(stats.get('count', 'N/A'))))
                self.features_table.setItem(i, 3, QTableWidgetItem(f"{stats.get('mean', 0):.3f}"))
                self.features_table.setItem(i, 4, QTableWidgetItem(f"{stats.get('std', 0):.3f}"))
                self.features_table.setItem(i, 5, QTableWidgetItem(f"{stats.get('min', 0):.2f} - {stats.get('max', 0):.2f}"))
            else:
                self.features_table.setItem(i, 2, QTableWidgetItem(str(stats.get('unique_values', 'N/A'))))
                self.features_table.setItem(i, 3, QTableWidgetItem(str(stats.get('most_frequent', 'N/A'))))
                self.features_table.setItem(i, 4, QTableWidgetItem(f"{stats.get('entropy', 0):.3f}"))
                self.features_table.setItem(i, 5, QTableWidgetItem(str(stats.get('most_frequent_count', 'N/A'))))
            
            self.features_table.setItem(i, 6, QTableWidgetItem(f"{missing_pct:.1f}%"))
        
        self.features_table.resizeColumnsToContents()
        
        header = self.features_table.horizontalHeader()
        header.setMinimumSectionSize(120)  # Minimum column width
        
        total_width = 1500  # Available width for columns (accounting for scrollbars)
        self.features_table.setColumnWidth(0, 260)  # Feature name - wider for full names
        self.features_table.setColumnWidth(1, 100)  # Type - slightly larger
        self.features_table.setColumnWidth(2, 160)  # Count/Unique - larger
        self.features_table.setColumnWidth(3, 180)  # Mean/Mode - larger  
        self.features_table.setColumnWidth(4, 180)  # Std/Entropy - larger
        self.features_table.setColumnWidth(5, 250)  # Range/Top_Value - much wider
        self.features_table.setColumnWidth(6, 130)  # Missing % - larger
        
        self.features_table.setHorizontalScrollMode(self.features_table.ScrollPerPixel)
        
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.Interactive)  # Feature names can be resized
        
    def updateTargetAnalysisTab(self):
        """Update the target analysis tab"""
        if 'target_analysis' not in self.statistics:
            return
            
        target_analysis = self.statistics['target_analysis']
        
        if 'error' in target_analysis:
            self.target_column_label.setText('Error')
            self.target_type_label.setText(target_analysis['error'])
            return
        
        self.target_column_label.setText(str(target_analysis.get('target_column', 'N/A')))
        self.target_type_label.setText(str(target_analysis.get('type', 'N/A')))
        self.target_missing_label.setText(f"{target_analysis.get('missing_values', 0)} ({target_analysis.get('missing_percentage', 0):.1f}%)")
        self.target_unique_label.setText(str(target_analysis.get('unique_values', 'N/A')))
        
        if target_analysis.get('type') == 'categorical':
            self.target_dominant_label.setText(f"{target_analysis.get('majority_class', 'N/A')} ({target_analysis.get('majority_percentage', 0):.1f}%)")
            
            class_dist = target_analysis.get('class_distribution', {})
            self.target_distribution_table.setRowCount(len(class_dist))
            
            for i, (class_val, info) in enumerate(class_dist.items()):
                value_item = QTableWidgetItem(str(class_val))
                count_item = QTableWidgetItem(str(info.get('count', 0)))
                pct_item = QTableWidgetItem(f"{info.get('percentage', 0):.1f}%")
                
                self.target_distribution_table.setItem(i, 0, value_item)
                self.target_distribution_table.setItem(i, 1, count_item)
                self.target_distribution_table.setItem(i, 2, pct_item)
            
            self.numeric_target_group.setVisible(False)
                
        else:
            self.target_dominant_label.setText(f"Mean: {target_analysis.get('mean', 0):.3f}")
            
            self.target_distribution_table.setRowCount(0)
            
            self.numeric_target_group.setVisible(True)
            if hasattr(self, 'target_mean_label'):
                self.target_mean_label.setText(f"{target_analysis.get('mean', 0):.4f}")
                self.target_std_label.setText(f"{target_analysis.get('std', 0):.4f}")
                self.target_min_label.setText(f"{target_analysis.get('min', 0):.4f}")
                self.target_max_label.setText(f"{target_analysis.get('max', 0):.4f}")
                self.target_median_label.setText(f"{target_analysis.get('median', 0):.4f}")
        
        self.target_distribution_table.resizeColumnsToContents()
            
    def updateSplitAnalysisTab(self):
        """Update the split analysis tab"""
        if 'split_analysis' not in self.statistics:
            return
            
        split = self.statistics['split_analysis']
        
        if 'error' in split:
            error_msg = split.get('error', 'Unknown error')
            if hasattr(self, 'split_feature_detail_label'):
                self.split_feature_detail_label.setText(f"Error: {error_msg}")
            return
        
        self.split_feature_detail_label.setText(str(split.get('feature', 'N/A')))
        self.split_value_detail_label.setText(str(split.get('value', 'N/A')))
        self.split_operator_detail_label.setText(str(split.get('operator', 'N/A')))
        self.split_improvement_label.setText(f"{split.get('improvement', 0):.4f}")
        
        if 'left_samples' in split:
            self.left_samples_label.setText(f"{split['left_samples']} "
                                           f"({split.get('left_percentage', 0):.1f}%)")
            self.right_samples_label.setText(f"{split['right_samples']} "
                                            f"({split.get('right_percentage', 0):.1f}%)")
            
        if 'balance_score' in split:
            self.balance_score_label.setText(f"{split['balance_score']:.4f}")
        
        if hasattr(self, 'left_purity_label'):
            self.left_purity_label.setText(f"{split.get('left_purity', 0):.4f}")
            self.right_purity_label.setText(f"{split.get('right_purity', 0):.4f}")
            variance_reduction = split.get('variance_reduction', 0)
            if hasattr(self, 'variance_reduction_label'):
                self.variance_reduction_label.setText(f"{variance_reduction:.4f}")

    def updatePerformanceTab(self):
        """Update the performance tab"""
        if 'performance_metrics' not in self.statistics:
            return
            
        perf = self.statistics['performance_metrics']
        if 'error' in perf:
            return
            
        if perf.get('type') == 'classification':
            accuracy = perf.get('accuracy', 0)
            self.accuracy_label.setText(f"{accuracy:.3f} ({accuracy*100:.1f}%)")
            self.error_rate_label.setText(f"{1-accuracy:.3f} ({(1-accuracy)*100:.1f}%)")
            
            class_dist = perf.get('class_distribution', {})
            if class_dist:
                max_class_count = max(class_dist.values())
                total_count = sum(class_dist.values())
                confidence = max_class_count / total_count if total_count > 0 else 0
                self.prediction_confidence_label.setText(f"{confidence:.3f}")
            else:
                self.prediction_confidence_label.setText("N/A")
                
        elif perf.get('type') == 'regression':
            mae = perf.get('mae', 0)
            rmse = perf.get('rmse', 0)
            r_squared = perf.get('r_squared', 0)
            
            self.accuracy_label.setText(f"R²: {r_squared:.3f}")
            self.error_rate_label.setText(f"RMSE: {rmse:.3f}")
            self.prediction_confidence_label.setText(f"MAE: {mae:.3f}")
            
        detail_lines = []
        detail_lines.append(f"Prediction: {perf.get('prediction', 'N/A')}")
        detail_lines.append(f"Sample Count: {perf.get('sample_count', 0)}")
        
        if perf.get('type') == 'classification':
            detail_lines.append(f"Correct Predictions: {perf.get('correct_predictions', 0)}")
            detail_lines.append(f"Incorrect Predictions: {perf.get('incorrect_predictions', 0)}")
            
        self.performance_text.setPlainText('\n'.join(detail_lines))

    def updateFeatureImportanceTab(self):
        """Update the feature importance tab"""
        if 'feature_importance' not in self.statistics:
            return
            
        feature_importance = self.statistics['feature_importance']
        if 'error' in feature_importance:
            self.importance_method_label.setText("Error")
            self.features_analyzed_label.setText(feature_importance['error'])
            return
            
        self.importance_method_label.setText("Correlation-based analysis")
        total_features = feature_importance.get('total_features_analyzed', 0)
        self.features_analyzed_label.setText(str(total_features))
        
        if 'feature_importance' in feature_importance and isinstance(feature_importance['feature_importance'], dict):
            importance_data = feature_importance['feature_importance']
            self.importance_table.setRowCount(len(importance_data))
            
            for i, (feature, score) in enumerate(importance_data.items()):
                self.importance_table.setItem(i, 0, QTableWidgetItem(feature))
                self.importance_table.setItem(i, 1, QTableWidgetItem(f"{score:.4f}"))
        elif 'top_5_features' in feature_importance:
            top_features = feature_importance['top_5_features']
            self.importance_table.setRowCount(len(top_features))
            
            for i, (feature, score) in enumerate(top_features.items()):
                self.importance_table.setItem(i, 0, QTableWidgetItem(feature))
                self.importance_table.setItem(i, 1, QTableWidgetItem(f"{score:.4f}"))
        else:
            self.importance_table.setRowCount(1)
            self.importance_table.setItem(0, 0, QTableWidgetItem("No data available"))
            self.importance_table.setItem(0, 1, QTableWidgetItem("N/A"))
            
        self.importance_table.resizeColumnsToContents()
        
    def updateAdvancedAnalysisTab(self):
        """Update the advanced analysis tab with calculated statistics"""
        try:
            advanced_stats = self.statistics.get('advanced_analysis', {})
            
            if 'error' in advanced_stats:
                error_text = advanced_stats['error']
                self.sample_density_label.setText(f"Error: {error_text}")
                self.missing_percentage_label.setText(f"Error: {error_text}")
                self.completeness_score_label.setText(f"Error: {error_text}")
                self.correlations_table.setRowCount(0)
                self.anomalies_table.setRowCount(0)
                return
            
            data_insights = advanced_stats.get('data_insights', {})
            self.sample_density_label.setText(f"{data_insights.get('sample_density', 0):.2f}")
            self.missing_percentage_label.setText(f"{data_insights.get('missing_data_percentage', 0):.2f}%")
            self.completeness_score_label.setText(f"{data_insights.get('data_completeness_score', 0):.4f}")
            
            feature_correlations = advanced_stats.get('feature_correlations', {})
            high_correlations = feature_correlations.get('high_correlations', [])
            
            self.correlations_table.setRowCount(len(high_correlations))
            for i, corr in enumerate(high_correlations):
                self.correlations_table.setItem(i, 0, QTableWidgetItem(str(corr.get('feature1', ''))))
                self.correlations_table.setItem(i, 1, QTableWidgetItem(str(corr.get('feature2', ''))))
                self.correlations_table.setItem(i, 2, QTableWidgetItem(f"{corr.get('correlation', 0):.4f}"))
            
            self.correlations_table.resizeColumnsToContents()
            
            anomaly_detection = advanced_stats.get('anomaly_detection', {})
            outlier_info = anomaly_detection.get('outlier_features', [])
            
            self.anomalies_table.setRowCount(len(outlier_info))
            for i, outlier in enumerate(outlier_info):
                self.anomalies_table.setItem(i, 0, QTableWidgetItem(str(outlier.get('feature', ''))))
                self.anomalies_table.setItem(i, 1, QTableWidgetItem(str(outlier.get('outlier_count', 0))))
                self.anomalies_table.setItem(i, 2, QTableWidgetItem(f"{outlier.get('outlier_percentage', 0):.2f}%"))
            
            self.anomalies_table.resizeColumnsToContents()
            
        except Exception as e:
            logger.error(f"Error updating advanced analysis tab: {str(e)}", exc_info=True)
            error_msg = f"Update Error: {str(e)}"
            self.sample_density_label.setText(error_msg)
            self.missing_percentage_label.setText(error_msg)
            self.completeness_score_label.setText(error_msg)
            
    def updateVisualization(self):
        """Update the visualization based on selected type"""
        viz_type = self.viz_type_combo.currentText()
        
        self.figure.clear()
        
        try:
            if viz_type == "Target Distribution":
                self._plot_target_distribution()
            elif viz_type == "Feature Distributions":
                self._plot_feature_distributions()
            elif viz_type == "Split Visualization":
                self._plot_split_visualization()
            elif viz_type == "Node Purity":
                self._plot_node_purity()
                
        except Exception as e:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Visualization Error:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            
        self.figure.tight_layout()
        self.canvas.draw()
        
    def _plot_target_distribution(self):
        """Plot target variable distribution"""
        if (self.node_data is None or len(self.node_data) == 0 or 
            self.target_column not in self.node_data.columns):
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No target data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        ax = self.figure.add_subplot(111)
        target_data = self.node_data[self.target_column].dropna()
        
        if pd.api.types.is_numeric_dtype(target_data):
            if len(target_data) > 0:
                ax.hist(target_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_xlabel(self.target_column)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Target Distribution in Node {getattr(self.node, "node_id", "unknown")}')
            else:
                ax.text(0.5, 0.5, 'No numeric target data to plot', 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            value_counts = target_data.value_counts()
            if len(value_counts) > 0:
                ax.bar(range(len(value_counts)), value_counts.values, color='lightcoral')
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                ax.set_xlabel(self.target_column)
                ax.set_ylabel('Count')
                ax.set_title(f'Target Distribution in Node {getattr(self.node, "node_id", "unknown")}')
            else:
                ax.text(0.5, 0.5, 'No categorical data to plot', 
                       ha='center', va='center', transform=ax.transAxes)
            
    def _plot_feature_distributions(self):
        """Plot feature distributions"""
        if self.node_data is None or len(self.node_data) == 0:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        numeric_cols = self.node_data.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)
            
        if not numeric_cols:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No numeric features to plot', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        n_plots = min(4, len(numeric_cols))
        
        for i, col in enumerate(numeric_cols[:n_plots]):
            ax = self.figure.add_subplot(2, 2, i+1)
            data = self.node_data[col].dropna()
            if len(data) > 0:
                ax.hist(data, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
                ax.set_title(f'{col}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                
    def _plot_split_visualization(self):
        """Plot split visualization"""
        if getattr(self.node, 'is_terminal', True):
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No split to visualize\n(Leaf node)', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        split_feature = getattr(self.node, 'split_feature', None)
        if (self.node_data is None or len(self.node_data) == 0 or 
            split_feature is None or split_feature not in self.node_data.columns):
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Split feature data not available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        ax = self.figure.add_subplot(111)
        
        feature = getattr(self.node, 'split_feature', None)
        threshold = getattr(self.node, 'split_value', None)
        
        if feature is None or threshold is None:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Split information not available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        feature_data = self.node_data[feature].dropna()
        
        if pd.api.types.is_numeric_dtype(feature_data):
            if len(feature_data) > 0:
                ax.hist(feature_data, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
                ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                          label=f'Split at {threshold}')
                ax.set_xlabel(feature)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Split Visualization: {feature}')
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No numeric data to visualize', 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            value_counts = feature_data.value_counts()
            if len(value_counts) > 0:
                colors = ['red' if val == threshold else 'lightblue' for val in value_counts.index]
                ax.bar(range(len(value_counts)), value_counts.values, color=colors)
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                ax.set_xlabel(feature)
                ax.set_ylabel('Count')
                ax.set_title(f'Split Visualization: {feature} == {threshold}')
            else:
                ax.text(0.5, 0.5, 'No categorical data to visualize', 
                       ha='center', va='center', transform=ax.transAxes)
            
    def _plot_node_purity(self):
        """Plot node purity metrics"""
        if 'purity_metrics' not in self.statistics:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Purity metrics not available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
            
        purity = self.statistics['purity_metrics']
        
        if purity.get('type') == 'categorical':
            metrics = ['Gini Impurity', 'Entropy', 'Misclassification Error']
            values = [
                purity.get('gini_impurity', 0),
                purity.get('entropy', 0),
                purity.get('misclassification_error', 0)
            ]
            
            ax = self.figure.add_subplot(111)
            bars = ax.bar(metrics, values, color=['lightcoral', 'lightgreen', 'lightblue'])
            ax.set_ylabel('Impurity')
            ax.set_title(f'Node Purity Metrics - {getattr(self.node, "node_id", "unknown")}')
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
                       
        else:
            ax = self.figure.add_subplot(111)
            variance = purity.get('variance', 0)
            std = purity.get('std', 0)
            
            metrics = ['Variance', 'Standard Deviation']
            values = [variance, std]
            
            bars = ax.bar(metrics, values, color=['orange', 'purple'])
            ax.set_ylabel('Value')
            ax.set_title(f'Node Variability Metrics - {getattr(self.node, "node_id", "unknown")}')
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
                       
    def exportReport(self):
        """Export comprehensive node report"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Node Report", 
            f"node_report_{self.node.node_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt);;JSON Files (*.json);;All Files (*)"
        )
        
        if not filename:
            return
            
        try:
            if filename.endswith('.json'):
                report_data = {
                    'node_info': {
                        'node_id': self.node.node_id,
                        'depth': self.node.depth,
                        'samples': self.node.samples,
                        'is_terminal': self.node.is_terminal,
                        'prediction': self.node.prediction if self.node.is_terminal else None,
                        'split_feature': self.node.split_feature if not self.node.is_terminal else None,
                        'split_value': self.node.split_value if not self.node.is_terminal else None
                    },
                    'statistics': self.statistics,
                    'export_info': {
                        'timestamp': datetime.now().isoformat(),
                        'target_column': self.target_column,
                        'data_samples': len(self.node_data) if self.node_data is not None else 0
                    }
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, default=str)
                    
            else:
                report_lines = []
                report_lines.append(f"Node Report: {self.node.node_id}")
                report_lines.append("=" * 50)
                report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append("")
                
                report_lines.append("BASIC INFORMATION")
                report_lines.append("-" * 20)
                report_lines.append(f"Node ID: {self.node.node_id}")
                report_lines.append(f"Depth: {self.node.depth}")
                report_lines.append(f"Type: {'Leaf' if self.node.is_terminal else 'Internal'}")
                report_lines.append(f"Samples: {self.node.samples}")
                
                if self.node.is_terminal:
                    report_lines.append(f"Prediction: {self.node.prediction}")
                else:
                    report_lines.append(f"Split Feature: {self.node.split_feature}")
                    report_lines.append(f"Split Value: {self.node.split_value}")
                    report_lines.append(f"Split Operator: {getattr(self.node, 'split_operator', '<=')}")
                    
                report_lines.append("")
                
                for section_name, section_data in self.statistics.items():
                    if isinstance(section_data, dict) and 'error' not in section_data:
                        report_lines.append(f"{section_name.upper().replace('_', ' ')}")
                        report_lines.append("-" * len(section_name))
                        
                        for key, value in section_data.items():
                            if isinstance(value, dict):
                                report_lines.append(f"{key}:")
                                for subkey, subvalue in value.items():
                                    report_lines.append(f"  {subkey}: {subvalue}")
                            else:
                                report_lines.append(f"{key}: {value}")
                                
                        report_lines.append("")
                        
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(report_lines))
                    
            QMessageBox.information(self, "Export Successful", 
                                  f"Node report exported to:\n{filename}")
                                  
        except Exception as e:
            QMessageBox.critical(self, "Export Error", 
                               f"Error exporting report: {str(e)}")
            
    def exportNodeData(self):
        """Export node data to CSV"""
        if self.node_data is None or len(self.node_data) == 0:
            QMessageBox.warning(self, "No Data", "No data available to export.")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Node Data", 
            f"node_data_{self.node.node_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not filename:
            return
            
        try:
            self.node_data.to_csv(filename, index=False)
            QMessageBox.information(self, "Export Successful", 
                                  f"Node data exported to:\n{filename}")
                                  
        except Exception as e:
            QMessageBox.critical(self, "Export Error", 
                               f"Error exporting data: {str(e)}")

    def exportCharts(self):
        """Export visualization charts to image files"""
        if not MATPLOTLIB_AVAILABLE:
            QMessageBox.warning(self, "Charts Not Available", 
                              "Matplotlib is not available. Cannot export charts.")
            return
            
        if not hasattr(self, 'canvas') or self.canvas is None:
            QMessageBox.warning(self, "No Charts", "No charts available to export.")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Charts", 
            f"node_charts_{self.node.node_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
        )
        
        if not filename:
            return
            
        try:
            self.figure.savefig(filename, dpi=300, bbox_inches='tight', 
                              facecolor='white', edgecolor='none')
            QMessageBox.information(self, "Export Successful", 
                                  f"Charts exported to:\n{filename}")
                                  
        except Exception as e:
            QMessageBox.critical(self, "Export Error", 
                               f"Error exporting charts: {str(e)}")

    def compareWithParent(self):
        """Compare this node with its parent node"""
        try:
            if not hasattr(self.node, 'parent') or self.node.parent is None:
                QMessageBox.information(self, "No Parent", 
                                      "This node has no parent to compare with (it's the root node).")
                return
            
            parent_node = self.node.parent
            
            comparison_dialog = QDialog(self)
            comparison_dialog.setWindowTitle(f"Compare Node {self.node.node_id} with Parent {parent_node.node_id}")
            comparison_dialog.setModal(True)
            comparison_dialog.resize(800, 600)
            
            layout = QVBoxLayout(comparison_dialog)
            
            table = QTableWidget()
            table.setColumnCount(3)
            table.setHorizontalHeaderLabels(['Property', f'Node {self.node.node_id}', f'Parent {parent_node.node_id}'])
            
            comparisons = [
                ('Node ID', str(self.node.node_id), str(parent_node.node_id)),
                ('Depth', str(getattr(self.node, 'depth', 'N/A')), str(getattr(parent_node, 'depth', 'N/A'))),
                ('Samples', str(getattr(self.node, 'samples', 'N/A')), str(getattr(parent_node, 'samples', 'N/A'))),
                ('Is Terminal', str(getattr(self.node, 'is_terminal', 'N/A')), str(getattr(parent_node, 'is_terminal', 'N/A'))),
                ('Impurity', f"{getattr(self.node, 'impurity', 0):.4f}", f"{getattr(parent_node, 'impurity', 0):.4f}"),
                ('Majority Class', str(getattr(self.node, 'majority_class', 'N/A')), str(getattr(parent_node, 'majority_class', 'N/A'))),
                ('Prediction', str(getattr(self.node, 'prediction', 'N/A')), str(getattr(parent_node, 'prediction', 'N/A')))
            ]
            
            if not getattr(self.node, 'is_terminal', True):
                comparisons.extend([
                    ('Split Feature', str(getattr(self.node, 'split_feature', 'N/A')), 'N/A'),
                    ('Split Value', str(getattr(self.node, 'split_value', 'N/A')), 'N/A'),
                    ('Split Type', str(getattr(self.node, 'split_type', 'N/A')), 'N/A')
                ])
            
            if not getattr(parent_node, 'is_terminal', True):
                parent_split_comparisons = [
                    ('Parent Split Feature', 'N/A', str(getattr(parent_node, 'split_feature', 'N/A'))),
                    ('Parent Split Value', 'N/A', str(getattr(parent_node, 'split_value', 'N/A'))),
                    ('Parent Split Type', 'N/A', str(getattr(parent_node, 'split_type', 'N/A')))
                ]
                comparisons.extend(parent_split_comparisons)
            
            table.setRowCount(len(comparisons))
            
            for i, (prop, node_val, parent_val) in enumerate(comparisons):
                table.setItem(i, 0, QTableWidgetItem(prop))
                table.setItem(i, 1, QTableWidgetItem(node_val))
                table.setItem(i, 2, QTableWidgetItem(parent_val))
            
            table.resizeColumnsToContents()
            layout.addWidget(table)
            
            improvement_group = QGroupBox("Improvement Metrics")
            improvement_layout = QFormLayout()
            
            node_samples = getattr(self.node, 'samples', 0)
            parent_samples = getattr(parent_node, 'samples', 0)
            sample_ratio = node_samples / parent_samples if parent_samples > 0 else 0
            
            node_impurity = getattr(self.node, 'impurity', 0)
            parent_impurity = getattr(parent_node, 'impurity', 0)
            impurity_improvement = parent_impurity - node_impurity
            
            improvement_layout.addRow("Sample Ratio:", QLabel(f"{sample_ratio:.3f} ({node_samples}/{parent_samples})"))
            improvement_layout.addRow("Impurity Improvement:", QLabel(f"{impurity_improvement:.4f}"))
            improvement_layout.addRow("Relative Impurity Reduction:", 
                                    QLabel(f"{(impurity_improvement/parent_impurity*100):.1f}%" if parent_impurity > 0 else "N/A"))
            
            improvement_group.setLayout(improvement_layout)
            layout.addWidget(improvement_group)
            
            close_button = QPushButton("Close")
            close_button.clicked.connect(comparison_dialog.accept)
            layout.addWidget(close_button)
            
            comparison_dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "Comparison Error", 
                               f"Error comparing with parent: {str(e)}")
            logger.error(f"Error in compareWithParent: {str(e)}", exc_info=True)
            
