#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Split Statistics Calculator for Bespoke Utility
Calculates various statistics for decision tree splits.

[SplitStatisticsCalculator.__init__ -> Initializes the SplitStatisticsCalculator -> dependent functions are None]
[SplitStatisticsCalculator.set_criterion -> Sets the impurity criterion -> dependent functions are None]
[SplitStatisticsCalculator.calculate_impurity -> Calculates impurity based on the selected criterion -> dependent functions are None]
[SplitStatisticsCalculator.calculate_split_quality -> Calculates comprehensive split quality metrics -> dependent functions are calculate_impurity]
[SplitStatisticsCalculator.evaluate_split_effectiveness -> Evaluates the effectiveness of a split and provides recommendations -> dependent functions are None]
[SplitStatisticsCalculator.calculate_feature_importance -> Calculates the feature importance contribution from this split -> dependent functions are calculate_split_quality]
[SplitStatisticsCalculator.suggest_optimal_bins -> Suggests an optimal bin configuration for maximum information gain -> dependent functions are calculate_split_quality, evaluate_split_effectiveness]
[SplitStatisticsCalculator.calculate_split_stability -> Calculates the stability of split quality across bootstrap samples -> dependent functions are calculate_split_quality]
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Literal
import pandas as pd
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SplitQualityMetrics:
    """Container for split quality metrics"""
    impurity_before: float
    impurity_after: float
    impurity_decrease: float
    gain_ratio: float
    weighted_impurity: float
    bin_metrics: List[Dict[str, Any]]


class SplitStatisticsCalculator:
    """Calculates split quality metrics and statistics"""
    
    def __init__(self, criterion: Literal['gini', 'entropy', 'log_loss'] = 'gini'):
        self.criterion = criterion
        
    def set_criterion(self, criterion: Literal['gini', 'entropy', 'log_loss']):
        """Set the impurity criterion"""
        self.criterion = criterion
        
    def calculate_impurity(self, target_data: pd.Series) -> float:
        """Calculate impurity based on selected criterion"""
        if len(target_data) == 0:
            return 0.0
            
        class_counts = target_data.value_counts()
        total = len(target_data)
        
        if total == 0 or len(class_counts) <= 1:
            return 0.0
            
        try:
            if self.criterion == "gini":
                impurity = 1.0 - sum((count / total) ** 2 for count in class_counts)
            elif self.criterion == "entropy":
                impurity = -sum((count / total) * np.log2(count / total + 1e-10) for count in class_counts)
            else:  # log_loss
                impurity = -sum((count / total) * np.log(count / total + 1e-10) for count in class_counts)
        except Exception as e:
            logger.error(f"Error calculating {self.criterion} impurity: {e}")
            impurity = 0.0
            
        return impurity
        
    def calculate_split_quality(self, 
                              feature_data: pd.Series, 
                              target_data: pd.Series,
                              bins: List[Tuple[float, float]],
                              feature_type: str = 'numeric') -> SplitQualityMetrics:
        """Calculate comprehensive split quality metrics"""
        
        if len(feature_data) != len(target_data):
            raise ValueError("Feature and target data must have same length")
            
        if len(bins) < 2:
            raise ValueError("At least 2 bins required for split quality calculation")
            
        current_impurity = self.calculate_impurity(target_data)
        
        total_samples = len(target_data)
        weighted_impurity = 0.0
        bin_metrics = []
        
        for i, (min_val, max_val) in enumerate(bins):
            if feature_type == 'numeric':
                if i == len(bins) - 1:  # Last bin includes max
                    mask = (feature_data >= min_val) & (feature_data <= max_val)
                else:
                    mask = (feature_data >= min_val) & (feature_data < max_val)
                range_text = f"[{min_val:.3f}, {max_val:.3f}]"
            else:
                mask = feature_data == min_val
                range_text = str(min_val)
                
            bin_target_data = target_data[mask]
            bin_count = len(bin_target_data)
            
            bin_impurity = self.calculate_impurity(bin_target_data)
            if total_samples > 0:
                weighted_impurity += (bin_count / total_samples) * bin_impurity
                
            if bin_count > 0:
                target_dist = bin_target_data.value_counts().to_dict()
                purity = 1.0 - bin_impurity
            else:
                target_dist = {}
                purity = 0.0
                
            bin_metric = {
                'bin_id': i,
                'range': range_text,
                'count': bin_count,
                'percentage': (100 * bin_count / total_samples) if total_samples > 0 else 0,
                'target_distribution': target_dist,
                'impurity': bin_impurity,
                'purity': purity,
                'is_empty': bin_count == 0,
                'is_small': bin_count < max(1, total_samples // 20)  # Less than 5% of data
            }
            
            bin_metrics.append(bin_metric)
            
        impurity_decrease = current_impurity - weighted_impurity
        gain_ratio = impurity_decrease / current_impurity if current_impurity > 0 else 0
        
        return SplitQualityMetrics(
            impurity_before=current_impurity,
            impurity_after=weighted_impurity,
            impurity_decrease=impurity_decrease,
            gain_ratio=gain_ratio,
            weighted_impurity=weighted_impurity,
            bin_metrics=bin_metrics
        )
        
    def evaluate_split_effectiveness(self, metrics: SplitQualityMetrics) -> Dict[str, Any]:
        """Evaluate the effectiveness of a split and provide recommendations"""
        evaluation = {
            'quality_score': 0.0,
            'recommendation': 'poor',
            'issues': [],
            'strengths': []
        }
        
        empty_bins = [m for m in metrics.bin_metrics if m['is_empty']]
        if empty_bins:
            evaluation['issues'].append(f"{len(empty_bins)} empty bins detected")
            
        small_bins = [m for m in metrics.bin_metrics if m['is_small'] and not m['is_empty']]
        if small_bins:
            evaluation['issues'].append(f"{len(small_bins)} bins with very few samples")
            
        if metrics.impurity_decrease <= 0:
            evaluation['issues'].append("No impurity improvement")
        elif metrics.impurity_decrease < 0.01:
            evaluation['issues'].append("Very small impurity improvement")
        else:
            evaluation['strengths'].append(f"Good impurity decrease: {metrics.impurity_decrease:.4f}")
            
        if metrics.gain_ratio < 0.1:
            evaluation['issues'].append("Low information gain ratio")
        elif metrics.gain_ratio > 0.3:
            evaluation['strengths'].append(f"High information gain: {metrics.gain_ratio:.4f}")
            
        bin_counts = [m['count'] for m in metrics.bin_metrics]
        if bin_counts:
            max_count = max(bin_counts)
            min_count = min([c for c in bin_counts if c > 0])  # Exclude empty bins
            if min_count > 0 and max_count / min_count > 10:
                evaluation['issues'].append("Highly imbalanced bins")
            elif min_count > 0 and max_count / min_count < 3:
                evaluation['strengths'].append("Well-balanced bins")
                
        base_score = min(metrics.gain_ratio * 100, 50)  # Max 50 points for gain ratio
        
        penalty = len(evaluation['issues']) * 10
        
        bonus = len(evaluation['strengths']) * 5
        
        evaluation['quality_score'] = max(0, base_score - penalty + bonus)
        
        if evaluation['quality_score'] >= 70:
            evaluation['recommendation'] = 'excellent'
        elif evaluation['quality_score'] >= 50:
            evaluation['recommendation'] = 'good'
        elif evaluation['quality_score'] >= 30:
            evaluation['recommendation'] = 'fair'
        elif evaluation['quality_score'] >= 10:
            evaluation['recommendation'] = 'poor'
        else:
            evaluation['recommendation'] = 'very_poor'
            
        return evaluation
        
    def calculate_feature_importance(self, 
                                   feature_data: pd.Series, 
                                   target_data: pd.Series,
                                   bins: List[Tuple[float, float]],
                                   total_samples: int) -> float:
        """Calculate feature importance contribution from this split"""
        try:
            metrics = self.calculate_split_quality(feature_data, target_data, bins)
            
            sample_weight = len(feature_data) / total_samples if total_samples > 0 else 0
            
            importance = metrics.impurity_decrease * sample_weight
            
            return importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return 0.0
            
    def suggest_optimal_bins(self, 
                           feature_data: pd.Series, 
                           target_data: pd.Series,
                           max_bins: int = 10,
                           min_samples_per_bin: int = 5) -> List[Tuple[float, float]]:
        """Suggest optimal bin configuration for maximum information gain"""
        
        if len(feature_data) != len(target_data):
            raise ValueError("Feature and target data must have same length")
            
        best_bins = []
        best_score = -1
        
        for num_bins in range(2, min(max_bins + 1, len(feature_data) // min_samples_per_bin + 1)):
            try:
                quantiles = np.linspace(0, 1, num_bins + 1)
                bin_edges = feature_data.quantile(quantiles).values
                bin_edges = np.unique(bin_edges)
                
                if len(bin_edges) < 2:
                    continue
                    
                bins = []
                for i in range(len(bin_edges) - 1):
                    bins.append((bin_edges[i], bin_edges[i + 1]))
                    
                metrics = self.calculate_split_quality(feature_data, target_data, bins)
                evaluation = self.evaluate_split_effectiveness(metrics)
                
                valid_bins = all(m['count'] >= min_samples_per_bin or m['count'] == 0 
                               for m in metrics.bin_metrics)
                
                if valid_bins and evaluation['quality_score'] > best_score:
                    best_score = evaluation['quality_score']
                    best_bins = bins.copy()
                    
            except Exception as e:
                logger.debug(f"Error testing {num_bins} bins: {e}")
                continue
                
        if not best_bins:
            median_val = feature_data.median()
            data_min = feature_data.min()
            data_max = feature_data.max()
            best_bins = [(data_min, median_val), (median_val, data_max)]
            
        return best_bins
        
    def calculate_split_stability(self, 
                                feature_data: pd.Series, 
                                target_data: pd.Series,
                                bins: List[Tuple[float, float]],
                                bootstrap_samples: int = 100) -> Dict[str, float]:
        """Calculate stability of split quality across bootstrap samples"""
        
        gain_ratios = []
        impurity_decreases = []
        
        for _ in range(bootstrap_samples):
            try:
                sample_indices = np.random.choice(len(feature_data), size=len(feature_data), replace=True)
                sample_feature = feature_data.iloc[sample_indices].reset_index(drop=True)
                sample_target = target_data.iloc[sample_indices].reset_index(drop=True)
                
                metrics = self.calculate_split_quality(sample_feature, sample_target, bins)
                gain_ratios.append(metrics.gain_ratio)
                impurity_decreases.append(metrics.impurity_decrease)
                
            except Exception as e:
                logger.debug(f"Bootstrap sample failed: {e}")
                continue
                
        if not gain_ratios:
            return {'stability_score': 0.0, 'gain_ratio_std': 0.0, 'impurity_decrease_std': 0.0}
            
        gain_ratio_std = np.std(gain_ratios)
        impurity_decrease_std = np.std(impurity_decreases)
        
        mean_gain_ratio = np.mean(gain_ratios)
        stability_score = 1.0 / (1.0 + gain_ratio_std) if mean_gain_ratio > 0 else 0.0
        
        return {
            'stability_score': stability_score,
            'gain_ratio_std': gain_ratio_std,
            'impurity_decrease_std': impurity_decrease_std,
            'mean_gain_ratio': mean_gain_ratio,
            'mean_impurity_decrease': np.mean(impurity_decreases)
        }