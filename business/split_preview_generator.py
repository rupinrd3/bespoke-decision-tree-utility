#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Split Preview Generator for Bespoke Utility
Generates previews of potential splits for decision tree nodes.

[SplitPreviewGenerator.__init__ -> Initializes the SplitPreviewGenerator -> dependent functions are None]
[SplitPreviewGenerator.generate_preview -> Generates a comprehensive split preview -> dependent functions are calculate_split_quality, evaluate_split_effectiveness, _generate_recommendations, _generate_warnings, _create_bin_details, _create_summary]
[SplitPreviewGenerator._generate_recommendations -> Generates actionable recommendations for the split -> dependent functions are None]
[SplitPreviewGenerator._generate_warnings -> Generates warnings about potential issues with the split -> dependent functions are None]
[SplitPreviewGenerator._create_bin_details -> Creates detailed bin information for UI display -> dependent functions are _calculate_bin_quality_score]
[SplitPreviewGenerator._calculate_bin_quality_score -> Calculates a quality score for an individual bin -> dependent functions are None]
[SplitPreviewGenerator._create_summary -> Creates summary information for the split preview -> dependent functions are None]
[SplitPreviewGenerator.generate_comparison_preview -> Generates a comparison between multiple split configurations -> dependent functions are _generate_comparison_recommendations, balance_score]
[SplitPreviewGenerator.balance_score -> Calculates a balance score for a given configuration -> dependent functions are None]
[SplitPreviewGenerator._generate_comparison_recommendations -> Generates recommendations for comparing multiple configurations -> dependent functions are None]
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass

from business.split_statistics_calculator import SplitStatisticsCalculator, SplitQualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class SplitPreview:
    """Container for split preview information"""
    quality_metrics: SplitQualityMetrics
    evaluation: Dict[str, Any]
    recommendations: List[str]
    warnings: List[str]
    bin_details: List[Dict[str, Any]]
    summary: Dict[str, Any]


class SplitPreviewGenerator:
    """Generates comprehensive preview information for split configurations"""
    
    def __init__(self, statistics_calculator: Optional[SplitStatisticsCalculator] = None):
        self.stats_calc = statistics_calculator or SplitStatisticsCalculator()
        
    def generate_preview(self, 
                        feature_data: pd.Series,
                        target_data: pd.Series,
                        bins: List[Tuple[float, float]],
                        feature_name: str,
                        feature_type: str = 'numeric') -> SplitPreview:
        """Generate comprehensive split preview"""
        
        if len(feature_data) != len(target_data):
            raise ValueError("Feature and target data must have same length")
            
        quality_metrics = self.stats_calc.calculate_split_quality(
            feature_data, target_data, bins, feature_type
        )
        
        evaluation = self.stats_calc.evaluate_split_effectiveness(quality_metrics)
        
        recommendations = self._generate_recommendations(quality_metrics, evaluation)
        warnings = self._generate_warnings(quality_metrics, evaluation)
        
        bin_details = self._create_bin_details(quality_metrics, feature_type)
        
        summary = self._create_summary(quality_metrics, evaluation, feature_name, len(bins))
        
        return SplitPreview(
            quality_metrics=quality_metrics,
            evaluation=evaluation,
            recommendations=recommendations,
            warnings=warnings,
            bin_details=bin_details,
            summary=summary
        )
        
    def _generate_recommendations(self, 
                                 metrics: SplitQualityMetrics, 
                                 evaluation: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        empty_bins = [m for m in metrics.bin_metrics if m['is_empty']]
        if empty_bins:
            empty_indices = [m['bin_id'] + 1 for m in empty_bins]  # 1-based for user
            recommendations.append(
                f"Remove empty bins: {', '.join(map(str, empty_indices))}"
            )
            
        small_bins = [m for m in metrics.bin_metrics if m['is_small'] and not m['is_empty']]
        if small_bins:
            recommendations.append(
                "Consider merging small bins with adjacent bins for better statistical power"
            )
            
        if metrics.gain_ratio < 0.1:
            recommendations.append(
                "Low information gain - consider choosing a different feature or split strategy"
            )
            
        bin_counts = [m['count'] for m in metrics.bin_metrics if m['count'] > 0]
        if bin_counts and max(bin_counts) / min(bin_counts) > 10:
            recommendations.append(
                "Highly imbalanced bins - consider equal frequency binning for better balance"
            )
            
        if evaluation['recommendation'] in ['good', 'excellent']:
            recommendations.append("Split configuration looks good - ready to apply")
        elif evaluation['recommendation'] == 'fair':
            recommendations.append("Split is acceptable but could be improved")
            
        num_bins = len(metrics.bin_metrics)
        total_samples = sum(m['count'] for m in metrics.bin_metrics)
        
        if num_bins > total_samples // 10:
            recommendations.append(
                f"Consider reducing number of bins - {num_bins} bins for {total_samples} samples may be too granular"
            )
        elif num_bins == 2 and metrics.gain_ratio > 0.3:
            recommendations.append(
                "Excellent binary split - this feature shows strong predictive power"
            )
            
        return recommendations
        
    def _generate_warnings(self, 
                          metrics: SplitQualityMetrics, 
                          evaluation: Dict[str, Any]) -> List[str]:
        """Generate warnings about potential issues"""
        warnings = []
        
        empty_bins = [m for m in metrics.bin_metrics if m['is_empty']]
        if empty_bins:
            warnings.append(
                f"⚠️ {len(empty_bins)} empty bins will not contribute to the split"
            )
            
        if metrics.impurity_decrease <= 0:
            warnings.append(
                "⚠️ No improvement in impurity - this split may not be beneficial"
            )
            
        very_small_bins = [m for m in metrics.bin_metrics if m['count'] < 3 and m['count'] > 0]
        if very_small_bins:
            warnings.append(
                f"⚠️ {len(very_small_bins)} bins have fewer than 3 samples - results may be unreliable"
            )
            
        if metrics.gain_ratio < 0.05:
            warnings.append(
                "⚠️ Very low information gain - split may not generalize well"
            )
            
        total_samples = sum(m['count'] for m in metrics.bin_metrics)
        largest_bin = max((m['count'] for m in metrics.bin_metrics), default=0)
        if largest_bin > total_samples * 0.8:
            warnings.append(
                "⚠️ One bin contains >80% of data - split may not be effective"
            )
            
        pure_bins = [m for m in metrics.bin_metrics if m['purity'] > 0.95 and m['count'] > 0]
        if len(pure_bins) == len(metrics.bin_metrics):
            warnings.append(
                "⚠️ All bins are nearly pure - may indicate overfitting"
            )
            
        return warnings
        
    def _create_bin_details(self, 
                           metrics: SplitQualityMetrics,
                           feature_type: str) -> List[Dict[str, Any]]:
        """Create detailed bin information for UI display"""
        bin_details = []
        
        total_samples = sum(m['count'] for m in metrics.bin_metrics)
        
        for i, bin_metric in enumerate(metrics.bin_metrics):
            status = 'normal'
            if bin_metric['is_empty']:
                status = 'empty'
            elif bin_metric['is_small']:
                status = 'small'
            elif bin_metric['purity'] > 0.9:
                status = 'pure'
                
            target_dist = bin_metric['target_distribution']
            if target_dist:
                dist_text = ', '.join([f"{k}: {v}" for k, v in target_dist.items()])
                most_common = max(target_dist.items(), key=lambda x: x[1])
                most_common_text = f"{most_common[0]} ({most_common[1]})"
            else:
                dist_text = "No data"
                most_common_text = "N/A"
                
            detail = {
                'bin_id': i,
                'bin_number': i + 1,  # 1-based for display
                'range_text': bin_metric['range'],
                'sample_count': bin_metric['count'],
                'percentage': bin_metric['percentage'],
                'target_distribution_text': dist_text,
                'most_common_class': most_common_text,
                'purity': bin_metric['purity'],
                'impurity': bin_metric['impurity'],
                'status': status,
                'is_empty': bin_metric['is_empty'],
                'is_small': bin_metric['is_small'],
                'contribution': (bin_metric['count'] / total_samples) if total_samples > 0 else 0,
                'quality_score': self._calculate_bin_quality_score(bin_metric)
            }
            
            bin_details.append(detail)
            
        return bin_details
        
    def _calculate_bin_quality_score(self, bin_metric: Dict[str, Any]) -> float:
        """Calculate a quality score for an individual bin"""
        if bin_metric['is_empty']:
            return 0.0
            
        score = 50.0  # Base score
        
        if bin_metric['is_small']:
            score -= 20.0
            
        if bin_metric['purity'] > 0.7:
            score += 20.0
        elif bin_metric['purity'] > 0.5:
            score += 10.0
            
        if bin_metric['purity'] < 0.1:
            score -= 15.0
            
        if bin_metric['count'] >= 10:
            score += 10.0
        elif bin_metric['count'] >= 5:
            score += 5.0
            
        return max(0.0, min(100.0, score))
        
    def _create_summary(self, 
                       metrics: SplitQualityMetrics,
                       evaluation: Dict[str, Any],
                       feature_name: str,
                       num_bins: int) -> Dict[str, Any]:
        """Create summary information"""
        
        total_samples = sum(m['count'] for m in metrics.bin_metrics)
        non_empty_bins = len([m for m in metrics.bin_metrics if not m['is_empty']])
        
        return {
            'feature_name': feature_name,
            'num_bins_total': num_bins,
            'num_bins_non_empty': non_empty_bins,
            'total_samples': total_samples,
            'impurity_before': metrics.impurity_before,
            'impurity_after': metrics.impurity_after,
            'impurity_decrease': metrics.impurity_decrease,
            'gain_ratio': metrics.gain_ratio,
            'quality_score': evaluation['quality_score'],
            'recommendation': evaluation['recommendation'],
            'is_beneficial': metrics.impurity_decrease > 0,
            'has_issues': len(evaluation['issues']) > 0,
            'has_warnings': len([m for m in metrics.bin_metrics if m['is_empty'] or m['is_small']]) > 0,
            'average_bin_purity': np.mean([m['purity'] for m in metrics.bin_metrics if not m['is_empty']]) if non_empty_bins > 0 else 0.0,
            'min_bin_size': min([m['count'] for m in metrics.bin_metrics if not m['is_empty']]) if non_empty_bins > 0 else 0,
            'max_bin_size': max([m['count'] for m in metrics.bin_metrics]) if metrics.bin_metrics else 0
        }
        
    def generate_comparison_preview(self, 
                                  previews: List[SplitPreview],
                                  preview_names: List[str]) -> Dict[str, Any]:
        """Generate comparison between multiple split configurations"""
        
        if len(previews) != len(preview_names):
            raise ValueError("Number of previews must match number of names")
            
        if not previews:
            return {}
            
        comparison = {
            'configurations': [],
            'best_overall': None,
            'best_gain_ratio': None,
            'best_balance': None,
            'recommendations': []
        }
        
        for i, (preview, name) in enumerate(zip(previews, preview_names)):
            config_summary = {
                'name': name,
                'index': i,
                'quality_score': preview.evaluation['quality_score'],
                'gain_ratio': preview.quality_metrics.gain_ratio,
                'impurity_decrease': preview.quality_metrics.impurity_decrease,
                'num_bins': len(preview.quality_metrics.bin_metrics),
                'num_empty_bins': len([m for m in preview.quality_metrics.bin_metrics if m['is_empty']]),
                'recommendation': preview.evaluation['recommendation'],
                'has_warnings': len(preview.warnings) > 0
            }
            comparison['configurations'].append(config_summary)
            
        if comparison['configurations']:
            best_overall = max(comparison['configurations'], key=lambda x: x['quality_score'])
            comparison['best_overall'] = best_overall['index']
            
            best_gain = max(comparison['configurations'], key=lambda x: x['gain_ratio'])
            comparison['best_gain_ratio'] = best_gain['index']
            
            def balance_score(config):
                return config['quality_score'] - (config['num_bins'] - 2) * 5  # Penalty for complexity
                
            best_balance = max(comparison['configurations'], key=balance_score)
            comparison['best_balance'] = best_balance['index']
            
        if len(comparison['configurations']) > 1:
            comparison['recommendations'] = self._generate_comparison_recommendations(comparison)
            
        return comparison
        
    def _generate_comparison_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate recommendations for comparing multiple configurations"""
        recommendations = []
        configs = comparison['configurations']
        
        if not configs:
            return recommendations
            
        best_quality = max(config['quality_score'] for config in configs)
        worst_quality = min(config['quality_score'] for config in configs)
        
        if best_quality - worst_quality > 30:
            best_config = configs[comparison['best_overall']]
            recommendations.append(
                f"Configuration '{best_config['name']}' has significantly better quality "
                f"(score: {best_config['quality_score']:.1f})"
            )
            
        min_bins = min(config['num_bins'] for config in configs)
        max_bins = max(config['num_bins'] for config in configs)
        
        if max_bins > min_bins * 2:
            simple_configs = [c for c in configs if c['num_bins'] == min_bins]
            if simple_configs and simple_configs[0]['quality_score'] > best_quality * 0.8:
                recommendations.append(
                    f"Consider simpler configuration with {min_bins} bins - "
                    f"only {best_quality - simple_configs[0]['quality_score']:.1f} points lower quality"
                )
                
        configs_with_empty = [c for c in configs if c['num_empty_bins'] > 0]
        if configs_with_empty:
            recommendations.append(
                f"{len(configs_with_empty)} configurations have empty bins - "
                "consider configurations without empty bins"
            )
            
        best_gain = max(config['gain_ratio'] for config in configs)
        if best_gain > 0.3:
            best_gain_config = configs[comparison['best_gain_ratio']]
            recommendations.append(
                f"Configuration '{best_gain_config['name']}' shows excellent information gain "
                f"({best_gain_config['gain_ratio']:.3f})"
            )
        elif best_gain < 0.1:
            recommendations.append(
                "All configurations show low information gain - "
                "consider a different feature or preprocessing"
            )
            
        return recommendations