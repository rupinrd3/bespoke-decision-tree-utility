#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Binary Target Validator for Bespoke Utility
Validates and ensures binary targets throughout the workflow
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BinaryTargetValidator:
    """Validator for binary classification target variables"""
    
    def __init__(self):
        self.common_binary_patterns = {
            'positive': {'yes', 'y', 'true', 't', '1', 1, True, 'positive', 'pos', 'high', 'good', 'success'},
            'negative': {'no', 'n', 'false', 'f', '0', 0, False, 'negative', 'neg', 'low', 'bad', 'failure'}
        }
        
    def validate_binary_target(self, data: pd.Series, column_name: str = None) -> Dict[str, Any]:
        """
        Validate if a column is suitable for binary classification
        
        Args:
            data: Pandas Series containing the target data
            column_name: Name of the column (for reporting)
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'is_valid': False,
            'unique_count': 0,
            'unique_values': [],
            'null_count': 0,
            'recommendations': [],
            'warnings': [],
            'mapping_suggestions': {},
            'confidence': 0.0
        }
        
        try:
            if data.empty:
                result['warnings'].append("Target column is empty")
                return result
                
            clean_data = data.dropna()
            if clean_data.empty:
                result['warnings'].append("Target column contains only null values")
                return result
                
            unique_values = clean_data.unique()
            result['unique_count'] = len(unique_values)
            result['unique_values'] = list(unique_values)
            result['null_count'] = data.isnull().sum()
            
            if result['unique_count'] == 2:
                result['is_valid'] = True
                result['confidence'] = 1.0
                result['recommendations'].append("Perfect binary target with exactly 2 unique values")
                
                value_counts = clean_data.value_counts()
                min_count = value_counts.min()
                max_count = value_counts.max()
                imbalance_ratio = min_count / max_count
                
                if imbalance_ratio < 0.1:
                    result['warnings'].append(f"Severe class imbalance detected (ratio: {imbalance_ratio:.3f})")
                elif imbalance_ratio < 0.3:
                    result['warnings'].append(f"Moderate class imbalance detected (ratio: {imbalance_ratio:.3f})")
                    
                values = list(unique_values)
                result['mapping_suggestions'] = self._suggest_binary_mapping(values)
                
            elif result['unique_count'] == 1:
                result['warnings'].append("Only one unique value - not suitable for classification")
                result['recommendations'].append("Need at least 2 different values for binary classification")
                
            elif result['unique_count'] <= 10:
                result['confidence'] = max(0.0, 1.0 - (result['unique_count'] - 2) * 0.2)
                result['recommendations'].append(f"Column has {result['unique_count']} unique values")
                result['recommendations'].append("Consider grouping values into 2 categories for binary classification")
                
                grouping_suggestions = self._suggest_groupings(unique_values)
                if grouping_suggestions:
                    result['mapping_suggestions'] = grouping_suggestions
                    result['recommendations'].append("See mapping suggestions for possible binary groupings")
                    
            else:
                result['warnings'].append(f"Too many unique values ({result['unique_count']}) for binary classification")
                result['recommendations'].append("Consider creating binary categories based on thresholds or business rules")
                
                if pd.api.types.is_numeric_dtype(data):
                    threshold_suggestions = self._suggest_numeric_thresholds(clean_data)
                    if threshold_suggestions:
                        result['mapping_suggestions'] = threshold_suggestions
                        result['recommendations'].append("See threshold suggestions for numeric binary splits")
                        
        except Exception as e:
            logger.error(f"Error validating binary target: {e}")
            result['warnings'].append(f"Validation error: {str(e)}")
            
        return result
        
    def _suggest_binary_mapping(self, values: List[Any]) -> Dict[str, Any]:
        """Suggest mapping for binary values to standard 0/1 encoding"""
        mapping = {}
        
        if len(values) != 2:
            return mapping
            
        val1, val2 = values[0], values[1]
        
        val1_str = str(val1).lower().strip()
        val2_str = str(val2).lower().strip()
        
        if val1_str in self.common_binary_patterns['positive']:
            mapping = {val1: 1, val2: 0}
        elif val2_str in self.common_binary_patterns['positive']:
            mapping = {val2: 1, val1: 0}
        elif val1_str in self.common_binary_patterns['negative']:
            mapping = {val1: 0, val2: 1}
        elif val2_str in self.common_binary_patterns['negative']:
            mapping = {val2: 0, val1: 1}
        else:
            mapping = {val1: 0, val2: 1}
            
        return {
            'binary_mapping': mapping,
            'description': f"Map {list(mapping.keys())} to {list(mapping.values())}"
        }
        
    def _suggest_groupings(self, values: List[Any]) -> Dict[str, Any]:
        """Suggest ways to group multiple values into binary categories"""
        suggestions = {}
        
        str_values = [str(v).lower().strip() for v in values]
        
        positive_matches = []
        negative_matches = []
        neutral_values = []
        
        for i, val_str in enumerate(str_values):
            if any(pattern in val_str for pattern in self.common_binary_patterns['positive']):
                positive_matches.append(values[i])
            elif any(pattern in val_str for pattern in self.common_binary_patterns['negative']):
                negative_matches.append(values[i])
            else:
                neutral_values.append(values[i])
                
        if positive_matches and negative_matches:
            suggestions['pattern_based'] = {
                'positive_group': positive_matches,
                'negative_group': negative_matches + neutral_values,
                'description': 'Group based on positive/negative patterns'
            }
            
        suggestions['manual_grouping'] = {
            'group_1': values[:len(values)//2],
            'group_2': values[len(values)//2:],
            'description': 'Manual grouping - review and adjust as needed'
        }
        
        return suggestions
        
    def _suggest_numeric_thresholds(self, data: pd.Series) -> Dict[str, Any]:
        """Suggest threshold-based binary splits for numeric data"""
        suggestions = {}
        
        try:
            median = data.median()
            mean = data.mean()
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            
            suggestions['statistical_thresholds'] = {
                'median': {
                    'threshold': median,
                    'description': f'Split at median ({median:.2f}): <= median vs > median'
                },
                'mean': {
                    'threshold': mean,
                    'description': f'Split at mean ({mean:.2f}): <= mean vs > mean'
                },
                'quartile': {
                    'threshold': q3,
                    'description': f'Split at 75th percentile ({q3:.2f}): <= Q3 vs > Q3'
                }
            }
            
            if data.max() - data.min() > 10:
                round_suggestions = []
                for threshold in [10, 50, 100, 500, 1000]:
                    if data.min() < threshold < data.max():
                        below_count = (data <= threshold).sum()
                        above_count = (data > threshold).sum()
                        total = len(data)
                        balance = min(below_count, above_count) / max(below_count, above_count)
                        
                        round_suggestions.append({
                            'threshold': threshold,
                            'below_count': below_count,
                            'above_count': above_count,
                            'balance_ratio': balance,
                            'description': f'Split at {threshold}: {below_count}/{above_count} (balance: {balance:.2f})'
                        })
                        
                if round_suggestions:
                    round_suggestions.sort(key=lambda x: abs(x['balance_ratio'] - 0.5))
                    suggestions['round_thresholds'] = round_suggestions[:3]  # Top 3
                    
        except Exception as e:
            logger.warning(f"Error generating numeric threshold suggestions: {e}")
            
        return suggestions
        
    def create_binary_target(self, data: pd.Series, mapping: Dict[Any, int]) -> pd.Series:
        """
        Create a binary target series using the provided mapping
        
        Args:
            data: Original target data
            mapping: Dictionary mapping original values to 0/1
            
        Returns:
            Binary target series
        """
        try:
            binary_target = data.map(mapping)
            
            unmapped_mask = binary_target.isnull() & data.notnull()
            if unmapped_mask.any():
                logger.warning(f"Found {unmapped_mask.sum()} values not in mapping, setting to NaN")
                
            return binary_target.astype('Int64')  # Nullable integer type
            
        except Exception as e:
            logger.error(f"Error creating binary target: {e}")
            raise
            
    def validate_workflow_target(self, dataframe: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Validate target column in the context of a workflow
        
        Args:
            dataframe: Full dataset
            target_column: Name of target column
            
        Returns:
            Validation results with workflow-specific recommendations
        """
        result = {
            'is_valid': False,
            'target_exists': False,
            'validation_results': {},
            'workflow_recommendations': []
        }
        
        if target_column not in dataframe.columns:
            result['workflow_recommendations'].append(f"Target column '{target_column}' not found in dataset")
            return result
            
        result['target_exists'] = True
        
        target_data = dataframe[target_column]
        validation = self.validate_binary_target(target_data, target_column)
        result['validation_results'] = validation
        
        if validation['is_valid']:
            result['is_valid'] = True
            result['workflow_recommendations'].append("Target is ready for binary classification")
            
            total_samples = len(dataframe)
            target_counts = target_data.value_counts()
            min_class_size = target_counts.min()
            
            if total_samples < 100:
                result['workflow_recommendations'].append("Small dataset - consider gathering more data")
            elif min_class_size < 10:
                result['workflow_recommendations'].append("Very small minority class - consider data augmentation")
                
        else:
            if validation['unique_count'] == 1:
                result['workflow_recommendations'].append("Cannot proceed with binary classification - need multiple target values")
            elif validation['unique_count'] > 2:
                result['workflow_recommendations'].append("Multi-class target detected - convert to binary or use multi-class methods")
            else:
                result['workflow_recommendations'].append("Target validation failed - review data quality")
                
        return result
        
    def get_binary_validation_summary(self, validation_result: Dict[str, Any]) -> str:
        """Generate a human-readable summary of validation results"""
        if not validation_result:
            return "No validation results available"
            
        summary_parts = []
        
        if validation_result.get('is_valid', False):
            summary_parts.append(f"✓ Valid binary target")
            if validation_result.get('confidence', 0) < 1.0:
                summary_parts.append(f"(confidence: {validation_result['confidence']:.1%})")
        else:
            summary_parts.append("⚠ Not a valid binary target")
            
        unique_count = validation_result.get('unique_count', 0)
        unique_values = validation_result.get('unique_values', [])
        
        if unique_count == 2:
            summary_parts.append(f"Values: {unique_values}")
        elif unique_count == 1:
            summary_parts.append(f"Only one value: {unique_values[0]}")
        elif unique_count > 2:
            summary_parts.append(f"{unique_count} unique values")
            
        null_count = validation_result.get('null_count', 0)
        if null_count > 0:
            summary_parts.append(f"{null_count} null values")
            
        warnings = validation_result.get('warnings', [])
        if warnings:
            summary_parts.extend([f"⚠ {warning}" for warning in warnings])
            
        return " | ".join(summary_parts)