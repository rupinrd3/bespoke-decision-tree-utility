#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Split Configuration Module for Bespoke Utility
Standardized split configuration classes to prevent data corruption and inconsistencies
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Literal, Any
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation operations"""
    
    def __init__(self, is_valid: bool = True):
        self.is_valid = is_valid
        self.errors = []
        self.warnings = []
        
    def add_error(self, error: str):
        """Add an error and mark as invalid"""
        self.errors.append(error)
        self.is_valid = False
        
    def add_warning(self, warning: str):
        """Add a warning"""
        self.warnings.append(warning)


@dataclass
class SplitConfiguration(ABC):
    """Abstract base class for split configurations"""
    feature: str
    split_type: Literal['numeric', 'categorical']
    
    @abstractmethod
    def validate(self, data: Optional[pd.DataFrame] = None) -> ValidationResult:
        """Validate the split configuration"""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for apply_manual_split"""
        pass
    
    @abstractmethod
    def get_child_assignment(self, values: pd.Series) -> pd.Series:
        """Get child node assignment for each value"""
        pass


@dataclass
class NumericSplit(SplitConfiguration):
    """Configuration for numeric splits"""
    split_type: Literal['numeric'] = 'numeric'
    threshold: float = 0.0
    operator: Literal['<=', '<', '>=', '>'] = '<='
    
    def validate(self, data: Optional[pd.DataFrame] = None) -> ValidationResult:
        """Validate numeric split configuration"""
        result = ValidationResult(is_valid=True)
        
        if not self.feature:
            result.add_error("Feature name cannot be empty")
            
        if not isinstance(self.threshold, (int, float)):
            result.add_error(f"Threshold must be numeric, got {type(self.threshold)}")
        elif math.isnan(self.threshold):
            result.add_error("Threshold cannot be NaN")
        elif math.isinf(self.threshold):
            result.add_error("Threshold cannot be infinite")
        elif abs(self.threshold) > 1e15:
            result.add_warning(f"Threshold value {self.threshold} is extremely large and may cause numerical issues")
            
        if data is not None and self.feature in data.columns:
            feature_data = data[self.feature]
            
            if not pd.api.types.is_numeric_dtype(feature_data):
                result.add_error(f"Feature '{self.feature}' is not numeric")
            else:
                clean_data = feature_data.dropna()
                if len(clean_data) > 0:
                    data_min, data_max = clean_data.min(), clean_data.max()
                    
                    if self.threshold < data_min:
                        result.add_warning(f"Threshold {self.threshold} is below minimum data value {data_min}")
                    elif self.threshold > data_max:
                        result.add_warning(f"Threshold {self.threshold} is above maximum data value {data_max}")
                    
                    if data_min == data_max:
                        result.add_warning(f"Feature '{self.feature}' has constant value {data_min}")
                        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'feature': self.feature,
            'split_type': self.split_type,
            'split_value': self.threshold,
            'split_operator': self.operator
        }
    
    def get_child_assignment(self, values: pd.Series) -> pd.Series:
        """Get child node assignment (0=left, 1=right) for each value"""
        if self.operator in ['<=', '<']:
            return (values > self.threshold).astype(int)
        else:  # '>=' or '>'
            return (values < self.threshold).astype(int)


@dataclass
class NumericMultiBinSplit(SplitConfiguration):
    """Configuration for multi-bin numeric splits"""
    split_type: Literal['numeric_multi_bin'] = 'numeric_multi_bin'
    thresholds: List[float] = field(default_factory=list)
    num_bins: int = 2
    
    def validate(self, data: Optional[pd.DataFrame] = None) -> ValidationResult:
        """Validate multi-bin numeric split configuration"""
        result = ValidationResult(is_valid=True)
        
        if not self.feature:
            result.add_error("Feature name cannot be empty")
            
        if not self.thresholds:
            result.add_error("Multi-bin split requires at least one threshold")
        elif len(self.thresholds) < 1:
            result.add_error("Multi-bin split requires at least one threshold")
        elif len(self.thresholds) != self.num_bins - 1:
            result.add_warning(f"Number of thresholds ({len(self.thresholds)}) should be num_bins - 1 ({self.num_bins - 1})")
            
        for i, threshold in enumerate(self.thresholds):
            if not isinstance(threshold, (int, float)):
                result.add_error(f"Threshold {i} must be numeric, got {type(threshold)}")
            elif math.isnan(threshold):
                result.add_error(f"Threshold {i} cannot be NaN")
            elif math.isinf(threshold):
                result.add_error(f"Threshold {i} cannot be infinite")
                
        if len(self.thresholds) > 1:
            for i in range(1, len(self.thresholds)):
                if self.thresholds[i] <= self.thresholds[i-1]:
                    result.add_error(f"Thresholds must be in ascending order: {self.thresholds[i-1]} >= {self.thresholds[i]}")
                    
        if data is not None and self.feature in data.columns:
            feature_data = data[self.feature]
            
            if not pd.api.types.is_numeric_dtype(feature_data):
                result.add_error(f"Feature '{self.feature}' is not numeric")
            else:
                min_val, max_val = feature_data.min(), feature_data.max()
                for i, threshold in enumerate(self.thresholds):
                    if threshold < min_val or threshold > max_val:
                        result.add_warning(f"Threshold {i} ({threshold}) is outside data range [{min_val}, {max_val}]")
                        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'feature': self.feature,
            'split_type': self.split_type,
            'thresholds': self.thresholds,
            'num_bins': self.num_bins
        }
    
    def get_child_assignment(self, values: pd.Series) -> pd.Series:
        """Get child node assignment (0, 1, 2, ...) for each value based on thresholds"""
        assignments = pd.Series(0, index=values.index, dtype=int)
        
        for i, threshold in enumerate(self.thresholds):
            assignments[values >= threshold] = i + 1
            
        return assignments


@dataclass  
class CategoricalSplit(SplitConfiguration):
    """Configuration for categorical splits"""
    split_type: Literal['categorical'] = 'categorical'
    category_mapping: Dict[str, int] = field(default_factory=dict)
    left_categories: List[str] = field(default_factory=list)
    right_categories: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize category mapping from left/right categories if needed"""
        if not self.category_mapping and (self.left_categories or self.right_categories):
            self.category_mapping = {}
            for cat in self.left_categories:
                self.category_mapping[cat] = 0
            for cat in self.right_categories:
                self.category_mapping[cat] = 1
        
        elif self.category_mapping and not (self.left_categories or self.right_categories):
            bins = {}
            for cat, bin_idx in self.category_mapping.items():
                if bin_idx not in bins:
                    bins[bin_idx] = []
                bins[bin_idx].append(cat)
            
            if len(bins) == 2 and 0 in bins and 1 in bins:
                self.left_categories = bins[0]
                self.right_categories = bins[1]
    
    def validate(self, data: Optional[pd.DataFrame] = None) -> ValidationResult:
        """Validate categorical split configuration"""
        result = ValidationResult(is_valid=True)
        
        if not self.feature:
            result.add_error("Feature name cannot be empty")
            
        if not self.category_mapping and not (self.left_categories or self.right_categories):
            result.add_error("Either category_mapping or left_categories/right_categories must be provided")
            
        if self.left_categories and self.right_categories:
            left_set = set(self.left_categories)
            right_set = set(self.right_categories)
            overlap = left_set & right_set
            if overlap:
                result.add_error(f"Categories appear in both left and right: {overlap}")
                
        if self.category_mapping:
            mapped_categories = set(self.category_mapping.keys())
            if self.left_categories:
                left_set = set(self.left_categories)
                if not left_set.issubset(mapped_categories):
                    result.add_error("Some left_categories not in category_mapping")
            if self.right_categories:
                right_set = set(self.right_categories)
                if not right_set.issubset(mapped_categories):
                    result.add_error("Some right_categories not in category_mapping")
                    
        if data is not None and self.feature in data.columns:
            feature_data = data[self.feature]
            
            if pd.api.types.is_numeric_dtype(feature_data) and not isinstance(feature_data.dtype, pd.CategoricalDtype):
                result.add_warning(f"Feature '{self.feature}' appears to be numeric, not categorical")
                
            unique_values = set(feature_data.dropna().unique())
            mapped_categories = set(self.category_mapping.keys())
            
            missing_in_data = mapped_categories - unique_values
            if missing_in_data:
                result.add_warning(f"Categories in mapping not found in data: {missing_in_data}")
                
            missing_in_mapping = unique_values - mapped_categories
            if missing_in_mapping:
                result.add_warning(f"Data categories not in mapping: {missing_in_mapping}")
                
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        if len(set(self.category_mapping.values())) == 2:
            left_cats = [cat for cat, idx in self.category_mapping.items() if idx == 0]
            right_cats = [cat for cat, idx in self.category_mapping.items() if idx == 1]
            return {
                'feature': self.feature,
                'split_type': self.split_type,
                'left_categories': left_cats,
                'right_categories': right_cats
            }
        else:
            return {
                'feature': self.feature,
                'split_type': self.split_type,
                'split_categories': self.category_mapping
            }
    
    def get_child_assignment(self, values: pd.Series) -> pd.Series:
        """Get child node assignment for each value"""
        return values.map(self.category_mapping).fillna(-1)  # -1 for unmapped categories


class SplitConfigurationFactory:
    """Factory for creating split configurations from various formats"""
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> SplitConfiguration:
        """Create split configuration from dictionary with type normalization"""
        split_type = config_dict.get('split_type', config_dict.get('type'))
        feature = config_dict.get('feature')
        
        if not feature:
            raise ValueError("Feature name is required")
        
        if split_type in ['numeric', 'numerical', 'numeric_multi_bin', 'numerical_multi_bin']:
            if split_type in ['numeric_multi_bin', 'numerical_multi_bin']:
                split_type = 'numeric'
            
            threshold = config_dict.get('split_value', config_dict.get('threshold'))
            if threshold is None:
                raise ValueError("Numerical split requires threshold/split_value")
                
            operator = config_dict.get('split_operator', '<=')
            return NumericSplit(
                feature=feature,
                threshold=float(threshold),
                operator=operator
            )
            
        elif split_type == 'categorical':
            category_mapping = config_dict.get('split_categories', {})
            left_categories = config_dict.get('left_categories', [])
            right_categories = config_dict.get('right_categories', [])
            
            return CategoricalSplit(
                feature=feature,
                category_mapping=category_mapping,
                left_categories=left_categories,
                right_categories=right_categories
            )
        else:
            raise ValueError(f"Invalid split_type: {split_type}. Must be 'numeric'/'numerical'/'numeric_multi_bin' or 'categorical'")
    
    @staticmethod
    def from_enhanced_dialog_format(config_dict: Dict[str, Any]) -> SplitConfiguration:
        """Create split configuration from enhanced dialog format"""
        config_type = config_dict.get('type')
        feature = config_dict.get('feature')
        
        if config_type == 'numeric_binary':
            return NumericSplit(
                feature=feature,
                threshold=config_dict['split_value'],
                operator='<='
            )
        elif config_type == 'categorical_binary':
            return CategoricalSplit(
                feature=feature,
                left_categories=config_dict['left_categories'],
                right_categories=config_dict['right_categories']
            )
        elif config_type == 'categorical_multi_bin':
            return CategoricalSplit(
                feature=feature,
                category_mapping=config_dict['split_categories']
            )
        elif config_type == 'numeric_multi_bin':
            thresholds = config_dict.get('thresholds', [])
            if not thresholds:
                raise ValueError("Multi-bin numeric split requires thresholds array")
            return NumericMultiBinSplit(
                feature=feature,
                thresholds=thresholds,
                num_bins=config_dict.get('num_bins', len(thresholds) + 1)
            )
        else:
            raise ValueError(f"Unsupported enhanced dialog format type: {config_type}")


def validate_split_consistency(parent_config: SplitConfiguration, 
                             child_configs: List[SplitConfiguration],
                             data: pd.DataFrame) -> ValidationResult:
    """Validate consistency between parent and child splits"""
    result = ValidationResult(is_valid=True)
    
    try:
        if parent_config.feature not in data.columns:
            result.add_error(f"Parent feature '{parent_config.feature}' not found in data")
            return result
            
        parent_assignments = parent_config.get_child_assignment(data[parent_config.feature])
        unique_assignments = set(parent_assignments.dropna().unique())
        
        if len(child_configs) != len(unique_assignments):
            result.add_error(f"Number of child configs ({len(child_configs)}) doesn't match parent assignments ({len(unique_assignments)})")
        
        total_samples = 0
        for child_config in child_configs:
            if child_config.feature in data.columns:
                child_data = data[data[parent_config.feature].map(
                    lambda x: parent_config.get_child_assignment(pd.Series([x])).iloc[0]
                ) == 0]  # Just check first child for now
                total_samples += len(child_data)
                
        if total_samples > len(data):
            result.add_error("Data leakage detected: child nodes contain more samples than parent")
            
    except Exception as e:
        result.add_error(f"Error validating split consistency: {e}")
        
    return result