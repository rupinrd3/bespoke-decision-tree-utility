#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bin Manager Component for Bespoke Utility
Manages bin configuration and validation - extracted from enhanced edit dialog
"""

import logging
from typing import List, Tuple, Dict, Optional, Any
from PyQt5.QtCore import QObject, pyqtSignal
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BinManager(QObject):
    """Manages bin configuration and validation"""
    
    binsChanged = pyqtSignal()
    validationError = pyqtSignal(str)
    binAdded = pyqtSignal(int, float, float)  # bin_id, min_val, max_val
    binRemoved = pyqtSignal(int)  # bin_id
    
    def __init__(self):
        super().__init__()
        self._bins: List[Tuple[float, float]] = []
        self._feature_type: Optional[str] = None
        self._feature_data: Optional[pd.Series] = None
        self._data_min: Optional[float] = None
        self._data_max: Optional[float] = None
        
    def set_feature_data(self, feature_data: pd.Series, feature_type: str = 'numeric'):
        """Set the feature data for validation"""
        self._feature_data = feature_data.dropna()
        self._feature_type = feature_type
        
        if feature_type == 'numeric' and len(self._feature_data) > 0:
            self._data_min = float(self._feature_data.min())
            self._data_max = float(self._feature_data.max())
        else:
            self._data_min = None
            self._data_max = None
            
        logger.debug(f"Set feature data: type={feature_type}, min={self._data_min}, max={self._data_max}")
        
    def add_bin(self, min_val: float, max_val: float) -> bool:
        """Add a bin with validation"""
        if not self._validate_bin(min_val, max_val):
            return False
            
        self._bins.append((min_val, max_val))
        bin_id = len(self._bins) - 1
        
        logger.debug(f"Added bin {bin_id}: [{min_val}, {max_val}]")
        self.binAdded.emit(bin_id, min_val, max_val)
        self.binsChanged.emit()
        return True
        
    def remove_bin(self, index: int) -> bool:
        """Remove a bin by index"""
        if not (0 <= index < len(self._bins)):
            self.validationError.emit(f"Invalid bin index: {index}")
            return False
            
        if len(self._bins) <= 2:
            self.validationError.emit("Cannot remove bin: at least 2 bins are required")
            return False
            
        removed_bin = self._bins.pop(index)
        logger.debug(f"Removed bin {index}: {removed_bin}")
        
        self.binRemoved.emit(index)
        self.binsChanged.emit()
        return True
        
    def update_bin(self, index: int, min_val: float, max_val: float) -> bool:
        """Update an existing bin"""
        if not (0 <= index < len(self._bins)):
            self.validationError.emit(f"Invalid bin index: {index}")
            return False
            
        old_bin = self._bins.pop(index)
        
        if self._validate_bin(min_val, max_val):
            self._bins.insert(index, (min_val, max_val))
            logger.debug(f"Updated bin {index}: {old_bin} -> [{min_val}, {max_val}]")
            self.binsChanged.emit()
            return True
        else:
            self._bins.insert(index, old_bin)
            return False
            
    def clear_bins(self):
        """Clear all bins"""
        self._bins.clear()
        logger.debug("Cleared all bins")
        self.binsChanged.emit()
        
    def get_bins(self) -> List[Tuple[float, float]]:
        """Get all bins"""
        return self._bins.copy()
        
    def get_bin_count(self) -> int:
        """Get number of bins"""
        return len(self._bins)
        
    def create_equal_width_bins(self, num_bins: int) -> bool:
        """Create equal width bins"""
        if self._data_min is None or self._data_max is None:
            self.validationError.emit("No feature data set for bin creation")
            return False
            
        if num_bins < 2:
            self.validationError.emit("At least 2 bins are required")
            return False
            
        self.clear_bins()
        
        data_min = self._data_min
        data_max = self._data_max
        
        if data_max == data_min:
            range_extend = max(0.001, abs(data_min) * 0.001)
            data_max = data_min + range_extend
            
        bin_width = (data_max - data_min) / num_bins
        
        for i in range(num_bins):
            bin_min = data_min + i * bin_width
            bin_max = data_min + (i + 1) * bin_width
            
            if i == num_bins - 1:  # Last bin includes max value
                bin_max = data_max
                
            self._bins.append((bin_min, bin_max))
            
        logger.info(f"Created {num_bins} equal width bins")
        self.binsChanged.emit()
        return True
        
    def create_equal_frequency_bins(self, num_bins: int) -> bool:
        """Create equal frequency bins"""
        if self._feature_data is None or len(self._feature_data) == 0:
            self.validationError.emit("No feature data set for bin creation")
            return False
            
        if num_bins < 2:
            self.validationError.emit("At least 2 bins are required")
            return False
            
        self.clear_bins()
        
        try:
            quantiles = np.linspace(0, 1, num_bins + 1)
            bin_edges = self._feature_data.quantile(quantiles).values
            
            unique_edges = []
            for edge in bin_edges:
                if not unique_edges or edge != unique_edges[-1]:
                    unique_edges.append(edge)
                    
            for i in range(len(unique_edges) - 1):
                self._bins.append((unique_edges[i], unique_edges[i + 1]))
                
            logger.info(f"Created {len(self._bins)} equal frequency bins")
            self.binsChanged.emit()
            return True
            
        except Exception as e:
            logger.error(f"Error creating equal frequency bins: {e}")
            self.validationError.emit(f"Failed to create equal frequency bins: {e}")
            return False
            
    def create_quantile_bins(self, num_bins: int) -> bool:
        """Create quantile-based bins"""
        if self._feature_data is None or len(self._feature_data) == 0:
            self.validationError.emit("No feature data set for bin creation")
            return False
            
        if num_bins < 2:
            self.validationError.emit("At least 2 bins are required")
            return False
            
        self.clear_bins()
        
        if num_bins == 2:
            quantiles = [0, 0.5, 1]
        elif num_bins == 3:
            quantiles = [0, 0.33, 0.67, 1]
        elif num_bins == 4:
            quantiles = [0, 0.25, 0.5, 0.75, 1]
        elif num_bins == 5:
            quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1]
        else:
            quantiles = np.linspace(0, 1, num_bins + 1)
            
        try:
            bin_edges = self._feature_data.quantile(quantiles).values
            bin_edges = np.unique(bin_edges)
            
            for i in range(len(bin_edges) - 1):
                self._bins.append((bin_edges[i], bin_edges[i + 1]))
                
            logger.info(f"Created {len(self._bins)} quantile bins")
            self.binsChanged.emit()
            return True
            
        except Exception as e:
            logger.error(f"Error creating quantile bins: {e}")
            self.validationError.emit(f"Failed to create quantile bins: {e}")
            return False
            
    def _validate_bin(self, min_val: float, max_val: float) -> bool:
        """Validate a bin doesn't overlap with existing bins"""
        if min_val >= max_val:
            self.validationError.emit(f"Min value ({min_val:.3f}) must be less than max value ({max_val:.3f})")
            return False
            
        for i, (existing_min, existing_max) in enumerate(self._bins):
            if min_val < existing_max and max_val > existing_min:
                self.validationError.emit(
                    f"Bin [{min_val:.3f}, {max_val:.3f}] overlaps with existing bin {i} "
                    f"[{existing_min:.3f}, {existing_max:.3f}]"
                )
                return False
                
        return True
        
    def validate_coverage(self) -> bool:
        """Validate bins cover the data range completely"""
        if not self._bins:
            self.validationError.emit("No bins defined")
            return False
            
        if self._data_min is None or self._data_max is None:
            if self._feature_type == 'categorical':
                return True
            self.validationError.emit("No data range defined for coverage validation")
            return False
            
        sorted_bins = sorted(self._bins)
        
        if sorted_bins[0][0] > self._data_min:
            self.validationError.emit(f"Bins don't cover minimum data value {self._data_min:.3f}")
            return False
            
        if sorted_bins[-1][1] < self._data_max:
            self.validationError.emit(f"Bins don't cover maximum data value {self._data_max:.3f}")
            return False
            
        for i in range(len(sorted_bins) - 1):
            gap = sorted_bins[i+1][0] - sorted_bins[i][1]
            if abs(gap) > 1e-10:  # Allow for floating point precision
                self.validationError.emit(
                    f"Gap between bins: {sorted_bins[i][1]:.3f} to {sorted_bins[i+1][0]:.3f}"
                )
                return False
                
        return True
        
    def get_bin_statistics(self, bin_index: int, target_data: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Get statistics for a specific bin"""
        if not (0 <= bin_index < len(self._bins)):
            return {}
            
        if self._feature_data is None:
            return {}
            
        min_val, max_val = self._bins[bin_index]
        
        if self._feature_type == 'numeric':
            if bin_index == len(self._bins) - 1:  # Last bin includes max
                mask = (self._feature_data >= min_val) & (self._feature_data <= max_val)
            else:
                mask = (self._feature_data >= min_val) & (self._feature_data < max_val)
        else:
            mask = self._feature_data == min_val
            
        sample_count = mask.sum()
        
        stats = {
            'count': sample_count,
            'percentage': (100.0 * sample_count / len(self._feature_data)) if len(self._feature_data) > 0 else 0.0,
            'range': f"[{min_val:.3f}, {max_val:.3f}]" if self._feature_type == 'numeric' else str(min_val)
        }
        
        if target_data is not None and len(target_data) == len(self._feature_data):
            bin_targets = target_data[mask]
            if len(bin_targets) > 0:
                target_dist = bin_targets.value_counts().to_dict()
                stats['target_distribution'] = target_dist
                
                total = len(bin_targets)
                if total > 0:
                    gini_impurity = 1.0 - sum((count / total) ** 2 for count in target_dist.values())
                    stats['purity'] = 1.0 - gini_impurity
                else:
                    stats['purity'] = 0.0
            else:
                stats['target_distribution'] = {}
                stats['purity'] = 0.0
        
        return stats
        
    def merge_small_bins(self, min_samples_threshold: int) -> int:
        """Merge bins with fewer samples than threshold. Returns number of merges performed."""
        if self._feature_data is None or len(self._bins) <= 2:
            return 0
            
        merged_count = 0
        i = 0
        
        while i < len(self._bins) - 1:
            stats = self.get_bin_statistics(i)
            sample_count = stats.get('count', 0)
            
            if sample_count < min_samples_threshold:
                min_val, _ = self._bins[i]
                _, max_val = self._bins[i + 1]
                
                self._bins.pop(i + 1)  # Remove second bin first
                self._bins[i] = (min_val, max_val)  # Update first bin
                
                merged_count += 1
                logger.debug(f"Merged bins {i} and {i+1}: [{min_val:.3f}, {max_val:.3f}]")
            else:
                i += 1
                
        if merged_count > 0:
            self.binsChanged.emit()
            
        return merged_count