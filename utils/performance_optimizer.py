#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Performance Optimization Utilities for Split Operations
"""

import logging
import threading
import time
from typing import Dict, Any, Optional, Callable
from functools import lru_cache
import weakref
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SplitQualityCache:
    """Thread-safe cache for split quality calculations"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.RLock()
        self.access_times = {}
        self.creation_times = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if available and not expired"""
        with self.lock:
            if key in self.cache:
                if time.time() - self.creation_times.get(key, 0) < self.ttl_seconds:
                    self.access_times[key] = time.time()
                    return self.cache[key]
                else:
                    self._remove_key(key)
            return None
            
    def set(self, key: str, value: Any):
        """Set cached value"""
        with self.lock:
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            current_time = time.time()
            self.cache[key] = value
            self.access_times[key] = current_time
            self.creation_times[key] = current_time
            
    def _remove_key(self, key: str):
        """Remove key from all tracking dicts"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.creation_times.pop(key, None)
        
    def _evict_oldest(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
            
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_key(oldest_key)
        
    def clear(self):
        """Clear all cached items"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.creation_times.clear()
            
    def size(self) -> int:
        """Get current cache size"""
        with self.lock:
            return len(self.cache)


class BinStatisticsCalculator:
    """Optimized bin statistics calculator with caching"""
    
    def __init__(self, cache_size: int = 500):
        self.cache = SplitQualityCache(max_size=cache_size, ttl_seconds=60)
        
    def calculate_bin_statistics(
        self, 
        data: pd.Series, 
        target: pd.Series, 
        bins: list, 
        is_categorical: bool = False
    ) -> list:
        """Calculate statistics for all bins efficiently"""
        
        data_hash = hash(tuple(data.values)) if len(data) < 1000 else hash(str(data.shape) + str(data.iloc[0]))
        target_hash = hash(tuple(target.values)) if len(target) < 1000 else hash(str(target.shape))
        bins_hash = hash(str(bins))
        cache_key = f"bin_stats_{data_hash}_{target_hash}_{bins_hash}_{is_categorical}"
        
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
            
        if is_categorical:
            result = self._calculate_categorical_bin_stats(data, target, bins)
        else:
            result = self._calculate_numerical_bin_stats(data, target, bins)
            
        self.cache.set(cache_key, result)
        return result
        
    def _calculate_numerical_bin_stats(self, data: pd.Series, target: pd.Series, bins: list) -> list:
        """Calculate statistics for numerical bins - FIXED to handle NaN values like synchronous version"""
        bin_stats = []
        
        total_count = len(data)
        missing_mask = data.isna()
        missing_count = missing_mask.sum()
        valid_data_mask = ~missing_mask
        
        bin_counts = []
        for i, (min_val, max_val) in enumerate(bins):
            try:
                if i == len(bins) - 1:  # Last bin includes upper bound
                    mask = valid_data_mask & (data >= min_val) & (data <= max_val)
                else:
                    mask = valid_data_mask & (data >= min_val) & (data < max_val)
                
                bin_data = data[mask]
                bin_target = target[mask]
                count = mask.sum()
                bin_counts.append(count)
                
                stats = {
                    'bin_id': i,
                    'range': (min_val, max_val),
                    'count': count,  # Will be updated after missing value assignment
                    'percentage': 0,  # Will be calculated after missing value assignment
                    'target_distribution': bin_target.value_counts().to_dict() if len(bin_target) > 0 else {},
                    'mean_target': bin_target.mean() if len(bin_target) > 0 else 0,
                    'gini_impurity': self._calculate_gini_impurity(bin_target)
                }
                
                bin_stats.append(stats)
                
            except Exception as e:
                logger.error(f"Error calculating stats for bin {i}: {e}")
                bin_stats.append({
                    'bin_id': i,
                    'range': (min_val, max_val),
                    'count': 0,
                    'percentage': 0,
                    'target_distribution': {},
                    'mean_target': 0,
                    'gini_impurity': 0.5,
                    'error': str(e)
                })
        
        if missing_count > 0 and bin_counts:
            largest_bin_idx = bin_counts.index(max(bin_counts))
            bin_counts[largest_bin_idx] += missing_count
            
            if largest_bin_idx < len(bin_stats):
                bin_stats[largest_bin_idx]['count'] = bin_counts[largest_bin_idx]
        
        for i, stats in enumerate(bin_stats):
            if 'error' not in stats:
                final_count = bin_counts[i] if i < len(bin_counts) else stats['count']
                stats['count'] = final_count
                stats['percentage'] = (final_count / total_count) * 100 if total_count > 0 else 0
                
        return bin_stats
        
    def _calculate_categorical_bin_stats(self, data: pd.Series, target: pd.Series, bins: list) -> list:
        """Calculate statistics for categorical bins"""
        bin_stats = []
        
        for i, categories in enumerate(bins):
            try:
                if isinstance(categories, dict):
                    categories_list = categories.get('categories', [])
                elif isinstance(categories, (list, tuple)):
                    categories_list = list(categories)
                else:
                    categories_list = [categories]
                
                mask = data.isin(categories_list)
                bin_data = data[mask]
                bin_target = target[mask]
                
                stats = {
                    'bin_id': i,
                    'categories': categories_list,
                    'count': len(bin_data),
                    'percentage': len(bin_data) / len(data) * 100 if len(data) > 0 else 0,
                    'target_distribution': bin_target.value_counts().to_dict() if len(bin_target) > 0 else {},
                    'mean_target': bin_target.mean() if len(bin_target) > 0 else 0,
                    'gini_impurity': self._calculate_gini_impurity(bin_target)
                }
                
                bin_stats.append(stats)
                
            except Exception as e:
                logger.error(f"Error calculating stats for categorical bin {i}: {e}")
                bin_stats.append({
                    'bin_id': i,
                    'categories': categories_list if 'categories_list' in locals() else [],
                    'count': 0,
                    'percentage': 0,
                    'target_distribution': {},
                    'mean_target': 0,
                    'gini_impurity': 0.5,
                    'error': str(e)
                })
                
        return bin_stats
        
    def _calculate_gini_impurity(self, target_data: pd.Series) -> float:
        """Calculate Gini impurity for target data"""
        if len(target_data) == 0:
            return 0.5  # Maximum impurity for binary case
            
        value_counts = target_data.value_counts()
        total = len(target_data)
        
        gini = 1.0
        for count in value_counts:
            probability = count / total
            gini -= probability ** 2
            
        return gini


class BackgroundStatsCalculator:
    """Background thread for calculating statistics without blocking UI"""
    
    def __init__(self):
        self.calculator = BinStatisticsCalculator()
        self.pending_requests = {}
        self.lock = threading.Lock()
        
    def calculate_async(
        self, 
        request_id: str,
        data: pd.Series, 
        target: pd.Series, 
        bins: list, 
        callback: Callable,
        is_categorical: bool = False
    ):
        """Schedule asynchronous calculation"""
        
        def worker():
            try:
                result = self.calculator.calculate_bin_statistics(
                    data, target, bins, is_categorical
                )
                
                if callback:
                    callback(request_id, result, None)
                    
            except Exception as e:
                logger.error(f"Error in background calculation: {e}")
                if callback:
                    callback(request_id, None, str(e))
            finally:
                with self.lock:
                    self.pending_requests.pop(request_id, None)
        
        with self.lock:
            if request_id in self.pending_requests:
                pass
                
            thread = threading.Thread(target=worker, daemon=True)
            self.pending_requests[request_id] = thread
            thread.start()


_split_quality_cache = None
_background_calculator = None


def get_split_quality_cache() -> SplitQualityCache:
    """Get global split quality cache instance"""
    global _split_quality_cache
    if _split_quality_cache is None:
        _split_quality_cache = SplitQualityCache()
    return _split_quality_cache


def get_background_calculator() -> BackgroundStatsCalculator:
    """Get global background calculator instance"""
    global _background_calculator
    if _background_calculator is None:
        _background_calculator = BackgroundStatsCalculator()
    return _background_calculator