#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory Management Utilities for Bespoke Utility
Provides functions to optimize memory usage for large datasets
"""

import logging
import gc
import os
import sys
import psutil
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def configure_memory_settings(config: Dict[str, Any] = None):
    """
    Configure memory settings for the application
    
    Args:
        config: Configuration dictionary with memory settings
    """
    if config is None:
        config = {}
    
    memory_config = config.get('memory', {})
    
    max_rows = memory_config.get('max_display_rows', 100)
    max_cols = memory_config.get('max_display_cols', 50)
    
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', max_cols)
    
    gc_threshold = memory_config.get('gc_threshold', 100)
    gc.set_threshold(gc_threshold)
    
    total_memory = psutil.virtual_memory().total / (1024 ** 3)  # GB
    logger.info(f"System memory: {total_memory:.2f} GB")
    logger.info(f"Memory settings configured: max_rows={max_rows}, max_cols={max_cols}, gc_threshold={gc_threshold}")

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize memory usage of a DataFrame by choosing appropriate data types
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Optimized DataFrame
    """
    start_mem = get_dataframe_memory_usage(df)
    logger.debug(f"DataFrame memory usage before optimization: {start_mem:.2f} MB")
    
    result = df.copy()
    
    for col in result.columns:
        if not pd.api.types.is_numeric_dtype(result[col].dtype):
            continue
        
        if pd.api.types.is_integer_dtype(result[col].dtype):
            has_null = result[col].isna().any()
            
            col_min = result[col].min()
            col_max = result[col].max()
            
            if has_null:
                if col_min >= 0:  # Unsigned
                    if col_max <= 255:
                        result[col] = result[col].astype(pd.UInt8Dtype())
                    elif col_max <= 65535:
                        result[col] = result[col].astype(pd.UInt16Dtype())
                    elif col_max <= 4294967295:
                        result[col] = result[col].astype(pd.UInt32Dtype())
                    else:
                        result[col] = result[col].astype(pd.UInt64Dtype())
                else:  # Signed
                    if col_min >= -128 and col_max <= 127:
                        result[col] = result[col].astype(pd.Int8Dtype())
                    elif col_min >= -32768 and col_max <= 32767:
                        result[col] = result[col].astype(pd.Int16Dtype())
                    elif col_min >= -2147483648 and col_max <= 2147483647:
                        result[col] = result[col].astype(pd.Int32Dtype())
                    else:
                        result[col] = result[col].astype(pd.Int64Dtype())
            else:
                if col_min >= 0:  # Unsigned
                    if col_max <= 255:
                        result[col] = result[col].astype(np.uint8)
                    elif col_max <= 65535:
                        result[col] = result[col].astype(np.uint16)
                    elif col_max <= 4294967295:
                        result[col] = result[col].astype(np.uint32)
                    else:
                        result[col] = result[col].astype(np.uint64)
                else:  # Signed
                    if col_min >= -128 and col_max <= 127:
                        result[col] = result[col].astype(np.int8)
                    elif col_min >= -32768 and col_max <= 32767:
                        result[col] = result[col].astype(np.int16)
                    elif col_min >= -2147483648 and col_max <= 2147483647:
                        result[col] = result[col].astype(np.int32)
                    else:
                        result[col] = result[col].astype(np.int64)
        
        elif pd.api.types.is_float_dtype(result[col].dtype):
            if np.finfo(np.float32).min <= result[col].min() and result[col].max() <= np.finfo(np.float32).max:
                result[col] = result[col].astype(np.float32)
    
    for col in result.columns:
        if pd.api.types.is_object_dtype(result[col].dtype):
            num_unique = result[col].nunique()
            num_total = len(result[col])
            
            if num_unique > 0 and num_unique < num_total * 0.5 and num_unique >= 10:
                result[col] = result[col].astype('category')
    
    for col in result.columns:
        if pd.api.types.is_datetime64_dtype(result[col].dtype):
            min_date = result[col].min()
            max_date = result[col].max()
            
            if min_date and max_date:
                date_range_years = (max_date - min_date).days / 365.25
                if date_range_years < 50:  # If dates span less than 50 years
                    result[col] = pd.to_datetime(result[col], format='%Y-%m-%d')
    
    end_mem = get_dataframe_memory_usage(result)
    reduction = (start_mem - end_mem) / start_mem * 100
    
    logger.info(f"DataFrame optimized: {start_mem:.2f} MB → {end_mem:.2f} MB ({reduction:.1f}% reduction)")
    
    return result

def get_dataframe_memory_usage(df: pd.DataFrame) -> float:
    """
    Calculate memory usage of a DataFrame in MB
    
    Args:
        df: DataFrame to measure
        
    Returns:
        Memory usage in megabytes
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / (1024 * 1024)  # Convert to MB
    return memory_mb

def get_system_memory_info() -> Dict[str, float]:
    """
    Get system memory information
    
    Returns:
        Dictionary with memory information in GB:
            - total: Total physical memory
            - available: Available memory
            - used: Used memory
            - percent: Percentage of memory used
    """
    memory = psutil.virtual_memory()
    
    return {
        'total': memory.total / (1024 ** 3),
        'available': memory.available / (1024 ** 3),
        'used': memory.used / (1024 ** 3),
        'percent': memory.percent
    }

def monitor_memory_usage(threshold_percent: float = 80.0) -> bool:
    """
    Monitor memory usage and log warnings if it exceeds the threshold
    
    Args:
        threshold_percent: Percentage threshold to trigger warning
        
    Returns:
        True if memory usage is below threshold, False otherwise
    """
    memory_info = get_system_memory_info()
    
    if memory_info['percent'] > threshold_percent:
        logger.warning(f"High memory usage: {memory_info['percent']:.1f}% "
                      f"({memory_info['used']:.1f} GB / {memory_info['total']:.1f} GB)")
        
        process_info = get_top_memory_processes(5)
        logger.info(f"Top memory-consuming processes: {process_info}")
        
        logger.info("Suggesting garbage collection...")
        gc.collect()
        
        return False
    
    return True

def get_top_memory_processes(n: int = 5) -> Dict[str, float]:
    """
    Get the top N processes consuming the most memory
    
    Args:
        n: Number of processes to return
        
    Returns:
        Dictionary mapping process names to memory usage in MB
    """
    processes = {}
    
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            proc_info = proc.info
            memory_mb = proc_info['memory_info'].rss / (1024 * 1024)
            
            processes[f"{proc_info['name']} (PID: {proc_info['pid']})"] = memory_mb
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    top_processes = dict(sorted(processes.items(), key=lambda x: x[1], reverse=True)[:n])
    
    return top_processes

def clear_memory(deep_clean: bool = False) -> float:
    """
    Clear memory by forcing garbage collection
    
    Args:
        deep_clean: Whether to perform a more aggressive memory cleanup
        
    Returns:
        Amount of memory freed in MB
    """
    before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    
    gc.collect()
    
    if deep_clean:
        for _ in range(3):
            gc.collect()
        
        try:
            if hasattr(pd.core.common, '_possibly_cast_to_datetime') and hasattr(pd.core.common._possibly_cast_to_datetime, 'cache_clear'):
                pd.core.common._possibly_cast_to_datetime.cache_clear()
        except (AttributeError, TypeError) as e:
            logger.debug(f"Could not clear pandas cache: {e}")
    
    after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    
    freed_mb = before - after
    
    logger.info(f"Memory cleanup: {before:.1f} MB → {after:.1f} MB (Freed: {freed_mb:.1f} MB)")
    
    return freed_mb

def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 50000) -> list:
    """
    Split a DataFrame into chunks for processing large datasets
    
    Args:
        df: DataFrame to split
        chunk_size: Number of rows per chunk
        
    Returns:
        List of DataFrame chunks
    """
    num_chunks = (len(df) + chunk_size - 1) // chunk_size  # Ceiling division
    
    logger.info(f"Splitting DataFrame with {len(df)} rows into {num_chunks} chunks")
    
    return [df.iloc[i * chunk_size:(i + 1) * chunk_size].copy() for i in range(num_chunks)]
