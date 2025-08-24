#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration Module for Bespoke Utility
Handles loading, validating, and saving application configuration
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "application": {
        "name": "Bespoke Decision Tree Utility",
        "version": "1.0.0",
        "portable_mode": True,
        "temp_dir": "temp",
        "log_dir": "logs",
        "data_dir": "data"
    },
    
    "memory": {
        "max_display_rows": 100,
        "max_display_cols": 50,
        "gc_threshold": 100,
        "low_memory_mode": False,
        "memory_warning_threshold": 80.0,  # Percentage of memory usage to trigger warning
        "chunk_size": 50000  # Default chunk size for processing large datasets
    },
    
    "ui": {
        "theme": "light",
        "window_width": 1200,
        "window_height": 800,
        "font_size": 10,
        "show_toolbar": True,
        "show_statusbar": True,
        "auto_save": True,
        "confirmation_dialogs": True,
        "decimal_places": 4,
        "tree_visualization": {
            "node_size": 100,
            "level_distance": 150,
            "sibling_distance": 50,
            "orientation": "top_down",  # or "left_right"
            "show_node_ids": False,
            "show_samples": True,
            "show_percentages": True,
            "show_importance": True
        }
    },
    
    "data_loader": {
        "default_encoding": "utf-8",
        "default_delimiter": ",",
        "chunk_size": 100000,
        "max_rows_preview": 1000,
        "detect_encoding": True,
        "detect_delimiter": True,
        "max_file_size_mb": 800,  # Maximum file size in MB
        "sample_size": 5000  # Rows to sample for data type inference
    },
    
    "missing_values": {
        "strategy": "impute",  # Options: "remove_rows", "impute", "raise"
        "impute_method": "mean",  # Options: "mean", "median", "mode", "constant"
        "fill_value": 0,  # For constant imputation
        "threshold": 0.5  # Maximum ratio of missing values to handle
    },
    
    "decision_tree": {
        "criterion": "gini",  # Options: "gini", "entropy", "information_gain", "misclassification"
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_impurity_decrease": 0.0,
        "max_features": None,  # None = use all features
        "max_leaf_nodes": None,  # None = unlimited
        "growth_mode": "automatic",  # Options: "automatic", "manual", "hybrid"
        "class_weight": None,  # None = balanced
        "random_state": 42,
        "pruning_enabled": True,
        "pruning_method": "cost_complexity",
        "pruning_alpha": 0.01  # Complexity parameter
    },
    
    "metrics": {
        "cross_validation": {
            "enabled": True,
            "n_folds": 5,
            "stratified": True
        },
        "primary_metrics": ["accuracy", "precision", "recall", "f1_score", "roc_auc"],
        "threshold": 0.5,  # Decision threshold for binary classification
        "positive_class": None  # None = second class by default
    },
    
    "export": {
        "default_format": "pmml",  # Options: "pmml", "json", "python", "proprietary"
        "include_metadata": True,
        "add_comments": True,
        "include_data_summary": True
    }
}

def get_config_path() -> Path:
    """
    Get the path to the configuration file
    
    Returns:
        Path to the configuration file
    """
    script_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    config_path = script_dir / "config.json"
    
    return config_path

def load_configuration() -> Dict[str, Any]:
    """
    Load configuration from file, falling back to defaults if not found
    
    Returns:
        Configuration dictionary
    """
    config_path = get_config_path()
    
    try:
        if config_path.exists():
            logger.info(f"Loading configuration from {config_path}")
            
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            config = merge_configs(DEFAULT_CONFIG, user_config)
            
            logger.info("Configuration loaded successfully")
        else:
            logger.info("No configuration file found, using defaults")
            config = DEFAULT_CONFIG.copy()
            
            save_configuration(config)
        
        validate_configuration(config)
        
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}", exc_info=True)
        logger.warning("Falling back to default configuration")
        return DEFAULT_CONFIG.copy()

def save_configuration(config: Dict[str, Any]) -> bool:
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if saved successfully, False otherwise
    """
    config_path = get_config_path()
    
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Configuration saved to {config_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}", exc_info=True)
        return False

def merge_configs(default_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge user configuration with defaults
    
    Args:
        default_config: Default configuration dictionary
        user_config: User configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged = default_config.copy()
    
    for key, value in user_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged

def validate_configuration(config: Dict[str, Any]) -> bool:
    """
    Validate configuration values and log warnings for invalid settings
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if all values are valid, False otherwise
    """
    valid = True
    
    dt_config = config.get('decision_tree', {})
    
    criterion = dt_config.get('criterion')
    if criterion not in ['gini', 'entropy', 'information_gain', 'misclassification']:
        logger.warning(f"Invalid criterion: {criterion}, using 'gini' instead")
        dt_config['criterion'] = 'gini'
        valid = False
    
    growth_mode = dt_config.get('growth_mode')
    if growth_mode not in ['automatic', 'manual', 'hybrid']:
        logger.warning(f"Invalid growth mode: {growth_mode}, using 'automatic' instead")
        dt_config['growth_mode'] = 'automatic'
        valid = False
    
    for param, min_val, default in [
        ('max_depth', 1, 10),
        ('min_samples_split', 2, 2),
        ('min_samples_leaf', 1, 1),
        ('min_impurity_decrease', 0.0, 0.0),
        ('pruning_alpha', 0.0, 0.01)
    ]:
        value = dt_config.get(param)
        if not isinstance(value, (int, float)) or value < min_val:
            logger.warning(f"Invalid {param}: {value}, using {default} instead")
            dt_config[param] = default
            valid = False
    
    memory_config = config.get('memory', {})
    
    for param, min_val, default in [
        ('max_display_rows', 1, 100),
        ('max_display_cols', 1, 50),
        ('gc_threshold', 1, 100),
        ('memory_warning_threshold', 1.0, 80.0),
        ('chunk_size', 100, 50000)
    ]:
        value = memory_config.get(param)
        if not isinstance(value, (int, float)) or value < min_val:
            logger.warning(f"Invalid memory.{param}: {value}, using {default} instead")
            memory_config[param] = default
            valid = False
    
    ui_config = config.get('ui', {})
    
    if ui_config.get('theme') not in ['light', 'dark', 'system']:
        logger.warning(f"Invalid theme: {ui_config.get('theme')}, using 'light' instead")
        ui_config['theme'] = 'light'
        valid = False
    
    loader_config = config.get('data_loader', {})
    
    if loader_config.get('max_file_size_mb', 0) <= 0:
        logger.warning(f"Invalid max_file_size_mb: {loader_config.get('max_file_size_mb')}, using 800MB instead")
        loader_config['max_file_size_mb'] = 800
        valid = False
    
    missing_config = config.get('missing_values', {})
    
    if missing_config.get('strategy') not in ['remove_rows', 'impute', 'raise']:
        logger.warning(f"Invalid missing value strategy: {missing_config.get('strategy')}, using 'impute' instead")
        missing_config['strategy'] = 'impute'
        valid = False
    
    if missing_config.get('impute_method') not in ['mean', 'median', 'mode', 'constant']:
        logger.warning(f"Invalid impute method: {missing_config.get('impute_method')}, using 'mean' instead")
        missing_config['impute_method'] = 'mean'
        valid = False
    
    return valid

def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation for nested dictionaries
    
    Args:
        config: Configuration dictionary
        key_path: Key path using dot notation (e.g., 'decision_tree.max_depth')
        default: Default value if key not found
        
    Returns:
        Configuration value or default if not found
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default

def set_config_value(config: Dict[str, Any], key_path: str, value: Any) -> bool:
    """
    Set a configuration value using dot notation for nested dictionaries
    
    Args:
        config: Configuration dictionary
        key_path: Key path using dot notation (e.g., 'decision_tree.max_depth')
        value: Value to set
        
    Returns:
        True if successful, False otherwise
    """
    keys = key_path.split('.')
    target = config
    
    try:
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        
        target[keys[-1]] = value
        return True
    except Exception as e:
        logger.error(f"Error setting config value {key_path}: {str(e)}")
        return False

def create_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary directories based on configuration
    
    Args:
        config: Configuration dictionary
    """
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    temp_dir = base_dir / config.get('application', {}).get('temp_dir', 'temp')
    log_dir = base_dir / config.get('application', {}).get('log_dir', 'logs')
    data_dir = base_dir / config.get('application', {}).get('data_dir', 'data')
    
    for directory in [temp_dir, log_dir, data_dir]:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory created/verified: {directory}")
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {str(e)}")

def get_application_dirs(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Get application directory paths
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of directory paths
    """
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    return {
        'base': base_dir,
        'temp': base_dir / config.get('application', {}).get('temp_dir', 'temp'),
        'logs': base_dir / config.get('application', {}).get('log_dir', 'logs'),
        'data': base_dir / config.get('application', {}).get('data_dir', 'data')
    }
