#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logging Utilities for Bespoke Utility
Sets up application logging with file and console handlers
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

def setup_logging(log_dir: Optional[str] = None, 
                 log_level: int = logging.INFO,
                 log_format: Optional[str] = None,
                 enable_console: bool = True,
                 max_log_size: int = 10485760,  # 10MB
                 backup_count: int = 5) -> logging.Logger:
    """
    Set up application logging with file and console handlers
    
    Args:
        log_dir: Directory for log files (default: 'logs' in app directory)
        log_level: Logging level (default: INFO)
        log_format: Log message format (default: defined in function)
        enable_console: Whether to enable console logging
        max_log_size: Maximum size for log files before rotation (bytes)
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured root logger
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(log_format)
    
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        root_logger.addHandler(console_handler)
    
    if log_dir is None:
        app_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log_dir = app_dir / 'logs'
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'bespoke_utility_{timestamp}.log'
    
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=max_log_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized: {log_file}")
    logger.info(f"Log level: {logging.getLevelName(log_level)}")
    
    return root_logger

def set_log_level(logger_name: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Set logging level for a specific logger or the root logger
    
    Args:
        logger_name: Name of the logger to modify (None for root logger)
        level: New logging level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    for handler in logger.handlers:
        handler.setLevel(level)
    
    logging.getLogger(__name__).info(
        f"Log level for {'root' if logger_name is None else logger_name} "
        f"set to {logging.getLevelName(level)}"
    )

def create_module_logger(module_name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger for a specific module with appropriate configuration
    
    Args:
        module_name: Name of the module
        level: Logging level for this module
        
    Returns:
        Configured logger for the module
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    
    if not logger.handlers and not logger.propagate:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            logger.addHandler(handler)
    
    return logger

def flush_logs() -> None:
    """
    Flush all loggers' handlers to ensure logs are written
    """
    root_logger = logging.getLogger()
    
    for handler in root_logger.handlers:
        try:
            handler.flush()
        except Exception as e:
            print(f"Error flushing log handler: {str(e)}")

def log_exception(e: Exception, logger: Optional[logging.Logger] = None) -> None:
    """
    Log an exception with traceback
    
    Args:
        e: Exception to log
        logger: Logger to use (defaults to root logger)
    """
    if logger is None:
        logger = logging.getLogger()
    
    logger.error(f"Exception: {str(e)}", exc_info=True)

def log_system_info() -> Dict[str, Any]:
    """
    Log system information for diagnostics
    
    Returns:
        Dictionary with system information
    """
    import platform
    import sys
    import locale
    
    try:
        import psutil
        memory_available = True
    except ImportError:
        memory_available = False
    
    logger = logging.getLogger(__name__)
    
    system_info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': sys.version,
        'python_implementation': platform.python_implementation(),
        'locale': locale.getdefaultlocale(),
        'encoding': sys.getdefaultencoding(),
        'filesystem_encoding': sys.getfilesystemencoding()
    }
    
    if memory_available:
        memory = psutil.virtual_memory()
        system_info.update({
            'memory_total': f"{memory.total / (1024**3):.2f} GB",
            'memory_available': f"{memory.available / (1024**3):.2f} GB",
            'memory_percent_used': f"{memory.percent}%"
        })
    
    logger.info("System information:")
    for key, value in system_info.items():
        logger.info(f"  {key}: {value}")
    
    return system_info

class LogCapture:
    """Context manager to capture logs during a specific operation"""
    
    def __init__(self, logger_name: Optional[str] = None, level: int = logging.INFO):
        """
        Initialize log capture
        
        Args:
            logger_name: Name of the logger to capture (None for root)
            level: Minimum log level to capture
        """
        self.logger_name = logger_name
        self.level = level
        self.logger = logging.getLogger(logger_name)
        self.log_records = []
        self.handler = None
    
    def __enter__(self):
        """Set up log capture when entering context"""
        self.handler = _MemoryHandler(self.log_records)
        self.handler.setLevel(self.level)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(formatter)
        
        self.logger.addHandler(self.handler)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up when exiting context"""
        if self.handler:
            self.logger.removeHandler(self.handler)
    
    def get_logs(self) -> list:
        """
        Get captured log records
        
        Returns:
            List of log records
        """
        return self.log_records
    
    def get_log_messages(self) -> list:
        """
        Get captured log messages as strings
        
        Returns:
            List of formatted log messages
        """
        if not self.handler:
            return []
        
        formatter = self.handler.formatter
        return [formatter.format(record) for record in self.log_records]

class _MemoryHandler(logging.Handler):
    """Custom handler to store log records in memory"""
    
    def __init__(self, records_list):
        """
        Initialize with a list to store records
        
        Args:
            records_list: List to store log records
        """
        super().__init__()
        self.records_list = records_list
    
    def emit(self, record):
        """
        Store the log record
        
        Args:
            record: Log record to store
        """
        self.records_list.append(record)
