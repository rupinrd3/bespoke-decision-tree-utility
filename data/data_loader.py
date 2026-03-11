#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Loader Module for Bespoke Utility
Handles importing data from various sources and formats
"""


import os
import io
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import psutil

import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QMessageBox
from joblib import Parallel, delayed

from utils.memory_management import optimize_dataframe

logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading data from various sources with optimizations for large datasets"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize DataLoader with configuration"""
        self.config = config or {}
        self.supported_formats = {
            'csv': self._load_csv,
            'xlsx': self._load_excel,
            'xls': self._load_excel,
            'txt': self._load_text,
            'parquet': self._load_parquet,
            'pqt': self._load_parquet,
            'feather': self._load_feather,
            'ft': self._load_feather,
            'pkl': self._load_pickle,
            'pickle': self._load_pickle,
        }
        
        pd.set_option('mode.chained_assignment', None)  # Suppress SettingWithCopyWarning
        self.chunk_size = self.config.get('data_loader', {}).get('chunk_size', 100000)
        
        self.use_parallel_processing = self.config.get('data_loader', {}).get('use_parallel_processing', True)
        self.memory_threshold_gb = self.config.get('data_loader', {}).get('memory_threshold_gb', 8.0)  # Enable parallel for datasets > 8GB available memory
        self.max_parallel_columns = self.config.get('data_loader', {}).get('max_parallel_columns', 100)  # Process columns in batches
        
        self.datasets = {}
        
    def load_file(self, file_path: str, **kwargs) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Load data from a file with appropriate format handler
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments for the loader
            
        Returns:
            Tuple containing:
                - DataFrame with the loaded data (or None if failed)
                - Metadata dictionary
        """
        try:
            file_path = Path(file_path)
            file_extension = file_path.suffix.lower().lstrip('.')
            
            if file_extension not in self.supported_formats:
                error_msg = f"Unsupported file format: {file_extension}"
                logger.error(error_msg)
                return None, {"error": error_msg}
            
            df, metadata = self.supported_formats[file_extension](file_path, **kwargs)
            
            if df is not None:
                df = optimize_dataframe(df)
                
                dataset_name = kwargs.get('name', file_path.stem)
                self.datasets[dataset_name] = {
                    'path': str(file_path),
                    'rows': len(df),
                    'columns': df.columns.tolist(),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                    'memory_usage': df.memory_usage(deep=True).sum(),
                    **metadata
                }
                
                logger.info(f"Successfully loaded dataset '{dataset_name}' with {len(df)} rows and {len(df.columns)} columns")
                return df, self.datasets[dataset_name]
            else:
                return None, metadata
                
        except Exception as e:
            error_msg = f"Error loading file {file_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, {"error": error_msg}
    
    def _load_csv(self, file_path: Path, **kwargs) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Load a CSV file with optimizations for large files
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments:
                - delimiter: CSV delimiter character (default: ',')
                - header: Header row (default: 0)
                - encoding: File encoding (default: 'utf-8')
                - use_chunks: Whether to use chunking for large files (default: True)
                
        Returns:
            Tuple with DataFrame and metadata
        """
        metadata: Dict[str, Any] = {}
        
        delimiter = kwargs.get('delimiter', ',')
        header = kwargs.get('header', 0)
        encoding = kwargs.get('encoding', 'utf-8')
        use_chunks = kwargs.get('use_chunks', True)
        nrows = kwargs.get('nrows')
        skiprows = kwargs.get('skiprows', 0)
        skipfooter = kwargs.get('skipfooter', 0)
        usecols = kwargs.get('usecols')
        na_values = kwargs.get('na_values', None)
        decimal = kwargs.get('decimal', '.')
        thousands = kwargs.get('thousands', '')
        quotechar = self._sanitize_char_option(kwargs.get('quotechar', '"'))
        escapechar = self._sanitize_char_option(kwargs.get('escapechar'))
        parse_dates = kwargs.get('parse_dates')
        date_columns = kwargs.get('date_columns')
        if parse_dates is None and date_columns:
            parse_dates = date_columns
        target_date_columns = parse_dates or date_columns
        date_format = kwargs.get('date_format')
        engine = kwargs.get('engine')
        if not engine and skipfooter:
            engine = 'python'

        thousands = thousands if thousands else None

        if nrows is not None or skipfooter:
            use_chunks = False

        try:
            file_size = file_path.stat().st_size
            logger.debug(f"CSV file size: {file_size/1024/1024:.2f} MB")
            
            read_csv_kwargs = {
                'delimiter': delimiter,
                'header': header,
                'low_memory': False
            }
            if encoding:
                read_csv_kwargs['encoding'] = encoding
            if quotechar:
                read_csv_kwargs['quotechar'] = quotechar
            if escapechar:
                read_csv_kwargs['escapechar'] = escapechar
            if skiprows:
                read_csv_kwargs['skiprows'] = skiprows
            if skipfooter:
                read_csv_kwargs['skipfooter'] = skipfooter
            if usecols:
                read_csv_kwargs['usecols'] = usecols
            if na_values is not None:
                read_csv_kwargs['na_values'] = na_values
            if decimal:
                read_csv_kwargs['decimal'] = decimal
            if thousands:
                read_csv_kwargs['thousands'] = thousands
            if parse_dates and not date_format:
                read_csv_kwargs['parse_dates'] = parse_dates
            if engine:
                read_csv_kwargs['engine'] = engine
            
            if file_size > 100*1024*1024 and use_chunks:  # > 100 MB
                logger.info(f"Large CSV detected, using chunked loading with chunk size {self.chunk_size}")
                
                sample_kwargs = dict(read_csv_kwargs)
                sample_kwargs['nrows'] = 5000 if nrows is None else min(nrows, 5000)
                sample_kwargs['low_memory'] = True
                sample = pd.read_csv(file_path, **sample_kwargs)
                
                binary_cols = self._detect_binary_columns(sample)
                metadata['potential_binary_targets'] = binary_cols
                
                chunks = []
                
                for chunk in pd.read_csv(
                    file_path,
                    chunksize=self.chunk_size,
                    **read_csv_kwargs
                ):
                    chunks.append(chunk)
                
                df = pd.concat(chunks, ignore_index=True)
                metadata['loaded_in_chunks'] = True
                metadata['num_chunks'] = len(chunks)
                
            else:
                direct_kwargs = dict(read_csv_kwargs)
                if nrows is not None:
                    direct_kwargs['nrows'] = nrows
                df = pd.read_csv(file_path, **direct_kwargs)
                
                binary_cols = self._detect_binary_columns(df)
                metadata['potential_binary_targets'] = binary_cols
                
                metadata['loaded_in_chunks'] = False

            df = self._apply_custom_date_format(df, target_date_columns, date_format)
            
            df = self._optimize_dtypes_existing_df(df)
            
            metadata['delimiter'] = delimiter
            metadata['encoding'] = encoding
            metadata['file_size'] = file_size
            metadata['quotechar'] = quotechar or 'default'
            metadata['escapechar'] = escapechar or 'none'
            metadata['skiprows'] = skiprows
            metadata['skipfooter'] = skipfooter
            metadata['decimal'] = decimal
            metadata['thousands'] = thousands or 'none'
            
            return df, metadata
            
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}", exc_info=True)
            
            try:
                logger.info("Attempting fallback CSV loading with string types...")
                fallback_kwargs = {
                    'delimiter': delimiter,
                    'header': header,
                    'encoding': encoding,
                    'skiprows': skiprows,
                    'skipfooter': skipfooter,
                    'dtype': str,  # Load everything as string initially
                    'low_memory': False
                }
                if quotechar:
                    fallback_kwargs['quotechar'] = quotechar
                if escapechar:
                    fallback_kwargs['escapechar'] = escapechar
                if skipfooter:
                    fallback_kwargs['engine'] = 'python'
                df = pd.read_csv(file_path, **fallback_kwargs)
                
                df = self._auto_detect_and_convert_types(df)
                df = self._apply_custom_date_format(df, target_date_columns, date_format)
                
                binary_cols = self._detect_binary_columns(df)
                metadata = {
                    'potential_binary_targets': binary_cols,
                    'fallback_loading': True,
                    'delimiter': delimiter,
                    'encoding': encoding,
                    'file_size': file_path.stat().st_size,
                    'loaded_in_chunks': False,
                    'skiprows': skiprows,
                    'skipfooter': skipfooter,
                    'decimal': decimal,
                    'thousands': thousands or 'none'
                }
                
                logger.info("Fallback CSV loading successful")
                return df, metadata
                
            except Exception as fallback_error:
                logger.error(f"Fallback CSV loading also failed: {str(fallback_error)}")
                return None, {"error": f"CSV loading error: {str(e)}. Fallback error: {str(fallback_error)}"}
    
    def _load_excel(self, file_path: Path, **kwargs) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Load an Excel file with optimizations
        
        Args:
            file_path: Path to the Excel file
            **kwargs: Additional arguments:
                - sheet_name: Sheet to load (default: 0)
                - header: Header row (default: 0)
                
        Returns:
            Tuple with DataFrame and metadata
        """
        metadata: Dict[str, Any] = {}
        
        sheet_name = kwargs.get('sheet_name', 0)
        header = kwargs.get('header', 0)
        skiprows = kwargs.get('skiprows', 0)
        skipfooter = kwargs.get('skipfooter', 0)
        nrows = kwargs.get('nrows')
        na_values = kwargs.get('na_values', None)
        parse_dates = kwargs.get('parse_dates')
        date_columns = kwargs.get('date_columns')
        if parse_dates is None and date_columns:
            parse_dates = date_columns
        target_date_columns = parse_dates or date_columns
        date_format = kwargs.get('date_format')
        decimal = kwargs.get('decimal', '.')
        thousands = kwargs.get('thousands', '')
        
        try:
            file_size = file_path.stat().st_size
            logger.debug(f"Excel file size: {file_size/1024/1024:.2f} MB")
            
            excel_file = pd.ExcelFile(file_path)
            metadata['available_sheets'] = excel_file.sheet_names

            read_excel_kwargs = {
                'sheet_name': sheet_name,
                'header': header,
                'skiprows': skiprows,
                'nrows': nrows,
                'na_values': na_values,
                'decimal': decimal,
                'thousands': thousands if thousands else None
            }
            if parse_dates and not date_format:
                read_excel_kwargs['parse_dates'] = parse_dates

            df = pd.read_excel(excel_file, **read_excel_kwargs)

            if isinstance(skipfooter, int) and skipfooter > 0:
                if skipfooter < len(df):
                    df = df.iloc[:-skipfooter]
                else:
                    df = df.iloc[0:0]

            df = self._apply_custom_date_format(df, target_date_columns, date_format)
            
            df = self._optimize_dtypes_existing_df(df)
            
            binary_cols = self._detect_binary_columns(df)
            metadata['potential_binary_targets'] = binary_cols
            
            metadata['sheet_name'] = sheet_name
            metadata['file_size'] = file_size
            metadata['skiprows'] = skiprows
            metadata['skipfooter'] = skipfooter
            metadata['decimal'] = decimal
            metadata['thousands'] = thousands or 'none'
            
            return df, metadata
            
        except Exception as e:
            logger.error(f"Error loading Excel file: {str(e)}", exc_info=True)
            return None, {"error": f"Excel loading error: {str(e)}"}
    
    def _load_text(self, file_path: Path, **kwargs) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Load a delimited text file
        
        Args:
            file_path: Path to the text file
            **kwargs: Additional arguments:
                - delimiter: Field delimiter (default tries to detect)
                - header: Header row (default: 0)
                - encoding: File encoding (default: 'utf-8')
                
        Returns:
            Tuple with DataFrame and metadata
        """
        metadata = {}
        
        delimiter = kwargs.get('delimiter', None)  # Try to detect if not provided
        header = kwargs.get('header', 0)
        encoding = kwargs.get('encoding', 'utf-8')
        
        try:
            if delimiter is None:
                with open(file_path, 'r', encoding=encoding) as f:
                    sample = f.read(10000)  # Read first 10KB
                
                delimiters = [',', '\t', ';', '|', ' ']
                delimiter_counts = {d: sample.count(d) for d in delimiters}
                
                if max(delimiter_counts.values()) > 0:
                    delimiter = max(delimiter_counts.items(), key=lambda x: x[1])[0]
                    logger.info(f"Detected delimiter: '{delimiter}'")
                else:
                    delimiter = '\t'  # Default to tab if detection fails
                    logger.warning("Could not detect delimiter, using tab as default")
                
                metadata['detected_delimiter'] = delimiter
            
            kwargs['delimiter'] = delimiter
            kwargs['header'] = header
            kwargs['encoding'] = encoding
            df, csv_metadata = self._load_csv(file_path, **kwargs)
            metadata.update(csv_metadata)
            return df, metadata
            
        except Exception as e:
            logger.error(f"Error loading text file: {str(e)}", exc_info=True)
            return None, {"error": f"Text file loading error: {str(e)}"}
    
    def _sanitize_char_option(self, value: Optional[str]) -> Optional[str]:
        """Normalize single-character options like quote or escape characters"""
        if value is None:
            return None
        value_str = str(value)
        if not value_str:
            return None
        try:
            decoded_value = bytes(value_str, 'utf-8').decode('unicode_escape')
        except Exception as exc:
            logger.warning(f"Could not decode escape sequence '{value_str}': {exc}")
            decoded_value = value_str
        if not decoded_value:
            return None
        return decoded_value[0]
    
    def _apply_custom_date_format(self, df: pd.DataFrame, date_columns: Optional[List[str]], date_format: Optional[str]) -> pd.DataFrame:
        """Apply custom date parsing to specified columns if requested"""
        if df is None or not date_columns:
            return df
        for col in date_columns:
            if col not in df.columns:
                continue
            try:
                if date_format:
                    df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
                elif not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception as e:
                logger.warning(f"Could not parse column '{col}' with format '{date_format}': {e}")
        return df
    
    def _apply_row_limits(self, df: pd.DataFrame, skiprows: Optional[int] = None,
                          skipfooter: Optional[int] = None, nrows: Optional[int] = None) -> pd.DataFrame:
        """Apply row trimming for formats that do not support native parameters"""
        if df is None or df.empty:
            return df
        
        result = df
        if isinstance(skiprows, int) and skiprows > 0:
            result = result.iloc[skiprows:]
        if isinstance(skipfooter, int) and skipfooter > 0:
            if skipfooter < len(result):
                result = result.iloc[:-skipfooter]
            else:
                result = result.iloc[0:0]
        if isinstance(nrows, int) and nrows > 0:
            result = result.head(nrows)
        return result
    
    def _load_parquet(self, file_path: Path, **kwargs) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Load a Parquet file with optional column selection
        """
        metadata: Dict[str, Any] = {}
        columns = kwargs.get('columns')  # Optional subset of columns
        engine = kwargs.get('engine')  # Optional override
        nrows = kwargs.get('nrows')
        skiprows = kwargs.get('skiprows', 0)
        skipfooter = kwargs.get('skipfooter', 0)
        date_columns = kwargs.get('date_columns') or kwargs.get('parse_dates')
        date_format = kwargs.get('date_format')
        
        try:
            logger.info(f"Loading Parquet file: {file_path}")
            read_parquet_kwargs = {}
            if columns is not None:
                read_parquet_kwargs['columns'] = columns
            if engine is not None:
                read_parquet_kwargs['engine'] = engine

            df = pd.read_parquet(file_path, **read_parquet_kwargs)
            df = self._apply_row_limits(df, skiprows=skiprows, skipfooter=skipfooter, nrows=nrows)
            df = self._apply_custom_date_format(df, date_columns, date_format)
            
            df = self._optimize_dtypes_existing_df(df)
            
            metadata['potential_binary_targets'] = self._detect_binary_columns(df)
            
            metadata['file_size'] = file_path.stat().st_size
            if columns is not None:
                metadata['columns_requested'] = columns
            if engine:
                metadata['engine'] = engine
            metadata['skiprows'] = skiprows
            metadata['skipfooter'] = skipfooter
            metadata['nrows'] = nrows
            
            return df, metadata
        
        except ImportError as e:
            error_msg = ("Parquet support requires either 'pyarrow' or 'fastparquet'. "
                         f"Install an engine to proceed. Original error: {str(e)}")
            logger.error(error_msg)
            return None, {"error": error_msg}
        except Exception as e:
            logger.error(f"Error loading Parquet file: {str(e)}", exc_info=True)
            return None, {"error": f"Parquet loading error: {str(e)}"}
    
    def _load_feather(self, file_path: Path, **kwargs) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Load a Feather file with optional column selection
        """
        metadata: Dict[str, Any] = {}
        columns = kwargs.get('columns')
        use_threads = kwargs.get('use_threads', True)
        nrows = kwargs.get('nrows')
        skiprows = kwargs.get('skiprows', 0)
        skipfooter = kwargs.get('skipfooter', 0)
        date_columns = kwargs.get('date_columns') or kwargs.get('parse_dates')
        date_format = kwargs.get('date_format')
        
        try:
            logger.info(f"Loading Feather file: {file_path}")
            df = pd.read_feather(file_path, columns=columns, use_threads=use_threads)
            df = self._apply_row_limits(df, skiprows=skiprows, skipfooter=skipfooter, nrows=nrows)
            df = self._apply_custom_date_format(df, date_columns, date_format)
            
            df = self._optimize_dtypes_existing_df(df)
            
            metadata['potential_binary_targets'] = self._detect_binary_columns(df)
            metadata['file_size'] = file_path.stat().st_size
            metadata['use_threads'] = use_threads
            if columns is not None:
                metadata['columns_requested'] = columns
            metadata['skiprows'] = skiprows
            metadata['skipfooter'] = skipfooter
            metadata['nrows'] = nrows
            
            return df, metadata
        
        except ImportError as e:
            error_msg = ("Feather import requires 'pyarrow'. Install it to enable Feather support. "
                         f"Original error: {str(e)}")
            logger.error(error_msg)
            return None, {"error": error_msg}
        except Exception as e:
            logger.error(f"Error loading Feather file: {str(e)}", exc_info=True)
            return None, {"error": f"Feather loading error: {str(e)}"}
    
    def _load_pickle(self, file_path: Path, **kwargs) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Load a pickled pandas DataFrame (or convertible object)
        """
        metadata: Dict[str, Any] = {}
        compression = kwargs.get('compression', 'infer')
        nrows = kwargs.get('nrows')
        skiprows = kwargs.get('skiprows', 0)
        skipfooter = kwargs.get('skipfooter', 0)
        date_columns = kwargs.get('date_columns') or kwargs.get('parse_dates')
        date_format = kwargs.get('date_format')
        
        try:
            logger.info(f"Loading pickle file: {file_path}")
            obj = pd.read_pickle(file_path, compression=compression)
            metadata['object_type'] = type(obj).__name__
            
            if isinstance(obj, pd.DataFrame):
                df = obj
            else:
                try:
                    df = pd.DataFrame(obj)
                    metadata['converted_from'] = type(obj).__name__
                except Exception:
                    error_msg = ("Pickle file did not contain a DataFrame or convertible structure. "
                                 f"Found type: {type(obj).__name__}")
                    logger.error(error_msg)
                    return None, {"error": error_msg}
            
            df = self._apply_row_limits(df, skiprows=skiprows, skipfooter=skipfooter, nrows=nrows)
            df = self._apply_custom_date_format(df, date_columns, date_format)
            
            df = self._optimize_dtypes_existing_df(df)
            metadata['potential_binary_targets'] = self._detect_binary_columns(df)
            metadata['file_size'] = file_path.stat().st_size
            metadata['compression'] = compression
            metadata['skiprows'] = skiprows
            metadata['skipfooter'] = skipfooter
            metadata['nrows'] = nrows
            
            return df, metadata
        
        except Exception as e:
            logger.error(f"Error loading pickle file: {str(e)}", exc_info=True)
            return None, {"error": f"Pickle loading error: {str(e)}"}
    
    def _detect_binary_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Detect columns that could be binary target variables
        Uses parallel processing for large datasets
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of column names that have exactly two unique values
        """
        if self._should_use_parallel_processing(df):
            return self._detect_binary_columns_parallel(df)
        else:
            return self._detect_binary_columns_sequential(df)
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Determine optimal data types for each column to reduce memory usage
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping column names to optimal data types
        """
        dtypes = {}
        
        for col in df.columns:
            if df[col].dtype == 'category' or df[col].dtype == 'datetime64[ns]':
                continue
                
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].isna().sum() == 0 and df[col].apply(lambda x: x == int(x)).all():
                    col_min, col_max = df[col].min(), df[col].max()
                    
                    if col_min >= 0:  # Unsigned
                        if col_max <= 255:
                            dtypes[col] = 'uint8'
                        elif col_max <= 65535:
                            dtypes[col] = 'uint16'
                        elif col_max <= 4294967295:
                            dtypes[col] = 'uint32'
                        else:
                            dtypes[col] = 'uint64'
                    else:  # Signed
                        if col_min >= -128 and col_max <= 127:
                            dtypes[col] = 'int8'
                        elif col_min >= -32768 and col_max <= 32767:
                            dtypes[col] = 'int16'
                        elif col_min >= -2147483648 and col_max <= 2147483647:
                            dtypes[col] = 'int32'
                        else:
                            dtypes[col] = 'int64'
                else:
                    if df[col].min() >= -3.4e38 and df[col].max() <= 3.4e38:
                        dtypes[col] = 'float32'
                    else:
                        dtypes[col] = 'float64'
            
            elif df[col].dtype == 'object':
                num_unique = df[col].nunique()
                if num_unique <= min(100, len(df) // 10):  # Either ≤100 unique values or <10% of rows
                    dtypes[col] = 'category'
        
        return dtypes
    
    def _optimize_dtypes_existing_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize the data types of an existing DataFrame to reduce memory
        Uses parallel processing for large datasets
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        if self._should_use_parallel_processing(df):
            dtypes = self._optimize_dtypes_parallel(df)
        else:
            dtypes = self._optimize_dtypes(df)
        
        optimized_count = 0
        for col, dtype in dtypes.items():
            try:
                if dtype == 'category':
                    df[col] = df[col].astype('category')
                else:
                    df[col] = df[col].astype(dtype)
                optimized_count += 1
            except Exception as e:
                logger.warning(f"Could not convert column {col} to {dtype}: {str(e)}")
        
        logger.info(f"Successfully optimized {optimized_count} columns out of {len(dtypes)} candidates")
        return df
    
    def _auto_detect_and_convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Auto-detect and convert column types from string DataFrame
        Uses parallel processing for large datasets
        
        Args:
            df: DataFrame with string columns to convert
            
        Returns:
            DataFrame with appropriate types
        """
        if self._should_use_parallel_processing(df):
            return self._auto_detect_and_convert_types_parallel(df)
        else:
            return self._auto_detect_and_convert_types_sequential(df)
    
    def _should_use_parallel_processing(self, df: pd.DataFrame) -> bool:
        """
        Determine if parallel processing should be used based on dataset size and available memory
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            True if parallel processing should be used
        """
        if not self.use_parallel_processing:
            return False
            
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        try:
            df_memory_gb = df.memory_usage(deep=True).sum() / (1024**3)
        except:
            df_memory_gb = len(df) * len(df.columns) * 8 / (1024**3)  # Rough estimate
        
        
        memory_safe = available_memory_gb > self.memory_threshold_gb
        dataset_large_enough = (len(df.columns) > 50 or df_memory_gb > 0.1)  # >50 cols or >100MB
        memory_headroom = available_memory_gb > (df_memory_gb * 2)  # At least 2x dataset size available
        
        use_parallel = memory_safe and dataset_large_enough and memory_headroom
        
        if use_parallel:
            logger.info(f"Using parallel processing: {available_memory_gb:.2f}GB available memory, {df_memory_gb:.2f}GB dataset, {len(df.columns)} columns")
        else:
            logger.info(f"Using sequential processing: {available_memory_gb:.2f}GB available memory, {df_memory_gb:.2f}GB dataset, {len(df.columns)} columns")
            
        return use_parallel
    
    def _analyze_single_column_dtype(self, col_data: pd.Series, col_name: str) -> Tuple[str, Optional[str]]:
        """
        Analyze a single column for optimal data type - designed for parallel execution
        
        Args:
            col_data: Column data as pandas Series
            col_name: Column name for logging
            
        Returns:
            Tuple of (column_name, optimal_dtype)
        """
        try:
            if col_data.dtype == 'category' or col_data.dtype == 'datetime64[ns]':
                return col_name, None
                
            if pd.api.types.is_numeric_dtype(col_data):
                non_null_data = col_data.dropna()
                if len(non_null_data) == 0:
                    return col_name, None
                    
                if col_data.isna().sum() == 0 and non_null_data.apply(lambda x: x == int(x) if pd.notna(x) else True).all():
                    col_min, col_max = non_null_data.min(), non_null_data.max()
                    
                    if col_min >= 0:  # Unsigned
                        if col_max <= 255:
                            return col_name, 'uint8'
                        elif col_max <= 65535:
                            return col_name, 'uint16'
                        elif col_max <= 4294967295:
                            return col_name, 'uint32'
                        else:
                            return col_name, 'uint64'
                    else:  # Signed
                        if col_min >= -128 and col_max <= 127:
                            return col_name, 'int8'
                        elif col_min >= -32768 and col_max <= 32767:
                            return col_name, 'int16'
                        elif col_min >= -2147483648 and col_max <= 2147483647:
                            return col_name, 'int32'
                        else:
                            return col_name, 'int64'
                else:
                    if non_null_data.min() >= -3.4e38 and non_null_data.max() <= 3.4e38:
                        return col_name, 'float32'
                    else:
                        return col_name, 'float64'
            
            elif col_data.dtype == 'object':
                num_unique = col_data.nunique()
                total_count = len(col_data)
                if num_unique <= min(100, total_count // 10):  # Either ≤100 unique values or <10% of rows
                    return col_name, 'category'
        
        except Exception as e:
            logger.warning(f"Error analyzing column {col_name}: {e}")
            
        return col_name, None
    
    def _optimize_dtypes_parallel(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Determine optimal data types for columns using parallel processing
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping column names to optimal data types
        """
        try:
            columns = list(df.columns)
            batch_size = min(self.max_parallel_columns, len(columns))
            
            all_results = []
            
            for i in range(0, len(columns), batch_size):
                batch_columns = columns[i:i + batch_size]
                
                batch_data = [(df[col].copy(), col) for col in batch_columns]
                
                batch_results = Parallel(n_jobs=-2, backend='loky')(
                    delayed(self._analyze_single_column_dtype)(col_data, col_name) 
                    for col_data, col_name in batch_data
                )
                
                all_results.extend(batch_results)
                
                if len(columns) > 100:
                    logger.info(f"Processed dtype optimization batch {i//batch_size + 1}/{(len(columns) + batch_size - 1)//batch_size}")
            
            dtypes = {col_name: dtype for col_name, dtype in all_results if dtype is not None}
            
            logger.info(f"Parallel dtype optimization completed: {len(dtypes)} columns optimized out of {len(columns)}")
            return dtypes
            
        except Exception as e:
            logger.error(f"Error in parallel dtype optimization: {e}")
            logger.info("Falling back to sequential dtype optimization")
            return self._optimize_dtypes(df)
    
    def _auto_detect_single_column(self, col_data: pd.Series, col_name: str) -> Tuple[str, pd.Series]:
        """
        Auto-detect and convert a single column type - designed for parallel execution
        
        Args:
            col_data: Column data as pandas Series
            col_name: Column name
            
        Returns:
            Tuple of (column_name, converted_series)
        """
        try:
            if col_data.dtype != 'object' and col_data.dtype != 'string':
                return col_name, col_data  # Already converted
                
            try:
                non_null_values = col_data.dropna()
                if len(non_null_values) > 0:
                    numeric_series = pd.to_numeric(non_null_values, errors='coerce')
                    
                    if numeric_series.notna().all():
                        converted_series = pd.to_numeric(col_data, errors='coerce')
                        return col_name, converted_series
                        
            except Exception:
                pass
            
            try:
                unique_values = col_data.dropna().astype(str).str.lower().unique()
                if len(unique_values) <= 2 and all(val in ['true', 'false', '1', '0', 'yes', 'no', 'y', 'n'] for val in unique_values):
                    bool_map = {'true': True, 'false': False, '1': True, '0': False, 
                               'yes': True, 'no': False, 'y': True, 'n': False}
                    converted_series = col_data.astype(str).str.lower().map(bool_map)
                    return col_name, converted_series
            except Exception:
                pass
            
            try:
                unique_count = col_data.nunique()
                total_count = len(col_data)
                
                if unique_count < 50 or (unique_count / total_count) < 0.05:
                    converted_series = col_data.astype('category')
                    return col_name, converted_series
                    
            except Exception:
                pass
            
        except Exception as e:
            logger.warning(f"Error auto-detecting column {col_name}: {e}")
            
        return col_name, col_data  # Return original if no conversion possible
    
    def _auto_detect_and_convert_types_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Auto-detect and convert column types using parallel processing
        
        Args:
            df: DataFrame with string columns to convert
            
        Returns:
            DataFrame with appropriate types
        """
        try:
            string_columns = [col for col in df.columns 
                            if df[col].dtype == 'object' or df[col].dtype == 'string']
            
            if not string_columns:
                return df
            
            logger.info(f"Starting parallel auto-type detection for {len(string_columns)} string columns")
            
            batch_size = min(self.max_parallel_columns, len(string_columns))
            
            for i in range(0, len(string_columns), batch_size):
                batch_columns = string_columns[i:i + batch_size]
                
                batch_data = [(df[col].copy(), col) for col in batch_columns]
                
                batch_results = Parallel(n_jobs=-2, backend='loky')(
                    delayed(self._auto_detect_single_column)(col_data, col_name) 
                    for col_data, col_name in batch_data
                )
                
                conversions_applied = 0
                for col_name, converted_series in batch_results:
                    if not converted_series.equals(df[col_name]):
                        df[col_name] = converted_series
                        conversions_applied += 1
                
                if len(string_columns) > 50:
                    logger.info(f"Auto-detection batch {i//batch_size + 1}/{(len(string_columns) + batch_size - 1)//batch_size} completed, {conversions_applied} conversions applied")
            
            logger.info("Parallel auto-detection and type conversion completed")
            return df
            
        except Exception as e:
            logger.error(f"Error in parallel auto-type detection: {e}")
            logger.info("Falling back to sequential auto-type detection")
            return self._auto_detect_and_convert_types_sequential(df)
            
    def _auto_detect_and_convert_types_sequential(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sequential version of auto-detect for fallback
        """
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype == 'string':
                _, converted_series = self._auto_detect_single_column(df[col], col)
                if not converted_series.equals(df[col]):
                    df[col] = converted_series
        
        logger.info("Sequential auto-detection and type conversion completed")
        return df
    
    def _detect_single_binary_column(self, col_data: pd.Series, col_name: str) -> Tuple[str, bool, Optional[List]]:
        """
        Detect if a single column is binary - designed for parallel execution
        
        Args:
            col_data: Column data as pandas Series
            col_name: Column name
            
        Returns:
            Tuple of (column_name, is_binary, unique_values_list)
        """
        try:
            if col_data.isna().sum() > len(col_data) * 0.2:  # 20% or more are null
                return col_name, False, None
                
            unique_values = col_data.dropna().unique()
            if len(unique_values) == 2:
                return col_name, True, unique_values.tolist()
                
        except Exception as e:
            logger.warning(f"Error detecting binary column {col_name}: {e}")
            
        return col_name, False, None
    
    def _detect_binary_columns_parallel(self, df: pd.DataFrame) -> List[str]:
        """
        Detect binary columns using parallel processing
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of column names that have exactly two unique values
        """
        try:
            columns = list(df.columns)
            logger.info(f"Starting parallel binary column detection for {len(columns)} columns")
            
            batch_size = min(self.max_parallel_columns, len(columns))
            all_results = []
            
            for i in range(0, len(columns), batch_size):
                batch_columns = columns[i:i + batch_size]
                
                batch_data = [(df[col].copy(), col) for col in batch_columns]
                
                batch_results = Parallel(n_jobs=-2, backend='loky')(
                    delayed(self._detect_single_binary_column)(col_data, col_name) 
                    for col_data, col_name in batch_data
                )
                
                all_results.extend(batch_results)
                
                if len(columns) > 100:
                    logger.info(f"Binary detection batch {i//batch_size + 1}/{(len(columns) + batch_size - 1)//batch_size} completed")
            
            binary_cols = []
            for col_name, is_binary, unique_values in all_results:
                if is_binary:
                    binary_cols.append(col_name)
                    logger.debug(f"Detected binary column: {col_name} with values {unique_values}")
            
            logger.info(f"Parallel binary column detection completed: found {len(binary_cols)} binary columns out of {len(columns)}")
            return binary_cols
            
        except Exception as e:
            logger.error(f"Error in parallel binary column detection: {e}")
            logger.info("Falling back to sequential binary column detection")
            return self._detect_binary_columns_sequential(df)
    
    def _detect_binary_columns_sequential(self, df: pd.DataFrame) -> List[str]:
        """
        Sequential version of binary column detection for fallback
        """
        binary_cols = []
        
        for col in df.columns:
            _, is_binary, unique_values = self._detect_single_binary_column(df[col], col)
            if is_binary:
                binary_cols.append(col)
                logger.debug(f"Detected binary column: {col} with values {unique_values}")
        
        return binary_cols
