#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Processor Module for Bespoke Utility
Handles data cleaning, transformation, and feature engineering tasks.

[DataProcessor.__init__ -> Initializes the DataProcessor -> dependent functions are None]
[DataProcessor.filter_data -> Filters a DataFrame based on specified conditions -> dependent functions are _apply_comparison, optimize_dataframe]
[DataProcessor._apply_comparison -> Applies a comparison operation to create a boolean mask -> dependent functions are None]
[DataProcessor.handle_missing_values -> Handles missing values in the DataFrame -> dependent functions are optimize_dataframe]
[DataProcessor.transform_column -> Applies a transformation to a column -> dependent functions are optimize_dataframe]
[DataProcessor.get_processing_history -> Gets the processing history for a dataset -> dependent functions are None]
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import pandas as pd
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from utils.memory_management import optimize_dataframe

logger = logging.getLogger(__name__)

class DataProcessor(QObject):
    """Class for processing and filtering datasets"""
    
    progressUpdated = pyqtSignal(int, str)
    processingComplete = pyqtSignal(str, object)
    processingError = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize DataProcessor with configuration"""
        super().__init__()
        self.config = config or {}
        
        self.processing_history = {}
        self.filter_conditions = {}
        
    def filter_data(self, df: pd.DataFrame, conditions: List[Dict[str, Any]], 
                   dataset_name: str = "dataset", inplace: bool = False) -> Optional[pd.DataFrame]:
        """
        Filter a DataFrame based on specified conditions
        
        Args:
            df: Input DataFrame to filter
            conditions: List of filter conditions, where each condition is a dict with:
                - column: Column name to filter on
                - operator: Comparison operator ('>', '<', '=', '!=', 'contains', etc.)
                - value: Value to compare against
                - logic: 'AND' or 'OR' (how to combine with the previous condition)
            dataset_name: Name of the dataset for tracking
            inplace: Whether to modify the input DataFrame or return a new one
            
        Returns:
            Filtered DataFrame or None if error occurred
        """
        try:
            if len(conditions) == 0:
                logger.warning("No filter conditions provided")
                return df if inplace else df.copy()
            
            self.filter_conditions[dataset_name] = conditions
            
            original_count = len(df)
            
            mask = None
            current_mask = None
            
            max_conditions = self.config.get('processor', {}).get('max_filter_conditions', 20)
            if len(conditions) > max_conditions:
                logger.warning(f"Number of filter conditions ({len(conditions)}) exceeds recommended maximum ({max_conditions})")
            
            for i, condition in enumerate(conditions):
                column = condition.get('column')
                operator = condition.get('operator')
                value = condition.get('value')
                logic = condition.get('logic', 'AND')
                
                if column not in df.columns:
                    logger.error(f"Column {column} not found in dataset")
                    self.processingError.emit(f"Column '{column}' not found in dataset")
                    continue
                
                if len(conditions) > 5:
                    progress = int((i / len(conditions)) * 100)
                    self.progressUpdated.emit(progress, f"Applying filter condition {i+1}/{len(conditions)}")
                
                try:
                    logger.debug(f"Applying condition {i}: {column} {operator} {value} (logic: {logic})")
                    current_mask = self._apply_comparison(df, column, operator, value)
                    logger.debug(f"Condition {i} result: {current_mask.sum()} rows match out of {len(current_mask)}")
                    
                    if mask is None:
                        mask = current_mask
                        logger.debug(f"Initial mask: {mask.sum()} rows")
                    elif logic.upper() == 'AND':
                        mask = mask & current_mask
                        logger.debug(f"After AND: {mask.sum()} rows")
                    elif logic.upper() == 'OR':
                        mask = mask | current_mask
                        logger.debug(f"After OR: {mask.sum()} rows")
                    else:
                        logger.warning(f"Unknown logic operator: {logic}, using AND")
                        mask = mask & current_mask
                        logger.debug(f"After default AND: {mask.sum()} rows")
                        
                except Exception as e:
                    error_msg = f"Error applying condition on column '{column}': {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    self.processingError.emit(error_msg)
                    continue
            
            if mask is not None:
                if inplace:
                    result_df = df[mask]
                    df.drop(df.index, inplace=True)
                    df = df.append(result_df, ignore_index=True)
                    filtered_df = df
                else:
                    filtered_df = df[mask].copy()
                
                filtered_count = len(filtered_df)
                reduction = original_count - filtered_count
                reduction_pct = (reduction / original_count) * 100 if original_count > 0 else 0
                
                if reduction == 0:
                    logger.warning(f"Filter on {dataset_name}: NO ROWS REMOVED - {original_count} → {filtered_count} rows "
                                 f"(Check if filter conditions match any data)")
                else:
                    logger.info(f"Filtered {dataset_name}: {original_count} → {filtered_count} rows "
                               f"(removed {reduction} rows, {reduction_pct:.2f}%)")
                
                self.processing_history[dataset_name] = self.processing_history.get(dataset_name, [])
                self.processing_history[dataset_name].append({
                    'operation': 'filter',
                    'conditions': conditions,
                    'rows_before': original_count,
                    'rows_after': filtered_count
                })
                
                filtered_df = optimize_dataframe(filtered_df)
                
                self.processingComplete.emit(dataset_name, filtered_df)
                return filtered_df
            else:
                logger.warning("No valid filter conditions could be applied")
                self.processingError.emit("No valid filter conditions could be applied")
                return df if inplace else df.copy()
                
        except Exception as e:
            error_msg = f"Error filtering data: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.processingError.emit(error_msg)
            return None
    
    def _apply_comparison(self, df: pd.DataFrame, column: str, operator: str, value: Any) -> pd.Series:
        """
        Apply a comparison operation to create a boolean mask
        
        Args:
            df: DataFrame to operate on
            column: Column name
            operator: Comparison operator
            value: Value to compare against
            
        Returns:
            Boolean Series (mask) with the comparison result
        """
        col_type = df[column].dtype
        
        if pd.api.types.is_object_dtype(col_type) or pd.api.types.is_string_dtype(col_type):
            if operator == '=':
                return df[column].astype(str) == str(value)
            elif operator == '!=':
                return df[column].astype(str) != str(value)
            elif operator == 'contains':
                return df[column].astype(str).str.contains(str(value), na=False)
            elif operator == 'starts_with':
                return df[column].astype(str).str.startswith(str(value), na=False)
            elif operator == 'ends_with':
                return df[column].astype(str).str.endswith(str(value), na=False)
            elif operator == 'is_null':
                return df[column].isna()
            elif operator == 'is_not_null':
                return ~df[column].isna()
            else:
                raise ValueError(f"Unsupported operator '{operator}' for string column")
        
        elif pd.api.types.is_numeric_dtype(col_type):
            if value == 'null' and operator in ['=', '!=']:
                if operator == '=':
                    return df[column].isna()
                else:  # !=
                    return ~df[column].isna()
            else:
                try:
                    if pd.api.types.is_integer_dtype(col_type):
                        if str(col_type).startswith('uint8') and isinstance(value, str) and value.upper() in ['Y', 'N', 'YES', 'NO']:
                            num_value = 1 if value.upper() in ['Y', 'YES'] else 0
                        else:
                            num_value = int(value)
                    else:
                        num_value = float(value)
                    
                    if operator == '>':
                        return df[column] > num_value
                    elif operator == '>=':
                        return df[column] >= num_value
                    elif operator == '<':
                        return df[column] < num_value
                    elif operator == '<=':
                        return df[column] <= num_value
                    elif operator == '==':
                        return df[column] == num_value
                    elif operator == '!=':
                        return df[column] != num_value
                    elif operator == 'is_null':
                        return df[column].isna()
                    elif operator == 'is_not_null':
                        return ~df[column].isna()
                    else:
                        raise ValueError(f"Unsupported operator '{operator}' for numeric column")
                except (ValueError, TypeError):
                    if operator in ['==', '!=']:
                        logger.warning(f"Failed to convert '{value}' to number for column '{column}', treating as categorical comparison")
                        if operator == '==':
                            return df[column].astype(str) == str(value)
                        else:  # !=
                            return df[column].astype(str) != str(value)
                    else:
                        raise ValueError(f"Could not convert '{value}' to a number for comparison")
        
        elif pd.api.types.is_categorical_dtype(col_type):
            if operator == '==' or operator == '=':
                return df[column] == value
            elif operator == '!=':
                return df[column] != value
            elif operator == 'is_null':
                return df[column].isna()
            elif operator == 'is_not_null':
                return ~df[column].isna()
            else:
                raise ValueError(f"Unsupported operator '{operator}' for categorical column")
        
        elif pd.api.types.is_datetime64_dtype(col_type):
            try:
                date_value = pd.to_datetime(value)
                
                if operator == '>':
                    return df[column] > date_value
                elif operator == '>=':
                    return df[column] >= date_value
                elif operator == '<':
                    return df[column] < date_value
                elif operator == '<=':
                    return df[column] <= date_value
                elif operator == '==' or operator == '=':
                    return df[column] == date_value
                elif operator == '!=':
                    return df[column] != date_value
                elif operator == 'is_null':
                    return df[column].isna()
                elif operator == 'is_not_null':
                    return ~df[column].isna()
                else:
                    raise ValueError(f"Unsupported operator '{operator}' for date column")
            except Exception:
                raise ValueError(f"Could not convert '{value}' to a date for comparison")
        
        elif pd.api.types.is_bool_dtype(col_type):
            if isinstance(value, str):
                if value.lower() in ['true', 'yes', 'y', '1']:
                    bool_value = True
                elif value.lower() in ['false', 'no', 'n', '0']:
                    bool_value = False
                else:
                    raise ValueError(f"Could not convert '{value}' to a boolean")
            else:
                bool_value = bool(value)
                
            if operator == '==' or operator == '=':
                return df[column] == bool_value
            elif operator == '!=':
                return df[column] != bool_value
            elif operator == 'is_null':
                return df[column].isna()
            elif operator == 'is_not_null':
                return ~df[column].isna()
            else:
                raise ValueError(f"Unsupported operator '{operator}' for boolean column")
        
        else:
            if operator == '==' or operator == '=':
                return df[column].astype(str) == str(value)
            elif operator == '!=':
                return df[column].astype(str) != str(value)
            elif operator == 'contains':
                return df[column].astype(str).str.contains(str(value), na=False)
            elif operator == 'starts_with':
                return df[column].astype(str).str.startswith(str(value), na=False)
            elif operator == 'ends_with':
                return df[column].astype(str).str.endswith(str(value), na=False)
            elif operator == 'is_null':
                return df[column].isna()
            elif operator == 'is_not_null':
                return ~df[column].isna()
            else:
                raise ValueError(f"Unsupported operator '{operator}' for column type {col_type}")
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: Dict[str, Any], 
                             dataset_name: str = "dataset") -> pd.DataFrame:
        """
        Handle missing values in the DataFrame
        
        Args:
            df: Input DataFrame
            strategy: Dictionary defining the strategy:
                - method: 'remove_rows', 'fill_constant', 'fill_mean', 'fill_median', 'fill_mode'
                - columns: List of columns to apply to (or 'all')
                - fill_value: Value to use for 'fill_constant'
            dataset_name: Name of the dataset for tracking
            
        Returns:
            Processed DataFrame
        """
        try:
            method = strategy.get('method', 'remove_rows')
            columns = strategy.get('columns', 'all')
            fill_value = strategy.get('fill_value', 0)
            
            result_df = df.copy()
            
            original_count = len(df)
            missing_counts = {col: df[col].isna().sum() for col in df.columns}
            total_missing = sum(missing_counts.values())
            
            logger.info(f"Handling missing values in {dataset_name}: {total_missing} missing values across {len(missing_counts)} columns")
            
            cols_to_process = df.columns if columns == 'all' else columns
            
            if method == 'remove_rows':
                result_df = result_df.dropna(subset=cols_to_process)
                
            elif method == 'fill_constant':
                for col in cols_to_process:
                    if col in result_df.columns and result_df[col].isna().any():
                        if pd.api.types.is_numeric_dtype(result_df[col].dtype):
                            try:
                                typed_value = float(fill_value) if '.' in str(fill_value) else int(fill_value)
                                result_df[col].fillna(typed_value, inplace=True)
                            except (ValueError, TypeError):
                                logger.warning(f"Could not convert '{fill_value}' to a number for column {col}")
                                result_df[col].fillna(fill_value, inplace=True)
                        else:
                            result_df[col].fillna(str(fill_value), inplace=True)
            
            elif method == 'fill_mean':
                for col in cols_to_process:
                    if col in result_df.columns and result_df[col].isna().any():
                        if pd.api.types.is_numeric_dtype(result_df[col].dtype):
                            result_df[col].fillna(result_df[col].mean(), inplace=True)
                        else:
                            logger.warning(f"Cannot use mean imputation for non-numeric column {col}")
            
            elif method == 'fill_median':
                for col in cols_to_process:
                    if col in result_df.columns and result_df[col].isna().any():
                        if pd.api.types.is_numeric_dtype(result_df[col].dtype):
                            result_df[col].fillna(result_df[col].median(), inplace=True)
                        else:
                            logger.warning(f"Cannot use median imputation for non-numeric column {col}")
            
            elif method == 'fill_mode':
                for col in cols_to_process:
                    if col in result_df.columns and result_df[col].isna().any():
                        mode_value = result_df[col].mode()
                        if not mode_value.empty:
                            result_df[col].fillna(mode_value[0], inplace=True)
            
            else:
                logger.warning(f"Unknown missing value handling method: {method}")
                return df
            
            final_count = len(result_df)
            removed_rows = original_count - final_count
            new_missing_counts = {col: result_df[col].isna().sum() for col in result_df.columns}
            total_new_missing = sum(new_missing_counts.values())
            
            logger.info(f"Missing value handling results: {removed_rows} rows removed, "
                       f"{total_missing - total_new_missing} missing values resolved")
            
            self.processing_history[dataset_name] = self.processing_history.get(dataset_name, [])
            self.processing_history[dataset_name].append({
                'operation': 'handle_missing',
                'strategy': strategy,
                'rows_before': original_count,
                'rows_after': final_count,
                'missing_before': total_missing,
                'missing_after': total_new_missing
            })
            
            result_df = optimize_dataframe(result_df)
            
            self.processingComplete.emit(dataset_name, result_df)
            return result_df
            
        except Exception as e:
            error_msg = f"Error handling missing values: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.processingError.emit(error_msg)
            return df
    
    def transform_column(self, df: pd.DataFrame, column: str, transformation: Dict[str, Any],
                        new_column: Optional[str] = None, dataset_name: str = "dataset") -> pd.DataFrame:
        """
        Apply a transformation to a column
        
        Args:
            df: Input DataFrame
            column: Column to transform
            transformation: Dictionary defining the transformation:
                - type: Type of transformation ('log', 'sqrt', 'normalize', 'encode', etc.)
                - params: Additional parameters specific to the transformation
            new_column: Name for the new column (if None, overwrite existing)
            dataset_name: Name of the dataset for tracking
            
        Returns:
            DataFrame with transformed column
        """
        try:
            if column not in df.columns:
                error_msg = f"Column {column} not found in dataset"
                logger.error(error_msg)
                self.processingError.emit(error_msg)
                return df
            
            result_df = df.copy()
            
            output_column = new_column if new_column else column
            
            trans_type = transformation.get('type', '').lower()
            params = transformation.get('params', {})
            
            if trans_type == 'log':
                base = params.get('base', 10)
                
                if pd.api.types.is_numeric_dtype(df[column].dtype):
                    offset = params.get('offset', 0)
                    min_val = df[column].min()
                    
                    if min_val <= 0:
                        if offset == 0:
                            offset = abs(min_val) + 1
                            logger.info(f"Auto-calculated offset {offset} for log transform to handle non-positive values")
                        
                        result_df[output_column] = np.log(df[column] + offset) / np.log(base)
                    else:
                        result_df[output_column] = np.log(df[column]) / np.log(base)
                else:
                    logger.warning(f"Cannot apply log transformation to non-numeric column {column}")
                    return df
            
            elif trans_type == 'sqrt':
                if pd.api.types.is_numeric_dtype(df[column].dtype):
                    offset = params.get('offset', 0)
                    min_val = df[column].min()
                    
                    if min_val < 0:
                        if offset == 0:
                            offset = abs(min_val)
                            logger.info(f"Auto-calculated offset {offset} for sqrt transform to handle negative values")
                        
                        result_df[output_column] = np.sqrt(df[column] + offset)
                    else:
                        result_df[output_column] = np.sqrt(df[column])
                else:
                    logger.warning(f"Cannot apply sqrt transformation to non-numeric column {column}")
                    return df
            
            elif trans_type == 'normalize':
                if pd.api.types.is_numeric_dtype(df[column].dtype):
                    min_val = df[column].min()
                    max_val = df[column].max()
                    
                    if min_val == max_val:
                        logger.warning(f"Cannot normalize column {column} with identical min and max values")
                        result_df[output_column] = 0  # All values map to 0 if min=max
                    else:
                        result_df[output_column] = (df[column] - min_val) / (max_val - min_val)
                else:
                    logger.warning(f"Cannot apply normalization to non-numeric column {column}")
                    return df
            
            elif trans_type == 'standardize':
                if pd.api.types.is_numeric_dtype(df[column].dtype):
                    mean_val = df[column].mean()
                    std_val = df[column].std()
                    
                    if std_val == 0:
                        logger.warning(f"Cannot standardize column {column} with zero standard deviation")
                        result_df[output_column] = 0  # All values map to 0 if std=0
                    else:
                        result_df[output_column] = (df[column] - mean_val) / std_val
                else:
                    logger.warning(f"Cannot apply standardization to non-numeric column {column}")
                    return df
            
            elif trans_type == 'encode':
                # Categorical encoding
                encode_type = params.get('method', 'one-hot')
                
                if encode_type == 'one-hot':
                    # One-hot encoding (dummy variables)
                    dummies = pd.get_dummies(df[column], prefix=column)
                    
                    for dummy_col in dummies.columns:
                        result_df[dummy_col] = dummies[dummy_col]
                
                elif encode_type == 'label':
                    # Label encoding (convert categories to integers)
                    categories = df[column].astype('category')
                    result_df[output_column] = categories.cat.codes
                
                elif encode_type == 'binary':
                    # Binary encoding for binary columns
                    if df[column].nunique() <= 2:
                        unique_vals = df[column].dropna().unique()
                        if len(unique_vals) == 2:
                            mapping = params.get('mapping', {})
                            if not mapping:
                                mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                            
                            result_df[output_column] = df[column].map(mapping)
                        elif len(unique_vals) == 1:
                            logger.warning(f"Column {column} has only one unique value, encoding as all 0s")
                            result_df[output_column] = 0
                    else:
                        logger.warning(f"Cannot apply binary encoding to column {column} with >2 unique values")
                        return df
            
            elif trans_type == 'bin':
                num_bins = params.get('bins', 5)
                bin_method = params.get('method', 'equal_width')
                labels = params.get('labels', None)
                
                if pd.api.types.is_numeric_dtype(df[column].dtype):
                    if bin_method == 'equal_width':
                        result_df[output_column] = pd.cut(df[column], bins=num_bins, labels=labels)
                    elif bin_method == 'equal_freq':
                        result_df[output_column] = pd.qcut(df[column], q=num_bins, labels=labels, duplicates='drop')
                    elif bin_method == 'custom':
                        bin_edges = params.get('bin_edges', [])
                        if len(bin_edges) >= 2:
                            result_df[output_column] = pd.cut(df[column], bins=bin_edges, labels=labels)
                        else:
                            logger.warning("Custom binning requires at least 2 bin edges")
                            return df
                else:
                    logger.warning(f"Cannot apply binning to non-numeric column {column}")
                    return df
            
            else:
                logger.warning(f"Unknown transformation type: {trans_type}")
                return df
            
            self.processing_history[dataset_name] = self.processing_history.get(dataset_name, [])
            self.processing_history[dataset_name].append({
                'operation': 'transform_column',
                'column': column,
                'output_column': output_column,
                'transformation': transformation
            })
            
            result_df = optimize_dataframe(result_df)
            
            self.processingComplete.emit(dataset_name, result_df)
            return result_df
            
        except Exception as e:
            error_msg = f"Error transforming column {column}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.processingError.emit(error_msg)
            return df
    
    def get_processing_history(self, dataset_name: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get the processing history for a dataset
        
        Args:
            dataset_name: Name of the dataset (or None for all)
            
        Returns:
            Dictionary containing the processing history
        """
        if dataset_name:
            return {dataset_name: self.processing_history.get(dataset_name, [])}
        else:
            return self.processing_history
