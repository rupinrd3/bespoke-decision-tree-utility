#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Engineering Module for Bespoke Utility
Handles creation of new variables through formulas and transformations

"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import pandas as pd
import numpy as np
import re
from PyQt5.QtCore import QObject, pyqtSignal

from utils.memory_management import optimize_dataframe

logger = logging.getLogger(__name__)

class FormulaParser:
    """Class for parsing and evaluating formulas to create new variables"""
    
    def __init__(self):
        """Initialize the formula parser"""
        self.operators = {
            '+': np.add,
            '-': np.subtract,
            '*': np.multiply,
            '/': np.divide,
            '^': np.power,
            '%': np.mod
        }
        
        self.functions = {
            'LOG': np.log10,
            'LN': np.log,
            'EXP': np.exp,
            'SQRT': np.sqrt,
            'ABS': np.abs,
            'ROUND': np.round,
            'FLOOR': np.floor,
            'CEIL': np.ceil,
            'SIN': np.sin,
            'COS': np.cos,
            'TAN': np.tan,
            'MIN': np.minimum,
            'MAX': np.maximum,
            'MEAN': lambda *args: np.mean(np.column_stack(args), axis=1),
            'SUM': lambda *args: np.sum(np.column_stack(args), axis=1)
        }
        
        self.patterns = {
            'variable': r'[A-Za-z][A-Za-z0-9_]*',
            'number': r'-?\d+(\.\d+)?',
            'function': r'[A-Z]+\(',
            'operators': r'[\+\-\*/\^%]',
            'parentheses': r'[\(\)]',
            'comma': r',',
            'whitespace': r'\s+'
        }
    
    def tokenize(self, formula: str) -> List[str]:
        """
        Split formula into tokens
        
        Args:
            formula: Formula string
            
        Returns:
            List of tokens
        """
        pattern = '|'.join(f'({p})' for p in self.patterns.values())
        
        tokens = []
        for match in re.finditer(pattern, formula):
            token = match.group(0)
            if not re.match(self.patterns['whitespace'], token):  # Skip whitespace
                tokens.append(token)
        
        return tokens
    
    def validate_syntax(self, formula: str) -> Tuple[bool, str]:
        """
        Check if formula has valid syntax
        
        Args:
            formula: Formula string
            
        Returns:
            Tuple with (is_valid, error_message)
        """
        tokens = self.tokenize(formula)
        
        paren_count = 0
        for token in tokens:
            if token == '(':
                paren_count += 1
            elif token == ')':
                paren_count -= 1
                
            if paren_count < 0:
                return False, "Unbalanced parentheses: too many closing parentheses"
        
        if paren_count > 0:
            return False, "Unbalanced parentheses: missing closing parentheses"
        
        func_pattern = re.compile(r'([A-Z]+)\(')
        for token in tokens:
            match = func_pattern.match(token)
            if match:
                func_name = match.group(1)
                if func_name not in self.functions:
                    return False, f"Unknown function: {func_name}"
        
        for i in range(len(tokens) - 1):
            current = tokens[i]
            next_token = tokens[i + 1]
            
            if current in self.operators and next_token in self.operators and next_token != '-':
                return False, f"Invalid operator sequence: {current} {next_token}"
            
            if current == ')' and (re.match(self.patterns['variable'], next_token) or 
                                  re.match(self.patterns['number'], next_token)):
                return False, f"Missing operator between {current} and {next_token}"
            
            if (re.match(self.patterns['variable'], current) or 
                re.match(self.patterns['number'], current)) and next_token == '(':
                return False, f"Missing operator between {current} and {next_token}"
        
        return True, ""
    
    def evaluate(self, formula: str, df: pd.DataFrame) -> pd.Series:
        """
        Evaluate formula on dataframe
        
        Args:
            formula: Formula string
            df: Dataframe with variables
            
        Returns:
            Series with formula result
        """
        if 'IF(' in formula.upper():
            return self._evaluate_if_function(formula, df)
        
        for col in df.columns:
            pattern = r'\b' + re.escape(col) + r'\b'
            formula = re.sub(pattern, f"df['{col}']", formula)
        
        for func_name, func in self.functions.items():
            formula = formula.replace(f"{func_name}(", f"self.functions['{func_name}'](")
        
        formula = formula.replace('^', '**')  # Power operator
        
        try:
            result = eval(formula)
            return result
        except Exception as e:
            logger.error(f"Error evaluating formula: {formula}", exc_info=True)
            raise ValueError(f"Failed to evaluate formula: {str(e)}")
    
    def _evaluate_if_function(self, formula: str, df: pd.DataFrame) -> pd.Series:
        """
        Evaluate IF statements in formula
        
        Args:
            formula: Formula with IF statements
            df: Dataframe with variables
            
        Returns:
            Series with result
        """
        if_pattern = r'IF\s*\(\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*\)'
        
        matches = re.finditer(if_pattern, formula, re.IGNORECASE)
        
        result_formula = formula
        for match in matches:
            if_expr = match.group(0)
            condition = match.group(1)
            true_val = match.group(2)
            false_val = match.group(3)
            
            for col in df.columns:
                pattern = r'\b' + re.escape(col) + r'\b'
                condition = re.sub(pattern, f"df['{col}']", condition)
            
            for func_name, func in self.functions.items():
                condition = condition.replace(f"{func_name}(", f"self.functions['{func_name}'](")
            
            condition = condition.replace('^', '**')
            
            try:
                condition_result = eval(condition)
                
                np_statement = f"np.where({condition_result}, {true_val}, {false_val})"
                
                result_formula = result_formula.replace(if_expr, np_statement)
            except Exception as e:
                logger.error(f"Error evaluating IF condition: {condition}", exc_info=True)
                raise ValueError(f"Failed to evaluate IF condition: {str(e)}")
        
        return self.evaluate(result_formula, df)


class FeatureEngineering(QObject):
    """Class for creating new variables and features"""
    
    featureCreated = pyqtSignal(str, str)  # dataset_name, column_name
    featureError = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize FeatureEngineering with configuration"""
        super().__init__()
        self.config = config or {}
        self.formula_parser = FormulaParser()
        
        self.formula_history = {}
    
    def create_formula_variable(self, df: pd.DataFrame, formula: str, var_name: str, 
                              var_type: str = 'float', dataset_name: str = "dataset") -> pd.DataFrame:
        """
        Create a new variable based on a formula
        
        Args:
            df: Input DataFrame
            formula: Formula string defining the variable
            var_name: Name for the new variable
            var_type: Data type for the new variable (float, int, str, category)
            dataset_name: Name of the dataset for tracking
            
        Returns:
            DataFrame with the new variable added
        """
        try:
            is_valid, error_msg = self.formula_parser.validate_syntax(formula)
            if not is_valid:
                logger.error(f"Invalid formula syntax: {error_msg}")
                self.featureError.emit(f"Invalid formula syntax: {error_msg}")
                return df
            
            if var_name in df.columns:
                logger.warning(f"Column {var_name} already exists, will be replaced")
            
            result_df = df.copy()
            
            try:
                result = self.formula_parser.evaluate(formula, df)
                
                if var_type == 'float':
                    result = pd.to_numeric(result, errors='coerce')
                elif var_type == 'int':
                    result = pd.to_numeric(result, errors='coerce').astype('Int64')  # Nullable integer
                elif var_type == 'str':
                    result = result.astype(str)
                elif var_type == 'category':
                    result = result.astype('category')
                
                result_df[var_name] = result
                
                self.formula_history[dataset_name] = self.formula_history.get(dataset_name, {})
                self.formula_history[dataset_name][var_name] = {
                    'formula': formula,
                    'type': var_type
                }
                
                result_df = optimize_dataframe(result_df)
                
                logger.info(f"Created new variable '{var_name}' with formula: {formula}")
                self.featureCreated.emit(dataset_name, var_name)
                
                return result_df
                
            except Exception as e:
                error_msg = f"Error evaluating formula: {str(e)}"
                logger.error(error_msg, exc_info=True)
                self.featureError.emit(error_msg)
                return df
                
        except Exception as e:
            error_msg = f"Error creating variable: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.featureError.emit(error_msg)
            return df
    
    def create_binned_variable(self, df: pd.DataFrame, source_column: str, var_name: str,
                            num_bins: int = 5, bin_method: str = 'equal_width', 
                            labels: Optional[List[str]] = None, dataset_name: str = "dataset") -> pd.DataFrame:
        """
        Create a binned variable from a numeric column
        
        Args:
            df: Input DataFrame
            source_column: Column to bin
            var_name: Name for the new binned variable
            num_bins: Number of bins
            bin_method: Binning method ('equal_width', 'equal_freq', 'custom')
            labels: Labels for the bins (optional)
            dataset_name: Name of the dataset for tracking
            
        Returns:
            DataFrame with the new variable added
        """
        try:
            if source_column not in df.columns:
                error_msg = f"Column {source_column} not found in dataset"
                logger.error(error_msg)
                self.featureError.emit(error_msg)
                return df
                
            if var_name in df.columns:
                logger.warning(f"Column {var_name} already exists, will be replaced")
            
            if not pd.api.types.is_numeric_dtype(df[source_column].dtype):
                error_msg = f"Column {source_column} must be numeric for binning"
                logger.error(error_msg)
                self.featureError.emit(error_msg)
                return df
            
            result_df = df.copy()
            
            if bin_method == 'equal_width':
                result_df[var_name] = pd.cut(df[source_column], bins=num_bins, labels=labels)
                bin_edges = pd.cut(df[source_column], bins=num_bins).categories.to_tuples()
            
            elif bin_method == 'equal_freq':
                result_df[var_name] = pd.qcut(df[source_column], q=num_bins, labels=labels, duplicates='drop')
                bin_edges = pd.qcut(df[source_column], q=num_bins, duplicates='drop').categories.to_tuples()
            
            elif bin_method == 'custom':
                custom_bins = self.config.get('feature_engineering', {}).get('custom_bins', [])
                
                if custom_bins:
                    result_df[var_name] = pd.cut(df[source_column], bins=custom_bins, labels=labels)
                    bin_edges = custom_bins
                else:
                    percentiles = np.linspace(0, 100, num_bins + 1)
                    bin_edges = np.percentile(df[source_column].dropna(), percentiles)
                    result_df[var_name] = pd.cut(df[source_column], bins=bin_edges, labels=labels)
            
            else:
                error_msg = f"Unknown binning method: {bin_method}"
                logger.error(error_msg)
                self.featureError.emit(error_msg)
                return df
            
            self.formula_history[dataset_name] = self.formula_history.get(dataset_name, {})
            self.formula_history[dataset_name][var_name] = {
                'type': 'binned',
                'source_column': source_column,
                'bin_method': bin_method,
                'num_bins': num_bins,
                'bin_edges': bin_edges
            }
            
            result_df = optimize_dataframe(result_df)
            
            logger.info(f"Created binned variable '{var_name}' from {source_column} with {bin_method} method")
            self.featureCreated.emit(dataset_name, var_name)
            
            return result_df
            
        except Exception as e:
            error_msg = f"Error creating binned variable: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.featureError.emit(error_msg)
            return df
    
    def create_interaction_term(self, df: pd.DataFrame, columns: List[str], var_name: str,
                               interaction_type: str = 'multiply', dataset_name: str = "dataset") -> pd.DataFrame:
        """
        Create an interaction term between multiple columns
        
        Args:
            df: Input DataFrame
            columns: List of columns to combine
            var_name: Name for the new interaction variable
            interaction_type: Type of interaction ('multiply', 'add', 'subtract', 'divide')
            dataset_name: Name of the dataset for tracking
            
        Returns:
            DataFrame with the new variable added
        """
        try:
            for col in columns:
                if col not in df.columns:
                    error_msg = f"Column {col} not found in dataset"
                    logger.error(error_msg)
                    self.featureError.emit(error_msg)
                    return df
            
            for col in columns:
                if not pd.api.types.is_numeric_dtype(df[col].dtype):
                    error_msg = f"Column {col} must be numeric for interaction terms"
                    logger.error(error_msg)
                    self.featureError.emit(error_msg)
                    return df
            
            if var_name in df.columns:
                logger.warning(f"Column {var_name} already exists, will be replaced")
            
            result_df = df.copy()
            
            if interaction_type == 'multiply':
                result = df[columns[0]].copy()
                for col in columns[1:]:
                    result *= df[col]
                
                formula = ' * '.join(columns)
                
            elif interaction_type == 'add':
                result = df[columns[0]].copy()
                for col in columns[1:]:
                    result += df[col]
                
                formula = ' + '.join(columns)
                
            elif interaction_type == 'subtract':
                result = df[columns[0]].copy()
                for col in columns[1:]:
                    result -= df[col]
                
                formula = ' - '.join(columns)
                
            elif interaction_type == 'divide':
                result = df[columns[0]].copy()
                for col in columns[1:]:
                    denominator = df[col].replace(0, np.nan)
                    result /= denominator
                
                formula = ' / '.join(columns)
                
            else:
                error_msg = f"Unknown interaction type: {interaction_type}"
                logger.error(error_msg)
                self.featureError.emit(error_msg)
                return df
            
            result_df[var_name] = result
            
            self.formula_history[dataset_name] = self.formula_history.get(dataset_name, {})
            self.formula_history[dataset_name][var_name] = {
                'type': 'interaction',
                'columns': columns,
                'interaction_type': interaction_type,
                'formula': formula
            }
            
            result_df = optimize_dataframe(result_df)
            
            logger.info(f"Created interaction variable '{var_name}' using {interaction_type} from {columns}")
            self.featureCreated.emit(dataset_name, var_name)
            
            return result_df
            
        except Exception as e:
            error_msg = f"Error creating interaction term: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.featureError.emit(error_msg)
            return df
    
    def create_binary_variable(self, df: pd.DataFrame, condition: Dict[str, Any], var_name: str,
                              dataset_name: str = "dataset") -> pd.DataFrame:
        """
        Create a binary variable (0/1) based on a condition
        
        Args:
            df: Input DataFrame
            condition: Dictionary defining the condition:
                - column: Column to check
                - operator: Comparison operator ('>', '<', '=', etc.)
                - value: Value to compare against
            var_name: Name for the new binary variable
            dataset_name: Name of the dataset for tracking
            
        Returns:
            DataFrame with the new variable added
        """
        try:
            column = condition.get('column')
            operator = condition.get('operator')
            value = condition.get('value')
            
            if column not in df.columns:
                error_msg = f"Column {column} not found in dataset"
                logger.error(error_msg)
                self.featureError.emit(error_msg)
                return df
            
            if var_name in df.columns:
                logger.warning(f"Column {var_name} already exists, will be replaced")
            
            result_df = df.copy()
            
            if operator == '>':
                formula = f"IF({column} > {value}, 1, 0)"
            elif operator == '>=':
                formula = f"IF({column} >= {value}, 1, 0)"
            elif operator == '<':
                formula = f"IF({column} < {value}, 1, 0)"
            elif operator == '<=':
                formula = f"IF({column} <= {value}, 1, 0)"
            elif operator == '=':
                formula = f"IF({column} = {value}, 1, 0)"
            elif operator == '!=':
                formula = f"IF({column} != {value}, 1, 0)"
            elif operator == 'contains':
                if pd.api.types.is_string_dtype(df[column].dtype) or pd.api.types.is_object_dtype(df[column].dtype):
                    result_df[var_name] = df[column].str.contains(str(value), na=False).astype(int)
                    return result_df
                else:
                    error_msg = "Contains operator only works with string columns"
                    logger.error(error_msg)
                    self.featureError.emit(error_msg)
                    return df
            else:
                error_msg = f"Unsupported operator: {operator}"
                logger.error(error_msg)
                self.featureError.emit(error_msg)
                return df
            
            result = self.formula_parser.evaluate(formula, df)
            result_df[var_name] = result
            
            self.formula_history[dataset_name] = self.formula_history.get(dataset_name, {})
            self.formula_history[dataset_name][var_name] = {
                'type': 'binary',
                'condition': condition,
                'formula': formula
            }
            
            result_df = optimize_dataframe(result_df)
            
            logger.info(f"Created binary variable '{var_name}' with condition: {column} {operator} {value}")
            self.featureCreated.emit(dataset_name, var_name)
            
            return result_df
            
        except Exception as e:
            error_msg = f"Error creating binary variable: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.featureError.emit(error_msg)
            return df
            
    def get_formula_history(self, dataset_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get the formula history for a dataset
        
        Args:
            dataset_name: Name of the dataset (or None for all)
            
        Returns:
            Dictionary containing the formula history
        """
        if dataset_name:
            return {dataset_name: self.formula_history.get(dataset_name, {})}
        else:
            return self.formula_history
