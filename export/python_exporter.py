#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct Tree Model Python Exporter for Retro Scoring
Extracts node_id, node_number, and node_logic directly from the decision tree model
Generates Python code for scoring new datasets with same field names
"""

import logging
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class GenericNodeExporter:
    """
    Direct Tree Model Python Exporter for Retro Scoring
    Extracts terminal node data directly from BespokeDecisionTree model
    Uses same node numbering and logic as the core utility (NOT CSV files)
    Generates Python code for scoring new datasets with retro scoring capability
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model_categorical_features = set()  # Store categorical features from model
        self.model_numerical_features = set()   # Store numerical features from model
        logger.info("Direct Tree Model Python Exporter initialized")
    
    def export_model_to_python(self, model, filepath: str, node_report_csv: str = None) -> bool:
        """
        Export decision tree model directly to Python retro scoring code
        
        Args:
            model: BespokeDecisionTree fitted model with tree structure
            filepath: Output Python file path for retro scoring code
            node_report_csv: IGNORED - uses direct tree access only
            
        Returns:
            True if successful retro scoring code generation
        """
        try:
            self._extract_feature_types_from_model(model)
            
            terminal_nodes = self._extract_terminal_node_conditions(model, node_report_csv)
            
            if not terminal_nodes:
                logger.error("No terminal nodes found - cannot generate code")
                return False
            
            logger.info(f"Found {len(terminal_nodes)} terminal nodes for code generation")
            
            python_code = self._generate_complete_python_code(model, terminal_nodes)
            python_code = python_code.replace('\\n\\n', '\n\n').replace('\n\\n', '\n')

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(python_code)

            logger.info(f"Successfully generated Python code with {len(terminal_nodes)} terminal nodes")
            return True
            
        except Exception as e:
            logger.error(f"Error in generic export: {str(e)}", exc_info=True)
            return False
    
    def _extract_feature_types_from_model(self, model) -> None:
        """
        Extract feature type information from the fitted model
        Critical for using correct default values in Python conditions
        """
        try:
            if hasattr(model, 'categorical_features') and model.categorical_features:
                self.model_categorical_features = set(model.categorical_features)
                logger.debug(f"Found {len(self.model_categorical_features)} categorical features")
            
            if hasattr(model, 'numerical_features') and model.numerical_features:
                self.model_numerical_features = set(model.numerical_features)
                logger.debug(f"Found {len(self.model_numerical_features)} numerical features")
                
            if hasattr(model, 'feature_names') and model.feature_names:
                if not self.model_categorical_features and not self.model_numerical_features:
                    self.model_numerical_features = set(model.feature_names)
                    logger.warning("No feature type info found - defaulting all features to numerical")
                    
        except Exception as e:
            logger.warning(f"Could not extract feature types from model: {e}")
            if hasattr(model, 'feature_names') and model.feature_names:
                self.model_numerical_features = set(model.feature_names)
    
    def _is_categorical_feature(self, feature_name: str) -> bool:
        """Check if a feature is categorical based on model information"""
        return feature_name in self.model_categorical_features
    
    def _get_feature_default_value(self, feature_name: str) -> str:
        """Get appropriate default value for a feature based on its type"""
        if self._is_categorical_feature(feature_name):
            return "''"  # Empty string for categorical
        else:
            return "0"   # Zero for numerical
    
    def _extract_terminal_node_conditions(self, model, node_report_csv: str = None) -> Dict[int, Dict]:
        """
        Extract terminal node conditions DIRECTLY from decision tree model
        Uses the same node numbering and logic extraction as the core utility
        
        Args:
            model: BespokeDecisionTree instance with fitted tree
            node_report_csv: IGNORED - direct tree access only
            
        Returns:
            {
                node_no: {
                    'python_condition': 'python_logic',
                    'logic_text': 'human-readable rule chain',
                    'metadata': {...},
                    'node_id': 'tree-node-id'
                }
            }
        """
        terminal_nodes = {}
        
        if not hasattr(model, 'root') or not model.root:
            logger.error("Model has no root node - cannot extract terminal conditions")
            return {}
            
        try:
            terminal_node_objects = self._extract_terminal_nodes_from_model(model.root)
            
            if not terminal_node_objects:
                logger.error("No terminal nodes found in decision tree")
                return {}
                
            node_number_mapping = self._create_node_number_mapping_direct(model.root)
            
            for node_obj in terminal_node_objects:
                node_number = node_number_mapping.get(node_obj.node_id, 0)
                if node_number == 0:
                    logger.warning(f"Could not find node number for terminal node {node_obj.node_id}")
                    continue
                    
                python_conditions, display_logic = self._build_node_path_conditions_direct(node_obj)
                
                metadata = self._extract_node_metadata_direct(node_obj, node_number)
                
                terminal_nodes[node_number] = {
                    'python_condition': python_conditions,
                    'logic_text': display_logic,
                    'metadata': metadata,
                    'node_id': node_obj.node_id
                }
                
            logger.info(f"Extracted {len(terminal_nodes)} terminal nodes directly from tree model")
            return terminal_nodes
            
        except Exception as e:
            logger.error(f"Error extracting terminal node conditions from tree model: {e}")
            return {}
    
    
    def _old_convert_logic_to_python(self, logic_str: str) -> str:
        """
        Convert any node logic string to Python conditions
        Handles any combination of variables and operators
        
        Examples:
        "[EXT_SOURCE_2] > 0.57 AND [EXT_SOURCE_3] <= 0.596" 
        -> "sample.get('EXT_SOURCE_2', 0) > 0.57 and sample.get('EXT_SOURCE_3', 0) <= 0.596"
        
        "[CODE_GENDER] IN ['F', 'XNA']"
        -> "sample.get('CODE_GENDER', '') in ['F', 'XNA']"
        """
        
        def replace_feature(match):
            feature = match.group(1)
            return f"sample.get('{feature}', 0)"
        
        python_logic = re.sub(r'\[([A-Z_0-9]+)\]', replace_feature, logic_str)
        
        python_logic = python_logic.replace(' AND ', ' and ')
        python_logic = python_logic.replace(' OR ', ' or ')
        
        in_pattern = r"sample\.get\('([^']+)', 0\) IN \[([^\]]+)\]"
        
        def replace_in_condition(match):
            feature = match.group(1)
            values_str = match.group(2)
            
            if "'" in values_str or '"' in values_str:
                quoted_values = re.findall(r"['\"]([^'\"]+)['\"]", values_str)
                if quoted_values:
                    values_list = quoted_values
                else:
                    values_list = [v.strip().strip("'\"") for v in values_str.split(',')]
            else:
                values_list = [values_str.strip()]
            
            return f"sample.get('{feature}', '') in {repr(values_list)}"
        
        python_logic = re.sub(in_pattern, replace_in_condition, python_logic)
        
        equals_pattern = r"sample\.get\('([^']+)', 0\) = (['\"][^'\"]+['\"]|[A-Za-z_][A-Za-z0-9_]*)"
        
        def replace_equals_condition(match):
            feature = match.group(1)
            value = match.group(2).strip("'\"")
            return f"sample.get('{feature}', '') == '{value}'"
        
        python_logic = re.sub(equals_pattern, replace_equals_condition, python_logic)
        
        return python_logic
    
    def _extract_from_tree_structure(self, root_node) -> Dict[int, Dict]:
        """
        Extract terminal nodes by traversing actual tree structure
        Works with any tree complexity
        """
        terminal_nodes = {}
        
        def traverse_node(node, path_conditions=None, node_id=1):
            if path_conditions is None:
                path_conditions = []
            
            if not node:
                return node_id
            
            is_terminal = getattr(node, 'is_terminal', True)
            
            if is_terminal:
                if path_conditions:
                    python_conditions = ' and '.join(path_conditions)
                else:
                    python_conditions = "True  # Root node"
                
                samples = getattr(node, 'samples', 0)
                target_count = getattr(node, 'target_count', 0)
                target_rate = (target_count / samples * 100) if samples > 0 else 0.0
                
                terminal_nodes[node_id] = {
                    'conditions': python_conditions,
                    'metadata': {
                        'samples': samples,
                        'target_count': target_count,
                        'target_rate': target_rate,
                        'description': f"Node {node_id}: {len(path_conditions)} conditions ({samples} samples, {target_rate:.1f}% target rate)"
                    }
                }
                return node_id + 1
            else:
                split_feature = getattr(node, 'split_feature', None)
                split_value = getattr(node, 'split_value', None)
                children = getattr(node, 'children', [])
                
                current_id = node_id + 1
                
                if split_feature and split_value is not None and children:
                    left_condition = f"sample.get('{split_feature}', 0) <= {split_value}"
                    right_condition = f"sample.get('{split_feature}', 0) > {split_value}"
                    
                    if len(children) > 0:
                        left_path = path_conditions + [left_condition]
                        current_id = traverse_node(children[0], left_path, current_id)
                    
                    if len(children) > 1:
                        right_path = path_conditions + [right_condition]
                        current_id = traverse_node(children[1], right_path, current_id)
                else:
                    for i, child in enumerate(children):
                        child_path = path_conditions.copy()
                        current_id = traverse_node(child, child_path, current_id)
                
                return current_id
        
        if root_node:
            traverse_node(root_node)
        
        return terminal_nodes
    
    def _extract_terminal_nodes_from_model(self, root_node) -> List:
        """
        Extract terminal node objects directly from tree structure
        Uses same method as NodeReportGenerator._extract_terminal_nodes
        """
        if not root_node:
            return []
        
        terminal_nodes = []
        visited = set()
        max_depth = 50  # Prevent infinite recursion
        
        def traverse(node, depth=0):
            if depth > max_depth:
                logger.warning("Maximum tree depth exceeded during traversal")
                return
            
            if id(node) in visited:
                logger.warning("Circular reference detected in tree structure")
                return
            
            visited.add(id(node))
            
            try:
                if hasattr(node, 'is_terminal') and node.is_terminal:
                    if hasattr(node, 'samples') and node.samples > 0:
                        terminal_nodes.append(node)
                    else:
                        logger.warning(f"Terminal node {getattr(node, 'node_id', 'unknown')} has zero samples - excluding")
                elif hasattr(node, 'children') and node.children:
                    for child in node.children:
                        if child:  # Ensure child is not None
                            traverse(child, depth + 1)
            except Exception as e:
                logger.warning(f"Error traversing node {getattr(node, 'node_id', 'unknown')}: {e}")
        
        traverse(root_node)
        return terminal_nodes
    
    def _create_node_number_mapping_direct(self, root_node) -> Dict[str, int]:
        """
        Create node number mapping using EXACT same algorithm as NodeReportGenerator
        This ensures perfect consistency between tree visualization and exported code
        """
        try:
            all_nodes = root_node.get_subtree_nodes()  # Gets ALL nodes (internal + terminal)
            
            if not all_nodes:
                logger.warning("No nodes found in tree for numbering")
                return {}
            
            node_number_mapping = {}
            for i, node in enumerate(all_nodes):
                node_number_mapping[node.node_id] = i + 1  # Sequential: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
            
            logger.debug(f"Created node number mapping: {len(node_number_mapping)} nodes")
            return node_number_mapping
            
        except Exception as e:
            logger.error(f"Error creating node number mapping: {e}")
            return {}
    
    def _build_node_path_conditions_direct(self, node) -> Tuple[str, str]:
        """
        Build Python and human-readable conditions directly from tree node path
        
        Returns:
            Tuple[str, str]: (python_condition, human_readable_condition)
        """
        try:
            python_conditions: List[str] = []
            text_conditions: List[str] = []
            current = node
            
            while current and current.parent:
                parent = current.parent
                python_cond, text_cond = self._build_parent_child_conditions_direct(parent, current)
                if python_cond:
                    python_conditions.append(python_cond)
                if text_cond:
                    text_conditions.append(text_cond)
                current = parent
            
            python_conditions.reverse()
            text_conditions.reverse()
            
            python_expr = " and ".join(python_conditions) if python_conditions else "True  # Root node"
            text_expr = " AND ".join(text_conditions) if text_conditions else "All records (root node)"
            return python_expr, text_expr
                
        except Exception as e:
            logger.warning(f"Could not build path conditions for node {getattr(node, 'node_id', 'unknown')}: {e}")
            return ("True  # Error building conditions", "Logic unavailable")
    
    def _extract_node_metadata_direct(self, node, node_number: int) -> Dict[str, Any]:
        """
        Extract node metadata directly from TreeNode object
        Uses same metadata extraction as NodeReportGenerator
        """
        try:
            node_size = getattr(node, 'samples', 0)
            class_counts = getattr(node, 'class_counts', {})
            
            target_count = 0
            if class_counts:
                if 1 in class_counts:
                    target_count = class_counts[1]
                elif True in class_counts:
                    target_count = class_counts[True]
                else:
                    target_count = sum(count for class_val, count in class_counts.items()
                                       if class_val in [1, True, 'Yes', 'yes', 'Y', 'y'])
                                     
            target_rate = (target_count * 100.0 / node_size) if node_size > 0 else 0.0
            
            return {
                'samples': node_size,
                'target_count': target_count,
                'target_rate': round(target_rate, 2),
                'lift': 0.0,  # Will be calculated if needed
                'description': f"Node {node_number}: {node_size} samples, {target_rate:.1f}% target rate",
                'class_counts': class_counts
            }
            
        except Exception as e:
            logger.warning(f"Error extracting metadata for node {getattr(node, 'node_id', 'unknown')}: {e}")
            return {
                'samples': 0,
                'target_count': 0,
                'target_rate': 0.0,
                'lift': 0.0,
                'description': f"Node {node_number}: Error extracting metadata"
            }
    
    def _format_value_for_display(self, value: Any) -> str:
        """Format values for human-readable logic strings."""
        try:
            if isinstance(value, float):
                return f"{value:.6g}"
            if isinstance(value, (int, bool)):
                return str(value)
            if value is None:
                return "NULL"
            if isinstance(value, str):
                return repr(value)
            if isinstance(value, (list, tuple, set)):
                formatted = ", ".join(self._format_value_for_display(v) for v in value)
                return f"[{formatted}]"
            return repr(value)
        except Exception:
            return repr(value)

    def _build_parent_child_conditions_direct(self, parent_node, child_node) -> Tuple[Optional[str], Optional[str]]:
        """
        Build condition strings for parent->child relationship
        Uses EXACT same logic as NodeReportGenerator._build_condition
        """
        try:
            feature = getattr(parent_node, 'split_feature', None)
            if not feature:
                return None, None
            
            split_type = getattr(parent_node, 'split_type', 'numeric')
            
            try:
                child_index = parent_node.children.index(child_node)
            except (ValueError, AttributeError):
                safe_feature = repr(feature)
                return (
                    f"sample.get({safe_feature}, 0)  # index not found",
                    f"[{feature}] split (child index unknown)"
                )
            
            default_value = self._get_feature_default_value(feature)
            
            safe_feature = repr(feature)
            feature_label = f"[{feature}]"
            
            if split_type == 'numeric':
                threshold = getattr(parent_node, 'split_value', 0)
                if child_index == 0:  # Left child
                    return (
                        f"sample.get({safe_feature}, {default_value}) <= {threshold}",
                        f"{feature_label} <= {self._format_value_for_display(threshold)}"
                    )
                else:  # Right child
                    return (
                        f"sample.get({safe_feature}, {default_value}) > {threshold}",
                        f"{feature_label} > {self._format_value_for_display(threshold)}"
                    )
                        
            elif split_type == 'numeric_multi_bin':
                if hasattr(parent_node, 'split_thresholds') and parent_node.split_thresholds:
                    thresholds = parent_node.split_thresholds
                    if child_index == 0:
                        upper = thresholds[0]
                        return (
                            f"sample.get({safe_feature}, {default_value}) <= {upper}",
                            f"{feature_label} <= {self._format_value_for_display(upper)}"
                        )
                    elif child_index == len(thresholds):
                        lower = thresholds[-1]
                        return (
                            f"sample.get({safe_feature}, {default_value}) > {lower}",
                            f"{feature_label} > {self._format_value_for_display(lower)}"
                        )
                    else:
                        lower = thresholds[child_index-1]
                        upper = thresholds[child_index]
                        return (
                            f"sample.get({safe_feature}, {default_value}) > {lower} and sample.get({safe_feature}, {default_value}) <= {upper}",
                            f"{feature_label} in ({self._format_value_for_display(lower)}, {self._format_value_for_display(upper)}]"
                        )
                else:
                    return (
                        f"sample.get({safe_feature}, {default_value})  # bin {child_index}",
                        f"{feature_label} bin {child_index}"
                    )
                        
            elif split_type == 'categorical':
                categories = getattr(parent_node, 'split_categories', {})
                if categories:
                    child_categories = [cat for cat, idx in categories.items() if idx == child_index]
                    if child_categories:
                        if len(child_categories) == 1:
                            category_value = repr(child_categories[0])  # Handles quotes and special chars
                            return (
                                f"sample.get({safe_feature}, {default_value}) == {category_value}",
                                f"{feature_label} == {self._format_value_for_display(child_categories[0])}"
                            )
                        else:
                            return (
                                f"sample.get({safe_feature}, {default_value}) in {repr(child_categories)}",
                                f"{feature_label} IN {self._format_value_for_display(child_categories)}"
                            )
                    else:
                        return (
                            f"sample.get({safe_feature}, {default_value})  # no categories for child {child_index}",
                            f"{feature_label} categorical branch {child_index}"
                        )
                else:
                    return (
                        f"sample.get({safe_feature}, {default_value})  # categorical",
                        f"{feature_label} categorical split"
                    )
            
            elif split_type == 'missing':
                if child_index == 0:  # Missing branch
                    return (
                        f"sample.get({safe_feature}, None) is None",
                        f"{feature_label} IS NULL"
                    )
                else:  # Not missing branch
                    return (
                        f"sample.get({safe_feature}, None) is not None",
                        f"{feature_label} IS NOT NULL"
                    )
            
            else:
                split_value = getattr(parent_node, 'split_value', 'N/A')
                return (
                    f"sample.get({safe_feature}, {default_value})  # {split_type} split",
                    f"{feature_label} {split_type} split on {self._format_value_for_display(split_value)}"
                )
        
        except Exception as e:
            logger.debug(f"Could not build condition: {e}")
            feature = getattr(parent_node, 'split_feature', 'Unknown')
            safe_feature = repr(feature)
            return (
                f"sample.get({safe_feature}, 0)  # condition parsing failed",
                f"[{feature}] split (unavailable)"
            )


    def _format_python_dict(self, data: Dict, indent_level: int = 1) -> str:
        """
        Format dictionary as valid Python code (not JSON)
        Handles proper boolean, None, and numeric formatting
        """
        indent = "    " * indent_level
        base_indent = "    " * (indent_level - 1)
        
        if not isinstance(data, dict):
            return self._format_python_value(data)
        
        if not data:
            return "{}"
        
        lines = ["{"]
        
        for key, value in data.items():
            if isinstance(key, str):
                key_str = repr(key)  # Handles quotes and escaping
            else:
                key_str = str(key)  # For integer keys
            
            value_str = self._format_python_value(value, indent_level + 1)
            
            lines.append(f"{indent}{key_str}: {value_str},")
        
        lines.append(f"{base_indent}}}")
        return "\n".join(lines)
    
    def _format_python_value(self, value: Any, indent_level: int = 1) -> str:
        """
        Format a Python value correctly (not JSON)
        """
        if value is None:
            return "None"
        elif isinstance(value, bool):
            return "True" if value else "False"  # Python boolean format
        elif isinstance(value, str):
            return repr(value)  # Handles quotes and escaping
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, (list, tuple)):
            if not value:
                return "[]" if isinstance(value, list) else "()"
            
            formatted_items = [self._format_python_value(item, indent_level) for item in value]
            
            if isinstance(value, list):
                if len(formatted_items) <= 3 and all(len(str(item)) < 20 for item in formatted_items):
                    return f"[{', '.join(formatted_items)}]"
                else:
                    indent = "    " * indent_level
                    items_str = f",\n{indent}".join(formatted_items)
                    base_indent = "    " * (indent_level - 1)
                    return f"[\n{indent}{items_str}\n{base_indent}]"
            else:  # tuple
                return f"({', '.join(formatted_items)})"
                
        elif isinstance(value, dict):
            return self._format_python_dict(value, indent_level)
        else:
            return repr(value)  # Fallback
    
    def _sanitize_python_literal(self, value: Any) -> Any:
        """
        Convert numpy/pandas scalar types to native Python equivalents for safe code generation.
        """
        try:
            import numpy as np  # Local import to avoid mandatory dependency at runtime
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                return float(value)
            if isinstance(value, (np.bool_,)):
                return bool(value)
        except Exception:
            pass
        
        if hasattr(value, 'item') and callable(getattr(value, 'item')):
            try:
                return value.item()
            except Exception:
                pass
        
        if isinstance(value, dict):
            return {self._sanitize_python_literal(k): self._sanitize_python_literal(v) for k, v in value.items()}
        
        if isinstance(value, list):
            return [self._sanitize_python_literal(v) for v in value]
        
        if isinstance(value, tuple):
            return tuple(self._sanitize_python_literal(v) for v in value)
        
        if isinstance(value, set):
            return [self._sanitize_python_literal(v) for v in sorted(value, key=lambda x: repr(x))]
        
        return value
    
    def _generate_complete_python_code(self, model, terminal_nodes: Dict[int, Dict]) -> str:
        """
        Generate complete Python file with node assignment logic and scoring helpers.
        """
        model_name = getattr(model, 'model_name', 'Exported_Decision_Tree')
        feature_names = list(getattr(model, 'feature_names', []))
        generation_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        node_metadata: Dict[int, Dict[str, Any]] = {}
        for node_no in sorted(terminal_nodes.keys()):
            node_data = terminal_nodes[node_no]
            metadata = dict(node_data.get('metadata', {}))
            metadata.setdefault('description', f"Node {node_no}")
            metadata['node_no'] = node_no
            metadata['node_id'] = node_data.get('node_id')
            metadata['node_logic'] = node_data.get('logic_text')
            node_metadata[node_no] = self._sanitize_python_literal(metadata)
        
        model_info = self._sanitize_python_literal({
            'name': model_name,
            'generated': generation_timestamp,
            'terminal_nodes': len(terminal_nodes),
            'features': len(feature_names),
            'purpose': 'retro_scoring',
            'extraction_method': 'direct_tree_model',
            'csv_dependency': False,
            'adds_columns': ['node_no', 'node_logic', 'node_id']
        })
        
        feature_config = self._sanitize_python_literal({
            'names': feature_names,
            'categorical': sorted(self.model_categorical_features),
            'numerical': sorted(self.model_numerical_features)
        })
        
        model_info_str = self._format_python_dict(model_info)
        feature_config_str = self._format_python_dict(feature_config)
        node_metadata_str = self._format_python_dict(node_metadata)
        
        sections: List[str] = []
        
        sections.append(f'''#!/usr/bin/env python3
"""
Decision Tree Retro Scoring Script
Generated by Bespoke Decision Tree Utility

Model: {model_name}
Generated: {generation_timestamp}
Terminal Nodes: {len(terminal_nodes)}

This file can be executed directly or imported as a module. It assigns every
record to a terminal node and appends three new columns:
    - node_no   : visualization node number consistent with the desktop app
    - node_logic: human-readable rule chain applied to reach the node
    - node_id   : internal model node identifier

The feature names in your scoring dataset must match the development dataset.
Column order may differ—the logic looks up fields by name.
"""
''')
        
        sections.append('''from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
''')
        
        sections.append('''# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
DEFAULT_NODE = 0                       # Returned when no conditions match
STRICT_FEATURE_VALIDATION = False      # Raise if inputs miss required features
PERFORMANCE_MODE = False               # True skips exceptions for speed
ENABLE_NODE_LOGGING = False            # Print node assignments during scoring

DEFAULT_OUTPUT_FORMAT = 'csv'          # csv | parquet | excel
OUTPUT_SUFFIX = '_scored'              # Suffix when output path not provided
SUPPORTED_OUTPUT_FORMATS = {'csv', 'parquet', 'excel'}
''')
        
        sections.append(f'''# ============================================================================
# MODEL METADATA AND NODE INFORMATION
# ============================================================================
MODEL_INFO = {model_info_str}

FEATURE_CONFIG = {feature_config_str}

NODE_METADATA = {node_metadata_str}
''')
        
        node_assignment_code = self._generate_node_assignment_function(terminal_nodes)
        sections.append(node_assignment_code)
        sections.append(self._generate_utility_functions())
        sections.append(self._generate_main_interface())
        sections.append(self._generate_examples(model_name, len(terminal_nodes)))
        
        return '\n\n'.join(sections)

    def _generate_node_assignment_function(self, terminal_nodes: Dict[int, Dict]) -> str:
        """
        Generate the core node assignment function with ALL terminal node conditions
        Works with any number of nodes (45, 463, etc.)
        """
        code_lines = [
            "# ============================================================================",
            "# TERMINAL NODE ASSIGNMENT LOGIC",
            f"# Handles {len(terminal_nodes)} terminal nodes with dynamic conditions",
            "# ============================================================================",
            "def _assign_to_node(sample):",
            "    \"\"\"",
            "    Assign record to correct terminal node using extracted tree conditions",
            f"    Handles {len(terminal_nodes)} terminal nodes with any variable combinations",
            "    \"\"\"",
            "    try:"
        ]
        
        if not terminal_nodes:
            code_lines.extend([
                "        # No terminal nodes found - return default",
                "        return DEFAULT_NODE  # No terminal nodes available",
            ])
        else:
            for i, (node_no, node_data) in enumerate(sorted(terminal_nodes.items())):
                conditions = node_data['python_condition']
                description = node_data['metadata']['description']
                
                if not conditions or conditions.strip() == '':
                    conditions = 'True  # Empty condition - always matches'
                
                if i == 0:
                    code_lines.append(f"        # {description}")
                    code_lines.append(f"        if {conditions}:")
                else:
                    code_lines.append(f"        # {description}")  
                    code_lines.append(f"        elif {conditions}:")
                
                code_lines.append(f"            return {node_no}")
                code_lines.append("")
            
            code_lines.extend([
                "        else:",
                "            return DEFAULT_NODE  # No conditions matched",
            "",
            "    except KeyError as e:",
            "        if STRICT_FEATURE_VALIDATION:",
            "            raise ValueError(f'Missing required feature: {e}')",
            "        else:",
            "            return DEFAULT_NODE",
            "    except Exception as e:",
            "        if PERFORMANCE_MODE:",
            "            return DEFAULT_NODE", 
            "        else:",
            "            raise RuntimeError(f'Node assignment error: {e}')"
        ])
        
        return '\n'.join(code_lines)
    
    def _generate_utility_functions(self) -> str:
        """Generate lookup and assignment helper functions"""
        return '''# ============================================================================
# LOOKUP & ASSIGNMENT HELPERS
# ============================================================================
def get_model_info() -> Dict[str, Any]:
    """Return metadata about the exported tree."""
    return MODEL_INFO


def get_feature_names() -> List[str]:
    """Return expected feature names in training order."""
    return FEATURE_CONFIG['names']


def get_all_node_numbers() -> List[int]:
    """Return all available terminal node numbers."""
    return sorted(NODE_METADATA.keys())


def get_feature_type(feature_name: str) -> str:
    """Return the recorded feature type ('categorical', 'numerical', or 'unknown')."""
    if feature_name in FEATURE_CONFIG['categorical']:
        return 'categorical'
    if feature_name in FEATURE_CONFIG['numerical']:
        return 'numerical'
    return 'unknown'


def get_node_details(node_numbers: Union[int, List[int]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Return metadata dictionaries for one or more terminal nodes."""
    if not isinstance(node_numbers, list):
        node_numbers = [node_numbers]
    
    details: List[Dict[str, Any]] = []
    for node_no in node_numbers:
        info = NODE_METADATA.get(node_no)
        if info is None:
            info = {
                'node_no': node_no,
                'description': f'Unknown node {node_no}',
                'samples': 0,
                'target_rate': 0.0,
                'target_count': 0
            }
        details.append(info)
    
    return details[0] if len(details) == 1 else details


def _validate_features(sample_data: Dict[str, Any]) -> None:
    """Raise if required features are missing when strict validation is enabled."""
    required_features = set(FEATURE_CONFIG['names'])
    missing = required_features - set(sample_data.keys())
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")


def assign_node(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Assign a single record to a terminal node and return metadata."""
    if STRICT_FEATURE_VALIDATION:
        _validate_features(sample)
    
    node_no = _assign_to_node(sample)
    metadata = NODE_METADATA.get(node_no, {})
    
    return {
        'node_no': node_no,
        'node_logic': metadata.get('node_logic', ''),
        'node_id': metadata.get('node_id', ''),
        'metadata': metadata
    }


def _prepare_records(X: Union[pd.DataFrame, Dict[str, Any], List[Any], tuple]) -> List[Dict[str, Any]]:
    """Normalise different input types into a list of dictionaries."""
    if isinstance(X, pd.DataFrame):
        return X.to_dict('records')
    
    if isinstance(X, dict):
        return [X]
    
    if isinstance(X, (list, tuple)):
        if not X:
            return []
        
        first = X[0]
        feature_names = get_feature_names()
        
        if isinstance(first, dict):
            return list(X)
        
        if isinstance(first, (list, tuple)):
            records = []
            for row in X:
                row_values = list(row)
                record = {
                    feature_names[i]: row_values[i] if i < len(row_values) else None
                    for i in range(len(feature_names))
                }
                records.append(record)
            return records
        
        # Treat as a single row of scalars
        return [{
            feature_names[i]: X[i] if i < len(feature_names) else None
            for i in range(len(feature_names))
        }]
    
    raise ValueError(f"Unsupported input type: {type(X)}")


def assign_nodes(X: Union[pd.DataFrame, Dict[str, Any], List[Any], tuple]) -> Union[int, List[int]]:
    """
    Assign node numbers for any supported input type.
    Returns int for a single record, otherwise a list.
    """
    records = _prepare_records(X)
    assignments: List[int] = []
    
    for sample in records:
        try:
            result = assign_node(sample)
            assignments.append(result['node_no'])
            if ENABLE_NODE_LOGGING:
                print(f"Assigned node {result['node_no']}: {result['node_logic']}")
        except Exception as exc:
            if PERFORMANCE_MODE:
                assignments.append(DEFAULT_NODE)
            else:
                raise RuntimeError(f"Node assignment failed: {exc}") from exc
    
    if len(assignments) == 1:
        return assignments[0]
    return assignments


def assign_nodes_with_metadata(X: Union[pd.DataFrame, Dict[str, Any], List[Any], tuple]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Assign nodes and return metadata (node_no, node_logic, node_id) for each record.
    Returns a single dictionary for one record, or a list otherwise.
    """
    records = _prepare_records(X)
    results: List[Dict[str, Any]] = []
    
    for sample in records:
        try:
            results.append(assign_node(sample))
        except Exception as exc:
            if PERFORMANCE_MODE:
                results.append({
                    'node_no': DEFAULT_NODE,
                    'node_logic': '',
                    'node_id': '',
                    'metadata': {'error': str(exc)}
                })
            else:
                raise RuntimeError(f"Node assignment failed: {exc}") from exc
    
    if len(results) == 1:
        return results[0]
    return results'''
    
    def _generate_main_interface(self) -> str:
        """Generate scoring helpers and file processing pipeline"""
        return '''# ============================================================================
# SCORING PIPELINE
# ============================================================================
def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the DataFrame with node_no, node_logic, and node_id columns appended.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("score_dataframe expects a pandas DataFrame")
    
    assignments = assign_nodes_with_metadata(df)
    if isinstance(assignments, dict):
        assignments = [assignments]
    
    scored = df.copy()
    node_numbers: List[int] = []
    node_logic: List[str] = []
    node_ids: List[str] = []
    
    for result in assignments:
        node_numbers.append(result.get('node_no', DEFAULT_NODE))
        node_logic.append(result.get('node_logic', ''))
        node_ids.append(result.get('node_id', ''))
    
    scored['node_no'] = node_numbers
    scored['node_logic'] = node_logic
    scored['node_id'] = node_ids
    return scored


def _detect_file_format(path: Path, fallback: str) -> str:
    """Infer file format from suffix or fall back to provided default."""
    suffix = path.suffix.lower()
    if suffix in {'.csv'}:
        return 'csv'
    if suffix in {'.parquet', '.pq'}:
        return 'parquet'
    if suffix in {'.xlsx', '.xls'}:
        return 'excel'
    return fallback


def _read_input_dataset(path: Path, file_format: str) -> pd.DataFrame:
    """Load dataset according to the requested format."""
    if file_format == 'csv':
        return pd.read_csv(path)
    if file_format == 'parquet':
        return pd.read_parquet(path)
    if file_format == 'excel':
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input format: {file_format}")


def _save_scored_dataframe(df: pd.DataFrame, path: Path, file_format: str) -> None:
    """Persist scored data in the requested format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if file_format == 'csv':
        df.to_csv(path, index=False)
    elif file_format == 'parquet':
        df.to_parquet(path, index=False)
    elif file_format == 'excel':
        df.to_excel(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {file_format}")


def _normalise_output_path(input_path: Path, suggested_path: Optional[str], output_format: str) -> Path:
    """Determine output path when the user does not supply one."""
    if suggested_path:
        path = Path(suggested_path).expanduser()
    else:
        suffix = {
            'csv': '.csv',
            'parquet': '.parquet',
            'excel': '.xlsx'
        }[output_format]
        path = input_path.with_name(f"{input_path.stem}{OUTPUT_SUFFIX}{suffix}")
    
    if not path.suffix:
        suffix = {
            'csv': '.csv',
            'parquet': '.parquet',
            'excel': '.xlsx'
        }[output_format]
        path = path.with_suffix(suffix)
    
    return path


def process_file(input_path: Union[str, Path],
                 output_path: Optional[Union[str, Path]] = None,
                 file_format: Optional[str] = None) -> Tuple[pd.DataFrame, Path]:
    """
    Score the dataset located at input_path and persist the result.
    
    Args:
        input_path: Path to the dataset to score.
        output_path: Optional destination path; auto-generated when omitted.
        file_format: Optional override for input format (csv|parquet|excel).
    
    Returns:
        Tuple of (scored_dataframe, output_path)
    """
    path = Path(input_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    
    requested_format = (file_format or '').lower()
    if requested_format and requested_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported file format override: {file_format}")
    
    input_format = requested_format or _detect_file_format(path, DEFAULT_OUTPUT_FORMAT)
    dataset = _read_input_dataset(path, input_format)
    scored = score_dataframe(dataset)
    
    if output_path is not None:
        supplied_path = Path(output_path).expanduser()
        output_format = requested_format or _detect_file_format(supplied_path, input_format)
        if output_format not in SUPPORTED_OUTPUT_FORMATS:
            output_format = DEFAULT_OUTPUT_FORMAT
        final_output_path = _normalise_output_path(path, str(supplied_path), output_format)
    else:
        output_format = input_format if input_format in SUPPORTED_OUTPUT_FORMATS else DEFAULT_OUTPUT_FORMAT
        final_output_path = _normalise_output_path(path, None, output_format)
    
    _save_scored_dataframe(scored, final_output_path, output_format)
    return scored, final_output_path


def _prompt_user_inputs() -> Dict[str, Optional[str]]:
    """Collect interactive inputs when the script is executed directly."""
    print("=" * 60)
    print("Decision Tree Retro Scoring")
    print("Provide the location of the dataset you would like to score.")
    print("=" * 60)
    
    input_path = input("Path to dataset (CSV/Parquet/Excel): ").strip()
    if not input_path:
        return {'input_path': None, 'output_path': None, 'file_format': None}
    
    output_path = input("Where should the scored dataset be saved? (leave blank for default): ").strip() or None
    file_format = input("Input format override [csv/parquet/excel or leave blank]: ").strip().lower() or None
    
    if file_format and file_format not in SUPPORTED_OUTPUT_FORMATS:
        print(f"Unrecognised format '{file_format}'. Falling back to auto-detection.")
        file_format = None
    
    return {
        'input_path': input_path,
        'output_path': output_path,
        'file_format': file_format
    }'''
    
    def _generate_examples(self, model_name: str, num_nodes: int) -> str:
        """Generate command-line entry point for the scored script"""
        safe_model_name = repr(model_name)
        return f'''# ============================================================================
# COMMAND LINE ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    user_inputs = _prompt_user_inputs()
    input_path = user_inputs.get('input_path')
    
    if not input_path:
        print("No dataset provided. Exiting.")
        sys.exit(0)
    
    try:
        scored_df, output_path = process_file(
            input_path,
            output_path=user_inputs.get('output_path'),
            file_format=user_inputs.get('file_format')
        )
    except Exception as exc:
        print(f"[ERROR] Scoring failed: {{exc}}")
        sys.exit(1)
    
    print("\\n" + "=" * 60)
    print("Scoring complete!")
    print(f"Model: {safe_model_name}")
    print(f"Terminal nodes in model: {num_nodes}")
    print(f"Input rows processed: {{len(scored_df)}}")
    print(f"Scored dataset saved to: {{output_path}}")
    print("Columns added: ['node_no', 'node_logic', 'node_id']")
    print("=" * 60)'''


class PythonExporter(GenericNodeExporter):
    """Wrapper to maintain compatibility with existing code"""
    
    def export_model(self, model, filepath: str) -> bool:
        """Export model - compatibility wrapper"""
        return self.export_model_to_python(model, filepath)
    
    def _generate_professional_python_code(self, model) -> str:
        """Compatibility wrapper for old interface"""
        terminal_nodes = self._extract_terminal_node_conditions(model)
        if not terminal_nodes:
            return "# Error: Could not extract terminal node conditions"
        return self._generate_complete_python_code(model, terminal_nodes)
