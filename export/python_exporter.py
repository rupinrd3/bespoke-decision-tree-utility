#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generic Node Assignment Python Exporter
Works with ANY decision tree structure - 45 nodes, 463 nodes, any variables, any conditions
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
    Completely generic Python exporter that works with any decision tree structure
    Extracts actual terminal node conditions and generates appropriate Python code
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        logger.info("Generic Node Exporter initialized")
    
    def export_model_to_python(self, model, filepath: str, node_report_csv: str = None) -> bool:
        """
        Export any decision tree model to Python node assignment code
        
        Args:
            model: Decision tree model
            filepath: Output Python file path  
            node_report_csv: Path to node_report.csv (optional)
            
        Returns:
            True if successful
        """
        try:
            terminal_nodes = self._extract_terminal_node_conditions(model, node_report_csv)
            
            if not terminal_nodes:
                logger.error("No terminal nodes found - cannot generate code")
                return False
            
            logger.info(f"Found {len(terminal_nodes)} terminal nodes for code generation")
            
            python_code = self._generate_complete_python_code(model, terminal_nodes)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(python_code)
            
            logger.info(f"Successfully generated Python code with {len(terminal_nodes)} terminal nodes")
            return True
            
        except Exception as e:
            logger.error(f"Error in generic export: {str(e)}", exc_info=True)
            return False
    
    def _extract_terminal_node_conditions(self, model, node_report_csv: str = None) -> Dict[int, Dict]:
        """
        Extract terminal node conditions from tree structure or CSV
        Works with any tree complexity and any number of nodes
        
        Returns:
            {node_no: {'conditions': 'python_logic', 'metadata': {...}}}
        """
        terminal_nodes = {}
        
        if node_report_csv and os.path.exists(node_report_csv):
            terminal_nodes = self._extract_from_csv(node_report_csv)
            if terminal_nodes:
                logger.info(f"Extracted {len(terminal_nodes)} nodes from CSV")
                return terminal_nodes
        
        if hasattr(model, 'root') and model.root:
            terminal_nodes = self._extract_from_tree_structure(model.root)
            if terminal_nodes:
                logger.info(f"Extracted {len(terminal_nodes)} nodes from tree structure")
                return terminal_nodes
        
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', 'node_report.csv'),
            'node_report.csv',
            '../node_report.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                terminal_nodes = self._extract_from_csv(path)
                if terminal_nodes:
                    logger.info(f"Auto-found CSV at {path} with {len(terminal_nodes)} nodes")
                    return terminal_nodes
        
        logger.warning("Could not extract terminal node conditions from any source")
        return {}
    
    def _extract_from_csv(self, csv_path: str) -> Dict[int, Dict]:
        """Extract terminal node conditions from node_report.csv"""
        try:
            df = pd.read_csv(csv_path)
            terminal_nodes = {}
            
            for _, row in df.iterrows():
                node_no = int(row['node_no'])
                logic_str = row['node_logic']
                
                python_conditions = self._convert_logic_to_python(logic_str)
                
                metadata = {
                    'samples': int(row['node_size']),
                    'target_count': int(row['target_count']),
                    'target_rate': float(row['target_rate']),
                    'lift': float(row.get('lift', 0)),
                    'description': f"Node {node_no}: {logic_str} ({row['node_size']} samples, {row['target_rate']}% target rate)"
                }
                
                terminal_nodes[node_no] = {
                    'conditions': python_conditions,
                    'metadata': metadata
                }
            
            return terminal_nodes
            
        except Exception as e:
            logger.error(f"Error reading CSV {csv_path}: {e}")
            return {}
    
    def _convert_logic_to_python(self, logic_str: str) -> str:
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
    
    def _generate_complete_python_code(self, model, terminal_nodes: Dict[int, Dict]) -> str:
        """
        Generate complete Python file with node assignment logic for ANY number of terminal nodes
        """
        model_name = getattr(model, 'model_name', 'Exported_Decision_Tree')
        feature_names = getattr(model, 'feature_names', [])
        
        sections = []
        
        sections.append(f'''#!/usr/bin/env python3
"""
Decision Tree Node Assignment Code - GENERIC VERSION
Generated by Bespoke Decision Tree Utility

Model: {model_name}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Terminal Nodes: {len(terminal_nodes)}
Purpose: Assign records to decision tree terminal nodes

This file contains logic for {len(terminal_nodes)} terminal nodes.
Works with ANY decision tree structure and ANY variable combinations.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')''')
        
        sections.append('''# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
DEFAULT_NODE = 0                       # Default when no conditions match
STRICT_FEATURE_VALIDATION = False      # Require all features present
PERFORMANCE_MODE = False               # True = speed, False = safety
ENABLE_NODE_LOGGING = False            # Log each assignment''')
        
        node_info_dict = {node_no: data['metadata'] for node_no, data in terminal_nodes.items()}
        
        sections.append(f'''# ============================================================================
# MODEL METADATA AND NODE INFORMATION
# ============================================================================
MODEL_INFO = {{
    'name': '{model_name}',
    'generated': '{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}',
    'terminal_nodes': {len(terminal_nodes)},
    'features': {len(feature_names)},
    'purpose': 'node_assignment'
}}

FEATURE_CONFIG = {{
    'names': {feature_names}
}}

NODE_INFO = {node_info_dict}''')
        
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
        
        for i, (node_no, node_data) in enumerate(sorted(terminal_nodes.items())):
            conditions = node_data['conditions']
            description = node_data['metadata']['description']
            
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
        """Generate standard utility functions"""
        return '''# ============================================================================
# UTILITY FUNCTIONS  
# ============================================================================
def get_model_info() -> Dict[str, Any]:
    """Return model information"""
    return MODEL_INFO

def get_feature_names() -> List[str]:
    """Return expected feature names"""
    return FEATURE_CONFIG['names']

def get_all_node_numbers() -> List[int]:
    """Return all possible terminal node numbers"""
    return sorted(list(NODE_INFO.keys()))

def get_node_details(node_numbers) -> Union[Dict, List[Dict]]:
    """Get detailed information about nodes"""
    if not isinstance(node_numbers, list):
        node_numbers = [node_numbers]
    
    node_details = []
    for node_no in node_numbers:
        info = NODE_INFO.get(node_no, {
            'node_no': node_no,
            'samples': 0,
            'target_rate': 0.0,
            'description': f'Unknown node {node_no}'
        })
        node_details.append(info)
    
    return node_details[0] if len(node_details) == 1 else node_details'''
    
    def _generate_main_interface(self) -> str:
        """Generate main assignment interface"""
        return '''# ============================================================================
# MAIN NODE ASSIGNMENT INTERFACE
# ============================================================================
def assign_nodes(X) -> Union[int, List[int]]:
    """
    Assign input records to terminal decision tree nodes
    
    Args:
        X: Input data (DataFrame, dict, array, list)
        
    Returns:
        Terminal node assignments
    """
    try:
        # Handle different input types
        if isinstance(X, pd.DataFrame):
            prepared_data = X.to_dict('records')
        elif isinstance(X, dict):
            prepared_data = [X]
        elif isinstance(X, (list, tuple)):
            if len(X) > 0 and isinstance(X[0], dict):
                prepared_data = list(X)
            else:
                # Convert array to dict using feature names
                feature_names = FEATURE_CONFIG['names'][:len(X)]
                prepared_data = [{name: X[i] if i < len(X) else 0 
                                for i, name in enumerate(feature_names)}]
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")
        
        # Assign nodes
        node_assignments = []
        for sample in prepared_data:
            node_no = _assign_to_node(sample)
            node_assignments.append(node_no)
        
        if ENABLE_NODE_LOGGING:
            print(f"Assigned {len(node_assignments)} samples to nodes")
        
        return node_assignments[0] if len(node_assignments) == 1 else node_assignments
        
    except Exception as e:
        if PERFORMANCE_MODE:
            return DEFAULT_NODE
        else:
            raise RuntimeError(f"Node assignment failed: {str(e)}")'''
    
    def _generate_examples(self, model_name: str, num_nodes: int) -> str:
        """Generate usage examples"""
        return f'''# ============================================================================
# EXAMPLE USAGE - {num_nodes} TERMINAL NODES
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generic Decision Tree Node Assignment")
    print(f"Model: {model_name}")
    print(f"Terminal Nodes: {num_nodes}")
    print("=" * 60)
    
    # Example 1: Show model info
    print("\\n1. Model Information:")
    model_info = get_model_info()
    print(f"   Generated: {{model_info['generated']}}")
    print(f"   Terminal Nodes: {{model_info['terminal_nodes']}}")
    print(f"   Available Nodes: {{sorted(get_all_node_numbers())}}")
    
    # Example 2: Sample assignment (customize with your features)
    print("\\n2. Sample Node Assignment:")
    try:
        # Replace with your actual feature names and values
        sample_data = {{
            # Add your features here based on FEATURE_CONFIG['names']
        }}
        
        if sample_data:
            node_assignment = assign_nodes(sample_data) 
            node_details = get_node_details(node_assignment)
            print(f"   Assigned to Node: {{node_assignment}}")
            print(f"   Node Details: {{node_details.get('description', 'N/A')}}")
        else:
            print("   No sample data provided - customize with your features")
            
    except Exception as e:
        print(f"   Error: {{e}}")
    
    print("\\n" + "=" * 60)
    print(f"Generic node assignment system ready!")
    print(f"Works with {{num_nodes}} terminal nodes and any feature combination")
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