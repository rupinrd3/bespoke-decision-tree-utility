#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Correct Node Assignment Generator
Reads actual node_report.csv and generates accurate Python logic
"""

import pandas as pd
import re
from typing import Dict, List, Tuple


def parse_node_logic_from_csv(csv_path: str) -> Dict[int, str]:
    """
    Parse the actual node logic from node_report.csv
    Returns mapping of node_no -> logic condition
    """
    try:
        df = pd.read_csv(csv_path)
        node_logic_map = {}
        
        for _, row in df.iterrows():
            node_no = int(row['node_no'])
            logic = row['node_logic']
            node_logic_map[node_no] = logic
            
        return node_logic_map
        
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return {}


def convert_logic_to_python(logic_str: str) -> str:
    """
    Convert node logic string to Python code
    
    Example:
    "[EXT_SOURCE_2] > 0.57 AND [EXT_SOURCE_3] <= 0.596 AND [NAME_EDUCATION_TYPE] IN [Secondary]"
    becomes:
    "sample.get('EXT_SOURCE_2', 0) > 0.57 and sample.get('EXT_SOURCE_3', 0) <= 0.596 and sample.get('NAME_EDUCATION_TYPE', '') in ['Secondary / secondary special']"
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
        
        if "'" in values_str:
            values_list = [v.strip().strip("'\"") for v in values_str.split(',')]
        else:
            values_list = [values_str.strip()]
        
        formatted_values = repr(values_list)
        return f"sample.get('{feature}', '') in {formatted_values}"
    
    python_logic = re.sub(in_pattern, replace_in_condition, python_logic)
    
    return python_logic


def generate_correct_node_assignment_code(csv_path: str) -> str:
    """Generate correct Python node assignment code from node_report.csv"""
    
    node_logic_map = parse_node_logic_from_csv(csv_path)
    
    if not node_logic_map:
        return "# Error: Could not parse node_report.csv"
    
    code_lines = [
        "def _assign_to_node_correct(sample):",
        '    """',
        '    Correct node assignment based on actual node_report.csv logic',
        '    """',
        '    try:'
    ]
    
    for i, (node_no, logic_str) in enumerate(sorted(node_logic_map.items())):
        python_condition = convert_logic_to_python(logic_str)
        
        if i == 0:
            code_lines.append(f"        if {python_condition}:")
        else:
            code_lines.append(f"        elif {python_condition}:")
        
        code_lines.append(f"            return {node_no}  # Node {node_no}")
        code_lines.append("")
    
    code_lines.extend([
        "        else:",
        "            return DEFAULT_NODE  # No matching condition",
        "",
        "    except Exception as e:",
        "        if PERFORMANCE_MODE:",
        "            return DEFAULT_NODE",
        "        else:",
        "            raise RuntimeError(f'Node assignment error: {e}')"
    ])
    
    return '\n'.join(code_lines)


if __name__ == "__main__":
    csv_path = "D:/Decision_tree_utility/utility/node_report.csv"
    
    print("Parsing node_report.csv...")
    node_logic = parse_node_logic_from_csv(csv_path)
    
    print(f"Found {len(node_logic)} nodes")
    for node_no, logic in list(node_logic.items())[:3]:  # Show first 3
        print(f"Node {node_no}: {logic}")
        python_logic = convert_logic_to_python(logic)
        print(f"Python: {python_logic}")
        print()
    
    print("Generating correct Python code...")
    correct_code = generate_correct_node_assignment_code(csv_path)
    print(correct_code[:500] + "...")  # Show first 500 chars