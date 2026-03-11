#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Serialization Utilities for Bespoke Utility

Provides safe JSON serialization functions that handle numpy data types,
pandas objects, and other complex Python objects that are not natively
JSON serializable.
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


def make_json_serializable(obj: Any) -> Any:
    """
    Convert objects to JSON-serializable formats.
    
    This function recursively processes objects to ensure they can be
    serialized to JSON, converting numpy types, pandas objects, and
    other complex types to their JSON-compatible equivalents.
    
    Args:
        obj: Object to make JSON serializable
        
    Returns:
        JSON-serializable version of the object
    """
    if obj is None:
        return None
    
    if isinstance(obj, (bool, int, float, str)):
        return obj
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    
    elif isinstance(obj, Path):
        return str(obj)
    
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    
    elif isinstance(obj, dict):
        return {make_json_serializable(k): make_json_serializable(v) 
                for k, v in obj.items()}
    
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    
    elif isinstance(obj, set):
        return list(obj)
    
    else:
        try:
            return str(obj)
        except Exception as e:
            logger.warning(f"Could not serialize object of type {type(obj)}: {e}")
            return f"<non-serializable: {type(obj).__name__}>"


def safe_pandas_dtypes_to_dict(dtypes: pd.Series) -> Dict[str, str]:
    """
    Safely convert pandas dtypes Series to a JSON-serializable dictionary.
    
    This function specifically handles the case where pandas dtypes.to_dict()
    might return numpy data types as keys, which are not JSON serializable.
    
    Args:
        dtypes: pandas Series containing data types
        
    Returns:
        Dictionary with string keys and string values representing dtypes
    """
    try:
        dtypes_dict = {}
        for column, dtype in dtypes.items():
            key = str(column)
            value = str(dtype)
            dtypes_dict[key] = value
        
        return dtypes_dict
        
    except Exception as e:
        logger.error(f"Error converting pandas dtypes to dict: {e}")
        return {}


def safe_json_dump(obj: Any, file_path: Union[str, Path], indent: int = 2) -> bool:
    """
    Safely dump an object to JSON file with proper error handling.
    
    Args:
        obj: Object to serialize
        file_path: Path to output file
        indent: JSON indentation
        
    Returns:
        True if successful, False otherwise
    """
    try:
        serializable_obj = make_json_serializable(obj)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_obj, f, indent=indent, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        logger.error(f"Error writing JSON to {file_path}: {e}")
        return False


def safe_json_dumps(obj: Any, indent: int = 2) -> str:
    """
    Safely serialize an object to JSON string with proper error handling.
    
    Args:
        obj: Object to serialize
        indent: JSON indentation
        
    Returns:
        JSON string or empty string if serialization fails
    """
    try:
        serializable_obj = make_json_serializable(obj)
        
        return json.dumps(serializable_obj, indent=indent, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error serializing object to JSON: {e}")
        return "{}"


def create_serializable_metadata(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """
    Create JSON-serializable metadata for a pandas DataFrame.
    
    Args:
        df: DataFrame to create metadata for
        name: Name of the dataset
        
    Returns:
        Dictionary containing serializable metadata
    """
    try:
        metadata = {
            'name': str(name),
            'shape': list(df.shape),  # Convert tuple to list
            'dtypes': safe_pandas_dtypes_to_dict(df.dtypes),
            'memory_usage': int(df.memory_usage(deep=True).sum()),  # Convert to int
            'columns': [str(col) for col in df.columns.tolist()],  # Ensure strings
            'index_name': str(df.index.name) if df.index.name is not None else None,
            'creation_date': datetime.now().isoformat()
        }
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error creating DataFrame metadata: {e}")
        return {
            'name': str(name),
            'error': f"Could not create metadata: {str(e)}",
            'creation_date': datetime.now().isoformat()
        }


class NumpyJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles numpy data types.
    
    This encoder can be used as a fallback when the make_json_serializable
    function is not sufficient.
    """
    
    def default(self, obj):
        """Handle objects that are not JSON serializable by default."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        
        elif isinstance(obj, Path):
            return str(obj)
        
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        
        elif isinstance(obj, set):
            return list(obj)
        
        else:
            try:
                return str(obj)
            except Exception:
                return f"<non-serializable: {type(obj).__name__}>"


def test_serialization():
    """Test the serialization utilities with various data types."""
    import numpy as np
    import pandas as pd
    
    test_data = {
        'numpy_int': np.int32(42),
        'numpy_float': np.float64(3.14159),
        'numpy_bool': np.bool_(True),
        'numpy_array': np.array([1, 2, 3, 4, 5]),
        'datetime': datetime.now(),
        'path': Path('/some/path'),
        'set': {1, 2, 3},
        'nested_dict': {
            'numpy_types': {
                np.uint8(255): 'max_uint8',
                'normal_key': np.float32(2.71828)
            }
        }
    }
    
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c'],
        'col3': [1.1, 2.2, 3.3]
    })
    
    metadata = create_serializable_metadata(df, 'test_dataframe')
    
    print("Test data serialization:")
    print(safe_json_dumps(test_data, indent=2))
    
    print("\nDataFrame metadata:")
    print(safe_json_dumps(metadata, indent=2))
    
    print("\nTest completed successfully!")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    test_serialization()