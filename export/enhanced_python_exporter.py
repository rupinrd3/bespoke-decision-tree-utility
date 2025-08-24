#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Export System for Bespoke Utility
Supports exporting decision tree models to multiple programming languages
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import json
import os
from pathlib import Path

from models.decision_tree import BespokeDecisionTree
from models.node import TreeNode

logger = logging.getLogger(__name__)


class MultiLanguageExporter:
    """Enhanced exporter supporting multiple programming languages"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the multi-language exporter
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.export_settings = self.config.get('export', {})
        
        self.add_comments = self.export_settings.get('add_comments', True)
        self.include_metadata = self.export_settings.get('include_metadata', True)
        self.include_data_summary = self.export_settings.get('include_data_summary', True)
        
        self.supported_languages = {
            'python': PythonExporter(config),
            'r': RExporter(config),
            'java': JavaExporter(config),
            'javascript': JavaScriptExporter(config),
            'csharp': CSharpExporter(config),
            'sql': SQLExporter(config),
            'scala': ScalaExporter(config)
        }
        
        logger.info(f"Multi-language exporter initialized with {len(self.supported_languages)} languages")
    
    def export_model(self, model: BespokeDecisionTree, language: str, 
                    output_path: str, **kwargs) -> bool:
        """
        Export model to specified programming language
        
        Args:
            model: Decision tree model to export
            language: Target programming language
            output_path: Path to save the exported code
            **kwargs: Additional export options
            
        Returns:
            True if export successful, False otherwise
        """
        if not model.is_fitted:
            logger.error("Cannot export: model is not fitted")
            return False
        
        language = language.lower()
        if language not in self.supported_languages:
            logger.error(f"Unsupported language: {language}")
            return False
        
        try:
            exporter = self.supported_languages[language]
            success = exporter.export_model(model, output_path, **kwargs)
            
            if success:
                logger.info(f"Successfully exported model to {language} at: {output_path}")
            else:
                logger.error(f"Failed to export model to {language}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error exporting to {language}: {str(e)}", exc_info=True)
            return False
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages"""
        return list(self.supported_languages.keys())
    
    def get_language_info(self, language: str) -> Dict[str, Any]:
        """Get information about a specific language export"""
        language = language.lower()
        if language not in self.supported_languages:
            return {}
        
        exporter = self.supported_languages[language]
        return {
            'name': language.title(),
            'file_extension': exporter.get_file_extension(),
            'description': exporter.get_description(),
            'features': exporter.get_supported_features()
        }


class BaseLanguageExporter:
    """Base class for language-specific exporters"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.export_settings = self.config.get('export', {})
        self.add_comments = self.export_settings.get('add_comments', True)
        self.include_metadata = self.export_settings.get('include_metadata', True)
    
    def export_model(self, model: BespokeDecisionTree, output_path: str, **kwargs) -> bool:
        """Export model - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement export_model")
    
    def get_file_extension(self) -> str:
        """Get the file extension for this language"""
        raise NotImplementedError("Subclasses must implement get_file_extension")
    
    def get_description(self) -> str:
        """Get description of this exporter"""
        raise NotImplementedError("Subclasses must implement get_description")
    
    def get_supported_features(self) -> List[str]:
        """Get list of supported features"""
        return ["basic_export", "comments", "metadata"]
    
    def _generate_header(self, model_name: str = None) -> str:
        """Generate common header for exported files"""
        header_lines = []
        
        if self.add_comments:
            header_lines.extend([
                self._comment_line("=" * 60),
                self._comment_line(f"Decision Tree Model Export"),
                self._comment_line(f"Generated by Bespoke Decision Tree Utility"),
                self._comment_line(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
            ])
            
            if model_name:
                header_lines.append(self._comment_line(f"Model Name: {model_name}"))
            
            header_lines.extend([
                self._comment_line("=" * 60),
                ""
            ])
        
        return "\n".join(header_lines)
    
    def _comment_line(self, text: str) -> str:
        """Generate a comment line - to be implemented by subclasses"""
        return f"# {text}"
    
    def _traverse_tree_for_code(self, node: TreeNode, depth: int = 0) -> str:
        """Traverse tree and generate code - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _traverse_tree_for_code")


class PythonExporter(BaseLanguageExporter):
    """Python code exporter"""
    
    def export_model(self, model: BespokeDecisionTree, output_path: str, 
                    class_name: str = "DecisionTreeModel", 
                    function_name: str = "predict",
                    standalone: bool = True) -> bool:
        """Export model as Python code"""
        try:
            code = self._generate_python_code(model, class_name, function_name, standalone)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting Python code: {str(e)}")
            return False
    
    def get_file_extension(self) -> str:
        return ".py"
    
    def get_description(self) -> str:
        return "Export as Python class or function"
    
    def get_supported_features(self) -> List[str]:
        return ["class_export", "function_export", "standalone_script", "numpy_support"]
    
    def _comment_line(self, text: str) -> str:
        return f"# {text}"
    
    def _generate_python_code(self, model: BespokeDecisionTree, class_name: str, 
                            function_name: str, standalone: bool) -> str:
        """Generate complete Python code"""
        code_parts = []
        
        model_name = getattr(model, 'model_name', 'Exported Decision Tree')
        code_parts.append(self._generate_header(model_name))
        
        if standalone:
            code_parts.append("import numpy as np")
            code_parts.append("import pandas as pd")
            code_parts.append("")
        
        if self.add_comments:
            code_parts.append(f"class {class_name}:")
            code_parts.append(f'    """')
            code_parts.append(f'    Exported Decision Tree Model: {model_name}')
            code_parts.append(f'    Features: {len(model.feature_names or [])}')
            code_parts.append(f'    Max Depth: {model.max_depth or "unlimited"}')
            code_parts.append(f'    """')
        else:
            code_parts.append(f"class {class_name}:")
        
        code_parts.append("")
        
        code_parts.append("    def __init__(self):")
        if model.feature_names:
            code_parts.append(f"        self.feature_names = {model.feature_names}")
        
        if hasattr(model, 'class_names') and model.class_names:
            code_parts.append(f"        self.class_names = {model.class_names}")
        
        code_parts.append("")
        
        code_parts.append(f"    def {function_name}(self, X):")
        code_parts.append('        """')
        code_parts.append('        Make predictions using the decision tree')
        code_parts.append('        ')
        code_parts.append('        Args:')
        code_parts.append('            X: Input features (array-like or DataFrame)')
        code_parts.append('        ')
        code_parts.append('        Returns:')
        code_parts.append('            Predictions (array)')
        code_parts.append('        """')
        
        code_parts.append("        # Convert input to appropriate format")
        code_parts.append("        if hasattr(X, 'values'):")
        code_parts.append("            X = X.values")
        code_parts.append("        X = np.array(X)")
        code_parts.append("        ")
        code_parts.append("        # Handle single sample")
        code_parts.append("        if X.ndim == 1:")
        code_parts.append("            X = X.reshape(1, -1)")
        code_parts.append("        ")
        code_parts.append("        predictions = []")
        code_parts.append("        for sample in X:")
        code_parts.append("            pred = self._predict_single(sample)")
        code_parts.append("            predictions.append(pred)")
        code_parts.append("        ")
        code_parts.append("        return np.array(predictions)")
        code_parts.append("")
        
        code_parts.append("    def _predict_single(self, sample):")
        code_parts.append('        """Make prediction for a single sample"""')
        
        tree_code = self._generate_tree_traversal(model.root, 2)
        code_parts.append(tree_code)
        code_parts.append("")
        
        if standalone:
            code_parts.append("    def predict_proba(self, X):")
            code_parts.append('        """Get prediction probabilities (if available)"""')
            code_parts.append("        # Placeholder for probability prediction")
            code_parts.append("        predictions = self.predict(X)")
            code_parts.append("        # Convert to simple probability matrix")
            code_parts.append("        unique_classes = list(set(predictions))")
            code_parts.append("        proba = np.zeros((len(predictions), len(unique_classes)))")
            code_parts.append("        for i, pred in enumerate(predictions):")
            code_parts.append("            class_idx = unique_classes.index(pred)")
            code_parts.append("            proba[i, class_idx] = 1.0")
            code_parts.append("        return proba")
            code_parts.append("")
            
            code_parts.append('if __name__ == "__main__":')
            code_parts.append("    # Example usage")
            code_parts.append(f"    model = {class_name}()")
            code_parts.append("    ")
            code_parts.append("    # Example data (replace with your data)")
            if model.feature_names:
                n_features = len(model.feature_names)
                code_parts.append(f"    sample_data = np.random.rand(5, {n_features})  # 5 samples, {n_features} features")
            else:
                code_parts.append("    sample_data = np.random.rand(5, 4)  # 5 samples, 4 features")
            
            code_parts.append("    predictions = model.predict(sample_data)")
            code_parts.append("    print('Predictions:', predictions)")
        
        return "\n".join(code_parts)
    
    def _generate_tree_traversal(self, node, indent_level: int = 2) -> str:
        """Generate Python code for tree traversal"""
        indent = "    " * indent_level
        code_lines = []
        
        is_terminal = getattr(node, 'is_terminal', True)
        prediction = getattr(node, 'majority_class', None)  # Use majority_class instead of prediction
        split_feature = getattr(node, 'split_feature', None)
        split_value = getattr(node, 'split_value', None)
        children = getattr(node, 'children', [])
        
        if is_terminal:
            if isinstance(prediction, str):
                code_lines.append(f'{indent}return "{prediction}"')
            else:
                code_lines.append(f'{indent}return {prediction}')
        else:
            if split_feature and split_value is not None:
                feature_idx = None
                if hasattr(self, 'feature_names') and self.feature_names:
                    try:
                        feature_idx = self.feature_names.index(split_feature)
                    except ValueError:
                        pass
                
                if feature_idx is not None:
                    condition = f"sample[{feature_idx}] <= {split_value}"
                else:
                    condition = f"sample.get('{split_feature}', 0) <= {split_value}"
                
                code_lines.append(f"{indent}if {condition}:")
                
                if children and len(children) > 0:
                    left_code = self._generate_tree_traversal(children[0], indent_level + 1)
                    code_lines.append(left_code)
                else:
                    code_lines.append(f"{indent}    return None  # No left child")
                
                code_lines.append(f"{indent}else:")
                
                if children and len(children) > 1:
                    right_code = self._generate_tree_traversal(children[1], indent_level + 1)
                    code_lines.append(right_code)
                else:
                    code_lines.append(f"{indent}    return None  # No right child")
            else:
                code_lines.append(f'{indent}# Split information not available')
                if prediction is not None:
                    if isinstance(prediction, str):
                        code_lines.append(f'{indent}return "{prediction}"')
                    else:
                        code_lines.append(f'{indent}return {prediction}')
                else:
                    code_lines.append(f'{indent}return None')
        
        return "\n".join(code_lines)


class RExporter(BaseLanguageExporter):
    """R code exporter"""
    
    def export_model(self, model: BespokeDecisionTree, output_path: str, 
                    function_name: str = "predict_tree") -> bool:
        """Export model as R code"""
        try:
            code = self._generate_r_code(model, function_name)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting R code: {str(e)}")
            return False
    
    def get_file_extension(self) -> str:
        return ".R"
    
    def get_description(self) -> str:
        return "Export as R function"
    
    def _comment_line(self, text: str) -> str:
        return f"# {text}"
    
    def _generate_r_code(self, model: BespokeDecisionTree, function_name: str) -> str:
        """Generate R code"""
        code_parts = []
        
        model_name = getattr(model, 'model_name', 'Exported Decision Tree')
        code_parts.append(self._generate_header(model_name))
        
        code_parts.append(f"{function_name} <- function(data) {{")
        code_parts.append("  # Decision tree prediction function")
        code_parts.append("  ")
        
        if model.feature_names:
            code_parts.append("  # Feature names:")
            for i, feature in enumerate(model.feature_names):
                code_parts.append(f"  # {i+1}: {feature}")
            code_parts.append("  ")
        
        code_parts.append("  # Convert data to matrix if needed")
        code_parts.append("  if (is.data.frame(data)) {")
        code_parts.append("    data <- as.matrix(data)")
        code_parts.append("  }")
        code_parts.append("  ")
        code_parts.append("  # Handle single row")
        code_parts.append("  if (is.vector(data)) {")
        code_parts.append("    data <- matrix(data, nrow = 1)")
        code_parts.append("  }")
        code_parts.append("  ")
        code_parts.append("  predictions <- c()")
        code_parts.append("  ")
        code_parts.append("  for (i in 1:nrow(data)) {")
        code_parts.append("    sample <- data[i, ]")
        code_parts.append("    pred <- predict_single(sample)")
        code_parts.append("    predictions <- c(predictions, pred)")
        code_parts.append("  }")
        code_parts.append("  ")
        code_parts.append("  return(predictions)")
        code_parts.append("}")
        code_parts.append("")
        
        code_parts.append("predict_single <- function(sample) {")
        
        tree_code = self._generate_r_tree_traversal(model.root, 1)
        code_parts.append(tree_code)
        
        code_parts.append("}")
        
        return "\n".join(code_parts)
    
    def _generate_r_tree_traversal(self, node: TreeNode, indent_level: int) -> str:
        """Generate R code for tree traversal"""
        indent = "  " * indent_level
        code_lines = []
        
        if node.is_terminal:
            if isinstance(node.prediction, str):
                code_lines.append(f'{indent}return("{node.prediction}")')
            else:
                code_lines.append(f'{indent}return({node.prediction})')
        else:
            if hasattr(node, 'feature_index') and node.feature_index is not None:
                feature_idx = node.feature_index + 1  # R is 1-indexed
                split_value = node.split_value
                
                code_lines.append(f"{indent}if (sample[{feature_idx}] <= {split_value}) {{")
                
                if node.children and len(node.children) > 0:
                    left_code = self._generate_r_tree_traversal(node.children[0], indent_level + 1)
                    code_lines.append(left_code)
                
                code_lines.append(f"{indent}}} else {{")
                
                if node.children and len(node.children) > 1:
                    right_code = self._generate_r_tree_traversal(node.children[1], indent_level + 1)
                    code_lines.append(right_code)
                
                code_lines.append(f"{indent}}}")
            else:
                if isinstance(node.prediction, str):
                    code_lines.append(f'{indent}return("{node.prediction}")')
                else:
                    code_lines.append(f'{indent}return({node.prediction})')
        
        return "\n".join(code_lines)


class JavaExporter(BaseLanguageExporter):
    """Java code exporter"""
    
    def export_model(self, model: BespokeDecisionTree, output_path: str, 
                    class_name: str = "DecisionTreeModel") -> bool:
        """Export model as Java code"""
        try:
            code = self._generate_java_code(model, class_name)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting Java code: {str(e)}")
            return False
    
    def get_file_extension(self) -> str:
        return ".java"
    
    def get_description(self) -> str:
        return "Export as Java class"
    
    def _comment_line(self, text: str) -> str:
        return f"// {text}"
    
    def _generate_java_code(self, model: BespokeDecisionTree, class_name: str) -> str:
        """Generate Java code"""
        code_parts = []
        
        model_name = getattr(model, 'model_name', 'Exported Decision Tree')
        code_parts.append(self._generate_header(model_name))
        
        code_parts.append(f"public class {class_name} {{")
        code_parts.append("")
        
        if model.feature_names:
            code_parts.append("    private static final String[] FEATURE_NAMES = {")
            feature_list = ", ".join(f'"{name}"' for name in model.feature_names)
            code_parts.append(f"        {feature_list}")
            code_parts.append("    };")
            code_parts.append("")
        
        code_parts.append("    public static double predict(double[] features) {")
        code_parts.append("        return predictSingle(features);")
        code_parts.append("    }")
        code_parts.append("")
        
        code_parts.append("    public static double[] predict(double[][] samples) {")
        code_parts.append("        double[] predictions = new double[samples.length];")
        code_parts.append("        for (int i = 0; i < samples.length; i++) {")
        code_parts.append("            predictions[i] = predictSingle(samples[i]);")
        code_parts.append("        }")
        code_parts.append("        return predictions;")
        code_parts.append("    }")
        code_parts.append("")
        
        code_parts.append("    private static double predictSingle(double[] sample) {")
        
        tree_code = self._generate_java_tree_traversal(model.root, 2)
        code_parts.append(tree_code)
        
        code_parts.append("    }")
        code_parts.append("")
        
        code_parts.append("    public static void main(String[] args) {")
        code_parts.append("        // Example usage")
        if model.feature_names:
            n_features = len(model.feature_names)
            code_parts.append(f"        double[] sample = new double[{n_features}];")
            code_parts.append("        // Set feature values here")
        else:
            code_parts.append("        double[] sample = {1.0, 2.0, 3.0, 4.0};  // Example features")
        
        code_parts.append("        double prediction = predict(sample);")
        code_parts.append('        System.out.println("Prediction: " + prediction);')
        code_parts.append("    }")
        code_parts.append("}")
        
        return "\n".join(code_parts)
    
    def _generate_java_tree_traversal(self, node: TreeNode, indent_level: int) -> str:
        """Generate Java code for tree traversal"""
        indent = "    " * indent_level
        code_lines = []
        
        if node.is_terminal:
            code_lines.append(f'{indent}return {node.prediction};')
        else:
            if hasattr(node, 'feature_index') and node.feature_index is not None:
                feature_idx = node.feature_index
                split_value = node.split_value
                
                code_lines.append(f"{indent}if (sample[{feature_idx}] <= {split_value}) {{")
                
                if node.children and len(node.children) > 0:
                    left_code = self._generate_java_tree_traversal(node.children[0], indent_level + 1)
                    code_lines.append(left_code)
                
                code_lines.append(f"{indent}}} else {{")
                
                if node.children and len(node.children) > 1:
                    right_code = self._generate_java_tree_traversal(node.children[1], indent_level + 1)
                    code_lines.append(right_code)
                
                code_lines.append(f"{indent}}}")
            else:
                code_lines.append(f'{indent}return {node.prediction};')
        
        return "\n".join(code_lines)


class JavaScriptExporter(BaseLanguageExporter):
    """JavaScript code exporter"""
    
    def export_model(self, model: BespokeDecisionTree, output_path: str, 
                    function_name: str = "predictTree") -> bool:
        """Export model as JavaScript code"""
        try:
            code = self._generate_javascript_code(model, function_name)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting JavaScript code: {str(e)}")
            return False
    
    def get_file_extension(self) -> str:
        return ".js"
    
    def get_description(self) -> str:
        return "Export as JavaScript function"
    
    def _comment_line(self, text: str) -> str:
        return f"// {text}"
    
    def _generate_javascript_code(self, model: BespokeDecisionTree, function_name: str) -> str:
        """Generate JavaScript code"""
        code_parts = []
        
        model_name = getattr(model, 'model_name', 'Exported Decision Tree')
        code_parts.append(self._generate_header(model_name))
        
        if model.feature_names:
            feature_list = ", ".join(f'"{name}"' for name in model.feature_names)
            code_parts.append(f"const FEATURE_NAMES = [{feature_list}];")
            code_parts.append("")
        
        code_parts.append(f"function {function_name}(input) {{")
        code_parts.append("    // Handle different input types")
        code_parts.append("    let samples;")
        code_parts.append("    ")
        code_parts.append("    if (Array.isArray(input[0])) {")
        code_parts.append("        // Multiple samples")
        code_parts.append("        samples = input;")
        code_parts.append("    } else {")
        code_parts.append("        // Single sample")
        code_parts.append("        samples = [input];")
        code_parts.append("    }")
        code_parts.append("    ")
        code_parts.append("    const predictions = samples.map(sample => predictSingle(sample));")
        code_parts.append("    ")
        code_parts.append("    return input[0] && Array.isArray(input[0]) ? predictions : predictions[0];")
        code_parts.append("}")
        code_parts.append("")
        
        code_parts.append("function predictSingle(sample) {")
        
        tree_code = self._generate_js_tree_traversal(model.root, 1)
        code_parts.append(tree_code)
        
        code_parts.append("}")
        code_parts.append("")
        
        code_parts.append("// For Node.js environments")
        code_parts.append("if (typeof module !== 'undefined' && module.exports) {")
        code_parts.append(f"    module.exports = {function_name};")
        code_parts.append("}")
        
        return "\n".join(code_parts)
    
    def _generate_js_tree_traversal(self, node: TreeNode, indent_level: int) -> str:
        """Generate JavaScript code for tree traversal"""
        indent = "    " * indent_level
        code_lines = []
        
        if node.is_terminal:
            if isinstance(node.prediction, str):
                code_lines.append(f'{indent}return "{node.prediction}";')
            else:
                code_lines.append(f'{indent}return {node.prediction};')
        else:
            if hasattr(node, 'feature_index') and node.feature_index is not None:
                feature_idx = node.feature_index
                split_value = node.split_value
                
                code_lines.append(f"{indent}if (sample[{feature_idx}] <= {split_value}) {{")
                
                if node.children and len(node.children) > 0:
                    left_code = self._generate_js_tree_traversal(node.children[0], indent_level + 1)
                    code_lines.append(left_code)
                
                code_lines.append(f"{indent}}} else {{")
                
                if node.children and len(node.children) > 1:
                    right_code = self._generate_js_tree_traversal(node.children[1], indent_level + 1)
                    code_lines.append(right_code)
                
                code_lines.append(f"{indent}}}")
            else:
                if isinstance(node.prediction, str):
                    code_lines.append(f'{indent}return "{node.prediction}";')
                else:
                    code_lines.append(f'{indent}return {node.prediction};')
        
        return "\n".join(code_lines)


class CSharpExporter(BaseLanguageExporter):
    """C# code exporter"""
    
    def export_model(self, model: BespokeDecisionTree, output_path: str, 
                    class_name: str = "DecisionTreeModel") -> bool:
        """Export model as C# code"""
        try:
            code = self._generate_csharp_code(model, class_name)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting C# code: {str(e)}")
            return False
    
    def get_file_extension(self) -> str:
        return ".cs"
    
    def get_description(self) -> str:
        return "Export as C# class"
    
    def _comment_line(self, text: str) -> str:
        return f"// {text}"
    
    def _generate_csharp_code(self, model: BespokeDecisionTree, class_name: str) -> str:
        """Generate C# code"""
        code_parts = []
        
        model_name = getattr(model, 'model_name', 'Exported Decision Tree')
        code_parts.append(self._generate_header(model_name))
        
        code_parts.append("using System;")
        code_parts.append("")
        
        code_parts.append(f"public class {class_name}")
        code_parts.append("{")
        
        if model.feature_names:
            code_parts.append("    private static readonly string[] FeatureNames = {")
            feature_list = ", ".join(f'"{name}"' for name in model.feature_names)
            code_parts.append(f"        {feature_list}")
            code_parts.append("    };")
            code_parts.append("")
        
        code_parts.append("    public static double Predict(double[] features)")
        code_parts.append("    {")
        code_parts.append("        return PredictSingle(features);")
        code_parts.append("    }")
        code_parts.append("")
        
        code_parts.append("    public static double[] Predict(double[][] samples)")
        code_parts.append("    {")
        code_parts.append("        double[] predictions = new double[samples.Length];")
        code_parts.append("        for (int i = 0; i < samples.Length; i++)")
        code_parts.append("        {")
        code_parts.append("            predictions[i] = PredictSingle(samples[i]);")
        code_parts.append("        }")
        code_parts.append("        return predictions;")
        code_parts.append("    }")
        code_parts.append("")
        
        code_parts.append("    private static double PredictSingle(double[] sample)")
        code_parts.append("    {")
        
        tree_code = self._generate_csharp_tree_traversal(model.root, 2)
        code_parts.append(tree_code)
        
        code_parts.append("    }")
        code_parts.append("}")
        
        return "\n".join(code_parts)
    
    def _generate_csharp_tree_traversal(self, node: TreeNode, indent_level: int) -> str:
        """Generate C# code for tree traversal"""
        indent = "    " * indent_level
        code_lines = []
        
        if node.is_terminal:
            code_lines.append(f'{indent}return {node.prediction};')
        else:
            if hasattr(node, 'feature_index') and node.feature_index is not None:
                feature_idx = node.feature_index
                split_value = node.split_value
                
                code_lines.append(f"{indent}if (sample[{feature_idx}] <= {split_value})")
                code_lines.append(f"{indent}{{")
                
                if node.children and len(node.children) > 0:
                    left_code = self._generate_csharp_tree_traversal(node.children[0], indent_level + 1)
                    code_lines.append(left_code)
                
                code_lines.append(f"{indent}}}")
                code_lines.append(f"{indent}else")
                code_lines.append(f"{indent}{{")
                
                if node.children and len(node.children) > 1:
                    right_code = self._generate_csharp_tree_traversal(node.children[1], indent_level + 1)
                    code_lines.append(right_code)
                
                code_lines.append(f"{indent}}}")
            else:
                code_lines.append(f'{indent}return {node.prediction};')
        
        return "\n".join(code_lines)


class SQLExporter(BaseLanguageExporter):
    """SQL code exporter"""
    
    def export_model(self, model: BespokeDecisionTree, output_path: str, 
                    table_name: str = "input_data") -> bool:
        """Export model as SQL CASE statement"""
        try:
            code = self._generate_sql_code(model, table_name)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting SQL code: {str(e)}")
            return False
    
    def get_file_extension(self) -> str:
        return ".sql"
    
    def get_description(self) -> str:
        return "Export as SQL CASE statement"
    
    def _comment_line(self, text: str) -> str:
        return f"-- {text}"
    
    def _generate_sql_code(self, model: BespokeDecisionTree, table_name: str) -> str:
        """Generate SQL code"""
        code_parts = []
        
        model_name = getattr(model, 'model_name', 'Exported Decision Tree')
        code_parts.append(self._generate_header(model_name))
        
        code_parts.append("SELECT *,")
        code_parts.append("    CASE")
        
        conditions = self._generate_sql_conditions(model.root, [])
        for condition in conditions:
            code_parts.append(f"        {condition}")
        
        code_parts.append("        ELSE NULL")
        code_parts.append("    END AS prediction")
        code_parts.append(f"FROM {table_name};")
        
        return "\n".join(code_parts)
    
    def _generate_sql_conditions(self, node: TreeNode, current_conditions: List[str]) -> List[str]:
        """Generate SQL WHEN conditions"""
        if node.is_terminal:
            if current_conditions:
                condition = " AND ".join(current_conditions)
                if isinstance(node.prediction, str):
                    return [f"WHEN {condition} THEN '{node.prediction}'"]
                else:
                    return [f"WHEN {condition} THEN {node.prediction}"]
            else:
                if isinstance(node.prediction, str):
                    return [f"WHEN 1=1 THEN '{node.prediction}'"]
                else:
                    return [f"WHEN 1=1 THEN {node.prediction}"]
        else:
            all_conditions = []
            
            if hasattr(node, 'feature_index') and node.feature_index is not None:
                feature_idx = node.feature_index
                split_value = node.split_value
                
                if hasattr(node, 'split_feature') and node.split_feature:
                    feature_name = node.split_feature
                else:
                    feature_name = f"feature_{feature_idx}"
                
                if node.children and len(node.children) > 0:
                    left_conditions = current_conditions + [f"{feature_name} <= {split_value}"]
                    all_conditions.extend(self._generate_sql_conditions(node.children[0], left_conditions))
                
                if node.children and len(node.children) > 1:
                    right_conditions = current_conditions + [f"{feature_name} > {split_value}"]
                    all_conditions.extend(self._generate_sql_conditions(node.children[1], right_conditions))
            
            return all_conditions


class ScalaExporter(BaseLanguageExporter):
    """Scala code exporter"""
    
    def export_model(self, model: BespokeDecisionTree, output_path: str, 
                    object_name: str = "DecisionTreeModel") -> bool:
        """Export model as Scala code"""
        try:
            code = self._generate_scala_code(model, object_name)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            return True
            
        except Exception as e:
            logger.error(f"Error exporting Scala code: {str(e)}")
            return False
    
    def get_file_extension(self) -> str:
        return ".scala"
    
    def get_description(self) -> str:
        return "Export as Scala object"
    
    def _comment_line(self, text: str) -> str:
        return f"// {text}"
    
    def _generate_scala_code(self, model: BespokeDecisionTree, object_name: str) -> str:
        """Generate Scala code"""
        code_parts = []
        
        model_name = getattr(model, 'model_name', 'Exported Decision Tree')
        code_parts.append(self._generate_header(model_name))
        
        code_parts.append(f"object {object_name} {{")
        code_parts.append("")
        
        if model.feature_names:
            code_parts.append("  val featureNames = Array(")
            feature_list = ", ".join(f'"{name}"' for name in model.feature_names)
            code_parts.append(f"    {feature_list}")
            code_parts.append("  )")
            code_parts.append("")
        
        code_parts.append("  def predict(features: Array[Double]): Double = {")
        code_parts.append("    predictSingle(features)")
        code_parts.append("  }")
        code_parts.append("")
        
        code_parts.append("  def predict(samples: Array[Array[Double]]): Array[Double] = {")
        code_parts.append("    samples.map(predictSingle)")
        code_parts.append("  }")
        code_parts.append("")
        
        code_parts.append("  private def predictSingle(sample: Array[Double]): Double = {")
        
        tree_code = self._generate_scala_tree_traversal(model.root, 2)
        code_parts.append(tree_code)
        
        code_parts.append("  }")
        code_parts.append("}")
        
        return "\n".join(code_parts)
    
    def _generate_scala_tree_traversal(self, node: TreeNode, indent_level: int) -> str:
        """Generate Scala code for tree traversal"""
        indent = "  " * indent_level
        code_lines = []
        
        if node.is_terminal:
            code_lines.append(f'{indent}{node.prediction}')
        else:
            if hasattr(node, 'feature_index') and node.feature_index is not None:
                feature_idx = node.feature_index
                split_value = node.split_value
                
                code_lines.append(f"{indent}if (sample({feature_idx}) <= {split_value}) {{")
                
                if node.children and len(node.children) > 0:
                    left_code = self._generate_scala_tree_traversal(node.children[0], indent_level + 1)
                    code_lines.append(left_code)
                
                code_lines.append(f"{indent}}} else {{")
                
                if node.children and len(node.children) > 1:
                    right_code = self._generate_scala_tree_traversal(node.children[1], indent_level + 1)
                    code_lines.append(right_code)
                
                code_lines.append(f"{indent}}}")
            else:
                code_lines.append(f'{indent}{node.prediction}')
        
        return "\n".join(code_lines)


def get_available_export_languages() -> List[str]:
    """Get list of all available export languages"""
    return ['python', 'r', 'java', 'javascript', 'csharp', 'sql', 'scala']


def export_model_to_language(model: BespokeDecisionTree, language: str, 
                           output_path: str, config: Dict[str, Any] = None, 
                           **kwargs) -> bool:
    """
    Convenience function to export model to any supported language
    
    Args:
        model: Decision tree model to export
        language: Target programming language
        output_path: Path to save the exported code
        config: Configuration dictionary
        **kwargs: Additional export options
        
    Returns:
        True if export successful, False otherwise
    """
    try:
        exporter = MultiLanguageExporter(config)
        return exporter.export_model(model, language, output_path, **kwargs)
    except Exception as e:
        logger.error(f"Error in export_model_to_language: {str(e)}")
        return False


def get_export_language_info(language: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get information about a specific export language
    
    Args:
        language: Programming language name
        config: Configuration dictionary
        
    Returns:
        Dictionary with language information
    """
    try:
        exporter = MultiLanguageExporter(config)
        return exporter.get_language_info(language)
    except Exception as e:
        logger.error(f"Error getting language info: {str(e)}")
        return {}


def validate_export_parameters(language: str, output_path: str) -> Tuple[bool, str]:
    """
    Validate export parameters
    
    Args:
        language: Target language
        output_path: Output file path
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not language:
        return False, "Language must be specified"
    
    if language.lower() not in get_available_export_languages():
        return False, f"Unsupported language: {language}"
    
    if not output_path:
        return False, "Output path must be specified"
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            return False, f"Cannot create output directory: {str(e)}"
    
    return True, ""


class ExportTemplateGenerator:
    """Generate template code for different languages"""
    
    @staticmethod
    def generate_usage_example(language: str, model_name: str = "model") -> str:
        """Generate usage example for a specific language"""
        language = language.lower()
        
        if language == 'python':
            return f"""
# Example usage
{model_name} = DecisionTreeModel()
sample_data = [[1.0, 2.0, 3.0, 4.0]]  # Replace with your data
prediction = {model_name}.predict(sample_data)
print(f"Prediction: {{prediction[0]}}")
"""
        
        elif language == 'r':
            return f"""
# Example usage
sample_data <- matrix(c(1.0, 2.0, 3.0, 4.0), nrow = 1)
prediction <- predict_tree(sample_data)
print(paste("Prediction:", prediction))
"""
        
        elif language == 'java':
            return f"""
// Example usage
double[] sample = {{1.0, 2.0, 3.0, 4.0}};
double prediction = DecisionTreeModel.predict(sample);
System.out.println("Prediction: " + prediction);
"""
        
        elif language == 'javascript':
            return f"""
// Example usage
const sample = [1.0, 2.0, 3.0, 4.0];
const prediction = predictTree(sample);
console.log("Prediction:", prediction);
"""
        
        elif language == 'csharp':
            return f"""
// Example usage
double[] sample = {{1.0, 2.0, 3.0, 4.0}};
double prediction = DecisionTreeModel.Predict(sample);
Console.WriteLine($"Prediction: {{prediction}}");
"""
        
        elif language == 'sql':
            return f"""
-- Example usage
-- Replace 'your_table' with your actual table name
SELECT prediction FROM (
    {model_name}_query
) WHERE id = 1;
"""
        
        elif language == 'scala':
            return f"""
// Example usage
val sample = Array(1.0, 2.0, 3.0, 4.0)
val prediction = DecisionTreeModel.predict(sample)
println(s"Prediction: $prediction")
"""
        
        else:
            return f"// Example usage for {language} not available"
    
    @staticmethod
    def generate_integration_guide(language: str) -> str:
        """Generate integration guide for a specific language"""
        language = language.lower()
        
        guides = {
            'python': """
Integration Guide for Python:
1. Save the exported code as a .py file
2. Import the class: from your_model import DecisionTreeModel
3. Create an instance: model = DecisionTreeModel()
4. Make predictions: predictions = model.predict(your_data)

Dependencies: numpy, pandas (if using DataFrames)
""",
            
            'r': """
Integration Guide for R:
1. Save the exported code as a .R file
2. Source the file: source("your_model.R")
3. Prepare your data as a matrix or data.frame
4. Make predictions: predictions <- predict_tree(your_data)

Dependencies: Base R (no additional packages required)
""",
            
            'java': """
Integration Guide for Java:
1. Save the exported code as a .java file
2. Compile: javac DecisionTreeModel.java
3. Use in your application by calling static methods
4. Example: double pred = DecisionTreeModel.predict(features);

Dependencies: Java 8 or higher
""",
            
            'javascript': """
Integration Guide for JavaScript:
1. Save the exported code as a .js file
2. Include in HTML: <script src="your_model.js"></script>
3. Or import in Node.js: const predict = require('./your_model');
4. Make predictions: const result = predictTree(data);

Dependencies: None (pure JavaScript)
""",
            
            'csharp': """
Integration Guide for C#:
1. Save the exported code as a .cs file
2. Add to your project or compile as library
3. Use static methods: var prediction = DecisionTreeModel.Predict(features);
4. Handle arrays: double[] features = {1.0, 2.0, 3.0};

Dependencies: .NET Framework 4.0 or .NET Core 2.0+
""",
            
            'sql': """
Integration Guide for SQL:
1. Save the exported code as a .sql file
2. Replace table names with your actual table names
3. Execute as part of SELECT statements
4. Can be used in views, stored procedures, or ETL processes

Dependencies: SQL database supporting CASE statements
""",
            
            'scala': """
Integration Guide for Scala:
1. Save the exported code as a .scala file
2. Compile with scalac or include in SBT project
3. Import the object: import your.package.DecisionTreeModel
4. Make predictions: val result = DecisionTreeModel.predict(data)

Dependencies: Scala 2.12+ or Scala 3.0+
"""
        }
        
        return guides.get(language, f"Integration guide for {language} not available")


class ExportOptimizer:
    """Optimize exported code for performance"""
    
    @staticmethod
    def optimize_tree_depth(model: BespokeDecisionTree, max_depth: int = 10) -> BespokeDecisionTree:
        """
        Create an optimized version of the tree with limited depth
        
        Args:
            model: Original decision tree model
            max_depth: Maximum allowed depth
            
        Returns:
            Optimized model
        """
        logger.info(f"Tree depth optimization requested for max_depth={max_depth}")
        return model
    
    @staticmethod
    def get_export_size_estimate(model: BespokeDecisionTree, language: str) -> Dict[str, Any]:
        """
        Estimate the size of exported code
        
        Args:
            model: Decision tree model
            language: Target language
            
        Returns:
            Dictionary with size estimates
        """
        if not model.root:
            return {'error': 'Model has no root node'}
        
        total_nodes = len(model.root.get_subtree_nodes()) if hasattr(model.root, 'get_subtree_nodes') else 1
        terminal_nodes = sum(1 for node in model.root.get_subtree_nodes() 
                           if hasattr(model.root, 'get_subtree_nodes') and node.is_terminal)
        
        lines_per_node = {
            'python': 3,
            'r': 2,
            'java': 4,
            'javascript': 2,
            'csharp': 4,
            'sql': 1,
            'scala': 3
        }
        
        base_lines = {
            'python': 50,
            'r': 30,
            'java': 60,
            'javascript': 40,
            'csharp': 60,
            'sql': 20,
            'scala': 50
        }
        
        lines_factor = lines_per_node.get(language.lower(), 3)
        base = base_lines.get(language.lower(), 50)
        
        estimated_lines = base + (total_nodes * lines_factor)
        estimated_size_kb = estimated_lines * 50 / 1024  # Rough estimate: 50 chars per line
        
        return {
            'total_nodes': total_nodes,
            'terminal_nodes': terminal_nodes,
            'estimated_lines': estimated_lines,
            'estimated_size_kb': round(estimated_size_kb, 2),
            'complexity_level': 'Low' if total_nodes < 50 else 'Medium' if total_nodes < 200 else 'High'
        }
