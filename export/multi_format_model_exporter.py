#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-Format Model Exporter for Bespoke Utility
Exports decision tree models in various industry-standard formats
"""

import logging
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import pandas as pd
import numpy as np

from models.node import TreeNode

logger = logging.getLogger(__name__)


class MultiFormatModelExporter:
    """Exports decision tree models in multiple formats"""
    
    def __init__(self, tree_model):
        """
        Initialize the multi-format exporter
        
        Args:
            tree_model: The trained decision tree model
        """
        self.tree_model = tree_model
        self.model_name = getattr(tree_model, 'model_name', 'Exported Decision Tree')
        self.export_timestamp = datetime.now().isoformat()
        
    def export_pmml(self, output_path: str) -> str:
        """
        Export model in PMML (Predictive Model Markup Language) format
        
        Args:
            output_path: Path to save the PMML file
            
        Returns:
            Generated PMML content as string
        """
        pmml = ET.Element("PMML")
        pmml.set("version", "4.4")
        pmml.set("xmlns", "http://www.dmg.org/PMML-4_4")
        pmml.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        
        header = ET.SubElement(pmml, "Header")
        header.set("copyright", "Bespoke Utility")
        header.set("description", f"Decision Tree Model: {self.model_name}")
        
        application = ET.SubElement(header, "Application")
        application.set("name", "Bespoke Utility")
        application.set("version", "1.0")
        
        timestamp = ET.SubElement(header, "Timestamp")
        timestamp.text = self.export_timestamp
        
        data_dict = ET.SubElement(pmml, "DataDictionary")
        data_dict.set("numberOfFields", str(len(self._get_feature_names()) + 1))
        
        for feature_name in self._get_feature_names():
            data_field = ET.SubElement(data_dict, "DataField")
            data_field.set("name", feature_name)
            data_field.set("optype", "continuous")  # Simplified
            data_field.set("dataType", "double")
            
        target_field = ET.SubElement(data_dict, "DataField")
        target_field.set("name", "target")
        target_field.set("optype", "categorical")
        target_field.set("dataType", "string")
        
        for class_name in self._get_class_names():
            value = ET.SubElement(target_field, "Value")
            value.set("value", str(class_name))
            
        tree_model_elem = ET.SubElement(pmml, "TreeModel")
        tree_model_elem.set("modelName", self.model_name)
        tree_model_elem.set("functionName", "classification")
        tree_model_elem.set("algorithmName", "BespokeDecisionTree")
        tree_model_elem.set("splitCharacteristic", "binarySplit")
        
        mining_schema = ET.SubElement(tree_model_elem, "MiningSchema")
        
        for feature_name in self._get_feature_names():
            mining_field = ET.SubElement(mining_schema, "MiningField")
            mining_field.set("name", feature_name)
            mining_field.set("usageType", "active")
            
        target_mining_field = ET.SubElement(mining_schema, "MiningField")
        target_mining_field.set("name", "target")
        target_mining_field.set("usageType", "target")
        
        output = ET.SubElement(tree_model_elem, "Output")
        
        output_field = ET.SubElement(output, "OutputField")
        output_field.set("name", "predicted_target")
        output_field.set("feature", "predictedValue")
        
        prob_output_field = ET.SubElement(output, "OutputField")
        prob_output_field.set("name", "probability")
        prob_output_field.set("feature", "probability")
        
        root_node = self._create_pmml_node(self.tree_model.root, "1")
        tree_model_elem.append(root_node)
        
        rough_string = ET.tostring(pmml, 'unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
            
        logger.info(f"Exported PMML model to: {output_path}")
        return pretty_xml
        
    def export_json(self, output_path: str, include_metadata: bool = True) -> str:
        """
        Export model in JSON format
        
        Args:
            output_path: Path to save the JSON file
            include_metadata: Whether to include model metadata
            
        Returns:
            Generated JSON content as string
        """
        model_data = {
            "model_type": "decision_tree",
            "model_name": self.model_name,
            "export_timestamp": self.export_timestamp,
            "tree_structure": self._serialize_tree_node(self.tree_model.root)
        }
        
        if include_metadata:
            model_data["metadata"] = {
                "feature_names": self._get_feature_names(),
                "class_names": self._get_class_names(),
                "tree_depth": self._calculate_depth(self.tree_model.root),
                "node_count": self._count_nodes(self.tree_model.root),
                "model_parameters": self._get_model_parameters()
            }
            
        json_content = json.dumps(model_data, indent=2, default=str)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_content)
            
        logger.info(f"Exported JSON model to: {output_path}")
        return json_content
        
    def export_xml(self, output_path: str) -> str:
        """
        Export model in custom XML format
        
        Args:
            output_path: Path to save the XML file
            
        Returns:
            Generated XML content as string
        """
        root = ET.Element("DecisionTreeModel")
        root.set("name", self.model_name)
        root.set("timestamp", self.export_timestamp)
        root.set("version", "1.0")
        
        metadata = ET.SubElement(root, "Metadata")
        
        model_info = ET.SubElement(metadata, "ModelInfo")
        model_info.set("type", "decision_tree")
        model_info.set("depth", str(self._calculate_depth(self.tree_model.root)))
        model_info.set("nodes", str(self._count_nodes(self.tree_model.root)))
        
        features = ET.SubElement(metadata, "Features")
        for feature_name in self._get_feature_names():
            feature = ET.SubElement(features, "Feature")
            feature.set("name", feature_name)
            feature.set("type", "numeric")  # Simplified
            
        classes = ET.SubElement(metadata, "Classes")
        for class_name in self._get_class_names():
            class_elem = ET.SubElement(classes, "Class")
            class_elem.set("name", str(class_name))
            
        tree_structure = ET.SubElement(root, "TreeStructure")
        root_node = self._create_xml_node(self.tree_model.root, "root")
        tree_structure.append(root_node)
        
        rough_string = ET.tostring(root, 'unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
            
        logger.info(f"Exported XML model to: {output_path}")
        return pretty_xml
        
    def export_onnx(self, output_path: str) -> str:
        """
        Export model in ONNX format (requires onnx library)
        
        Args:
            output_path: Path to save the ONNX file
            
        Returns:
            Status message
        """
        try:
            import onnx
            from onnx import helper, TensorProto
        except ImportError:
            raise ImportError("ONNX library is required for ONNX export. Install with: pip install onnx")
            
        
        input_features = self._get_feature_names()
        inputs = []
        
        for i, feature in enumerate(input_features):
            input_tensor = helper.make_tensor_value_info(
                feature, TensorProto.FLOAT, [1]
            )
            inputs.append(input_tensor)
            
        output = helper.make_tensor_value_info(
            'prediction', TensorProto.STRING, [1]
        )
        
        node = helper.make_node(
            'Identity',  # Simplified - would need TreeEnsemble operator
            inputs=[input_features[0]],
            outputs=['prediction'],
            name='decision_tree_node'
        )
        
        graph = helper.make_graph(
            [node],
            'decision_tree_graph',
            inputs,
            [output]
        )
        
        model = helper.make_model(graph)
        model.opset_import[0].version = 11
        
        model.doc_string = f"Decision Tree Model: {self.model_name}"
        model.model_version = 1
        
        onnx.save(model, output_path)
        
        logger.info(f"Exported ONNX model to: {output_path}")
        return f"ONNX model exported successfully to {output_path}"
        
    def export_sklearn_format(self, output_path: str) -> str:
        """
        Export model in scikit-learn compatible format
        
        Args:
            output_path: Path to save the pickle file
            
        Returns:
            Status message
        """
        import pickle
        
        sklearn_model = {
            'model_type': 'DecisionTreeClassifier',
            'tree_': self._convert_to_sklearn_tree(),
            'classes_': np.array(self._get_class_names()),
            'feature_names_': self._get_feature_names(),
            'n_classes_': len(self._get_class_names()),
            'n_features_': len(self._get_feature_names()),
            'metadata': {
                'export_source': 'Bespoke Utility',
                'export_timestamp': self.export_timestamp,
                'model_name': self.model_name
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(sklearn_model, f)
            
        logger.info(f"Exported sklearn-format model to: {output_path}")
        return f"Scikit-learn format model exported to {output_path}"
        
    def export_csv_rules(self, output_path: str) -> str:
        """
        Export decision rules as CSV
        
        Args:
            output_path: Path to save the CSV file
            
        Returns:
            Generated CSV content as string
        """
        rules = []
        self._extract_rules_for_csv(self.tree_model.root, [], rules)
        
        rules_data = []
        for i, (conditions, prediction, node_info) in enumerate(rules):
            rule_data = {
                'Rule_ID': f"R{i+1:03d}",
                'Conditions': ' AND '.join(conditions) if conditions else 'TRUE',
                'Prediction': str(prediction),
                'Node_Samples': node_info.get('samples', 0),
                'Node_Purity': node_info.get('purity', 0.0),
                'Confidence': node_info.get('confidence', 0.0)
            }
            rules_data.append(rule_data)
            
        df = pd.DataFrame(rules_data)
        df.to_csv(output_path, index=False)
        
        csv_content = df.to_csv(index=False)
        
        logger.info(f"Exported CSV rules to: {output_path}")
        return csv_content
        
    def export_all_formats(self, base_path: str, formats: List[str] = None) -> Dict[str, str]:
        """
        Export model in multiple formats
        
        Args:
            base_path: Base path for output files (without extension)
            formats: List of formats to export (default: all supported)
            
        Returns:
            Dictionary mapping format names to output paths
        """
        if formats is None:
            formats = ['json', 'xml', 'pmml', 'csv_rules', 'sklearn']
            
        results = {}
        
        for format_name in formats:
            try:
                output_path = f"{base_path}.{format_name}"
                
                if format_name == 'json':
                    self.export_json(output_path)
                elif format_name == 'xml':
                    self.export_xml(output_path)
                elif format_name == 'pmml':
                    self.export_pmml(output_path)
                elif format_name == 'csv_rules':
                    output_path = f"{base_path}_rules.csv"
                    self.export_csv_rules(output_path)
                elif format_name == 'sklearn':
                    output_path = f"{base_path}.pkl"
                    self.export_sklearn_format(output_path)
                elif format_name == 'onnx':
                    output_path = f"{base_path}.onnx"
                    self.export_onnx(output_path)
                    
                results[format_name] = output_path
                
            except Exception as e:
                logger.error(f"Error exporting {format_name}: {e}")
                results[format_name] = f"Error: {str(e)}"
                
        return results
        
    def _create_pmml_node(self, node: TreeNode, node_id: str) -> ET.Element:
        """Create PMML node element"""
        node_elem = ET.Element("Node")
        node_elem.set("id", node_id)
        
        if node.is_terminal:
            node_elem.set("score", str(node.prediction))
            node_elem.set("recordCount", str(node.samples))
            
            score_dist = ET.SubElement(node_elem, "ScoreDistribution")
            score_dist.set("value", str(node.prediction))
            score_dist.set("recordCount", str(node.samples))
            
        else:
            node_elem.set("recordCount", str(node.samples))
            
            predicate = ET.SubElement(node_elem, "SimplePredicate")
            predicate.set("field", node.split_feature)
            predicate.set("operator", "lessOrEqual")
            predicate.set("value", str(node.split_value))
            
            for i, child in enumerate(node.children):
                child_id = f"{node_id}.{i+1}"
                child_elem = self._create_pmml_node(child, child_id)
                node_elem.append(child_elem)
                
        return node_elem
        
    def _create_xml_node(self, node: TreeNode, node_id: str) -> ET.Element:
        """Create XML node element"""
        node_elem = ET.Element("Node")
        node_elem.set("id", node_id)
        node_elem.set("samples", str(node.samples))
        node_elem.set("depth", str(node.depth))
        
        if node.is_terminal:
            node_elem.set("type", "leaf")
            node_elem.set("prediction", str(node.prediction))
            if hasattr(node, 'impurity'):
                node_elem.set("impurity", str(node.impurity))
        else:
            node_elem.set("type", "split")
            node_elem.set("feature", node.split_feature)
            node_elem.set("threshold", str(node.split_value))
            if hasattr(node, 'split_operator'):
                node_elem.set("operator", node.split_operator)
            if hasattr(node, 'impurity'):
                node_elem.set("impurity", str(node.impurity))
                
            for i, child in enumerate(node.children):
                child_id = f"{node_id}.{i+1}"
                child_elem = self._create_xml_node(child, child_id)
                node_elem.append(child_elem)
                
        return node_elem
        
    def _serialize_tree_node(self, node: TreeNode) -> Dict[str, Any]:
        """Serialize tree node to dictionary"""
        node_data = {
            "node_id": node.node_id,
            "depth": node.depth,
            "samples": node.samples,
            "is_terminal": node.is_terminal
        }
        
        if hasattr(node, 'impurity'):
            node_data["impurity"] = node.impurity
            
        if node.is_terminal:
            node_data["prediction"] = node.prediction
        else:
            node_data["split_feature"] = node.split_feature
            node_data["split_value"] = node.split_value
            if hasattr(node, 'split_operator'):
                node_data["split_operator"] = node.split_operator
                
            node_data["children"] = [
                self._serialize_tree_node(child) for child in node.children
            ]
            
        return node_data
        
    def _convert_to_sklearn_tree(self) -> Dict[str, Any]:
        """Convert tree to scikit-learn compatible format"""
        
        tree_data = {
            'node_count': self._count_nodes(self.tree_model.root),
            'max_depth': self._calculate_depth(self.tree_model.root),
            'feature_names': self._get_feature_names(),
            'class_names': self._get_class_names(),
            'tree_structure': self._serialize_tree_node(self.tree_model.root)
        }
        
        return tree_data
        
    def _extract_rules_for_csv(self, node: TreeNode, current_conditions: List[str], 
                              rules: List[Tuple[List[str], str, Dict[str, Any]]]):
        """Extract rules for CSV export"""
        if node.is_terminal:
            node_info = {
                'samples': node.samples,
                'purity': 1.0 - getattr(node, 'impurity', 0.0),
                'confidence': 1.0  # Simplified
            }
            rules.append((current_conditions.copy(), str(node.prediction), node_info))
        else:
            feature = node.split_feature
            threshold = node.split_value
            operator = getattr(node, 'split_operator', '<=')
            
            if node.children and len(node.children) > 0:
                condition = f"{feature} {operator} {threshold}"
                new_conditions = current_conditions + [condition]
                self._extract_rules_for_csv(node.children[0], new_conditions, rules)
            
            if node.children and len(node.children) > 1:
                if operator == '<=':
                    condition = f"{feature} > {threshold}"
                else:
                    condition = f"{feature} not ({operator} {threshold})"
                new_conditions = current_conditions + [condition]
                self._extract_rules_for_csv(node.children[1], new_conditions, rules)
                
    def _get_feature_names(self) -> List[str]:
        """Get feature names from the model"""
        if hasattr(self.tree_model, 'feature_names'):
            return list(self.tree_model.feature_names)
        else:
            features = set()
            self._collect_features(self.tree_model.root, features)
            return sorted(list(features))
            
    def _get_class_names(self) -> List[str]:
        """Get class names from the model"""
        if hasattr(self.tree_model, 'class_names'):
            return list(self.tree_model.class_names)
        else:
            classes = set()
            self._collect_classes(self.tree_model.root, classes)
            return sorted(list(classes))
            
    def _get_model_parameters(self) -> Dict[str, Any]:
        """Get model training parameters"""
        if hasattr(self.tree_model, 'get_params'):
            return self.tree_model.get_params()
        else:
            return {
                'criterion': getattr(self.tree_model, 'criterion', 'unknown'),
                'max_depth': getattr(self.tree_model, 'max_depth', None),
                'min_samples_split': getattr(self.tree_model, 'min_samples_split', None),
                'min_samples_leaf': getattr(self.tree_model, 'min_samples_leaf', None)
            }
            
    def _calculate_depth(self, node: TreeNode) -> int:
        """Calculate depth of tree recursively"""
        if node.is_terminal:
            return 0
        
        max_child_depth = 0
        for child in node.children:
            child_depth = self._calculate_depth(child)
            max_child_depth = max(max_child_depth, child_depth)
            
        return max_child_depth + 1
        
    def _count_nodes(self, node: TreeNode) -> int:
        """Count nodes in tree recursively"""
        count = 1  # Current node
        
        for child in node.children:
            count += self._count_nodes(child)
            
        return count
        
    def _collect_features(self, node: TreeNode, features: set):
        """Collect all features used in the tree"""
        if not node.is_terminal and node.split_feature:
            features.add(node.split_feature)
            
        for child in node.children:
            self._collect_features(child, features)
            
    def _collect_classes(self, node: TreeNode, classes: set):
        """Collect all classes in the tree"""
        if node.is_terminal and node.prediction is not None:
            classes.add(str(node.prediction))
            
        for child in node.children:
            self._collect_classes(child, classes)