#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Saver Module for Bespoke Utility
Provides functionality for saving decision tree models in various formats
"""

import os
import json
import logging
import xml.dom.minidom
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import uuid
import pickle

from models.decision_tree import BespokeDecisionTree
from models.node import TreeNode

logger = logging.getLogger(__name__)

class ModelSaver:
    """Class for saving decision tree models in various formats"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the model saver
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        self.export_settings = self.config.get('export', {})
        
        self.default_format = self.export_settings.get('default_format', 'pmml')
        
        self.include_metadata = self.export_settings.get('include_metadata', True)
        
        self.add_comments = self.export_settings.get('add_comments', True)
        
        self.include_data_summary = self.export_settings.get('include_data_summary', True)
        
        logger.info(f"Model saver initialized with default format: {self.default_format}")
    
    def save_model(self, model: BespokeDecisionTree, filepath: str, format_type: Optional[str] = None) -> bool:
        """
        Save a model to a file in the specified format
        
        Args:
            model: Decision tree model to save
            filepath: Path to save the model to
            format_type: Format to save in (pmml, json, python, proprietary)
            
        Returns:
            True if the model was saved successfully, False otherwise
        """
        if format_type is None:
            format_type = self.default_format
        
        format_type = format_type.lower()
        
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            if format_type == 'pmml':
                return self.save_pmml(model, filepath)
            elif format_type == 'json':
                return self.save_json(model, filepath)
            elif format_type == 'proprietary':
                return self.save_proprietary(model, filepath)
            elif format_type == 'python':
                if hasattr(model, 'export_to_python'):
                    model.export_to_python(filepath)
                    return True
                else:
                    logger.error("Model does not support export to Python")
                    return False
            else:
                logger.error(f"Unsupported format: {format_type}")
                return False
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}", exc_info=True)
            return False
    
    def save_pmml(self, model: BespokeDecisionTree, filepath: str) -> bool:
        """
        Save a model in PMML format
        
        Args:
            model: Decision tree model to save
            filepath: Path to save the model to
            
        Returns:
            True if the model was saved successfully, False otherwise
        """
        try:
            if not model.is_fitted:
                logger.error("Cannot export to PMML: model is not fitted")
                return False
            
            pmml = ET.Element("PMML", {
                "version": "4.4",
                "xmlns": "http://www.dmg.org/PMML-4_4",
                "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance"
            })
            
            header = ET.SubElement(pmml, "Header")
            header.set("copyright", f"Copyright (c) {datetime.now().year} Bespoke Analytics")
            
            timestamp = ET.SubElement(header, "Timestamp")
            timestamp.text = datetime.now().isoformat()
            
            if self.include_metadata:
                application = ET.SubElement(header, "Application")
                application.set("name", "Bespoke Decision Tree Utility")
                application.set("version", self.config.get('application', {}).get('version', '1.0.0'))
            
            data_dict = ET.SubElement(pmml, "DataDictionary")
            
            for feature in model.feature_names:
                field = ET.SubElement(data_dict, "DataField")
                field.set("name", feature)
                
                if feature in model.categorical_features:
                    field.set("optype", "categorical")
                    field.set("dataType", "string")
                else:
                    field.set("optype", "continuous")
                    field.set("dataType", "double")
            
            if model.target_name:
                field = ET.SubElement(data_dict, "DataField")
                field.set("name", model.target_name)
                field.set("optype", "categorical")
                field.set("dataType", "string")
                
                for value in model.target_values:
                    value_elem = ET.SubElement(field, "Value")
                    value_elem.set("value", str(value))
            
            tree_model = ET.SubElement(pmml, "TreeModel")
            tree_model.set("modelName", model.model_name)
            tree_model.set("functionName", "classification")
            tree_model.set("algorithmName", "Decision Tree")
            
            mining_schema = ET.SubElement(tree_model, "MiningSchema")
            
            for feature in model.feature_names:
                mining_field = ET.SubElement(mining_schema, "MiningField")
                mining_field.set("name", feature)
                mining_field.set("usageType", "active")
            
            if model.target_name:
                mining_field = ET.SubElement(mining_schema, "MiningField")
                mining_field.set("name", model.target_name)
                mining_field.set("usageType", "target")
            
            if self.include_data_summary:
                model_stats = ET.SubElement(tree_model, "ModelStats")
                
                for feature in model.feature_names:
                    univ_stat = ET.SubElement(model_stats, "UnivariateStats")
                    univ_stat.set("field", feature)
                    
                    counts = ET.SubElement(univ_stat, "Counts")
                    counts.set("totalFreq", str(model.training_samples))
            
            output = ET.SubElement(tree_model, "Output")
            
            for value in model.target_values:
                output_field = ET.SubElement(output, "OutputField")
                output_field.set("name", f"Probability_{value}")
                output_field.set("feature", "probability")
                output_field.set("value", str(value))
                output_field.set("dataType", "double")
            
            self._add_pmml_nodes(tree_model, model.root)
            
            xml_string = ET.tostring(pmml, encoding="utf-8")
            dom = xml.dom.minidom.parseString(xml_string)
            pretty_xml = dom.toprettyxml(indent="  ")
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(pretty_xml)
            
            logger.info(f"Model saved in PMML format to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model in PMML format: {str(e)}", exc_info=True)
            return False
    
    def _add_pmml_nodes(self, parent_elem: ET.Element, node: TreeNode, node_id: int = 1) -> int:
        """
        Recursively add tree nodes to PMML document
        
        Args:
            parent_elem: Parent XML element
            node: Current tree node
            node_id: Current node ID (for PMML)
            
        Returns:
            Next available node ID
        """
        node_elem = ET.SubElement(parent_elem, "Node")
        node_elem.set("id", str(node_id))
        
        if node.majority_class is not None:
            node_elem.set("score", str(node.majority_class))
        
        node_elem.set("recordCount", str(node.samples))
        
        if node.class_counts:
            for cls, count in node.class_counts.items():
                score_dist = ET.SubElement(node_elem, "ScoreDistribution")
                score_dist.set("value", str(cls))
                score_dist.set("recordCount", str(count))
                
                probability = count / node.samples if node.samples > 0 else 0
                score_dist.set("probability", str(probability))
        
        if node.is_terminal:
            return node_id + 1
        
        if node.split_feature:
            if node.split_type == 'numeric':
                if len(node.children) >= 2:
                    left_child = node.children[0]
                    next_id = node_id + 1
                    
                    node_elem = ET.SubElement(parent_elem, "Node")
                    node_elem.set("id", str(next_id))
                    
                    predicate = ET.SubElement(node_elem, "SimplePredicate")
                    predicate.set("field", node.split_feature)
                    predicate.set("operator", "lessOrEqual")
                    predicate.set("value", str(node.split_value))
                    
                    if left_child.majority_class is not None:
                        node_elem.set("score", str(left_child.majority_class))
                    
                    node_elem.set("recordCount", str(left_child.samples))
                    
                    if left_child.class_counts:
                        for cls, count in left_child.class_counts.items():
                            score_dist = ET.SubElement(node_elem, "ScoreDistribution")
                            score_dist.set("value", str(cls))
                            score_dist.set("recordCount", str(count))
                            
                            probability = count / left_child.samples if left_child.samples > 0 else 0
                            score_dist.set("probability", str(probability))
                    
                    next_id = self._add_pmml_nodes(node_elem, left_child, next_id + 1)
                    
                    if len(node.children) >= 2:
                        right_child = node.children[1]
                        
                        node_elem = ET.SubElement(parent_elem, "Node")
                        node_elem.set("id", str(next_id))
                        
                        predicate = ET.SubElement(node_elem, "SimplePredicate")
                        predicate.set("field", node.split_feature)
                        predicate.set("operator", "greaterThan")
                        predicate.set("value", str(node.split_value))
                        
                        if right_child.majority_class is not None:
                            node_elem.set("score", str(right_child.majority_class))
                        
                        node_elem.set("recordCount", str(right_child.samples))
                        
                        if right_child.class_counts:
                            for cls, count in right_child.class_counts.items():
                                score_dist = ET.SubElement(node_elem, "ScoreDistribution")
                                score_dist.set("value", str(cls))
                                score_dist.set("recordCount", str(count))
                                
                                probability = count / right_child.samples if right_child.samples > 0 else 0
                                score_dist.set("probability", str(probability))
                        
                        next_id = self._add_pmml_nodes(node_elem, right_child, next_id + 1)
                        
                    return next_id
            
            elif node.split_type == 'categorical':
                next_id = node_id + 1
                
                categories_by_child = {}
                for cat, child_idx in node.split_categories.items():
                    categories_by_child.setdefault(child_idx, []).append(cat)
                
                for child_idx, categories in categories_by_child.items():
                    if child_idx < len(node.children):
                        child = node.children[child_idx]
                        
                        node_elem = ET.SubElement(parent_elem, "Node")
                        node_elem.set("id", str(next_id))
                        
                        if len(categories) == 1:
                            predicate = ET.SubElement(node_elem, "SimplePredicate")
                            predicate.set("field", node.split_feature)
                            predicate.set("operator", "equal")
                            predicate.set("value", str(categories[0]))
                        else:
                            predicate = ET.SubElement(node_elem, "SimpleSetPredicate")
                            predicate.set("field", node.split_feature)
                            predicate.set("operator", "isIn")
                            
                            array = ET.SubElement(predicate, "Array")
                            array.set("type", "string")
                            array.text = " ".join([str(cat) for cat in categories])
                        
                        if child.majority_class is not None:
                            node_elem.set("score", str(child.majority_class))
                        
                        node_elem.set("recordCount", str(child.samples))
                        
                        if child.class_counts:
                            for cls, count in child.class_counts.items():
                                score_dist = ET.SubElement(node_elem, "ScoreDistribution")
                                score_dist.set("value", str(cls))
                                score_dist.set("recordCount", str(count))
                                
                                probability = count / child.samples if child.samples > 0 else 0
                                score_dist.set("probability", str(probability))
                        
                        next_id = self._add_pmml_nodes(node_elem, child, next_id + 1)
                
                return next_id
        
        return node_id + 1
    
    def save_json(self, model: BespokeDecisionTree, filepath: str) -> bool:
        """
        Save a model in JSON format
        
        Args:
            model: Decision tree model to save
            filepath: Path to save the model to
            
        Returns:
            True if the model was saved successfully, False otherwise
        """
        try:
            model_dict = model.to_dict()
            
            if self.include_metadata:
                model_dict['metadata'] = {
                    'created_by': 'Bespoke Decision Tree Utility',
                    'version': self.config.get('application', {}).get('version', '1.0.0'),
                    'timestamp': datetime.now().isoformat(),
                    'format_version': '1.0'
                }
            
            if self.add_comments and isinstance(model_dict, dict):
                model_dict['__comment'] = "Generated by Bespoke Decision Tree Utility"
            
            model_dict['export_config'] = {
                'include_metadata': self.include_metadata,
                'include_data_summary': self.include_data_summary,
                'export_format': 'json'
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(model_dict, f, indent=2)
            
            logger.info(f"Model saved in JSON format to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model in JSON format: {str(e)}", exc_info=True)
            return False
    
    def save_proprietary(self, model: BespokeDecisionTree, filepath: str) -> bool:
        """
        Save a model in proprietary format (pickle)
        
        Args:
            model: Decision tree model to save
            filepath: Path to save the model to
            
        Returns:
            True if the model was saved successfully, False otherwise
        """
        try:
            
            data = {
                'model': model,
                'timestamp': datetime.now().isoformat(),
                'format_version': '1.0'
            }
            
            if self.include_metadata:
                data['metadata'] = {
                    'created_by': 'Bespoke Decision Tree Utility',
                    'version': self.config.get('application', {}).get('version', '1.0.0'),
                    'model_name': model.model_name,
                    'model_id': model.model_id
                }
            
            if self.include_data_summary and model.is_fitted:
                data['summary'] = {
                    'num_nodes': model.num_nodes,
                    'num_leaves': model.num_leaves,
                    'max_depth': model.max_depth,
                    'feature_importance': model.feature_importance,
                    'training_samples': model.training_samples
                }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Model saved in proprietary format to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model in proprietary format: {str(e)}", exc_info=True)
            return False
    
    def load_model(self, filepath: str, format_type: Optional[str] = None) -> Optional[BespokeDecisionTree]:
        """
        Load a model from a file
        
        Args:
            filepath: Path to load the model from
            format_type: Format of the model file
            
        Returns:
            Loaded model, or None if loading failed
        """
        if format_type is None:
            _, ext = os.path.splitext(filepath)
            ext = ext.lower()
            
            if ext == '.pmml':
                format_type = 'pmml'
            elif ext == '.json':
                format_type = 'json'
            elif ext in ['.pkl', '.pickle']:
                format_type = 'proprietary'
            else:
                logger.error(f"Cannot determine format from file extension: {ext}")
                return None
        
        format_type = format_type.lower()
        
        try:
            if format_type == 'json':
                return self.load_json(filepath)
            elif format_type == 'proprietary':
                return self.load_proprietary(filepath)
            elif format_type == 'pmml':
                logger.error("Loading from PMML format is not implemented")
                return None
            else:
                logger.error(f"Unsupported format: {format_type}")
                return None
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            return None
    
    def load_json(self, filepath: str) -> Optional[BespokeDecisionTree]:
        """
        Load a model from a JSON file
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model, or None if loading failed
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                model_dict = json.load(f)
            
            from models.decision_tree import BespokeDecisionTree
            model = BespokeDecisionTree.from_dict(model_dict)
            
            logger.info(f"Model loaded from JSON file: {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from JSON: {str(e)}", exc_info=True)
            return None
    
    def load_proprietary(self, filepath: str) -> Optional[BespokeDecisionTree]:
        """
        Load a model from a proprietary format file
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model, or None if loading failed
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict) and 'model' in data:
                model = data['model']
                logger.info(f"Model loaded from proprietary format file: {filepath}")
                return model
            elif isinstance(data, BespokeDecisionTree):
                logger.info(f"Model loaded from proprietary format file: {filepath}")
                return data
            else:
                logger.error("Invalid data format in proprietary file")
                return None
            
        except Exception as e:
            logger.error(f"Error loading model from proprietary format: {str(e)}", exc_info=True)
            return None
    
    def check_file_format(self, filepath: str) -> Optional[str]:
        """
        Check the format of a model file
        
        Args:
            filepath: Path to the model file
            
        Returns:
            Format type string, or None if unknown
        """
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        
        if ext == '.pmml':
            return 'pmml'
        elif ext == '.json':
            return 'json'
        elif ext in ['.pkl', '.pickle']:
            return 'proprietary'
        
        try:
            with open(filepath, 'rb') as f:
                header = f.read(10)
            
            if header.startswith(b'{'):
                return 'json'
            
            if header.startswith(b'<?xml') or header.startswith(b'<PMML'):
                return 'pmml'
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking file format: {str(e)}", exc_info=True)
            return None


class PMMLExporter:
    """Helper class for exporting decision trees to PMML format"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the PMML exporter
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    def export_model(self, model: BespokeDecisionTree, filepath: str) -> bool:
        """
        Export a model to PMML format
        
        Args:
            model: Decision tree model to export
            filepath: Path to save the PMML file
            
        Returns:
            True if export was successful, False otherwise
        """
        saver = ModelSaver(self.config)
        return saver.save_pmml(model, filepath)


class JSONExporter:
    """Helper class for exporting decision trees to JSON format"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the JSON exporter
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    def export_model(self, model: BespokeDecisionTree, filepath: str) -> bool:
        """
        Export a model to JSON format
        
        Args:
            model: Decision tree model to export
            filepath: Path to save the JSON file
            
        Returns:
            True if export was successful, False otherwise
        """
        saver = ModelSaver(self.config)
        return saver.save_json(model, filepath)


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    from models.decision_tree import BespokeDecisionTree
    from models.node import TreeNode
    
    model = BespokeDecisionTree()
    model.feature_names = ["Age", "Income", "CreditScore"]
    model.target_name = "Default"
    model.target_values = [0, 1]
    model.categorical_features = set()
    model.numerical_features = {"Age", "Income", "CreditScore"}
    model.training_samples = 1000
    model.is_fitted = True
    
    root = TreeNode(node_id="root", depth=0)
    root.update_stats(samples=1000, class_counts={0: 800, 1: 200}, impurity=0.32)
    root.set_split("Income", 50000, "numeric")
    
    left_child = TreeNode(node_id="root_L", parent=root, depth=1)
    left_child.update_stats(samples=600, class_counts={0: 550, 1: 50}, impurity=0.16)
    left_child.is_terminal = True
    
    right_child = TreeNode(node_id="root_R", parent=root, depth=1)
    right_child.update_stats(samples=400, class_counts={0: 250, 1: 150}, impurity=0.48)
    right_child.set_split("CreditScore", 700, "numeric")
    
    right_left = TreeNode(node_id="root_R_L", parent=right_child, depth=2)
    right_left.update_stats(samples=250, class_counts={0: 200, 1: 50}, impurity=0.32)
    right_left.is_terminal = True
    
    right_right = TreeNode(node_id="root_R_R", parent=right_child, depth=2)
    right_right.update_stats(samples=150, class_counts={0: 50, 1: 100}, impurity=0.28)
    right_right.is_terminal = True
    
    root.add_child(left_child)
    root.add_child(right_child)
    right_child.add_child(right_left)
    right_child.add_child(right_right)
    
    model.root = root
    model.num_nodes = 5
    model.num_leaves = 3
    model.max_depth = 2
    
    model.metrics = {
        'accuracy': 0.85,
        'precision': 0.75,
        'recall': 0.6,
        'f1_score': 0.67
    }
    
    saver = ModelSaver()
    
    saver.save_pmml(model, "test_model.pmml")
    saver.save_json(model, "test_model.json")
    saver.save_proprietary(model, "test_model.pkl")
    
    print("Model saved in multiple formats")