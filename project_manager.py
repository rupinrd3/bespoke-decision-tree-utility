# utility/project_manager.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Project Manager Module for Bespoke Utility
Handles comprehensive saving and loading of the entire project state, including workflow,
dataset references, model states, configurations, analysis results, and export settings.
"""

from __future__ import annotations 

import logging
import json
import os
import pickle
import zipfile
import tempfile
import shutil
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import hashlib

from ui.workflow_canvas import (WorkflowCanvas, WorkflowNode, ConnectionData, 
                               DatasetNode, ModelNode) # For type hinting and config
from models.decision_tree import BespokeDecisionTree
from data.data_loader import DataLoader # For reloading datasets based on stored info
from utils.serialization_utils import (make_json_serializable, safe_json_dump, 
                                      create_serializable_metadata, 
                                      safe_pandas_dtypes_to_dict)

logger = logging.getLogger(__name__)

PROJECT_FILE_EXTENSION = ".bspk"
PROJECT_FILE_VERSION = "2.0" # Enhanced version with comprehensive state preservation

class EnhancedProjectManager:
    """
    Enhanced project manager for Bespoke Utility projects.
    
    A comprehensive project includes:
    - Workflow canvas state (nodes, connections, positions, configurations)
    - Complete dataset information (data, metadata, transformations)
    - Model configurations and trained states (including tree structure)
    - Analysis results and performance metrics
    - Variable importance calculations and selections
    - Export settings and format preferences
    - UI state and view configurations
    - Project metadata and version history
    """
    
    PROJECT_FILE_EXTENSION = ".bspk"
    PROJECT_FILE_VERSION = "2.0"

    def __init__(self, config: Dict[str, Any], main_window_ref=None):
        self.app_config = config
        self.current_project_path: Optional[Path] = None
        self.is_modified: bool = False
        self.main_window_ref = main_window_ref
        
        self.data_loader = DataLoader(self.app_config)
        
        self.project_metadata = {
            'created_date': None,
            'last_modified': None,
            'version_history': [],
            'description': '',
            'tags': [],
            'author': '',
            'application_version': self.app_config.get("application", {}).get("version", "1.0.0")
        }
        
        self._cached_datasets = {}
        self._cached_models = {}
        self._cached_analysis_results = {}
        
        logger.info("Enhanced ProjectManager initialized.")

    def new_project(self, description: str = "", author: str = "", tags: List[str] = None):
        """Create a new project with enhanced metadata."""
        self.current_project_path = None
        self.is_modified = False
        
        self.project_metadata = {
            'created_date': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat(),
            'version_history': [{'version': 1, 'date': datetime.now().isoformat(), 'description': 'Project created'}],
            'description': description,
            'tags': tags or [],
            'author': author,
            'application_version': self.app_config.get("application", {}).get("version", "1.0.0")
        }
        
        self._cached_datasets.clear()
        self._cached_models.clear()
        self._cached_analysis_results.clear()
        
        logger.info("New enhanced project created.")

    def save_project_comprehensive(self, file_path: str, 
                                 workflow_canvas: WorkflowCanvas = None,
                                 datasets: Dict[str, pd.DataFrame] = None,
                                 models: Dict[str, BespokeDecisionTree] = None,
                                 analysis_results: Dict[str, Any] = None,
                                 variable_importance: Dict[str, Any] = None,
                                 performance_metrics: Dict[str, Any] = None,
                                 export_settings: Dict[str, Any] = None,
                                 ui_state: Dict[str, Any] = None,
                                 include_data: bool = True) -> bool:
        """
        Save comprehensive project state to a .bspk file (ZIP format).
        
        Args:
            file_path: Path to save the project
            workflow_canvas: Workflow canvas state
            datasets: Dictionary of datasets
            models: Dictionary of trained models
            analysis_results: Analysis results and cached computations
            variable_importance: Variable importance calculations
            performance_metrics: Model performance metrics
            export_settings: Export format preferences and settings
            ui_state: UI state (window positions, selections, etc.)
            include_data: Whether to include actual data or just references
            
        Returns:
            True if successful, False otherwise
        """
        if not file_path.lower().endswith(self.PROJECT_FILE_EXTENSION):
            file_path += self.PROJECT_FILE_EXTENSION
        
        path_obj = Path(file_path)
        logger.info(f"Saving comprehensive project to: {path_obj}")
        
        self.project_metadata['last_modified'] = datetime.now().isoformat()
        version_num = len(self.project_metadata['version_history']) + 1
        self.project_metadata['version_history'].append({
            'version': version_num,
            'date': datetime.now().isoformat(),
            'description': f'Project saved (version {version_num})'
        })
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                main_config = {
                    "project_file_version": self.PROJECT_FILE_VERSION,
                    "project_metadata": self.project_metadata,
                    "workflow": self._serialize_workflow(workflow_canvas),
                    "datasets_meta": self._serialize_datasets_metadata(datasets),
                    "models_meta": self._serialize_models_metadata(models),
                    "analysis_results": analysis_results or {},
                    "variable_importance": variable_importance or {},
                    "performance_metrics": performance_metrics or {},
                    "export_settings": export_settings or {},
                    "ui_state": ui_state or {},
                    "include_data": include_data
                }
                
                if not safe_json_dump(main_config, temp_path / "project.json"):
                    logger.error("Failed to save main project configuration")
                    return False
                
                if datasets and include_data:
                    datasets_dir = temp_path / "datasets"
                    datasets_dir.mkdir()
                    
                    for name, df in datasets.items():
                        if isinstance(df, pd.DataFrame):
                            df.to_parquet(datasets_dir / f"{name}.parquet")
                            
                            metadata = create_serializable_metadata(df, name)
                            
                            if not safe_json_dump(metadata, datasets_dir / f"{name}_meta.json"):
                                logger.warning(f"Failed to save metadata for dataset {name}")
                
                if models:
                    models_dir = temp_path / "models"
                    models_dir.mkdir()
                    
                    for name, model in models.items():
                        if isinstance(model, BespokeDecisionTree):
                            model_dict = model.to_dict() if hasattr(model, 'to_dict') else {'error': 'Model serialization not available'}
                            
                            if not safe_json_dump(model_dict, models_dir / f"{name}.json"):
                                logger.warning(f"Failed to save model {name} as JSON")
                            
                            try:
                                with open(models_dir / f"{name}.pkl", 'wb') as f:
                                    pickle.dump(model, f)
                            except Exception as e:
                                logger.warning(f"Could not pickle model {name}: {e}")
                
                if analysis_results:
                    analysis_dir = temp_path / "analysis"
                    analysis_dir.mkdir()
                    
                    for analysis_type, results in analysis_results.items():
                        if not safe_json_dump(results, analysis_dir / f"{analysis_type}.json"):
                            logger.warning(f"Failed to save analysis results for {analysis_type}")
                
                if variable_importance:
                    importance_dir = temp_path / "importance"
                    importance_dir.mkdir()
                    
                    if not safe_json_dump(variable_importance, importance_dir / "variable_importance.json"):
                        logger.warning("Failed to save variable importance data")
                
                if performance_metrics:
                    metrics_dir = temp_path / "metrics"
                    metrics_dir.mkdir()
                    
                    if not safe_json_dump(performance_metrics, metrics_dir / "performance_metrics.json"):
                        logger.warning("Failed to save performance metrics")
                
                with zipfile.ZipFile(path_obj, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in temp_path.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(temp_path)
                            zipf.write(file_path, arcname)
            
            self.current_project_path = path_obj
            self.is_modified = False
            logger.info(f"Enhanced project saved successfully to {path_obj}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving enhanced project to {path_obj}: {e}", exc_info=True)
            return False

    def load_project_comprehensive(self, file_path: str, 
                                 workflow_canvas: WorkflowCanvas = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Load comprehensive project state from a .bspk file.
        
        Args:
            file_path: Path to the project file
            workflow_canvas: Workflow canvas to restore state to
            
        Returns:
            Tuple of (success, project_data) where project_data contains all loaded components
        """
        path_obj = Path(file_path)
        logger.info(f"Loading comprehensive project from: {path_obj}")
        
        if not path_obj.exists():
            logger.error(f"Project file not found: {path_obj}")
            return False, {}
        
        project_data = {
            'datasets': {},
            'models': {},
            'analysis_results': {},
            'variable_importance': {},
            'performance_metrics': {},
            'export_settings': {},
            'ui_state': {},
            'workflow_config': {},
            'metadata': {}
        }
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                with zipfile.ZipFile(path_obj, 'r') as zipf:
                    zipf.extractall(temp_path)
                
                main_config_path = temp_path / "project.json"
                if main_config_path.exists():
                    with open(main_config_path, 'r', encoding='utf-8') as f:
                        main_config = json.load(f)
                    
                    project_data['metadata'] = main_config.get('project_metadata', {})
                    project_data['workflow_config'] = main_config.get('workflow', {})
                    project_data['analysis_results'] = main_config.get('analysis_results', {})
                    project_data['variable_importance'] = main_config.get('variable_importance', {})
                    project_data['performance_metrics'] = main_config.get('performance_metrics', {})
                    project_data['export_settings'] = main_config.get('export_settings', {})
                    project_data['ui_state'] = main_config.get('ui_state', {})
                    
                    file_version = main_config.get("project_file_version", "1.0")
                    if file_version != self.PROJECT_FILE_VERSION:
                        logger.warning(f"Project file version {file_version} differs from current {self.PROJECT_FILE_VERSION}")
                
                datasets_dir = temp_path / "datasets"
                if datasets_dir.exists():
                    for parquet_file in datasets_dir.glob("*.parquet"):
                        dataset_name = parquet_file.stem
                        try:
                            df = pd.read_parquet(parquet_file)
                            project_data['datasets'][dataset_name] = df
                            
                            meta_file = datasets_dir / f"{dataset_name}_meta.json"
                            if meta_file.exists():
                                with open(meta_file, 'r') as f:
                                    meta = json.load(f)
                                project_data['datasets'][f"{dataset_name}_meta"] = meta
                                
                            logger.info(f"Loaded dataset: {dataset_name} with shape {df.shape}")
                            
                        except Exception as e:
                            logger.error(f"Error loading dataset {dataset_name}: {e}")
                
                models_dir = temp_path / "models"
                if models_dir.exists():
                    for model_file in models_dir.glob("*.json"):
                        model_name = model_file.stem
                        try:
                            with open(model_file, 'r') as f:
                                model_dict = json.load(f)
                            
                            pickle_file = models_dir / f"{model_name}.pkl"
                            if pickle_file.exists():
                                try:
                                    with open(pickle_file, 'rb') as f:
                                        model = pickle.load(f)
                                    project_data['models'][model_name] = model
                                    logger.info(f"Loaded model from pickle: {model_name}")
                                    continue
                                except Exception as e:
                                    logger.warning(f"Could not load pickle model {model_name}: {e}")
                            
                            if hasattr(BespokeDecisionTree, 'from_dict'):
                                model = BespokeDecisionTree.from_dict(model_dict, self.app_config)
                                project_data['models'][model_name] = model
                                logger.info(f"Loaded model from JSON: {model_name}")
                            else:
                                project_data['models'][model_name] = model_dict
                                logger.warning(f"Loaded model as dict (no from_dict method): {model_name}")
                                
                        except Exception as e:
                            logger.error(f"Error loading model {model_name}: {e}")
                
                analysis_dir = temp_path / "analysis"
                if analysis_dir.exists():
                    for analysis_file in analysis_dir.glob("*.json"):
                        analysis_type = analysis_file.stem
                        try:
                            with open(analysis_file, 'r') as f:
                                analysis_data = json.load(f)
                            project_data['analysis_results'][analysis_type] = analysis_data
                            
                        except Exception as e:
                            logger.error(f"Error loading analysis {analysis_type}: {e}")
                
                importance_dir = temp_path / "importance"
                if importance_dir.exists():
                    importance_file = importance_dir / "variable_importance.json"
                    if importance_file.exists():
                        try:
                            with open(importance_file, 'r') as f:
                                importance_data = json.load(f)
                            project_data['variable_importance'] = importance_data
                            
                        except Exception as e:
                            logger.error(f"Error loading variable importance: {e}")
                
                metrics_dir = temp_path / "metrics"
                if metrics_dir.exists():
                    metrics_file = metrics_dir / "performance_metrics.json"
                    if metrics_file.exists():
                        try:
                            with open(metrics_file, 'r') as f:
                                metrics_data = json.load(f)
                            project_data['performance_metrics'] = metrics_data
                            
                        except Exception as e:
                            logger.error(f"Error loading performance metrics: {e}")
                
                self._cached_datasets = project_data['datasets']
                self._cached_models = project_data['models']
                self._cached_analysis_results = project_data['analysis_results']
                
                if workflow_canvas and project_data['workflow_config']:
                    try:
                        self._restore_workflow(workflow_canvas, project_data['workflow_config'])
                    except Exception as e:
                        logger.error(f"Error restoring workflow: {e}")
                
                self.current_project_path = path_obj
                self.is_modified = False
                self.project_metadata = project_data['metadata']
                
                logger.info(f"Enhanced project loaded successfully from {path_obj}")
                return True, project_data
                
        except Exception as e:
            logger.error(f"Error loading enhanced project from {path_obj}: {e}", exc_info=True)
            return False, {}

    def _serialize_workflow(self, workflow_canvas: WorkflowCanvas) -> Dict[str, Any]:
        """Serialize workflow canvas state."""
        if not workflow_canvas:
            return {}
        
        try:
            return workflow_canvas.scene.get_config() if hasattr(workflow_canvas, 'scene') else {}
        except Exception as e:
            logger.error(f"Error serializing workflow: {e}")
            return {}

    def _serialize_datasets_metadata(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Serialize dataset metadata."""
        if not datasets:
            return {}
        
        metadata = {}
        for name, df in datasets.items():
            if isinstance(df, pd.DataFrame):
                base_metadata = create_serializable_metadata(df, name)
                base_metadata['checksum'] = self._calculate_dataframe_checksum(df)
                metadata[name] = base_metadata
        
        return metadata

    def _serialize_models_metadata(self, models: Dict[str, BespokeDecisionTree]) -> Dict[str, Any]:
        """Serialize model metadata."""
        if not models:
            return {}
        
        metadata = {}
        for name, model in models.items():
            if isinstance(model, BespokeDecisionTree):
                metadata[name] = {
                    'name': name,
                    'model_type': 'BespokeDecisionTree',
                    'creation_date': datetime.now().isoformat(),
                    'tree_depth': getattr(model, 'max_depth', None),
                    'n_features': getattr(model, 'n_features_', None),
                    'n_classes': getattr(model, 'n_classes_', None)
                }
        
        return metadata

    def _calculate_dataframe_checksum(self, df: pd.DataFrame) -> str:
        """Calculate checksum for dataframe integrity verification."""
        try:
            content = df.to_string().encode('utf-8')
            return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate dataframe checksum: {e}")
            return ""

    def _restore_workflow(self, workflow_canvas: WorkflowCanvas, workflow_config: Dict[str, Any]):
        """Restore workflow canvas state."""
        try:
            if hasattr(workflow_canvas, 'scene') and hasattr(workflow_canvas.scene, 'set_config'):
                if hasattr(self, 'main_window_ref') and self.main_window_ref:
                    workflow_canvas.scene.main_window_ref = self.main_window_ref
                    
                    if hasattr(self.main_window_ref, 'models') and hasattr(self, '_cached_models'):
                        original_models = getattr(self.main_window_ref, 'models', {}).copy()
                        self.main_window_ref.models = self._cached_models.copy()
                        logger.info(f"Temporarily populated main window with {len(self._cached_models)} models for workflow restoration")
                        
                        try:
                            workflow_canvas.scene.set_config(workflow_config)
                            logger.info("Workflow restored successfully with model associations")
                        except Exception as workflow_error:
                            self.main_window_ref.models = original_models
                            logger.error(f"Error during workflow restoration: {workflow_error}")
                            raise
                    else:
                        workflow_canvas.scene.set_config(workflow_config)
                        logger.info("Workflow restored successfully (no models to associate)")
                else:
                    workflow_canvas.scene.set_config(workflow_config)
                    logger.info("Workflow restored successfully (no main window reference)")
        except Exception as e:
            logger.error(f"Error restoring workflow: {e}")

    def get_project_info(self) -> Dict[str, Any]:
        """Get comprehensive project information."""
        return {
            'current_path': str(self.current_project_path) if self.current_project_path else None,
            'project_name': self.get_project_name(),
            'is_modified': self.is_modified,
            'metadata': self.project_metadata,
            'cached_datasets': list(self._cached_datasets.keys()),
            'cached_models': list(self._cached_models.keys()),
            'cached_analysis': list(self._cached_analysis_results.keys())
        }

    def export_project_summary(self, output_path: str) -> bool:
        """Export a summary of the project to a text file."""
        try:
            info = self.get_project_info()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("BESPOKE UTILITY PROJECT SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Project Name: {info['project_name']}\n")
                f.write(f"Current Path: {info['current_path']}\n")
                f.write(f"Is Modified: {info['is_modified']}\n\n")
                
                metadata = info['metadata']
                f.write("PROJECT METADATA\n")
                f.write("-" * 20 + "\n")
                f.write(f"Created: {metadata.get('created_date', 'Unknown')}\n")
                f.write(f"Last Modified: {metadata.get('last_modified', 'Unknown')}\n")
                f.write(f"Author: {metadata.get('author', 'Unknown')}\n")
                f.write(f"Description: {metadata.get('description', 'No description')}\n")
                f.write(f"Tags: {', '.join(metadata.get('tags', []))}\n")
                f.write(f"Application Version: {metadata.get('application_version', 'Unknown')}\n\n")
                
                f.write("VERSION HISTORY\n")
                f.write("-" * 15 + "\n")
                for version in metadata.get('version_history', []):
                    f.write(f"Version {version.get('version', 'N/A')}: {version.get('date', 'Unknown')} - {version.get('description', 'No description')}\n")
                f.write("\n")
                
                f.write("CACHED DATA\n")
                f.write("-" * 11 + "\n")
                f.write(f"Datasets: {', '.join(info['cached_datasets']) if info['cached_datasets'] else 'None'}\n")
                f.write(f"Models: {', '.join(info['cached_models']) if info['cached_models'] else 'None'}\n")
                f.write(f"Analysis Results: {', '.join(info['cached_analysis']) if info['cached_analysis'] else 'None'}\n")
                
            logger.info(f"Project summary exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting project summary: {e}")
            return False

    def set_modified(self, modified: bool = True):
        """Mark project as modified or unmodified."""
        self.is_modified = modified
        if self.main_window_ref:
            self.main_window_ref.set_project_modified(modified, update_manager=False)

    def get_current_project_path(self) -> Optional[Path]:
        """Get the current project file path."""
        return self.current_project_path

    def get_project_name(self) -> str:
        """Get the project name."""
        if self.current_project_path:
            return self.current_project_path.stem
        return "New Project"

    def save_project(self, file_path: str, workflow_canvas: WorkflowCanvas,
                     datasets: Dict[str, pd.DataFrame], 
                     models: Dict[str, BespokeDecisionTree]) -> bool:
        """Legacy method - delegates to comprehensive save."""
        logger.info("Using legacy save_project method - delegating to comprehensive save")
        return self.save_project_comprehensive(
            file_path=file_path,
            workflow_canvas=workflow_canvas,
            datasets=datasets,
            models=models,
            include_data=True
        )

    def load_project(self, file_path: str, workflow_canvas: WorkflowCanvas) -> \
                     Tuple[bool, Dict[str, pd.DataFrame], Dict[str, BespokeDecisionTree]]:
        """Legacy method - delegates to comprehensive load."""
        logger.info("Using legacy load_project method - delegating to comprehensive load")
        success, project_data = self.load_project_comprehensive(file_path, workflow_canvas)
        
        return success, project_data.get('datasets', {}), project_data.get('models', {})


ProjectManager = EnhancedProjectManager


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    pm = EnhancedProjectManager(config={"application": {"version": "1.0.0"}})
    print(f"Enhanced ProjectManager initialized. Default project path: {pm.get_current_project_path()}")
    
    pm.new_project(
        description="Test enhanced project", 
        author="Bespoke Utility", 
        tags=["test", "enhanced", "comprehensive"]
    )
    
    print(f"Project info: {pm.get_project_info()}")
    
    if pm.export_project_summary("test_project_summary.txt"):
        print("Project summary exported successfully")
        if os.path.exists("test_project_summary.txt"):
            os.remove("test_project_summary.txt")
            print("Cleaned up test summary file")