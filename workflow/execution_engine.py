# utility/workflow/execution_engine.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Workflow Execution Engine for Bespoke Utility
Handles the execution of the analysis workflow defined on the canvas

"""

import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal, QRunnable, QThreadPool, QMetaObject, Qt
from models.decision_tree import BespokeDecisionTree

from ui.workflow_canvas import (
    WorkflowNode, DatasetNode, ModelNode, FilterNode, TransformNode,
    EvaluationNode, VisualizationNode, ExportNode, ConnectionData,
    NODE_TYPE_DATASET, NODE_TYPE_MODEL, NODE_TYPE_FILTER,
    NODE_TYPE_TRANSFORM, NODE_TYPE_EVALUATION,
    NODE_TYPE_VISUALIZATION, NODE_TYPE_EXPORT
)
from data.data_loader import DataLoader # Though loading might be handled by ProjectManager
from data.data_processor import DataProcessor
from data.feature_engineering import FeatureEngineering
from models.decision_tree import BespokeDecisionTree
from analytics.performance_metrics import MetricsCalculator
from export.model_saver import ModelSaver

logger = logging.getLogger(__name__)

class WorkflowTask(QRunnable):
    """
    A runnable task for executing a part of the workflow.
    This allows for potential background execution.
    
    Note: QRunnable cannot have signals directly. Signals are handled by the parent
    WorkflowExecutionEngine which inherits from QObject.
    """

    def __init__(self, execution_engine: 'WorkflowExecutionEngine', start_node_id: Optional[str] = None):
        super().__init__()
        self.execution_engine = execution_engine
        self.start_node_id = start_node_id
    
    def _emit_signal_threadsafe(self, signal_name, *args):
        """Emit signal in a thread-safe manner"""
        try:
            from PyQt5.QtCore import QTimer
            
            def emit_signal():
                try:
                    self.execution_engine.emit_signal_proxy(signal_name, *args)
                except Exception as e:
                    logger.error(f"Error in delayed signal emission: {e}")
            
            QTimer.singleShot(0, emit_signal)
            
        except Exception as e:
            logger.error(f"Error scheduling signal emission: {e}")
            try:
                self.execution_engine.emit_signal_proxy(signal_name, *args)
            except Exception as e2:
                logger.error(f"Direct signal emission also failed: {e2}")

    def run(self):
        """Execute the workflow."""
        try:
            if self.start_node_id:
                logger.info(f"Starting workflow execution from node: {self.start_node_id}")
                # TODO: Implement logic to execute only a part of the graph
                ordered_nodes_to_run = self.execution_engine.get_downstream_nodes(self.start_node_id)
                if not ordered_nodes_to_run:
                    logger.error(f"Could not determine execution path from node {self.start_node_id}")
                    self._emit_signal_threadsafe("workflowExecutionError", f"Cannot run from node {self.start_node_id}.")
                    return
                self.execution_engine._execute_graph(ordered_nodes_to_run)

            else:
                logger.info("Starting full workflow execution.")
                if not self.execution_engine.execution_order:
                    logger.error("Execution order is empty. Cannot run workflow.")
                    self._emit_signal_threadsafe("workflowExecutionError", "Execution order not determined.")
                    return
                self.execution_engine._execute_graph(self.execution_engine.execution_order)

            self._emit_signal_threadsafe("workflowExecutionSuccess", "Workflow executed successfully.")
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            self._emit_signal_threadsafe("workflowExecutionError", str(e))
        finally:
            self.execution_engine.is_running = False
            self._emit_signal_threadsafe("workflowExecutionFinished")


class WorkflowExecutionEngine(QObject):
    """
    Manages the execution of a visual workflow.
    It takes a workflow definition (nodes and connections) and processes it.
    """
    workflowExecutionStarted = pyqtSignal()
    workflowExecutionFinished = pyqtSignal() # Emitted on success or failure after completion
    workflowExecutionError = pyqtSignal(str)
    workflowExecutionSuccess = pyqtSignal(str) # Message on success
    nodeProcessingStarted = pyqtSignal(str) # node_id
    nodeProcessingFinished = pyqtSignal(str, str) # node_id, status ("success", "failure", "skipped")
    nodeOutputReady = pyqtSignal(str, object, str) # node_id, output_data, output_type (e.g. 'dataframe', 'model')
    requestNodeConfig = pyqtSignal(str) # node_id, emits when a node's config is needed if not already on node object

    def __init__(self, config: Dict[str, Any], project_data_manager = None): # Use ProjectManager type hint if available
        super().__init__()
        self.config = config
        self.workflow_nodes: Dict[str, WorkflowNode] = {}
        self.workflow_connections: List[ConnectionData] = [] # Store as list
        self.execution_order: List[str] = []
        self.node_outputs: Dict[str, Dict[str, Any]] = {} # Stores outputs: {node_id: {'port_name': data, ...}}
        self.is_running = False
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(1) # For now, run sequentially in a background thread

        self.data_loader = DataLoader(config) # May not be directly used if project_manager handles all data
        self.data_processor = DataProcessor(config)
        self.feature_engineer = FeatureEngineering(config)
        self.metrics_calculator = MetricsCalculator()
        self.model_saver = ModelSaver(config)
        
        self.project_data_manager = project_data_manager
        if self.project_data_manager:
            logger.info("WorkflowExecutionEngine initialized with ProjectDataManager.")
        else:
            logger.warning("WorkflowExecutionEngine initialized WITHOUT ProjectDataManager. DatasetNode functionality will be limited.")

        logger.info("WorkflowExecutionEngine initialized.")

    def emit_signal_proxy(self, signal_name, *args):
        """Thread-safe signal emission proxy method"""
        try:
            if hasattr(self, signal_name):
                getattr(self, signal_name).emit(*args)
            else:
                logger.error(f"Signal {signal_name} not found")
        except Exception as e:
            logger.error(f"Error in signal proxy: {e}")

    def set_workflow(self, nodes: Dict[str, WorkflowNode], connections: List[ConnectionData]):
        """
        Sets the current workflow to be executed.
        Args:
            nodes: A dictionary of WorkflowNode objects, keyed by node_id.
            connections: A list of ConnectionData objects.
        """
        self.workflow_nodes = nodes
        self.workflow_connections = connections # Expecting a list now
        self.node_outputs = {}
        logger.info(f"Workflow set with {len(nodes)} nodes and {len(connections)} connections.")
        if not self._determine_execution_order():
            self.workflowExecutionError.emit("Invalid workflow structure (cycle detected or disconnected).")
            self.execution_order = []


    def _determine_execution_order(self) -> bool:
        """
        Determines the execution order of nodes using topological sort.
        Only includes connected nodes in execution order, ignoring isolated nodes.
        Returns:
            True if a valid execution order is found, False otherwise.
        """
        connected_nodes = set()
        for conn in self.workflow_connections:
            connected_nodes.add(conn.source_node_id)
            connected_nodes.add(conn.target_node_id)
        
        if not connected_nodes and self.workflow_nodes:
            self.execution_order = list(self.workflow_nodes.keys())
            logger.info(f"No connections found. All {len(self.execution_order)} nodes are independent.")
            return True
        
        adj: Dict[str, List[str]] = {node_id: [] for node_id in connected_nodes}
        in_degree: Dict[str, int] = {node_id: 0 for node_id in connected_nodes}

        for conn in self.workflow_connections:
            source_id = conn.source_node_id
            target_id = conn.target_node_id
            if target_id not in adj[source_id]:
                adj[source_id].append(target_id)
            in_degree[target_id] += 1

        queue = [node_id for node_id in connected_nodes if in_degree[node_id] == 0]
        self.execution_order = []

        while queue:
            u = queue.pop(0)
            self.execution_order.append(u)
            if u in adj:
                for v_node_id in adj[u]:
                    in_degree[v_node_id] -= 1
                    if in_degree[v_node_id] == 0:
                        queue.append(v_node_id)

        if len(self.execution_order) == len(connected_nodes):
            unconnected_nodes = [node_id for node_id in self.workflow_nodes if node_id not in connected_nodes]
            if unconnected_nodes:
                logger.info(f"Found {len(unconnected_nodes)} unconnected nodes - adding to execution order")
                self.execution_order.extend(unconnected_nodes)
            
            logger.info(f"Execution order determined: {self.execution_order}")
            logger.info(f"Connected nodes: {len(connected_nodes)}, Unconnected nodes: {len(unconnected_nodes)}")
            return True
        else:
            problem_nodes = connected_nodes - set(self.execution_order)
            logger.error(f"Cycle detected in connected components. Problematic nodes: {problem_nodes}. Cannot execute.")
            self.execution_order = []
            return False

    def get_downstream_nodes(self, start_node_id: str) -> List[str]:
        """Determines the execution order for nodes downstream of start_node_id, including itself."""
        if not self.workflow_nodes or start_node_id not in self.workflow_nodes:
            return []

        adj: Dict[str, List[str]] = {node_id: [] for node_id in self.workflow_nodes}
        for conn in self.workflow_connections:
            source_id = conn.source_node_id
            target_id = conn.target_node_id
            if target_id not in adj[source_id]:
                adj[source_id].append(target_id)

        downstream_execution_order = []
        queue = [start_node_id]
        visited_for_subgraph = {start_node_id} # Keep track of nodes added to the current subgraph execution

        if not self.execution_order:
            if not self._determine_execution_order(): # Try to build full order if not present
                 return [] # Cannot proceed

        try:
            start_index = self.execution_order.index(start_node_id)
        except ValueError:
            logger.error(f"Start node {start_node_id} not found in the determined execution order.")
            return []

        
        
        nodes_in_subgraph = set()
        processing_q = [start_node_id]
        visited_bfs_dfs = set()

        while processing_q:
            curr = processing_q.pop(0)
            if curr in visited_bfs_dfs:
                continue
            visited_bfs_dfs.add(curr)
            nodes_in_subgraph.add(curr)
            if curr in adj:
                for neighbor in adj[curr]:
                    if neighbor not in visited_bfs_dfs:
                        processing_q.append(neighbor)
        
        downstream_execution_order = [node for node in self.execution_order if node in nodes_in_subgraph]

        if start_node_id not in downstream_execution_order or downstream_execution_order[0] != start_node_id:
             logger.warning(f"Could not form a valid downstream execution path starting with {start_node_id}. Full run might be required.")
             if start_node_id not in downstream_execution_order: return []


        logger.info(f"Determined downstream execution order from {start_node_id}: {downstream_execution_order}")
        return downstream_execution_order

    def _get_upstream_nodes(self, target_node_id: str) -> List[str]:
        """Get all nodes that are upstream dependencies of the target node."""
        if not self.workflow_nodes or target_node_id not in self.workflow_nodes:
            return []
            
        upstream_nodes = []
        for conn in self.workflow_connections:
            if conn.target_node_id == target_node_id:
                if conn.source_node_id not in upstream_nodes:
                    upstream_nodes.append(conn.source_node_id)
                    
        return upstream_nodes

    def run_workflow(self, start_node_id: Optional[str] = None):
        if self.is_running:
            logger.warning("Workflow is already running.")
            return

        if not self.workflow_nodes:
            self.workflowExecutionError.emit("Workflow is empty. Cannot run.")
            return

        nodes_to_run_list = []
        if start_node_id:
            downstream_nodes = self.get_downstream_nodes(start_node_id)
            if not downstream_nodes:
                 self.workflowExecutionError.emit(f"Cannot determine execution path from {start_node_id}.")
                 return
                 
            all_required_nodes = set(downstream_nodes)
            
            for node_id in downstream_nodes:
                upstream_deps = self._get_upstream_nodes(node_id)
                for upstream_id in upstream_deps:
                    if upstream_id not in self.node_outputs or "error" in self.node_outputs.get(upstream_id, {}):
                        all_required_nodes.add(upstream_id)
            
            nodes_to_run_list = [node for node in self.execution_order if node in all_required_nodes]
            
            for node_id_to_clear in nodes_to_run_list:
                if node_id_to_clear in self.node_outputs:
                    del self.node_outputs[node_id_to_clear]
                    logger.debug(f"Cleared previous output for node {node_id_to_clear} for partial run.")
        else: # Full run
            if not self.execution_order: # If not set during set_workflow or became invalid
                if not self._determine_execution_order():
                    self.workflowExecutionError.emit("Invalid workflow structure. Cannot run.")
                    return
            self.node_outputs = {} # Clear all previous outputs for a full run
            nodes_to_run_list = self.execution_order


        if not nodes_to_run_list:
            self.workflowExecutionError.emit("No nodes to execute.")
            return

        self.is_running = True
        self.workflowExecutionStarted.emit()
        

        current_task = WorkflowTask(self, start_node_id=start_node_id) # The task will internally decide the list
        self.thread_pool.start(current_task)


    def _execute_graph(self, ordered_nodes_to_process: List[str]):
        """
        Internal method to execute the graph based on a specific ordered list of node IDs.
        """
        for node_id in ordered_nodes_to_process:
            node = self.workflow_nodes.get(node_id)
            if not node:
                logger.warning(f"Node {node_id} not found in workflow_nodes during execution. Skipping.")
                self.nodeProcessingFinished.emit(node_id, "skipped")
                continue

            self.nodeProcessingStarted.emit(node_id)
            logger.info(f"Executing node: {node_id} ({node.title})")

            try:
                inputs = self._get_node_inputs(node_id)
                node_output_dict = self._execute_node(node, inputs)

                if node_output_dict is None: # Node execution itself indicated failure or no output
                    raise RuntimeError(f"Node {node_id} ({node.title}) did not produce an output dictionary.")

                self.node_outputs[node_id] = node_output_dict
                
                for port_name, data_val in node_output_dict.items():
                    output_type = "unknown"
                    if isinstance(data_val, pd.DataFrame): output_type = "dataframe"
                    elif isinstance(data_val, BespokeDecisionTree): output_type = "model"
                    elif isinstance(data_val, dict) and "accuracy" in data_val : output_type = "metrics"
                    
                    self.nodeOutputReady.emit(node_id, data_val, port_name) # Send port_name as output_type for now
                
                self.nodeProcessingFinished.emit(node_id, "success")
                logger.info(f"Finished executing node: {node_id}. Outputs: {list(node_output_dict.keys())}")

            except Exception as e:
                logger.error(f"Error executing node {node_id} ({node.title}): {e}", exc_info=True)
                self.nodeProcessingFinished.emit(node_id, "failure")
                self.node_outputs[node_id] = {"error": str(e)} # Mark as failed with error
                raise # Re-raise to be caught by the WorkflowTask runner

        self.is_running = False # Ensure this is set at the end of graph execution


    def _get_node_inputs(self, node_id: str) -> Dict[str, Any]:
        """
        Retrieves the necessary inputs for a given node from the outputs of its predecessors.
        Inputs are keyed by the target node's input port name.
        """
        inputs: Dict[str, Any] = {}
        node = self.workflow_nodes.get(node_id)
        if not node: return {}

        
        required_input_ports = {port_def['name'] for port_def in getattr(node, 'input_ports', [])}
        
        for conn in self.workflow_connections:
            if conn.target_node_id == node_id:
                source_node_id = conn.source_node_id
                source_port_name = conn.source_port.get('name', 'default_output') # From WorkflowNode.output_ports definition
                target_input_port_name = conn.target_port.get('name', 'default_input') # From WorkflowNode.input_ports definition

                if source_node_id in self.node_outputs:
                    source_node_all_outputs = self.node_outputs[source_node_id]
                    if source_port_name in source_node_all_outputs:
                        inputs[target_input_port_name] = source_node_all_outputs[source_port_name]
                        logger.debug(f"Input for {node_id}.{target_input_port_name} from {source_node_id}.{source_port_name}: {type(inputs[target_input_port_name])}")
                        if target_input_port_name in required_input_ports:
                            required_input_ports.remove(target_input_port_name)
                    else:
                        logger.warning(f"Output port '{source_port_name}' not found in outputs of source node {source_node_id} for target {node_id}.{target_input_port_name}. Available outputs: {list(source_node_all_outputs.keys())}")
                else:
                    logger.warning(f"Output from source node {source_node_id} not found for node {node_id}.{target_input_port_name}. Upstream node might have failed or not run.")

        if required_input_ports: # Some required inputs were not connected or source failed
            logger.warning(f"Node {node_id} is missing connections for required input ports: {required_input_ports}")

        return inputs

    def _execute_node(self, node: WorkflowNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a single node based on its type and stored configuration.
        Returns a dictionary of outputs, keyed by the node's output port names.
        """
        node_type = node.node_type
        outputs: Dict[str, Any] = {} # { 'Port Name': data_object }


        if node_type == NODE_TYPE_DATASET and isinstance(node, DatasetNode):
            if self.project_data_manager and hasattr(self.project_data_manager, 'datasets') and node.dataset_name in self.project_data_manager.datasets:
                df = self.project_data_manager.datasets[node.dataset_name]
                logger.info(f"DatasetNode '{node.node_id}': Loaded dataset '{node.dataset_name}' with {len(df)} rows, {len(df.columns)} cols.")
                outputs['Data Output'] = df # Assuming DatasetNode has one output port named 'Data Output'
            else:
                error_msg = f"DatasetNode '{node.node_id}': Dataset '{node.dataset_name}' not found in ProjectDataManager."
                logger.error(error_msg)
                raise ValueError(error_msg)

        elif node_type == NODE_TYPE_FILTER and isinstance(node, FilterNode):
            df_input = inputs.get('Data Input') # Name of the input port
            if not isinstance(df_input, pd.DataFrame):
                raise ValueError(f"FilterNode '{node.node_id}' expects a DataFrame for 'Data Input', got {type(df_input)}")
            
            filter_conditions = getattr(node, 'conditions', []) # Default to empty list
            if not filter_conditions:
                logger.warning(f"FilterNode '{node.node_id}' has no conditions defined. Passing data through.")
                outputs['Filtered Data'] = df_input.copy() # Pass copy to avoid modification issues
            else:
                logger.info(f"FilterNode '{node.node_id}': Applying {len(filter_conditions)} conditions.")
                # Log filter conditions and data statistics for debugging
                logger.info(f"FilterNode '{node.node_id}' conditions: {filter_conditions}")
                logger.info(f"FilterNode '{node.node_id}' input data shape: {df_input.shape}")
                
                for i, cond in enumerate(filter_conditions):
                    col_name = cond.get('column')
                    operator = cond.get('operator')
                    value = cond.get('value')
                    
                    if col_name in df_input.columns:
                        col_data = df_input[col_name]
                        logger.info(f"Condition {i}: {col_name} {operator} {value}")
                        logger.info(f"  Column '{col_name}' - type: {col_data.dtype}, min: {col_data.min()}, max: {col_data.max()}")
                        
                        try:
                            if operator == '>':
                                test_mask = col_data > float(value) if isinstance(value, str) else col_data > value
                                matching_rows = test_mask.sum()
                                logger.info(f"  Rows matching '{col_name} {operator} {value}': {matching_rows}/{len(col_data)} ({matching_rows/len(col_data)*100:.1f}%)")
                                if matching_rows == 0:
                                    logger.warning(f"  NO ROWS match condition '{col_name} {operator} {value}' - this may explain why filter has no effect")
                        except Exception as e:
                            logger.error(f"  Error testing condition: {e}")
                    else:
                        logger.error(f"Column '{col_name}' not found in input data!")
                
                filtered_df = self.data_processor.filter_data(df_input, filter_conditions, dataset_name=node.title)
                outputs['Filtered Data'] = filtered_df
                logger.info(f"FilterNode '{node.node_id}' output: {len(filtered_df)} rows.")

        elif node_type == NODE_TYPE_TRANSFORM and isinstance(node, TransformNode):
            df_input = inputs.get('Data Input')
            if not isinstance(df_input, pd.DataFrame):
                raise ValueError(f"TransformNode '{node.node_id}' expects a DataFrame for 'Data Input', got {type(df_input)}")

            transformations = getattr(node, 'transformations', []) # Default to empty list
            if not transformations:
                logger.warning(f"TransformNode '{node.node_id}' has no transformations. Passing data through.")
                outputs['Transformed Data'] = df_input.copy()
            else:
                logger.info(f"TransformNode '{node.node_id}': Applying {len(transformations)} transformations.")
                current_df = df_input.copy() # Work on a copy
                for transform_config in transformations:
                    ttype = transform_config.get('type')
                    try:
                        if ttype == 'formula':
                            current_df = self.feature_engineer.create_formula_variable(
                                current_df,
                                transform_config['formula'],
                                transform_config['new_column'],
                                transform_config.get('var_type', 'float')
                            )
                        elif ttype == 'create_variable':
                            formula = transform_config.get('formula', '')
                            target_column = transform_config.get('target_column', transform_config.get('new_column', 'new_variable'))
                            if formula:
                                current_df = self.feature_engineer.create_formula_variable(
                                    current_df,
                                    formula,
                                    target_column,
                                    transform_config.get('var_type', 'float')
                                )
                                logger.info(f"TransformNode '{node.node_id}': Created variable '{target_column}' using formula '{formula}'")
                            else:
                                logger.warning(f"TransformNode '{node.node_id}': create_variable transformation missing formula")
                        elif ttype == 'derive_ratio':
                            source_columns = transform_config.get('source_columns', [])
                            target_column = transform_config.get('target_column', 'ratio_variable')
                            if len(source_columns) >= 2:
                                formula = f"{source_columns[0]} / {source_columns[1]}"
                                current_df = self.feature_engineer.create_formula_variable(
                                    current_df, formula, target_column, 'float'
                                )
                                logger.info(f"TransformNode '{node.node_id}': Created ratio '{target_column}' = {formula}")
                            else:
                                logger.warning(f"TransformNode '{node.node_id}': derive_ratio needs at least 2 source columns")
                        elif ttype == 'derive_difference':
                            source_columns = transform_config.get('source_columns', [])
                            target_column = transform_config.get('target_column', 'diff_variable')
                            if len(source_columns) >= 2:
                                formula = f"{source_columns[0]} - {source_columns[1]}"
                                current_df = self.feature_engineer.create_formula_variable(
                                    current_df, formula, target_column, 'float'
                                )
                                logger.info(f"TransformNode '{node.node_id}': Created difference '{target_column}' = {formula}")
                            else:
                                logger.warning(f"TransformNode '{node.node_id}': derive_difference needs at least 2 source columns")
                        elif ttype == 'derive_sum':
                            source_columns = transform_config.get('source_columns', [])
                            target_column = transform_config.get('target_column', 'sum_variable')
                            if source_columns:
                                formula = " + ".join(source_columns)
                                current_df = self.feature_engineer.create_formula_variable(
                                    current_df, formula, target_column, 'float'
                                )
                                logger.info(f"TransformNode '{node.node_id}': Created sum '{target_column}' = {formula}")
                            else:
                                logger.warning(f"TransformNode '{node.node_id}': derive_sum needs source columns")
                        elif ttype == 'encode_categorical':
                            # Simple categorical encoding (could be enhanced with one-hot encoding)
                            source_columns = transform_config.get('source_columns', [])
                            target_column = transform_config.get('target_column', 'encoded_variable')
                            if source_columns:
                                source_col = source_columns[0]
                                if source_col in current_df.columns:
                                    # Simple label encoding
                                    from sklearn.preprocessing import LabelEncoder
                                    le = LabelEncoder()
                                    current_df[target_column] = le.fit_transform(current_df[source_col].astype(str))
                                    logger.info(f"TransformNode '{node.node_id}': Encoded categorical '{source_col}' to '{target_column}'")
                                else:
                                    logger.warning(f"TransformNode '{node.node_id}': Source column '{source_col}' not found")
                            else:
                                logger.warning(f"TransformNode '{node.node_id}': encode_categorical needs source column")
                        elif ttype == 'log_transform':
                            source_columns = transform_config.get('source_columns', [])
                            target_column = transform_config.get('target_column', 'log_variable')
                            if source_columns:
                                source_col = source_columns[0]
                                if source_col in current_df.columns:
                                    import numpy as np
                                    current_df[target_column] = np.log(current_df[source_col] + 1)
                                    logger.info(f"TransformNode '{node.node_id}': Log transformed '{source_col}' to '{target_column}'")
                                else:
                                    logger.warning(f"TransformNode '{node.node_id}': Source column '{source_col}' not found")
                            else:
                                logger.warning(f"TransformNode '{node.node_id}': log_transform needs source column")
                        elif ttype == 'standardize':
                            source_columns = transform_config.get('source_columns', [])
                            target_column = transform_config.get('target_column', 'standardized_variable')
                            if source_columns:
                                source_col = source_columns[0]
                                if source_col in current_df.columns:
                                    from sklearn.preprocessing import StandardScaler
                                    scaler = StandardScaler()
                                    current_df[target_column] = scaler.fit_transform(current_df[[source_col]]).flatten()
                                    logger.info(f"TransformNode '{node.node_id}': Standardized '{source_col}' to '{target_column}'")
                                else:
                                    logger.warning(f"TransformNode '{node.node_id}': Source column '{source_col}' not found")
                            else:
                                logger.warning(f"TransformNode '{node.node_id}': standardize needs source column")
                        elif ttype == 'binning':
                            source_columns = transform_config.get('source_columns', [])
                            target_column = transform_config.get('target_column', 'binned_variable')
                            bins = transform_config.get('bins', 5)
                            if source_columns:
                                source_col = source_columns[0]
                                if source_col in current_df.columns:
                                    current_df[target_column] = pd.cut(current_df[source_col], bins=bins, labels=False)
                                    logger.info(f"TransformNode '{node.node_id}': Binned '{source_col}' to '{target_column}' with {bins} bins")
                                else:
                                    logger.warning(f"TransformNode '{node.node_id}': Source column '{source_col}' not found")
                            else:
                                logger.warning(f"TransformNode '{node.node_id}': binning needs source column")
                        else:
                            logger.warning(f"TransformNode '{node.node_id}': Unknown transformation type '{ttype}'. Skipping.")
                    except Exception as e:
                        logger.error(f"TransformNode '{node.node_id}': Error applying transformation '{ttype}': {e}")
                outputs['Transformed Data'] = current_df
                logger.info(f"TransformNode '{node.node_id}' output: {len(current_df)} rows.")


        elif node_type == NODE_TYPE_MODEL and isinstance(node, ModelNode):
            df_input = inputs.get('Data Input')
            if not isinstance(df_input, pd.DataFrame):
                raise ValueError(f"ModelNode '{node.node_id}' expects a DataFrame input for 'Data Input', got {type(df_input)}")
            
            if df_input.empty:
                raise ValueError(f"ModelNode '{node.node_id}' received empty DataFrame")

            model_instance = getattr(node, 'model', None) # Get pre-configured model from the node
            
            if not isinstance(model_instance, BespokeDecisionTree):
                logger.warning(f"ModelNode '{node.node_id}' missing model instance, creating new one")
                try:
                    model_instance = BespokeDecisionTree(self.config or {})
                    node.model = model_instance
                    
                    node_config = getattr(node, 'get_config', lambda: {})()
                    saved_target = node_config.get('target_variable')
                    if saved_target:
                        model_instance.target_name = saved_target
                        logger.info(f"Restored target variable '{saved_target}' for newly created ModelNode {node.node_id}")
                    
                    logger.info(f"Created new BespokeDecisionTree instance for node {node.node_id}")
                except Exception as e:
                    raise ValueError(f"ModelNode '{node.node_id}' does not have a valid BespokeDecisionTree model instance and could not create one: {e}")
            
            target_variable = getattr(model_instance, 'target_name', None)
            available_columns = list(df_input.columns)
            
            if not target_variable:
                node_config = getattr(node, 'get_config', lambda: {})()
                saved_target = node_config.get('target_variable')
                
                if saved_target and saved_target in df_input.columns:
                    model_instance.target_name = saved_target
                    target_variable = saved_target
                    logger.info(f"ModelNode '{node.node_id}' restored target variable from saved config: '{saved_target}'")
                else:
                    potential_target = df_input.columns[-1]
                    logger.warning(f"ModelNode '{node.node_id}' has no target variable set, using last column: '{potential_target}'")
                    model_instance.target_name = potential_target
                    target_variable = potential_target
            
            if target_variable not in available_columns:
                raise ValueError(f"Target column '{target_variable}' (from model config) not found in input data for ModelNode '{node.node_id}'. "
                               f"Available columns: {', '.join(available_columns[:10])}{'...' if len(available_columns) > 10 else ''}")
            
            target_series = df_input[target_variable]
            if target_series.isnull().all():
                raise ValueError(f"Target variable '{target_variable}' contains only null values for ModelNode '{node.node_id}'")
            
            unique_values = target_series.dropna().unique()
            if len(unique_values) < 2:
                raise ValueError(f"Target variable '{target_variable}' must have at least 2 unique values for classification. "
                               f"Found {len(unique_values)} unique values: {unique_values}")
            
            logger.info(f"ModelNode '{node.node_id}': Target variable '{target_variable}' has {len(unique_values)} unique values: {unique_values}")
            
            def has_manual_splits(model):
                """Check if model has manually created splits that should be preserved"""
                if not model.is_fitted or not model.root:
                    return False
                
                has_children = hasattr(model.root, 'children') and model.root.children and len(model.root.children) > 0
                is_manual_mode = hasattr(model, 'growth_mode') and model.growth_mode.value == 'manual'
                
                return has_children and is_manual_mode
            
            skip_retraining = False
            if model_instance.is_fitted and has_manual_splits(model_instance):
                current_features = set(df_input.columns) - {target_variable}
                model_features = set(getattr(model_instance, 'feature_names', []))
                
                if (model_instance.target_name == target_variable and 
                    current_features == model_features and
                    len(df_input) > 0):
                    skip_retraining = True
                    actual_node_count = len(model_instance._get_all_nodes()) if hasattr(model_instance, '_get_all_nodes') else model_instance.num_nodes
                    logger.info(f"ModelNode '{node.node_id}': Preserving manually built tree with {actual_node_count} nodes. Skipping retraining.")
            
            if not skip_retraining:
                X = df_input.drop(columns=[target_variable])
                y = df_input[target_variable]
                
                if X.empty or len(X.columns) == 0:
                    raise ValueError(f"ModelNode '{node.node_id}' has no feature columns after removing target variable '{target_variable}'")
                
                logger.info(f"ModelNode '{node.node_id}': Training model '{model_instance.model_name}' with {len(X)} samples and {len(X.columns)} features...")
                
                
                try:
                    model_instance.fit(X, y) # This should be synchronous for now or use QRunnable for model fitting too
                    logger.info(f"ModelNode '{node.node_id}': Model '{model_instance.model_name}' trained successfully.")
                except Exception as training_error:
                    logger.error(f"ModelNode '{node.node_id}': Model training failed: {str(training_error)}")
                    raise ValueError(f"ModelNode '{node.node_id}': Model training failed: {str(training_error)}")
            else:
                X = df_input.drop(columns=[target_variable])
                y = df_input[target_variable]
                
                if hasattr(model_instance, '_cached_X') and model_instance._cached_X is None:
                    model_instance._cached_X = X.copy()
                    model_instance._cached_y = y.copy()
                    logger.info(f"ModelNode '{node.node_id}': Updated cached training data for fitted model.")
            
            outputs['Model Output'] = model_instance # The output is the trained model

        elif node_type == NODE_TYPE_EVALUATION and isinstance(node, EvaluationNode):
            model_input = inputs.get('Model Input') # Expects a trained BespokeDecisionTree model
            data_input = inputs.get('Data Input')   # Expects a DataFrame for evaluation

            if not isinstance(model_input, BespokeDecisionTree):
                raise ValueError(f"EvaluationNode '{node.node_id}' expects a BespokeDecisionTree for 'Model Input', got {type(model_input)}")
            if not model_input.is_fitted:
                raise ValueError(f"Model for EvaluationNode '{node.node_id}' is not fitted.")
            if not isinstance(data_input, pd.DataFrame):
                raise ValueError(f"EvaluationNode '{node.node_id}' expects a DataFrame for 'Data Input', got {type(data_input)}")
            
            target_name = model_input.target_name
            if not target_name or target_name not in data_input.columns:
                 raise ValueError(f"Target column '{target_name}' not found in data for EvaluationNode '{node.node_id}'.")

            X_eval = data_input.drop(columns=[target_name])
            y_eval = data_input[target_name]
            
            
            from workflow.node_logic.evaluation_node_logic import EvaluationNodeLogic
            
            evaluation_logic = EvaluationNodeLogic(node.node_id, {})
            logger.info(f"EvaluationNode '{node.node_id}': Evaluating model '{model_input.model_name}'.")
            
            evaluation_inputs = {
                'Model Input': model_input,
                'Data Input': data_input
            }
            computed_metrics = evaluation_logic.execute(evaluation_inputs, node)

            outputs['Evaluation Results'] = computed_metrics # Dictionary of metrics
            logger.info(f"EvaluationNode '{node.node_id}' complete. Accuracy: {computed_metrics.get('accuracy', 'N/A')}")


        elif node_type == NODE_TYPE_VISUALIZATION and isinstance(node, VisualizationNode):
            viz_type = getattr(node, 'visualization_type', 'tree') # Configured on the node
            
            logger.info(f"VisualizationNode '{node.node_id}': Preparing for visualization type '{viz_type}'.")
            logger.info(f"Available inputs: {list(inputs.keys())}")
            
            model_input = inputs.get('Model Input')
            results_input = inputs.get('Results Input') 
            eval_results = inputs.get('Evaluation Results')
            data_input = inputs.get('Data Input')
            
            logger.info(f"Model Input type: {type(model_input)}")
            logger.info(f"Results Input type: {type(results_input)}")
            logger.info(f"Evaluation Results type: {type(eval_results)}")
            
            if viz_type == 'tree':
                if isinstance(model_input, BespokeDecisionTree):
                    data_ref = model_input
                    logger.info("Using Model Input as data_ref")
                elif isinstance(data_input, BespokeDecisionTree):
                    data_ref = data_input  
                    logger.info("Using Data Input as data_ref")
                elif isinstance(results_input, dict) and 'accuracy' in results_input:
                    data_ref = results_input
                    logger.info("Tree visualization received evaluation results - using Results Input as data_ref")
                else:
                    data_ref = "No valid tree model found"
                    logger.warning(f"No BespokeDecisionTree found in inputs: {list(inputs.keys())}")
            elif viz_type == 'metrics':
                if isinstance(results_input, dict) and 'accuracy' in results_input:
                    data_ref = results_input
                    logger.info("Using Results Input (evaluation metrics) as data_ref")
                elif isinstance(eval_results, dict) and 'accuracy' in eval_results:
                    data_ref = eval_results
                    logger.info("Using Evaluation Results as data_ref") 
                else:
                    data_ref = "No valid metrics found"
                    logger.warning(f"No metrics found in inputs: {list(inputs.keys())}")
            else:
                if isinstance(results_input, dict):
                    data_ref = results_input
                    logger.info("Using Results Input as data_ref")
                else:
                    data_ref = model_input or eval_results or data_input or "No input"
            
            logger.info(f"Final data_ref type: {type(data_ref)}")
            outputs['Visualization Trigger'] = {'type': viz_type, 'data_ref': data_ref}


        elif node_type == NODE_TYPE_EXPORT and isinstance(node, ExportNode):
            model_to_export = inputs.get('Model Input')
            data_to_export = inputs.get('Data Input') # Could be DataFrame or other results

            export_path = getattr(node, 'export_path', None)
            export_format = getattr(node, 'export_format', 'pmml') # Configured on the node
            
            if not export_path:
                raise ValueError(f"ExportNode '{node.node_id}': Export path not set.")

            if model_to_export and isinstance(model_to_export, BespokeDecisionTree):
                self.model_saver.save_model(model_to_export, export_path, format_type=export_format)
                outputs['Export Status'] = f"Model exported to {export_path} as {export_format}"
            elif data_to_export is not None and isinstance(data_to_export, pd.DataFrame):
                if export_format == 'csv': data_to_export.to_csv(export_path, index=False)
                elif export_format == 'excel': data_to_export.to_excel(export_path, index=False)
                else: raise ValueError(f"Unsupported export format '{export_format}' for DataFrame.")
                outputs['Export Status'] = f"Data exported to {export_path} as {export_format}"
            else:
                raise ValueError(f"ExportNode '{node.node_id}': No valid model or data input to export.")
            logger.info(outputs['Export Status'])

        else:
            logger.warning(f"Execution logic for node type '{node_type}' (Class: {node.__class__.__name__}) not fully implemented.")
            outputs['default_output'] = None # Or raise NotImplementedError(f"Execution for {node_type} not implemented")

        return outputs


    def stop_workflow(self):
        """Attempts to stop the current workflow execution."""
        if self.is_running:
            logger.info("Attempting to stop workflow execution.")
            self.is_running = False
            self.thread_pool.clear() # Clears tasks that haven't started
            self.workflowExecutionError.emit("Workflow execution was requested to stop.")
            self.workflowExecutionFinished.emit() # Signal that it's "finished" in a stopped state
            logger.info("Workflow stop requested. Running task may complete.")


    def get_node_output(self, node_id: str, port_name: str) -> Any:
        """
        Retrieves a specific output from a previously executed node.
        """
        node_outputs_dict = self.node_outputs.get(node_id, {})
        return node_outputs_dict.get(port_name)

    def get_all_node_outputs(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves all outputs of a previously executed node."""
        return self.node_outputs.get(node_id)

    def clear_outputs(self, specific_node_id: Optional[str] = None):
        """Clears stored node outputs, optionally for a specific node."""
        if specific_node_id:
            if specific_node_id in self.node_outputs:
                del self.node_outputs[specific_node_id]
                logger.info(f"Cleared outputs for node {specific_node_id}.")
        else:
            self.node_outputs = {}
            logger.info("Cleared all stored node outputs.")