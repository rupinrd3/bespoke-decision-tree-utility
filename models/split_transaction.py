#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Split Transaction Module for Bespoke Utility
Provides atomic split operations with rollback capability
"""

import logging
import copy
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
import uuid
from datetime import datetime

from models.node import TreeNode
from models.split_configuration import SplitConfiguration, ValidationResult
from models.split_validator import SplitValidator

logger = logging.getLogger(__name__)


class NodeSnapshot:
    """Snapshot of a node's state for rollback purposes"""
    
    def __init__(self, node: TreeNode):
        self.node_id = node.node_id
        self.parent_id = node.parent.node_id if node.parent else None
        self.depth = node.depth
        self.is_terminal = node.is_terminal
        
        self.split_feature = getattr(node, 'split_feature', None)
        self.split_value = getattr(node, 'split_value', None)
        self.split_rule = getattr(node, 'split_rule', None)
        self.split_type = getattr(node, 'split_type', None)
        self.split_categories = copy.deepcopy(getattr(node, 'split_categories', {}))
        self.split_operator = getattr(node, 'split_operator', None)
        
        self.samples = getattr(node, 'samples', 0)
        self.class_counts = copy.deepcopy(getattr(node, 'class_counts', {}))
        self.impurity = getattr(node, 'impurity', 0.0)
        self.majority_class = getattr(node, 'majority_class', None)
        
        self.child_ids = [child.node_id for child in node.children]
        
        self.timestamp = datetime.now()
        
    def restore_to_node(self, node: TreeNode, all_nodes: Dict[str, TreeNode]):
        """Restore this snapshot to a node"""
        node.depth = self.depth
        node.is_terminal = self.is_terminal
        
        node.split_feature = self.split_feature
        node.split_value = self.split_value
        node.split_rule = self.split_rule
        node.split_type = self.split_type
        node.split_categories = copy.deepcopy(self.split_categories)
        node.split_operator = self.split_operator
        
        node.samples = self.samples
        node.class_counts = copy.deepcopy(self.class_counts)
        node.impurity = self.impurity
        node.majority_class = self.majority_class
        
        node.children = []
        for child_id in self.child_ids:
            if child_id in all_nodes:
                child = all_nodes[child_id]
                child.parent = node
                node.children.append(child)


class SplitTransaction:
    """Manages atomic split operations with rollback capability"""
    
    def __init__(self, model, validator: Optional[SplitValidator] = None):
        self.model = model
        self.validator = validator or SplitValidator()
        self.transaction_id = str(uuid.uuid4())
        self.snapshots: Dict[str, NodeSnapshot] = {}
        self.created_nodes: List[str] = []
        self.deleted_nodes: List[str] = []
        self.rollback_callbacks: List[Callable] = []
        self.is_active = False
        self.is_committed = False
        self.start_time = None
        
        logger.debug(f"Created split transaction {self.transaction_id}")
        
    def __enter__(self):
        """Start the transaction"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the transaction - commit or rollback based on success"""
        if exc_type is not None:
            logger.error(f"Exception in transaction {self.transaction_id}: {exc_val}")
            self.rollback()
            return False  # Don't suppress the exception
        else:
            self.commit()
            return True
            
    def start(self):
        """Start the transaction"""
        if self.is_active:
            raise RuntimeError("Transaction is already active")
            
        self.is_active = True
        self.start_time = datetime.now()
        logger.info(f"Started split transaction {self.transaction_id}")
        
    def backup_node(self, node: TreeNode):
        """Create a backup snapshot of a node"""
        if not self.is_active:
            raise RuntimeError("Transaction is not active")
            
        if node.node_id not in self.snapshots:
            self.snapshots[node.node_id] = NodeSnapshot(node)
            logger.debug(f"Backed up node {node.node_id}")

    def backup_node_state(self, node: TreeNode):
        """Alias for backup_node for compatibility"""
        self.backup_node(node)
        
    def backup_tree_branch(self, root: TreeNode):
        """Backup an entire branch of the tree"""
        def backup_recursive(node):
            self.backup_node(node)
            for child in node.children:
                backup_recursive(child)
                
        backup_recursive(root)
        
    def record_node_creation(self, node_id: str):
        """Record that a node was created in this transaction"""
        if not self.is_active:
            raise RuntimeError("Transaction is not active")
            
        self.created_nodes.append(node_id)
        logger.debug(f"Recorded node creation: {node_id}")

    def record_split_creation(self, parent_node: TreeNode, child_nodes: List[TreeNode]):
        """Record split creation with parent and child nodes"""
        if not self.is_active:
            raise RuntimeError("Transaction is not active")
            
        self.backup_node(parent_node)
        
        for child in child_nodes:
            self.record_node_creation(child.node_id)
            
        logger.debug(f"Recorded split creation on {parent_node.node_id} with {len(child_nodes)} children")
        
    def record_node_deletion(self, node_id: str):
        """Record that a node was deleted in this transaction"""
        if not self.is_active:
            raise RuntimeError("Transaction is not active")
            
        self.deleted_nodes.append(node_id)
        logger.debug(f"Recorded node deletion: {node_id}")
        
    def add_rollback_callback(self, callback: Callable):
        """Add a callback to be executed during rollback"""
        self.rollback_callbacks.append(callback)
        
    def validate_transaction(self) -> ValidationResult:
        """Validate the current state of the transaction"""
        result = ValidationResult(is_valid=True)
        
        if not self.is_active:
            result.add_error("Transaction is not active")
            return result
            
        if hasattr(self.model, 'root') and self.model.root:
            tree_result = self.validator.validate_tree_consistency(self.model.root)
            result.errors.extend(tree_result.errors)
            result.warnings.extend(tree_result.warnings)
            if not tree_result.is_valid:
                result.is_valid = False
                
        return result
        
    def commit(self):
        """Commit the transaction"""
        if not self.is_active:
            raise RuntimeError("Transaction is not active")
            
        if self.is_committed:
            raise RuntimeError("Transaction is already committed")
            
        try:
            validation_result = self.validate_transaction()
            if not validation_result.is_valid:
                logger.error(f"Transaction validation failed: {validation_result.errors}")
                self.rollback()
                raise RuntimeError(f"Transaction validation failed: {validation_result.errors}")
                
            self.snapshots.clear()
            self.created_nodes.clear()
            self.deleted_nodes.clear()
            self.rollback_callbacks.clear()
            
            self.is_committed = True
            self.is_active = False
            
            duration = datetime.now() - self.start_time
            logger.info(f"Committed transaction {self.transaction_id} after {duration.total_seconds():.3f}s")
            
        except Exception as e:
            logger.error(f"Error committing transaction {self.transaction_id}: {e}")
            self.rollback()
            raise
            
    def rollback(self):
        """Rollback the transaction"""
        if not self.is_active:
            logger.warning(f"Attempted to rollback inactive transaction {self.transaction_id}")
            return
            
        try:
            logger.info(f"Rolling back transaction {self.transaction_id}")
            
            for callback in reversed(self.rollback_callbacks):
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in rollback callback: {e}")
                    
            all_nodes = {}
            if hasattr(self.model, 'root') and self.model.root:
                def collect_nodes(node):
                    all_nodes[node.node_id] = node
                    for child in node.children:
                        collect_nodes(child)
                collect_nodes(self.model.root)
                
            for node_id in self.created_nodes:
                if node_id in all_nodes:
                    node = all_nodes[node_id]
                    if node.parent:
                        node.parent.children = [c for c in node.parent.children if c.node_id != node_id]
                    if hasattr(self.model, 'nodes') and node_id in self.model.nodes:
                        del self.model.nodes[node_id]
                    logger.debug(f"Removed created node: {node_id}")
                    
            for node_id, snapshot in self.snapshots.items():
                if node_id in all_nodes:
                    snapshot.restore_to_node(all_nodes[node_id], all_nodes)
                    logger.debug(f"Restored node: {node_id}")
                    
            if hasattr(self.model, 'num_nodes'):
                self.model.num_nodes -= len(self.created_nodes)
            if hasattr(self.model, 'num_leaves'):
                if hasattr(self.model, 'root') and self.model.root:
                    leaf_count = 0
                    def count_leaves(node):
                        nonlocal leaf_count
                        if node.is_terminal:
                            leaf_count += 1
                        else:
                            for child in node.children:
                                count_leaves(child)
                    count_leaves(self.model.root)
                    self.model.num_leaves = leaf_count
                    
            self.is_active = False
            duration = datetime.now() - self.start_time
            logger.info(f"Rolled back transaction {self.transaction_id} after {duration.total_seconds():.3f}s")
            
        except Exception as e:
            logger.error(f"Error during rollback of transaction {self.transaction_id}: {e}")
            self.is_active = False  # Mark as inactive even if rollback failed
            

@contextmanager
def atomic_split_operation(model, validator: Optional[SplitValidator] = None):
    """Context manager for atomic split operations"""
    transaction = SplitTransaction(model, validator)
    try:
        transaction.start()
        yield transaction
        transaction.commit()
    except Exception as e:
        transaction.rollback()
        raise


def safe_apply_split(model, node_id: str, split_config: SplitConfiguration,
                    validator: Optional[SplitValidator] = None) -> bool:
    """Safely apply a split with automatic rollback on failure"""
    try:
        with atomic_split_operation(model, validator) as transaction:
            node = model.get_node_by_id(node_id)
            if not node:
                logger.error(f"Node {node_id} not found")
                return False
                
            transaction.backup_node(node)
            
            validation_result = transaction.validator.validate_split_config(
                split_config, 
                getattr(model, '_cached_X', None)
            )
            
            if not validation_result.is_valid:
                logger.error(f"Split validation failed: {validation_result.errors}")
                return False
                
            success = model.apply_manual_split(node_id, split_config.to_dict())
            
            if success:
                for child in node.children:
                    transaction.record_node_creation(child.node_id)
                    
                logger.info(f"Successfully applied split to node {node_id}")
                return True
            else:
                logger.error(f"Failed to apply split to node {node_id}")
                return False
                
    except Exception as e:
        logger.error(f"Exception applying split to node {node_id}: {e}")
        return False