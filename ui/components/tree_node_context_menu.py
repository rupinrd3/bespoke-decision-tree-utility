#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tree Node Context Menu
Enhanced context menu for decision tree nodes with modern styling

"""

import logging
from typing import Dict, Optional, Callable
from PyQt5.QtWidgets import QMenu, QAction
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtGui import QKeySequence

logger = logging.getLogger(__name__)

class TreeNodeContextMenu(QMenu):
    """Enhanced context menu for tree nodes with modern styling and comprehensive actions"""
    
    viewStatisticsRequested = pyqtSignal(str)  # node_id
    editSplitRequested = pyqtSignal(str)  # node_id
    findOptimalSplitRequested = pyqtSignal(str)  # node_id
    manualSplitRequested = pyqtSignal(str)  # node_id
    pruneSubtreeRequested = pyqtSignal(str)  # node_id
    copyNodeInfoRequested = pyqtSignal(str)  # node_id
    pasteNodeInfoRequested = pyqtSignal(str)  # node_id
    expandSubtreeRequested = pyqtSignal(str)  # node_id
    collapseSubtreeRequested = pyqtSignal(str)  # node_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_node_id = None
        self.current_node_type = None
        self.current_node_state = None
        
        self.setup_styling()
        logger.debug("TreeNodeContextMenu initialized")
        
    def setup_styling(self):
        """Apply modern styling to context menu"""
        self.setStyleSheet("""
            QMenu {
                background-color: #ffffff;
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                padding: 4px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
            }
            QMenu::item {
                background-color: transparent;
                padding: 8px 12px 8px 32px;
                margin: 1px;
                border-radius: 4px;
                min-width: 140px;
            }
            QMenu::item:selected {
                background-color: #f1f5f9;
                color: #1e293b;
            }
            QMenu::item:disabled {
                color: #94a3b8;
            }
            QMenu::separator {
                height: 1px;
                background-color: #e2e8f0;
                margin: 4px 8px;
            }
            QMenu::icon {
                left: 8px;
                width: 16px;
                height: 16px;
            }
        """)
        
    def create_node_menu(self, node_id: str, node_type: str, node_state: Dict):
        """Create context menu based on node properties"""
        self.current_node_id = node_id
        self.current_node_type = node_type
        self.current_node_state = node_state
        
        self.clear()
        
        statistics_action = QAction("📊 View Node Statistics", self)
        statistics_action.setShortcut(QKeySequence("Ctrl+I"))
        statistics_action.triggered.connect(lambda: self.viewStatisticsRequested.emit(node_id))
        self.addAction(statistics_action)
        
        self.addSeparator()
        
        if node_type == "internal":
            self._add_internal_node_actions(node_id, node_state)
        elif node_type == "terminal" or node_type == "leaf":
            self._add_terminal_node_actions(node_id, node_state)
        elif node_type == "root":
            self._add_root_node_actions(node_id, node_state)
            
        self.addSeparator()
        
        self._add_navigation_actions(node_id, node_state)
        
        self.addSeparator()
        
        self._add_utility_actions(node_id, node_state)
        
        if node_type != "root":
            self.addSeparator()
            self._add_destructive_actions(node_id, node_state)
            
    def _add_internal_node_actions(self, node_id: str, node_state: Dict):
        """Add actions specific to internal nodes"""
        
        find_split_action = QAction("🔍 Find Optimal Split", self)
        find_split_action.setShortcut(QKeySequence("Ctrl+F"))
        find_split_action.setStatusTip("Find optimal alternative split for this node")
        find_split_action.triggered.connect(lambda: self.findOptimalSplitRequested.emit(node_id))
        self.addAction(find_split_action)
        
        edit_action = QAction("✂️ Edit Split", self)
        edit_action.setShortcut(QKeySequence("Ctrl+E"))
        edit_action.setStatusTip("Edit the split condition for this node")
        edit_action.triggered.connect(lambda: self.editSplitRequested.emit(node_id))
        self.addAction(edit_action)
        
        manual_split_action = QAction("🎯 Manual Split", self)
        manual_split_action.setShortcut(QKeySequence("Ctrl+M"))
        manual_split_action.setStatusTip("Manually configure split for this node")
        manual_split_action.triggered.connect(lambda: self.manualSplitRequested.emit(node_id))
        self.addAction(manual_split_action)
        
        copy_action = QAction("📋 Copy Node Info", self)
        copy_action.setShortcut(QKeySequence("Ctrl+C"))
        copy_action.setStatusTip("Copy node information to clipboard")
        copy_action.triggered.connect(lambda: self.copyNodeInfoRequested.emit(node_id))
        self.addAction(copy_action)
        
        paste_action = QAction("📋 Paste Node Structure", self)
        paste_action.setShortcut(QKeySequence("Ctrl+V"))
        paste_action.setStatusTip("Paste node structure from clipboard (replaces current subtree)")
        paste_action.triggered.connect(lambda: self.pasteNodeInfoRequested.emit(node_id))
        self.addAction(paste_action)
        
        prune_action = QAction("✂️ Prune Subtree", self)
        prune_action.setStatusTip("Remove all child nodes (convert to leaf)")
        prune_action.triggered.connect(lambda: self.pruneSubtreeRequested.emit(node_id))
        self.addAction(prune_action)
        
        
    def _add_terminal_node_actions(self, node_id: str, node_state: Dict):
        """Add actions specific to terminal/leaf nodes"""
        can_split = node_state.get('can_split', True)
        sample_count = node_state.get('sample_count', 0)
        
        find_split_action = QAction("🔍 Find Optimal Split", self)
        find_split_action.setShortcut(QKeySequence("Ctrl+F"))
        find_split_action.setStatusTip("Find the best split for this leaf node")
        find_split_action.setEnabled(can_split and sample_count > 1)
        find_split_action.triggered.connect(lambda: self.findOptimalSplitRequested.emit(node_id))
        self.addAction(find_split_action)
        
        manual_split_action = QAction("🎯 Manual Split", self)
        manual_split_action.setShortcut(QKeySequence("Ctrl+M"))
        manual_split_action.setStatusTip("Manually configure split for this node")
        manual_split_action.setEnabled(can_split and sample_count > 1)
        manual_split_action.triggered.connect(lambda: self.manualSplitRequested.emit(node_id))
        self.addAction(manual_split_action)
        
        copy_action = QAction("📋 Copy Node Info", self)
        copy_action.setShortcut(QKeySequence("Ctrl+C"))
        copy_action.setStatusTip("Copy node information to clipboard")
        copy_action.triggered.connect(lambda: self.copyNodeInfoRequested.emit(node_id))
        self.addAction(copy_action)
        
        paste_action = QAction("📋 Paste Node Structure", self)
        paste_action.setShortcut(QKeySequence("Ctrl+V"))
        paste_action.setStatusTip("Paste node structure from clipboard (adds as children)")
        paste_action.triggered.connect(lambda: self.pasteNodeInfoRequested.emit(node_id))
        self.addAction(paste_action)
        
        if not can_split or sample_count <= 1:
            explanation_action = QAction("ℹ️ Cannot Split (Insufficient Data)", self)
            explanation_action.setEnabled(False)
            self.addAction(explanation_action)
            
    def _add_root_node_actions(self, node_id: str, node_state: Dict):
        """Add actions specific to root node"""
        is_terminal = node_state.get('is_terminal', True)
        
        if is_terminal:
            self._add_terminal_node_actions(node_id, node_state)
        else:
            self._add_internal_node_actions(node_id, node_state)
            
        pass
        
    def _add_navigation_actions(self, node_id: str, node_state: Dict):
        """Add tree navigation actions"""
        has_children = node_state.get('has_children', False)
        is_expanded = node_state.get('is_expanded', True)
        
        if has_children:
            if is_expanded:
                collapse_action = QAction("🔽 Collapse Subtree", self)
                collapse_action.setStatusTip("Collapse this node's subtree")
                collapse_action.triggered.connect(lambda: self.collapseSubtreeRequested.emit(node_id))
                self.addAction(collapse_action)
            else:
                expand_action = QAction("🔼 Expand Subtree", self)
                expand_action.setStatusTip("Expand this node's subtree")
                expand_action.triggered.connect(lambda: self.expandSubtreeRequested.emit(node_id))
                self.addAction(expand_action)
                
        
    def _add_utility_actions(self, node_id: str, node_state: Dict):
        """Add utility actions - now empty since Copy Node Info moved to main actions"""
        pass
        
    def _add_destructive_actions(self, node_id: str, node_state: Dict):
        """Add destructive actions with appropriate styling"""
        
    def show_for_node(self, node_id: str, node_type: str, node_state: Dict, position=None):
        """Show context menu for a specific node"""
        self.create_node_menu(node_id, node_type, node_state)
        
        if position is not None:
            if hasattr(position, 'toPoint'):
                global_pos = position.toPoint()
            else:
                global_pos = position
        else:
            from PyQt5.QtGui import QCursor
            global_pos = QCursor.pos()
            
        self.exec_(global_pos)
        
        logger.debug(f"Context menu shown for node {node_id} of type {node_type} at position {global_pos}")
        
    def show_at_cursor(self, node_id: str, node_type: str, node_state: Dict):
        """Show context menu at current mouse cursor position"""
        self.show_for_node(node_id, node_type, node_state, position=None)
        
    def get_available_actions(self, node_type: str, node_state: Dict) -> Dict[str, bool]:
        """Get available actions for a node type and state"""
        actions = {
            'view_statistics': True,
            'edit_split': False,
            'find_split': False,
            'manual_split': False,
            'prune_subtree': False,
            'expand_subtree': False,
            'collapse_subtree': False,
            'copy_info': True
        }
        
        if node_type == "internal":
            actions.update({
                'edit_split': True,
                'find_split': True,
                'manual_split': True,
                'prune_subtree': True
            })
        elif node_type in ["terminal", "leaf"]:
            can_split = node_state.get('can_split', True)
            sample_count = node_state.get('sample_count', 0)
            actions.update({
                'find_split': can_split and sample_count > 1,
                'manual_split': can_split and sample_count > 1
            })
        elif node_type == "root":
            pass  # Root node specific handling if needed
            
        has_children = node_state.get('has_children', False)
        is_expanded = node_state.get('is_expanded', True)
        
        if has_children:
            actions['expand_subtree'] = not is_expanded
            actions['collapse_subtree'] = is_expanded
            
        return actions