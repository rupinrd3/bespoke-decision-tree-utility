#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tree Navigation Dialog for Bespoke Utility
Advanced navigation tools for large decision trees
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np
import re

from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, QStringListModel
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette, QKeySequence
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QTabWidget, QWidget, QListWidget, QListWidgetItem,
    QSplitter, QProgressBar, QMessageBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QFrame, QSizePolicy, QTreeWidget, QTreeWidgetItem, QShortcut,
    QCompleter
)

from models.node import TreeNode

logger = logging.getLogger(__name__)


class TreeSearchEngine:
    """Search engine for decision tree nodes"""
    
    def __init__(self, root_node: TreeNode):
        self.root_node = root_node
        self.all_nodes = []
        self.node_index = {}
        self._build_index()
        
    def _build_index(self):
        """Build searchable index of all nodes"""
        self.all_nodes.clear()
        self.node_index.clear()
        
        self._index_node_recursive(self.root_node)
        
    def _index_node_recursive(self, node: TreeNode):
        """Recursively index nodes"""
        self.all_nodes.append(node)
        
        search_text = []
        search_text.append(f"id:{node.node_id}")
        search_text.append(f"depth:{node.depth}")
        search_text.append(f"samples:{node.samples}")
        
        if node.is_terminal:
            search_text.append("type:leaf")
            if node.majority_class is not None:
                search_text.append(f"prediction:{node.majority_class}")
        else:
            search_text.append("type:internal")
            if node.split_feature:
                search_text.append(f"feature:{node.split_feature}")
            if node.split_value is not None:
                search_text.append(f"value:{node.split_value}")
            if hasattr(node, 'split_operator'):
                search_text.append(f"operator:{node.split_operator}")
                
        if hasattr(node, 'impurity') and node.impurity is not None:
            search_text.append(f"impurity:{node.impurity:.4f}")
            
        self.node_index[node.node_id] = {
            'node': node,
            'search_text': ' '.join(search_text).lower(),
            'path': self._get_node_path(node)
        }
        
        for child in node.children:
            self._index_node_recursive(child)
            
    def _get_node_path(self, target_node: TreeNode) -> List[str]:
        """Get path from root to target node"""
        path = []
        self._find_path_recursive(self.root_node, target_node, path)
        return path
        
    def _find_path_recursive(self, current_node: TreeNode, target_node: TreeNode, path: List[str]) -> bool:
        """Recursively find path to target node"""
        if current_node.node_id == target_node.node_id:
            path.append(current_node.node_id)
            return True
            
        path.append(current_node.node_id)
        
        for child in current_node.children:
            if self._find_path_recursive(child, target_node, path):
                return True
                
        path.pop()
        return False
        
    def search(self, query: str, search_type: str = "text") -> List[Dict[str, Any]]:
        """
        Search nodes based on query
        
        Args:
            query: Search query
            search_type: Type of search ("text", "regex", "advanced")
            
        Returns:
            List of matching node information
        """
        if not query.strip():
            return []
            
        results = []
        query_lower = query.lower().strip()
        
        if search_type == "text":
            results = self._search_text(query_lower)
        elif search_type == "regex":
            results = self._search_regex(query)
        elif search_type == "advanced":
            results = self._search_advanced(query_lower)
        elif search_type == "xpath":
            results = self._search_xpath(query)
            
        return results
        
    def _search_text(self, query: str) -> List[Dict[str, Any]]:
        """Simple text search"""
        results = []
        
        for node_id, node_info in self.node_index.items():
            if query in node_info['search_text']:
                results.append({
                    'node': node_info['node'],
                    'node_id': node_id,
                    'match_reason': 'Text match',
                    'path': node_info['path'],
                    'relevance': self._calculate_relevance(query, node_info['search_text'])
                })
                
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results
        
    def _search_regex(self, pattern: str) -> List[Dict[str, Any]]:
        """Regex pattern search"""
        results = []
        
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            
            for node_id, node_info in self.node_index.items():
                if regex.search(node_info['search_text']):
                    results.append({
                        'node': node_info['node'],
                        'node_id': node_id,
                        'match_reason': f'Regex match: {pattern}',
                        'path': node_info['path'],
                        'relevance': 1.0
                    })
                    
        except re.error as e:
            logger.warning(f"Invalid regex pattern: {pattern}, error: {e}")
            
        return results
        
    def _search_advanced(self, query: str) -> List[Dict[str, Any]]:
        """Advanced search with field-specific queries"""
        results = []
        
        conditions = self._parse_advanced_query(query)
        
        if not conditions:
            return self._search_text(query)
            
        for node_id, node_info in self.node_index.items():
            node = node_info['node']
            
            if self._evaluate_conditions(node, conditions):
                results.append({
                    'node': node,
                    'node_id': node_id,
                    'match_reason': f'Advanced query: {query}',
                    'path': node_info['path'],
                    'relevance': 1.0
                })
                
        return results
        
    def _search_xpath(self, xpath: str) -> List[Dict[str, Any]]:
        """XPath-like search for tree structure"""
        results = []
        
        
        if xpath == "//leaf":
            for node_id, node_info in self.node_index.items():
                if node_info['node'].is_terminal:
                    results.append({
                        'node': node_info['node'],
                        'node_id': node_id,
                        'match_reason': 'XPath: all leaf nodes',
                        'path': node_info['path'],
                        'relevance': 1.0
                    })
                    
        elif xpath == "//internal":
            for node_id, node_info in self.node_index.items():
                if not node_info['node'].is_terminal:
                    results.append({
                        'node': node_info['node'],
                        'node_id': node_id,
                        'match_reason': 'XPath: all internal nodes',
                        'path': node_info['path'],
                        'relevance': 1.0
                    })
                    
        
        return results
        
    def _parse_advanced_query(self, query: str) -> List[Dict[str, Any]]:
        """Parse advanced query into conditions"""
        conditions = []
        
        parts = query.split(' and ')
        
        for part in parts:
            part = part.strip()
            if ':' in part:
                field, value = part.split(':', 1)
                field = field.strip()
                value = value.strip()
                
                operator = '='
                if value.startswith('>='):
                    operator = '>='
                    value = value[2:].strip()
                elif value.startswith('<='):
                    operator = '<='
                    value = value[2:].strip()
                elif value.startswith('>'):
                    operator = '>'
                    value = value[1:].strip()
                elif value.startswith('<'):
                    operator = '<'
                    value = value[1:].strip()
                elif value.startswith('!='):
                    operator = '!='
                    value = value[2:].strip()
                    
                conditions.append({
                    'field': field,
                    'operator': operator,
                    'value': value
                })
                
        return conditions
        
    def _evaluate_conditions(self, node: TreeNode, conditions: List[Dict[str, Any]]) -> bool:
        """Evaluate if node matches all conditions"""
        for condition in conditions:
            if not self._evaluate_single_condition(node, condition):
                return False
        return True
        
    def _evaluate_single_condition(self, node: TreeNode, condition: Dict[str, Any]) -> bool:
        """Evaluate single condition against node"""
        field = condition['field']
        operator = condition['operator']
        value = condition['value']
        
        node_value = None
        
        if field == 'id':
            node_value = node.node_id
        elif field == 'depth':
            node_value = node.depth
        elif field == 'samples':
            node_value = node.samples
        elif field == 'type':
            node_value = 'leaf' if node.is_terminal else 'internal'
        elif field == 'feature':
            node_value = node.split_feature
        elif field == 'value':
            node_value = node.split_value
        elif field == 'prediction':
            node_value = node.prediction
        elif field == 'impurity':
            node_value = getattr(node, 'impurity', None)
        else:
            return False
            
        if node_value is None:
            return False
            
        try:
            if isinstance(node_value, (int, float)) and value.replace('.', '').replace('-', '').isdigit():
                value = float(value)
            elif isinstance(node_value, str):
                value = str(value)
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Error converting value for comparison: {e}")
            pass
            
        if operator == '=':
            return str(node_value).lower() == str(value).lower()
        elif operator == '!=':
            return str(node_value).lower() != str(value).lower()
        elif operator == '>':
            return node_value > value
        elif operator == '<':
            return node_value < value
        elif operator == '>=':
            return node_value >= value
        elif operator == '<=':
            return node_value <= value
            
        return False
        
    def _calculate_relevance(self, query: str, text: str) -> float:
        """Calculate search relevance score"""
        if not query or not text:
            return 0.0
            
        score = 0.0
        
        if query in text:
            score += 1.0
            
        query_words = query.split()
        for word in query_words:
            if word in text:
                score += 0.5
                
        score = score / (len(text) / 100.0 + 1.0)
        
        return score
        
    def get_node_by_id(self, node_id: str) -> Optional[TreeNode]:
        """Get node by ID"""
        if node_id in self.node_index:
            return self.node_index[node_id]['node']
        return None
        
    def get_node_path(self, node_id: str) -> List[str]:
        """Get path to node"""
        if node_id in self.node_index:
            return self.node_index[node_id]['path']
        return []
        
    def get_all_node_ids(self) -> List[str]:
        """Get all node IDs for autocomplete"""
        return list(self.node_index.keys())
        
    def get_tree_statistics(self) -> Dict[str, Any]:
        """Get tree statistics"""
        total_nodes = len(self.all_nodes)
        leaf_nodes = sum(1 for node in self.all_nodes if node.is_terminal)
        internal_nodes = total_nodes - leaf_nodes
        
        max_depth = max(node.depth for node in self.all_nodes) if self.all_nodes else 0
        
        features_used = {}
        for node in self.all_nodes:
            if not node.is_terminal and node.split_feature:
                features_used[node.split_feature] = features_used.get(node.split_feature, 0) + 1
                
        return {
            'total_nodes': total_nodes,
            'leaf_nodes': leaf_nodes,
            'internal_nodes': internal_nodes,
            'max_depth': max_depth,
            'features_used': features_used
        }


class TreeNavigationDialog(QDialog):
    """Dialog for advanced tree navigation and search"""
    
    nodeSelected = pyqtSignal(str)  # node_id
    pathRequested = pyqtSignal(list)  # path as list of node_ids
    splitRequested = pyqtSignal(str)  # node_id for find split
    editRequested = pyqtSignal(str)  # node_id for edit split
    
    def __init__(self, tree_model, parent=None):
        super().__init__(parent)
        
        self.tree_model = tree_model
        self.search_engine = None
        self.current_results = []
        self.history = []
        self.bookmarks = {}
        
        if tree_model and tree_model.root:
            self.search_engine = TreeSearchEngine(tree_model.root)
            
        self.setWindowTitle("Tree Navigation")
        self.setModal(False)
        self.resize(900, 700)
        
        self.setupUI()
        self.setupShortcuts()
        self.updateTreeStatistics()
        
    def setupUI(self):
        """Setup the main UI"""
        layout = QVBoxLayout()
        
        header_label = QLabel("Tree Navigation & Search")
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header_label)
        
        self.tab_widget = QTabWidget()
        
        search_tab = self.createSearchTab()
        self.tab_widget.addTab(search_tab, "Search")
        
        browse_tab = self.createBrowseTab()
        self.tab_widget.addTab(browse_tab, "Browse")
        
        bookmarks_tab = self.createBookmarksTab()
        self.tab_widget.addTab(bookmarks_tab, "Bookmarks")
        
        split_mgmt_tab = self.createSplitManagementTab()
        self.tab_widget.addTab(split_mgmt_tab, "Split Management")
        
        stats_tab = self.createStatisticsTab()
        self.tab_widget.addTab(stats_tab, "Statistics")
        
        layout.addWidget(self.tab_widget)
        
        nav_bar = self.createNavigationBar()
        layout.addWidget(nav_bar)
        
        button_layout = QHBoxLayout()
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def createSearchTab(self) -> QWidget:
        """Create the search tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        search_group = QGroupBox("Search")
        search_layout = QVBoxLayout()
        
        input_layout = QHBoxLayout()
        
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search nodes... (e.g., 'feature:age', 'depth:>3', 'type:leaf')")
        self.search_edit.returnPressed.connect(self.performSearch)
        
        if self.search_engine:
            completer_model = QStringListModel()
            
            autocomplete_items = self.search_engine.get_all_node_ids()
            autocomplete_items.extend([
                'type:leaf', 'type:internal', 'depth:', 'samples:', 'feature:', 
                'value:', 'prediction:', 'impurity:', 'id:'
            ])
            
            completer_model.setStringList(autocomplete_items)
            completer = QCompleter(completer_model)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            self.search_edit.setCompleter(completer)
        
        input_layout.addWidget(self.search_edit)
        
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.performSearch)
        input_layout.addWidget(self.search_button)
        
        search_layout.addLayout(input_layout)
        
        options_layout = QHBoxLayout()
        
        self.search_type_combo = QComboBox()
        self.search_type_combo.addItems(['Text', 'Regex', 'Advanced', 'XPath'])
        options_layout.addWidget(QLabel("Type:"))
        options_layout.addWidget(self.search_type_combo)
        
        self.case_sensitive_checkbox = QCheckBox("Case Sensitive")
        options_layout.addWidget(self.case_sensitive_checkbox)
        
        options_layout.addStretch()
        search_layout.addLayout(options_layout)
        
        quick_layout = QHBoxLayout()
        
        self.all_leaves_button = QPushButton("All Leaves")
        self.all_leaves_button.clicked.connect(lambda: self.quickSearch("//leaf"))
        
        self.deep_nodes_button = QPushButton("Deep Nodes (>5)")
        self.deep_nodes_button.clicked.connect(lambda: self.quickSearch("depth:>5"))
        
        self.small_samples_button = QPushButton("Small Samples (<10)")
        self.small_samples_button.clicked.connect(lambda: self.quickSearch("samples:<10"))
        
        quick_layout.addWidget(self.all_leaves_button)
        quick_layout.addWidget(self.deep_nodes_button)
        quick_layout.addWidget(self.small_samples_button)
        quick_layout.addStretch()
        
        search_layout.addLayout(quick_layout)
        
        search_group.setLayout(search_layout)
        layout.addWidget(search_group)
        
        results_group = QGroupBox("Search Results")
        results_layout = QVBoxLayout()
        
        self.results_info_label = QLabel("No search performed")
        results_layout.addWidget(self.results_info_label)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            'Node ID', 'Type', 'Depth', 'Samples', 'Feature/Prediction', 'Actions'
        ])
        
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        results_layout.addWidget(self.results_table)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        widget.setLayout(layout)
        return widget
        
    def createBrowseTab(self) -> QWidget:
        """Create the browse tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        browser_group = QGroupBox("Tree Browser")
        browser_layout = QVBoxLayout()
        
        self.tree_browser = QTreeWidget()
        self.tree_browser.setHeaderLabel("Tree Structure")
        self.tree_browser.itemClicked.connect(self.onBrowserItemClicked)
        self.tree_browser.itemDoubleClicked.connect(self.onBrowserItemDoubleClicked)
        browser_layout.addWidget(self.tree_browser)
        
        controls_layout = QHBoxLayout()
        
        self.expand_all_button = QPushButton("Expand All")
        self.expand_all_button.clicked.connect(self.tree_browser.expandAll)
        
        self.collapse_all_button = QPushButton("Collapse All")
        self.collapse_all_button.clicked.connect(self.tree_browser.collapseAll)
        
        self.refresh_browser_button = QPushButton("Refresh")
        self.refresh_browser_button.clicked.connect(self.refreshBrowser)
        
        controls_layout.addWidget(self.expand_all_button)
        controls_layout.addWidget(self.collapse_all_button)
        controls_layout.addWidget(self.refresh_browser_button)
        controls_layout.addStretch()
        
        browser_layout.addLayout(controls_layout)
        
        browser_group.setLayout(browser_layout)
        layout.addWidget(browser_group)
        
        details_group = QGroupBox("Node Details")
        details_layout = QFormLayout()
        
        self.selected_node_id_label = QLabel("None")
        self.selected_node_type_label = QLabel("None")
        self.selected_node_depth_label = QLabel("None")
        self.selected_node_samples_label = QLabel("None")
        self.selected_node_info_label = QLabel("None")
        
        details_layout.addRow("Node ID:", self.selected_node_id_label)
        details_layout.addRow("Type:", self.selected_node_type_label)
        details_layout.addRow("Depth:", self.selected_node_depth_label)
        details_layout.addRow("Samples:", self.selected_node_samples_label)
        details_layout.addRow("Details:", self.selected_node_info_label)
        
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)
        
        widget.setLayout(layout)
        return widget
        
    def createBookmarksTab(self) -> QWidget:
        """Create the bookmarks tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        bookmarks_group = QGroupBox("Bookmarks")
        bookmarks_layout = QVBoxLayout()
        
        self.bookmarks_list = QListWidget()
        self.bookmarks_list.itemDoubleClicked.connect(self.goToBookmark)
        bookmarks_layout.addWidget(self.bookmarks_list)
        
        controls_layout = QHBoxLayout()
        
        self.add_bookmark_button = QPushButton("Add Current")
        self.add_bookmark_button.clicked.connect(self.addBookmark)
        
        self.remove_bookmark_button = QPushButton("Remove")
        self.remove_bookmark_button.clicked.connect(self.removeBookmark)
        
        self.clear_bookmarks_button = QPushButton("Clear All")
        self.clear_bookmarks_button.clicked.connect(self.clearBookmarks)
        
        controls_layout.addWidget(self.add_bookmark_button)
        controls_layout.addWidget(self.remove_bookmark_button)
        controls_layout.addWidget(self.clear_bookmarks_button)
        controls_layout.addStretch()
        
        bookmarks_layout.addLayout(controls_layout)
        
        bookmarks_group.setLayout(bookmarks_layout)
        layout.addWidget(bookmarks_group)
        
        history_group = QGroupBox("Navigation History")
        history_layout = QVBoxLayout()
        
        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(self.goToHistoryItem)
        history_layout.addWidget(self.history_list)
        
        history_controls = QHBoxLayout()
        
        self.clear_history_button = QPushButton("Clear History")
        self.clear_history_button.clicked.connect(self.clearHistory)
        
        history_controls.addWidget(self.clear_history_button)
        history_controls.addStretch()
        
        history_layout.addLayout(history_controls)
        
        history_group.setLayout(history_layout)
        layout.addWidget(history_group)
        
        widget.setLayout(layout)
        return widget
        
    def createStatisticsTab(self) -> QWidget:
        """Create the statistics tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        stats_group = QGroupBox("Tree Statistics")
        stats_layout = QFormLayout()
        
        self.total_nodes_label = QLabel("0")
        self.leaf_nodes_label = QLabel("0")
        self.internal_nodes_label = QLabel("0")
        self.max_depth_label = QLabel("0")
        self.avg_depth_label = QLabel("0")
        
        stats_layout.addRow("Total Nodes:", self.total_nodes_label)
        stats_layout.addRow("Leaf Nodes:", self.leaf_nodes_label)
        stats_layout.addRow("Internal Nodes:", self.internal_nodes_label)
        stats_layout.addRow("Maximum Depth:", self.max_depth_label)
        stats_layout.addRow("Average Depth:", self.avg_depth_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        features_group = QGroupBox("Feature Usage")
        features_layout = QVBoxLayout()
        
        self.features_table = QTableWidget()
        self.features_table.setColumnCount(2)
        self.features_table.setHorizontalHeaderLabels(['Feature', 'Usage Count'])
        
        header = self.features_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        
        features_layout.addWidget(self.features_table)
        features_group.setLayout(features_layout)
        layout.addWidget(features_group)
        
        widget.setLayout(layout)
        return widget
        
    def createNavigationBar(self) -> QWidget:
        """Create quick navigation bar"""
        nav_frame = QFrame()
        nav_frame.setFrameStyle(QFrame.StyledPanel)
        layout = QHBoxLayout()
        
        self.quick_nav_edit = QLineEdit()
        self.quick_nav_edit.setPlaceholderText("Enter node ID to jump to...")
        self.quick_nav_edit.returnPressed.connect(self.quickNavigate)
        
        self.go_button = QPushButton("Go")
        self.go_button.clicked.connect(self.quickNavigate)
        
        self.back_button = QPushButton("← Back")
        self.back_button.clicked.connect(self.goBack)
        self.back_button.setEnabled(False)
        
        self.forward_button = QPushButton("Forward →")
        self.forward_button.clicked.connect(self.goForward)
        self.forward_button.setEnabled(False)
        
        layout.addWidget(QLabel("Quick Go:"))
        layout.addWidget(self.quick_nav_edit)
        layout.addWidget(self.go_button)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.back_button)
        layout.addWidget(self.forward_button)
        layout.addStretch()
        
        nav_frame.setLayout(layout)
        return nav_frame
        
    def setupShortcuts(self):
        """Setup keyboard shortcuts"""
        search_shortcut = QShortcut(QKeySequence("Ctrl+F"), self)
        search_shortcut.activated.connect(self.focusSearch)
        
        nav_shortcut = QShortcut(QKeySequence("Ctrl+G"), self)
        nav_shortcut.activated.connect(self.focusQuickNav)
        
        back_shortcut = QShortcut(QKeySequence("Alt+Left"), self)
        back_shortcut.activated.connect(self.goBack)
        
        forward_shortcut = QShortcut(QKeySequence("Alt+Right"), self)
        forward_shortcut.activated.connect(self.goForward)
        
    def focusSearch(self):
        """Focus on search input"""
        self.tab_widget.setCurrentIndex(0)  # Search tab
        self.search_edit.setFocus()
        self.search_edit.selectAll()
        
    def focusQuickNav(self):
        """Focus on quick navigation input"""
        self.quick_nav_edit.setFocus()
        self.quick_nav_edit.selectAll()
        
    def performSearch(self):
        """Perform search based on current input"""
        if not self.search_engine:
            QMessageBox.warning(self, "No Tree", "No tree loaded for searching.")
            return
            
        query = self.search_edit.text().strip()
        if not query:
            return
            
        search_type = self.search_type_combo.currentText().lower()
        
        results = self.search_engine.search(query, search_type)
        self.current_results = results
        
        self.updateSearchResults(results, query)
        
    def quickSearch(self, query: str):
        """Perform quick search"""
        self.search_edit.setText(query)
        self.search_type_combo.setCurrentText("XPath" if query.startswith("//") else "Advanced")
        self.performSearch()
        
    def updateSearchResults(self, results: List[Dict[str, Any]], query: str):
        """Update search results display"""
        self.results_info_label.setText(f"Found {len(results)} results for: '{query}'")
        
        self.results_table.setRowCount(len(results))
        
        for i, result in enumerate(results):
            node = result['node']
            
            id_item = QTableWidgetItem(node.node_id)
            self.results_table.setItem(i, 0, id_item)
            
            type_item = QTableWidgetItem("Leaf" if node.is_terminal else "Internal")
            self.results_table.setItem(i, 1, type_item)
            
            depth_item = QTableWidgetItem(str(node.depth))
            self.results_table.setItem(i, 2, depth_item)
            
            samples_item = QTableWidgetItem(str(node.samples))
            self.results_table.setItem(i, 3, samples_item)
            
            if node.is_terminal:
                info_text = f"Prediction: {node.prediction}"
            else:
                info_text = f"Feature: {node.split_feature} {getattr(node, 'split_operator', '<=')} {node.split_value}"
            info_item = QTableWidgetItem(info_text)
            self.results_table.setItem(i, 4, info_item)
            
            actions_layout = QHBoxLayout()
            
            go_button = QPushButton("Go")
            go_button.clicked.connect(lambda checked, nid=node.node_id: self.navigateToNode(nid))
            
            bookmark_button = QPushButton("★")
            bookmark_button.setMaximumWidth(30)
            bookmark_button.clicked.connect(lambda checked, nid=node.node_id: self.addBookmarkForNode(nid))
            
            actions_widget = QWidget()
            actions_layout.addWidget(go_button)
            actions_layout.addWidget(bookmark_button)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            actions_widget.setLayout(actions_layout)
            
            self.results_table.setCellWidget(i, 5, actions_widget)
            
    def refreshBrowser(self):
        """Refresh the tree browser"""
        if not self.tree_model or not self.tree_model.root:
            return
            
        self.tree_browser.clear()
        
        root_item = self.createBrowserItem(self.tree_model.root)
        self.tree_browser.addTopLevelItem(root_item)
        self.tree_browser.expandToDepth(2)  # Expand first few levels
        
    def createBrowserItem(self, node: TreeNode) -> QTreeWidgetItem:
        """Create browser item for tree node"""
        if node.is_terminal:
            text = f"[{node.node_id}] Leaf: {node.prediction} ({node.samples} samples)"
        else:
            text = f"[{node.node_id}] {node.split_feature} {getattr(node, 'split_operator', '<=')} {node.split_value} ({node.samples} samples)"
            
        item = QTreeWidgetItem([text])
        item.setData(0, Qt.UserRole, node.node_id)
        
        for child in node.children:
            child_item = self.createBrowserItem(child)
            item.addChild(child_item)
            
        return item
        
    def onBrowserItemClicked(self, item: QTreeWidgetItem, column: int):
        """Handle browser item click"""
        node_id = item.data(0, Qt.UserRole)
        if node_id and self.search_engine:
            node = self.search_engine.get_node_by_id(node_id)
            if node:
                self.updateNodeDetails(node)
                
    def onBrowserItemDoubleClicked(self, item: QTreeWidgetItem, column: int):
        """Handle browser item double click"""
        node_id = item.data(0, Qt.UserRole)
        if node_id:
            self.navigateToNode(node_id)
            
    def updateNodeDetails(self, node: TreeNode):
        """Update node details display"""
        self.selected_node_id_label.setText(node.node_id)
        self.selected_node_type_label.setText("Leaf" if node.is_terminal else "Internal")
        self.selected_node_depth_label.setText(str(node.depth))
        self.selected_node_samples_label.setText(str(node.samples))
        
        if node.is_terminal:
            info_text = f"Prediction: {node.prediction}"
        else:
            info_text = f"Split: {node.split_feature} {getattr(node, 'split_operator', '<=')} {node.split_value}"
            
        if hasattr(node, 'impurity') and node.impurity is not None:
            info_text += f", Impurity: {node.impurity:.4f}"
            
        self.selected_node_info_label.setText(info_text)
        
    def navigateToNode(self, node_id: str):
        """Navigate to specific node"""
        if not self.search_engine:
            return
            
        node = self.search_engine.get_node_by_id(node_id)
        if node:
            self.addToHistory(node_id)
            
            self.updateNodeDetails(node)
            
            self.nodeSelected.emit(node_id)
            
            path = self.search_engine.get_node_path(node_id)
            if path:
                self.pathRequested.emit(path)
                
    def quickNavigate(self):
        """Quick navigation to node"""
        node_id = self.quick_nav_edit.text().strip()
        if node_id:
            self.navigateToNode(node_id)
            self.quick_nav_edit.clear()
            
    def addToHistory(self, node_id: str):
        """Add node to navigation history"""
        if node_id not in self.history:
            self.history.append(node_id)
            
            self.history_list.addItem(f"{node_id} - {self.getNodeDescription(node_id)}")
            
        self.back_button.setEnabled(len(self.history) > 1)
        
    def getNodeDescription(self, node_id: str) -> str:
        """Get short description of node"""
        if not self.search_engine:
            return "Unknown"
            
        node = self.search_engine.get_node_by_id(node_id)
        if not node:
            return "Unknown"
            
        if node.is_terminal:
            return f"Leaf: {node.prediction}"
        else:
            return f"Split: {node.split_feature}"
            
    def goBack(self):
        """Go back in navigation history"""
        if len(self.history) > 1:
            self.history.pop()
            if self.history:
                previous_node = self.history[-1]
                self.navigateToNode(previous_node)
                
    def goForward(self):
        """Go forward in navigation history"""
        pass
        
    def addBookmark(self):
        """Add current node to bookmarks"""
        current_node_id = self.selected_node_id_label.text()
        if current_node_id and current_node_id != "None":
            self.addBookmarkForNode(current_node_id)
            
    def addBookmarkForNode(self, node_id: str):
        """Add specific node to bookmarks"""
        if node_id not in self.bookmarks:
            description = self.getNodeDescription(node_id)
            self.bookmarks[node_id] = description
            
            self.bookmarks_list.addItem(f"{node_id} - {description}")
            
    def removeBookmark(self):
        """Remove selected bookmark"""
        current_item = self.bookmarks_list.currentItem()
        if current_item:
            text = current_item.text()
            node_id = text.split(" - ")[0]
            
            if node_id in self.bookmarks:
                del self.bookmarks[node_id]
                
            self.bookmarks_list.takeItem(self.bookmarks_list.row(current_item))
            
    def clearBookmarks(self):
        """Clear all bookmarks"""
        self.bookmarks.clear()
        self.bookmarks_list.clear()
        
    def goToBookmark(self, item: QListWidgetItem):
        """Go to bookmarked node"""
        text = item.text()
        node_id = text.split(" - ")[0]
        self.navigateToNode(node_id)
        
    def goToHistoryItem(self, item: QListWidgetItem):
        """Go to history item"""
        text = item.text()
        node_id = text.split(" - ")[0]
        self.navigateToNode(node_id)
        
    def clearHistory(self):
        """Clear navigation history"""
        self.history.clear()
        self.history_list.clear()
        self.back_button.setEnabled(False)
        
    def updateTreeStatistics(self):
        """Update tree statistics display"""
        if not self.search_engine:
            return
            
        stats = self.search_engine.get_tree_statistics()
        
        self.total_nodes_label.setText(str(stats['total_nodes']))
        self.leaf_nodes_label.setText(str(stats['leaf_nodes']))
        self.internal_nodes_label.setText(str(stats['internal_nodes']))
        self.max_depth_label.setText(str(stats['max_depth']))
        
        if self.search_engine.all_nodes:
            avg_depth = sum(node.depth for node in self.search_engine.all_nodes) / len(self.search_engine.all_nodes)
            self.avg_depth_label.setText(f"{avg_depth:.2f}")
        else:
            self.avg_depth_label.setText("0")
            
        features_used = stats['features_used']
        self.features_table.setRowCount(len(features_used))
        
        for i, (feature, count) in enumerate(sorted(features_used.items(), key=lambda x: x[1], reverse=True)):
            feature_item = QTableWidgetItem(feature)
            count_item = QTableWidgetItem(str(count))
            
            self.features_table.setItem(i, 0, feature_item)
            self.features_table.setItem(i, 1, count_item)
            
    def createSplitManagementTab(self):
        """Create the split management tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        nav_group = QGroupBox("Quick Navigation")
        nav_layout = QVBoxLayout()
        
        goto_layout = QHBoxLayout()
        goto_layout.addWidget(QLabel("Go to Node:"))
        
        self.goto_node_input = QLineEdit()
        self.goto_node_input.setPlaceholderText("Enter node ID or split condition")
        self.goto_node_input.returnPressed.connect(self.goToSpecificNode)
        goto_layout.addWidget(self.goto_node_input)
        
        goto_btn = QPushButton("Go")
        goto_btn.clicked.connect(self.goToSpecificNode)
        goto_layout.addWidget(goto_btn)
        
        nav_layout.addLayout(goto_layout)
        
        quick_nav_layout = QGridLayout()
        
        root_btn = QPushButton("Root Node")
        root_btn.clicked.connect(lambda: self.navigateToNode("root"))
        quick_nav_layout.addWidget(root_btn, 0, 0)
        
        deepest_btn = QPushButton("Deepest Node")
        deepest_btn.clicked.connect(self.goToDeepestNode)
        quick_nav_layout.addWidget(deepest_btn, 0, 1)
        
        largest_btn = QPushButton("Largest Node")
        largest_btn.clicked.connect(self.goToLargestNode)
        quick_nav_layout.addWidget(largest_btn, 1, 0)
        
        impure_btn = QPushButton("Most Impure")
        impure_btn.clicked.connect(self.goToMostImpureNode)
        quick_nav_layout.addWidget(impure_btn, 1, 1)
        
        nav_layout.addLayout(quick_nav_layout)
        nav_group.setLayout(nav_layout)
        layout.addWidget(nav_group)
        
        ops_group = QGroupBox("Split Operations")
        ops_layout = QVBoxLayout()
        
        selection_layout = QHBoxLayout()
        selection_layout.addWidget(QLabel("Selected Node:"))
        
        self.selected_node_label = QLabel("None")
        self.selected_node_label.setStyleSheet("QLabel { font-weight: bold; color: blue; }")
        selection_layout.addWidget(self.selected_node_label)
        selection_layout.addStretch()
        
        ops_layout.addLayout(selection_layout)
        
        ops_buttons_layout = QGridLayout()
        
        self.find_split_btn = QPushButton("Find Split")
        self.find_split_btn.setToolTip("Find optimal split for selected node")
        self.find_split_btn.clicked.connect(self.requestFindSplit)
        self.find_split_btn.setEnabled(False)
        ops_buttons_layout.addWidget(self.find_split_btn, 0, 0)
        
        self.edit_split_btn = QPushButton("Edit Split")
        self.edit_split_btn.setToolTip("Edit existing split for selected node")
        self.edit_split_btn.clicked.connect(self.requestEditSplit)
        self.edit_split_btn.setEnabled(False)
        ops_buttons_layout.addWidget(self.edit_split_btn, 0, 1)
        
        view_details_btn = QPushButton("View Details")
        view_details_btn.setToolTip("View detailed node information")
        view_details_btn.clicked.connect(self.viewNodeDetails)
        ops_buttons_layout.addWidget(view_details_btn, 1, 0)
        
        compare_btn = QPushButton("Compare Nodes")
        compare_btn.setToolTip("Compare selected nodes")
        compare_btn.clicked.connect(self.compareNodes)
        ops_buttons_layout.addWidget(compare_btn, 1, 1)
        
        ops_layout.addLayout(ops_buttons_layout)
        ops_group.setLayout(ops_layout)
        layout.addWidget(ops_group)
        
        splits_group = QGroupBox("All Splits in Tree")
        splits_layout = QVBoxLayout()
        
        self.splits_table = QTableWidget()
        self.splits_table.setColumnCount(6)
        self.splits_table.setHorizontalHeaderLabels([
            'Node ID', 'Feature', 'Condition', 'Samples', 'Impurity', 'Depth'
        ])
        
        header = self.splits_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents) 
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        
        self.splits_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.splits_table.setAlternatingRowColors(True)
        self.splits_table.setSortingEnabled(True)
        self.splits_table.itemSelectionChanged.connect(self.onSplitTableSelectionChanged)
        self.splits_table.itemDoubleClicked.connect(self.goToSplitFromTable)
        
        splits_layout.addWidget(self.splits_table)
        
        refresh_btn = QPushButton("Refresh Splits")
        refresh_btn.clicked.connect(self.refreshSplitsTable)
        splits_layout.addWidget(refresh_btn)
        
        splits_group.setLayout(splits_layout)
        layout.addWidget(splits_group)
        
        self.refreshSplitsTable()
        
        widget.setLayout(layout)
        return widget
        
    def goToSpecificNode(self):
        """Navigate to a specific node based on input"""
        query = self.goto_node_input.text().strip()
        if not query:
            return
            
        if self.search_engine:
            for node in self.search_engine.all_nodes:
                if node.node_id == query:
                    self.navigateToNode(query)
                    return
                    
        self.performSearch(query, "text")
        if self.current_results:
            node_id = self.current_results[0]['node_id']
            self.navigateToNode(node_id)
            
    def goToDeepestNode(self):
        """Navigate to the deepest node in the tree"""
        if not self.search_engine:
            return
            
        deepest_node = max(self.search_engine.all_nodes, key=lambda n: n.depth, default=None)
        if deepest_node:
            self.navigateToNode(deepest_node.node_id)
            
    def goToLargestNode(self):
        """Navigate to the node with the most samples"""
        if not self.search_engine:
            return
            
        largest_node = max(self.search_engine.all_nodes, key=lambda n: n.samples, default=None)
        if largest_node:
            self.navigateToNode(largest_node.node_id)
            
    def goToMostImpureNode(self):
        """Navigate to the node with highest impurity"""
        if not self.search_engine:
            return
            
        most_impure = max(
            (n for n in self.search_engine.all_nodes if hasattr(n, 'impurity') and n.impurity is not None),
            key=lambda n: n.impurity,
            default=None
        )
        if most_impure:
            self.navigateToNode(most_impure.node_id)
            
    def requestFindSplit(self):
        """Request to find split for selected node"""
        if hasattr(self, 'current_selected_node') and self.current_selected_node:
            self.splitRequested.emit(self.current_selected_node)
            
    def requestEditSplit(self):
        """Request to edit split for selected node"""
        if hasattr(self, 'current_selected_node') and self.current_selected_node:
            self.editRequested.emit(self.current_selected_node)
            
    def viewNodeDetails(self):
        """View detailed information about selected node"""
        if not hasattr(self, 'current_selected_node') or not self.current_selected_node:
            QMessageBox.information(self, "No Selection", "Please select a node first.")
            return
            
        node = None
        if self.search_engine:
            for n in self.search_engine.all_nodes:
                if n.node_id == self.current_selected_node:
                    node = n
                    break
                    
        if not node:
            return
            
        details = f"Node Details: {node.node_id}\n\n"
        details += f"Type: {'Leaf' if node.is_terminal else 'Internal'}\n"
        details += f"Depth: {node.depth}\n"
        details += f"Samples: {node.samples}\n"
        
        if hasattr(node, 'impurity') and node.impurity is not None:
            details += f"Impurity: {node.impurity:.4f}\n"
            
        if not node.is_terminal:
            if node.split_feature:
                details += f"Split Feature: {node.split_feature}\n"
            if node.split_value is not None:
                details += f"Split Value: {node.split_value}\n"
        else:
            if node.prediction is not None:
                details += f"Prediction: {node.prediction}\n"
                
        if hasattr(node, 'class_distribution') and node.class_distribution:
            details += f"\nClass Distribution:\n"
            for cls, count in node.class_distribution.items():
                details += f"  {cls}: {count}\n"
                
        QMessageBox.information(self, f"Node {node.node_id}", details)
        
    def compareNodes(self):
        """Compare multiple selected nodes"""
        QMessageBox.information(self, "Compare Nodes", "Node comparison feature coming soon!")
        
    def refreshSplitsTable(self):
        """Refresh the splits table with current tree data"""
        if not self.search_engine:
            self.splits_table.setRowCount(0)
            return
            
        split_nodes = [n for n in self.search_engine.all_nodes if not n.is_terminal]
        
        self.splits_table.setRowCount(len(split_nodes))
        
        for i, node in enumerate(split_nodes):
            self.splits_table.setItem(i, 0, QTableWidgetItem(node.node_id))
            
            feature = getattr(node, 'split_feature', 'Unknown')
            self.splits_table.setItem(i, 1, QTableWidgetItem(str(feature)))
            
            condition = "N/A"
            if hasattr(node, 'split_feature') and hasattr(node, 'split_value'):
                if node.split_feature and node.split_value is not None:
                    condition = f"{node.split_feature} <= {node.split_value}"
            self.splits_table.setItem(i, 2, QTableWidgetItem(condition))
            
            self.splits_table.setItem(i, 3, QTableWidgetItem(str(node.samples)))
            
            impurity = getattr(node, 'impurity', 0.0)
            self.splits_table.setItem(i, 4, QTableWidgetItem(f"{impurity:.4f}"))
            
            self.splits_table.setItem(i, 5, QTableWidgetItem(str(node.depth)))
            
    def onSplitTableSelectionChanged(self):
        """Handle selection change in splits table"""
        selected_items = self.splits_table.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            node_id = self.splits_table.item(row, 0).text()
            
            self.current_selected_node = node_id
            self.selected_node_label.setText(node_id)
            
            self.find_split_btn.setEnabled(True)
            
            if self.search_engine:
                for node in self.search_engine.all_nodes:
                    if node.node_id == node_id and not node.is_terminal:
                        self.edit_split_btn.setEnabled(
                            hasattr(node, 'split_feature') and node.split_feature is not None
                        )
                        break
        else:
            if hasattr(self, 'current_selected_node'):
                delattr(self, 'current_selected_node')
            self.selected_node_label.setText("None")
            self.find_split_btn.setEnabled(False)
            self.edit_split_btn.setEnabled(False)
            
    def goToSplitFromTable(self, item):
        """Navigate to split selected from table"""
        row = item.row()
        node_id = self.splits_table.item(row, 0).text()
        self.navigateToNode(node_id)
            
        self.refreshBrowser()