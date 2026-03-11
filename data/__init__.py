#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Module for Bespoke Utility
Handles data loading, processing, and feature engineering
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .feature_engineering import FeatureEngineering
from formula_editor import FormulaEditorDialog

from .data_loader import DataLoader
from .data_processor import DataProcessor

__all__ = ['DataLoader', 'DataProcessor', 'FeatureEngineering', 'FormulaEditorDialog']