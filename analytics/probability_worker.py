#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Worker utilities for computing terminal node probabilities in parallel.

The functions in this module are intentionally free of Qt/QObject dependencies
so they can be executed safely inside Joblib's ``loky`` backend.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from models.node import TreeNode


def compute_probabilities_chunk(
    tree_payload: Dict[str, Any],
    records: np.ndarray,
    feature_names: Sequence[str],
    positive_class: Any,
) -> List[float]:
    """
    Compute positive-class probabilities for a contiguous chunk of records.

    Args:
        tree_payload: Dictionary representation of the root ``TreeNode``.
        records: 2D numpy array slice for the current chunk.
        feature_names: Column names (ordered) that align with ``records``.
        positive_class: Requested positive class, may be ``None``.

    Returns:
        List of floats representing per-row positive probabilities.
    """
    root = TreeNode.from_dict(tree_payload)
    results: List[float] = []

    for row in records:
        sample_mapping = _build_sample_mapping(row, feature_names)
        terminal_node = root._traverse_to_terminal(sample_mapping)
        probability = _extract_probability(terminal_node, positive_class)
        results.append(float(probability))

    return results


def _build_sample_mapping(row: np.ndarray, feature_names: Sequence[str]) -> Dict[str, Any]:
    return {feature_names[idx]: row[idx] for idx in range(len(feature_names))}


def _extract_probability(node: TreeNode, positive_class: Any) -> float:
    if getattr(node, "class_counts", None):
        class_counts = node.class_counts
        total = sum(class_counts.values())
        classes = list(class_counts.keys())
        resolved_positive = _normalize_positive_class(classes, positive_class)
        positive = class_counts.get(resolved_positive, 0)
        return positive / total if total else 0.0

    node_probability = getattr(node, "probability", None)
    if node_probability is not None:
        majority = getattr(node, "majority_class", None)
        if positive_class is None or majority is None:
            return float(node_probability)
        if str(majority) == str(positive_class):
            return float(node_probability)
        return float(1.0 - float(node_probability))

    return 0.5


def _normalize_positive_class(classes: Iterable[Any], candidate: Any) -> Any:
    classes = list(classes)
    if not classes:
        return candidate
    if candidate is None:
        return _guess_positive_class(classes)
    if candidate in classes:
        return candidate
    for cls in classes:
        if str(cls) == str(candidate):
            return cls
    if isinstance(candidate, str):
        for cls in classes:
            if isinstance(cls, (int, float)):
                try:
                    if float(candidate) == float(cls):
                        return cls
                except ValueError:
                    continue
    if isinstance(candidate, (int, float)):
        for cls in classes:
            if isinstance(cls, str):
                if cls == str(candidate):
                    return cls
                try:
                    if float(cls) == float(candidate):
                        return cls
                except ValueError:
                    continue
    return _guess_positive_class(classes)


def _guess_positive_class(classes: Sequence[Any]) -> Any:
    if len(classes) == 2:
        preferred_tokens = [
            1,
            True,
            "1",
            "True",
            "true",
            "Yes",
            "yes",
            "Y",
            "y",
            "positive",
            "Positive",
        ]
        for token in preferred_tokens:
            for cls in classes:
                if str(cls).lower() == str(token).lower():
                    return cls
        return classes[1]

    try:
        return sorted(classes)[-1]
    except TypeError:
        return classes[-1]

