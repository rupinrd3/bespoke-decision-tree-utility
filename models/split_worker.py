#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Joblib-friendly helpers for evaluating candidate splits outside the Qt object graph.

The functions here mirror the logic inside ``BespokeDecisionTree`` but operate on
plain numpy arrays and simple dataclasses so they can be used within the loky
backend without hitting QObject pickling issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SplitTask:
    feature: str
    feature_values: np.ndarray
    target_values: np.ndarray
    weights: np.ndarray
    current_impurity: float
    feature_type: str  # 'numeric' or 'categorical'
    target_classes: Sequence[Any]
    min_samples_leaf: float
    criterion: str
    positive_class: Optional[Any]


def evaluate_split(task: SplitTask) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Evaluate the best split for a single feature.

    Returns:
        Tuple of (feature_name, split_info) or ``None`` when no viable split exists.
    """
    try:
        if task.feature_type == "categorical":
            split_found, split_info = _evaluate_categorical(task)
        else:
            split_found, split_info = _evaluate_numeric(task)
    except Exception:
        return None

    if not split_found or not split_info or split_info.get("gain", 0) <= 0:
        return None

    return task.feature, split_info


def _evaluate_numeric(task: SplitTask) -> Tuple[bool, Dict[str, Any]]:
    feature_values = np.asarray(task.feature_values)
    target_values = np.asarray(task.target_values)
    weights = np.asarray(task.weights, dtype=float)
    total_samples_considered = feature_values.size

    valid_mask = ~np.isnan(feature_values.astype(float, copy=False))
    if not np.any(valid_mask):
        return False, {}

    feature_values = feature_values[valid_mask]
    target_values = target_values[valid_mask]
    weights = weights[valid_mask]
    valid_sample_count = feature_values.size
    sample_coverage = (
        valid_sample_count / total_samples_considered
        if total_samples_considered > 0 else 0.0
    )

    parent_counts = {}
    for target_value, weight in zip(target_values, weights):
        parent_counts[target_value] = parent_counts.get(target_value, 0.0) + float(weight)
    parent_impurity = _calculate_impurity(parent_counts, task.criterion)

    sort_idx = np.argsort(feature_values)
    feature_values = feature_values[sort_idx]
    target_values = target_values[sort_idx]
    weights = weights[sort_idx]

    unique_values = np.unique(feature_values)
    if unique_values.size <= 1:
        return False, {}

    split_points = (unique_values[1:] + unique_values[:-1]) / 2.0

    classes = list(task.target_classes)
    cls_index = {cls: i for i, cls in enumerate(classes)}
    if not classes:
        for cls in target_values:
            if cls not in cls_index:
                cls_index[cls] = len(cls_index)
        classes = list(cls_index.keys())
    n_classes = len(classes)

    n_samples = feature_values.shape[0]
    W = np.zeros((n_samples, n_classes), dtype=float)
    for i in range(n_samples):
        j = cls_index.get(target_values[i])
        if j is not None:
            W[i, j] = weights[i]

    left_counts_arr = np.zeros(n_classes, dtype=float)
    right_counts_arr = W.sum(axis=0)

    best_gain = -np.inf
    best_split: Optional[Dict[str, Any]] = None
    sample_idx = 0

    for threshold in split_points:
        while sample_idx < n_samples and feature_values[sample_idx] <= threshold:
            left_counts_arr += W[sample_idx]
            right_counts_arr -= W[sample_idx]
            sample_idx += 1

        left_weight = float(left_counts_arr.sum())
        right_weight = float(right_counts_arr.sum())
        total_weight = left_weight + right_weight
        if total_weight <= 0:
            continue
        if left_weight < task.min_samples_leaf or right_weight < task.min_samples_leaf:
            continue

        left_impurity = _calculate_impurity(left_counts_arr, task.criterion)
        right_impurity = _calculate_impurity(right_counts_arr, task.criterion)

        weighted_impurity = (
            (left_weight / total_weight) * left_impurity
            + (right_weight / total_weight) * right_impurity
        )
        gain = float(parent_impurity - weighted_impurity)

        if gain > best_gain:
            best_gain = gain
            left_counts = {cls: float(left_counts_arr[idx]) for cls, idx in cls_index.items()}
            right_counts = {cls: float(right_counts_arr[idx]) for cls, idx in cls_index.items()}
            best_split = {
                "feature": task.feature,
                "threshold": float(threshold),
                "gain": float(gain),
                "impurity_decrease": float(gain),
                "left_impurity": float(left_impurity),
                "right_impurity": float(right_impurity),
                "left_counts": left_counts,
                "right_counts": right_counts,
                "split_type": "numeric",
                "left_samples": int(left_weight),
                "right_samples": int(right_weight),
            }

    if best_split is None:
        return False, {}

    left_total = sum(best_split.get("left_counts", {}).values())
    right_total = sum(best_split.get("right_counts", {}).values())
    best_split["samples_evaluated"] = int(valid_sample_count)
    best_split["sample_coverage"] = sample_coverage
    best_split["total_samples_considered"] = int(total_samples_considered)
    best_split["missing_value_child_idx"] = 0 if left_total >= right_total else 1
    best_split["parent_impurity"] = parent_impurity
    return True, best_split


def _evaluate_categorical(task: SplitTask) -> Tuple[bool, Dict[str, Any]]:
    feature_values = np.asarray(task.feature_values)
    target_values = np.asarray(task.target_values)
    weights = np.asarray(task.weights)
    total_samples_considered = feature_values.size

    valid_mask = ~pd_isna(feature_values)
    if not np.any(valid_mask):
        return False, {}

    feature_values = feature_values[valid_mask]
    target_values = target_values[valid_mask]
    weights = weights[valid_mask]
    valid_sample_count = feature_values.size
    sample_coverage = (
        valid_sample_count / total_samples_considered
        if total_samples_considered > 0 else 0.0
    )

    parent_counts = {}
    for target_value, weight in zip(target_values, weights):
        parent_counts[target_value] = parent_counts.get(target_value, 0.0) + float(weight)
    parent_impurity = _calculate_impurity(parent_counts, task.criterion)

    categories = np.unique(feature_values)
    if categories.size <= 1:
        return False, {}

    if categories.size > 10:
        split_found, split_info = _evaluate_grouped_categories(
            task, categories, feature_values, target_values, weights, parent_impurity
        )
    else:
        split_found, split_info = _evaluate_binary_categories(
            task, categories, feature_values, target_values, weights, parent_impurity
        )

    if split_found and split_info:
        left_total = sum(split_info.get("left_counts", {}).values())
        right_total = sum(split_info.get("right_counts", {}).values())
        split_info["samples_evaluated"] = int(valid_sample_count)
        split_info["sample_coverage"] = sample_coverage
        split_info["total_samples_considered"] = int(total_samples_considered)
        split_info["missing_value_child_idx"] = 0 if left_total >= right_total else 1
        split_info["parent_impurity"] = parent_impurity
        return True, split_info

    return False, {}


def _evaluate_grouped_categories(
    task: SplitTask,
    categories: np.ndarray,
    feature_values: np.ndarray,
    target_values: np.ndarray,
    weights: np.ndarray,
    parent_impurity: float,
) -> Tuple[bool, Dict[str, Any]]:
    category_stats: Dict[Any, Dict[str, Any]] = {}

    for cat in categories:
        cat_mask = feature_values == cat
        if not np.any(cat_mask):
            continue

        cat_targets = target_values[cat_mask]
        cat_weights = weights[cat_mask]

        counts = {cls: 0.0 for cls in task.target_classes}
        for cls in task.target_classes:
            cls_mask = cat_targets == cls
            if np.any(cls_mask):
                counts[cls] = cat_weights[cls_mask].sum()

        total_weight = sum(counts.values())
        if total_weight <= 0:
            continue

        positive_prop = _resolve_positive_proportion(counts, task.positive_class, task.target_classes)
        category_stats[cat] = {
            "counts": counts,
            "positive_prop": positive_prop,
            "total_weight": total_weight,
        }

    sorted_cats = sorted(category_stats.keys(), key=lambda c: category_stats[c]["positive_prop"])
    if len(sorted_cats) <= 1:
        return False, {}

    best_gain = -np.inf
    best_split: Optional[Dict[str, Any]] = None

    for idx in range(1, len(sorted_cats)):
        left_cats = set(sorted_cats[:idx])

        left_counts = {cls: 0.0 for cls in task.target_classes}
        right_counts = {cls: 0.0 for cls in task.target_classes}

        for cat, stats in category_stats.items():
            target_counts = stats["counts"]
            if cat in left_cats:
                for cls, count in target_counts.items():
                    left_counts[cls] += count
            else:
                for cls, count in target_counts.items():
                    right_counts[cls] += count

        split_found, split_info = _finalise_categorical_split(
            task,
            left_counts,
            right_counts,
            left_cats,
            categories,
            best_gain,
            parent_impurity,
        )

        if split_found and split_info["gain"] > best_gain:
            best_gain = split_info["gain"]
            best_split = split_info

    if best_split is None:
        return False, {}

    return True, best_split


def _evaluate_binary_categories(
    task: SplitTask,
    categories: np.ndarray,
    feature_values: np.ndarray,
    target_values: np.ndarray,
    weights: np.ndarray,
    parent_impurity: float,
) -> Tuple[bool, Dict[str, Any]]:
    n_categories = len(categories)
    max_combinations = 2 ** (n_categories - 1) - 1
    max_to_try = 100 if n_categories > 7 else max_combinations

    best_gain = -np.inf
    best_split: Optional[Dict[str, Any]] = None
    combinations_tried = 0

    for subset_size in range(1, n_categories):
        for subset in combinations(range(n_categories), subset_size):
            combinations_tried += 1
            if combinations_tried > max_to_try:
                break

            left_cats = set(categories[list(subset)])
            left_counts = {cls: 0.0 for cls in task.target_classes}
            right_counts = {cls: 0.0 for cls in task.target_classes}

            for cat, target, weight in zip(feature_values, target_values, weights):
                if cat in left_cats:
                    left_counts[target] += weight
                else:
                    right_counts[target] += weight

            split_found, split_info = _finalise_categorical_split(
                task,
                left_counts,
                right_counts,
                left_cats,
                categories,
                best_gain,
                parent_impurity,
            )

            if split_found and split_info["gain"] > best_gain:
                best_gain = split_info["gain"]
                best_split = split_info

        if combinations_tried > max_to_try:
            break

    if best_split is None:
        return False, {}

    return True, best_split


def _finalise_categorical_split(
    task: SplitTask,
    left_counts: Dict[Any, float],
    right_counts: Dict[Any, float],
    left_categories: Iterable[Any],
    all_categories: np.ndarray,
    current_best_gain: float,
    parent_impurity: float,
) -> Tuple[bool, Dict[str, Any]]:
    left_weight = sum(left_counts.values())
    right_weight = sum(right_counts.values())
    total_weight = left_weight + right_weight

    if total_weight <= 0:
        return False, {}

    left_impurity = _calculate_impurity(left_counts, task.criterion)
    right_impurity = _calculate_impurity(right_counts, task.criterion)

    weighted_impurity = (
        (left_weight / total_weight) * left_impurity
        + (right_weight / total_weight) * right_impurity
    )

    gain = parent_impurity - weighted_impurity
    if gain <= current_best_gain:
        return False, {}

    split_info = {
        "feature": task.feature,
        "left_categories": list(left_categories),
        "right_categories": [cat for cat in all_categories if cat not in left_categories],
        "gain": float(gain),
        "impurity_decrease": float(gain),
        "left_impurity": float(left_impurity),
        "right_impurity": float(right_impurity),
        "left_counts": left_counts.copy(),
        "right_counts": right_counts.copy(),
        "split_type": "categorical",
        "left_samples": int(left_weight),
        "right_samples": int(right_weight),
        "parent_impurity": parent_impurity,
    }

    return True, split_info


def _calculate_impurity(class_counts, criterion: str) -> float:
    """Accepts dict[class->count] or 1D numpy array of counts."""
    if isinstance(class_counts, dict):
        total_count = sum(class_counts.values())
        if total_count <= 0:
            return 0.0
        proportions = [count / total_count for count in class_counts.values() if count > 0]
        if not proportions:
            return 0.0
        if criterion == "gini":
            return 1.0 - sum(p * p for p in proportions)
        if criterion in {"entropy", "information_gain"}:
            return float(-sum(p * np.log2(p) for p in proportions))
        if criterion == "misclassification":
            return 1.0 - max(proportions)
        return 1.0 - sum(p * p for p in proportions)

    counts = np.asarray(class_counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    if p.size == 0:
        return 0.0
    if criterion == "gini":
        return float(1.0 - np.sum(p * p))
    if criterion in {"entropy", "information_gain"}:
        return float(-np.sum(p * np.log2(p)))
    if criterion == "misclassification":
        return float(1.0 - np.max(p))
    return float(1.0 - np.sum(p * p))


def _resolve_positive_proportion(
    counts: Dict[Any, float],
    positive_class: Optional[Any],
    classes: Sequence[Any],
) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    resolved_positive = _normalize_positive_class(classes, positive_class)
    return counts.get(resolved_positive, 0.0) / total


def _normalize_positive_class(classes: Sequence[Any], candidate: Optional[Any]) -> Any:
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
        preferred = [
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
        for token in preferred:
            for cls in classes:
                if str(cls).lower() == str(token).lower():
                    return cls
        return classes[1]
    try:
        return sorted(classes)[-1]
    except TypeError:
        return classes[-1]


def pd_isna(values: np.ndarray) -> np.ndarray:
    """
    Pandas-like ``isna`` helpers without importing pandas inside worker processes.
    """
    if values.dtype.kind in {"f", "c"}:
        return np.isnan(values)
    if values.dtype.kind in {"O", "U", "S"}:
        return np.equal(values, None) | (values == "") | (values != values)
    return np.zeros_like(values, dtype=bool)
