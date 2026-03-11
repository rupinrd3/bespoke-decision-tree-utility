# models Folder Overview

## __init__.py
- Declares the `models` package and documents that it houses core modeling primitives.

## decision_tree.py
- `BespokeDecisionTree` is the interactive model backing the UI. It inherits `QObject` to emit progress signals, manages training/validation parameters (`_set_default_parameters`, `fit`, `predict`, `predict_proba`), and exposes manual-edit entry points (`apply_manual_split`, `merge_nodes`, `prune_subtree`, `undo_last_operation`). Support routines handle serialization (`to_dict`, `from_dict`), caching training data for manual edits, and coordinating enhanced validation with split configuration/transactions.
- Enumerations `SplitCriterion` and `TreeGrowthMode` enumerate supported growth strategies for downstream modules.

## node.py
- `TreeNode` represents each tree vertex, storing split metadata, statistics, and visualization hints. Utility methods (`add_child`, `remove_child`, `to_dict`, `from_dict`, `clone_subtree`, `get_path_rules`) maintain structural integrity during edits and exports.

## split_finder.py
- `SplitFinder` evaluates candidate splits across numeric and categorical features; `find_best_split` ranks binning options, while helpers (`_select_features_for_split`, `_analyze_variable_characteristics`, `_generate_numeric_bins`, `_evaluate_categorical_split`, `_calculate_split_score`) balance information gain with stability. A local `SplitCriterion` enum mirrors the tree’s scoring modes.

## split_configuration.py
- Defines validated configuration objects. `SplitConfiguration` abstracts shared behaviour, with `NumericSplit`, `CategoricalSplit`, and `MultiBinSplit` implementing `validate`, `to_dict`, and `get_child_assignment`. `ValidationResult` captures issues, and `SplitConfigurationFactory` streamlines building the correct variant from UI input.

## split_validator.py
- `SplitValidator` inspects pending splits against training data, enforcing node capacity, monotonicity, and bin safety. `validate_split`, `_validate_numeric_split`, and `_validate_categorical_split` surface actionable warnings/errors for the manual workflow.

## split_transaction.py
- Transaction utilities guarantee safe tree mutations: `SplitTransaction` encapsulates proposed operations, `atomic_split_operation` wraps apply/rollback semantics, and `safe_apply_split` handles error propagation. Helper methods (`_capture_pre_split_state`, `_apply_split`, `_rollback`) keep node statistics consistent.

## surrogate_handler.py
- `SurrogateSplitHandler` identifies surrogate splits for handling missing values or failed predicates; `find_surrogate_splits`, `_calculate_surrogate_gain`, and `_rank_surrogates` compute backups when the primary split cannot evaluate cleanly.

## tree_pruning.py
- `TreePruner` offers automatic pruning strategies. `cost_complexity_pruning`, `reduced_error_pruning`, and `minimal_depth_pruning` compute pruning candidates, while `_evaluate_pruned_tree` and `_calculate_complexity_penalty` quantify impact.
