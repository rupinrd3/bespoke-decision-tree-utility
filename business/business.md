# business Folder Overview

## split_statistics_calculator.py
- `SplitStatisticsCalculator` measures split quality: `calculate_impurity` supports Gini/entropy/log-loss, `calculate_split_quality` compiles bin-level metrics and overall impurity drop, while `evaluate_split_effectiveness`, `calculate_feature_importance`, `suggest_optimal_bins`, and `calculate_split_stability` turn those metrics into actionable guidance.
- `SplitQualityMetrics` dataclass captures the summary returned to calling code (before/after impurity, gain ratio, and per-bin diagnostics).

## split_preview_generator.py
- `SplitPreviewGenerator` turns raw split metrics into UI-friendly content; `generate_preview` queries `SplitStatisticsCalculator`, ` _create_bin_details`, and `_create_summary`, then surfaces human-readable recommendations via `_generate_recommendations` and `_generate_warnings`.
- `generate_comparison_preview` and `balance_score` compare multiple binning strategies, while the `SplitPreview` dataclass packages results for display.
