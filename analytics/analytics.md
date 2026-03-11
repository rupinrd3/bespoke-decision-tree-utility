# analytics Folder Overview

## variable_importance.py
- `VariableImportance` (Qt `QObject`) orchestrates feature-importance calculations; `compute_importance` switches between impurity, permutation, drop-column, and gain workflows, `calculate_gini_importance` traverses `TreeNode`s to score features, and reporting helpers (`calculate_feature_interactions`, `get_top_features`, `analyze_importance`, `export_importance_report`, `plot_importance`) keep UI summaries in sync.
- `InteractionImportance` runs pairwise permutations against the fitted tree (`compute_interactions`, `_tree_score`, `_predict_sample`) and can visualize cross-feature lift via `plot_interactions`.

## performance_metrics.py
- `MetricsCalculator` batches accuracy, confusion matrices, ROC/PR curves, class reports, log loss, and bootstrap stability; `compute_metrics` and `calculate_comprehensive_metrics` decide when to offload work in parallel, while helpers (`_calculate_advanced_metrics`, `_calculate_simple_cv_metrics`, `_accuracy` fallbacks) keep results robust when scikit-learn metrics are unavailable.
- `MetricsVisualizer` wraps Matplotlib charting (`plot_confusion_matrix`, `plot_roc_curve`, `plot_precision_recall_curve`, `plot_metrics_summary`, `plot_lift_chart`) for the UI dialogs.

## node_statistics.py
- `NodeAnalyzer` surfaces tree-level diagnostics: `get_node_report` assembles per-node metrics, `analyze_tree` and `_calculate_balance_factor` summarise shape and balance, while utilities like `get_node_path_rules`, `plot_node_class_distribution`, and `calculate_terminal_node_metrics` feed the visualization panes.

## node_report_generator.py
- `NodeReportGenerator` converts a fitted tree and evaluation dataset into a tabular report; `generate_node_report` validates inputs, walks terminal nodes, calculates lift/precision via `_calculate_node_statistics`, and applies `_calculate_cumulative_metrics` before export. Excel export support lives in `export_to_excel` and `_format_excel_sheet`.
