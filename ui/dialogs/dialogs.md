# dialogs Folder Overview

## __init__.py
- Declares the dialogs package supplying modal and wizard flows for the desktop client.

## advanced_data_filter_dialog.py
- `AdvancedDataFilterDialog` builds a multi-condition filter designer with previews and logical groupings. It relies on `FilterCondition` DTOs and per-row `ConditionWidget`s to capture ranges, lists, null checks, and chaining logic before calling back into `data.DataProcessor`.

## advanced_filter_dialog.py
- `AdvancedFilterDialog` offers a streamlined filter interface for quick edits; it composes `BasicConditionWidget` rows, validates inputs, and emits final criteria lists.

## binary_classification_dashboard.py
- `BinaryClassificationDashboard` hosts a metrics + visualization dashboard for binary models. It uses the local `MetricsCalculator` helper to compute confusion/ROC summaries and `MetricsVisualizationWidget` to render charts; `show_binary_classification_dashboard` exposes a convenience launcher.

## categorical_split_grouping_dialog.py
- `CategoricalSplitGroupingDialog` assists in grouping categories for a candidate split. A background `CategoricalGroupingWorker` computes statistics while the dialog lets users drag categories between bins and confirm assignments.

## cloud_import_options_dialog.py
- `CloudImportOptionsDialog` captures provider credentials and object-selection parameters, delegating connection tests to `CloudTestThread` so the UI remains responsive.

## csv_options_dialog.py
- `CsvImportOptionsDialog` captures encoding, delimiter, quote/escape characters, skip-row counts, numeric separators, and optional date column/format overrides before delegating to `data.DataLoader`.

## data_import_wizard.py
- `DataImportWizard` provides a multi-page wizard covering source selection, format configuration, previews, data typing, missing-value rules, and final confirmation. Dedicated `QWizardPage` subclasses (`IntroPage`, `SourceSelectionPage`, `FormatConfigurationPage`, `DataPreviewPage`, `DataTypesPage`, `MissingValuesPage`, `SummaryPage`) encapsulate each step, and recent updates add quote/escape character inputs, skip-top/bottom row spinners, column selection checklists, decimal/thousands separator fields, and reusable date-format parsing that propagate to the final dataset load.

## data_transformation_dialog.py
- `DataTransformationDialog` exposes a library of reusable transformations, letting users stack operations, preview results, and persist the plan back to `DataProcessor`.

## database_import_options_dialog.py
- `DatabaseImportOptionsDialog` prompts for connection parameters for supported RDBMS engines. `DatabaseTestThread` runs connection and metadata checks asynchronously, surfacing schema/table listings when available.

## dataset_properties_dialog.py
- `DatasetPropertiesDialog` summarizes dataset health with `BasicStatsTableModel` feeding descriptive statistics, type breakdowns, memory usage, and target balance to the UI.

## enhanced_edit_split_dialog.py
- `EnhancedEditSplitDialog` lets users revise existing splits. `BinConfigWidget` renders the current binning, supports regrouping via the component managers, and returns a validated `SplitConfiguration`.

## enhanced_filter_dialog.py
- `EnhancedFilterDialog` is a UX refresh of the filter builder featuring expression previews. `FilterConditionWidget` wraps condition entry and validation while the dialog coordinates apply/reset actions.

## enhanced_formula_editor_dialog.py
- `EnhancedFormulaEditorDialog` embeds syntax highlighting (`FormulaSyntaxHighlighter`), validation (`FormulaValidator`), and safe evaluation (`FormulaEvaluator`) to offer a richer experience than the inline editor.

## enhanced_transform_dialog.py
- `EnhancedTransformDialog` modernizes transformation editing. `TransformationWidget` surfaces individual steps, while `TransformationEditDialog` handles advanced parameterization per operation.

## enhanced_tree_configuration_dialog.py
- `EnhancedTreeConfigurationDialog` aggregates tree-training knobs (split criteria, depth, pruning, monotonicity). It loads defaults, validates numeric ranges, and emits a normalized configuration map.

## excel_options_dialog.py
- `ExcelImportOptionsDialog` collects sheet selection, header and skip-row settings, numeric separators, and optional date parsing hints to mirror the wizard experience.

## filter_data_dialog.py
- `FilterDataDialog` supports quick, inline filtering with `FilterConditionWidget` controls, persisting conditions and previewing row counts before execution.

## find_split_dialog.py
- `FindSplitDialog` queues a `SplitFinderWorker` to search for optimal splits and displays ranked results with metric breakdowns for manual approval.

## manual_tree_growth_dialog.py
- `ManualTreeGrowthDialog` supplies a full-screen manual tree constructor, visualizing nodes via `TreeGraphicsNode`, streaming training data summaries, and exposing inline `EditSplitDialog` for the selected node.

## missing_values_dialog.py
- `MissingValuesDialog` centralizes imputation strategies, letting users select columns, choose fill methods, inspect missingness summaries, and apply a batch operation.

## node_reporting_dialog.py
- `NodeReportingDialog` consumes `NodeStatisticsCalculator` results to present multi-tab reports (overview, performance, lift, path rules) with export options.

## performance_evaluation_dialog.py
- `PerformanceEvaluationDialog` executes metrics calculations via `PerformanceEvaluationWorker`, offering summary tabs, threshold analysis, and export/report buttons once the worker returns.

## split_configuration_dialog.py
- `SplitConfigurationDialog` captures settings for a new split (feature selection, bin strategy, monotonic constraints) and validates them prior to generating a `SplitConfiguration`.

## text_options_dialog.py
- `TextImportOptionsDialog` mirrors the CSV controls for text files, covering encoding, delimiter, quote/escape characters, skip rows, numeric separators, and date parsing inputs.

## tree_configuration_dialog.py
- `TreeConfigurationDialog` exposes baseline model parameters (depth, min samples, pruning flags) and applies toggles that enable/disable advanced options dynamically.

## tree_navigation_dialog.py
- `TreeNavigationDialog` helps users locate nodes via search, bookmarks, or hierarchical browsing. `TreeSearchEngine` indexes nodes and returns paths for highlighting in the visualizer.

## unified_split_dialog.py
- `UnifiedSplitDialog` unifies auto, manual, and edit workflows in a single surface. It uses `SplitFinderWorker` for discovery, `ModernSplitCandidateTable` to display ranked options, and commits the chosen split back to the tree.

## variable_importance_dialog.py
- `VariableImportanceDialog` wraps the analytics service, launching `VariableImportanceWorker`, plotting bar charts/interaction heatmaps, and providing export buttons when results are ready.

## variable_selection_dialog.py
- `VariableSelectionDialog` simplifies variable subset selection, featuring drag-drop between available/selected lists, metadata summaries, and validation before updates propagate to downstream nodes.
