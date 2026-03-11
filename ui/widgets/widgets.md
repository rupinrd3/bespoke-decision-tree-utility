# widgets Folder Overview

## variable_selector_widget.py
- `VariableSelectorWidget` presents a searchable table of variables with checkbox selection, importance scores, and filters. It emits `selectionChanged` when the selection set updates and provides utilities like `_populate_table`, `_filter_variables`, `_select_all`, and `_deselect_all`.

## model_properties_widget.py
- `ModelPropertiesWidget` visualizes metadata about a `BespokeDecisionTree`. `set_model` populates labels for IDs, target, feature counts, depth, parameters, and training history, while `update_properties` refreshes when the underlying model changes.

## lift_chart_widget.py
- `LiftChartWidget` embeds a Matplotlib canvas for lift and gain charts. `plot_lift_chart` expects sorted predictions, draws lift/gain curves, and annotates key percentiles; `clear_chart` and `_clear_plot` reset the figure when data is unavailable.
