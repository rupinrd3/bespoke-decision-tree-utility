# node_logic Folder Overview

## evaluation_node_logic.py
- `EvaluationNodeLogic` defines how an Evaluation node behaves during workflow execution: `execute` validates inputs (trained `BespokeDecisionTree` plus evaluation `DataFrame`), computes performance metrics via `MetricsCalculator`, optionally renders charts through `MetricsVisualizer`, and returns a metrics dictionary for downstream nodes. `get_configurable_parameters` lists knob settings surfaced in the UI.
