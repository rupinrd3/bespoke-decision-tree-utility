# ui Folder Overview

## __init__.py
- Marks the UI package and references the desktop-facing modules bundled here.

## base_detail_window.py
- `BaseDetailWindow` standardizes dialog chrome, sizing, and status widgets for detail windows (`setup_window_chrome`, `setup_content_area`, `apply_base_styling`).
- `DataDetailWindow` and `ModelDetailWindow` specialize the base class, wiring dataset/model actions, embedding viewers, and linking to live tree signals (`connect_tree_signals`, `on_tree_node_selected`).

## data_viewer.py
- `PandasModel` exposes a pandas `DataFrame` to Qt’s model/view APIs (`rowCount`, `columnCount`, `data`, `headerData`).
- `DataViewerWidget` wraps the table, providing `set_dataframe`, `apply_filter`, and header-driven stats for the detail panes.

## main_window.py
- `MainWindow` orchestrates the desktop shell: `init_ui` builds menus, toolbars, status bar, breadcrumbs, and dock widgets; numerous event handlers (`on_workflow_node_context_menu`, `configure_workflow_node`, `run_workflow_from_canvas`, `manual_tree_building_mode`) synchronize the workflow canvas, tree visualizer, and dialogs. It also manages project lifecycle (`new_project`, `open_project`, `save_project`) and analysis/report launches.
- `EnhancedExportDialog` supplies the multi-language export workflow, updating file extensions and dispatching to exporter services.

## node_editor.py
- `NodePropertiesWidget`, `SplitFinderWidget`, and `NodeReportWidget` present editable node metadata, trigger split searches, and render analytics respectively.
- `NodeEditorDialog` coordinates the widgets, routing selection changes, persisting modifications, and broadcasting updates back to the tree.

## streamlined_manual_split_dialog.py
- `StreamlinedManualSplitDialog` is the guided manual split workflow: it previews distributions (`DataDistributionWidget`), manages categorical grouping (`CategoryGroupWidget`), wires shortcuts, and applies user-confirmed thresholds back to the model.

## streamlined_split_dialog.py
- `StreamlinedSplitDialog` provides a compact auto-split chooser. It calls `find_splits_automatically`, populates ranked candidates, and returns the selected configuration via helper functions `setRowData`/`getRowData`.

## tree_visualizer.py
- Scene graph implementation for rendered trees: `EnhancedTreeNodeItem` and `EnhancedTreeEdgeItem` draw nodes/edges with tooltips, `EnhancedTreeScene` lays out coordinates (`update_tree`, `_calculate_node_positions`), and `TreeVisualizerWidget` hosts toolbar controls (zoom, focus, highlighting) while relaying selection events.

## variable_viewer.py
- `VariableViewerWidget` lists available variables with metadata, supporting quick refresh (`update_variables`) and clearing when datasets change.

## window_manager.py
- `WindowManager` tracks stacked widgets, instantiates registered windows on demand (`register_window`, `_create_window_instance`, `show_window`), and keeps breadcrumb navigation (`BreadcrumbWidget`) in sync with navigation history.

## workflow_canvas.py
- Node classes (`WorkflowNode`, `DatasetNode`, `ModelNode`, `FilterNode`, `TransformNode`, `EvaluationNode`, `VisualizationNode`, `ExportNode`) encapsulate node-specific configuration and rendering. `ConnectionData` stores edge metadata for serialization.
- `WorkflowScene` and `WorkflowCanvas` manage node placement, drag-drop wiring, and execution hooks (`add_node`, `add_connection`, `remove_node`, `run_workflow_from_canvas`). `NodeConfigDialog` offers in-place editing for generic node settings.
