# components Folder Overview

## __init__.py
- Declares the UI components package used across dialogs and widgets.

## bin_manager.py
- `BinManager` provides shared logic for bin editing panels, supporting dataset injection (`set_feature_data`), bin CRUD (`add_bin`, `remove_bin`, `update_bin`), distribution summaries, and emitting change callbacks.

## categorical_bin_manager.py
- `CategoricalBinManager` assembles drag-and-drop grouping controls around `BinWidget` containers and the `DraggableCategoryList`. It loads category frequency data (`load_data`), populates unassigned tables, auto-groups small categories, and exposes the final mapping for split configuration.

## numerical_bin_manager.py
- `EnhancedNumericalBinManager` renders histogram previews and interactive sliders for numeric binning. Key methods include `load_data`, `set_precision`, `set_auto_adjust`, `recalculate_bins`, and `preview_split_quality`.

## modern_toolbar.py
- `ModernToolbar` assembles the ribbon-like toolbar seen in the main window. It applies bespoke styling and exposes section builders (`add_universal_section`, `add_context_section`, `add_workflow_actions`) that wire custom buttons to host callbacks.

## widget_toolbar.py
- `WidgetToolbar` delivers a lightweight, embeddable toolbar for dialogs. It supports button/separator creation and styling, while `ToolbarContainer` offers a standardized chrome for hosting those toolbars inside panels.

## enhanced_status_bar.py
- `EnhancedStatusBar` extends Qt’s status bar with `BusyIndicator` animations, a `MemoryIndicator`, and clock + message panes. It surfaces `start_time_updates`, `update_time`, `show_memory_warning`, and `set_busy` for the main window.

## tree_node_context_menu.py
- `TreeNodeContextMenu` constructs right-click menus for tree nodes; it tailors actions for internal versus terminal nodes (`_add_internal_node_actions`, `_add_terminal_node_actions`) and funnels selection back to the `MainWindow`.
