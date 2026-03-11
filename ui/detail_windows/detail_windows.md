# detail_windows Folder Overview

## __init__.py
- Package marker noting that specialized detail windows live in this subpackage.

## visualization_detail_window.py
- `VisualizationDetailWindow` extends the base detail framework to host charts and dashboards. It configures visualization-specific toolbars (`setup_visualization_actions`), embeds plotting widgets (`setup_visualization_content`), and uses safeguards like `_safe_set_text`/`_safe_set_pixmap` to avoid repaint issues when switching data sources.
