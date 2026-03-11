# styling Folder Overview

## __init__.py
- Declares the styling subpackage that hosts theme resources.

## theme_manager.py
- `ThemeManager` centralizes application theming. It generates light/dark palettes (`_get_light_theme`, `_get_dark_theme`), composes the global stylesheet (`_generate_global_stylesheet`, `_apply_palette_to_app`), and exposes `apply_theme`/`apply_palette` to update the running Qt application.
