# export Folder Overview

## __init__.py
- Declares the `export` package and documents that it contains model export utilities.

## python_exporter.py
- `GenericNodeExporter` traverses a fitted `BespokeDecisionTree`, captures terminal-node logic, and generates standalone Python scoring code via `export_model_to_python`. Helpers extract feature types (`_extract_feature_types_from_model`), materialize branch conditions (`_extract_terminal_node_conditions`, `_generate_complete_python_code`), and map categorical fallbacks.

## enhanced_python_exporter.py
- `MultiLanguageExporter` brokers exports across Python, R, Java, JavaScript, C#, SQL, and Scala; `export_model` dispatches to language-specific subclasses while `get_supported_languages` and `get_language_info` feed the UI.
- Each `*Exporter` subclass (e.g., `PythonExporter`, `RExporter`) implements `export_model` plus metadata helpers (`get_file_extension`, `get_description`, `_generate_*_code`). `ExportTemplateGenerator` and `ExportOptimizer` provide additional integration guides and code-size diagnostics, while module-level functions expose convenience wrappers (`export_model_to_language`, `validate_export_parameters`).

## multi_format_model_exporter.py
- `MultiFormatModelExporter` serializes trees into PMML, JSON, CSV summaries, and quick-look tables; `export_pmml` builds an XML tree and `export_json`/`export_csv_summary` (and related helpers like `_serialize_tree_node`, `_create_pmml_node`, `_collect_nodes_by_depth`) shape the fields expected by downstream systems.

## model_saver.py
- `ModelSaver` coordinates persistence: `save_model` writes pickle bundles with metadata, while `save_pmml`, `save_json`, and `save_python` delegate to format-specific exporters. The companion `PMMLExporter` and `JSONExporter` classes encapsulate their respective serialization logic.

## correct_node_assignment_generator.py
- Functional helpers (`parse_node_logic_from_csv`, `convert_logic_to_python`, `generate_correct_node_assignment_code`) translate legacy CSV-based logic descriptors into executable Python snippets, ensuring exported scorers stay aligned with the interactive tree.
