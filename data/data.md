# data Folder Overview

## __init__.py
- Marks the `data` package and documents that it hosts ingestion and transformation helpers.

## data_loader.py
- `DataLoader` handles file-based ingestion: `load_file` routes to format-specific loaders (`_load_csv`, `_load_excel`, `_load_text`, `_load_parquet`, `_load_feather`, `_load_pickle`), applies `optimize_dataframe`, and tracks dataset metadata.
- CSV/Text loaders now honour quote/escape characters, skip-top/bottom rows, decimal/thousands separators, and `usecols` filters while auto-switching to the python engine when `skipfooter` is enabled.
- Columnar loaders (`parquet`, `feather`, `pickle`) accept column subsets, apply row trimming post-load, and support custom date parsing via `_apply_custom_date_format`.
- Internals like `_detect_binary_columns`, `_optimize_dtypes_existing_df`, and `_auto_detect_and_convert_types` reduce memory footprint while surfacing potential targets.

## data_processor.py
- `DataProcessor` (Qt `QObject`) records transformations. `filter_data` builds boolean masks via `_apply_comparison`, `handle_missing_values` wraps common imputation strategies, and `transform_column` applies user-defined functions; `progressUpdated`/`processingComplete` signals keep the UI reactive and `get_processing_history` exposes an audit trail.

## feature_engineering.py
- `FormulaParser` tokenizes, validates, and evaluates formula expressions, including an `IF` helper for conditional logic.
- `FeatureEngineering` encapsulates derived-field creation (`create_formula_variable`, `create_binned_variable`, `create_interaction_term`, `create_binary_variable`) and logs history for reproducibility.

## database_importer.py
- `DatabaseImporter` abstracts relational sources: `connect` branches across SQLite/PostgreSQL/MySQL/SQL Server/Oracle (with lazy connector imports), `import_data_from_query` and `import_table` pull pandas frames, and administration helpers (`list_tables_or_views`, `list_schemas`, `get_table_info`, `test_connection`) aid the import dialogs. Custom exceptions `DatabaseConnectionError` and `DatabaseQueryError` clarify failure modes.

## cloud_importer.py
- `CloudImporter` centralizes bucket access for AWS S3, Google Cloud Storage, and Azure Blob (`_get_s3_client`, `_get_gcs_client`, `_get_azure_client`). `list_buckets`, `list_files`, and `download_file_to_dataframe` expose common workflows, while `get_file_info` and `test_connection` provide diagnostics. Supported providers and formats are enumerated for UI validation.
