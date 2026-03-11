# utils Folder Overview

## __init__.py
- Declares the shared utilities package.

## binary_target_validator.py
- `BinaryTargetValidator` inspects candidate targets to ensure they are binary, recommending recodes via `_suggest_binary_mapping`, `_suggest_groupings`, and `_suggest_numeric_thresholds` when the dataset isn’t clean.

## config.py
- Configuration helpers: `get_config_path` locates `config.json`, `load_configuration` merges defaults with overrides, `save_configuration` persists edits, and `merge_configs`/`validate_configuration` enforce structure. `get_config_value`/`set_config_value` provide safe accessors, while `create_directories` and `get_application_dirs` bootstrap folder layout.

## logging_utils.py
- `setup_logging` wires rotating file handlers and optional console logging; `set_log_level`, `create_module_logger`, and `flush_logs` adjust running loggers. `LogCapture` is a context manager for capturing log output in tests, `_MemoryHandler` buffers records in memory, and `log_exception`/`log_system_info` emit diagnostics.

## memory_management.py
- Functions for memory hygiene: `configure_memory_settings` tunes pandas/numpy settings, `optimize_dataframe` downcasts columns, `monitor_memory_usage` and `get_top_memory_processes` surface system state, and `chunk_dataframe` helps stream large datasets.

## metrics_calculator.py
- `CentralMetricsCalculator` supplies lightweight aggregations (percentage contributions, lift, class distributions, tree-level statistics) for analytics UIs. `MetricsValidator` sanity-checks node metric inputs before reporting.

## performance_optimizer.py
- Caching and background helpers: `SplitQualityCache` memoizes split analytics with LRU eviction, `BinStatisticsCalculator` computes per-bin stats and Gini metrics, and `BackgroundStatsCalculator` can offload long-running calculations. Module functions expose singleton instances.

## serialization_utils.py
- JSON-safe serialization helpers: `make_json_serializable` and `safe_pandas_dtypes_to_dict` coerce numpy/pandas objects, `safe_json_dump(s)` write data with consistent encoding, `create_serializable_metadata` assembles standard project metadata, and `NumpyJSONEncoder` plugs into the standard library for custom dumping.
