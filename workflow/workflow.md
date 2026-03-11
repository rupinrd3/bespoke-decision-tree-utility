# workflow Folder Overview

## execution_engine.py
- `WorkflowExecutionEngine` coordinates execution of the canvas graph. `set_workflow` records nodes and determines topological order, `run_workflow`/`WorkflowTask` execute either the full graph or a downstream subset, and `_execute_node` dispatches to node-type specific handlers (data load, transform, model train, evaluation, export). Output caching (`get_node_output`, `clear_outputs`) and graceful stop (`stop_workflow`) keep long-running runs manageable.
- `WorkflowTask` allows the engine to run inside Qt’s thread pool, emitting completion/error signals back to the UI via `_emit_signal_threadsafe`.
