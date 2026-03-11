# Repository Guidelines

## Project Structure & Module Organization
The desktop entry point is `main.py`; domain packages include `analytics/` (metrics), `business/` (scoring logic), `models/` (tree engine), `workflow/` (canvas orchestration), and `ui/` (components, dialogs, widgets). Data connectors sit in `data/`, export pipelines in `export/`, reusable helpers in `utils/`, and platform scripts in `scripts/`. Generated artifacts land in `logs/` and `export/`; keep them untracked.

## Build, Run, and Tooling
Spin up a venv with `python3.12 -m venv venv && source venv/bin/activate`, then `pip install -r requirements.txt`. Launch locally using `python main.py`; `bash scripts/run_ubuntu.sh` preps Qt on Linux, while Windows developers can lean on `scripts/run_portable.bat`. Logs rotate under `logs/`—check the newest file when chasing startup issues. Use `scripts/build_portable.bat` only when packaging tagged releases.

## Coding Style & Naming Conventions
Adhere to PEP 8 with 4-space indentation and keep line length near 88 chars. Favor type hints and module-level docstrings that describe data shapes and side effects. Qt classes should use descriptive suffixes (`DecisionTreeDialog`, `WorkflowCanvasWidget`) so imports stay readable. Before adding new helpers, look in `utils/` or extend existing modules to preserve the project's domain-driven layout.

## Testing Guidelines
Pytest is the target harness. If no `tests/` directory is present, create one that mirrors source packages (e.g., `tests/workflow/test_executor.py`). Name files `test_<subject>.py`, rely on fixtures for sizeable datasets, and cover both numerical outputs and workflow edge cases. Execute suites with `python -m pytest` and attach relevant `logs/` snippets when reporting flakiness.

## Commit & Pull Request Guidelines
Write concise, imperative commit subjects (`Add workflow cache invalidation`) with optional scopes (`ui:`). Reference issues where applicable and align bullet points with the `CHANGELOG.md` headings (`Added`, `Changed`, `Fixed`) to simplify release notes. Pull requests need a problem statement, test evidence (`python -m pytest` output or UI screenshots), and a note on data/schema impacts. Request review from the maintainer responsible for the touched module.

## Configuration & Data Care
Runtime defaults originate in `config.json` and `utils/config.py`; update them together and never hard-code credentials. Staged datasets belong under `data/` only after anonymization. Keep large exports, cached models, and log files out of version control—existing `.gitignore` rules already cover them, so avoid manual overrides.
