# Changelog

All notable changes to the Bespoke Decision Tree Utility will be documented in this file.

## [1.0] - 2026-03-11 — First Stable Release

### Overview
Version 1.0 is the first stable production release of the Bespoke Decision Tree Utility. It delivers a fully functional, tested desktop application for interactive decision tree modelling with a focus on credit risk assessment and binary classification.

### Added
- **Visual Workflow Canvas** — drag-and-drop pipeline builder with node-based ML workflow execution
- **Binary Classification Dashboard** — comprehensive metrics including Gini, KS statistic, ROC-AUC, and Population Stability Index (PSI)
- **Variable Importance Analysis** — Gini impurity, permutation importance, and information gain methods
- **Node Reporting** — detailed statistical reports per tree node with CSV/Excel/Markdown export
- **Multi-format Model Export** — PMML, JSON, and Python code generation for deployment
- **Enhanced Split Editing** — interactive dialog for manual split threshold adjustment with preview
- **Surrogate Splits** — missing value handling via surrogate variable mechanism
- **Tree Pruning** — cost-complexity and minimum impurity decrease pruning algorithms
- **Performance Evaluation Dialog** — lift charts, Lorenz curves, and confusion matrix visualisation
- **Formula Editor** — derived variable creation with expression builder
- **Cloud & Database Import** — connectors for cloud data sources and SQL databases
- **Light/Dark Theme** — full theme system with persistent user preference
- **Memory Monitoring** — real-time memory usage display with configurable warning thresholds
- **Large Dataset Support** — optimised handling of datasets up to 800 MB / 400K records with chunk processing
- **Platform Launchers** — one-click startup scripts for Windows (Anaconda) and Ubuntu/Linux

### Changed
- Application version bumped to 1.0 (stable) from 0.50 (beta)
- Workflow execution engine refactored for reliable topological sort and error propagation
- Data loader optimised for memory efficiency with automatic dtype downcast

### Fixed
- Split statistics calculation corrected for edge cases with uniform node distributions
- Negative information gain edge case resolved in split finder
- Main window state restoration on project reload for multi-window layouts

---

## [0.50] - 2025-08-23

### Added
- Project structure and documentation
- Comprehensive .gitignore file for Python projects
- Platform-specific setup guides for Windows (Anaconda) and Ubuntu
- Comprehensive USER_MANUAL.md with step-by-step instructions


### Changed
- Updated Python requirement from 3.8+ to 3.12+ for optimal performance


### Removed
- Bundled Python environment (moved to development artifacts)
- Development logs and temporary files


---

## Version Naming Convention

- **Major versions** (X.0.0): Significant architectural changes or major new features
- **Minor versions** (X.Y.0): New features, enhancements, and improvements
- **Patch versions** (X.Y.Z): Bug fixes, minor improvements, and maintenance updates

## Release Notes

Each release includes:
- **Added**: New features and capabilities
- **Changed**: Modifications to existing functionality  
- **Fixed**: Bug fixes and issue resolutions
- **Removed**: Deprecated or removed features
- **Enhanced**: Performance improvements and optimizations

For detailed technical information about each release, see the commit history and release tags in the repository.