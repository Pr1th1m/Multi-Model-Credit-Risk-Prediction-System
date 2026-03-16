# AI Agents Instruction File

This file serves as a reference for AI coding agents operating in this workspace.

## System Context
- **OS**: Windows
- **Project Type**: Python Machine Learning Pipeline

## Workflow & Coding Standards
- **Source of Truth**: The `src/` directory contains all production code. `notebooks/` is for experimentation only.
- **Dependencies**: Any new packages added must be reflected in `requirements.txt`.
- **Experiment Tracking**: We use `mlflow` to track all models. Do not commit the `mlruns/` directory. If a new metric needs tracking, edit the `mlflow.start_run()` block in `src/pipeline.py`.
- **Imports**: All custom internal imports should use absolute paths based on the project root (e.g., `import src.data_loader` or `from src.evaluation import ...`). Do not use relative imports like `from . import config`.
- **Execution**: To run tests or the main script, ensure the working directory is the project root `c:\Users\prath\MultilModel`.

If you encounter an `ImportError` regarding the `src` module in tools/IDEs, it has been resolved by placing `PYTHONPATH=.` in `.env` and setting `python.analysis.extraPaths` to `.` in `.vscode/settings.json`.
