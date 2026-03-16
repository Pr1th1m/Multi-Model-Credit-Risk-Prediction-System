# AI Assistant Guidelines (Claude/Agents)

## Project Context
- **Name**: Multi-Model Credit Risk Pipeline
- **Domain**: Machine Learning / Credit Risk Prediction
- **Language**: Python 3
- **Primary Execution**: `python main.py`

## Development Rules
1. **Python Imports**: Imports are resolved from the project root. When adding or modifying code in `src/`, use absolute imports starting with `src.` (e.g., `from src.config import ...`).
2. **Environment**: This project uses a `.venv` folder for dependencies. The `.env` file specifies `PYTHONPATH=.` so tools like Pylance can resolve imports.
3. **Data & Results**: Do not commit datasets to `data/` or model outputs to `results/`. These directories are gitignored.
4. **Pipelines**: Modifications to the ML flow should be made modularly within `src/` and orchestrated in `src/pipeline.py`.
5. **Linting/Formatting**: Ensure code follows standard PEP 8 conventions.

## Common Tasks
- When asked to add a new model, update `src/models.py` and register it in `src/pipeline.py` or the appropriate configuration registry.
- When doing EDA, use the `notebooks/` directory rather than polluting the `src/` directory with exploration scripts.
