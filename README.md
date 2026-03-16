# Multi-Model Credit Risk Prediction

A machine learning pipeline for credit risk prediction using various models (Logistic Regression, Decision Tree, RandomForest, XGBoost, LightGBM, KNN, and SVM).

## Project Structure

- `main.py`: Entry point for the pipeline.
- `src/`: Source code package.
  - `config.py`: Configuration settings.
  - `data_loader.py`: Data loading utilities.
  - `preprocessing.py`: Data preprocessing and feature engineering.
  - `models.py`: Model definitions and training wrappers.
  - `evaluation.py`: Model evaluation metrics and reporting.
  - `pipeline.py`: Main execution pipeline connecting data, models, and evaluation.
- `data/`: Directory for input datasets (ignored in git).
- `results/`: Directory for model outputs metrics (ignored in git).
- `mlruns/`: Directory created automatically to store MLflow artifacts and metrics.
- `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA).

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:
   - **Windows**: `.venv\Scripts\activate`
   - **Linux/Mac**: `source .venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main pipeline. This will train all models, log their hyperparameters and metrics using MLflow, and determine the best performing algorithm:

```bash
python main.py
```

### Viewing MLflow Dashboard

After running the pipeline, you can view the detailed model comparisons, metrics, and saved artifacts in the MLflow UI:

1. Open a terminal in the project root.
2. Run the MLflow UI server:
   ```bash
   mlflow ui
   ```
3. Open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).
