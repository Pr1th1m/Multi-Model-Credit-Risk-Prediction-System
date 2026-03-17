# Multi-Model Credit Risk Prediction

A machine learning pipeline for credit risk prediction using various models (Logistic Regression, Decision Tree, RandomForest, XGBoost, LightGBM, CatBoost, KNN, and SVM).

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

## Results

Here are the evaluation metrics for the models implemented in this pipeline, sorted by AUC-ROC:

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| --- | --- | --- | --- | --- | --- |
| **SVM** | 0.730 | 0.9135 | 0.6786 | 0.7787 | 0.8470 |
| **Logistic Regression** | 0.760 | 0.8833 | 0.7571 | 0.8154 | 0.8260 |
| **Random Forest** | 0.775 | 0.8276 | 0.8571 | 0.8421 | 0.8243 |
| **CatBoost** | 0.785 | 0.8345 | 0.8643 | 0.8491 | 0.8152 |
| **XGBoost** | 0.750 | 0.7394 | 0.9929 | 0.8476 | 0.8088 |
| **LightGBM** | 0.760 | 0.8239 | 0.8357 | 0.8298 | 0.8040 |
| **KNN** | 0.735 | 0.7636 | 0.9000 | 0.8262 | 0.7545 |
| **Decision Tree** | 0.680 | 0.8455 | 0.6643 | 0.7440 | 0.6859 |

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
