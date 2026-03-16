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
- `results/`: Directory for model outputs and evaluation reports (ignored in git).
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

Run the main pipeline:

```bash
python main.py
```
