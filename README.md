# Forest Prediction

This project implements a reproducible forest cover type classification pipeline for the `forest-prediction` subject. The workflow follows the required split strategy, evaluates only the five requested model families, saves the best full pipeline as a pickle file, and uses that saved pipeline to score the external `data/test.csv`.

## Goal

The goal is to predict `Cover_Type` from cartographic variables, while keeping model selection honest and reproducible:

- `train.csv` is split into Train(1) and Test(1) with an 80/20 stratified split.
- model selection is performed only on Train(1) with 5-fold stratified cross-validation.
- the saved artifact is the full best pipeline, not a bare estimator.
- `data/test.csv` is used only in `scripts/predict.py`.

## Repository Structure

```text
project/
  README.md
  requirements.txt
  environment.yml
  data/
    train.csv
    test.csv
    covtype.info
  notebook/
    EDA.ipynb
  scripts/
    preprocessing_feature_engineering.py
    model_selection.py
    predict.py
  results/
    plots/
      confusion_matrix_heatmap.png
      learning_curve_best_model.png
    test_predictions.csv
    best_model.pkl
```

## Environment Setup

### `venv` + `pip`

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### `conda`

```bash
conda env create -f environment.yml
conda activate forest-prediction
```

## How To Run

```bash
python scripts/model_selection.py
python scripts/predict.py
jupyter notebook notebook/EDA.ipynb
```

## Python Files Summary

### `scripts/preprocessing_feature_engineering.py`

- centralizes paths, constants, dataset loading, and `X / y` splitting
- defines the reusable feature engineering step used in both training and prediction
- builds the preprocessing logic used inside every model pipeline

### `scripts/model_selection.py`

- loads `data/train.csv`
- creates the fixed 80/20 stratified Train(1) / Test(1) split
- runs separate `GridSearchCV` pipelines for the five required model families:
  - Gradient Boosting
  - KNN
  - Random Forest
  - SVM (`LinearSVC`)
  - Logistic Regression
- selects the best model by cross-validation accuracy
- evaluates the selected model on Test(1)
- builds the confusion matrix DataFrame and saves the required plots
- refits the best pipeline on the full train set (0) and saves it to `results/best_model.pkl`

### `scripts/predict.py`

- loads the saved best pipeline from `results/best_model.pkl`
- loads `data/test.csv`
- applies the same preprocessing and feature engineering through the saved pipeline
- computes external test accuracy
- saves predictions to `results/test_predictions.csv`

## Feature Engineering

The project uses two explicit engineered features and keeps feature creation intentionally limited:

1. `Distance_To_Hydrology`
   `sqrt(Horizontal_Distance_To_Hydrology^2 + Vertical_Distance_To_Hydrology^2)`
2. `Fire_Road_Distance_Diff`
   `Horizontal_Distance_To_Fire_Points - Horizontal_Distance_To_Roadways`

These features are created inside the reusable preprocessing module and are applied identically during model selection and final prediction. No target leakage is used.

## Model Selection Approach

- target column: `Cover_Type`
- Train(1) / Test(1) split: stratified 80/20 with `random_state=42`
- cross-validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- scoring metric for all model comparisons: accuracy
- preprocessing is embedded in every pipeline to keep model selection leakage-free
- scaling is applied only for KNN, SVM, and Logistic Regression pipelines
- the best model is chosen by CV accuracy and then evaluated on Test(1)

## Metrics

- Best model family: `Random Forest`
- Train accuracy on train set (0): `0.9631`
- Final test accuracy on external test set (0): `0.7032`

## Artifacts

Running `python scripts/model_selection.py` creates:

- `results/best_model.pkl`
- `results/plots/confusion_matrix_heatmap.png`
- `results/plots/learning_curve_best_model.png`

Running `python scripts/predict.py` creates:

- `results/test_predictions.csv`
