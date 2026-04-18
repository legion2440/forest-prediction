"""Reusable data loading, feature engineering, and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RANDOM_STATE = 42
HOLDOUT_TEST_SIZE = 0.2
CV_SPLITS = 5
TARGET_COLUMN = "Cover_Type"

NUMERIC_FEATURE_COLUMNS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]

WILDERNESS_COLUMNS = [f"Wilderness_Area{i}" for i in range(1, 5)]
SOIL_TYPE_COLUMNS = [f"Soil_Type{i}" for i in range(1, 41)]
BINARY_FEATURE_COLUMNS = WILDERNESS_COLUMNS + SOIL_TYPE_COLUMNS
ENGINEERED_FEATURE_COLUMNS = [
    "Distance_To_Hydrology",
    "Fire_Road_Distance_Diff",
]


def resolve_path(path_like: str | Path) -> Path:
    """Resolve a dataset path relative to the project data directory when needed."""
    path = Path(path_like)
    if path.exists():
        return path
    return DATA_DIR / path


def load_dataset(path_like: str | Path) -> pd.DataFrame:
    """Load a CSV dataset from an explicit path or from ``data/``."""
    dataset_path = resolve_path(path_like)
    return pd.read_csv(dataset_path)


def split_features_target(
    dataframe: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a labeled dataset into the feature matrix and target series."""
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")

    # Drop the target explicitly so training code cannot accidentally keep it in X.
    features = dataframe.drop(columns=[target_column]).copy()
    target = dataframe[target_column].copy()
    return features, target


def create_engineered_features(features: pd.DataFrame) -> pd.DataFrame:
    """Create the engineered features shared by training and final inference."""
    required_columns = set(NUMERIC_FEATURE_COLUMNS + BINARY_FEATURE_COLUMNS)
    missing_columns = sorted(required_columns.difference(features.columns))
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {missing_columns}")

    transformed = features.copy()
    # Keep feature engineering in one place so model selection and predict.py use
    # exactly the same derived columns.
    transformed["Distance_To_Hydrology"] = np.sqrt(
        transformed["Horizontal_Distance_To_Hydrology"] ** 2
        + transformed["Vertical_Distance_To_Hydrology"] ** 2
    )
    transformed["Fire_Road_Distance_Diff"] = (
        transformed["Horizontal_Distance_To_Fire_Points"]
        - transformed["Horizontal_Distance_To_Roadways"]
    )
    return transformed


def get_model_feature_columns() -> list[str]:
    """Return the full feature list expected by the model pipelines."""
    return NUMERIC_FEATURE_COLUMNS + BINARY_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS


def get_scaled_feature_columns() -> list[str]:
    """Return only the continuous features that may require scaling."""
    return NUMERIC_FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS


class ForestFeatureEngineer(BaseEstimator, TransformerMixin):
    """scikit-learn compatible transformer for deterministic feature engineering."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ForestFeatureEngineer":
        """Store input feature names for downstream pipeline introspection."""
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the shared engineered-feature logic to the incoming data frame."""
        return create_engineered_features(X)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Return feature names after the engineered columns are appended."""
        if input_features is None:
            input_features = list(getattr(self, "feature_names_in_", get_model_feature_columns()))
        return np.asarray(list(input_features) + ENGINEERED_FEATURE_COLUMNS, dtype=object)


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Build the column-wise preprocessing step used inside each model pipeline."""
    if scale_numeric:
        # Scale only the continuous variables for scale-sensitive models while
        # keeping the binary indicator columns untouched.
        transformers = [
            ("scaled_numeric", StandardScaler(), get_scaled_feature_columns()),
            ("binary_passthrough", "passthrough", BINARY_FEATURE_COLUMNS),
        ]
    else:
        transformers = [("all_features", "passthrough", get_model_feature_columns())]

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(estimator: BaseEstimator, scale_numeric: bool) -> Pipeline:
    """Assemble the full modeling pipeline, including preprocessing."""
    return Pipeline(
        steps=[
            # Keeping feature engineering and preprocessing inside the pipeline
            # ensures that cross-validation runs without data leakage.
            ("feature_engineering", ForestFeatureEngineer()),
            ("preprocessing", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )


def ensure_output_directories() -> None:
    """Create the output folders expected by training and prediction scripts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
