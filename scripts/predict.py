"""Load the saved pipeline, score it on the external test file, and export predictions."""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

from preprocessing_feature_engineering import (
    DATA_DIR,
    RESULTS_DIR,
    TARGET_COLUMN,
    ensure_output_directories,
    load_dataset,
    split_features_target,
)


def main() -> None:
    """Run final inference on ``data/test.csv`` using the saved full pipeline."""
    ensure_output_directories()

    model_path = RESULTS_DIR / "best_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            "The trained model was not found. Run scripts/model_selection.py first."
        )

    # Unlike the internal holdout in model_selection.py, this is the external
    # test file (0) used only after the best pipeline has already been chosen.
    model = joblib.load(model_path)
    external_test_df = load_dataset(DATA_DIR / "test.csv")
    X_test_0, y_test_0 = split_features_target(external_test_df, target_column=TARGET_COLUMN)

    # The saved artifact is the full pipeline, so prediction automatically reuses
    # the same feature engineering and preprocessing as during training.
    predictions = model.predict(X_test_0)
    accuracy = accuracy_score(y_test_0, predictions)

    predictions_df = pd.DataFrame(
        {
            "row_id": range(len(predictions)),
            "predicted_cover_type": predictions,
        }
    )
    predictions_path = RESULTS_DIR / "test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    print(f"Final external test accuracy: {accuracy:.4f}")
    print(f"Saved predictions to: {predictions_path}")


if __name__ == "__main__":
    main()
