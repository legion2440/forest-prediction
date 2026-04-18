"""Model selection workflow for the required forest cover classifiers."""

from __future__ import annotations

import time
from dataclasses import dataclass

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocessing_feature_engineering import (
    CV_SPLITS,
    DATA_DIR,
    HOLDOUT_TEST_SIZE,
    PLOTS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    TARGET_COLUMN,
    build_model_pipeline,
    ensure_output_directories,
    load_dataset,
    split_features_target,
)

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


@dataclass(frozen=True)
class ModelConfig:
    """Configuration bundle for one required model family."""

    name: str
    estimator: object
    scale_numeric: bool
    param_grid: dict[str, list[object]]


def get_model_configs() -> list[ModelConfig]:
    """Return the fixed model families and search spaces used in the project."""
    return [
        ModelConfig(
            name="Gradient Boosting",
            estimator=GradientBoostingClassifier(random_state=RANDOM_STATE),
            scale_numeric=False,
            param_grid={
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3],
                "model__min_samples_leaf": [10],
            },
        ),
        ModelConfig(
            name="KNN",
            estimator=KNeighborsClassifier(),
            scale_numeric=True,
            param_grid={
                "model__n_neighbors": [9, 21],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
        ),
        ModelConfig(
            name="Random Forest",
            estimator=RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            scale_numeric=False,
            param_grid={
                "model__n_estimators": [200],
                "model__max_depth": [12, 16],
                "model__min_samples_leaf": [2, 4],
                "model__max_features": ["sqrt"],
            },
        ),
        ModelConfig(
            name="SVM",
            estimator=LinearSVC(
                random_state=RANDOM_STATE,
                dual=False,
                max_iter=5000,
            ),
            scale_numeric=True,
            param_grid={
                "model__C": [0.1, 1.0, 3.0],
            },
        ),
        ModelConfig(
            name="Logistic Regression",
            estimator=LogisticRegression(
                random_state=RANDOM_STATE,
                solver="lbfgs",
                max_iter=2000,
            ),
            scale_numeric=True,
            param_grid={
                "model__C": [0.1, 1.0, 3.0],
            },
        ),
    ]


def build_search(config: ModelConfig, cv: StratifiedKFold) -> GridSearchCV:
    """Create a grid search that evaluates one model family inside its full pipeline."""
    pipeline = build_model_pipeline(
        estimator=config.estimator,
        scale_numeric=config.scale_numeric,
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid=config.param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
        error_score="raise",
    )


def create_confusion_matrix_dataframe(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> pd.DataFrame:
    """Return the confusion matrix as a labeled DataFrame for inspection."""
    labels = sorted(pd.unique(pd.concat([y_true, pd.Series(y_pred)])))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    dataframe = pd.DataFrame(matrix, index=labels, columns=labels)
    dataframe.index.name = "True label"
    dataframe.columns.name = "Predicted label"
    return dataframe


def plot_confusion_matrix(confusion_df: pd.DataFrame) -> None:
    """Save the confusion matrix heatmap required by the subject."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix for the Best Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_learning_curve_for_best_model(
    estimator: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> None:
    """Plot the learning curve for the selected best pipeline."""
    # Clone the estimator so the plotting step does not mutate the fitted best model.
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=clone(estimator),
        X=X_train,
        y=y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    learning_curve_df = pd.DataFrame(
        {
            "train_size": train_sizes,
            "train_mean": train_scores.mean(axis=1),
            "train_std": train_scores.std(axis=1),
            "validation_mean": validation_scores.mean(axis=1),
            "validation_std": validation_scores.std(axis=1),
        }
    )

    plt.figure(figsize=(9, 6))
    plt.plot(
        learning_curve_df["train_size"],
        learning_curve_df["train_mean"],
        marker="o",
        label="Training accuracy",
    )
    plt.plot(
        learning_curve_df["train_size"],
        learning_curve_df["validation_mean"],
        marker="o",
        label="Validation accuracy",
    )
    plt.fill_between(
        learning_curve_df["train_size"],
        learning_curve_df["train_mean"] - learning_curve_df["train_std"],
        learning_curve_df["train_mean"] + learning_curve_df["train_std"],
        alpha=0.15,
    )
    plt.fill_between(
        learning_curve_df["train_size"],
        learning_curve_df["validation_mean"] - learning_curve_df["validation_std"],
        learning_curve_df["validation_mean"] + learning_curve_df["validation_std"],
        alpha=0.15,
    )
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve for the Best Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "learning_curve_best_model.png", dpi=200, bbox_inches="tight")
    plt.close()


def run_model_selection() -> dict[str, object]:
    """Run the full training, selection, evaluation, and model export workflow."""
    ensure_output_directories()

    # Split train.csv into the feature matrix and target before any fitting step.
    dataset = load_dataset(DATA_DIR / "train.csv")
    X_full, y_full = split_features_target(dataset, target_column=TARGET_COLUMN)

    # This is the internal holdout required by the subject: it comes from train.csv
    # and is used only after model selection on Train(1).
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
        X_full,
        y_full,
        test_size=HOLDOUT_TEST_SIZE,
        stratify=y_full,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Each required model family gets its own grid search on Train(1), evaluated
    # with the same stratified CV setup so the comparison stays reproducible.
    summaries: list[dict[str, object]] = []
    fitted_searches: list[tuple[ModelConfig, GridSearchCV]] = []

    for config in get_model_configs():
        start_time = time.time()
        search = build_search(config=config, cv=cv)
        search.fit(X_train_1, y_train_1)
        elapsed_seconds = time.time() - start_time

        holdout_predictions = search.predict(X_test_1)
        holdout_accuracy = accuracy_score(y_test_1, holdout_predictions)

        summaries.append(
            {
                "model": config.name,
                "best_cv_accuracy": search.best_score_,
                "holdout_test_1_accuracy": holdout_accuracy,
                "best_params": search.best_params_,
                "fit_time_seconds": elapsed_seconds,
            }
        )
        fitted_searches.append((config, search))

    summary_df = (
        pd.DataFrame(summaries)
        .sort_values(by="best_cv_accuracy", ascending=False)
        .reset_index(drop=True)
    )
    print("Model selection summary:")
    print(summary_df.to_string(index=False))

    # Select the winner by cross-validation accuracy, then report its performance
    # on the internal holdout Test(1).
    best_row = summary_df.iloc[0]
    best_config, best_search = next(
        (config, search)
        for config, search in fitted_searches
        if config.name == best_row["model"]
    )

    best_holdout_predictions = best_search.predict(X_test_1)
    confusion_df = create_confusion_matrix_dataframe(y_test_1, best_holdout_predictions)
    print("\nConfusion matrix DataFrame:")
    print(confusion_df)
    plot_confusion_matrix(confusion_df)
    plot_learning_curve_for_best_model(best_search.best_estimator_, X_train_1, y_train_1, cv)

    # Refit the selected full pipeline on the whole training file (0) before
    # exporting it. Saving the full pipeline preserves feature engineering and
    # preprocessing for predict.py.
    final_model = clone(best_search.best_estimator_)
    final_model.fit(X_full, y_full)
    train_accuracy_full = accuracy_score(y_full, final_model.predict(X_full))
    model_path = RESULTS_DIR / "best_model.pkl"
    joblib.dump(final_model, model_path)

    print(f"\nBest model: {best_config.name}")
    print(f"Best CV accuracy: {best_search.best_score_:.4f}")
    print(f"Holdout Test(1) accuracy: {accuracy_score(y_test_1, best_holdout_predictions):.4f}")
    print(f"Train accuracy on train set (0): {train_accuracy_full:.4f}")
    print(f"Saved model to: {model_path}")

    return {
        "summary_df": summary_df,
        "best_model_name": best_config.name,
        "best_cv_accuracy": best_search.best_score_,
        "holdout_accuracy": accuracy_score(y_test_1, best_holdout_predictions),
        "train_accuracy_full": train_accuracy_full,
        "best_params": best_search.best_params_,
        "confusion_df": confusion_df,
    }


def main() -> None:
    """Execute the model selection workflow as a script."""
    run_model_selection()


if __name__ == "__main__":
    main()
