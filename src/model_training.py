"""Train and evaluate supervised response prediction models.

This script is limited to step 8 of the project. It compares controlled feature
sets to test whether cluster labels add predictive value, and whether historical
campaign behavior further improves prediction. It does not redefine the target,
perform clustering, tune models extensively, or deploy anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELING_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_data.csv"
MODEL_EVALUATION_PATH = PROJECT_ROOT / "reports" / "model_evaluation_summary.csv"
FEATURE_SET_COMPARISON_PATH = PROJECT_ROOT / "reports" / "feature_set_comparison_summary.csv"

TARGET_COLUMN = "response"
RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURE_SET_A = [
    "education",
    "marital_status",
    "age",
    "income",
    "children_total",
    "has_children",
    "household_size",
    "customer_tenure_days",
    "customer_tenure_years",
    "recency",
    "complain",
    "total_spending",
    "wine_share",
    "fruits_share",
    "meat_share",
    "fish_share",
    "sweets_share",
    "gold_share",
    "luxury_spending_ratio",
    "spending_per_tenure_year",
    "total_purchases",
    "total_store_web_catalog_purchases",
    "web_purchase_share",
    "store_purchase_share",
    "catalog_purchase_share",
    "deal_purchase_share",
    "average_spend_per_purchase",
    "num_web_visits_month",
]

FEATURE_SET_B = FEATURE_SET_A + ["cluster"]
FEATURE_SET_C = FEATURE_SET_B + ["total_campaign_acceptances", "any_campaign_acceptance"]

FEATURE_SETS = {
    "A_baseline_customer_features": FEATURE_SET_A,
    "B_baseline_plus_cluster": FEATURE_SET_B,
    "C_full_with_campaign_history": FEATURE_SET_C,
}

MODEL_NOTES = {
    "Dummy Majority": "Majority-class baseline.",
    "Logistic Regression": "Scaled linear classifier without class weighting.",
    "Logistic Regression Balanced": "Scaled linear classifier with class_weight='balanced'.",
    "Random Forest": "Tree ensemble without class weighting.",
    "Random Forest Balanced": "Tree ensemble with class_weight='balanced'.",
    "Gradient Boosting": "Gradient boosting classifier; no class weighting.",
}


@dataclass(frozen=True)
class SplitData:
    """Container for the shared stratified train/test split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_modeling_data(csv_path: Path = MODELING_DATA_PATH) -> pd.DataFrame:
    """Load modeling_data.csv."""
    return pd.read_csv(csv_path)


def define_feature_sets() -> dict[str, list[str]]:
    """Return the controlled feature sets used in the experiment."""
    return {name: features.copy() for name, features in FEATURE_SETS.items()}


def validate_modeling_data(df: pd.DataFrame) -> dict[str, int]:
    """Validate shape, missing values, infinite values, and target availability."""
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column not found: {TARGET_COLUMN}")

    target_values = set(df[TARGET_COLUMN].dropna().unique().tolist())
    if target_values != {0, 1}:
        raise ValueError(f"Expected binary target values {{0, 1}}, got {sorted(target_values)}")

    numeric_df = df.select_dtypes(include=[np.number])
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "infinite_values": int(np.isinf(numeric_df.to_numpy()).sum()),
        "positive_count": int((df[TARGET_COLUMN] == 1).sum()),
        "negative_count": int((df[TARGET_COLUMN] == 0).sum()),
    }


def make_train_test_split(df: pd.DataFrame) -> SplitData:
    """Create one shared stratified train/test split for all feature sets."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def identify_feature_types(
    df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[list[str], list[str]]:
    """Identify categorical and numerical features for a feature set."""
    categorical_features = [
        feature
        for feature in feature_names
        if is_string_dtype(df[feature]) or feature == "cluster"
    ]
    numerical_features = [feature for feature in feature_names if feature not in categorical_features]
    return numerical_features, categorical_features


def create_preprocessing_pipeline(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Create preprocessing that is fit only within each training pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_definitions() -> dict[str, BaseEstimator]:
    """Return the small model set used for every feature-set comparison."""
    return {
        "Dummy Majority": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Logistic Regression Balanced": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            min_samples_leaf=5,
            n_jobs=-1,
        ),
        "Random Forest Balanced": RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def build_model_pipeline(
    model: BaseEstimator,
    numerical_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Create a preprocessing + model pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessing_pipeline(numerical_features, categorical_features)),
            ("model", clone(model)),
        ]
    )


def get_positive_scores(pipeline: Pipeline, X_test: pd.DataFrame) -> np.ndarray:
    """Return positive-class scores from predict_proba or decision_function."""
    model = pipeline.named_steps["model"]

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)
        classes = list(model.classes_)
        if 1 in classes:
            return probabilities[:, classes.index(1)]
        return np.zeros(len(X_test))

    if hasattr(pipeline, "decision_function"):
        return pipeline.decision_function(X_test)

    return pipeline.predict(X_test)


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate predictions with class-imbalance-aware metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_positive": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_positive": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_positive": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def train_and_evaluate_models(
    df: pd.DataFrame,
    split_data: SplitData,
) -> tuple[pd.DataFrame, dict[tuple[str, str], Pipeline]]:
    """Train all model and feature-set combinations."""
    results: list[dict[str, float | int | str]] = []
    fitted_pipelines: dict[tuple[str, str], Pipeline] = {}
    feature_sets = define_feature_sets()
    model_definitions = build_model_definitions()

    for feature_set_name, feature_names in feature_sets.items():
        numerical_features, categorical_features = identify_feature_types(df, feature_names)
        X_train_subset = split_data.X_train[feature_names]
        X_test_subset = split_data.X_test[feature_names]

        for model_name, model in model_definitions.items():
            pipeline = build_model_pipeline(model, numerical_features, categorical_features)
            pipeline.fit(X_train_subset, split_data.y_train)

            y_pred = pipeline.predict(X_test_subset)
            y_score = get_positive_scores(pipeline, X_test_subset)
            metrics = evaluate_predictions(split_data.y_test, y_pred, y_score)

            results.append(
                {
                    "feature_set": feature_set_name,
                    "model_name": model_name,
                    **metrics,
                    "notes": MODEL_NOTES[model_name],
                }
            )
            fitted_pipelines[(feature_set_name, model_name)] = pipeline

    evaluation_df = pd.DataFrame(results)
    metric_cols = [
        "accuracy",
        "precision_positive",
        "recall_positive",
        "f1_positive",
        "roc_auc",
        "pr_auc",
    ]
    evaluation_df[metric_cols] = evaluation_df[metric_cols].round(6)
    return evaluation_df, fitted_pipelines


def compare_feature_sets(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """Select the best PR-AUC model for each feature set and interpret changes."""
    best_rows = (
        evaluation_df.sort_values(
            ["feature_set", "pr_auc", "f1_positive", "roc_auc"],
            ascending=[True, False, False, False],
        )
        .groupby("feature_set", as_index=False)
        .head(1)
        .copy()
    )

    interpretations = {
        "A_baseline_customer_features": (
            "Baseline using customer demographics, relationship, spending, and purchase behavior."
        ),
        "B_baseline_plus_cluster": (
            "Tests whether the unsupervised cluster label improves performance beyond baseline features."
        ),
        "C_full_with_campaign_history": (
            "Adds summarized historical campaign behavior; expected to improve response prediction if prior campaign behavior is informative."
        ),
    }

    comparison = best_rows[
        [
            "feature_set",
            "model_name",
            "accuracy",
            "precision_positive",
            "recall_positive",
            "f1_positive",
            "roc_auc",
            "pr_auc",
        ]
    ].rename(columns={"model_name": "best_model_for_feature_set"})
    comparison["interpretation"] = comparison["feature_set"].map(interpretations)
    return comparison.reset_index(drop=True)


def select_best_model(
    evaluation_df: pd.DataFrame,
    feature_set: str | None = None,
) -> tuple[str, str]:
    """Return the best model by PR-AUC, then F1, then ROC-AUC."""
    subset = evaluation_df.copy()
    if feature_set is not None:
        subset = subset[subset["feature_set"] == feature_set]

    best = subset.sort_values(
        ["pr_auc", "f1_positive", "roc_auc"],
        ascending=[False, False, False],
    ).iloc[0]
    return str(best["feature_set"]), str(best["model_name"])


def get_transformed_feature_names(pipeline: Pipeline) -> np.ndarray:
    """Return readable post-preprocessing feature names."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out()


def get_model_interpretation_table(
    pipeline: Pipeline,
    top_n: int = 15,
) -> pd.DataFrame:
    """Extract feature importance or coefficients for a fitted pipeline."""
    feature_names = get_transformed_feature_names(pipeline)
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        if len(feature_names) < len(values):
            feature_names = np.append(
                feature_names,
                [f"unnamed_feature_{index}" for index in range(len(feature_names), len(values))],
            )
        feature_names = feature_names[: len(values)]
        table = pd.DataFrame({"feature": feature_names, "importance": values})
        return table.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    if hasattr(model, "coef_"):
        values = model.coef_[0]
        if len(feature_names) < len(values):
            feature_names = np.append(
                feature_names,
                [f"unnamed_feature_{index}" for index in range(len(feature_names), len(values))],
            )
        feature_names = feature_names[: len(values)]
        table = pd.DataFrame({"feature": feature_names, "coefficient": values})
        table["abs_coefficient"] = table["coefficient"].abs()
        return table.sort_values("abs_coefficient", ascending=False).head(top_n).reset_index(drop=True)

    return pd.DataFrame({"message": ["No built-in coefficient or feature importance available."]})


def create_classification_report_df(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Create a classification report dataframe for a fitted pipeline."""
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).T


def evaluate_thresholds(
    y_true: pd.Series,
    y_score: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Evaluate probability thresholds for the positive class."""
    if thresholds is None:
        thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]

    rows = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": threshold,
                "precision_positive": precision_score(y_true, y_pred, zero_division=0),
                "recall_positive": recall_score(y_true, y_pred, zero_division=0),
                "f1_positive": f1_score(y_true, y_pred, zero_division=0),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "true_negatives": int(tn),
            }
        )
    return pd.DataFrame(rows)


def get_curve_data(
    y_true: pd.Series,
    y_score: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ROC and precision-recall curve data."""
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    roc_df = pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr, "threshold": roc_thresholds})
    pr_df = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": np.append(pr_thresholds, np.nan),
        }
    )
    return roc_df, pr_df


def save_evaluation_outputs(
    evaluation_df: pd.DataFrame,
    feature_set_comparison_df: pd.DataFrame,
    evaluation_path: Path = MODEL_EVALUATION_PATH,
    comparison_path: Path = FEATURE_SET_COMPARISON_PATH,
) -> None:
    """Save detailed and feature-set-level evaluation summaries."""
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_df.to_csv(evaluation_path, index=False)
    feature_set_comparison_df.to_csv(comparison_path, index=False)


def run_training_experiment() -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], Pipeline], SplitData]:
    """Run the full model training and evaluation experiment."""
    df = load_modeling_data()
    split_data = make_train_test_split(df)
    evaluation_df, fitted_pipelines = train_and_evaluate_models(df, split_data)
    feature_set_comparison_df = compare_feature_sets(evaluation_df)
    save_evaluation_outputs(evaluation_df, feature_set_comparison_df)
    return evaluation_df, feature_set_comparison_df, fitted_pipelines, split_data


def main() -> None:
    """Train models, save summaries, and print a concise validation summary."""
    df = load_modeling_data()
    validation = validate_modeling_data(df)
    split_data = make_train_test_split(df)
    evaluation_df, fitted_pipelines = train_and_evaluate_models(df, split_data)
    comparison_df = compare_feature_sets(evaluation_df)
    save_evaluation_outputs(evaluation_df, comparison_df)

    best_feature_set, best_model_name = select_best_model(evaluation_df)
    best_pipeline = fitted_pipelines[(best_feature_set, best_model_name)]
    y_score = get_positive_scores(best_pipeline, split_data.X_test[FEATURE_SETS[best_feature_set]])
    threshold_df = evaluate_thresholds(split_data.y_test, y_score)

    print("Supervised model training summary")
    print("=" * 80)
    print(f"Modeling dataset shape: {df.shape}")
    print(f"Train size: {len(split_data.y_train)}")
    print(f"Test size: {len(split_data.y_test)}")
    print("\nValidation:")
    for key, value in validation.items():
        print(f"{key}: {value}")

    print("\nTrain target distribution:")
    print(split_data.y_train.value_counts().sort_index().to_string())
    print("\nTest target distribution:")
    print(split_data.y_test.value_counts().sort_index().to_string())

    print("\nDetailed evaluation:")
    print(evaluation_df.to_string(index=False))

    print("\nBest model by feature set:")
    print(comparison_df.to_string(index=False))

    print(f"\nBest overall by PR-AUC: {best_feature_set} / {best_model_name}")
    print("\nThreshold analysis for best overall model:")
    print(threshold_df.round(4).to_string(index=False))

    print(f"\nSaved detailed evaluation to: {MODEL_EVALUATION_PATH}")
    print(f"Saved feature-set comparison to: {FEATURE_SET_COMPARISON_PATH}")


if __name__ == "__main__":
    main()
