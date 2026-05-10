"""Define the supervised target and create a modeling-ready dataset.

This script is limited to step 7 of the project. It defines `response` as the
supervised learning target, documents leakage exclusions, and creates a readable
modeling dataset for the next step. It does not train models, split data,
scale values, normalize values, or one-hot encode categorical variables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLUSTERED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customer_clustered.csv"
MODELING_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_data.csv"
TARGET_SUMMARY_PATH = PROJECT_ROOT / "reports" / "target_definition_summary.csv"

TARGET_COLUMN = "response"

CATEGORICAL_FEATURES = [
    "education",
    "marital_status",
    "cluster",
]

NUMERIC_FEATURES = [
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
    "total_campaign_acceptances",
    "any_campaign_acceptance",
]

SELECTED_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

LEAKAGE_EXCLUSIONS = [
    "id",
    "response",
    "dt_customer",
]

REDUNDANT_OR_UNUSED_EXCLUSIONS = [
    "year_birth",
    "kidhome",
    "teenhome",
    "mnt_wines",
    "mnt_fruits",
    "mnt_meat_products",
    "mnt_fish_products",
    "mnt_sweet_products",
    "mnt_gold_prods",
    "num_deals_purchases",
    "num_web_purchases",
    "num_catalog_purchases",
    "num_store_purchases",
    "z_cost_contact",
    "z_revenue",
    "accepted_cmp1",
    "accepted_cmp2",
    "accepted_cmp3",
    "accepted_cmp4",
    "accepted_cmp5",
]

LEAKAGE_NOTES = (
    "Excluded id, response, and raw dt_customer. Response is the target and is "
    "not used as a feature. Raw dates are excluded in favor of tenure features. "
    "Individual accepted_cmp1 through accepted_cmp5 columns are excluded to keep "
    "the feature set compact and avoid overly campaign-specific indicators."
)

MODELING_ASSUMPTIONS = (
    "The target predicts response to the most recent campaign. Summarized "
    "historical campaign behavior, total_campaign_acceptances and "
    "any_campaign_acceptance, is allowed because it represents prior campaign "
    "history rather than the current response outcome."
)


def load_clustered_data(csv_path: Path = CLUSTERED_DATA_PATH) -> pd.DataFrame:
    """Load the clustered customer dataset."""
    return pd.read_csv(csv_path, parse_dates=["dt_customer"])


def define_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.Series:
    """Return the supervised target after validating that it is binary."""
    if target_column not in df.columns:
        raise KeyError(f"Target column not found: {target_column}")

    target = df[target_column]
    unique_values = set(target.dropna().unique().tolist())
    if unique_values != {0, 1}:
        raise ValueError(f"Expected binary target values {{0, 1}}, got {sorted(unique_values)}")

    return target


def select_modeling_features() -> list[str]:
    """Return the selected feature names for the modeling dataset."""
    return SELECTED_FEATURES.copy()


def get_excluded_features() -> list[str]:
    """Return columns intentionally excluded from the modeling features."""
    return LEAKAGE_EXCLUSIONS + REDUNDANT_OR_UNUSED_EXCLUSIONS


def check_leakage_exclusions(
    selected_features: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
) -> None:
    """Validate that direct leakage columns are absent from selected features."""
    if selected_features is None:
        selected_features = select_modeling_features()

    direct_leakage_columns = {"id", target_column, "dt_customer"}
    leaked = sorted(direct_leakage_columns.intersection(selected_features))
    if leaked:
        raise ValueError(f"Leakage columns included as modeling features: {leaked}")


def create_modeling_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create the modeling dataset with selected features and target."""
    target = define_target(df)
    selected_features = select_modeling_features()
    missing_features = sorted(set(selected_features) - set(df.columns))
    if missing_features:
        raise KeyError(f"Selected features are missing from input data: {missing_features}")

    check_leakage_exclusions(selected_features)
    modeling_df = df[selected_features].copy()
    modeling_df[TARGET_COLUMN] = target

    return modeling_df


def validate_modeling_dataset(modeling_df: pd.DataFrame) -> dict[str, int]:
    """Return validation counts for the modeling dataset."""
    numeric_df = modeling_df.select_dtypes(include=[np.number])
    return {
        "rows": modeling_df.shape[0],
        "columns": modeling_df.shape[1],
        "missing_values": int(modeling_df.isna().sum().sum()),
        "infinite_values": int(np.isinf(numeric_df.to_numpy()).sum()),
        "feature_count": modeling_df.shape[1] - 1,
        "categorical_feature_count": len(CATEGORICAL_FEATURES),
        "numeric_feature_count": len(NUMERIC_FEATURES),
    }


def summarize_target(target: pd.Series) -> dict[str, float | int | str]:
    """Create target balance and baseline metrics."""
    counts = target.value_counts().sort_index()
    total_rows = int(len(target))
    positive_count = int(counts.get(1, 0))
    negative_count = int(counts.get(0, 0))
    positive_rate = positive_count / total_rows
    negative_rate = negative_count / total_rows
    baseline_accuracy = max(positive_rate, negative_rate)

    return {
        "target_name": TARGET_COLUMN,
        "target_positive_class_meaning": (
            "Customer accepted/responded to the most recent marketing campaign"
        ),
        "total_rows": total_rows,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": positive_rate,
        "negative_rate": negative_rate,
        "baseline_majority_class_accuracy": baseline_accuracy,
    }


def create_target_summary(modeling_df: pd.DataFrame) -> pd.DataFrame:
    """Create the one-row target definition summary table."""
    target_summary = summarize_target(modeling_df[TARGET_COLUMN])
    validation = validate_modeling_dataset(modeling_df)

    summary = {
        **target_summary,
        "selected_feature_count": validation["feature_count"],
        "categorical_feature_count": validation["categorical_feature_count"],
        "numeric_feature_count": validation["numeric_feature_count"],
        "leakage_notes": LEAKAGE_NOTES,
        "modeling_assumptions": MODELING_ASSUMPTIONS,
    }
    return pd.DataFrame([summary])


def save_modeling_data(
    modeling_df: pd.DataFrame,
    output_path: Path = MODELING_DATA_PATH,
) -> None:
    """Save the modeling dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    modeling_df.to_csv(output_path, index=False)


def save_target_summary(
    summary_df: pd.DataFrame,
    output_path: Path = TARGET_SUMMARY_PATH,
) -> None:
    """Save the target definition summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)


def main() -> None:
    """Create modeling_data.csv and target_definition_summary.csv."""
    clustered_df = load_clustered_data()
    modeling_df = create_modeling_dataset(clustered_df)
    summary_df = create_target_summary(modeling_df)

    save_modeling_data(modeling_df)
    save_target_summary(summary_df)

    target_counts = modeling_df[TARGET_COLUMN].value_counts().sort_index()
    target_pct = modeling_df[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100
    validation = validate_modeling_dataset(modeling_df)

    print("Target definition and modeling dataset summary")
    print("=" * 80)
    print(f"Input shape: {clustered_df.shape}")
    print(f"Modeling dataset shape: {modeling_df.shape}")
    print("\nTarget class counts:")
    print(target_counts.to_string())
    print("\nTarget class percentages:")
    print(target_pct.round(2).to_string())
    print(
        "\nBaseline majority-class accuracy: "
        f"{summary_df.loc[0, 'baseline_majority_class_accuracy']:.4f}"
    )
    print("\nSelected features:")
    for feature in select_modeling_features():
        print(f"- {feature}")
    print("\nExcluded columns:")
    for feature in get_excluded_features():
        print(f"- {feature}")
    print("\nValidation:")
    for key, value in validation.items():
        print(f"{key}: {value}")
    print(f"\nSaved modeling data to: {MODELING_DATA_PATH}")
    print(f"Saved target summary to: {TARGET_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
