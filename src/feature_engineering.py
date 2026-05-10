"""Create interpretable customer features for EDA, clustering, and modeling.

This script is intentionally limited to step 3 of the project. It adds
interpretable behavioral and demographic features, but does not scale values,
one-hot encode categories, create model targets, cluster customers, or train
models.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customer_clean.csv"
FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customer_features.csv"

SPENDING_COLUMNS = [
    "mnt_wines",
    "mnt_fruits",
    "mnt_meat_products",
    "mnt_fish_products",
    "mnt_sweet_products",
    "mnt_gold_prods",
]

PURCHASE_CHANNEL_COLUMNS = [
    "num_web_purchases",
    "num_catalog_purchases",
    "num_store_purchases",
]

DEAL_PURCHASE_COLUMN = "num_deals_purchases"
CAMPAIGN_COLUMNS = [f"accepted_cmp{i}" for i in range(1, 6)]

ENGINEERED_FEATURES = [
    "age",
    "customer_tenure_days",
    "customer_tenure_years",
    "children_total",
    "has_children",
    "household_size",
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
    "total_campaign_acceptances",
    "any_campaign_acceptance",
]


def load_clean_data(csv_path: Path = CLEAN_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned customer dataset and parse customer signup dates."""
    return pd.read_csv(csv_path, parse_dates=["dt_customer"])


def safe_divide(numerator: pd.Series, denominator: pd.Series | float | int) -> pd.Series:
    """Divide while returning 0 when the denominator is 0 or invalid."""
    result = numerator.div(denominator)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def estimate_household_size(marital_status: pd.Series, children_total: pd.Series) -> pd.Series:
    """Estimate household size from relationship status and children count.

    Married and Together records are treated as two-adult households. All other
    statuses are treated as one-adult households because the data does not give
    enough evidence to assume another adult is present.
    """
    two_adult_statuses = {"Married", "Together"}
    adult_count = marital_status.isin(two_adult_statuses).astype(int) + 1
    return adult_count + children_total


def add_demographic_features(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Add age, tenure, children, and estimated household-size features."""
    featured = df.copy()
    reference_year = reference_date.year

    # Age is measured at the latest customer date in the dataset so the value is
    # tied to the historical dataset rather than the current calendar year.
    featured["age"] = reference_year - featured["year_birth"]

    featured["customer_tenure_days"] = (reference_date - featured["dt_customer"]).dt.days
    featured["customer_tenure_years"] = featured["customer_tenure_days"] / 365.25

    featured["children_total"] = featured["kidhome"] + featured["teenhome"]
    featured["has_children"] = (featured["children_total"] > 0).astype(int)
    featured["household_size"] = estimate_household_size(
        featured["marital_status"],
        featured["children_total"],
    )

    return featured


def add_spending_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add product spending totals, shares, and safe rate features."""
    featured = df.copy()

    featured["total_spending"] = featured[SPENDING_COLUMNS].sum(axis=1)

    share_map = {
        "wine_share": "mnt_wines",
        "fruits_share": "mnt_fruits",
        "meat_share": "mnt_meat_products",
        "fish_share": "mnt_fish_products",
        "sweets_share": "mnt_sweet_products",
        "gold_share": "mnt_gold_prods",
    }
    for new_column, source_column in share_map.items():
        featured[new_column] = safe_divide(featured[source_column], featured["total_spending"])

    # Wine and gold are commonly interpreted as discretionary/luxury-oriented
    # categories in this dataset, so the ratio is useful for later segmentation.
    luxury_spending = featured["mnt_wines"] + featured["mnt_gold_prods"]
    featured["luxury_spending_ratio"] = safe_divide(luxury_spending, featured["total_spending"])

    # Customers with zero tenure days are protected by a minimum one-day
    # denominator. This avoids infinite annualized spending for the newest signup.
    tenure_years_safe = (featured["customer_tenure_days"].clip(lower=1)) / 365.25
    featured["spending_per_tenure_year"] = safe_divide(
        featured["total_spending"],
        tenure_years_safe,
    )

    return featured


def add_purchase_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add purchase-channel totals, shares, and average spending per purchase."""
    featured = df.copy()

    featured["total_store_web_catalog_purchases"] = featured[PURCHASE_CHANNEL_COLUMNS].sum(axis=1)

    # Deal purchases are kept in total_purchases because they are purchase
    # events, but channel-only totals are also retained to avoid ambiguity.
    featured["total_purchases"] = (
        featured["total_store_web_catalog_purchases"] + featured[DEAL_PURCHASE_COLUMN]
    )

    featured["web_purchase_share"] = safe_divide(
        featured["num_web_purchases"],
        featured["total_store_web_catalog_purchases"],
    )
    featured["store_purchase_share"] = safe_divide(
        featured["num_store_purchases"],
        featured["total_store_web_catalog_purchases"],
    )
    featured["catalog_purchase_share"] = safe_divide(
        featured["num_catalog_purchases"],
        featured["total_store_web_catalog_purchases"],
    )
    featured["deal_purchase_share"] = safe_divide(
        featured[DEAL_PURCHASE_COLUMN],
        featured["total_purchases"],
    )

    # Average spend uses the channel purchase total, not deal purchases, because
    # discounted purchases may overlap with web/catalog/store channels.
    featured["average_spend_per_purchase"] = safe_divide(
        featured["total_spending"],
        featured["total_store_web_catalog_purchases"],
    )

    return featured


def add_campaign_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add historical campaign acceptance counts without using response."""
    featured = df.copy()
    featured["total_campaign_acceptances"] = featured[CAMPAIGN_COLUMNS].sum(axis=1)
    featured["any_campaign_acceptance"] = (featured["total_campaign_acceptances"] > 0).astype(int)
    return featured


def create_customer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Create the full feature set from the cleaned customer dataset."""
    featured = df.copy()
    featured["dt_customer"] = pd.to_datetime(featured["dt_customer"], errors="raise")

    reference_date = featured["dt_customer"].max()

    featured = add_demographic_features(featured, reference_date)
    featured = add_spending_features(featured)
    featured = add_purchase_features(featured)
    featured = add_campaign_features(featured)

    return featured, reference_date


def validate_features(df: pd.DataFrame) -> dict[str, int]:
    """Return validation counts for missing and infinite engineered values."""
    numeric_df = df.select_dtypes(include=[np.number])
    infinite_count = int(np.isinf(numeric_df.to_numpy()).sum())

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "infinite_values": infinite_count,
    }


def save_feature_data(df: pd.DataFrame, output_path: Path = FEATURE_DATA_PATH) -> None:
    """Save the engineered feature dataset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    """Run the complete feature engineering workflow."""
    clean_df = load_clean_data()
    feature_df, reference_date = create_customer_features(clean_df)
    save_feature_data(feature_df)

    print("Feature engineering summary")
    print("=" * 80)
    print(f"Input shape: {clean_df.shape}")
    print(f"Output shape: {feature_df.shape}")
    print(f"Reference date: {reference_date.date()}")
    print("\nEngineered features:")
    for feature in ENGINEERED_FEATURES:
        print(f"- {feature}")

    print("\nValidation:")
    for key, value in validate_features(feature_df).items():
        print(f"{key}: {value}")

    print("\nExample engineered rows:")
    example_columns = [
        "id",
        "age",
        "customer_tenure_years",
        "children_total",
        "household_size",
        "total_spending",
        "total_purchases",
        "average_spend_per_purchase",
        "total_campaign_acceptances",
        "response",
    ]
    print(feature_df[example_columns].head())
    print(f"\nSaved engineered data to: {FEATURE_DATA_PATH}")


if __name__ == "__main__":
    main()
