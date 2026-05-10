"""Load and clean the Customer Personality Analysis dataset.

This script is intentionally limited to step 2 of the project: data loading,
basic validation, and conservative cleaning. It does not create model features,
targets, clusters, or scaled data.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "marketing_campaign.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customer_clean.csv"

MAX_REASONABLE_AGE_AT_SIGNUP = 100


def detect_delimiter(csv_path: Path) -> str:
    """Detect the CSV delimiter, falling back to the known Kaggle tab format."""
    sample = csv_path.read_text(encoding="utf-8", errors="replace")[:4096]

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample else ""
        if "\t" in first_line:
            return "\t"
        if ";" in first_line:
            return ";"
        return ","


def load_raw_data(csv_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw marketing campaign data with delimiter detection."""
    delimiter = detect_delimiter(csv_path)
    return pd.read_csv(csv_path, sep=delimiter, low_memory=False)


def to_snake_case(column_name: str) -> str:
    """Convert source column names like Year_Birth to year_birth."""
    cleaned = column_name.strip()
    cleaned = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_").lower()


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe with consistent snake_case columns."""
    renamed = df.copy()
    renamed.columns = [to_snake_case(column) for column in renamed.columns]
    return renamed


def inspect_dataframe(df: pd.DataFrame) -> None:
    """Print the raw structure requested for the loading step."""
    print("Shape:")
    print(df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing value counts:")
    print(df.isna().sum())
    print("\nNumeric summary statistics:")
    print(df.describe(include="number").T)
    print("\nCategorical/date summary statistics:")
    print(df.describe(include=["object", "string"]).T)


def investigate_missing_income(df: pd.DataFrame) -> None:
    """Print a short breakdown of rows with missing Income."""
    income_column = "Income" if "Income" in df.columns else "income"
    education_column = "Education" if "Education" in df.columns else "education"
    marital_column = "Marital_Status" if "Marital_Status" in df.columns else "marital_status"

    missing_income = df[df[income_column].isna()]
    print(f"\nRows with missing income: {len(missing_income)}")

    if missing_income.empty:
        return

    print("\nMissing income by education:")
    print(missing_income[education_column].value_counts(dropna=False))
    print("\nMissing income by marital status:")
    print(missing_income[marital_column].value_counts(dropna=False))


def quality_check_summary(df: pd.DataFrame) -> dict[str, int]:
    """Return counts for obvious data-quality issues in standardized columns."""
    count_columns = [column for column in df.columns if column.startswith("num_")]
    amount_columns = [column for column in df.columns if column.startswith("mnt_")]

    signup_year = df["dt_customer"].dt.year if pd.api.types.is_datetime64_any_dtype(df["dt_customer"]) else None
    if signup_year is not None:
        unrealistic_birth_years = (
            (df["year_birth"] > signup_year)
            | ((signup_year - df["year_birth"]) > MAX_REASONABLE_AGE_AT_SIGNUP)
        )
    else:
        unrealistic_birth_years = pd.Series(False, index=df.index)

    return {
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_income": int(df["income"].isna().sum()),
        "negative_income_rows": int((df["income"] < 0).sum()),
        "unrealistic_birth_year_rows": int(unrealistic_birth_years.sum()),
        "negative_purchase_count_rows": int((df[count_columns] < 0).any(axis=1).sum()),
        "negative_spending_amount_rows": int((df[amount_columns] < 0).any(axis=1).sum()),
    }


def clean_customer_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply conservative cleaning rules and return the cleaned data plus a report."""
    cleaned = standardize_column_names(raw_df)

    report: dict[str, int] = {
        "starting_rows": len(cleaned),
        "starting_columns": cleaned.shape[1],
    }

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates().copy()
    report["duplicate_rows_removed"] = before - len(cleaned)

    cleaned["dt_customer"] = pd.to_datetime(
        cleaned["dt_customer"],
        format="%d-%m-%Y",
        errors="coerce",
    )

    before = len(cleaned)
    cleaned = cleaned.dropna(subset=["dt_customer"]).copy()
    report["invalid_date_rows_removed"] = before - len(cleaned)

    before = len(cleaned)
    # Income is central to this project because it will support customer profiling
    # and later modeling. Only 24 rows are missing income (~1% of the raw data), so
    # dropping them is more conservative than imputing values before EDA.
    cleaned = cleaned.dropna(subset=["income"]).copy()
    report["missing_income_rows_removed"] = before - len(cleaned)

    before = len(cleaned)
    cleaned = cleaned[cleaned["income"] >= 0].copy()
    report["negative_income_rows_removed"] = before - len(cleaned)

    signup_year = cleaned["dt_customer"].dt.year
    unrealistic_birth_mask = (
        (cleaned["year_birth"] > signup_year)
        | ((signup_year - cleaned["year_birth"]) > MAX_REASONABLE_AGE_AT_SIGNUP)
    )
    before = len(cleaned)
    cleaned = cleaned[~unrealistic_birth_mask].copy()
    report["unrealistic_birth_year_rows_removed"] = before - len(cleaned)

    count_columns = [column for column in cleaned.columns if column.startswith("num_")]
    before = len(cleaned)
    cleaned = cleaned[~(cleaned[count_columns] < 0).any(axis=1)].copy()
    report["negative_purchase_count_rows_removed"] = before - len(cleaned)

    amount_columns = [column for column in cleaned.columns if column.startswith("mnt_")]
    before = len(cleaned)
    cleaned = cleaned[~(cleaned[amount_columns] < 0).any(axis=1)].copy()
    report["negative_spending_amount_rows_removed"] = before - len(cleaned)

    report["ending_rows"] = len(cleaned)
    report["ending_columns"] = cleaned.shape[1]

    return cleaned.reset_index(drop=True), report


def save_clean_data(df: pd.DataFrame, output_path: Path = CLEAN_DATA_PATH) -> None:
    """Save the cleaned dataset for later project steps."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    """Run the full loading and cleaning workflow."""
    raw_df = load_raw_data()

    print("Raw data inspection")
    print("=" * 80)
    inspect_dataframe(raw_df)
    investigate_missing_income(raw_df)

    cleaned_df, cleaning_report = clean_customer_data(raw_df)
    save_clean_data(cleaned_df)

    print("\nCleaning report")
    print("=" * 80)
    for key, value in cleaning_report.items():
        print(f"{key}: {value}")

    print("\nFinal cleaned dataframe shape:")
    print(cleaned_df.shape)
    print("\nRemaining missing values:")
    print(cleaned_df.isna().sum())
    print("\nExample cleaned rows:")
    print(cleaned_df.head())
    print(f"\nSaved cleaned data to: {CLEAN_DATA_PATH}")


if __name__ == "__main__":
    main()
