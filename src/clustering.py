"""Customer clustering pipeline for the Customer Personality Analysis project.

This script is limited to step 5: unsupervised clustering. It prepares a
clustering matrix, evaluates KMeans cluster counts, fits the final model, and
saves the original engineered dataset with a new `cluster` label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customer_features.csv"
CLUSTERED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "customer_clustered.csv"

RANDOM_STATE = 42
K_VALUES = list(range(2, 9))
FINAL_K = 4

RAW_CLUSTERING_FEATURES = [
    "age",
    "income",
    "children_total",
    "household_size",
    "recency",
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
    "web_purchase_share",
    "store_purchase_share",
    "catalog_purchase_share",
    "deal_purchase_share",
    "average_spend_per_purchase",
    "total_campaign_acceptances",
]

LOG_TRANSFORM_COLUMNS = [
    "income",
    "total_spending",
    "spending_per_tenure_year",
    "total_purchases",
    "average_spend_per_purchase",
]

WINSORIZE_COLUMNS = [
    "income",
    "spending_per_tenure_year",
]


@dataclass(frozen=True)
class ClusteringData:
    """Container for prepared clustering data and preprocessing metadata."""

    transformed_features: pd.DataFrame
    scaled_matrix: np.ndarray
    preprocessing_pipeline: Pipeline
    winsor_limits: dict[str, float]


def load_feature_data(csv_path: Path = FEATURE_DATA_PATH) -> pd.DataFrame:
    """Load the engineered customer features."""
    return pd.read_csv(csv_path, parse_dates=["dt_customer"])


def get_clustering_features() -> list[str]:
    """Return the raw feature names used as clustering inputs."""
    return RAW_CLUSTERING_FEATURES.copy()


def validate_input_data(df: pd.DataFrame) -> dict[str, int]:
    """Check missing and infinite values before clustering."""
    numeric_df = df.select_dtypes(include=[np.number])
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "infinite_values": int(np.isinf(numeric_df.to_numpy()).sum()),
    }


def calculate_winsor_limits(
    df: pd.DataFrame,
    columns: list[str] = WINSORIZE_COLUMNS,
    upper_quantile: float = 0.99,
) -> dict[str, float]:
    """Calculate upper caps for clustering-only winsorization."""
    return {column: float(df[column].quantile(upper_quantile)) for column in columns}


def transform_clustering_features(
    df: pd.DataFrame,
    winsor_limits: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Select and transform clustering features.

    Transformations are applied only to the clustering input matrix. The original
    engineered data remains unchanged and is used when saving cluster labels.
    """
    selected = df[get_clustering_features()].copy()

    if winsor_limits is None:
        winsor_limits = calculate_winsor_limits(selected)

    for column, upper_limit in winsor_limits.items():
        selected[column] = selected[column].clip(upper=upper_limit)

    for column in LOG_TRANSFORM_COLUMNS:
        # log1p reduces right skew while safely handling zero-valued customers.
        selected[f"{column}_log1p"] = np.log1p(selected[column].clip(lower=0))
        selected = selected.drop(columns=column)

    return selected, winsor_limits


def prepare_clustering_matrix(df: pd.DataFrame) -> ClusteringData:
    """Create transformed and scaled clustering inputs."""
    transformed, winsor_limits = transform_clustering_features(df)

    if transformed.isna().sum().sum() != 0:
        raise ValueError("Clustering features contain missing values after transformation.")

    if np.isinf(transformed.to_numpy()).sum() != 0:
        raise ValueError("Clustering features contain infinite values after transformation.")

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    scaled_matrix = pipeline.fit_transform(transformed)

    return ClusteringData(
        transformed_features=transformed,
        scaled_matrix=scaled_matrix,
        preprocessing_pipeline=pipeline,
        winsor_limits=winsor_limits,
    )


def evaluate_k_values(
    scaled_matrix: np.ndarray,
    k_values: list[int] = K_VALUES,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Evaluate KMeans inertia and silhouette score over a range of k values."""
    results = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(scaled_matrix)
        results.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": silhouette_score(scaled_matrix, labels),
            }
        )

    return pd.DataFrame(results)


def fit_final_kmeans(
    scaled_matrix: np.ndarray,
    n_clusters: int = FINAL_K,
    random_state: int = RANDOM_STATE,
) -> KMeans:
    """Fit the final KMeans clustering model."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    model.fit(scaled_matrix)
    return model


def add_cluster_labels(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Return a copy of the original engineered dataframe with cluster labels."""
    clustered = df.copy()
    clustered["cluster"] = labels.astype(int)
    return clustered


def create_pca_coordinates(
    scaled_matrix: np.ndarray,
    labels: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Create two-dimensional PCA coordinates for cluster visualization."""
    pca = PCA(n_components=2, random_state=random_state)
    coordinates = pca.fit_transform(scaled_matrix)
    return pd.DataFrame(
        {
            "pca_1": coordinates[:, 0],
            "pca_2": coordinates[:, 1],
            "cluster": labels.astype(int),
        }
    )


def save_clustered_output(
    clustered_df: pd.DataFrame,
    output_path: Path = CLUSTERED_DATA_PATH,
) -> None:
    """Save the original engineered data plus the final cluster label."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clustered_df.to_csv(output_path, index=False)


def run_clustering_pipeline(
    n_clusters: int = FINAL_K,
) -> tuple[pd.DataFrame, ClusteringData, pd.DataFrame, KMeans, pd.DataFrame]:
    """Run the full clustering workflow and save the clustered dataset."""
    feature_df = load_feature_data()
    clustering_data = prepare_clustering_matrix(feature_df)
    evaluation_df = evaluate_k_values(clustering_data.scaled_matrix)
    final_model = fit_final_kmeans(clustering_data.scaled_matrix, n_clusters=n_clusters)
    clustered_df = add_cluster_labels(feature_df, final_model.labels_)
    save_clustered_output(clustered_df)
    pca_df = create_pca_coordinates(clustering_data.scaled_matrix, final_model.labels_)
    return clustered_df, clustering_data, evaluation_df, final_model, pca_df


def main() -> None:
    """Run clustering and print a concise validation summary."""
    feature_df = load_feature_data()
    clustering_data = prepare_clustering_matrix(feature_df)
    evaluation_df = evaluate_k_values(clustering_data.scaled_matrix)
    final_model = fit_final_kmeans(clustering_data.scaled_matrix, n_clusters=FINAL_K)
    clustered_df = add_cluster_labels(feature_df, final_model.labels_)
    save_clustered_output(clustered_df)

    print("Clustering pipeline summary")
    print("=" * 80)
    print(f"Input shape: {feature_df.shape}")
    print(f"Output shape: {clustered_df.shape}")
    print("\nInput validation:")
    for key, value in validate_input_data(feature_df).items():
        print(f"{key}: {value}")

    print("\nSelected clustering features:")
    for feature in get_clustering_features():
        print(f"- {feature}")

    print("\nWinsorization caps used for clustering inputs only:")
    for column, cap in clustering_data.winsor_limits.items():
        print(f"- {column}: {cap:.4f}")

    print("\nLog1p transformed columns:")
    for column in LOG_TRANSFORM_COLUMNS:
        print(f"- {column}")

    print("\nKMeans evaluation:")
    print(evaluation_df.to_string(index=False))

    print(f"\nFinal k: {FINAL_K}")
    print("\nCluster counts:")
    print(clustered_df["cluster"].value_counts().sort_index().to_string())

    print("\nExample clustered rows:")
    print(
        clustered_df[
            [
                "id",
                "age",
                "income",
                "total_spending",
                "total_purchases",
                "total_campaign_acceptances",
                "cluster",
            ]
        ]
        .head()
        .to_string(index=False)
    )
    print(f"\nSaved clustered data to: {CLUSTERED_DATA_PATH}")


if __name__ == "__main__":
    main()
