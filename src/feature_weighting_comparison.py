
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "clustering_input.csv"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

goal_cols = [
    f"Goal {i}" for i in range(1, 17)
    if i != 14
]

X = df[goal_cols]

# Standardize original features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA retaining 90% variance
pca = PCA(n_components=0.90)
X_pca = pca.fit_transform(X_scaled)

print("Original number of features:", X_scaled.shape[1])
print("PCA-reduced number of features:", X_pca.shape[1])
print(
    "Variance retained:",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)

# Compare clustering for K = 2 to 8
results = []

for k in range(2, 9):

    # Original 15 standardized features
    original_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    original_labels = original_model.fit_predict(X_scaled)

    # PCA-reduced features
    pca_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    pca_labels = pca_model.fit_predict(X_pca)

    results.append({
        "K": k,
        "Original Silhouette": silhouette_score(
            X_scaled, original_labels
        ),
        "PCA Silhouette": silhouette_score(
            X_pca, pca_labels
        ),
        "Original DB Index": davies_bouldin_score(
            X_scaled, original_labels
        ),
        "PCA DB Index": davies_bouldin_score(
            X_pca, pca_labels
        ),
        "Original CH Score": calinski_harabasz_score(
            X_scaled, original_labels
        ),
        "PCA CH Score": calinski_harabasz_score(
            X_pca, pca_labels
        ),
        "Adjusted Rand Index": adjusted_rand_score(
            original_labels, pca_labels
        )
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULTS_DIR / "feature_weighting_comparison.csv",
    index=False
)

print("\nFeature comparison results:")
print(results_df.round(4).to_string(index=False))

print("\nResults saved in:", RESULTS_DIR)
