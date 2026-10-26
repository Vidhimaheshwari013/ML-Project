
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from scipy.cluster.hierarchy import linkage, dendrogram

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "clustering_input.csv"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Select SDG goal scores
goal_cols = [
    f"Goal {i}" for i in range(1, 17) if i != 14
]

X = df[goal_cols]

# Standardize scores
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create dendrogram
linked = linkage(X_scaled, method="ward")

plt.figure(figsize=(14, 7))
dendrogram(
    linked,
    labels=df["Region"].to_numpy(),
    leaf_rotation=90,
    leaf_font_size=7
)
plt.title("Hierarchical Clustering of Indian States and UTs")
plt.xlabel("State / Union Territory")
plt.ylabel("Ward Linkage Distance")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "hierarchical_dendrogram.png",
    dpi=300
)
plt.close()

# Evaluate different cluster counts
results = []

for k in range(2, 9):
    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )
    labels = model.fit_predict(X_scaled)

    results.append({
        "K": k,
        "Silhouette Score": silhouette_score(
            X_scaled, labels
        ),
        "Davies-Bouldin Index": davies_bouldin_score(
            X_scaled, labels
        ),
        "Calinski-Harabasz Score": calinski_harabasz_score(
            X_scaled, labels
        )
    })

results_df = pd.DataFrame(results)
results_df.to_csv(
    RESULTS_DIR / "hierarchical_metrics.csv",
    index=False
)

# Save assignments using K=2 as the initial comparison
final_model = AgglomerativeClustering(
    n_clusters=2,
    linkage="ward"
)
df["Hierarchical_Cluster"] = final_model.fit_predict(
    X_scaled
)

df.to_csv(
    RESULTS_DIR / "hierarchical_cluster_assignments.csv",
    index=False
)

print(results_df.round(4).to_string(index=False))
print("\nSaved hierarchical clustering results to:", RESULTS_DIR)
