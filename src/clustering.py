
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "clustering_input.csv"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load prepared dataset
df = pd.read_csv(DATA_PATH)

# Use only the 15 SDG goal scores as features
goal_cols = [
    f"Goal {i}" for i in range(1, 17)
    if i != 14
]

X = df[goal_cols]

# Standardize scores
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Evaluate K-means for different cluster counts
metrics = []
best_model = None
best_score = -1
best_k = None

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X_scaled)

    silhouette = silhouette_score(X_scaled, labels)
    db_index = davies_bouldin_score(X_scaled, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)

    metrics.append({
        "K": k,
        "Silhouette Score": silhouette,
        "Davies-Bouldin Index": db_index,
        "Calinski-Harabasz Score": ch_score
    })

    print(
        f"K={k} | Silhouette={silhouette:.4f} | "
        f"Davies-Bouldin={db_index:.4f} | "
        f"Calinski-Harabasz={ch_score:.2f}"
    )

    if silhouette > best_score:
        best_score = silhouette
        best_model = model
        best_k = k

# Save evaluation metrics
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(
    RESULTS_DIR / "kmeans_metrics.csv",
    index=False
)

# Save cluster assignments from the best silhouette result
df["KMeans_Cluster"] = best_model.labels_
df.to_csv(
    RESULTS_DIR / "kmeans_cluster_assignments.csv",
    index=False
)

# Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(
    metrics_df["K"],
    metrics_df["Silhouette Score"],
    marker="o"
)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("K-means Cluster Evaluation")
plt.xticks(range(2, 9))
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "silhouette_scores.png", dpi=300)
plt.show()

print("\nBest K by silhouette score:", best_k)
print("Best silhouette score:", round(best_score, 4))
print("Results saved in:", RESULTS_DIR)
