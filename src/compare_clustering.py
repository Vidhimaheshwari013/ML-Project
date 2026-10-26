
import os
import pandas as pd  # type: ignore[reportMissingModuleSource]
import matplotlib.pyplot as plt  # type: ignore[reportMissingModuleSource]

# File paths
KMEANS_PATH = "results/kmeans_metrics.csv"
HIERARCHICAL_PATH = "results/hierarchical_metrics.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load evaluation results
kmeans = pd.read_csv(KMEANS_PATH)
hierarchical = pd.read_csv(HIERARCHICAL_PATH)

# Add model names
kmeans["Model"] = "K-Means"
hierarchical["Model"] = "Hierarchical"

# Combine results
comparison = pd.concat(
    [kmeans, hierarchical],
    ignore_index=True
)

# Save combined comparison
comparison.to_csv(
    os.path.join(RESULTS_DIR, "clustering_model_comparison.csv"),
    index=False
)

print("CLUSTERING MODEL COMPARISON")
print("=" * 60)
print(comparison.to_string(index=False))

# Identify available metric columns
metric_columns = [
    col for col in comparison.columns
    if "silhouette" in col.lower()
    or "davies" in col.lower()
    or "calinski" in col.lower()
]

print("\nMetric columns detected:", metric_columns)

# Create a separate plot for each metric
for metric in metric_columns:
    plt.figure(figsize=(10, 5))

    for model in comparison["Model"].unique():
        model_data = comparison[comparison["Model"] == model]
        plt.plot(
            model_data["K"],
            model_data[metric],
            marker="o",
            label=model
        )

    plt.title(f"Clustering Comparison: {metric}")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel(metric)
    plt.xticks(sorted(comparison["K"].unique()))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = (
        metric.lower()
        .replace(" ", "_")
        .replace("-", "_")
        + "_comparison.png"
    )

    plt.savefig(
        os.path.join(RESULTS_DIR, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

print("\nComparison completed successfully!")
print("Files saved in:", os.path.abspath(RESULTS_DIR))
