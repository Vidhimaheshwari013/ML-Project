
import os
import pandas as pd  # type: ignore[reportMissingModuleSource]

DATA_PATH = "data/clustering_input.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load the prepared SDG dataset
data = pd.read_csv(DATA_PATH)

def find_column(df, keyword):
    matches = [
        col for col in df.columns
        if keyword in col.lower()
    ]
    return matches[0] if matches else None

def analyze_clusters(filename, model_name):
    path = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    assignments = pd.read_csv(path)

    # Identify region columns
    data_region = find_column(data, "region")
    assignment_region = find_column(assignments, "region")

    # Identify the cluster label column
    cluster_candidates = [
        col for col in assignments.columns
        if "cluster" in col.lower()
    ]

    if not data_region or not assignment_region:
        print(f"Could not identify region columns in {filename}")
        print("Columns:", assignments.columns.tolist())
        return

    if not cluster_candidates:
        print(f"Could not identify cluster column in {filename}")
        print("Columns:", assignments.columns.tolist())
        return

    cluster_col = cluster_candidates[0]

    # Match each region to its cluster
    merged = data.merge(
        assignments[[assignment_region, cluster_col]],
        left_on=data_region,
        right_on=assignment_region,
        how="inner"
    )

    goal_columns = [
        col for col in data.columns
        if col.startswith("Goal ")
    ]

    print(f"\n{'=' * 55}")
    print(f"{model_name.upper()} CLUSTER INTERPRETATION")
    print("=" * 55)

    # List regions in each cluster
    for cluster in sorted(merged[cluster_col].dropna().unique()):
        group = merged[merged[cluster_col] == cluster]

        print(f"\nCluster {cluster} — {len(group)} regions")
        print(", ".join(group[data_region].astype(str).tolist()))

    # Calculate average score for each goal in each cluster
    profiles = merged.groupby(cluster_col)[goal_columns].mean().round(2)

    print("\nAverage SDG scores by cluster:")
    print(profiles.to_string())

    # Save cluster profiles
    output_file = (
        model_name.lower().replace(" ", "_")
        + "_cluster_profiles.csv"
    )

    profiles.to_csv(
        os.path.join(RESULTS_DIR, output_file)
    )

    print(f"\nSaved: {output_file}")

    return merged

# Analyze both clustering methods
analyze_clusters(
    "kmeans_cluster_assignments.csv",
    "K-Means"
)

analyze_clusters(
    "hierarchical_cluster_assignments.csv",
    "Hierarchical"
)

print("\nCluster interpretation completed.")
