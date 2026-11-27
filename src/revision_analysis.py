
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Load data
df = pd.read_csv("data/clustering_input.csv")

goal_columns = [col for col in df.columns if col.startswith("Goal")]
X = df[goal_columns].values

# Standardise all 15 goal scores
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

K = 2
N_SEEDS = 50
N_BOOTSTRAPS = 100
N_PERMUTATIONS = 100

# Reference K-Means solution
reference_model = KMeans(
    n_clusters=K, random_state=42, n_init=20
)
reference_labels = reference_model.fit_predict(X_scaled)
reference_silhouette = silhouette_score(X_scaled, reference_labels)

# 1. Random-seed stability
seed_results = []

for seed in range(N_SEEDS):
    model = KMeans(n_clusters=K, random_state=seed, n_init=20)
    labels = model.fit_predict(X_scaled)

    seed_results.append({
        "Seed": seed,
        "Silhouette": silhouette_score(X_scaled, labels),
        "ARI_vs_reference": adjusted_rand_score(
            reference_labels, labels
        )
    })

seed_df = pd.DataFrame(seed_results)
seed_df.to_csv("results/revision_seed_stability.csv", index=False)

# 2. Bootstrap stability
bootstrap_results = []
rng = np.random.default_rng(42)

for iteration in range(N_BOOTSTRAPS):
    indices = rng.choice(len(X), size=len(X), replace=True)
    X_bootstrap = X[indices]

    bootstrap_scaler = StandardScaler()
    X_bootstrap_scaled = bootstrap_scaler.fit_transform(X_bootstrap)

    model = KMeans(
        n_clusters=K,
        random_state=iteration,
        n_init=20
    )
    model.fit(X_bootstrap_scaled)

    # Predict all original regions using the bootstrap-fitted model
    predictions = model.predict(bootstrap_scaler.transform(X))

    bootstrap_results.append({
        "Iteration": iteration + 1,
        "ARI_vs_reference": adjusted_rand_score(
            reference_labels, predictions
        )
    })

bootstrap_df = pd.DataFrame(bootstrap_results)
bootstrap_df.to_csv(
    "results/revision_bootstrap_stability.csv", index=False
)

# 3. States-only analysis (exclude Union Territories)
states_df = df[df["Region_Type"] == "State"].copy()
X_states = states_df[goal_columns].values

states_scaler = StandardScaler()
X_states_scaled = states_scaler.fit_transform(X_states)

states_model = KMeans(
    n_clusters=K, random_state=42, n_init=20
)
states_labels = states_model.fit_predict(X_states_scaled)

# Compare States-only labels with the original solution
original_state_labels = reference_labels[
    df["Region_Type"].values == "State"
]

states_ari = adjusted_rand_score(
    original_state_labels, states_labels
)

states_df["States_Only_Cluster"] = states_labels
states_df.to_csv(
    "results/revision_states_only_assignments.csv", index=False
)

pd.DataFrame([{
    "Number_of_States": len(states_df),
    "States_Only_Silhouette": silhouette_score(
        X_states_scaled, states_labels
    ),
    "ARI_vs_all_region_solution": states_ari
}]).to_csv("results/revision_states_only.csv", index=False)

# 4. Shuffled-feature null baseline
permutation_results = []

for iteration in range(N_PERMUTATIONS):
    shuffled = np.column_stack([
        rng.permutation(X[:, i])
        for i in range(X.shape[1])
    ])

    shuffled_scaled = StandardScaler().fit_transform(shuffled)

    model = KMeans(
        n_clusters=K,
        random_state=iteration,
        n_init=20
    )
    labels = model.fit_predict(shuffled_scaled)

    permutation_results.append({
        "Iteration": iteration + 1,
        "Silhouette": silhouette_score(shuffled_scaled, labels)
    })

permutation_df = pd.DataFrame(permutation_results)
permutation_df.to_csv(
    "results/revision_shuffled_feature_null.csv", index=False
)

# 5. Administrative grouping comparison: State vs UT
administrative_labels = (
    df["Region_Type"].values == "Union Territory"
).astype(int)

administrative_ari = adjusted_rand_score(
    reference_labels, administrative_labels
)

# Summary
print("\nREFERENCE SOLUTION")
print("Silhouette:", round(reference_silhouette, 4))

print("\nRANDOM-SEED STABILITY")
print(seed_df[["Silhouette", "ARI_vs_reference"]].agg(
    ["mean", "std"]
).round(4))

print("\nBOOTSTRAP STABILITY")
print(bootstrap_df["ARI_vs_reference"].agg(
    ["mean", "std", "min", "max"]
).round(4))

print("\nSTATES-ONLY ANALYSIS")
print("Number of States:", len(states_df))
print("Silhouette:", round(
    silhouette_score(X_states_scaled, states_labels), 4
))
print("ARI vs all-region solution:", round(states_ari, 4))

print("\nSHUFFLED-FEATURE NULL BASELINE")
print(permutation_df["Silhouette"].agg(
    ["mean", "std", "min", "max"]
).round(4))

print("\nADMINISTRATIVE GROUPING")
print("ARI (K-Means vs State/UT):", round(administrative_ari, 4))

print("\nAll revision analyses saved in results/")
