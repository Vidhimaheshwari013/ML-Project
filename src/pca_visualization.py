
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Define project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "clustering_input.csv"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 2. Load the prepared dataset
df = pd.read_csv(DATA_PATH)

# 3. Select the 15 complete SDG goal columns
# Goal 14 is excluded because it has many missing values
goal_cols = [f"Goal {i}" for i in range(1, 17) if i != 14]
X = df[goal_cols]

# 4. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Apply PCA to reduce 15 features to 2 components
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

# 6. Save PCA coordinates for every region
pca_df = pd.DataFrame({
    "Region": df["Region"],
    "Region_Type": df["Region_Type"],
    "PC1": components[:, 0],
    "PC2": components[:, 1]
})

pca_df.to_csv(RESULTS_DIR / "pca_coordinates.csv", index=False)

# 7. Save explained variance
variance_df = pd.DataFrame({
    "Principal Component": ["PC1", "PC2"],
    "Explained Variance (%)": pca.explained_variance_ratio_ * 100
})

variance_df.to_csv(
    RESULTS_DIR / "pca_explained_variance.csv",
    index=False
)

# 8. Create the PCA scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.8)

for _, row in pca_df.iterrows():
    plt.annotate(
        row["Region"],
        (row["PC1"], row["PC2"]),
        fontsize=7,
        xytext=(4, 3),
        textcoords="offset points"
    )

plt.xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)"
)
plt.ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)"
)
plt.title("PCA of SDG Scores Across Indian States and UTs")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "pca_visualization.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# 9. Create an explained-variance chart
plt.figure(figsize=(7, 5))
plt.bar(
    ["PC1", "PC2"],
    pca.explained_variance_ratio_ * 100
)
plt.ylabel("Explained Variance (%)")
plt.title("Variance Explained by the First Two Principal Components")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "pca_explained_variance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# 10. Display results
print("PCA completed successfully!")
print("\nExplained variance:")
print(variance_df.to_string(index=False))

total_variance = pca.explained_variance_ratio_.sum() * 100
print(f"\nTotal variance explained by PC1 and PC2: {total_variance:.2f}%")

print("\nFiles saved in:", RESULTS_DIR)

