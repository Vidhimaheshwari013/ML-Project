
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define project paths
DATA_PATH = "data/clustering_input.csv"
RESULTS_DIR = "results"

# Create results folder if it does not exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load prepared dataset
df = pd.read_csv(DATA_PATH)

# Select SDG goal columns only
goal_columns = [
    col for col in df.columns
    if col.startswith("Goal ")
]

sdg_data = df[goal_columns]

# Calculate correlation matrix
correlation_matrix = sdg_data.corr()

# Save correlation matrix
correlation_matrix.to_csv(
    os.path.join(RESULTS_DIR, "sdg_correlation_matrix.csv")
)

# Create correlation heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    fmt=".2f",
    square=True,
    linewidths=0.5
)

plt.title("Correlation Between SDG Goal Scores", fontsize=16)
plt.xlabel("SDG Goals")
plt.ylabel("SDG Goals")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

# Save heatmap
plt.savefig(
    os.path.join(RESULTS_DIR, "sdg_correlation_heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Correlation analysis completed successfully!")
print("\nCorrelation matrix:")
print(correlation_matrix.round(2))
print(f"\nFiles saved in: {os.path.abspath(RESULTS_DIR)}")
