
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "sdg_india_index.csv"
OUTPUT_PATH = PROJECT_DIR / "data" / "clustering_input.csv"

df = pd.read_csv(DATA_PATH)

# Keep the 15 goals with complete scores
goal_cols = [
    f"Goal {i}" for i in range(1, 17)
    if i != 14
]

# Select region information and the goal scores
features = df[["Region", "Region_Type"] + goal_cols].copy()

# Verify that the selected scores are complete
if features[goal_cols].isnull().any().any():
    raise ValueError("Missing values found in selected goal columns.")

if features["Region"].duplicated().any():
    raise ValueError("Duplicate region names found.")

# Save the prepared data
features.to_csv(OUTPUT_PATH, index=False)

print("Data prepared successfully!")
print("Number of regions:", len(features))
print("Number of goal features:", len(goal_cols))
print("Missing values:", int(features[goal_cols].isnull().sum().sum()))
print("Saved to:", OUTPUT_PATH)
print("\nPreview:")
print(features.head())
