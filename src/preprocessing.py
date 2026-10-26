
import pandas as pd
from pathlib import Path

# Locate the dataset relative to this script
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "sdg_india_index.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH)

# SDG goal columns used for machine learning
goal_cols = [f"Goal {i}" for i in range(1, 17)]

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nMissing values in each goal:")
print(df[goal_cols].isnull().sum())

print("\nDuplicate region names:", df["Region"].duplicated().sum())

# Save a basic data-quality report
report_path = PROJECT_DIR / "data" / "data_quality_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write(f"Dataset shape: {df.shape}\n")
    file.write("\nMissing values:\n")
    file.write(df[goal_cols].isnull().sum().to_string())
    file.write(f"\n\nDuplicate regions: {df['Region'].duplicated().sum()}\n")

print("\nData-quality report saved to:", report_path)

