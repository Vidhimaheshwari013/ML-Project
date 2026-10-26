
# Clustering Indian States Using SDG India Index Goal Scores

An unsupervised machine-learning project that explores similarities in Sustainable Development Goal (SDG) scores across Indian states and Union Territories.

## Objectives

- Group regions using K-Means clustering.
- Compare K-Means with Agglomerative Hierarchical Clustering.
- Visualize the data using Principal Component Analysis (PCA).
- Analyze correlations between SDG goal scores.
- Interpret cluster profiles and model limitations.

## Methods

- Data preprocessing and feature scaling
- K-Means clustering
- Agglomerative Hierarchical Clustering
- PCA visualization
- Correlation analysis
- Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score

## Key Findings

- Best K-Means Silhouette Score: approximately 0.1967 at K = 2.
- Best hierarchical Silhouette Score: approximately 0.1803 at K = 3.
- The first two principal components explain approximately 42.47% of the variance.
- Goal 4 and Goal 9 have the strongest positive correlation observed, approximately +0.67.
- Goal 5 and Goal 12 have the strongest negative correlation observed, approximately -0.55.

The low Silhouette Scores suggest that cluster separation is limited, so the groups should be interpreted cautiously.

## Project Structure

- `data/` — source and prepared datasets
- `src/` — Python scripts
- `results/` — generated metrics, cluster profiles, and visualizations
- `PROJECT_REPORT.md` — detailed project report
- `requirements.txt` — Python dependencies

## Setup

Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

Run the analysis scripts from the project root using Python. Start with preprocessing and data preparation, then run the clustering, PCA, correlation, comparison, and interpretation scripts.

## Limitations

Goal 14 was excluded because of substantial missing data. The dataset contains 36 regions, and the clustering results show limited separation. Correlations should not be interpreted as causal relationships.
