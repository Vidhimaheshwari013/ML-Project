# Clustering Indian States and Union Territories Using SDG India Index

A machine learning project that groups Indian States and Union Territories based on their Sustainable Development Goal (SDG) performance scores to explore similarities and differences in development indicators.

## Project Overview

The SDG India Index provides insights into India's progress toward the United Nations Sustainable Development Goals. This project applies unsupervised machine learning to identify groups of regions with similar SDG performance.

The project also uses dimensionality reduction and correlation analysis to explore relationships among SDG indicators.

## Objectives

- Group States and Union Territories based on their SDG goal scores.
- Compare K-Means and Hierarchical Clustering.
- Evaluate clustering performance using standard metrics.
- Visualize clusters and patterns using Principal Component Analysis (PCA).
- Examine relationships among SDG indicators through correlation analysis.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- PCA
- K-Means Clustering
- Hierarchical Clustering

## Methodology

1. **Data preprocessing:** Prepare the state-wise and Union Territory data for analysis.
2. **Feature preparation:** Select SDG goal scores and prepare the dataset for clustering.
3. **K-Means clustering:** Test different values of K and evaluate cluster quality.
4. **Hierarchical clustering:** Compare alternative groupings and examine cluster structure.
5. **Model evaluation:** Use Silhouette Score, Davies–Bouldin Index, and Calinski–Harabasz Score.
6. **PCA visualization:** Reduce the feature dimensions to visualize regional similarities.
7. **Correlation analysis:** Explore relationships among SDG goal scores.

## Results

- The highest Silhouette Score obtained for K-Means was **0.1967 at K = 2**.
- The highest Silhouette Score obtained for Hierarchical Clustering was **0.1803 at K = 3**.
- The first two principal components explained approximately **42.47% of the variance**.

These results suggest that the regions have some similarities in their SDG profiles, although the separation between clusters is limited. The findings should be interpreted as exploratory patterns rather than definitive rankings of regional development.

## Project Structure

```text
ML-Project/
├── data/
├── results/
├── src/
├── SDG_India_Index_Report.pdf
├── README.md
├── requirements.txt
└── .gitignore
```

- `data/` — prepared datasets and data-quality information.
- `results/` — clustering assignments, evaluation metrics, PCA outputs, and visualizations.
- `src/` — Python scripts for preprocessing, clustering, PCA, and correlation analysis.
- `SDG_India_Index_Report.pdf` — project report.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Vidhimaheshwari013/ML-Project.git
cd ML-Project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the analysis

Run the relevant scripts from the project root. For example:

```bash
python src/preprocessing.py
python src/clustering.py
python src/hierarchical_clustering.py
python src/pca_visualization.py
```

Run scripts in the appropriate order if one depends on the output of another.

## Future Improvements

- Explore additional clustering algorithms and parameter settings.
- Improve cluster interpretation using regional SDG profiles.
- Develop an interactive dashboard for exploring results.
- Investigate how clustering patterns change across different SDG indicators.

## Author

**Vidhi Maheshwari**

Computer Science Student | Machine Learning | Data Analysis

## Repository

[View the project on GitHub](https://github.com/Vidhimaheshwari013/ML-Project)

