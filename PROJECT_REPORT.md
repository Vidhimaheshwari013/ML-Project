
# Clustering Indian States Using SDG India Index Goal Scores

## 1. Abstract

This project applies unsupervised machine learning to group Indian states and Union Territories based on their Sustainable Development Goal (SDG) scores. K-Means and Agglomerative Hierarchical Clustering are used to identify broad patterns in SDG performance. PCA is used to visualize the data in two dimensions, while correlation analysis explores relationships between SDG goals.

## 2. Problem Statement

Different Indian states and Union Territories perform differently across sustainable development goals. This project aims to identify regions with similar SDG score patterns and examine how the goals are related to one another.

## 3. Objectives

- Prepare and validate the SDG India Index dataset.
- Group regions using K-Means clustering.
- Apply hierarchical clustering and compare its results with K-Means.
- Visualize regional patterns using Principal Component Analysis (PCA).
- Examine correlations between SDG goal scores.
- Interpret the resulting clusters and discuss the limitations.

## 4. Dataset

The dataset contains 36 regions, including states and Union Territories. It includes SDG goal scores and regional information.

Goal 14 was excluded from the clustering and related analyses because it contained 27 missing values. The remaining 15 goal features were used for analysis.

## 5. Methodology

### 5.1 Data Preprocessing
The data was inspected for missing values, duplicate regions, and consistency. A prepared dataset was created for the machine-learning analysis.

### 5.2 Feature Scaling
StandardScaler was used to standardize the SDG features so that differences in their scales would not disproportionately influence distance-based clustering.

### 5.3 K-Means Clustering
K-Means was evaluated for cluster counts from K = 2 to K = 8. Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score were used for evaluation.

### 5.4 Hierarchical Clustering
Agglomerative clustering using Ward linkage was evaluated for K = 2 to K = 8. A dendrogram was generated to visualize hierarchical relationships.

### 5.5 Principal Component Analysis
PCA reduced the standardized features to two principal components for visualization.

### 5.6 Correlation Analysis
A correlation matrix and heatmap were generated to examine relationships among SDG goal scores.

## 6. Results

### 6.1 Clustering Comparison

The best Silhouette Score achieved by K-Means was approximately 0.1967 at K = 2. The best Silhouette Score for hierarchical clustering was approximately 0.1803 at K = 3.

K-Means performed slightly better according to the best Silhouette Score. However, the scores indicate relatively weak separation between clusters.

The best cluster count depends on the evaluation metric, so no single value of K should be considered unquestionably optimal.

### 6.2 PCA Results

The first principal component explained approximately 27.27% of the variance, while the second explained approximately 15.20%.

Together, the two components explained approximately 42.47% of the total variance. The PCA plot therefore provides a useful but incomplete two-dimensional view of the dataset.

### 6.3 Correlation Results

The strongest positive correlation observed was between Goal 4 and Goal 9, at approximately +0.67.

The strongest negative correlation observed was between Goal 5 and Goal 12, at approximately -0.55.

These correlations indicate associations in the dataset and do not establish cause-and-effect relationships.

### 6.4 Cluster Interpretation

The cluster profiles show broad differences in average SDG scores between the groups. For K-Means, one cluster had higher average scores on most of the included goals, while the other scored higher on Goals 12 and 15.

Cluster labels are arbitrary identifiers, not rankings. The groups should be interpreted through their average goal scores rather than their numeric labels alone.

## 7. Limitations

- The dataset contains only 36 regions, which limits the strength of conclusions.
- Goal 14 was excluded because of substantial missing data.
- Silhouette Scores were relatively low, suggesting that the groups overlap.
- Results can depend on the selected algorithm, number of clusters, and feature selection.
- Correlation does not imply causation.
- The analysis identifies patterns and does not establish why a region received a particular score.

## 8. Conclusion

This project demonstrates how unsupervised machine-learning techniques can be used to explore similarities among Indian states and Union Territories based on SDG scores.

K-Means achieved a slightly higher best Silhouette Score than hierarchical clustering. PCA and correlation analysis provided additional ways to understand the dataset. However, the relatively low Silhouette Scores mean the clusters should be treated as exploratory groups rather than clearly separated categories.

The analysis can support further investigation into regional SDG performance, but the results should be interpreted alongside the dataset's limitations.

## 9. Future Scope

- Explore methods for handling missing SDG scores.
- Compare additional clustering algorithms.
- Test cluster stability and sensitivity to feature selection.
- Develop an interactive dashboard for exploring regional profiles.
- Incorporate data from additional years to examine changes over time.

## 10. Tools and Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- SciPy
- Visual Studio Code

