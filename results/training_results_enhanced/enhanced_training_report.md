# Enhanced PyOD Model Training Report

**Report Generated on**: 2025-06-27 18:05:07

---


## 1. Executive Summary

This report details the training and analysis of three unsupervised outlier detection models (IForest, ECOD, COPOD) on the CIC-IoT-Dataset-2023 benign traffic data. The process included data preprocessing, model training, and a detailed analysis of the resulting outlier scores. All artifacts, including trained models, scalers, and visualizations, have been saved for reproducibility.

### Summary of Trained Models

| Model | Training Time (s) | Mean Score | Median Score | Saved Model File |

|---|---|---|---|---|

| IFOREST | 2.81 | -0.1375 | -0.1475 | `iforest_model.joblib` |
| ECOD | 5.32 | 31.2863 | 29.9768 | `ecod_model.joblib` |
| COPOD | 2.09 | 25.5740 | 24.0084 | `copod_model.joblib` |


## 2. Setup and Preprocessing

- **Input Data**: `data\Benign_Final\BenignTraffic.pcap.csv` (362,361 rows × 39 features)

- **Preprocessing Pipeline**:

  1. Replaced `infinity` values with `NaN`.

  2. Imputed all `NaN` values using column-wise medians.

  3. Standardized all features using `sklearn.preprocessing.StandardScaler`.

- **Saved Artifacts**:

  - **Scaler**: `scaler.joblib`

  - **Feature List**: `training_columns.joblib`


---

## 3. Model-Specific Analysis


### 3.1 IFOREST Analysis

- **Training Time**: 2.81 seconds

- **Score Statistics**:
  - **Mean**: -0.1375
  - **Median**: -0.1475
  - **Std. Dev**: 0.0469
  - **Min / Max**: -0.2032 / 0.1093

#### Score Distribution

This histogram shows the distribution of outlier scores assigned to the training data. A long tail typically indicates good separation between inliers and potential outliers.

![iforest Score Distribution](visualizations\iforest_scores_distribution.png)

#### 2D PCA Scatter Plot

This plot shows the data in 2D (via PCA), with points colored by their outlier score. It helps visualize which regions of the data space are considered anomalous.

![iforest PCA Scatter Plot](visualizations\iforest_pca_scatter.png)


### 3.1 ECOD Analysis

- **Training Time**: 5.32 seconds

- **Score Statistics**:
  - **Mean**: 31.2863
  - **Median**: 29.9768
  - **Std. Dev**: 8.5022
  - **Min / Max**: 14.8824 / 114.1520

#### Score Distribution

This histogram shows the distribution of outlier scores assigned to the training data. A long tail typically indicates good separation between inliers and potential outliers.

![ecod Score Distribution](visualizations\ecod_scores_distribution.png)

#### 2D PCA Scatter Plot

This plot shows the data in 2D (via PCA), with points colored by their outlier score. It helps visualize which regions of the data space are considered anomalous.

![ecod PCA Scatter Plot](visualizations\ecod_pca_scatter.png)


### 3.1 COPOD Analysis

- **Training Time**: 2.09 seconds

- **Score Statistics**:
  - **Mean**: 25.5740
  - **Median**: 24.0084
  - **Std. Dev**: 7.6740
  - **Min / Max**: 12.1173 / 92.7206

#### Score Distribution

This histogram shows the distribution of outlier scores assigned to the training data. A long tail typically indicates good separation between inliers and potential outliers.

![copod Score Distribution](visualizations\copod_scores_distribution.png)

#### 2D PCA Scatter Plot

This plot shows the data in 2D (via PCA), with points colored by their outlier score. It helps visualize which regions of the data space are considered anomalous.

![copod PCA Scatter Plot](visualizations\copod_pca_scatter.png)
