# IoT Sentinel: (SEMI) --- Unsupervised Anomaly Detection in IoT Networks

This project focuses on identifying anomalous network traffic within an Internet of Things (IoT) environment using unsupervised machine learning techniques. By training models exclusively on benign traffic from the CIC-IoT-Dataset-2023, we build a baseline of "normal" behavior, then test on real backdoor‐malware captures to measure detection performance.

## 🚀 Project Overview

**Goal:** Detect deviations from benign IoT network behavior (zero‐day, signature‐free).

**Datasets:**
- **Benign:** BenignTraffic.pcap.csv
- **Malware:** Backdoor_Malware.pcap.csv

**Models:** PyOD outlier detectors trained on benign only, then evaluated on both benign & backdoor.
- Isolation Forest (IForest)
- LOF
- KNN
- AutoEncoder
- (You can also add ECOD & COPOD)

## 🛠️ Installation

git clone <your-repo-url>
cd IOT_SENTINEL

python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows
pip install -r requirements.txt

text

## 🔧 Usage

1. **Train on Benign**

python src/train.py
--input data/Benign_Final/BenignTraffic.pcap.csv
--output results/backdoor_malware

text

2. **Evaluate on Backdoor & Benign**

python src/evaluate.py
--benign data/Benign_Final/BenignTraffic.pcap.csv
--malware data/Backdoor_Malware/Backdoor_Malware.pcap.csv
--models results/backdoor_malware/*.joblib
--output results/evaluation_curves

text

## 📈 Evaluation

We perform semi‐supervised evaluation:

- Train each model on benign traffic only.
- Score both benign (for false-positive rate) and backdoor (for true-positive rate) traffic.
- Compute ROC & Precision–Recall curves, AUCs, and TPR @ 5% FPR.

**Example: evaluate_performance() Stub**

from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt

def evaluate_performance(model_name, y_true, y_scores, results_dir):
# ROC
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

text
# PR
prec, recall, _ = precision_recall_curve(y_true, y_scores)
pr_auc = auc(recall, prec)

# TPR @ 5% FPR
idx = np.searchsorted(fpr, 0.05)

# Save summary
with open(results_dir / f'eval_{model_name}.txt', 'w') as f:
    f.write(f"{model_name} ROC AUC: {roc_auc:.4f}\n")
    f.write(f"{model_name} PR  AUC: {pr_auc:.4f}\n")
    f.write(f"TPR @ FPR=5%: {tpr[idx]:.4f}\n")

# Plot ROC
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve: {model_name}")
plt.legend()
plt.savefig(results_dir / f'roc_{model_name}.png')

# Plot PR
plt.figure()
plt.plot(recall, prec, label=f"AUC={pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall: {model_name}")
plt.legend()
plt.savefig(results_dir / f'pr_{model_name}.png')
plt.close('all')
text

## 🔍 Next Steps

- **Feature Engineering:** rolling‐window rates, interarrival‐time features, payload‐entropy.
- **Hyperparameter Tuning:** sweep contamination & neighbor counts.
- **Ensemble Strategies:** majority‐vote or score‐averaging across detectors.

---

# Backdoor Malware Dataset Analysis Report

**Generated on:** 2025-06-27 20:00:59

## Dataset Overview

- **Dataset Shape:** 3218 rows × 39 columns
- **File Path:** `../../../data/Backdoor_Malware/Backdoor_Malware.pcap.csv`
- **Missing Values:** 0 total missing values

## Dataset Information

### Basic Statistics

- **Total Records:** 3,218
- **Total Features:** 39
- **Memory Usage:** 0.96 MB

### Data Types Distribution

- **float64:** 30 columns
- **int64:** 9 columns

### Missing Values Summary

✅ No missing values found in the dataset.

## Data Visualizations

### Feature Distributions

- **Header_Length**
- **Protocol Type**
- **Time_To_Live**
- **Rate**
- **fin_flag_number**

### Feature Correlation Analysis

- **Feature Correlation Heatmap**

## Statistical Summary

### Key Numeric Features Summary

| Feature            | Mean      | Std        | Min    | Max        |
|--------------------|-----------|------------|--------|------------|
| Header_Length      | 21.0175   | 7.8722     | 4.0000 | 34.4000    |
| Protocol Type      | 9.0059    | 5.0562     | 0.0000 | 17.0000    |
| Time_To_Live       | 108.9486  | 45.6627    | 35.6000| 248.6000   |
| Rate               | 2891.9642 | 30984.4099 | 0.0104 | 1233618.82 |
| fin_flag_number    | 0.0188    | 0.0487     | 0.0000 | 0.4000     |
| syn_flag_number    | 0.0284    | 0.0600     | 0.0000 | 0.4000     |
| rst_flag_number    | 0.0017    | 0.0157     | 0.0000 | 0.4000     |
| psh_flag_number    | 0.1987    | 0.1666     | 0.0000 | 0.9000     |
| ack_flag_number    | 0.5833    | 0.3078     | 0.0000 | 1.0000     |
| ece_flag_number    | 0.0000    | 0.0000     | 0.0000 | 0.0000     |

## Key Insights

1. **Dataset Size:** The dataset contains 3,218 records with 39 features, suitable for machine learning analysis.
2. **Data Quality:** ✅ Clean dataset with no missing values
3. **Feature Types:** 
   - Numeric features: 39
   - Categorical features: 0
4. **Potential for Analysis:** This dataset appears well-suited for:
   - Anomaly detection using PyOD
   - Classification tasks
   - Network security analysis
   - IoT malware detection

## Recommendations for PyOD Analysis

1. **Preprocessing Steps:**
   - Consider feature scaling/normalization
   - Handle categorical variables if needed
   - Remove highly correlated features if necessary
2. **Suitable PyOD Algorithms:**
   - Isolation Forest
   - Local Outlier Factor (LOF)
   - One-Class SVM
   - AutoEncoder-based methods
3. **Evaluation Strategy:**
   - Use the label column for evaluation (if available)
   - Apply train-test split
   - Consider cross-validation for robust results

## Files Generated

- `dataset_info.txt` - Basic dataset information
- `summary_statistics.csv` - Statistical summary
- `missing_values.csv` - Missing values analysis
- `categorical_analysis.txt` - Categorical features analysis
- `first_5_rows.csv` - Sample data preview
- Various PNG files for visualizations

---

# Comprehensive Anomaly Detection Report

## IsolationForest

### Overview

- **Total data points:** 3218
- **Detected outliers:** 161
- **Contamination rate:** 5.0031%

### Artifacts

- **Predictions CSV:** predictions_IsolationForest.csv
- **Text report:** summary_report_IsolationForest.txt

### Statistical Summaries

- **Inlier score mean:** -0.0880, std: 0.0342
- **Outlier score mean:** 0.0332, std: 0.0297

### Top 5 Most Anomalous Points

|      | anomaly_score |
|------|---------------|
| 272  | 0.134584      |
| 618  | 0.129274      |
| 1324 | 0.126925      |
| 115  | 0.119369      |
| 2600 | 0.102811      |

## LOF

### Overview

- **Total data points:** 3218
- **Detected outliers:** 161
- **Contamination rate:** 5.0031%

### Artifacts

- **Predictions CSV:** predictions_LOF.csv
- **Text report:** summary_report_LOF.txt

### Statistical Summaries

- **Inlier score mean:** 1.0897, std: 0.1189
- **Outlier score mean:** 3.9057, std: 5.2637

### Top 5 Most Anomalous Points

|      | anomaly_score |
|------|---------------|
| 944  | 34.0908       |
| 1249 | 24.9757       |
| 1012 | 24.2089       |
| 927  | 22.2564       |
| 1224 | 21.9304       |

## KNN

### Overview

- **Total data points:** 3218
- **Detected outliers:** 161
- **Contamination rate:** 5.0031%

### Artifacts

- **Predictions CSV:** predictions_KNN.csv
- **Text report:** summary_report_KNN.txt

### Statistical Summaries

- **Inlier score mean:** 1.4408, std: 0.7688
- **Outlier score mean:** 6.0460, std: 6.6041

### Top 5 Most Anomalous Points

|      | anomaly_score |
|------|---------------|
| 3217 | 56.7556       |
| 927  | 55.7658       |
| 2001 | 30.8562       |
| 997  | 28.465        |
| 2891 | 18.5361       |

## AutoEncoder

### Overview

- **Total data points:** 3218
- **Detected outliers:** 161
- **Contamination rate:** 5.0031%

### Artifacts

- **Predictions CSV:** predictions_AutoEncoder.csv
- **Text report:** summary_report_AutoEncoder.txt

### Statistical Summaries

- **Inlier score mean:** 3.2550, std: 1.2553
- **Outlier score mean:** 11.2495, std: 6.5634

### Top 5 Most Anomalous Points

|      | anomaly_score |
|------|---------------|
| 3217 | 56.7863       |
| 927  | 56.1826       |
| 2001 | 39.7269       |
| 2891 | 35.985        |
| 997  | 26.5697       |

---

*Report generated using automated analysis script*... BACKDOOR_MALWARE