IoT Sentinel: (SEMI) --- Unsupervised Anomaly Detection in IoT Networks
This project focuses on identifying anomalous network traffic within an Internet of Things (IoT) environment using unsupervised machine learning techniques. By training models exclusively on benign traffic from the CIC-IoT-Dataset-2023, we build a baseline of "normal" behavior, then test on real backdoor‐malware captures to measure detection performance.

🚀 Project Overview
Goal: Detect deviations from benign IoT network behavior (zero‐day, signature‐free).

Datasets:

Benign: BenignTraffic.pcap.csv

Malware: Backdoor_Malware.pcap.csv

Models: PyOD outlier detectors trained on benign only, then evaluated on both benign & backdoor.

Isolation Forest (IForest)

LOF

KNN

AutoEncoder

(You can also add ECOD & COPOD)

📁 Repository Structure
├── data/
│   ├── Benign_Final/
│   │   └── BenignTraffic.pcap.csv
│   └── Backdoor_Malware/
│       └── Backdoor_Malware.pcap.csv
├── notebooks/
│   ├── 0_data_ingest.ipynb
│   ├── 1_train_on_benign.ipynb
│   └── 2_test_on_backdoor.ipynb
├── results/
│   ├── backdoor_malware/
│   └── evaluation_curves/
├── src/
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
└── README.md

🛠️ Installation
git clone <your-repo-url>
cd IOT_SENTINEL

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

🔧 Usage
1. Train on Benign

python src/train.py \
  --input data/Benign_Final/BenignTraffic.pcap.csv \
  --output results/backdoor_malware

2. Evaluate on Backdoor & Benign

python src/evaluate.py \
  --benign data/Benign_Final/BenignTraffic.pcap.csv \
  --malware data/Backdoor_Malware/Backdoor_Malware.pcap.csv \
  --models results/backdoor_malware/*.joblib \
  --output results/evaluation_curves

📈 Evaluation
We perform semi‐supervised evaluation:

Train each model on benign traffic only.

Score both benign (for false-positive rate) and backdoor (for true-positive rate) traffic.

Compute ROC & Precision–Recall curves, AUCs, and TPR @ 5% FPR.

Example: evaluate_performance() Stub
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt

def evaluate_performance(model_name, y_true, y_scores, results_dir):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

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

🔍 Next Steps
Feature Engineering: rolling‐window rates, interarrival‐time features, payload‐entropy.

Hyperparameter Tuning: sweep contamination & neighbor counts.

Ensemble Strategies: majority‐vote or score‐averaging across detectors.