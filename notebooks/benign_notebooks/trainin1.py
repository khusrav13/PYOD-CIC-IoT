import pandas as pd
import numpy as np
import joblib
import time
import json
from pathlib import Path
from datetime import datetime

# Import PyOD models
from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD

# Import utilities
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("Enhanced PyOD Model Training & Analysis on CIC-IoT-Dataset-2023")
print("="*80)

# ============================================================================
# STEP 1: Set up Paths and Configuration
# ============================================================================
print("\n[STEP 1] Configuring paths and models...")

# Define project directories
DATA_FILE = Path('data/Benign_Final/BenignTraffic.pcap.csv')
BASE_RESULTS_DIR = Path('results/training_results_enhanced/')
VIZ_DIR = BASE_RESULTS_DIR / 'visualizations'

# Create directories if they don't exist
BASE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(exist_ok=True)
print(f"✓ Results will be saved to: {BASE_RESULTS_DIR}")
print(f"✓ Visualizations will be saved to: {VIZ_DIR}")


# ============================================================================
# STEP 2: Load and Preprocess Data
# ============================================================================
print("\n[STEP 2] Loading and preprocessing data...")

# Check if the data file exists
if not DATA_FILE.exists():
    print(f"❌ ERROR: Data file not found at {DATA_FILE}")
    exit(1)

df = pd.read_csv(DATA_FILE)
print(f"✓ Successfully loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# --- Preprocessing ---
# 1. Store original column order
original_columns = df.columns.tolist()
joblib.dump(original_columns, BASE_RESULTS_DIR / 'training_columns.joblib')
print(f"✓ Saved training column list to ensure consistent feature order.")

# 2. Handle infinity and missing values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
print(f"  - Found {df.isnull().any().sum()} columns with missing values before imputation.")
df.fillna(df.median(), inplace=True)
print(f"✓ Imputed all missing values using column medians.")

# 3. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
joblib.dump(scaler, BASE_RESULTS_DIR / 'scaler.joblib')
print(f"✓ Scaled data using StandardScaler and saved the scaler object.")

# 4. Dimensionality Reduction for Visualization (using PCA)
print("\n[STEP 3] Performing PCA for 2D visualization...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("✓ Reduced data to 2 dimensions using PCA.")


# ============================================================================
# STEP 4: Train Models, Analyze Scores, and Create Visuals
# ============================================================================
print("\n[STEP 4] Training models and generating artifacts...")

models_to_train = {
    'iforest': IForest(contamination=0.01, random_state=42, n_jobs=-1),
    'ecod': ECOD(contamination=0.01, n_jobs=-1),
    'copod': COPOD(contamination=0.01, n_jobs=-1)
}

training_results = {}

for name, model in models_to_train.items():
    print(f"\n--- Processing {name.upper()} ---")
    
    # --- Train Model ---
    start_time = time.time()
    model.fit(X_scaled)
    training_time = time.time() - start_time
    print(f"✓ Training complete in {training_time:.2f} seconds.")
    
    # Save the trained model
    model_path = BASE_RESULTS_DIR / f'{name}_model.joblib'
    joblib.dump(model, model_path)
    print(f"✓ Saved trained model to: {model_path}")
    
    # --- Analyze Scores ---
    decision_scores = model.decision_scores_
    score_stats = {
        'mean': np.mean(decision_scores),
        'median': np.median(decision_scores),
        'std': np.std(decision_scores),
        'min': np.min(decision_scores),
        'max': np.max(decision_scores)
    }
    print(f"✓ Score statistics: Mean={score_stats['mean']:.3f}, Median={score_stats['median']:.3f}, Std={score_stats['std']:.3f}")

    # --- Create Visualizations ---
    # 1. Histogram of decision scores
    plt.figure(figsize=(10, 6))
    sns.histplot(decision_scores, bins=100, kde=True, color='royalblue')
    plt.title(f'{name.upper()}: Distribution of Outlier Scores', fontsize=16)
    plt.xlabel('Decision Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    hist_path = VIZ_DIR / f'{name}_scores_distribution.png'
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved score distribution plot: {hist_path}")

    # 2. 2D PCA Scatter Plot with Outlier Scores
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=decision_scores, cmap='viridis', s=5, alpha=0.6)
    plt.colorbar(scatter, label='Outlier Score')
    plt.title(f'{name.upper()}: 2D PCA with Outlier Scores', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    scatter_path = VIZ_DIR / f'{name}_pca_scatter.png'
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved 2D scatter plot: {scatter_path}")

    # Store results for the report
    training_results[name] = {
        'model_path': model_path,
        'training_time': training_time,
        'score_stats': score_stats,
        'hist_path': hist_path,
        'scatter_path': scatter_path,
        'params': model.get_params()
    }


# ============================================================================
# STEP 5: Generate Comprehensive Markdown Report
# ============================================================================
print("\n[STEP 5] Generating enhanced summary report...")

report_path = BASE_RESULTS_DIR / 'enhanced_training_report.md'
report_lines = [
    f"# Enhanced PyOD Model Training Report\n",
    f"**Report Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    "---\n\n",
    "## 1. Executive Summary\n",
    "This report details the training and analysis of three unsupervised outlier detection models (IForest, ECOD, COPOD) on the CIC-IoT-Dataset-2023 benign traffic data. The process included data preprocessing, model training, and a detailed analysis of the resulting outlier scores. All artifacts, including trained models, scalers, and visualizations, have been saved for reproducibility.\n",
    
    "### Summary of Trained Models\n",
    "| Model | Training Time (s) | Mean Score | Median Score | Saved Model File |\n",
    "|---|---|---|---|---|\n"
]

for name, results in training_results.items():
    report_lines.append(
        f"| {name.upper()} | {results['training_time']:.2f} | {results['score_stats']['mean']:.4f} | {results['score_stats']['median']:.4f} | `{results['model_path'].name}` |"
    )

report_lines.extend([
    "\n\n## 2. Setup and Preprocessing\n",
    f"- **Input Data**: `{DATA_FILE}` ({df.shape[0]:,} rows × {df.shape[1]} features)\n",
    "- **Preprocessing Pipeline**:\n",
    "  1. Replaced `infinity` values with `NaN`.\n",
    "  2. Imputed all `NaN` values using column-wise medians.\n",
    "  3. Standardized all features using `sklearn.preprocessing.StandardScaler`.\n",
    f"- **Saved Artifacts**:\n",
    f"  - **Scaler**: `scaler.joblib`\n",
    f"  - **Feature List**: `training_columns.joblib`\n"
])

report_lines.append("\n---\n\n## 3. Model-Specific Analysis\n")

for name, results in training_results.items():
    report_lines.extend([
        f"\n### 3.1 {name.upper()} Analysis\n",
        f"- **Training Time**: {results['training_time']:.2f} seconds\n",
        f"- **Score Statistics**:\n"
        f"  - **Mean**: {results['score_stats']['mean']:.4f}\n"
        f"  - **Median**: {results['score_stats']['median']:.4f}\n"
        f"  - **Std. Dev**: {results['score_stats']['std']:.4f}\n"
        f"  - **Min / Max**: {results['score_stats']['min']:.4f} / {results['score_stats']['max']:.4f}\n",
        f"#### Score Distribution\n",
        "This histogram shows the distribution of outlier scores assigned to the training data. A long tail typically indicates good separation between inliers and potential outliers.\n",
        f"![{name} Score Distribution]({results['hist_path'].relative_to(BASE_RESULTS_DIR)})\n",
        f"#### 2D PCA Scatter Plot\n",
        "This plot shows the data in 2D (via PCA), with points colored by their outlier score. It helps visualize which regions of the data space are considered anomalous.\n",
        f"![{name} PCA Scatter Plot]({results['scatter_path'].relative_to(BASE_RESULTS_DIR)})\n"
    ])

try:
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"✓ Successfully saved enhanced report to: {report_path}")
except Exception as e:
    print(f"❌ ERROR saving report: {e}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ENHANCED TRAINING PROCESS COMPLETE!")
print("="*80)
print("Summary of all generated files:")
for f in sorted(BASE_RESULTS_DIR.rglob('*')):
    print(f"  - {f.relative_to(BASE_RESULTS_DIR.parent)}")
print("="*80)

