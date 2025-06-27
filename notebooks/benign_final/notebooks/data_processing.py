#!/usr/bin/env python3
"""
CIC IoT Dataset 2023 - Comprehensive Data Analysis Script
Author: AI Senior Engineer
Purpose: Analyze Benign_Final dataset and generate detailed report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from datetime import datetime
import json
from scipy import stats

warnings.filterwarnings('ignore')

# Set up directories
DATA_DIR = Path('data/Benign_Final/')
RESULTS_DIR = Path('results/')
RESULTS_DIR.mkdir(exist_ok=True)

# Configure visualization
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("CIC IoT Dataset 2023 - Benign Traffic Analysis")
print("="*80)

# ============================================================================
# STEP 1: Load Dataset
# ============================================================================
print("\n[STEP 1] Loading Dataset...")

# Define the file path
csv_file = DATA_DIR / 'BenignTraffic.pcap.csv'

# Check if file exists
if not csv_file.exists():
    print(f"ERROR: File not found at {csv_file}")
    print("Please ensure the dataset is in the correct location.")
    exit(1)

# Load the dataset
try:
    df = pd.read_csv(csv_file)
    print(f"✓ Successfully loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
except Exception as e:
    print(f"ERROR loading dataset: {e}")
    exit(1)

# ============================================================================
# STEP 2: Basic Information
# ============================================================================
print("\n[STEP 2] Analyzing Basic Information...")

# Create analysis report
report = []
report.append("# CIC IoT Dataset 2023 - Benign Traffic Analysis Report\n")
report.append(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
report.append("---\n\n")

# Basic stats
report.append("## 1. Dataset Overview\n")
report.append(f"- **Total Samples**: {df.shape[0]:,}\n")
report.append(f"- **Total Features**: {df.shape[1]}\n")
report.append(f"- **Memory Usage**: {df.memory_usage().sum() / 1024**2:.2f} MB\n\n")

# Column information
print("\n[INFO] Column Names and Types:")
column_info = pd.DataFrame({
    'Column': df.columns,
    'Type': df.dtypes.astype(str),
    'Non-Null Count': df.count(),
    'Null Count': df.isnull().sum(),
    'Null %': (df.isnull().sum() / len(df) * 100).round(2)
})

print(column_info.to_string())

# Save column info to report
report.append("## 2. Feature Information\n\n")
report.append("### Complete Feature List:\n")
report.append("| # | Feature Name | Data Type | Non-Null Count | Null Count | Null % |\n")
report.append("|---|--------------|-----------|----------------|------------|--------|\n")

for i, (idx, row) in enumerate(column_info.iterrows()):
    report.append(f"| {i+1} | {row['Column']} | {row['Type']} | "
                 f"{int(row['Non-Null Count']):,} | {int(row['Null Count']):,} | {row['Null %']}% |\n")

# ============================================================================
# STEP 3: Missing Values Analysis
# ============================================================================
print("\n[STEP 3] Analyzing Missing Values...")

missing_summary = df.isnull().sum()
missing_percent = (missing_summary / len(df)) * 100

report.append("\n## 3. Missing Values Analysis\n\n")
if missing_summary.sum() == 0:
    report.append("✓ **No missing values found in the dataset!**\n")
    print("✓ No missing values found!")
else:
    missing_df = pd.DataFrame({
        'Feature': missing_summary[missing_summary > 0].index,
        'Missing Count': missing_summary[missing_summary > 0].values,
        'Missing %': missing_percent[missing_summary > 0].values
    }).sort_values('Missing Count', ascending=False)
    
    report.append(f"⚠️ **Found {len(missing_df)} features with missing values:**\n\n")
    report.append("| Feature | Missing Count | Missing % |\n")
    report.append("|---------|--------------|------------|\n")
    for _, row in missing_df.iterrows():
        report.append(f"| {row['Feature']} | {row['Missing Count']:,} | {row['Missing %']:.2f}% |\n")

# ============================================================================
# STEP 4: Data Types Analysis
# ============================================================================
print("\n[STEP 4] Analyzing Data Types...")

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

report.append("\n## 4. Data Types Summary\n\n")
report.append(f"- **Numeric Features**: {len(numeric_cols)}\n")
report.append(f"- **Categorical Features**: {len(categorical_cols)}\n\n")

if categorical_cols:
    report.append("### Categorical Features:\n")
    for col in categorical_cols:
        unique_count = df[col].nunique()
        report.append(f"- **{col}**: {unique_count} unique values\n")
        if unique_count <= 20:  # Show value counts for small cardinality
            value_counts = df[col].value_counts()
            for val, count in value_counts.items():
                report.append(f"  - {val}: {count:,} ({count/len(df)*100:.2f}%)\n")

# ============================================================================
# STEP 5: Numeric Features Statistics
# ============================================================================
print("\n[STEP 5] Computing Numeric Statistics...")

if numeric_cols:
    # Basic statistics
    numeric_stats = df[numeric_cols].describe()
    
    # Additional statistics
    additional_stats = pd.DataFrame({
        'skewness': df[numeric_cols].skew(),
        'kurtosis': df[numeric_cols].kurtosis(),
        'zeros': (df[numeric_cols] == 0).sum(),
        'zeros_%': (df[numeric_cols] == 0).sum() / len(df) * 100,
        'unique': df[numeric_cols].nunique()
    })
    
    report.append("\n## 5. Numeric Features Analysis\n\n")
    report.append("### Top 10 Features by Mean Value:\n")
    top_features = numeric_stats.loc['mean'].sort_values(ascending=False).head(10)
    report.append("| Feature | Mean | Std | Min | Max |\n")
    report.append("|---------|------|-----|-----|-----|\n")
    
    for feat in top_features.index:
        report.append(f"| {feat} | {numeric_stats[feat]['mean']:.2e} | "
                     f"{numeric_stats[feat]['std']:.2e} | {numeric_stats[feat]['min']:.2e} | "
                     f"{numeric_stats[feat]['max']:.2e} |\n")

# ============================================================================
# STEP 6: Check for Infinity Values
# ============================================================================
print("\n[STEP 6] Checking for Infinity Values...")

inf_check = pd.DataFrame({
    'Feature': numeric_cols,
    'Positive Inf': [np.isinf(df[col]).sum() if col in numeric_cols else 0 for col in numeric_cols],
    'Negative Inf': [np.isneginf(df[col]).sum() if col in numeric_cols else 0 for col in numeric_cols]
})

inf_features = inf_check[(inf_check['Positive Inf'] > 0) | (inf_check['Negative Inf'] > 0)]

report.append("\n## 6. Infinity Values Check\n\n")
if len(inf_features) == 0:
    report.append("✓ **No infinity values found in the dataset!**\n")
    print("✓ No infinity values found!")
else:
    report.append(f"⚠️ **Found {len(inf_features)} features with infinity values:**\n\n")
    report.append("| Feature | Positive Inf | Negative Inf |\n")
    report.append("|---------|--------------|---------------|\n")
    for _, row in inf_features.iterrows():
        report.append(f"| {row['Feature']} | {row['Positive Inf']:,} | {row['Negative Inf']:,} |\n")

# ============================================================================
# STEP 7: Outlier Analysis
# ============================================================================
print("\n[STEP 7] Detecting Outliers...")

outlier_summary = {}
for col in numeric_cols[:20]:  # Analyze top 20 numeric columns
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_summary[col] = {
        'count': len(outliers),
        'percentage': len(outliers) / len(df) * 100
    }

# Sort by outlier percentage
outlier_df = pd.DataFrame(outlier_summary).T.sort_values('percentage', ascending=False).head(10)

report.append("\n## 7. Outlier Analysis (IQR Method)\n\n")
report.append("### Top 10 Features with Most Outliers:\n")
report.append("| Feature | Outlier Count | Outlier % |\n")
report.append("|---------|---------------|------------|\n")
for feat, data in outlier_df.iterrows():
    report.append(f"| {feat} | {int(data['count']):,} | {data['percentage']:.2f}% |\n")

# ============================================================================
# STEP 8: Correlation Analysis
# ============================================================================
print("\n[STEP 8] Computing Correlations...")

# Select numeric columns for correlation
if len(numeric_cols) > 0:
    # Sample data if too large
    if len(df) > 10000:
        sample_df = df[numeric_cols].sample(n=10000, random_state=42)
    else:
        sample_df = df[numeric_cols]
    
    # Compute correlation matrix
    corr_matrix = sample_df.corr()
    
    # Find highly correlated features
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    
    report.append("\n## 8. Correlation Analysis\n\n")
    if high_corr_pairs:
        report.append(f"### Highly Correlated Feature Pairs (|corr| > 0.8):\n")
        report.append("| Feature 1 | Feature 2 | Correlation |\n")
        report.append("|-----------|-----------|-------------|\n")
        for pair in high_corr_pairs[:10]:  # Show top 10
            report.append(f"| {pair['Feature 1']} | {pair['Feature 2']} | {pair['Correlation']:.3f} |\n")
    else:
        report.append("✓ No highly correlated feature pairs found (threshold: 0.8)\n")

# ============================================================================
# STEP 9: Generate Visualizations
# ============================================================================
print("\n[STEP 9] Creating Visualizations...")

# Create figure directory
fig_dir = RESULTS_DIR / 'figures'
fig_dir.mkdir(exist_ok=True)

# 1. Missing Values Heatmap (if any)
if missing_summary.sum() > 0:
    plt.figure(figsize=(12, 8))
    missing_df_viz = df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False)
    plt.bar(range(len(missing_df_viz)), missing_df_viz.values)
    plt.xticks(range(len(missing_df_viz)), missing_df_viz.index, rotation=45, ha='right')
    plt.ylabel('Missing Count')
    plt.title('Missing Values by Feature')
    plt.tight_layout()
    plt.savefig(fig_dir / 'missing_values.png', dpi=150)
    plt.close()

# 2. Feature Distribution Overview
if len(numeric_cols) > 0:
    # Select top 12 numeric features by variance
    top_features = df[numeric_cols].var().sort_values(ascending=False).head(12).index
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(top_features):
        data = df[col].dropna()
        if len(data) > 0:
            axes[idx].hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[idx].set_title(f'{col}\n(μ={data.mean():.2e}, σ={data.std():.2e})')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    plt.suptitle('Distribution of Top 12 Features (by Variance)', fontsize=16)
    plt.tight_layout()
    plt.savefig(fig_dir / 'feature_distributions.png', dpi=150)
    plt.close()

# 3. Correlation Heatmap (subset)
if len(numeric_cols) > 0:
    # Select subset of features for visualization
    subset_features = numeric_cols[:15]  # Top 15 features
    
    plt.figure(figsize=(12, 10))
    corr_subset = df[subset_features].corr()
    mask = np.triu(np.ones_like(corr_subset, dtype=bool))
    sns.heatmap(corr_subset, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, 
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix - Top 15 Features')
    plt.tight_layout()
    plt.savefig(fig_dir / 'correlation_heatmap.png', dpi=150)
    plt.close()

# 4. Box plots for outlier visualization
if len(numeric_cols) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols[:6]):
        data = df[col].dropna()
        if len(data) > 0:
            axes[idx].boxplot(data, vert=True, patch_artist=True,
                            boxprops=dict(facecolor='lightblue'),
                            medianprops=dict(color='red', linewidth=2))
            axes[idx].set_title(f'{col}')
            axes[idx].set_ylabel('Value')
            axes[idx].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.suptitle('Box Plots - Outlier Detection', fontsize=16)
    plt.tight_layout()
    plt.savefig(fig_dir / 'boxplots_outliers.png', dpi=150)
    plt.close()

report.append(f"\n## 9. Visualizations\n\n")
report.append(f"Generated {len(list(fig_dir.glob('*.png')))} visualizations in `{fig_dir}`\n")

# ============================================================================
# STEP 10: Save Results
# ============================================================================
print("\n[STEP 10] Saving Analysis Results...")

# Save the report
report_file = RESULTS_DIR / 'benign_traffic_analysis.md'
with open(report_file, 'w', encoding='utf-8') as f:
    f.writelines(report)

# Save detailed statistics as JSON
stats_dict = {
    'dataset_info': {
        'total_rows': int(df.shape[0]),
        'total_columns': int(df.shape[1]),
        'memory_usage_mb': float(df.memory_usage().sum() / 1024**2),
        'numeric_features': len(numeric_cols),
        'categorical_features': len(categorical_cols)
    },
    'missing_values': {
        'total_missing': int(missing_summary.sum()),
        'features_with_missing': int((missing_summary > 0).sum()),
        'missing_by_feature': missing_summary[missing_summary > 0].to_dict()
    },
    'numeric_stats': numeric_stats.to_dict() if len(numeric_cols) > 0 else {},
    'categorical_stats': {col: df[col].value_counts().to_dict() for col in categorical_cols},
    'outlier_summary': outlier_summary,
    'analysis_timestamp': datetime.now().isoformat()
}

stats_file = RESULTS_DIR / 'benign_traffic_stats.json'
with open(stats_file, 'w', encoding='utf-8') as f:
    json.dump(stats_dict, f, indent=2, default=str)

# Save column information as CSV
column_info.to_csv(RESULTS_DIR / 'column_information.csv', index=False, encoding='utf-8')

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"✓ Analysis report saved to: {report_file}")
print(f"✓ Detailed statistics saved to: {stats_file}")
print(f"✓ Column information saved to: {RESULTS_DIR / 'column_information.csv'}")
print(f"✓ Visualizations saved to: {fig_dir}")
print("="*80)

# Display summary statistics
print("\nDATASET SUMMARY:")
print(f"  Total Samples: {df.shape[0]:,}")
print(f"  Total Features: {df.shape[1]}")
print(f"  Missing Values: {missing_summary.sum():,}")
print(f"  Numeric Features: {len(numeric_cols)}")
print(f"  Categorical Features: {len(categorical_cols)}")

if categorical_cols and 'label' in categorical_cols:
    print("\nLABEL DISTRIBUTION:")
    for label, count in df['label'].value_counts().items():
        print(f"  {label}: {count:,} ({count/len(df)*100:.2f}%)")

print("\n✓ Analysis complete! Check the results folder for detailed reports.")