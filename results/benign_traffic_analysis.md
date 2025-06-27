# CIC IoT Dataset 2023 - Benign Traffic Analysis Report
**Analysis Date**: 2025-06-27 17:51:53
---

## 1. Dataset Overview
- **Total Samples**: 362,361
- **Total Features**: 39
- **Memory Usage**: 107.82 MB

## 2. Feature Information

### Complete Feature List:
| # | Feature Name | Data Type | Non-Null Count | Null Count | Null % |
|---|--------------|-----------|----------------|------------|--------|
| 1 | Header_Length | float64 | 362,361 | 0 | 0.0% |
| 2 | Protocol Type | int64 | 362,361 | 0 | 0.0% |
| 3 | Time_To_Live | float64 | 362,361 | 0 | 0.0% |
| 4 | Rate | float64 | 362,361 | 0 | 0.0% |
| 5 | fin_flag_number | float64 | 362,361 | 0 | 0.0% |
| 6 | syn_flag_number | float64 | 362,361 | 0 | 0.0% |
| 7 | rst_flag_number | float64 | 362,361 | 0 | 0.0% |
| 8 | psh_flag_number | float64 | 362,361 | 0 | 0.0% |
| 9 | ack_flag_number | float64 | 362,361 | 0 | 0.0% |
| 10 | ece_flag_number | float64 | 362,361 | 0 | 0.0% |
| 11 | cwr_flag_number | float64 | 362,361 | 0 | 0.0% |
| 12 | ack_count | int64 | 362,361 | 0 | 0.0% |
| 13 | syn_count | int64 | 362,361 | 0 | 0.0% |
| 14 | fin_count | int64 | 362,361 | 0 | 0.0% |
| 15 | rst_count | int64 | 362,361 | 0 | 0.0% |
| 16 | HTTP | float64 | 362,361 | 0 | 0.0% |
| 17 | HTTPS | float64 | 362,361 | 0 | 0.0% |
| 18 | DNS | float64 | 362,361 | 0 | 0.0% |
| 19 | Telnet | float64 | 362,361 | 0 | 0.0% |
| 20 | SMTP | float64 | 362,361 | 0 | 0.0% |
| 21 | SSH | float64 | 362,361 | 0 | 0.0% |
| 22 | IRC | float64 | 362,361 | 0 | 0.0% |
| 23 | TCP | float64 | 362,361 | 0 | 0.0% |
| 24 | UDP | float64 | 362,361 | 0 | 0.0% |
| 25 | DHCP | float64 | 362,361 | 0 | 0.0% |
| 26 | ARP | float64 | 362,361 | 0 | 0.0% |
| 27 | ICMP | float64 | 362,361 | 0 | 0.0% |
| 28 | IGMP | float64 | 362,361 | 0 | 0.0% |
| 29 | IPv | float64 | 362,361 | 0 | 0.0% |
| 30 | LLC | float64 | 362,361 | 0 | 0.0% |
| 31 | Tot sum | int64 | 362,361 | 0 | 0.0% |
| 32 | Min | int64 | 362,361 | 0 | 0.0% |
| 33 | Max | int64 | 362,361 | 0 | 0.0% |
| 34 | AVG | float64 | 362,361 | 0 | 0.0% |
| 35 | Std | float64 | 362,342 | 19 | 0.01% |
| 36 | Tot size | float64 | 362,361 | 0 | 0.0% |
| 37 | IAT | float64 | 362,361 | 0 | 0.0% |
| 38 | Number | int64 | 362,361 | 0 | 0.0% |
| 39 | Variance | float64 | 362,342 | 19 | 0.01% |

## 3. Missing Values Analysis

⚠️ **Found 2 features with missing values:**

| Feature | Missing Count | Missing % |
|---------|--------------|------------|
| Std | 19 | 0.01% |
| Variance | 19 | 0.01% |

## 4. Data Types Summary

- **Numeric Features**: 39
- **Categorical Features**: 0


## 5. Numeric Features Analysis

### Top 10 Features by Mean Value:
| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| Rate | inf | nan | 7.70e+00 | inf |
| Variance | 6.48e+05 | 2.00e+06 | 0.00e+00 | 1.36e+08 |
| Tot sum | 5.48e+03 | 6.38e+03 | 6.00e+01 | 9.43e+04 |
| Max | 1.52e+03 | 1.96e+03 | 6.00e+01 | 3.77e+04 |
| Tot size | 5.48e+02 | 6.39e+02 | 6.00e+01 | 1.36e+04 |
| AVG | 5.48e+02 | 6.39e+02 | 6.00e+01 | 1.36e+04 |
| Std | 4.96e+02 | 6.34e+02 | 0.00e+00 | 1.17e+04 |
| Min | 1.18e+02 | 2.71e+02 | 6.00e+01 | 1.36e+04 |
| Time_To_Live | 1.09e+02 | 4.84e+01 | 0.00e+00 | 2.51e+02 |
| Header_Length | 2.67e+01 | 6.98e+00 | 0.00e+00 | 6.00e+01 |

## 6. Infinity Values Check

⚠️ **Found 1 features with infinity values:**

| Feature | Positive Inf | Negative Inf |
|---------|--------------|---------------|
| Rate | 19 | 0 |

## 7. Outlier Analysis (IQR Method)

### Top 10 Features with Most Outliers:
| Feature | Outlier Count | Outlier % |
|---------|---------------|------------|
| syn_flag_number | 79,604 | 21.97% |
| syn_count | 79,604 | 21.97% |
| fin_flag_number | 68,303 | 18.85% |
| fin_count | 68,303 | 18.85% |
| Rate | 56,477 | 15.59% |
| HTTP | 48,678 | 13.43% |
| Protocol Type | 37,738 | 10.41% |
| DNS | 16,408 | 4.53% |
| Time_To_Live | 12,749 | 3.52% |
| Header_Length | 5,181 | 1.43% |

## 8. Correlation Analysis

### Highly Correlated Feature Pairs (|corr| > 0.8):
| Feature 1 | Feature 2 | Correlation |
|-----------|-----------|-------------|
| Header_Length | ack_flag_number | 0.893 |
| Header_Length | ack_count | 0.892 |
| Header_Length | TCP | 0.906 |
| Header_Length | UDP | -0.839 |
| fin_flag_number | fin_count | 1.000 |
| syn_flag_number | syn_count | 1.000 |
| rst_flag_number | rst_count | 1.000 |
| ack_flag_number | ack_count | 1.000 |
| ack_flag_number | TCP | 0.987 |
| ack_flag_number | UDP | -0.943 |

## 9. Visualizations

Generated 4 visualizations in `results\figures`
