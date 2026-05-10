# Paper Metrics Tables

## Dataset Sizes

| Dataset / artifact | Rows | Columns | Notes |
| --- | --- | --- | --- |
| Raw marketing campaign data | 2,240 | 29 | Tab-delimited Kaggle source file |
| Cleaned customer data | 2,213 | 29 | Removed missing income rows and clearly invalid birth years |
| Engineered customer features | 2,213 | 53 | Added demographic, spending, purchase, and campaign-history features |
| Clustered customer data | 2,213 | 54 | Added k = 4 cluster label |
| Modeling data | 2,213 | 32 | 31 selected features plus target |
| Train split | 1,770 | - | Stratified split, random_state = 42 |
| Test split | 443 | - | 20% holdout test set |

## Target Distribution

| Class | Meaning | Count | Percentage |
| --- | --- | --- | --- |
| 0 | Did not respond | 1880 | 84.95% |
| 1 | Responded | 333 | 15.05% |

Majority-class baseline accuracy: **0.850**.

## Cluster Counts and Segment Names

| Cluster | Segment name | Size | Pct | Median income | Avg spending | Avg purchases | Response rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Affluent High-Spend Catalog Customers | 667 | 30.14% | $73,926 | 1,249.31 | 20.52 | 25.04% |
| 1 | Younger Low-Spend Budget Customers | 413 | 18.66% | $26,224 | 93.51 | 7.54 | 10.17% |
| 2 | Established Family Wine-Oriented Shoppers | 640 | 28.92% | $56,362 | 678.67 | 19.47 | 13.44% |
| 3 | Large-Household Low-Spend Store Shoppers | 493 | 22.28% | $36,921 | 75.21 | 7.47 | 7.71% |

## Feature-Set Comparison Metrics

| Feature set | Best model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_baseline_customer_features | Gradient Boosting | 0.871 | 0.667 | 0.299 | 0.412 | 0.863 | 0.575 |
| B_baseline_plus_cluster | Gradient Boosting | 0.871 | 0.667 | 0.299 | 0.412 | 0.863 | 0.563 |
| C_full_with_campaign_history | Logistic Regression Balanced | 0.822 | 0.449 | 0.791 | 0.573 | 0.906 | 0.631 |

## Top Model / Feature-Set Combinations by PR-AUC

| Feature set | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C_full_with_campaign_history | Logistic Regression Balanced | 0.822 | 0.449 | 0.791 | 0.573 | 0.906 | 0.631 | 311 | 65 | 14 | 53 |
| C_full_with_campaign_history | Logistic Regression | 0.878 | 0.651 | 0.418 | 0.509 | 0.903 | 0.630 | 361 | 15 | 39 | 28 |
| C_full_with_campaign_history | Gradient Boosting | 0.887 | 0.718 | 0.418 | 0.528 | 0.894 | 0.612 | 365 | 11 | 39 | 28 |
| C_full_with_campaign_history | Random Forest Balanced | 0.865 | 0.559 | 0.493 | 0.524 | 0.878 | 0.584 | 350 | 26 | 34 | 33 |
| C_full_with_campaign_history | Random Forest | 0.865 | 0.640 | 0.239 | 0.348 | 0.880 | 0.577 | 367 | 9 | 51 | 16 |
| A_baseline_customer_features | Gradient Boosting | 0.871 | 0.667 | 0.299 | 0.412 | 0.863 | 0.575 | 366 | 10 | 47 | 20 |
| B_baseline_plus_cluster | Gradient Boosting | 0.871 | 0.667 | 0.299 | 0.412 | 0.863 | 0.563 | 366 | 10 | 47 | 20 |
| B_baseline_plus_cluster | Random Forest Balanced | 0.842 | 0.475 | 0.433 | 0.453 | 0.868 | 0.550 | 344 | 32 | 38 | 29 |

## Recommended Model Threshold Tradeoff

Recommended model: **Logistic Regression Balanced** on **C_full_with_campaign_history**.

| Threshold | Precision | Recall | F1 | False positives | False negatives |
| --- | --- | --- | --- | --- | --- |
| 0.20 | 0.3249 | 0.9552 | 0.4848 | 133 | 3 |
| 0.30 | 0.3728 | 0.9403 | 0.5339 | 106 | 4 |
| 0.40 | 0.4196 | 0.8955 | 0.5714 | 83 | 7 |
| 0.50 | 0.4492 | 0.7910 | 0.5730 | 65 | 14 |
| 0.60 | 0.5000 | 0.6866 | 0.5786 | 46 | 21 |
