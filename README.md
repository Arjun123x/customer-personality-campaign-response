# Customer Personality Analysis

Customer Personality Analysis: Segmentation and Campaign Response Prediction

GitHub link: `TODO: add repository URL`

## Project Objective

This project analyzes the Kaggle Customer Personality Analysis / marketing campaign dataset using a complete data science workflow. The project combines unsupervised learning for customer segmentation with supervised learning for campaign response prediction.

## Research Questions

1. Can customer demographics, spending behavior, purchase behavior, and campaign history be used to create interpretable customer segments?
2. Does the unsupervised cluster label add predictive value for predicting campaign response?
3. How much does historical campaign behavior improve supervised response prediction?

## Final Findings Summary

- Cleaned analysis dataset: 2,213 customers.
- Clustering solution: k = 4 retained because the clusters were interpretable after profiling.
- Segment labels:
  - Cluster 0: Affluent High-Spend Catalog Customers (30.14%)
  - Cluster 1: Younger Low-Spend Budget Customers (18.66%)
  - Cluster 2: Established Family Wine-Oriented Shoppers (28.92%)
  - Cluster 3: Large-Household Low-Spend Store Shoppers (22.28%)
- Supervised target: `response`, with 333 responders (15.05%) and 1880 non-responders (84.95%).
- Cluster labels were useful for interpretation but did not improve supervised prediction from Feature Set A to B.
- Historical campaign summaries improved predictive performance in Feature Set C.
- Recommended tested model: Logistic Regression Balanced on C_full_with_campaign_history with PR-AUC = 0.631, F1 = 0.573, and recall = 0.791.

## Project Structure

```text
data/
  raw/
    marketing_campaign.csv
  processed/
    customer_clean.csv
    customer_features.csv
    customer_clustered.csv
    modeling_data.csv
notebooks/
  01_data_cleaning.ipynb
  02_feature_engineering.ipynb
  03_eda.ipynb
  04_clustering.ipynb
  05_cluster_profiling.ipynb
  06_target_definition.ipynb
  07_model_training_evaluation.ipynb
  08_final_visuals.ipynb
src/
  data_cleaning.py
  feature_engineering.py
  clustering.py
  modeling_data.py
  model_training.py
reports/
  figures/
  cluster_profile_summary.csv
  target_definition_summary.csv
  model_evaluation_summary.csv
  feature_set_comparison_summary.csv
  final_visuals_summary.csv
  paper_source_notes.md
  paper_metrics_table.md
  paper_figure_plan.md
  reproducibility_checklist.md
  rubric_alignment_checklist.md
```

## Environment Setup

Create and activate a virtual environment from the project root.

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies only inside the virtual environment:

```bash
pip install -r requirements.txt
```

## Raw Data Placement

Download the Kaggle marketing campaign / customer personality dataset and place the raw, unmodified file at:

```text
data/raw/marketing_campaign.csv
```

The raw file should remain unchanged. The project loader handles the tab-delimited format.

## Notebook Execution Order

Run notebooks from the project root in this order:

1. `notebooks/01_data_cleaning.ipynb`
2. `notebooks/02_feature_engineering.ipynb`
3. `notebooks/03_eda.ipynb`
4. `notebooks/04_clustering.ipynb`
5. `notebooks/05_cluster_profiling.ipynb`
6. `notebooks/06_target_definition.ipynb`
7. `notebooks/07_model_training_evaluation.ipynb`
8. `notebooks/08_final_visuals.ipynb`

Launch Jupyter with:

```bash
jupyter notebook
```

## Script Descriptions

- `src/data_cleaning.py`: loads and cleans the raw dataset, then saves `data/processed/customer_clean.csv`.
- `src/feature_engineering.py`: creates interpretable customer features and saves `data/processed/customer_features.csv`.
- `src/clustering.py`: evaluates KMeans cluster counts, fits k = 4, and saves `data/processed/customer_clustered.csv`.
- `src/modeling_data.py`: defines `response` as the supervised target and saves `data/processed/modeling_data.csv`.
- `src/model_training.py`: trains and evaluates controlled supervised experiments and saves model evaluation reports.

## Final Outputs

- Final visuals: `reports/figures/final_*.png`
- Cluster profile summary: `reports/cluster_profile_summary.csv`
- Model evaluation summary: `reports/model_evaluation_summary.csv`
- Feature-set comparison summary: `reports/feature_set_comparison_summary.csv`
- Final visuals summary: `reports/final_visuals_summary.csv`
- Paper source notes: `reports/paper_source_notes.md`
- Paper metrics tables: `reports/paper_metrics_table.md`
- Paper figure plan: `reports/paper_figure_plan.md`
- Reproducibility checklist: `reports/reproducibility_checklist.md`
- Rubric alignment checklist: `reports/rubric_alignment_checklist.md`

## Important Modeling Notes

- `response` is the supervised target and is never used as a feature.
- Feature Set A excludes cluster and campaign-history variables.
- Feature Set B adds cluster only.
- Feature Set C adds summarized historical campaign behavior.
- Individual campaign flags `accepted_cmp1` through `accepted_cmp5` are excluded from final modeling.
- Metrics such as PR-AUC, recall, and F1 are emphasized because the positive response class is imbalanced.
