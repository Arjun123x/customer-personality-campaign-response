# Reproducibility Checklist

## Environment Setup

- Create a virtual environment in the project root: `python3 -m venv .venv`
- Activate on macOS/Linux: `source .venv/bin/activate`
- Activate on Windows PowerShell: `.venv\Scripts\Activate.ps1`
- Install dependencies: `pip install -r requirements.txt`
- Required packages are listed in `requirements.txt`: pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter.

## Raw Data Placement

- Place the Kaggle Customer Personality Analysis file at `data/raw/marketing_campaign.csv`.
- Keep the raw data unmodified.
- The project expects the Kaggle file to be tab-delimited.

## Notebook Execution Order

1. `notebooks/01_data_cleaning.ipynb`
2. `notebooks/02_feature_engineering.ipynb`
3. `notebooks/03_eda.ipynb`
4. `notebooks/04_clustering.ipynb`
5. `notebooks/05_cluster_profiling.ipynb`
6. `notebooks/06_target_definition.ipynb`
7. `notebooks/07_model_training_evaluation.ipynb`
8. `notebooks/08_final_visuals.ipynb`

## Script Descriptions

- `src/data_cleaning.py`: loads raw data, handles delimiter detection, standardizes names, removes missing income rows and clearly invalid birth years, saves `customer_clean.csv`.
- `src/feature_engineering.py`: creates demographic, spending, purchase, and campaign-history features, saves `customer_features.csv`.
- `src/clustering.py`: prepares clustering features, handles skew/outliers inside the clustering pipeline, evaluates k values, fits k = 4 KMeans, saves `customer_clustered.csv`.
- `src/modeling_data.py`: defines `response` as the supervised target, selects modeling features, saves `modeling_data.csv` and target summary.
- `src/model_training.py`: trains and evaluates controlled supervised experiments for Feature Sets A, B, and C, saves evaluation summaries.

## Expected Generated Outputs

- `data/processed/customer_clean.csv`
- `data/processed/customer_features.csv`
- `data/processed/customer_clustered.csv`
- `data/processed/modeling_data.csv`
- `reports/cluster_profile_summary.csv`
- `reports/target_definition_summary.csv`
- `reports/model_evaluation_summary.csv`
- `reports/feature_set_comparison_summary.csv`
- `reports/final_visuals_summary.csv`
- Final paper-ready visuals in `reports/figures/final_*.png`

## Random State Values

- Clustering KMeans: `random_state = 42`
- PCA visualization in clustering: `random_state = 42`
- Supervised train/test split: `random_state = 42`, `test_size = 0.20`, stratified by `response`
- Supervised models with random behavior: `random_state = 42`

## Leakage Prevention Notes

- `response` is never used as a clustering input or supervised feature.
- `id` is excluded from clustering and supervised modeling.
- Raw `dt_customer` is excluded from supervised modeling; tenure features are used instead.
- Individual `accepted_cmp1` through `accepted_cmp5` are excluded from final modeling data.
- `total_campaign_acceptances` and `any_campaign_acceptance` are used only under the assumption that historical campaign behavior occurred before the target campaign.
- Supervised preprocessing is fit inside scikit-learn pipelines after the train/test split.
- Feature Set A and Feature Set B exclude campaign-history variables to isolate the value of `cluster`.

## Final Report and Figure Locations

- Source notes: `reports/paper_source_notes.md`
- Metrics tables: `reports/paper_metrics_table.md`
- Figure plan: `reports/paper_figure_plan.md`
- Reproducibility checklist: `reports/reproducibility_checklist.md`
- Rubric checklist: `reports/rubric_alignment_checklist.md`
- Final visuals: `reports/figures/final_*.png`
