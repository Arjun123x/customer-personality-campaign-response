# Rubric Alignment Checklist

## Problem Formulation & Motivation

### What the project satisfies

- Defines a clear applied problem: customer segmentation and campaign response prediction.
- Includes two research directions: unsupervised segmentation and supervised classification.
- Tests whether segmentation adds predictive value through controlled feature sets.
- Uses an imbalanced campaign-response target and discusses metric choice.

### What still needs manual work

- Write final motivation in polished prose.
- Explicitly connect the project to CS439 course themes and any required course-paper framing.
- Add citations or references if required by the template.

### Self-assessment

- Excellent

## Data Handling & Preprocessing

### What the project satisfies

- Inspected raw data structure before transformations.
- Handled tab-delimited file loading.
- Removed missing income rows and clearly invalid birth years conservatively.
- Engineered interpretable demographic, spending, purchase, tenure, household, and campaign-history features.
- Preserved raw and intermediate processed datasets separately.

### What still needs manual work

- In the paper, clearly justify missing income handling and birth-year removal.
- Briefly mention the extreme income value and how clustering handled skew/outliers internally.

### Self-assessment

- Excellent

## Methodology & Technical Depth

### What the project satisfies

- Uses KMeans clustering with scaled transformed features.
- Evaluates k = 2 through k = 8 using inertia and silhouette score.
- Profiles k = 4 clusters for interpretability.
- Defines a supervised target and leakage-controlled modeling dataset.
- Compares controlled feature sets A, B, and C.
- Uses scikit-learn pipelines for preprocessing and modeling.

### What still needs manual work

- Explain why k = 4 was retained despite k = 2 having the strongest silhouette score.
- Present the historical campaign feature assumption carefully.
- Avoid making the methods section too long for the final paper format.

### Self-assessment

- Excellent

## Experiments, Evaluation & Results

### What the project satisfies

- Reports cluster sizes, segment names, and profile differences.
- Evaluates supervised models with accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and confusion-matrix counts.
- Includes DummyClassifier baseline.
- Compares Feature Set A vs B to test cluster predictive value.
- Compares Feature Set B vs C to test historical campaign behavior.
- Includes threshold analysis for the recommended model.

### What still needs manual work

- Choose which tables and figures fit within paper length.
- Convert selected markdown tables into LaTeX table format.
- Discuss results honestly without overstating predictive performance.

### Self-assessment

- Excellent

## Writing, Formatting & Reproducibility

### What the project satisfies

- Includes executed notebooks for each major step.
- Includes reusable scripts in `src/`.
- Includes final visuals and summary CSVs in `reports/`.
- Includes a reproducibility checklist and README instructions.
- Keeps generated outputs separate from raw data.

### What still needs manual work

- Write the final LaTeX paper manually using the course template.
- Add citations, figure references, table references, and final formatting.
- Confirm the final paper meets page, font, and submission requirements.
- Add GitHub link when repository is published.

### Self-assessment

- Proficient
