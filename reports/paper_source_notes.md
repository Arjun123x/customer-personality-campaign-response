# Paper Source Notes

Use these bullets as source material for the final CS439 paper. They are intentionally concise and are not written as final prose.

## Abstract Notes

- Project title: Customer Personality Analysis: Segmentation and Campaign Response Prediction.
- Dataset: Kaggle Customer Personality Analysis / marketing campaign customer dataset.
- Main tasks: unsupervised customer segmentation and supervised campaign response prediction.
- Cleaned analysis dataset: 2,213 customers after removing missing income rows and clearly unrealistic birth years.
- Clustering result: k = 4 retained because clusters were interpretable after profiling.
- Supervised target: `response`, where 1 means the customer responded to the most recent campaign.
- Response class balance: 333 responders (15.05%) and 1880 non-responders (84.95%).
- Main segmentation result: clusters differed in income, spending, purchase volume, household structure, campaign acceptance, and response rate.
- Main modeling result: cluster labels did not improve prediction from Feature Set A to B; historical campaign summaries improved Feature Set C.
- Recommended predictive model from tested set: C_full_with_campaign_history with Logistic Regression Balanced, PR-AUC = 0.631, F1 = 0.573, recall = 0.791.

## Introduction Notes

- Motivation: marketing campaigns benefit from understanding customer groups and predicting likely responders.
- Business-style question: can customer behavior data identify meaningful segments and predict campaign response?
- Data science question 1: do unsupervised clusters reveal interpretable customer segments?
- Data science question 2: does the unsupervised cluster label add predictive value for supervised response prediction?
- Data science question 3: how much does historical campaign behavior improve response prediction?
- Emphasize that the project combines segmentation for interpretation with classification for prediction.
- Avoid causal framing; results describe observed associations in this dataset.

## Related Work Notes

- Customer segmentation commonly uses unsupervised learning to group customers by demographics, spending, and behavioral variables.
- KMeans is a common baseline for interpretable segmentation when numeric features are scaled and skew/outliers are handled.
- Campaign response modeling is a binary classification problem often affected by class imbalance.
- For imbalanced classification, accuracy alone is insufficient; precision, recall, F1, ROC-AUC, and PR-AUC are more informative.
- Prior campaign behavior can be predictive if it occurs before the target campaign, but it requires a clear leakage assumption.

## Dataset and Preprocessing Notes

- Raw dataset: `data/raw/marketing_campaign.csv`.
- Raw file was tab-delimited despite the `.csv` extension.
- Raw data size: 2,240 rows, 29 columns.
- Cleaning output: `data/processed/customer_clean.csv`.
- Cleaned data size: 2,213 rows, 29 columns.
- Removed 24 rows with missing `Income`; income is central to segmentation and prediction, and missingness was about 1% of raw rows.
- Removed 3 clearly unrealistic birth-year records.
- Converted `Dt_Customer` to datetime and standardized columns to snake_case.
- Did not remove high-but-possible outliers during cleaning.
- Feature engineering output: `data/processed/customer_features.csv`, 2,213 rows and 53 columns.
- Engineered features included age, tenure, children/household variables, total spending, spending shares, purchase shares, average spend per purchase, and summarized campaign history.

## Methodology Notes

- Workflow: cleaning -> feature engineering -> EDA -> clustering -> cluster profiling -> target definition -> supervised modeling -> final visuals.
- Clustering used selected demographic, spending, purchase, recency, and campaign-history features.
- Clustering excluded `id`, `response`, raw dates, and raw categoricals.
- KMeans tested k = 2 through k = 8 using inertia and silhouette score.
- k = 4 was selected as a balance between interpretability and segmentation detail.
- Supervised target: `response`.
- Supervised experiments used one shared stratified 80/20 train/test split with `random_state = 42`.
- Preprocessing was fit inside scikit-learn pipelines to prevent test-data leakage.
- Categorical variables were one-hot encoded within the supervised modeling pipeline.

## Clustering Notes

- KMeans final k: 4.
- Final cluster counts:
- Cluster 0: Affluent High-Spend Catalog Customers; 667 customers (30.14%). Highest income and spending, strong catalog purchasing, fewest children, highest campaign acceptance and response rate.
- Cluster 1: Younger Low-Spend Budget Customers; 413 customers (18.66%). Youngest cluster with lowest median income, low spending and purchases, moderate child presence, and low campaign engagement.
- Cluster 2: Established Family Wine-Oriented Shoppers; 640 customers (28.92%). Older family-heavy customers with moderate-high income, strong purchase volume, high wine/luxury spending mix, and moderate campaign engagement.
- Cluster 3: Large-Household Low-Spend Store Shoppers; 493 customers (22.28%). Largest households and most children, low spending and purchases, strongest store/deal orientation, and lowest response rate.

- Cluster 0 had the highest response rate (25.04%) and highest spending.
- Cluster 3 had the lowest response rate (7.71%) and largest households.
- Clusters 1 and 3 were both low-spend segments but differed in household size, children, store/deal orientation, and response.
- Conclusion: clusters are useful for customer interpretation, even though they did not improve supervised prediction.

## Supervised Modeling Notes

- Input: `data/processed/modeling_data.csv`.
- Modeling data size: 2,213 rows, 32 columns.
- Target distribution: 333 positive responses (15.05%), 1880 negative responses (84.95%).
- Majority-class baseline accuracy: 0.850.
- Models tested for each feature set: Dummy Majority, Logistic Regression, Logistic Regression Balanced, Random Forest, Random Forest Balanced, Gradient Boosting.
- Feature Set A: baseline customer features only.
- Feature Set B: Feature Set A + `cluster`.
- Feature Set C: Feature Set B + `total_campaign_acceptances` and `any_campaign_acceptance`.
- Historical campaign features were excluded from A and B so the cluster-value test would not be obscured.
- Historical campaign summaries were allowed in C under the assumption that earlier campaign acceptances predate the target campaign response.
- Recommended model: Logistic Regression Balanced on C_full_with_campaign_history.

## Experiments/Results Notes

- Feature Set A best model: Gradient Boosting, PR-AUC = 0.575, F1 = 0.412, recall = 0.299.
- Feature Set B best model: Gradient Boosting, PR-AUC = 0.563, F1 = 0.412, recall = 0.299.
- Feature Set C best model: Logistic Regression Balanced, PR-AUC = 0.631, F1 = 0.573, recall = 0.791.
- Adding `cluster` changed PR-AUC by -0.012; this did not support predictive improvement.
- Adding campaign history changed PR-AUC by 0.068; this supported predictive improvement.
- Recommended model threshold 0.50: precision = 0.4492, recall = 0.7910, F1 = 0.5730, false positives = 65, false negatives = 14.
- Threshold 0.60 improved precision to 0.5000 and F1 to 0.5786 but increased false negatives to 21.

## Discussion Notes

- Clustering and classification answered different questions.
- Segments were interpretable and useful for describing customer groups.
- The cluster label did not add predictive value because it summarized information already present in the engineered features.
- Historical campaign behavior was more predictive than the cluster label.
- For campaign targeting, threshold selection depends on campaign cost and tolerance for false positives versus missed responders.
- Because the positive class is only about 15%, PR-AUC and recall are more informative than accuracy alone.

## Limitations Notes

- Dataset is historical and may not represent current customer behavior.
- Kaggle dataset does not provide all context about campaign timing and business costs.
- Historical campaign variables require the assumption that they precede the target campaign.
- Evaluation used one stratified train/test split rather than cross-validation.
- Models were not extensively tuned.
- Cluster labels are descriptive and should not be treated as causal explanations.
- Income includes an extreme value; clustering handled skew/outliers internally, but the original value remains in readable outputs.

## Future Work Notes

- Add cross-validation for more stable model estimates.
- Tune model hyperparameters systematically.
- Test alternative clustering algorithms or dimensionality-reduction approaches.
- Try cost-sensitive threshold selection based on campaign budget or contact cost.
- Explore calibrated probabilities.
- Compare with/without historical campaign variables under stricter leakage assumptions.
- Add a business-value simulation for targeting decisions.

## Conclusion Notes

- The project produced a full data science workflow from cleaning through segmentation, prediction, and final visuals.
- k = 4 segmentation is defensible for interpretation.
- Supervised modeling confirms the response target is imbalanced and requires metrics beyond accuracy.
- Cluster labels did not improve prediction when added to baseline customer features.
- Historical campaign summaries improved predictive performance.
- Final takeaway: segmentation is useful for understanding customers; historical behavior is more useful for predicting campaign response in this dataset.
