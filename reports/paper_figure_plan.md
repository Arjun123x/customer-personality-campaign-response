# Paper Figure Plan

| Filename | Suggested paper section | Caption draft | Key message | Placement |
| --- | --- | --- | --- | --- |
| final_project_workflow.png | Introduction / Methodology | Project workflow from raw customer campaign data through cleaning, feature engineering, exploratory analysis, clustering, cluster profiling, target definition, supervised modeling, and final findings. | Shows the full end-to-end data science pipeline. | Main paper |
| final_cluster_profile_heatmap.png | Clustering Results | Standardized cluster profile heatmap comparing each segment against the overall customer average on income, spending, purchase behavior, household structure, campaign acceptance, and response. | The four clusters differ in interpretable customer-profile dimensions. | Main paper |
| final_cluster_sizes.png | Clustering Results | Customer counts and percentages for the four retained KMeans clusters. | The retained segments are reasonably sized, ranging from about 19% to 30% of the dataset. | Appendix |
| final_cluster_segment_comparison.png | Clustering Results | Comparison of clusters on median income, average total spending, average purchase count, response rate, and average number of children. | Cluster 0 is high-income/high-spend, clusters 1 and 3 are lower-spend but structurally different, and cluster 2 is family-heavy with strong purchasing. | Main paper |
| final_response_rate_by_cluster.png | Clustering Results / Discussion | Recent campaign response rate by cluster. | Response rates differ by segment, with Cluster 0 highest and Cluster 3 lowest. | Main paper |
| final_feature_set_comparison.png | Supervised Modeling Results | Controlled feature-set comparison for baseline customer features, baseline plus cluster, and full model with historical campaign summaries. | Adding cluster did not improve prediction, while historical campaign summaries improved performance. | Main paper |
| final_model_comparison.png | Supervised Modeling Results | Top model and feature-set combinations compared by PR-AUC and positive-class F1. | The strongest PR-AUC came from Feature Set C with balanced logistic regression. | Appendix |
| final_threshold_tradeoff.png | Supervised Modeling Results / Discussion | Precision, recall, F1, false positives, and false negatives for the recommended model across decision thresholds. | Threshold selection controls the tradeoff between catching responders and creating false positives. | Main paper |

## Notes

- Use `final_cluster_profile_heatmap.png`, `final_cluster_segment_comparison.png`, `final_feature_set_comparison.png`, and `final_threshold_tradeoff.png` as the strongest main-paper figures.
- Use appendix placement for supporting visuals if the paper page limit is tight.
- Do not describe figure patterns as causal effects.
