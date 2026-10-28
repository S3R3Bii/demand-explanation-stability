# Model selection requires retained evidence

Effective October 28, 2026. This standard applies to every model, threshold, smoothing parameter and fine-tuning experiment in the 50-study program. It does not turn an unexecuted protocol into a completed study.

## Before fitting

Record the decision, target, units, primary selection metric, simple baseline, candidate configurations, search budget, random seeds and tie-breaking rule. Explain why the metric serves the decision. Regression normally reports MAE, MSE and RMSE with units; add an appropriate scale-normalized metric when series have different magnitudes. Classification requires prevalence, a confusion matrix, precision/recall and calibration where probabilities are used. State the operational threshold or cost assumptions. Statistical tests require assumptions, effect estimates and uncertainty; a p-value alone is insufficient.

Fix validation and final-test boundaries before examining outcomes. Use time, entity, participant or location separation as required by the data-generating process. Training labels, feature inputs, preprocessing, imputation, feature selection and threshold calibration must all be available at the prediction origin. Fit preprocessing inside each training fold. Keep related observations together. A date split alone does not prove an absence of leakage.

## During optimization

Retain every attempted configuration, fold score, status, runtime and seed, including failures. Select only from validation data using the declared objective. Compare the selected configuration with a meaningful simple baseline. Report boundary winners and search limitations; a finite grid does not establish a global optimum. Justify any search extension using validation evidence, and preserve the original results. Do not use a final-test score to choose the model, metric, threshold, feature set or narrative.

For pretrained-model fine-tuning, additionally retain the exact base model and revision, dataset rights, train/evaluation contamination checks, adapter or weight settings, optimizer, learning-rate schedule, effective batch size, epochs, checkpoint-selection rule, compute cost and task-specific evaluation. Never claim fine-tuning occurred if only prompting or inference was performed.

## Before claiming a result

Export observation-level validation and final-test predictions, labels, group/time identifiers and denominators. Independently recompute headline metrics from those exports without calling the fitting code's metric helpers. Check split disjointness, temporal availability, duplicates and source reconciliation. Use dependency-aware uncertainty (for example, participant or time-block resampling) where appropriate. Retain subgroup failures, robustness checks and all planned comparisons, including negative results.

Evaluate the frozen selection once on a previously unused test set. If a defect is repaired after test inspection, identify the corrected retrospective evaluation explicitly; do not describe the reused test as untouched. A new confirmatory claim needs a new untouched evaluation set. Report both predictive accuracy and relevant decision outcomes; lower MSE need not imply a useful operational improvement.

## Release evidence

Include the protocol, source hashes, split/availability manifest, complete search table, selected configuration, prediction exports, independent verification output, environment and reproducible commands. Classify each gate as pass, fail or unknown. Do not average away a failed correctness or leakage gate. Separate measured results from synthetic assumptions, prototypes from production and proposed contributions from established novelty. Manuscript drafts remain outside public repository history.
