# BAN6440 Final Project AI Interaction Log

This log records AI-assisted work chronologically. Prompts are preserved verbatim
where identified as project evidence. AI-generated recommendations are assessed
rather than automatically accepted.


## 2026-09-20 — Episode 1: Initial protocol review

**Prompt (verbatim):**
> I've attached my final project and a brief I'd like to stick to in drafting this. Review both documents and flag any anomaly before we start drafting the final project.

**Output summary:**
The AI review identified several weaknesses in the proposed experimental design.
The most important were reuse of the already-inspected Module 6 test set, an
incorrect campaign expected-value equation, and the proposal to average
deterministic models across random seeds.

**Assessment:**
The critique was materially correct. The Module 6 test set had already been
examined and had influenced subsequent analysis, so it could not reasonably be
treated as an untouched final-project holdout. The original business-value
equation also conflated offer acceptance with incremental retention uplift.
Repeating deterministic models across random seeds would not provide meaningful
performance variation.

I extended the critique by noting that a fresh split of the same 7,043 records is
still not completely pristine because preprocessing and ANN architecture choices
were developed with prior exposure to this dataset. I therefore retained that as
an explicit limitation.

**Action:**
The protocol was rewritten before final-project modelling. The final design uses
a fresh 80/20 stratified holdout, repeated stratified cross-validation on
development data only, pre-specified sigmoid calibration, an economically derived
decision threshold, separate demographic ablation and fairness analyses, and a
single final holdout evaluation after all choices are frozen.

**Verified by:**
Review of the previous Module 6 workflow and outputs, direct recomputation of the
campaign expected-value equation, and comparison with the final-project rubric.
## 2026-09-20 — Pre-registered modelling and business protocol

**Prompt (verbatim):**
> let's go !

**Output summary:**
The final-project execution workflow was started only after the evaluation
protocol, business assumptions, calibration method, model-selection rule and
split seed had been fixed.

**Assessment:**
These settings are treated as pre-registered decisions rather than choices to be
changed after observing model results.

**Action:**
Locked before the final-project split:

- Final holdout seed: 20260920
- Holdout size: 20%
- Development evaluation: 5-fold × 3-repeat RepeatedStratifiedKFold
- Primary model-selection metric: PR-AUC / average precision
- Calibration: sigmoid / Platt scaling
- Contact cost: $2
- Offer cost: $65
- Acceptance rate: 40%
- Incremental retention uplift: 15%
- Contribution margin: 30% of $65 per month
- Horizon: 24 months
- Exact break-even probability: 0.398860
- Operational threshold: 0.40

Pre-registered hypotheses:
1. Gradient boosting is expected to perform at least as well as logistic
   regression and the incumbent Module 6 ANN.
2. The incumbent ANN is expected to show some probability over-confidence.
3. If the selected model is over-confident, calibration may reduce the number of
   customers exceeding the 0.40 economic threshold.

**Verified by:**
The cost threshold was calculated directly from the assumptions above before any
new model comparison or final-project holdout evaluation.

## 2026-09-20 — Fresh holdout created and verified

**Output summary:**
The pre-registered stratified split was executed with seed 20260920.

**Assessment:**
The execution reproduced the expected split exactly. The 7,043-row dataset was
partitioned into 5,634 development records and 1,409 final-holdout records.
The development set contains 1,495 churn cases (26.53532%) and the holdout
contains 374 churn cases (26.54365%).

**Action:**
The split is now permanently locked. All subsequent development work will use
the recorded row assignments rather than generating another train/test split.

**Verified by:**
`outputs/split_manifest.json`, `outputs/split_assignments.csv`, and dataset
SHA-256:
88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a

## 2026-09-20 — Development-only data audit

**Output summary:**
The development audit examined 5,634 customers without accessing final-holdout
features. Eight blank `TotalCharges` values were found, and all eight occurred
where `tenure = 0`. Converting those values to zero left no missing values.

An IQR audit found no outliers in `tenure`, `MonthlyCharges`, or `TotalCharges`.
The audit also confirmed structural redundancy among internet-service variables
and a 0.999562 correlation between `TotalCharges` and
`tenure × MonthlyCharges`.

**Assessment:**
The pre-registered cleaning rule was supported by the development data. The eight
blank `TotalCharges` values represent new customers with zero tenure rather than
unknown charges, so setting them to zero is preferable to dropping the records or
statistically imputing them.

The proposed outlier policy also required no intervention. No development
observations fell outside the 1.5-IQR bounds for the three continuous variables.

**Action:**
All development rows are retained. No winsorisation, clipping, or outlier removal
will be applied. The structural redundancy is documented but the corresponding
features are retained because this project tests model choice and deployment
rather than conducting post hoc feature elimination.

**Verified by:**
`outputs/development_data_audit.json`,
`outputs/tables/development_data_quality.csv`, and
`outputs/tables/development_iqr_audit.csv`.

## 2026-09-20 — Model comparison configuration locked

**Output summary:**
The development preprocessing check reproduced 30 encoded model inputs from
19 raw predictors. The four numeric variables were standardized to mean 0 and
standard deviation 1. No final-holdout feature data were used.

**Assessment:**
The expected preprocessing structure was reproduced. The model comparison can
therefore proceed without changing the feature pipeline based on model results.

**Action:**
The following candidate configurations were fixed before running the comparison:

- Majority-class baseline: predict non-churn for classification; development-fold
  churn prevalence used as the constant probability score.
- Logistic regression: `lbfgs`, maximum 2,000 iterations.
- HistGradientBoosting: learning rate 0.10, 100 boosting iterations,
  31 maximum leaf nodes, no L2 regularization, no automatic early stopping.
- Incumbent Module 6 ANN: 32-unit ReLU layer, 0.30 dropout, 16-unit ReLU layer,
  sigmoid output, SGD learning rate 0.01, batch size 32, maximum 150 epochs,
  early-stopping patience 10.
- ANN early stopping uses a stratified 15% validation subset drawn only from each
  outer training fold.
- Classification metrics use a fixed 0.50 threshold.
- Primary model-selection metric remains average precision / PR-AUC across
  5 folds × 3 repeats.

No candidate will be hyperparameter-tuned after observing these comparison results.

**Verified by:**
`outputs/development_feature_schema.json`,
`outputs/tables/development_feature_names.csv`, and the locked build protocol.

## 2026-09-20 — Episode 2: Model comparison contradicted the pre-registered expectation

**Output summary:**
The four pre-specified models were evaluated using identical 5-fold × 3-repeat
stratified development folds. Logistic regression achieved the highest mean
PR-AUC at 0.6549, compared with 0.6532 for the incumbent Module 6 ANN and
0.6425 for HistGradientBoosting.

The paired-fold PR-AUC difference between logistic regression and the ANN was
0.001619 with a paired standard deviation of 0.009056. The difference between
logistic regression and HistGradientBoosting was 0.012400 with a paired standard
deviation of 0.013514. Under the pre-registered practical-tie rule, all three
predictive models were therefore treated as tied.

**Assessment:**
The result contradicted the pre-registered expectation that gradient boosting
would perform at least as well as logistic regression. Logistic regression
actually produced the highest mean PR-AUC, ROC-AUC, F1, accuracy, and the lowest
Brier score among the predictive candidates.

The ANN performed almost identically to logistic regression on PR-AUC, but its
additional architecture and training complexity did not produce a meaningful
development-set advantage.

**Action:**
I followed the pre-registered tie rule rather than changing the selection criterion
after seeing the results. Logistic regression was selected because it was the
simplest model among those practically tied with the highest mean PR-AUC.

No hyperparameters were changed and the final holdout remained sealed.

**Verified by:**
`outputs/tables/model_comparison_fold_metrics.csv`,
`outputs/tables/model_comparison_summary.csv`,
`outputs/tables/model_comparison_paired_ap.csv`, and
`outputs/model_selection.json`.
## 2026-09-20 — Protected-attribute ablation rule pre-registered

**Output summary:**
Before running the demographic-feature ablation, the governance decision rule was
fixed.

**Assessment:**
The ablation must not use a post hoc definition of "negligible performance loss."

**Action:**
The full logistic-regression model will be compared with a version excluding
`gender` and `SeniorCitizen` using the same 15 matched development folds and
PR-AUC as the primary metric.

The same paired practical-tie rule used during model selection will be applied.
If the reduced model is practically tied with the full model, the explicit
demographic features will be removed because they add governance sensitivity
without demonstrated predictive value. If the reduced model falls outside that
tie band, the attributes will be retained for the prototype and the trade-off
will be reported explicitly.

**Verified by:**
Rule recorded before demographic-ablation results were observed.

## 2026-09-20 — Episode 3: Calibration hypothesis was not supported

**Output summary:**
The selected logistic-regression model was assessed using nested out-of-fold
development predictions with the pre-specified sigmoid calibration method.

The uncalibrated model achieved a Brier score of 0.135817. Sigmoid calibration
produced 0.135867, a small deterioration of 0.000050. PR-AUC changed from
0.653747 to 0.653669 and ROC-AUC from 0.844042 to 0.844027.

At the pre-registered 0.40 economic threshold, the uncalibrated probabilities
selected 1,677 of 5,634 development customers, compared with 1,668 after
calibration, a difference of only nine customers.

**Assessment:**
The result did not support the pre-registered expectation that the selected model
would be materially over-confident. For the current full-feature logistic model,
sigmoid calibration provided no measurable improvement and slightly worsened
Brier loss.

The result was retained rather than changing to isotonic calibration after seeing
the data, because the calibration method had been pre-specified before execution.

**Action:**
Sigmoid remains the locked calibration method. No alternative calibration method
will be tested post hoc. If the protected-attribute ablation changes the final
feature set, the same calibration assessment will be repeated for that final
feature configuration before the model is frozen.

**Verified by:**
`outputs/tables/calibration_oof_predictions.csv`,
`outputs/calibration_development_summary.json`, and
`outputs/figures/development_reliability_diagram.png`.

## 2026-09-20 — Episode 4: Protected-attribute ablation supported removal

**Output summary:**
The selected logistic-regression model was retrained across the same 15 matched
development folds after removing `gender` and `SeniorCitizen`.

The full model achieved mean PR-AUC 0.654856, while the reduced model achieved
0.657584. The mean paired PR-AUC difference, calculated as full minus reduced,
was -0.002728 with a paired standard deviation of 0.003989.

**Assessment:**
The reduced model was practically tied with the full model under the
pre-registered governance rule. The reduced model also produced slightly higher
mean PR-AUC and slightly lower Brier loss, although those small numerical
differences are not interpreted as evidence of meaningful superiority.

The result shows that explicitly using `gender` and `SeniorCitizen` did not
provide enough predictive value to justify their additional governance
sensitivity.

**Action:**
`gender` and `SeniorCitizen` were removed from the final predictive feature set.
They are retained only as audit variables for evaluating group-level performance
after the model is frozen.

The final holdout remained sealed.

**Verified by:**
`outputs/tables/protected_attribute_ablation_folds.csv`,
`outputs/tables/protected_attribute_ablation_summary.csv`, and
`outputs/protected_attribute_decision.json`.
## 2026-09-20 — Final feature calibration confirmed negligible calibration effect

**Output summary:**
After removing `gender` and `SeniorCitizen`, the final reduced logistic-regression
feature set was reassessed using the locked sigmoid calibration procedure.

The raw out-of-fold Brier score was 0.135828 and the calibrated score was
0.135868, a deterioration of 0.000040. PR-AUC changed from 0.656295 to
0.655724 and ROC-AUC from 0.844102 to 0.844080.

At the 0.40 economic threshold, calibration changed the selected development
population from 1,682 to 1,674 customers, a reduction of eight.

**Assessment:**
The earlier finding was reproduced on the final feature set. Sigmoid calibration
did not materially improve probability quality and the pre-registered
over-confidence hypothesis was not supported.

**Action:**
The sigmoid calibrator remains part of the locked pipeline because the method was
pre-specified. No alternative calibration method will be tested after observing
these results.

**Verified by:**
`outputs/tables/final_feature_calibration_oof_predictions.csv`,
`outputs/final_feature_calibration_summary.json`, and
`outputs/figures/final_feature_development_reliability.png`.

## 2026-09-20 — Development fairness audit after demographic-feature removal

**Output summary:**
The final reduced logistic-regression model was audited using development
out-of-fold calibrated predictions. `gender` and `SeniorCitizen` were not model
inputs and were used only as grouping variables.

Gender results were similar across groups. Female and male selection rates were
29.83% and 29.60%, while recall was 65.73% and 65.77%.

Larger differences remained for `SeniorCitizen`. Senior customers had a churn
rate of 40.84% compared with 23.81% for non-seniors. Their selection rate was
49.17% versus 26.01%, recall was 75.00% versus 62.73%, and false-positive rate
was 31.33% versus 14.53%.

**Assessment:**
The gender audit showed little development-stage disparity in selection or error
rates. Larger differences remained across SeniorCitizen groups even though
SeniorCitizen was excluded from the predictive feature set.

These differences do not by themselves establish discriminatory treatment. The
groups have substantially different observed churn prevalence, and remaining
features may also correlate with age. However, the higher senior false-positive
rate and the approximately 2.27 percentage-point average underprediction of senior
churn risk are relevant governance findings.

**Action:**
No model changes were made because the fairness audit was conducted after the
protected-attribute feature decision had been fixed. The same group metrics will
be reported once on the untouched final holdout after the full pipeline is frozen.

**Verified by:**
`outputs/tables/development_fairness_audit.csv` and
`outputs/development_fairness_summary.json`.

## 2026-09-20 — Final development pipeline frozen

**Output summary:**
All development-stage model, feature, calibration, fairness and business-rule
decisions were completed before final-holdout evaluation.

**Assessment:**
Logistic regression was selected under the pre-registered model-selection rule.
`gender` and `SeniorCitizen` were removed following the pre-registered ablation
rule. Sigmoid calibration and the 0.40 economic threshold remained unchanged
despite development results showing negligible calibration benefit.

**Action:**
The pipeline was formally frozen. Post-holdout tuning is prohibited.

**Verified by:**
`outputs/pipeline_freeze.json`.

## 2026-09-20 — Automated validation before holdout evaluation

**Output summary:**
The frozen pipeline was tested using 11 automated tests before the final holdout
was accessed. All 11 tests passed in 4.61 seconds.

**Assessment:**
The tests confirmed the permanent split assignments, development counts,
TotalCharges cleaning rule, target encoding, preprocessing schema, protected-
attribute exclusion, locked protocol constants, model-selection artifact,
protected-attribute decision, and pipeline-freeze artifact.

**Action:**
No changes were required. The frozen pipeline remains unchanged.

**Verified by:**
`pytest -q`: 11 passed in 4.61 seconds.

## 2026-09-20 — Episode 5: Reporting bug found after holdout evaluation

**Output summary:**
The one-time final evaluation correctly used the pre-registered 0.40 threshold
for classification metrics, but the printed base-case business section called
the economic sensitivity function, which used the exact break-even threshold
of 0.398860. This caused the business summary to report 415 selected customers
while the operational model selected 414.

**Assessment:**
This was a reporting-code inconsistency, not a modelling or threshold-selection
error. The frozen operational threshold remained 0.40 and the immutable holdout
predictions had already been saved.

**Action:**
The model was not retrained and the holdout evaluation was not rerun. I
recomputed the operational business quantities directly from the saved calibrated
holdout probabilities using the locked 0.40 threshold. The corrected operational
result selected 414 customers, with expected campaign cost of $11,592.00,
expected benefit of $17,073.27, expected net value of $5,481.27, and ROI of
47.28%.

The exact 0.398860 scenario remains separately reported as a sensitivity check.

**Verified by:**
`outputs/tables/final_holdout_predictions.csv`,
`outputs/final_business_impact_operational.json`, and the frozen threshold in
`outputs/pipeline_freeze.json`.