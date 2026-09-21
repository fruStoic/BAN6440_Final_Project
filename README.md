\# BAN6440 Final Project  

\## End-to-End ML Architecture \& Interoperability Proposal



\*\*Student:\*\* Emmanuel Fru  

\*\*Course:\*\* BAN6440 Machine Learning  

\*\*Project:\*\* Telecom Churn Retention Decision System



\## 1. Project Overview



This project develops an end-to-end machine-learning decision system for telecommunications customer churn.



The objective is not only to predict churn, but to determine when a customer's estimated churn probability is high enough to justify a retention intervention.



The project extends the Module 6 churn-classification work by adding:



\- fresh development and holdout protocol;

\- repeated cross-validation;

\- model comparison;

\- probability calibration;

\- protected-attribute ablation;

\- fairness analysis;

\- economically derived decision threshold;

\- business-impact analysis;

\- deployment architecture;

\- model governance and monitoring;

\- reproducibility controls.



The final selected model is a \*\*logistic regression classifier\*\* with sigmoid probability calibration and a fixed operational threshold of \*\*0.40\*\*.



\---



\## 2. Dataset



The project uses:



```text

data/teleconnect.csv

```



Dataset summary:



```text

Rows:                 7,043

Columns:              21

Churn customers:      1,869

Churn prevalence:     26.54%

Development rows:     5,634

Final holdout rows:   1,409

```



The final project uses a fixed stratified split with:



```text

Random seed: 20260920

Development: 80%

Holdout:     20%

```



The holdout split is stored permanently and is not regenerated during model execution.



\---



\## 3. Final Model



The final predictive system uses:



```text

Model:                  Logistic Regression

Calibration:            Sigmoid

Operating threshold:    0.40

Raw predictive inputs:  17

Encoded features:       28

```



The following variables are excluded from predictive inputs:



```text

gender

SeniorCitizen

```



They are retained only for fairness auditing.



\---



\## 4. Project Structure



```text

BAN6440\_Final\_Project/

│

├── data/

│   └── teleconnect.csv

│

├── src/

│   ├── \_\_init\_\_.py

│   ├── config.py

│   ├── data\_loader.py

│   ├── create\_final\_project\_split.py

│   ├── data\_audit.py

│   ├── preprocessing.py

│   ├── preprocessing\_check.py

│   ├── model\_comparison.py

│   ├── calibration.py

│   ├── demographic\_ablation.py

│   ├── final\_feature\_calibration.py

│   ├── fairness\_analysis.py

│   ├── freeze\_pipeline.py

│   ├── holdout\_preflight.py

│   ├── final\_evaluation.py

│   └── final\_business\_postprocess.py

│

├── tests/

│   └── test\_pipeline.py

│

├── outputs/

│   ├── figures/

│   ├── tables/

│   ├── models/

│   ├── protocol\_lock.json

│   ├── split\_manifest.json

│   ├── split\_assignments.csv

│   ├── pipeline\_freeze.json

│   ├── final\_holdout\_results.json

│   └── final\_business\_impact\_operational.json

│

├── docs/

│   └── model\_card.md

│

├── report/

│

├── requirements.txt

├── requirements-lock.txt

└── README.md

```



\---



\## 5. Environment Setup



The project was developed using:



```text

Python 3.11.9

```



Create and activate a virtual environment.



\### Windows PowerShell



```powershell

python -m venv .venv

```



If PowerShell blocks activation:



```powershell

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

```



Activate:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



Install dependencies:



```powershell

pip install -r requirements.txt

```



For exact reproducibility:



```powershell

pip install -r requirements-lock.txt

```



\---



\## 6. Main Dependencies



The project uses:



```text

pandas

numpy

scikit-learn

scipy

matplotlib

tensorflow

joblib

pytest

```



The exact installed package versions are stored in:



```text

requirements-lock.txt

```



\---



\## 7. Reproducibility Sequence



The project was executed in a controlled sequence.



\### Step 1: Create the permanent final-project split



```powershell

python -m src.create\_final\_project\_split

```



This creates:



```text

outputs/protocol\_lock.json

outputs/split\_manifest.json

outputs/split\_assignments.csv

```



The split should not be regenerated after creation.



\---



\### Step 2: Audit development data



```powershell

python -m src.data\_audit

```



This produces development-only quality and outlier evidence.



\---



\### Step 3: Check preprocessing



```powershell

python -m src.preprocessing\_check

```



This validates:



```text

feature counts

encoded schema

scaling

missing values

finite values

```



\---



\### Step 4: Compare candidate models



```powershell

python -m src.model\_comparison

```



The compared models are:



```text

Majority baseline

Logistic Regression

HistGradientBoosting

Module 6 ANN

```



Model selection uses mean PR-AUC with a pre-registered practical-tie rule.



\---



\### Step 5: Evaluate calibration



```powershell

python -m src.calibration

```



Sigmoid calibration was specified before results were observed.



\---



\### Step 6: Run protected-attribute ablation



```powershell

python -m src.demographic\_ablation

```



This compares logistic regression with and without:



```text

gender

SeniorCitizen

```



\---



\### Step 7: Evaluate the final reduced feature set



```powershell

python -m src.final\_feature\_calibration

```



\---



\### Step 8: Run development fairness audit



```powershell

python -m src.fairness\_analysis

```



Fairness is assessed separately for:



```text

gender

SeniorCitizen

```



\---



\### Step 9: Freeze the pipeline



```powershell

python -m src.freeze\_pipeline

```



This records the final model, feature policy, calibration method, and decision threshold.



\---



\### Step 10: Run automated tests



```powershell

pytest -q

```



Final pre-holdout result:



```text

11 passed

```



\---



\### Step 11: Run holdout preflight



```powershell

python -m src.holdout\_preflight

```



The preflight checks that the frozen artifacts have not changed and that no previous holdout evaluation exists.



\---



\### Step 12: Final holdout evaluation



This command is intended to be run once only:



```powershell

python -m src.final\_evaluation

```



The script creates a holdout-open marker before loading the final holdout and prevents repeat final evaluations.



No model tuning is permitted after this stage.



\---



\### Step 13: Operational business post-processing



The original final-evaluation output used the exact economic threshold in one business-summary function even though the pre-registered operational threshold was 0.40.



The model was not retrained and the holdout was not rerun.



The corrected operational business result was calculated directly from the immutable saved holdout predictions:



```powershell

python -m src.final\_business\_postprocess

```



The result is stored in:



```text

outputs/final\_business\_impact\_operational.json

```



\---



\## 8. Final Holdout Results



At the fixed 0.40 threshold:



```text

Accuracy:        0.7899

Precision:       0.5942

Recall:          0.6578

F1:              0.6244

PR-AUC:          0.6484

ROC-AUC:         0.8470

Brier score:     0.1356

```



Confusion matrix:



```text

TN = 867

FP = 168

FN = 128

TP = 246

```



Selected customers:



```text

414 / 1,409

29.38%

```



\---



\## 9. Business Decision Rule



The intervention rule is:



```text

Contact customer if calibrated churn probability >= 0.40

```



The exact economic break-even probability is:



```text

0.398860

```



The operational threshold was pre-registered as:



```text

0.40

```



Base-case assumptions:



```text

Contact cost:                   $2

Offer cost:                     $65

Offer acceptance probability:   40%

Incremental retention uplift:   15%

Contribution margin rate:       30%

Monthly revenue proxy:          $65

Value horizon:                  24 months

```



These are planning assumptions rather than dataset facts.



Operational holdout result:



```text

Selected customers:       414

Expected prevented churn: 36.48

Campaign cost:            $11,592.00

Expected benefit:         $17,073.27

Expected net value:       $5,481.27

Expected ROI:             47.28%

```



These are expected values, not realized financial outcomes.



\---



\## 10. Key Output Artifacts



\### Protocol and model decisions



```text

outputs/protocol\_lock.json

outputs/split\_manifest.json

outputs/pipeline\_freeze.json

```



\### Model evaluation



```text

outputs/final\_holdout\_results.json

outputs/tables/final\_model\_comparison.csv

outputs/tables/final\_metric\_confidence\_intervals.csv

```



\### Holdout predictions



```text

outputs/tables/final\_holdout\_predictions.csv

```



\### Fairness



```text

outputs/tables/final\_holdout\_fairness.csv

```



\### Business analysis



```text

outputs/tables/business\_sensitivity\_grid.csv

outputs/final\_business\_impact.json

outputs/final\_business\_impact\_operational.json

```



\### Model artifact



```text

outputs/models/final\_model\_bundle.joblib

```



\### Figures



```text

outputs/figures/final\_confusion\_matrix.png

outputs/figures/final\_roc\_comparison.png

outputs/figures/final\_pr\_comparison.png

outputs/figures/final\_holdout\_reliability.png

```



\---



\## 11. Important Methodological Restrictions



The final holdout must not be used for:



```text

hyperparameter tuning

feature selection

threshold optimization

calibration-method selection

model selection

```



The holdout has already been consumed for final evaluation.



Any future model changes require a new validation protocol or genuinely new data.



\---



\## 12. Known Limitations



The project uses an internal random holdout from a dataset previously studied in Module 6. It is therefore not equivalent to external validation.



The dataset contains no timestamps, so temporal generalization cannot be evaluated.



The assumed 15% intervention uplift is not estimated from the dataset.



The model predicts churn risk rather than causal treatment effect.



The final expected ROI is sensitive to uplift and customer-value assumptions.



Group-level fairness metrics, particularly for `SeniorCitizen`, require continued monitoring.



\---



\## 13. Responsible Use



The model should be used as a retention decision-support tool.



It should not independently:



```text

change customer prices

alter contracts

deny services

make credit decisions

generate unrestricted offers

```



Campaign outcomes and treatment exposure should be recorded so that future analyses can separate natural churn from intervention-driven outcomes.



\---



\## 14. Supporting Documentation



The project includes:



```text

Final report

Architecture diagram

Model Card

AI Usage Disclosure

Detailed AI interaction log

Model evaluation tables

Business analysis

Fairness evidence

Source code

Environment requirements

```



