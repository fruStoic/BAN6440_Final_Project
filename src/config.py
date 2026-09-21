from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"

DATA_PATH = DATA_DIR / "teleconnect.csv"


# ============================================================
# FINAL PROJECT PROTOCOL
# ============================================================

FINAL_PROJECT_SPLIT_SEED = 20260920
HOLDOUT_SIZE = 0.20

CV_SPLITS = 5
CV_REPEATS = 3
CV_RANDOM_STATE = 20260920

PRIMARY_SELECTION_METRIC = "average_precision"


# ============================================================
# CALIBRATION
# ============================================================

CALIBRATION_METHOD = "sigmoid"
CALIBRATION_CV = 5


# ============================================================
# BUSINESS DECISION RULE
# ============================================================

CONTACT_COST = 2.00
OFFER_COST = 65.00
ACCEPTANCE_RATE = 0.40

# Incremental probability that the intervention prevents churn
# among customers who otherwise would have churned.
# Defined across all contacted customers.
INCREMENTAL_RETENTION_UPLIFT = 0.15

MONTHLY_REVENUE_PROXY = 65.00
CONTRIBUTION_MARGIN_RATE = 0.30
HORIZON_MONTHS = 24

TOTAL_CONTRIBUTION_MARGIN = (
    MONTHLY_REVENUE_PROXY
    * CONTRIBUTION_MARGIN_RATE
    * HORIZON_MONTHS
)

EXACT_BREAK_EVEN_THRESHOLD = (
    CONTACT_COST
    + ACCEPTANCE_RATE * OFFER_COST
) / (
    INCREMENTAL_RETENTION_UPLIFT
    * TOTAL_CONTRIBUTION_MARGIN
)

OPERATIONAL_THRESHOLD = 0.40


# ============================================================
# REPRODUCIBILITY
# ============================================================

GLOBAL_RANDOM_SEED = 20260920

# ============================================================
# MODEL COMPARISON
# Locked before observing Final Project model results
# ============================================================

LOGISTIC_MAX_ITER = 2000
LOGISTIC_SOLVER = "lbfgs"

HGB_LEARNING_RATE = 0.10
HGB_MAX_ITER = 100
HGB_MAX_LEAF_NODES = 31
HGB_L2_REGULARIZATION = 0.0

# Incumbent Module 6 ANN
ANN_HIDDEN_1 = 32
ANN_HIDDEN_2 = 16
ANN_DROPOUT = 0.30

ANN_OPTIMIZER = "sgd"
ANN_LEARNING_RATE = 0.01

ANN_BATCH_SIZE = 32
ANN_MAX_EPOCHS = 150
ANN_EARLY_STOPPING_PATIENCE = 10
ANN_INNER_VALIDATION_SIZE = 0.15

CLASSIFICATION_THRESHOLD = 0.50

# ============================================================
# PROTECTED-ATTRIBUTE GOVERNANCE
# Locked before demographic ablation
# ============================================================

PROTECTED_ATTRIBUTES = [
    "gender",
    "SeniorCitizen",
]

# Use the same paired-fold practical-tie rule as model selection.
# If removing these explicit demographic variables is practically
# tied with the full feature set on PR-AUC, remove them.
#
# If the reduced model falls outside the practical-tie band,
# retain them for the prototype and report the governance trade-off.

DROP_PROTECTED_IF_PRACTICALLY_TIED = True

# ============================================================
# FINAL FEATURE GOVERNANCE DECISION
# Determined from pre-registered development ablation
# ============================================================

FINAL_EXCLUDED_PREDICTORS = [
    "gender",
    "SeniorCitizen",
]

FINAL_PREDICTIVE_FEATURE_POLICY = (
    "Exclude gender and SeniorCitizen from model inputs. "
    "Retain them separately for fairness auditing only."
)