"""
ONE-TIME Final Project holdout evaluation.

IMPORTANT
---------
The pipeline has already been frozen.

This script:
1. Fits all required models using development data only.
2. Creates a permanent holdout-opened marker.
3. Loads the final holdout once.
4. Generates all final evaluation evidence.
5. Prohibits re-evaluation.

NO post-holdout tuning is permitted.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import gc

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import tensorflow as tf

from .config import (
    OUTPUT_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    MODELS_DIR,

    GLOBAL_RANDOM_SEED,

    LOGISTIC_SOLVER,
    LOGISTIC_MAX_ITER,

    CALIBRATION_METHOD,
    CALIBRATION_CV,

    FINAL_EXCLUDED_PREDICTORS,

    OPERATIONAL_THRESHOLD,
    EXACT_BREAK_EVEN_THRESHOLD,

    CONTACT_COST,
    OFFER_COST,
    ACCEPTANCE_RATE,
    INCREMENTAL_RETENTION_UPLIFT,
    MONTHLY_REVENUE_PROXY,
    CONTRIBUTION_MARGIN_RATE,
    HORIZON_MONTHS,

    ANN_HIDDEN_1,
    ANN_HIDDEN_2,
    ANN_DROPOUT,
    ANN_LEARNING_RATE,
    ANN_BATCH_SIZE,
    ANN_MAX_EPOCHS,
    ANN_EARLY_STOPPING_PATIENCE,
    ANN_INNER_VALIDATION_SIZE,
)

from .data_loader import (
    load_development_data,
    load_holdout_data,
)

from .preprocessing import (
    split_features_target,
    build_preprocessor,
)


# ============================================================
# OUTPUT PATHS
# ============================================================

FREEZE_PATH = (
    OUTPUT_DIR
    / "pipeline_freeze.json"
)

HOLDOUT_OPENED_PATH = (
    OUTPUT_DIR
    / "holdout_opened.json"
)

FINAL_RESULTS_PATH = (
    OUTPUT_DIR
    / "final_holdout_results.json"
)

PREDICTIONS_PATH = (
    TABLES_DIR
    / "final_holdout_predictions.csv"
)

COMPARISON_PATH = (
    TABLES_DIR
    / "final_model_comparison.csv"
)

CI_PATH = (
    TABLES_DIR
    / "final_metric_confidence_intervals.csv"
)

FAIRNESS_PATH = (
    TABLES_DIR
    / "final_holdout_fairness.csv"
)

SENSITIVITY_PATH = (
    TABLES_DIR
    / "business_sensitivity_grid.csv"
)

BUSINESS_PATH = (
    OUTPUT_DIR
    / "final_business_impact.json"
)

MODEL_BUNDLE_PATH = (
    MODELS_DIR
    / "final_model_bundle.joblib"
)

CONFUSION_PATH = (
    FIGURES_DIR
    / "final_confusion_matrix.png"
)

ROC_PATH = (
    FIGURES_DIR
    / "final_roc_comparison.png"
)

PR_PATH = (
    FIGURES_DIR
    / "final_pr_comparison.png"
)

RELIABILITY_PATH = (
    FIGURES_DIR
    / "final_holdout_reliability.png"
)


# ============================================================
# SAFETY
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def assert_evaluation_not_previously_run():
    if HOLDOUT_OPENED_PATH.exists():
        raise RuntimeError(
            "holdout_opened.json already exists.\n"
            "The final holdout has already been opened.\n"
            "Do not run this evaluation again."
        )

    if FINAL_RESULTS_PATH.exists():
        raise RuntimeError(
            "final_holdout_results.json already exists.\n"
            "Do not regenerate final holdout results."
        )


# ============================================================
# FINAL LOGISTIC MODEL
# ============================================================

def make_logistic_pipeline(X_train):

    return Pipeline([
        (
            "preprocessor",
            build_preprocessor(
                X_train
            ),
        ),
        (
            "model",
            LogisticRegression(
                solver=LOGISTIC_SOLVER,
                max_iter=LOGISTIC_MAX_ITER,
            ),
        ),
    ])


# ============================================================
# INCUMBENT MODULE 6 ANN
# ============================================================

def set_tensorflow_seed(seed):

    tf.keras.backend.clear_session()

    tf.keras.utils.set_random_seed(
        seed
    )

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_incumbent_ann(
    input_dim,
    seed,
):

    set_tensorflow_seed(
        seed
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(input_dim,)
        ),

        tf.keras.layers.Dense(
            ANN_HIDDEN_1,
            activation="relu",
        ),

        tf.keras.layers.Dropout(
            ANN_DROPOUT,
        ),

        tf.keras.layers.Dense(
            ANN_HIDDEN_2,
            activation="relu",
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid",
        ),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=ANN_LEARNING_RATE
        ),
        loss="binary_crossentropy",
    )

    return model


# ============================================================
# METRICS
# ============================================================

def classification_metrics(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        )
        .ravel()
    )

    return {
        "threshold": float(
            threshold
        ),

        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "average_precision": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),

        "brier": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),

        "selected": int(
            predictions.sum()
        ),

        "selected_rate": float(
            predictions.mean()
        ),
    }


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

def bootstrap_confidence_intervals(
    y_true,
    probabilities,
    threshold,
    n_bootstrap=2000,
    seed=20260920,
):

    y_true = np.asarray(
        y_true
    )

    probabilities = np.asarray(
        probabilities
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(y_true)

    records = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "average_precision": [],
        "roc_auc": [],
        "brier": [],
    }

    for _ in range(
        n_bootstrap
    ):

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        y_b = y_true[idx]
        p_b = probabilities[idx]

        # Extremely unlikely here, but protect AUC/AP.
        if len(
            np.unique(y_b)
        ) < 2:
            continue

        pred_b = (
            p_b >= threshold
        ).astype(int)

        records[
            "accuracy"
        ].append(
            accuracy_score(
                y_b,
                pred_b,
            )
        )

        records[
            "precision"
        ].append(
            precision_score(
                y_b,
                pred_b,
                zero_division=0,
            )
        )

        records[
            "recall"
        ].append(
            recall_score(
                y_b,
                pred_b,
                zero_division=0,
            )
        )

        records[
            "f1"
        ].append(
            f1_score(
                y_b,
                pred_b,
                zero_division=0,
            )
        )

        records[
            "average_precision"
        ].append(
            average_precision_score(
                y_b,
                p_b,
            )
        )

        records[
            "roc_auc"
        ].append(
            roc_auc_score(
                y_b,
                p_b,
            )
        )

        records[
            "brier"
        ].append(
            brier_score_loss(
                y_b,
                p_b,
            )
        )

    rows = []

    for metric, values in (
        records.items()
    ):

        values = np.asarray(
            values
        )

        rows.append({
            "metric": metric,

            "estimate": float(
                {
                    "accuracy":
                        accuracy_score(
                            y_true,
                            probabilities
                            >= threshold,
                        ),

                    "precision":
                        precision_score(
                            y_true,
                            probabilities
                            >= threshold,
                            zero_division=0,
                        ),

                    "recall":
                        recall_score(
                            y_true,
                            probabilities
                            >= threshold,
                            zero_division=0,
                        ),

                    "f1":
                        f1_score(
                            y_true,
                            probabilities
                            >= threshold,
                            zero_division=0,
                        ),

                    "average_precision":
                        average_precision_score(
                            y_true,
                            probabilities,
                        ),

                    "roc_auc":
                        roc_auc_score(
                            y_true,
                            probabilities,
                        ),

                    "brier":
                        brier_score_loss(
                            y_true,
                            probabilities,
                        ),
                }[
                    metric
                ]
            ),

            "ci_2_5": float(
                np.percentile(
                    values,
                    2.5,
                )
            ),

            "ci_97_5": float(
                np.percentile(
                    values,
                    97.5,
                )
            ),

            "bootstrap_samples": int(
                len(values)
            ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# FAIRNESS AUDIT
# ============================================================

def group_metrics(
    y_true,
    probability,
    threshold,
):

    y_true = np.asarray(
        y_true
    )

    probability = np.asarray(
        probability
    )

    prediction = (
        probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            prediction,
            labels=[0, 1],
        )
        .ravel()
    )

    prevalence = float(
        y_true.mean()
    )

    mean_probability = float(
        probability.mean()
    )

    return {
        "n": int(
            len(y_true)
        ),

        "actual_churn": int(
            y_true.sum()
        ),

        "actual_churn_rate": (
            prevalence
        ),

        "selected": int(
            prediction.sum()
        ),

        "selection_rate": float(
            prediction.mean()
        ),

        "precision": float(
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),

        "false_positive_rate": float(
            fp / (fp + tn)
            if (fp + tn) > 0
            else np.nan
        ),

        "false_negative_rate": float(
            fn / (fn + tp)
            if (fn + tp) > 0
            else np.nan
        ),

        "mean_predicted_probability": (
            mean_probability
        ),

        "calibration_gap": float(
            mean_probability
            - prevalence
        ),

        "brier_score": float(
            brier_score_loss(
                y_true,
                probability,
            )
        ),
    }


# ============================================================
# BUSINESS ANALYSIS
# ============================================================

def business_scenario(
    probabilities,
    y_true,
    uplift,
    horizon_months,
):

    margin = (
        CONTRIBUTION_MARGIN_RATE
        * MONTHLY_REVENUE_PROXY
        * horizon_months
    )

    expected_cost_per_contact = (
        CONTACT_COST
        + ACCEPTANCE_RATE
        * OFFER_COST
    )

    benefit_coefficient = (
        uplift * margin
    )

    threshold = (
        expected_cost_per_contact
        / benefit_coefficient
    )

    if threshold > 1:

        return {
            "uplift": uplift,
            "horizon_months": horizon_months,
            "threshold": threshold,
            "decision": "never_profitable",
            "selected": 0,
            "selected_percent": 0.0,
            "actual_precision": np.nan,
            "actual_recall": 0.0,
            "expected_churn_probability_sum": 0.0,
            "expected_prevented_churn": 0.0,
            "campaign_cost": 0.0,
            "expected_benefit": 0.0,
            "expected_net": 0.0,
            "roi": np.nan,
        }

    selected_mask = (
        probabilities >= threshold
    )

    selected_count = int(
        selected_mask.sum()
    )

    if selected_count == 0:

        return {
            "uplift": uplift,
            "horizon_months": horizon_months,
            "threshold": threshold,
            "decision": "run_but_no_customer_clears_threshold",
            "selected": 0,
            "selected_percent": 0.0,
            "actual_precision": np.nan,
            "actual_recall": 0.0,
            "expected_churn_probability_sum": 0.0,
            "expected_prevented_churn": 0.0,
            "campaign_cost": 0.0,
            "expected_benefit": 0.0,
            "expected_net": 0.0,
            "roi": np.nan,
        }

    selected_y = (
        np.asarray(y_true)[
            selected_mask
        ]
    )

    probability_sum = float(
        probabilities[
            selected_mask
        ].sum()
    )

    expected_prevented = (
        uplift
        * probability_sum
    )

    cost = (
        selected_count
        * expected_cost_per_contact
    )

    benefit = (
        benefit_coefficient
        * probability_sum
    )

    net = (
        benefit
        - cost
    )

    roi = (
        net / cost
        if cost > 0
        else np.nan
    )

    actual_precision = float(
        selected_y.mean()
    )

    actual_recall = float(
        selected_y.sum()
        / np.asarray(
            y_true
        ).sum()
    )

    return {
        "uplift": uplift,
        "horizon_months": horizon_months,
        "threshold": float(
            threshold
        ),

        "decision": "run_campaign",

        "selected": (
            selected_count
        ),

        "selected_percent": float(
            selected_count
            / len(probabilities)
        ),

        "actual_precision": (
            actual_precision
        ),

        "actual_recall": (
            actual_recall
        ),

        "expected_churn_probability_sum": (
            probability_sum
        ),

        "expected_prevented_churn": float(
            expected_prevented
        ),

        "campaign_cost": float(
            cost
        ),

        "expected_benefit": float(
            benefit
        ),

        "expected_net": float(
            net
        ),

        "roi": float(
            roi
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    assert_evaluation_not_previously_run()

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Verify freeze artifact before doing anything.
    # ---------------------------------------------------------

    if not FREEZE_PATH.exists():
        raise FileNotFoundError(
            "pipeline_freeze.json does not exist."
        )

    with FREEZE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        freeze = json.load(
            file
        )

    if (
        freeze["status"]
        != "FROZEN_BEFORE_HOLDOUT"
    ):
        raise RuntimeError(
            "Pipeline is not frozen."
        )

    print()
    print("=" * 82)
    print("ONE-TIME FINAL HOLDOUT EVALUATION")
    print("=" * 82)

    print(
        "Pipeline status      : FROZEN"
    )

    print(
        "Post-holdout tuning : NOT ALLOWED"
    )

    print()
    print(
        "Fitting all models on DEVELOPMENT DATA ONLY "
        "before opening holdout..."
    )

    # =========================================================
    # DEVELOPMENT DATA
    # =========================================================

    development = (
        load_development_data()
    )

    # ---------------------------------------------------------
    # Final selected logistic model
    # ---------------------------------------------------------

    X_dev_full, y_dev = (
        split_features_target(
            development
        )
    )

    X_dev_final = (
        X_dev_full.drop(
            columns=(
                FINAL_EXCLUDED_PREDICTORS
            )
        )
    )

    raw_logistic = (
        make_logistic_pipeline(
            X_dev_final
        )
    )

    raw_logistic.fit(
        X_dev_final,
        y_dev,
    )

    calibrated_logistic = (
        CalibratedClassifierCV(
            estimator=make_logistic_pipeline(
                X_dev_final
            ),
            method=CALIBRATION_METHOD,
            cv=CALIBRATION_CV,
        )
    )

    calibrated_logistic.fit(
        X_dev_final,
        y_dev,
    )

    # ---------------------------------------------------------
    # Incumbent Module 6 ANN
    #
    # It retains its original full 19-predictor policy.
    # ---------------------------------------------------------

    ann_preprocessor = (
        build_preprocessor(
            X_dev_full
        )
    )

    X_ann_dev = (
        ann_preprocessor
        .fit_transform(
            X_dev_full
        )
        .astype(
            "float32"
        )
    )

    y_ann_dev = (
        y_dev
        .to_numpy(
            dtype="int32"
        )
    )

    (
        X_ann_train,
        X_ann_validation,
        y_ann_train,
        y_ann_validation,
    ) = train_test_split(
        X_ann_dev,
        y_ann_dev,
        test_size=(
            ANN_INNER_VALIDATION_SIZE
        ),
        stratify=y_ann_dev,
        random_state=(
            GLOBAL_RANDOM_SEED
        ),
    )

    incumbent_ann = (
        build_incumbent_ann(
            input_dim=(
                X_ann_dev.shape[1]
            ),
            seed=(
                GLOBAL_RANDOM_SEED
            ),
        )
    )

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=(
                ANN_EARLY_STOPPING_PATIENCE
            ),
            restore_best_weights=True,
        )
    )

    ann_history = (
        incumbent_ann.fit(
            X_ann_train,
            y_ann_train,
            validation_data=(
                X_ann_validation,
                y_ann_validation,
            ),
            epochs=(
                ANN_MAX_EPOCHS
            ),
            batch_size=(
                ANN_BATCH_SIZE
            ),
            callbacks=[
                early_stopping
            ],
            verbose=0,
            shuffle=True,
        )
    )

    ann_best_epoch = (
        int(
            np.argmin(
                ann_history.history[
                    "val_loss"
                ]
            )
            + 1
        )
    )

    ann_epochs_run = (
        len(
            ann_history.history[
                "loss"
            ]
        )
    )

    print(
        "Development models fitted."
    )

    print(
        f"Incumbent ANN epochs : "
        f"{ann_epochs_run}"
    )

    print(
        f"Incumbent ANN best   : "
        f"{ann_best_epoch}"
    )

    # ---------------------------------------------------------
    # Save deployable final model BEFORE holdout evaluation.
    # ---------------------------------------------------------

    model_bundle = {
        "model": (
            calibrated_logistic
        ),

        "model_type": (
            "logistic_regression_sigmoid_calibrated"
        ),

        "excluded_predictors": (
            FINAL_EXCLUDED_PREDICTORS
        ),

        "operational_threshold": (
            OPERATIONAL_THRESHOLD
        ),

        "exact_economic_threshold": (
            EXACT_BREAK_EVEN_THRESHOLD
        ),

        "business_assumptions": {
            "contact_cost": (
                CONTACT_COST
            ),
            "offer_cost": (
                OFFER_COST
            ),
            "acceptance_rate": (
                ACCEPTANCE_RATE
            ),
            "incremental_retention_uplift": (
                INCREMENTAL_RETENTION_UPLIFT
            ),
            "monthly_revenue_proxy": (
                MONTHLY_REVENUE_PROXY
            ),
            "contribution_margin_rate": (
                CONTRIBUTION_MARGIN_RATE
            ),
            "horizon_months": (
                HORIZON_MONTHS
            ),
        },
    }

    joblib.dump(
        model_bundle,
        MODEL_BUNDLE_PATH,
    )

    # =========================================================
    # THE HOLDOUT IS NOW OPENED
    # =========================================================

    opened_record = {
        "opened_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "status": (
            "FINAL_HOLDOUT_OPENED"
        ),

        "pipeline_freeze_sha256": (
            sha256_file(
                FREEZE_PATH
            )
        ),

        "rule": (
            "No model, preprocessing, calibration, "
            "feature or threshold changes permitted "
            "after this timestamp."
        ),
    }

    with HOLDOUT_OPENED_PATH.open(
        "x",
        encoding="utf-8",
    ) as file:

        json.dump(
            opened_record,
            file,
            indent=2,
        )

    print()
    print(">>> FINAL HOLDOUT OPENED <<<")
    print(
        "No changes are permitted from this point."
    )
    print()

    holdout = (
        load_holdout_data()
    )

    X_holdout_full, y_holdout = (
        split_features_target(
            holdout
        )
    )

    X_holdout_final = (
        X_holdout_full.drop(
            columns=(
                FINAL_EXCLUDED_PREDICTORS
            )
        )
    )

    # =========================================================
    # FINAL LOGISTIC PREDICTIONS
    # =========================================================

    raw_probability = (
        raw_logistic.predict_proba(
            X_holdout_final
        )[:, 1]
    )

    calibrated_probability = (
        calibrated_logistic
        .predict_proba(
            X_holdout_final
        )[:, 1]
    )

    final_predictions = (
        calibrated_probability
        >= OPERATIONAL_THRESHOLD
    ).astype(int)

    # =========================================================
    # INCUMBENT ANN PREDICTIONS
    # =========================================================

    X_ann_holdout = (
        ann_preprocessor
        .transform(
            X_holdout_full
        )
        .astype(
            "float32"
        )
    )

    ann_probability = (
        incumbent_ann.predict(
            X_ann_holdout,
            verbose=0,
        )
        .reshape(-1)
    )

    # Incumbent technical classification threshold remains 0.50.
    ann_threshold = 0.50

    # =========================================================
    # TECHNICAL METRICS
    # =========================================================

    majority_accuracy = float(
        (
            y_holdout == 0
        ).mean()
    )

    raw_logistic_metrics = (
        classification_metrics(
            y_holdout,
            raw_probability,
            OPERATIONAL_THRESHOLD,
        )
    )

    calibrated_metrics = (
        classification_metrics(
            y_holdout,
            calibrated_probability,
            OPERATIONAL_THRESHOLD,
        )
    )

    incumbent_ann_metrics = (
        classification_metrics(
            y_holdout,
            ann_probability,
            ann_threshold,
        )
    )

    comparison = pd.DataFrame([
        {
            "model": (
                "majority_baseline"
            ),
            "threshold": np.nan,
            "accuracy": (
                majority_accuracy
            ),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "average_precision": float(
                y_holdout.mean()
            ),
            "roc_auc": 0.5,
            "brier": float(
                brier_score_loss(
                    y_holdout,
                    np.full(
                        len(y_holdout),
                        y_dev.mean(),
                    ),
                )
            ),
        },

        {
            "model": (
                "final_logistic_calibrated"
            ),
            **{
                key: value
                for key, value in (
                    calibrated_metrics.items()
                )
                if key not in {
                    "tn",
                    "fp",
                    "fn",
                    "tp",
                    "selected",
                    "selected_rate",
                }
            },
        },

        {
            "model": (
                "incumbent_module6_ann"
            ),
            **{
                key: value
                for key, value in (
                    incumbent_ann_metrics.items()
                )
                if key not in {
                    "tn",
                    "fp",
                    "fn",
                    "tp",
                    "selected",
                    "selected_rate",
                }
            },
        },
    ])

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
    )

    # =========================================================
    # CONFIDENCE INTERVALS
    # =========================================================

    ci = (
        bootstrap_confidence_intervals(
            y_true=(
                y_holdout.to_numpy()
            ),
            probabilities=(
                calibrated_probability
            ),
            threshold=(
                OPERATIONAL_THRESHOLD
            ),
            n_bootstrap=2000,
            seed=(
                GLOBAL_RANDOM_SEED
            ),
        )
    )

    ci.to_csv(
        CI_PATH,
        index=False,
    )

    # =========================================================
    # HOLDOUT PREDICTIONS TABLE
    # =========================================================

    predictions_table = pd.DataFrame({
        "customerID": (
            holdout[
                "customerID"
            ]
            .astype(str)
        ),

        "y_true": (
            y_holdout
            .to_numpy()
        ),

        "raw_logistic_probability": (
            raw_probability
        ),

        "calibrated_probability": (
            calibrated_probability
        ),

        "selected_at_0_40": (
            final_predictions
        ),

        "incumbent_ann_probability": (
            ann_probability
        ),

        "gender": (
            holdout[
                "gender"
            ].to_numpy()
        ),

        "SeniorCitizen": (
            holdout[
                "SeniorCitizen"
            ]
            .astype(int)
            .to_numpy()
        ),
    })

    predictions_table.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # =========================================================
    # ROUNDING CHECK
    # =========================================================

    exact_mask = (
        calibrated_probability
        >= EXACT_BREAK_EVEN_THRESHOLD
    )

    operational_mask = (
        calibrated_probability
        >= OPERATIONAL_THRESHOLD
    )

    exact_ids = set(
        holdout.loc[
            exact_mask,
            "customerID",
        ].astype(str)
    )

    operational_ids = set(
        holdout.loc[
            operational_mask,
            "customerID",
        ].astype(str)
    )

    rounding_check = {
        "exact_threshold": float(
            EXACT_BREAK_EVEN_THRESHOLD
        ),

        "operational_threshold": float(
            OPERATIONAL_THRESHOLD
        ),

        "selected_exact": int(
            exact_mask.sum()
        ),

        "selected_operational": int(
            operational_mask.sum()
        ),

        "count_difference": int(
            operational_mask.sum()
            - exact_mask.sum()
        ),

        "identical_customer_set": (
            exact_ids
            == operational_ids
        ),
    }

    # =========================================================
    # BUSINESS IMPACT
    # =========================================================

    base_case = business_scenario(
        probabilities=(
            calibrated_probability
        ),
        y_true=(
            y_holdout.to_numpy()
        ),
        uplift=(
            INCREMENTAL_RETENTION_UPLIFT
        ),
        horizon_months=(
            HORIZON_MONTHS
        ),
    )

    sensitivity_rows = []

    for uplift in [
        0.10,
        0.15,
        0.25,
    ]:

        for horizon in [
            12,
            24,
            36,
        ]:

            sensitivity_rows.append(
                business_scenario(
                    probabilities=(
                        calibrated_probability
                    ),
                    y_true=(
                        y_holdout
                        .to_numpy()
                    ),
                    uplift=(
                        uplift
                    ),
                    horizon_months=(
                        horizon
                    ),
                )
            )

    sensitivity = pd.DataFrame(
        sensitivity_rows
    )

    sensitivity.to_csv(
        SENSITIVITY_PATH,
        index=False,
    )

    business_record = {
        "base_case": (
            base_case
        ),

        "rounding_check": (
            rounding_check
        ),

        "cost_per_selected_customer": (
            CONTACT_COST
            + ACCEPTANCE_RATE
            * OFFER_COST
        ),

        "benefit_coefficient_base_case": (
            INCREMENTAL_RETENTION_UPLIFT
            * CONTRIBUTION_MARGIN_RATE
            * MONTHLY_REVENUE_PROXY
            * HORIZON_MONTHS
        ),
    }

    with BUSINESS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            business_record,
            file,
            indent=2,
        )

    # =========================================================
    # FINAL FAIRNESS AUDIT
    # =========================================================

    audit_df = pd.DataFrame({
        "y_true": (
            y_holdout
            .to_numpy()
        ),

        "probability": (
            calibrated_probability
        ),

        "gender": (
            holdout[
                "gender"
            ].to_numpy()
        ),

        "SeniorCitizen": (
            holdout[
                "SeniorCitizen"
            ]
            .astype(int)
            .to_numpy()
        ),
    })

    fairness_rows = []

    for attribute in [
        "gender",
        "SeniorCitizen",
    ]:

        for group, frame in (
            audit_df.groupby(
                attribute
            )
        ):

            metrics = (
                group_metrics(
                    y_true=(
                        frame[
                            "y_true"
                        ]
                        .to_numpy()
                    ),
                    probability=(
                        frame[
                            "probability"
                        ]
                        .to_numpy()
                    ),
                    threshold=(
                        OPERATIONAL_THRESHOLD
                    ),
                )
            )

            fairness_rows.append({
                "attribute": (
                    attribute
                ),
                "group": str(
                    group
                ),
                **metrics,
            })

    fairness = pd.DataFrame(
        fairness_rows
    )

    fairness.to_csv(
        FAIRNESS_PATH,
        index=False,
    )

    # =========================================================
    # FIGURE 1: CONFUSION MATRIX
    # =========================================================

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_holdout,
        final_predictions,
        display_labels=[
            "No Churn",
            "Churn",
        ],
        ax=ax,
        values_format="d",
    )

    ax.set_title(
        "Final Logistic Model at 0.40 Threshold"
    )

    fig.tight_layout()

    fig.savefig(
        CONFUSION_PATH,
        dpi=200,
    )

    plt.close(
        fig
    )

    # =========================================================
    # FIGURE 2: ROC
    # =========================================================

    final_fpr, final_tpr, _ = (
        roc_curve(
            y_holdout,
            calibrated_probability,
        )
    )

    ann_fpr, ann_tpr, _ = (
        roc_curve(
            y_holdout,
            ann_probability,
        )
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.plot(
        final_fpr,
        final_tpr,
        label=(
            "Final logistic "
            f"(AUC={calibrated_metrics['roc_auc']:.3f})"
        ),
    )

    ax.plot(
        ann_fpr,
        ann_tpr,
        label=(
            "Incumbent ANN "
            f"(AUC={incumbent_ann_metrics['roc_auc']:.3f})"
        ),
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Chance",
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "Final Holdout ROC Comparison"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        ROC_PATH,
        dpi=200,
    )

    plt.close(
        fig
    )

    # =========================================================
    # FIGURE 3: PRECISION-RECALL
    # =========================================================

    final_precision_curve, final_recall_curve, _ = (
        precision_recall_curve(
            y_holdout,
            calibrated_probability,
        )
    )

    ann_precision_curve, ann_recall_curve, _ = (
        precision_recall_curve(
            y_holdout,
            ann_probability,
        )
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.plot(
        final_recall_curve,
        final_precision_curve,
        label=(
            "Final logistic "
            f"(AP={calibrated_metrics['average_precision']:.3f})"
        ),
    )

    ax.plot(
        ann_recall_curve,
        ann_precision_curve,
        label=(
            "Incumbent ANN "
            f"(AP={incumbent_ann_metrics['average_precision']:.3f})"
        ),
    )

    ax.axhline(
        y=float(
            y_holdout.mean()
        ),
        linestyle="--",
        label=(
            "Holdout churn prevalence"
        ),
    )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        "Final Holdout Precision-Recall Comparison"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        PR_PATH,
        dpi=200,
    )

    plt.close(
        fig
    )

    # =========================================================
    # FIGURE 4: RELIABILITY
    # =========================================================

    raw_fraction, raw_mean = (
        calibration_curve(
            y_holdout,
            raw_probability,
            n_bins=10,
            strategy="uniform",
        )
    )

    calibrated_fraction, calibrated_mean = (
        calibration_curve(
            y_holdout,
            calibrated_probability,
            n_bins=10,
            strategy="uniform",
        )
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    ax.plot(
        raw_mean,
        raw_fraction,
        marker="o",
        label=(
            f"Raw logistic "
            f"(Brier={raw_logistic_metrics['brier']:.3f})"
        ),
    )

    ax.plot(
        calibrated_mean,
        calibrated_fraction,
        marker="o",
        label=(
            f"Sigmoid calibrated "
            f"(Brier={calibrated_metrics['brier']:.3f})"
        ),
    )

    ax.set_xlabel(
        "Mean Predicted Churn Probability"
    )

    ax.set_ylabel(
        "Observed Churn Rate"
    )

    ax.set_title(
        "Final Holdout Reliability"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        RELIABILITY_PATH,
        dpi=200,
    )

    plt.close(
        fig
    )

    # =========================================================
    # SAVE MASTER RESULTS IMMEDIATELY
    # =========================================================

    final_results = {
        "evaluated_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "status": (
            "FINAL_HOLDOUT_EVALUATED"
        ),

        "holdout": {
            "rows": int(
                len(y_holdout)
            ),

            "no_churn": int(
                (y_holdout == 0)
                .sum()
            ),

            "churn": int(
                (y_holdout == 1)
                .sum()
            ),

            "churn_rate": float(
                y_holdout.mean()
            ),

            "majority_accuracy": (
                majority_accuracy
            ),
        },

        "final_model": {
            "type": (
                "logistic_regression"
            ),

            "excluded_predictors": (
                FINAL_EXCLUDED_PREDICTORS
            ),

            "calibration": (
                CALIBRATION_METHOD
            ),

            "operational_threshold": (
                OPERATIONAL_THRESHOLD
            ),

            "raw_metrics_at_0_40": (
                raw_logistic_metrics
            ),

            "calibrated_metrics_at_0_40": (
                calibrated_metrics
            ),
        },

        "incumbent_module6_ann": {
            "threshold": (
                ann_threshold
            ),

            "epochs_run": (
                ann_epochs_run
            ),

            "best_epoch": (
                ann_best_epoch
            ),

            "metrics": (
                incumbent_ann_metrics
            ),
        },

        "business": (
            business_record
        ),

        "rounding_check": (
            rounding_check
        ),

        "rule": (
            "No post-holdout tuning permitted."
        ),
    }

    with FINAL_RESULTS_PATH.open(
        "x",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_results,
            file,
            indent=2,
        )

    # =========================================================
    # CONSOLE REPORT
    # =========================================================

    print()
    print("=" * 82)
    print("FINAL HOLDOUT RESULTS")
    print("=" * 82)

    print()
    print("HOLDOUT")
    print(
        f"Rows                 : "
        f"{len(y_holdout):,}"
    )
    print(
        f"No churn             : "
        f"{(y_holdout == 0).sum():,}"
    )
    print(
        f"Churn                : "
        f"{(y_holdout == 1).sum():,}"
    )
    print(
        f"Churn rate           : "
        f"{y_holdout.mean():.5%}"
    )
    print(
        f"Majority accuracy    : "
        f"{majority_accuracy:.4f}"
    )

    print()
    print("FINAL CALIBRATED LOGISTIC @ 0.40")

    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "average_precision",
        "roc_auc",
        "brier",
    ]:
        print(
            f"{key:<21}: "
            f"{calibrated_metrics[key]:.6f}"
        )

    print(
        f"TN / FP / FN / TP    : "
        f"{calibrated_metrics['tn']} / "
        f"{calibrated_metrics['fp']} / "
        f"{calibrated_metrics['fn']} / "
        f"{calibrated_metrics['tp']}"
    )

    print(
        f"Selected             : "
        f"{calibrated_metrics['selected']:,} "
        f"({calibrated_metrics['selected_rate']:.2%})"
    )

    print()
    print("HOLDOUT CALIBRATION")

    print(
        f"Raw Brier            : "
        f"{raw_logistic_metrics['brier']:.6f}"
    )

    print(
        f"Calibrated Brier     : "
        f"{calibrated_metrics['brier']:.6f}"
    )

    print(
        f"Brier change         : "
        f"{calibrated_metrics['brier'] - raw_logistic_metrics['brier']:+.6f}"
    )

    print()
    print("INCUMBENT MODULE 6 ANN @ 0.50")

    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "average_precision",
        "roc_auc",
        "brier",
    ]:
        print(
            f"{key:<21}: "
            f"{incumbent_ann_metrics[key]:.6f}"
        )

    print()
    print("BASE-CASE BUSINESS IMPACT")

    print(
        f"Threshold            : "
        f"{base_case['threshold']:.6f}"
    )

    print(
        f"Selected             : "
        f"{base_case['selected']:,} "
        f"({base_case['selected_percent']:.2%})"
    )

    print(
        f"Actual precision     : "
        f"{base_case['actual_precision']:.4f}"
    )

    print(
        f"Actual recall        : "
        f"{base_case['actual_recall']:.4f}"
    )

    print(
        f"Expected prevented   : "
        f"{base_case['expected_prevented_churn']:.2f}"
    )

    print(
        f"Campaign cost        : "
        f"${base_case['campaign_cost']:,.2f}"
    )

    print(
        f"Expected benefit     : "
        f"${base_case['expected_benefit']:,.2f}"
    )

    print(
        f"Expected net         : "
        f"${base_case['expected_net']:,.2f}"
    )

    print(
        f"ROI                  : "
        f"{base_case['roi']:.2%}"
    )

    print()
    print("0.398860 vs 0.40 ROUNDING CHECK")

    print(
        f"Selected exact       : "
        f"{rounding_check['selected_exact']}"
    )

    print(
        f"Selected operational : "
        f"{rounding_check['selected_operational']}"
    )

    print(
        f"Count difference     : "
        f"{rounding_check['count_difference']:+d}"
    )

    print(
        f"Identical set        : "
        f"{rounding_check['identical_customer_set']}"
    )

    print()
    print("FINAL HOLDOUT FAIRNESS")

    fairness_display = [
        "attribute",
        "group",
        "n",
        "actual_churn_rate",
        "selection_rate",
        "precision",
        "recall",
        "false_positive_rate",
        "false_negative_rate",
        "calibration_gap",
        "brier_score",
    ]

    print(
        fairness[
            fairness_display
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print()
    print("BUSINESS SENSITIVITY GRID")

    sensitivity_display = [
        "uplift",
        "horizon_months",
        "threshold",
        "decision",
        "selected",
        "selected_percent",
        "actual_precision",
        "actual_recall",
        "expected_net",
        "roi",
    ]

    print(
        sensitivity[
            sensitivity_display
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print()
    print("FILES CREATED")

    for path in [
        HOLDOUT_OPENED_PATH,
        FINAL_RESULTS_PATH,
        PREDICTIONS_PATH,
        COMPARISON_PATH,
        CI_PATH,
        FAIRNESS_PATH,
        SENSITIVITY_PATH,
        BUSINESS_PATH,
        MODEL_BUNDLE_PATH,
        CONFUSION_PATH,
        ROC_PATH,
        PR_PATH,
        RELIABILITY_PATH,
    ]:
        print(path)

    print()
    print("=" * 82)
    print(
        "FINAL HOLDOUT EVALUATION COMPLETE."
    )
    print(
        "THE HOLDOUT IS NOW SPENT."
    )
    print(
        "DO NOT TUNE OR RE-RUN THE MODEL."
    )
    print("=" * 82)

    del incumbent_ann
    gc.collect()


if __name__ == "__main__":
    main()