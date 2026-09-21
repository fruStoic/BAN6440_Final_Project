"""
Development-only calibration assessment for the final reduced
logistic-regression feature set.

gender and SeniorCitizen are excluded from predictive inputs.

The calibration method remains the pre-specified sigmoid method.

The final holdout is never loaded.
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .config import (
    OUTPUT_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    CV_RANDOM_STATE,
    LOGISTIC_MAX_ITER,
    LOGISTIC_SOLVER,
    CALIBRATION_METHOD,
    CALIBRATION_CV,
    OPERATIONAL_THRESHOLD,
    FINAL_EXCLUDED_PREDICTORS,
)

from .data_loader import (
    load_development_data,
)

from .preprocessing import (
    split_features_target,
    build_preprocessor,
)


PREDICTIONS_PATH = (
    TABLES_DIR
    / "final_feature_calibration_oof_predictions.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "final_feature_calibration_summary.json"
)

RELIABILITY_PATH = (
    FIGURES_DIR
    / "final_feature_development_reliability.png"
)


def make_pipeline(X_train):

    return Pipeline([
        (
            "preprocessor",
            build_preprocessor(X_train),
        ),
        (
            "model",
            LogisticRegression(
                solver=LOGISTIC_SOLVER,
                max_iter=LOGISTIC_MAX_ITER,
            ),
        ),
    ])


def main():

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    development = (
        load_development_data()
    )

    X, y = split_features_target(
        development
    )

    X = X.drop(
        columns=FINAL_EXCLUDED_PREDICTORS
    )

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )

    raw_oof = np.zeros(
        len(X),
        dtype=float,
    )

    calibrated_oof = np.zeros(
        len(X),
        dtype=float,
    )

    fold_ids = np.zeros(
        len(X),
        dtype=int,
    )

    print()
    print("=" * 72)
    print("FINAL FEATURE CALIBRATION ASSESSMENT")
    print("=" * 72)

    print(
        f"Rows                 : {len(X):,}"
    )

    print(
        f"Raw predictors       : {X.shape[1]}"
    )

    print(
        "Excluded predictors  : "
        f"{FINAL_EXCLUDED_PREDICTORS}"
    )

    print(
        f"Method               : {CALIBRATION_METHOD}"
    )

    print(
        f"Inner calibration    : {CALIBRATION_CV}-fold"
    )

    print(
        f"Operating point      : {OPERATIONAL_THRESHOLD:.2f}"
    )

    print()
    print(
        "FINAL HOLDOUT IS NOT LOADED BY THIS SCRIPT."
    )
    print()

    for fold_number, (
        train_idx,
        validation_idx,
    ) in enumerate(
        outer_cv.split(X, y),
        start=1,
    ):

        X_train = (
            X.iloc[train_idx]
            .copy()
        )

        y_train = (
            y.iloc[train_idx]
            .copy()
        )

        X_validation = (
            X.iloc[validation_idx]
            .copy()
        )

        y_validation = (
            y.iloc[validation_idx]
            .copy()
        )

        # -----------------------------------------------------
        # Raw logistic model
        # -----------------------------------------------------

        raw_model = make_pipeline(
            X_train
        )

        raw_model.fit(
            X_train,
            y_train,
        )

        raw_probabilities = (
            raw_model.predict_proba(
                X_validation
            )[:, 1]
        )

        # -----------------------------------------------------
        # Locked sigmoid calibration
        # -----------------------------------------------------

        calibrated_model = (
            CalibratedClassifierCV(
                estimator=make_pipeline(
                    X_train
                ),
                method=CALIBRATION_METHOD,
                cv=CALIBRATION_CV,
            )
        )

        calibrated_model.fit(
            X_train,
            y_train,
        )

        calibrated_probabilities = (
            calibrated_model.predict_proba(
                X_validation
            )[:, 1]
        )

        raw_oof[
            validation_idx
        ] = raw_probabilities

        calibrated_oof[
            validation_idx
        ] = calibrated_probabilities

        fold_ids[
            validation_idx
        ] = fold_number

        print(
            f"Fold {fold_number}: "
            f"raw Brier="
            f"{brier_score_loss(y_validation, raw_probabilities):.5f} | "
            f"calibrated="
            f"{brier_score_loss(y_validation, calibrated_probabilities):.5f}"
        )

    # ---------------------------------------------------------
    # Overall development OOF metrics
    # ---------------------------------------------------------

    raw_brier = brier_score_loss(
        y,
        raw_oof,
    )

    calibrated_brier = (
        brier_score_loss(
            y,
            calibrated_oof,
        )
    )

    raw_ap = average_precision_score(
        y,
        raw_oof,
    )

    calibrated_ap = average_precision_score(
        y,
        calibrated_oof,
    )

    raw_auc = roc_auc_score(
        y,
        raw_oof,
    )

    calibrated_auc = roc_auc_score(
        y,
        calibrated_oof,
    )

    raw_selected = int(
        (
            raw_oof
            >= OPERATIONAL_THRESHOLD
        ).sum()
    )

    calibrated_selected = int(
        (
            calibrated_oof
            >= OPERATIONAL_THRESHOLD
        ).sum()
    )

    raw_pct = (
        raw_selected / len(y)
    )

    calibrated_pct = (
        calibrated_selected / len(y)
    )

    # ---------------------------------------------------------
    # Save OOF predictions
    # ---------------------------------------------------------

    predictions = pd.DataFrame({
        "development_row": np.arange(
            len(y)
        ),
        "fold": fold_ids,
        "y_true": y.to_numpy(),
        "raw_probability": raw_oof,
        "calibrated_probability": (
            calibrated_oof
        ),
        "raw_selected_at_0_40": (
            raw_oof
            >= OPERATIONAL_THRESHOLD
        ).astype(int),
        "calibrated_selected_at_0_40": (
            calibrated_oof
            >= OPERATIONAL_THRESHOLD
        ).astype(int),
    })

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Reliability diagram
    # ---------------------------------------------------------

    raw_fraction, raw_mean = (
        calibration_curve(
            y,
            raw_oof,
            n_bins=10,
            strategy="uniform",
        )
    )

    calibrated_fraction, calibrated_mean = (
        calibration_curve(
            y,
            calibrated_oof,
            n_bins=10,
            strategy="uniform",
        )
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    plt.plot(
        raw_mean,
        raw_fraction,
        marker="o",
        label="Uncalibrated logistic",
    )

    plt.plot(
        calibrated_mean,
        calibrated_fraction,
        marker="o",
        label="Sigmoid calibrated",
    )

    plt.xlabel(
        "Mean predicted churn probability"
    )

    plt.ylabel(
        "Observed churn rate"
    )

    plt.title(
        "Final Feature Set: Development Reliability"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RELIABILITY_PATH,
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------

    summary = {
        "scope": (
            "development_oof_final_feature_set"
        ),

        "excluded_predictors": (
            FINAL_EXCLUDED_PREDICTORS
        ),

        "uncalibrated": {
            "brier": float(
                raw_brier
            ),
            "average_precision": float(
                raw_ap
            ),
            "roc_auc": float(
                raw_auc
            ),
            "selected_at_0_40": (
                raw_selected
            ),
            "selected_percent": float(
                raw_pct
            ),
        },

        "calibrated": {
            "brier": float(
                calibrated_brier
            ),
            "average_precision": float(
                calibrated_ap
            ),
            "roc_auc": float(
                calibrated_auc
            ),
            "selected_at_0_40": (
                calibrated_selected
            ),
            "selected_percent": float(
                calibrated_pct
            ),
        },

        "brier_change": float(
            calibrated_brier
            - raw_brier
        ),

        "selection_count_change": int(
            calibrated_selected
            - raw_selected
        ),
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    # ---------------------------------------------------------
    # Console
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL FEATURE CALIBRATION SUMMARY")
    print("=" * 72)

    print(
        f"Raw Brier          : {raw_brier:.6f}"
    )

    print(
        f"Calibrated Brier   : {calibrated_brier:.6f}"
    )

    print(
        f"Brier change       : "
        f"{calibrated_brier - raw_brier:+.6f}"
    )

    print()

    print(
        f"Raw PR-AUC         : {raw_ap:.6f}"
    )

    print(
        f"Calibrated PR-AUC  : {calibrated_ap:.6f}"
    )

    print(
        f"Raw ROC-AUC        : {raw_auc:.6f}"
    )

    print(
        f"Calibrated ROC-AUC : {calibrated_auc:.6f}"
    )

    print()

    print(
        f"Selected raw @0.40 : "
        f"{raw_selected:,} ({raw_pct:.2%})"
    )

    print(
        f"Selected cal @0.40 : "
        f"{calibrated_selected:,} ({calibrated_pct:.2%})"
    )

    print(
        f"Count change       : "
        f"{calibrated_selected - raw_selected:+,}"
    )

    print()

    print("FILES CREATED")
    print(PREDICTIONS_PATH)
    print(SUMMARY_PATH)
    print(RELIABILITY_PATH)

    print()
    print(
        "FINAL FEATURE POLICY IS LOCKED."
    )

    print(
        "SIGMOID METHOD REMAINS LOCKED."
    )

    print(
        "FINAL HOLDOUT REMAINS SEALED."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()