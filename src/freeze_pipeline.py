"""
Freeze the Final Project pipeline before opening the final holdout.

Run once after all development-stage decisions are complete.

This script DOES NOT load the holdout.
"""

from datetime import datetime, timezone
import hashlib
import json

from .config import (
    OUTPUT_DIR,
    FINAL_PROJECT_SPLIT_SEED,
    CV_SPLITS,
    CV_REPEATS,
    CV_RANDOM_STATE,
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
    CONTRIBUTION_MARGIN_RATE,
    MONTHLY_REVENUE_PROXY,
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


FREEZE_PATH = (
    OUTPUT_DIR
    / "pipeline_freeze.json"
)

SPLIT_MANIFEST_PATH = (
    OUTPUT_DIR
    / "split_manifest.json"
)

SELECTION_PATH = (
    OUTPUT_DIR
    / "model_selection.json"
)

PROTECTED_DECISION_PATH = (
    OUTPUT_DIR
    / "protected_attribute_decision.json"
)

CALIBRATION_PATH = (
    OUTPUT_DIR
    / "final_feature_calibration_summary.json"
)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main():

    if FREEZE_PATH.exists():
        raise RuntimeError(
            "Pipeline freeze already exists. "
            "Do not overwrite it."
        )

    required_files = [
        SPLIT_MANIFEST_PATH,
        SELECTION_PATH,
        PROTECTED_DECISION_PATH,
        CALIBRATION_PATH,
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required development artifacts missing:\n"
            + "\n".join(missing)
        )

    with SELECTION_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        selection = json.load(file)

    with PROTECTED_DECISION_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        protected_decision = json.load(file)

    if (
        selection["selected_model"]
        != "logistic_regression"
    ):
        raise ValueError(
            "Unexpected selected model."
        )

    if (
        protected_decision["decision"]
        != "drop_gender_and_SeniorCitizen"
    ):
        raise ValueError(
            "Unexpected protected-attribute decision."
        )

    freeze = {
        "frozen_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "status": (
            "FROZEN_BEFORE_HOLDOUT"
        ),

        "development_protocol": {
            "split_seed": (
                FINAL_PROJECT_SPLIT_SEED
            ),
            "cv_splits": (
                CV_SPLITS
            ),
            "cv_repeats": (
                CV_REPEATS
            ),
            "cv_random_state": (
                CV_RANDOM_STATE
            ),
            "primary_metric": (
                "average_precision"
            ),
        },

        "selected_model": {
            "type": (
                "logistic_regression"
            ),
            "solver": (
                LOGISTIC_SOLVER
            ),
            "max_iter": (
                LOGISTIC_MAX_ITER
            ),
        },

        "feature_policy": {
            "excluded_predictors": (
                FINAL_EXCLUDED_PREDICTORS
            ),
            "expected_raw_predictors": 17,
            "purpose_of_excluded_variables": (
                "Fairness auditing only"
            ),
        },

        "preprocessing": {
            "numeric": (
                "StandardScaler fitted on training data"
            ),
            "categorical": (
                "OneHotEncoder(drop='first', "
                "handle_unknown='ignore')"
            ),
        },

        "calibration": {
            "method": (
                CALIBRATION_METHOD
            ),
            "cv": (
                CALIBRATION_CV
            ),
            "note": (
                "Sigmoid retained because it was "
                "pre-specified despite negligible "
                "development Brier improvement."
            ),
        },

        "decision_rule": {
            "operational_threshold": (
                OPERATIONAL_THRESHOLD
            ),
            "exact_economic_threshold": (
                EXACT_BREAK_EVEN_THRESHOLD
            ),
        },

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

        "incumbent_comparator": {
            "name": (
                "Module 6 ANN"
            ),
            "architecture": (
                f"{ANN_HIDDEN_1} ReLU -> "
                f"Dropout({ANN_DROPOUT}) -> "
                f"{ANN_HIDDEN_2} ReLU -> "
                "sigmoid"
            ),
            "optimizer": (
                "SGD"
            ),
            "learning_rate": (
                ANN_LEARNING_RATE
            ),
            "batch_size": (
                ANN_BATCH_SIZE
            ),
            "max_epochs": (
                ANN_MAX_EPOCHS
            ),
            "early_stopping_patience": (
                ANN_EARLY_STOPPING_PATIENCE
            ),
            "inner_validation_fraction": (
                ANN_INNER_VALIDATION_SIZE
            ),
        },

        "holdout_plan": {
            "evaluate_selected_model": True,
            "evaluate_incumbent_ann": True,
            "final_group_audit": [
                "gender",
                "SeniorCitizen",
            ],
            "business_impact_analysis": True,
            "sensitivity_analysis": True,
            "no_post_holdout_tuning": True,
        },

        "development_artifact_hashes": {
            str(path.name): sha256_file(path)
            for path in required_files
        },
    }

    with FREEZE_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            freeze,
            file,
            indent=2,
        )

    print()
    print("=" * 72)
    print("FINAL PROJECT PIPELINE FROZEN")
    print("=" * 72)

    print(
        "Selected model       : logistic_regression"
    )

    print(
        "Raw predictors       : 17"
    )

    print(
        "Excluded             : "
        f"{FINAL_EXCLUDED_PREDICTORS}"
    )

    print(
        "Calibration          : "
        f"{CALIBRATION_METHOD}"
    )

    print(
        "Operating threshold  : "
        f"{OPERATIONAL_THRESHOLD:.2f}"
    )

    print(
        "Exact break-even     : "
        f"{EXACT_BREAK_EVEN_THRESHOLD:.6f}"
    )

    print()

    print(
        "Holdout evaluation   : ONE TIME ONLY"
    )

    print(
        "Post-holdout tuning  : NOT ALLOWED"
    )

    print()
    print(
        f"Freeze artifact      : {FREEZE_PATH}"
    )

    print()
    print(
        "THE FINAL HOLDOUT HAS NOT BEEN LOADED."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()