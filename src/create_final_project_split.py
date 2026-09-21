"""
Create the permanent Final Project development/holdout split.

IMPORTANT
---------
This script establishes the fresh Final Project holdout.

Run it once only.

All later scripts must load split membership from:
    outputs/split_assignments.csv

They must NOT call train_test_split() again.
"""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    OUTPUT_DIR,
    FINAL_PROJECT_SPLIT_SEED,
    HOLDOUT_SIZE,
    CV_SPLITS,
    CV_REPEATS,
    CV_RANDOM_STATE,
    PRIMARY_SELECTION_METRIC,
    CALIBRATION_METHOD,
    CALIBRATION_CV,
    CONTACT_COST,
    OFFER_COST,
    ACCEPTANCE_RATE,
    INCREMENTAL_RETENTION_UPLIFT,
    MONTHLY_REVENUE_PROXY,
    CONTRIBUTION_MARGIN_RATE,
    HORIZON_MONTHS,
    TOTAL_CONTRIBUTION_MARGIN,
    EXACT_BREAK_EVEN_THRESHOLD,
    OPERATIONAL_THRESHOLD,
)


MANIFEST_PATH = OUTPUT_DIR / "split_manifest.json"
ASSIGNMENTS_PATH = OUTPUT_DIR / "split_assignments.csv"
PROTOCOL_PATH = OUTPUT_DIR / "protocol_lock.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def validate_dataset(df: pd.DataFrame) -> None:
    required = {
        "customerID",
        "Churn",
        "TotalCharges",
        "tenure",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.shape != (7043, 21):
        raise ValueError(
            f"Expected dataset shape (7043, 21), found {df.shape}."
        )

    if df["customerID"].duplicated().any():
        raise ValueError(
            "Duplicate customerID values detected."
        )

    churn_counts = df["Churn"].value_counts().to_dict()

    expected = {
        "No": 5174,
        "Yes": 1869,
    }

    if churn_counts != expected:
        raise ValueError(
            "Unexpected Churn distribution. "
            f"Expected {expected}, found {churn_counts}."
        )


def protect_existing_split() -> None:
    """
    Prevent accidental regeneration after the split has been locked.
    """

    existing = [
        path
        for path in [
            MANIFEST_PATH,
            ASSIGNMENTS_PATH,
        ]
        if path.exists()
    ]

    if existing:
        names = "\n".join(str(path) for path in existing)

        raise RuntimeError(
            "FINAL PROJECT SPLIT ALREADY EXISTS.\n\n"
            "Do not redraw the holdout.\n"
            "Existing files:\n"
            f"{names}"
        )


def write_protocol_lock(dataset_hash: str) -> None:
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),

        "status": "LOCKED_BEFORE_FINAL_PROJECT_MODELING",

        "dataset": {
            "file": str(DATA_PATH),
            "sha256": dataset_hash,
        },

        "split": {
            "seed": FINAL_PROJECT_SPLIT_SEED,
            "holdout_fraction": HOLDOUT_SIZE,
            "stratified": True,
            "redraw_allowed": False,
        },

        "model_comparison": {
            "cross_validation": "RepeatedStratifiedKFold",
            "n_splits": CV_SPLITS,
            "n_repeats": CV_REPEATS,
            "random_state": CV_RANDOM_STATE,
            "matched_folds_across_models": True,
            "primary_metric": PRIMARY_SELECTION_METRIC,

            "practical_tie_rule": (
                "Treat the top models as practically tied when "
                "the absolute mean paired fold-level PR-AUC "
                "difference is less than or equal to the "
                "standard deviation of those paired differences."
            ),

            "tie_preference": [
                "logistic_regression",
                "hist_gradient_boosting",
                "ann",
            ],
        },

        "calibration": {
            "method": CALIBRATION_METHOD,
            "cv": CALIBRATION_CV,
            "method_pre_specified": True,
        },

        "business_decision": {
            "contact_cost": CONTACT_COST,
            "offer_cost": OFFER_COST,
            "acceptance_rate": ACCEPTANCE_RATE,

            "incremental_retention_uplift": (
                INCREMENTAL_RETENTION_UPLIFT
            ),

            "uplift_definition": (
                "Incremental probability that the retention "
                "intervention prevents churn among customers "
                "who otherwise would have churned, measured "
                "across all contacted customers."
            ),

            "monthly_revenue_proxy": MONTHLY_REVENUE_PROXY,

            "contribution_margin_rate": (
                CONTRIBUTION_MARGIN_RATE
            ),

            "horizon_months": HORIZON_MONTHS,

            "total_contribution_margin": (
                TOTAL_CONTRIBUTION_MARGIN
            ),

            "exact_break_even_threshold": (
                EXACT_BREAK_EVEN_THRESHOLD
            ),

            "operational_threshold": (
                OPERATIONAL_THRESHOLD
            ),
        },

        "hypotheses": [
            (
                "Gradient boosting is expected to perform "
                "at least as well as logistic regression "
                "and the incumbent Module 6 ANN."
            ),
            (
                "The incumbent Module 6 ANN is expected "
                "to show some probability over-confidence."
            ),
            (
                "If the selected model is over-confident, "
                "sigmoid calibration may reduce the number "
                "of customers exceeding the 0.40 economic "
                "threshold."
            ),
        ],

        "holdout_rule": (
            "The holdout will be evaluated once after model "
            "selection, calibration, threshold, protected-"
            "attribute decision and all development-stage "
            "choices are frozen."
        ),
    }

    with PROTOCOL_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            protocol,
            file,
            indent=2,
        )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Prevent accidental redraw.
    # ---------------------------------------------------------

    protect_existing_split()

    # ---------------------------------------------------------
    # Load and validate original course dataset.
    # ---------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    validate_dataset(df)

    dataset_hash = sha256_file(DATA_PATH)

    # ---------------------------------------------------------
    # Lock protocol before split generation.
    # ---------------------------------------------------------

    write_protocol_lock(dataset_hash)

    # ---------------------------------------------------------
    # Encode target only.
    # No feature preprocessing is performed here.
    # ---------------------------------------------------------

    y = (
        df["Churn"]
        .map({
            "No": 0,
            "Yes": 1,
        })
        .astype(int)
    )

    row_indices = np.arange(len(df))

    # ---------------------------------------------------------
    # Permanent stratified split.
    # ---------------------------------------------------------

    dev_indices, holdout_indices = train_test_split(
        row_indices,
        test_size=HOLDOUT_SIZE,
        stratify=y,
        random_state=FINAL_PROJECT_SPLIT_SEED,
    )

    dev_indices = np.sort(dev_indices)
    holdout_indices = np.sort(holdout_indices)

    # ---------------------------------------------------------
    # Store exact membership.
    # ---------------------------------------------------------

    assignments = pd.DataFrame({
        "row_index": row_indices,
        "customerID": df["customerID"],
        "churn": y,
        "split": "",
    })

    assignments.loc[
        dev_indices,
        "split",
    ] = "development"

    assignments.loc[
        holdout_indices,
        "split",
    ] = "holdout"

    assignments.to_csv(
        ASSIGNMENTS_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Split summaries.
    # ---------------------------------------------------------

    dev_y = y.iloc[dev_indices]
    holdout_y = y.iloc[holdout_indices]

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),

        "dataset_sha256": dataset_hash,

        "split_seed": FINAL_PROJECT_SPLIT_SEED,

        "holdout_fraction": HOLDOUT_SIZE,

        "full": {
            "rows": int(len(df)),
            "no_churn": int((y == 0).sum()),
            "churn": int((y == 1).sum()),
            "churn_rate": float(y.mean()),
        },

        "development": {
            "rows": int(len(dev_indices)),
            "no_churn": int((dev_y == 0).sum()),
            "churn": int((dev_y == 1).sum()),
            "churn_rate": float(dev_y.mean()),
        },

        "holdout": {
            "rows": int(len(holdout_indices)),
            "no_churn": int((holdout_y == 0).sum()),
            "churn": int((holdout_y == 1).sum()),
            "churn_rate": float(holdout_y.mean()),
        },

        "files": {
            "assignments": (
                "outputs/split_assignments.csv"
            ),
            "protocol_lock": (
                "outputs/protocol_lock.json"
            ),
        },
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    # ---------------------------------------------------------
    # Integrity checks.
    # ---------------------------------------------------------

    assert (
        len(dev_indices)
        + len(holdout_indices)
        == len(df)
    )

    assert not set(dev_indices).intersection(
        set(holdout_indices)
    )

    assert assignments["split"].ne("").all()

    assert assignments[
        "customerID"
    ].is_unique

    # ---------------------------------------------------------
    # Console evidence.
    # ---------------------------------------------------------

    print()
    print("=" * 68)
    print("FINAL PROJECT SPLIT LOCKED")
    print("=" * 68)

    print(
        f"Dataset SHA-256 : {dataset_hash}"
    )

    print(
        f"Split seed      : {FINAL_PROJECT_SPLIT_SEED}"
    )

    print()

    print("FULL DATASET")
    print(f"Rows            : {len(df):,}")
    print(f"No churn        : {(y == 0).sum():,}")
    print(f"Churn           : {(y == 1).sum():,}")
    print(f"Churn rate      : {y.mean():.5%}")

    print()

    print("DEVELOPMENT")
    print(f"Rows            : {len(dev_indices):,}")
    print(f"No churn        : {(dev_y == 0).sum():,}")
    print(f"Churn           : {(dev_y == 1).sum():,}")
    print(f"Churn rate      : {dev_y.mean():.5%}")

    print()

    print("FINAL HOLDOUT")
    print(f"Rows            : {len(holdout_indices):,}")
    print(f"No churn        : {(holdout_y == 0).sum():,}")
    print(f"Churn           : {(holdout_y == 1).sum():,}")
    print(f"Churn rate      : {holdout_y.mean():.5%}")

    print()

    print("BUSINESS RULE")
    print(
        "Exact threshold : "
        f"{EXACT_BREAK_EVEN_THRESHOLD:.6f}"
    )
    print(
        "Operational     : "
        f"{OPERATIONAL_THRESHOLD:.2f}"
    )

    print()

    print("FILES CREATED")
    print(PROTOCOL_PATH)
    print(MANIFEST_PATH)
    print(ASSIGNMENTS_PATH)

    print()
    print("DO NOT REDRAW THIS SPLIT.")
    print("=" * 68)


if __name__ == "__main__":
    main()