"""
Development-only protected-attribute ablation.

Compares the selected logistic-regression model:

1. Full feature set
2. Reduced feature set excluding gender and SeniorCitizen

Both versions use the same 5 x 3 repeated stratified folds.

The final holdout is never loaded.
"""

import json

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
)
from sklearn.pipeline import Pipeline

from .config import (
    OUTPUT_DIR,
    TABLES_DIR,
    CV_SPLITS,
    CV_REPEATS,
    CV_RANDOM_STATE,
    LOGISTIC_MAX_ITER,
    LOGISTIC_SOLVER,
    CLASSIFICATION_THRESHOLD,
    PROTECTED_ATTRIBUTES,
)

from .data_loader import (
    load_development_data,
)

from .preprocessing import (
    split_features_target,
    build_preprocessor,
)


FOLD_PATH = (
    TABLES_DIR
    / "protected_attribute_ablation_folds.csv"
)

SUMMARY_PATH = (
    TABLES_DIR
    / "protected_attribute_ablation_summary.csv"
)

DECISION_PATH = (
    OUTPUT_DIR
    / "protected_attribute_decision.json"
)


def build_pipeline(X_train):
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


def calculate_metrics(
    y_true,
    probabilities,
):
    predictions = (
        probabilities
        >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    return {
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
        "brier": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
    }


def main():

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    development = (
        load_development_data()
    )

    X_full, y = split_features_target(
        development
    )

    missing = [
        column
        for column in PROTECTED_ATTRIBUTES
        if column not in X_full.columns
    ]

    if missing:
        raise ValueError(
            f"Protected attributes missing: {missing}"
        )

    X_reduced = X_full.drop(
        columns=PROTECTED_ATTRIBUTES
    )

    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=CV_RANDOM_STATE,
    )

    folds = list(
        cv.split(
            X_full,
            y,
        )
    )

    rows = []

    print()
    print("=" * 72)
    print("PROTECTED-ATTRIBUTE ABLATION")
    print("=" * 72)

    print(
        f"Development rows     : {len(y):,}"
    )

    print(
        f"Full raw predictors  : {X_full.shape[1]}"
    )

    print(
        f"Reduced predictors   : {X_reduced.shape[1]}"
    )

    print(
        "Removed              : "
        f"{PROTECTED_ATTRIBUTES}"
    )

    print(
        f"Matched folds        : {len(folds)}"
    )

    print()
    print(
        "FINAL HOLDOUT IS NOT LOADED BY THIS SCRIPT."
    )
    print()

    for fold_id, (
        train_idx,
        validation_idx,
    ) in enumerate(
        folds,
        start=1,
    ):

        y_train = (
            y.iloc[train_idx]
        )

        y_validation = (
            y.iloc[validation_idx]
        )

        for label, X in [
            (
                "full",
                X_full,
            ),
            (
                "reduced",
                X_reduced,
            ),
        ]:

            X_train = (
                X.iloc[train_idx]
                .copy()
            )

            X_validation = (
                X.iloc[validation_idx]
                .copy()
            )

            pipeline = (
                build_pipeline(
                    X_train
                )
            )

            pipeline.fit(
                X_train,
                y_train,
            )

            probabilities = (
                pipeline.predict_proba(
                    X_validation
                )[:, 1]
            )

            metrics = calculate_metrics(
                y_validation,
                probabilities,
            )

            rows.append({
                "fold_id": fold_id,
                "feature_set": label,
                **metrics,
            })

        full_ap = rows[-2][
            "average_precision"
        ]

        reduced_ap = rows[-1][
            "average_precision"
        ]

        print(
            f"Fold {fold_id:02d}: "
            f"full AP={full_ap:.5f} | "
            f"reduced AP={reduced_ap:.5f} | "
            f"full-reduced={full_ap - reduced_ap:+.5f}"
        )

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        FOLD_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    metrics = [
        "average_precision",
        "roc_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "brier",
    ]

    summary_rows = []

    for feature_set in [
        "full",
        "reduced",
    ]:

        subset = results.loc[
            results[
                "feature_set"
            ] == feature_set
        ]

        row = {
            "feature_set": (
                feature_set
            )
        }

        for metric in metrics:

            row[
                f"{metric}_mean"
            ] = float(
                subset[
                    metric
                ].mean()
            )

            row[
                f"{metric}_sd"
            ] = float(
                subset[
                    metric
                ].std(
                    ddof=1
                )
            )

        summary_rows.append(
            row
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Paired AP decision
    # ---------------------------------------------------------

    pivot = results.pivot(
        index="fold_id",
        columns="feature_set",
        values="average_precision",
    )

    differences = (
        pivot["full"]
        - pivot["reduced"]
    )

    mean_difference = float(
        differences.mean()
    )

    sd_difference = float(
        differences.std(
            ddof=1
        )
    )

    practical_tie = bool(
        abs(mean_difference)
        <= sd_difference
    )

    if practical_tie:
        decision = (
            "drop_gender_and_SeniorCitizen"
        )

        reason = (
            "The reduced feature set is practically "
            "tied with the full feature set under "
            "the pre-registered paired PR-AUC rule. "
            "The explicit demographic attributes "
            "are therefore removed."
        )

    else:
        decision = (
            "retain_gender_and_SeniorCitizen"
        )

        reason = (
            "The reduced model falls outside the "
            "pre-registered practical-tie band. "
            "The explicit demographic attributes "
            "are retained for the prototype and "
            "the governance trade-off will be "
            "reported."
        )

    decision_record = {
        "protected_attributes": (
            PROTECTED_ATTRIBUTES
        ),

        "mean_full_minus_reduced_ap": (
            mean_difference
        ),

        "sd_paired_ap_difference": (
            sd_difference
        ),

        "practical_tie": (
            practical_tie
        ),

        "decision": (
            decision
        ),

        "reason": (
            reason
        ),
    }

    with DECISION_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            decision_record,
            file,
            indent=2,
        )

    # ---------------------------------------------------------
    # Console
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print("ABLATION SUMMARY")
    print("=" * 72)

    display_columns = [
        "feature_set",
        "average_precision_mean",
        "average_precision_sd",
        "roc_auc_mean",
        "f1_mean",
        "accuracy_mean",
        "brier_mean",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print()

    print(
        "Mean paired AP difference "
        f"(full - reduced): {mean_difference:+.6f}"
    )

    print(
        "SD paired AP difference        : "
        f"{sd_difference:.6f}"
    )

    print(
        "Practical tie                 : "
        f"{practical_tie}"
    )

    print(
        "Governance decision           : "
        f"{decision}"
    )

    print()
    print("FILES CREATED")
    print(FOLD_PATH)
    print(SUMMARY_PATH)
    print(DECISION_PATH)

    print()
    print(
        "FINAL HOLDOUT REMAINS SEALED."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()