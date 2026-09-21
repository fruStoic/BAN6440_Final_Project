"""
Development-stage fairness audit.

gender and SeniorCitizen are NOT predictive inputs in the final model.
They are used only as grouping variables for governance analysis.

This audit uses the already-created out-of-fold calibrated predictions
from the final reduced feature model.

The final holdout is never loaded.
"""

import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    brier_score_loss,
)

from .config import (
    OUTPUT_DIR,
    TABLES_DIR,
    OPERATIONAL_THRESHOLD,
)

from .data_loader import (
    load_development_data,
)


PREDICTIONS_PATH = (
    TABLES_DIR
    / "final_feature_calibration_oof_predictions.csv"
)

FAIRNESS_TABLE_PATH = (
    TABLES_DIR
    / "development_fairness_audit.csv"
)

FAIRNESS_SUMMARY_PATH = (
    OUTPUT_DIR
    / "development_fairness_summary.json"
)


def calculate_group_metrics(
    group_df: pd.DataFrame,
):
    y_true = (
        group_df["y_true"]
        .astype(int)
        .to_numpy()
    )

    probability = (
        group_df[
            "calibrated_probability"
        ]
        .astype(float)
        .to_numpy()
    )

    prediction = (
        probability
        >= OPERATIONAL_THRESHOLD
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
        np.mean(y_true)
    )

    selection_rate = float(
        np.mean(prediction)
    )

    precision = float(
        precision_score(
            y_true,
            prediction,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            y_true,
            prediction,
            zero_division=0,
        )
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else np.nan
    )

    fnr = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else np.nan
    )

    mean_probability = float(
        np.mean(probability)
    )

    calibration_gap = (
        mean_probability
        - prevalence
    )

    brier = float(
        brier_score_loss(
            y_true,
            probability,
        )
    )

    return {
        "n": int(
            len(group_df)
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

        "selection_rate": (
            selection_rate
        ),

        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),

        "precision": (
            precision
        ),

        "recall": (
            recall
        ),

        "false_positive_rate": float(
            fpr
        ),

        "false_negative_rate": float(
            fnr
        ),

        "mean_predicted_probability": (
            mean_probability
        ),

        "calibration_gap": float(
            calibration_gap
        ),

        "brier_score": (
            brier
        ),
    }


def audit_attribute(
    audit_df: pd.DataFrame,
    attribute: str,
):
    rows = []

    for group_value, group_df in (
        audit_df.groupby(
            attribute,
            dropna=False,
        )
    ):

        metrics = (
            calculate_group_metrics(
                group_df
            )
        )

        rows.append({
            "attribute": (
                attribute
            ),

            "group": str(
                group_value
            ),

            **metrics,
        })

    return rows


def main():

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Final-feature OOF predictions not found. "
            "Run final_feature_calibration first."
        )

    development = (
        load_development_data()
    )

    predictions = (
        pd.read_csv(
            PREDICTIONS_PATH
        )
    )

    if len(development) != len(
        predictions
    ):
        raise ValueError(
            "Development rows and OOF predictions "
            "do not have the same length."
        )

    # Verify target alignment before attaching audit groups.
    expected_target = (
        development["Churn"]
        .map({
            "No": 0,
            "Yes": 1,
        })
        .astype(int)
        .to_numpy()
    )

    observed_target = (
        predictions["y_true"]
        .astype(int)
        .to_numpy()
    )

    if not np.array_equal(
        expected_target,
        observed_target,
    ):
        raise ValueError(
            "OOF prediction rows do not align with "
            "development data."
        )

    audit_df = (
        predictions.copy()
    )

    # These attributes were deliberately excluded from model inputs.
    audit_df["gender"] = (
        development[
            "gender"
        ].to_numpy()
    )

    audit_df["SeniorCitizen"] = (
        development[
            "SeniorCitizen"
        ]
        .astype(int)
        .to_numpy()
    )

    rows = []

    rows.extend(
        audit_attribute(
            audit_df,
            "gender",
        )
    )

    rows.extend(
        audit_attribute(
            audit_df,
            "SeniorCitizen",
        )
    )

    fairness = pd.DataFrame(
        rows
    )

    fairness.to_csv(
        FAIRNESS_TABLE_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Difference summaries
    # ---------------------------------------------------------

    summary = {}

    for attribute in [
        "gender",
        "SeniorCitizen",
    ]:

        subset = (
            fairness.loc[
                fairness[
                    "attribute"
                ] == attribute
            ]
            .copy()
        )

        summary[
            attribute
        ] = {
            "groups": (
                subset[
                    "group"
                ].tolist()
            ),

            "selection_rate_range": float(
                subset[
                    "selection_rate"
                ].max()
                -
                subset[
                    "selection_rate"
                ].min()
            ),

            "precision_range": float(
                subset[
                    "precision"
                ].max()
                -
                subset[
                    "precision"
                ].min()
            ),

            "recall_range": float(
                subset[
                    "recall"
                ].max()
                -
                subset[
                    "recall"
                ].min()
            ),

            "fpr_range": float(
                subset[
                    "false_positive_rate"
                ].max()
                -
                subset[
                    "false_positive_rate"
                ].min()
            ),

            "fnr_range": float(
                subset[
                    "false_negative_rate"
                ].max()
                -
                subset[
                    "false_negative_rate"
                ].min()
            ),

            "calibration_gap_range": float(
                subset[
                    "calibration_gap"
                ].max()
                -
                subset[
                    "calibration_gap"
                ].min()
            ),
        }

    with FAIRNESS_SUMMARY_PATH.open(
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
    print("=" * 100)
    print(
        "DEVELOPMENT FAIRNESS AUDIT"
    )
    print("=" * 100)

    print(
        "Predictive model excludes gender "
        "and SeniorCitizen."
    )

    print(
        "These variables are used only "
        "for group-level auditing."
    )

    print(
        f"Decision threshold: "
        f"{OPERATIONAL_THRESHOLD:.2f}"
    )

    print()

    display_columns = [
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
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print()

    print("GROUP DIFFERENCE RANGES")

    for attribute, values in (
        summary.items()
    ):

        print()
        print(attribute)

        print(
            "  selection-rate range : "
            f"{values['selection_rate_range']:.4f}"
        )

        print(
            "  precision range      : "
            f"{values['precision_range']:.4f}"
        )

        print(
            "  recall range         : "
            f"{values['recall_range']:.4f}"
        )

        print(
            "  FPR range            : "
            f"{values['fpr_range']:.4f}"
        )

        print(
            "  FNR range            : "
            f"{values['fnr_range']:.4f}"
        )

        print(
            "  calibration-gap range: "
            f"{values['calibration_gap_range']:.4f}"
        )

    print()
    print("FILES CREATED")
    print(FAIRNESS_TABLE_PATH)
    print(FAIRNESS_SUMMARY_PATH)

    print()
    print(
        "THIS IS A DEVELOPMENT-STAGE "
        "GOVERNANCE AUDIT ONLY."
    )

    print(
        "FINAL SEGMENT METRICS WILL BE "
        "REPORTED ON THE HOLDOUT AFTER FREEZE."
    )

    print(
        "FINAL HOLDOUT REMAINS SEALED."
    )

    print("=" * 100)


if __name__ == "__main__":
    main()