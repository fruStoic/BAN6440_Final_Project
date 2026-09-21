"""
Development-only data audit.

No final-holdout feature data is inspected here.
"""

import json

import numpy as np
import pandas as pd

from .config import (
    OUTPUT_DIR,
    TABLES_DIR,
)

from .data_loader import (
    load_development_data,
)

from .preprocessing import (
    clean_raw_dataframe,
    split_features_target,
    get_feature_groups,
)


AUDIT_PATH = (
    OUTPUT_DIR
    / "development_data_audit.json"
)

QUALITY_TABLE_PATH = (
    TABLES_DIR
    / "development_data_quality.csv"
)

OUTLIER_TABLE_PATH = (
    TABLES_DIR
    / "development_iqr_audit.csv"
)


def main():

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = load_development_data()

    # ---------------------------------------------------------
    # Raw audit
    # ---------------------------------------------------------

    raw_total = (
        raw["TotalCharges"]
        .astype(str)
        .str.strip()
    )

    blank_total = int(
        raw_total.eq("").sum()
    )

    blank_tenure_zero = int(
        (
            raw_total.eq("")
            & raw["tenure"].eq(0)
        ).sum()
    )

    duplicate_customers = int(
        raw["customerID"]
        .duplicated()
        .sum()
    )

    # ---------------------------------------------------------
    # Cleaning
    # ---------------------------------------------------------

    cleaned = clean_raw_dataframe(
        raw
    )

    X, y = split_features_target(
        raw
    )

    numeric, categorical = (
        get_feature_groups(X)
    )

    # ---------------------------------------------------------
    # IQR audit
    # ---------------------------------------------------------

    outlier_records = []

    for column in [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]:

        series = cleaned[
            column
        ].astype(float)

        q1 = float(
            series.quantile(0.25)
        )

        q3 = float(
            series.quantile(0.75)
        )

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_mask = (
            (series < lower)
            | (series > upper)
        )

        outlier_records.append({
            "feature": column,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": int(
                outlier_mask.sum()
            ),
            "outlier_percent": float(
                outlier_mask.mean() * 100
            ),
            "observed_min": float(
                series.min()
            ),
            "observed_max": float(
                series.max()
            ),
        })

    outlier_df = pd.DataFrame(
        outlier_records
    )

    outlier_df.to_csv(
        OUTLIER_TABLE_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Development structural checks
    # ---------------------------------------------------------

    internet_no = (
        cleaned[
            "InternetService"
        ]
        .eq("No")
    )

    internet_no_count = int(
        internet_no.sum()
    )

    internet_service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    service_alignment = {}

    for column in internet_service_columns:

        aligned = (
            cleaned.loc[
                internet_no,
                column,
            ]
            .eq("No internet service")
            .all()
        )

        service_alignment[column] = bool(
            aligned
        )

    tenure_charge_proxy = (
        cleaned["tenure"]
        * cleaned["MonthlyCharges"]
    )

    total_charge_correlation = float(
        cleaned["TotalCharges"]
        .corr(
            tenure_charge_proxy
        )
    )

    # ---------------------------------------------------------
    # Before/after quality table
    # ---------------------------------------------------------

    quality = pd.DataFrame([
        {
            "stage": "raw_development",
            "rows": len(raw),
            "columns": raw.shape[1],
            "blank_TotalCharges": blank_total,
            "missing_values": int(
                raw.isna()
                .sum()
                .sum()
            ),
            "duplicate_customerIDs": (
                duplicate_customers
            ),
        },
        {
            "stage": "clean_development",
            "rows": len(cleaned),
            "columns": cleaned.shape[1],
            "blank_TotalCharges": 0,
            "missing_values": int(
                cleaned.isna()
                .sum()
                .sum()
            ),
            "duplicate_customerIDs": int(
                cleaned[
                    "customerID"
                ]
                .duplicated()
                .sum()
            ),
        },
    ])

    quality.to_csv(
        QUALITY_TABLE_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Main JSON record
    # ---------------------------------------------------------

    audit = {
        "scope": (
            "development_only"
        ),

        "rows": int(
            len(raw)
        ),

        "columns_raw": int(
            raw.shape[1]
        ),

        "predictors_raw": int(
            X.shape[1]
        ),

        "target": {
            "no_churn": int(
                (y == 0).sum()
            ),
            "churn": int(
                (y == 1).sum()
            ),
            "churn_rate": float(
                y.mean()
            ),
        },

        "TotalCharges": {
            "raw_blank_strings": (
                blank_total
            ),
            "blank_with_tenure_zero": (
                blank_tenure_zero
            ),
            "clean_missing": int(
                cleaned[
                    "TotalCharges"
                ]
                .isna()
                .sum()
            ),
            "minimum_after_cleaning": (
                float(
                    cleaned[
                        "TotalCharges"
                    ].min()
                )
            ),
            "maximum_after_cleaning": (
                float(
                    cleaned[
                        "TotalCharges"
                    ].max()
                )
            ),
        },

        "feature_groups": {
            "numeric": numeric,
            "categorical": categorical,
            "numeric_count": len(
                numeric
            ),
            "categorical_count": len(
                categorical
            ),
        },

        "internet_service_redundancy": {
            "internet_service_no_count": (
                internet_no_count
            ),
            "service_alignment": (
                service_alignment
            ),
        },

        "totalcharges_vs_tenure_x_monthlycharges_correlation": (
            total_charge_correlation
        ),

        "outlier_policy": (
            "IQR audit only. Extreme observations "
            "are retained unless shown to be invalid "
            "customer records."
        ),
    }

    with AUDIT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            audit,
            file,
            indent=2,
        )

    # ---------------------------------------------------------
    # Console output
    # ---------------------------------------------------------

    print()
    print("=" * 68)
    print(
        "DEVELOPMENT DATA AUDIT"
    )
    print("=" * 68)

    print(
        f"Rows                 : "
        f"{len(raw):,}"
    )

    print(
        f"Raw columns          : "
        f"{raw.shape[1]}"
    )

    print(
        f"Raw predictors       : "
        f"{X.shape[1]}"
    )

    print()

    print("TARGET")
    print(
        f"No churn             : "
        f"{(y == 0).sum():,}"
    )
    print(
        f"Churn                : "
        f"{(y == 1).sum():,}"
    )
    print(
        f"Churn rate           : "
        f"{y.mean():.5%}"
    )

    print()

    print("TOTAL CHARGES")
    print(
        f"Blank strings        : "
        f"{blank_total}"
    )
    print(
        f"Blank + tenure=0     : "
        f"{blank_tenure_zero}"
    )
    print(
        f"Missing after clean  : "
        f"{cleaned['TotalCharges'].isna().sum()}"
    )

    print()

    print("FEATURE GROUPS")
    print(
        f"Numeric              : "
        f"{len(numeric)}"
    )
    print(
        f"Categorical          : "
        f"{len(categorical)}"
    )

    print()

    print("STRUCTURAL CHECKS")
    print(
        f"InternetService=No   : "
        f"{internet_no_count:,}"
    )

    for column, aligned in (
        service_alignment.items()
    ):
        print(
            f"{column:<21}: "
            f"{aligned}"
        )

    print(
        "Corr(TotalCharges, "
        "tenure*MonthlyCharges): "
        f"{total_charge_correlation:.6f}"
    )

    print()

    print("IQR AUDIT")
    print(
        outlier_df.to_string(
            index=False
        )
    )

    print()

    print("FILES CREATED")
    print(AUDIT_PATH)
    print(QUALITY_TABLE_PATH)
    print(OUTLIER_TABLE_PATH)

    print()
    print(
        "FINAL HOLDOUT FEATURES WERE NOT "
        "USED IN THIS AUDIT."
    )
    print("=" * 68)


if __name__ == "__main__":
    main()