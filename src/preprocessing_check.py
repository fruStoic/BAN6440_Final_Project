"""
Development-only preprocessing schema check.

This script fits the preprocessor on the full development set ONLY
to inspect the resulting feature schema.

The fitted object produced here is NOT used for model evaluation.
During cross-validation, preprocessing will be fitted separately
inside each training fold through an sklearn Pipeline.
"""

import json

import numpy as np

from .config import (
    OUTPUT_DIR,
    TABLES_DIR,
)

from .data_loader import (
    load_development_data,
)

from .preprocessing import (
    split_features_target,
    build_preprocessor,
    get_feature_groups,
)


SCHEMA_PATH = (
    OUTPUT_DIR
    / "development_feature_schema.json"
)

FEATURE_TABLE_PATH = (
    TABLES_DIR
    / "development_feature_names.csv"
)


def main():

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Development data only
    # ---------------------------------------------------------

    development = (
        load_development_data()
    )

    X, y = (
        split_features_target(
            development
        )
    )

    numeric, categorical = (
        get_feature_groups(X)
    )

    # ---------------------------------------------------------
    # Schema inspection only
    # ---------------------------------------------------------

    preprocessor = (
        build_preprocessor(X)
    )

    X_encoded = (
        preprocessor.fit_transform(X)
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    # ---------------------------------------------------------
    # Basic checks
    # ---------------------------------------------------------

    if X_encoded.shape[0] != len(X):
        raise AssertionError(
            "Encoded row count changed."
        )

    if X_encoded.shape[1] != len(
        feature_names
    ):
        raise AssertionError(
            "Encoded feature count does not "
            "match feature-name count."
        )

    if not np.isfinite(
        X_encoded
    ).all():
        raise AssertionError(
            "Encoded matrix contains "
            "NaN or infinite values."
        )

    # Expected from the established pipeline.
    if X_encoded.shape[1] != 30:
        raise AssertionError(
            "Expected 30 encoded predictors, "
            f"found {X_encoded.shape[1]}."
        )

    # ---------------------------------------------------------
    # Check scaled numeric variables
    # ---------------------------------------------------------

    numeric_count = len(
        numeric
    )

    encoded_numeric = (
        X_encoded[
            :,
            :numeric_count
        ]
    )

    numeric_means = (
        encoded_numeric.mean(
            axis=0
        )
    )

    numeric_stds = (
        encoded_numeric.std(
            axis=0
        )
    )

    # ---------------------------------------------------------
    # Save feature names
    # ---------------------------------------------------------

    import pandas as pd

    feature_table = (
        pd.DataFrame({
            "feature_index": range(
                len(feature_names)
            ),
            "feature_name": (
                feature_names
            ),
        })
    )

    feature_table.to_csv(
        FEATURE_TABLE_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Save schema evidence
    # ---------------------------------------------------------

    schema = {
        "scope": (
            "development_only_schema_check"
        ),

        "raw_rows": int(
            X.shape[0]
        ),

        "raw_predictors": int(
            X.shape[1]
        ),

        "numeric_predictors": (
            numeric
        ),

        "categorical_predictors": (
            categorical
        ),

        "numeric_count": int(
            len(numeric)
        ),

        "categorical_count": int(
            len(categorical)
        ),

        "encoded_predictors": int(
            X_encoded.shape[1]
        ),

        "numeric_scaled_means": [
            float(value)
            for value
            in numeric_means
        ],

        "numeric_scaled_stds": [
            float(value)
            for value
            in numeric_stds
        ],

        "feature_names": (
            feature_names
        ),

        "important_note": (
            "This fitted preprocessor is used "
            "for schema inspection only. "
            "Cross-validation fits preprocessing "
            "inside each training fold."
        ),
    }

    with SCHEMA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            schema,
            file,
            indent=2,
        )

    # ---------------------------------------------------------
    # Console
    # ---------------------------------------------------------

    print()
    print("=" * 68)
    print(
        "DEVELOPMENT PREPROCESSING CHECK"
    )
    print("=" * 68)

    print(
        f"Rows                 : "
        f"{X.shape[0]:,}"
    )

    print(
        f"Raw predictors       : "
        f"{X.shape[1]}"
    )

    print(
        f"Numeric predictors   : "
        f"{len(numeric)}"
    )

    print(
        f"Categorical          : "
        f"{len(categorical)}"
    )

    print(
        f"Encoded predictors   : "
        f"{X_encoded.shape[1]}"
    )

    print()
    print("SCALED NUMERIC CHECK")

    for (
        feature,
        mean,
        std,
    ) in zip(
        numeric,
        numeric_means,
        numeric_stds,
    ):

        print(
            f"{feature:<18} "
            f"mean={mean: .6f} "
            f"std={std: .6f}"
        )

    print()
    print("FIRST 15 MODEL FEATURES")

    for index, feature in enumerate(
        feature_names[:15]
    ):
        print(
            f"{index:>2}: {feature}"
        )

    print()

    print("FILES CREATED")
    print(SCHEMA_PATH)
    print(FEATURE_TABLE_PATH)

    print()
    print(
        "NOTE: THIS PREPROCESSOR WAS FIT "
        "ONLY FOR SCHEMA INSPECTION."
    )
    print(
        "MODEL CV WILL REFIT PREPROCESSING "
        "INSIDE EACH TRAINING FOLD."
    )
    print("=" * 68)


if __name__ == "__main__":
    main()