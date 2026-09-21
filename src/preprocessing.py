"""
Cleaning and preprocessing utilities.

All fitted preprocessing objects must be trained on development
training folds only. Nothing here fits against the final holdout.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


def clean_raw_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply deterministic cleaning rules established before
    final-project model comparison.

    TotalCharges:
    whitespace -> NaN -> numeric.

    Blank TotalCharges is only accepted when tenure == 0.
    Such values are set to 0 because no charges have yet accrued.
    """

    cleaned = df.copy()

    raw_total = (
        cleaned["TotalCharges"]
        .astype(str)
        .str.strip()
    )

    blank_mask = raw_total.eq("")

    invalid_blank_mask = (
        blank_mask
        & cleaned["tenure"].ne(0)
    )

    if invalid_blank_mask.any():
        raise ValueError(
            "Found blank TotalCharges for customers "
            "with tenure > 0. Cleaning rule cannot "
            "safely resolve these rows."
        )

    cleaned["TotalCharges"] = pd.to_numeric(
        raw_total.replace("", np.nan),
        errors="raise",
    )

    cleaned.loc[
        blank_mask,
        "TotalCharges",
    ] = 0.0

    if cleaned["TotalCharges"].isna().any():
        raise ValueError(
            "Missing TotalCharges remains after cleaning."
        )

    if cleaned.isna().any().any():
        missing = (
            cleaned.isna()
            .sum()
            .loc[lambda x: x > 0]
            .to_dict()
        )

        raise ValueError(
            f"Unexpected missing values remain: {missing}"
        )

    return cleaned


def split_features_target(
    df: pd.DataFrame,
):
    """
    Separate predictors and binary target.
    """

    cleaned = clean_raw_dataframe(df)

    y = (
        cleaned[TARGET_COLUMN]
        .map({
            "No": 0,
            "Yes": 1,
        })
    )

    if y.isna().any():
        raise ValueError(
            "Unexpected Churn label detected."
        )

    y = y.astype(int)

    X = cleaned.drop(
        columns=[
            ID_COLUMN,
            TARGET_COLUMN,
        ]
    )

    return X, y


def get_feature_groups(
    X: pd.DataFrame,
):
    """
    Return numeric and categorical predictor names.
    """

    numeric = [
        column
        for column in NUMERIC_COLUMNS
        if column in X.columns
    ]

    categorical = [
        column
        for column in X.columns
        if column not in numeric
    ]

    return numeric, categorical


def build_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """
    Construct but do not fit the preprocessing pipeline.

    Numeric variables:
        StandardScaler

    Categorical variables:
        OneHotEncoder(drop='first')

    Fitting occurs inside CV training folds.
    """

    numeric, categorical = (
        get_feature_groups(X)
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )