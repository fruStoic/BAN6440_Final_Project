"""
Data loading utilities for the BAN6440 Final Project.

The development/holdout boundary is read from the permanently
recorded split assignments. This module never redraws the split.
"""

import pandas as pd

from .config import (
    DATA_PATH,
    OUTPUT_DIR,
)


ASSIGNMENTS_PATH = (
    OUTPUT_DIR
    / "split_assignments.csv"
)


def load_raw_data() -> pd.DataFrame:
    """Load the unchanged supplied course dataset."""
    return pd.read_csv(DATA_PATH)


def load_split_assignments() -> pd.DataFrame:
    """Load permanent Final Project split membership."""

    if not ASSIGNMENTS_PATH.exists():
        raise FileNotFoundError(
            "split_assignments.csv not found. "
            "The permanent split must exist first."
        )

    return pd.read_csv(
        ASSIGNMENTS_PATH
    )


def load_development_data() -> pd.DataFrame:
    """
    Return development rows only.

    The final holdout is not returned or inspected.
    """

    df = load_raw_data()
    assignments = load_split_assignments()

    dev_indices = (
        assignments.loc[
            assignments["split"] == "development",
            "row_index",
        ]
        .astype(int)
        .to_numpy()
    )

    return (
        df.iloc[dev_indices]
        .copy()
        .reset_index(drop=True)
    )


def load_holdout_data() -> pd.DataFrame:
    """
    Return final holdout rows.

    IMPORTANT:
    Do not call this function during development-stage
    model selection, calibration decisions, ablation,
    or preprocessing-policy selection.
    """

    df = load_raw_data()
    assignments = load_split_assignments()

    holdout_indices = (
        assignments.loc[
            assignments["split"] == "holdout",
            "row_index",
        ]
        .astype(int)
        .to_numpy()
    )

    return (
        df.iloc[holdout_indices]
        .copy()
        .reset_index(drop=True)
    )