import json

import numpy as np

from src.config import (
    OUTPUT_DIR,
    FINAL_PROJECT_SPLIT_SEED,
    CV_SPLITS,
    CV_REPEATS,
    CV_RANDOM_STATE,
    CALIBRATION_METHOD,
    OPERATIONAL_THRESHOLD,
    EXACT_BREAK_EVEN_THRESHOLD,
    FINAL_EXCLUDED_PREDICTORS,
)

from src.data_loader import (
    load_development_data,
    load_split_assignments,
)

from src.preprocessing import (
    clean_raw_dataframe,
    split_features_target,
    build_preprocessor,
)


def test_split_assignments_are_locked():
    assignments = load_split_assignments()

    assert len(assignments) == 7043

    assert assignments[
        "customerID"
    ].is_unique

    counts = (
        assignments["split"]
        .value_counts()
        .to_dict()
    )

    assert counts["development"] == 5634
    assert counts["holdout"] == 1409


def test_development_count_and_target():
    development = load_development_data()

    assert len(development) == 5634

    counts = (
        development["Churn"]
        .value_counts()
        .to_dict()
    )

    assert counts["No"] == 4139
    assert counts["Yes"] == 1495


def test_totalcharges_cleaning():
    development = load_development_data()

    raw_total = (
        development["TotalCharges"]
        .astype(str)
        .str.strip()
    )

    blank_mask = raw_total.eq("")

    assert blank_mask.sum() == 8

    assert (
        development.loc[
            blank_mask,
            "tenure",
        ]
        == 0
    ).all()

    cleaned = clean_raw_dataframe(
        development
    )

    assert (
        cleaned["TotalCharges"]
        .isna()
        .sum()
        == 0
    )

    assert (
        cleaned.loc[
            blank_mask,
            "TotalCharges",
        ]
        == 0
    ).all()


def test_feature_target_split():
    development = load_development_data()

    X, y = split_features_target(
        development
    )

    assert "customerID" not in X.columns
    assert "Churn" not in X.columns

    assert X.shape == (
        5634,
        19,
    )

    assert set(
        y.unique()
    ) == {
        0,
        1,
    }


def test_full_preprocessor_schema():
    development = load_development_data()

    X, _ = split_features_target(
        development
    )

    preprocessor = build_preprocessor(
        X
    )

    encoded = (
        preprocessor.fit_transform(
            X
        )
    )

    names = (
        preprocessor
        .get_feature_names_out()
    )

    assert encoded.shape == (
        5634,
        30,
    )

    assert len(names) == 30

    assert np.isfinite(
        encoded
    ).all()


def test_final_feature_policy():
    development = load_development_data()

    X, _ = split_features_target(
        development
    )

    reduced = X.drop(
        columns=FINAL_EXCLUDED_PREDICTORS
    )

    assert reduced.shape[1] == 17

    assert (
        "gender"
        not in reduced.columns
    )

    assert (
        "SeniorCitizen"
        not in reduced.columns
    )

    # Audit variables still exist in original data.
    assert (
        "gender"
        in development.columns
    )

    assert (
        "SeniorCitizen"
        in development.columns
    )


def test_final_reduced_preprocessor_is_finite():
    development = load_development_data()

    X, _ = split_features_target(
        development
    )

    X = X.drop(
        columns=FINAL_EXCLUDED_PREDICTORS
    )

    preprocessor = build_preprocessor(
        X
    )

    encoded = (
        preprocessor.fit_transform(
            X
        )
    )

    names = (
        preprocessor
        .get_feature_names_out()
    )

    assert encoded.shape[0] == 5634

    assert encoded.shape[1] == len(
        names
    )

    assert np.isfinite(
        encoded
    ).all()

    assert not any(
        name == "SeniorCitizen"
        or name.startswith(
            "gender_"
        )
        for name in names
    )


def test_locked_protocol_constants():
    assert (
        FINAL_PROJECT_SPLIT_SEED
        == 20260920
    )

    assert CV_SPLITS == 5
    assert CV_REPEATS == 3

    assert (
        CV_RANDOM_STATE
        == 20260920
    )

    assert (
        CALIBRATION_METHOD
        == "sigmoid"
    )

    assert (
        OPERATIONAL_THRESHOLD
        == 0.40
    )

    assert np.isclose(
        EXACT_BREAK_EVEN_THRESHOLD,
        28 / 70.2,
    )


def test_model_selection_artifact():
    path = (
        OUTPUT_DIR
        / "model_selection.json"
    )

    assert path.exists()

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        record = json.load(
            file
        )

    assert (
        record[
            "selected_model"
        ]
        == "logistic_regression"
    )


def test_protected_attribute_decision():
    path = (
        OUTPUT_DIR
        / "protected_attribute_decision.json"
    )

    assert path.exists()

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        record = json.load(
            file
        )

    assert record[
        "practical_tie"
    ] is True

    assert (
        record["decision"]
        == "drop_gender_and_SeniorCitizen"
    )


def test_pipeline_is_frozen():
    path = (
        OUTPUT_DIR
        / "pipeline_freeze.json"
    )

    assert path.exists()

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        freeze = json.load(
            file
        )

    assert (
        freeze["status"]
        == "FROZEN_BEFORE_HOLDOUT"
    )

    assert (
        freeze[
            "selected_model"
        ]["type"]
        == "logistic_regression"
    )

    assert (
        freeze[
            "feature_policy"
        ]["expected_raw_predictors"]
        == 17
    )

    assert (
        freeze[
            "calibration"
        ]["method"]
        == "sigmoid"
    )

    assert (
        freeze[
            "decision_rule"
        ]["operational_threshold"]
        == 0.40
    )

    assert (
        freeze[
            "holdout_plan"
        ]["no_post_holdout_tuning"]
        is True
    )