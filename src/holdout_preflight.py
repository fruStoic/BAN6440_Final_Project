"""
Preflight verification before the ONE-TIME final holdout evaluation.

This script does NOT load holdout feature data.
"""

import hashlib
import json

import numpy as np

from .config import (
    OUTPUT_DIR,
    FINAL_PROJECT_SPLIT_SEED,
    OPERATIONAL_THRESHOLD,
    EXACT_BREAK_EVEN_THRESHOLD,
    FINAL_EXCLUDED_PREDICTORS,
    CALIBRATION_METHOD,
)

from .data_loader import (
    load_development_data,
    load_split_assignments,
)

from .preprocessing import (
    split_features_target,
    build_preprocessor,
)


FREEZE_PATH = OUTPUT_DIR / "pipeline_freeze.json"
MANIFEST_PATH = OUTPUT_DIR / "split_manifest.json"
SELECTION_PATH = OUTPUT_DIR / "model_selection.json"
PROTECTED_PATH = OUTPUT_DIR / "protected_attribute_decision.json"
CALIBRATION_PATH = OUTPUT_DIR / "final_feature_calibration_summary.json"

FINAL_RESULTS_PATH = OUTPUT_DIR / "final_holdout_results.json"
HOLDOUT_OPENED_PATH = OUTPUT_DIR / "holdout_opened.json"


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

    print()
    print("=" * 72)
    print("FINAL HOLDOUT PREFLIGHT")
    print("=" * 72)

    # ---------------------------------------------------------
    # 1. No prior holdout evaluation.
    # ---------------------------------------------------------

    if FINAL_RESULTS_PATH.exists():
        raise RuntimeError(
            "Final holdout results already exist. "
            "Do not evaluate again."
        )

    if HOLDOUT_OPENED_PATH.exists():
        raise RuntimeError(
            "holdout_opened.json already exists. "
            "The final holdout has already been opened."
        )

    print("[PASS] No previous final-holdout evaluation found.")

    # ---------------------------------------------------------
    # 2. Required frozen artifacts exist.
    # ---------------------------------------------------------

    required = [
        FREEZE_PATH,
        MANIFEST_PATH,
        SELECTION_PATH,
        PROTECTED_PATH,
        CALIBRATION_PATH,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing frozen artifacts:\n"
            + "\n".join(missing)
        )

    print("[PASS] Required frozen artifacts exist.")

    # ---------------------------------------------------------
    # 3. Verify freeze artifact.
    # ---------------------------------------------------------

    with FREEZE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        freeze = json.load(file)

    assert freeze["status"] == "FROZEN_BEFORE_HOLDOUT"

    assert (
        freeze["selected_model"]["type"]
        == "logistic_regression"
    )

    assert (
        freeze["calibration"]["method"]
        == CALIBRATION_METHOD
        == "sigmoid"
    )

    assert (
        freeze["decision_rule"]["operational_threshold"]
        == OPERATIONAL_THRESHOLD
        == 0.40
    )

    assert np.isclose(
        freeze["decision_rule"]["exact_economic_threshold"],
        EXACT_BREAK_EVEN_THRESHOLD,
    )

    assert (
        freeze["feature_policy"]["excluded_predictors"]
        == FINAL_EXCLUDED_PREDICTORS
    )

    print("[PASS] Frozen model and decision rule verified.")

    # ---------------------------------------------------------
    # 4. Verify frozen artifact hashes.
    # ---------------------------------------------------------

    hashes = freeze[
        "development_artifact_hashes"
    ]

    hash_paths = {
        "split_manifest.json": MANIFEST_PATH,
        "model_selection.json": SELECTION_PATH,
        "protected_attribute_decision.json": PROTECTED_PATH,
        "final_feature_calibration_summary.json": CALIBRATION_PATH,
    }

    for name, path in hash_paths.items():

        actual = sha256_file(path)

        expected = hashes[name]

        if actual != expected:
            raise RuntimeError(
                f"Frozen artifact changed after freeze: {name}"
            )

    print("[PASS] Frozen development artifacts are unchanged.")

    # ---------------------------------------------------------
    # 5. Verify permanent split metadata WITHOUT loading holdout.
    # ---------------------------------------------------------

    assignments = load_split_assignments()

    counts = (
        assignments["split"]
        .value_counts()
        .to_dict()
    )

    assert counts["development"] == 5634
    assert counts["holdout"] == 1409

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    assert (
        manifest["split_seed"]
        == FINAL_PROJECT_SPLIT_SEED
        == 20260920
    )

    assert (
        manifest["holdout"]["rows"]
        == 1409
    )

    assert (
        manifest["holdout"]["churn"]
        == 374
    )

    print(
        "[PASS] Permanent split verified: "
        "5,634 development / 1,409 holdout."
    )

    # ---------------------------------------------------------
    # 6. Verify the FINAL predictive feature schema using
    #    development data only.
    # ---------------------------------------------------------

    development = load_development_data()

    X, y = split_features_target(
        development
    )

    X = X.drop(
        columns=FINAL_EXCLUDED_PREDICTORS
    )

    assert X.shape == (
        5634,
        17,
    )

    preprocessor = build_preprocessor(
        X
    )

    encoded = (
        preprocessor.fit_transform(X)
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    # Original 30 encoded predictors minus
    # SeniorCitizen and gender_Male.
    assert encoded.shape == (
        5634,
        28,
    )

    assert len(feature_names) == 28

    assert np.isfinite(
        encoded
    ).all()

    print(
        "[PASS] Final feature policy verified: "
        "17 raw predictors -> 28 encoded predictors."
    )

    # ---------------------------------------------------------
    # 7. Final summary.
    # ---------------------------------------------------------

    print()
    print("FROZEN FINAL PIPELINE")
    print(
        "Model               : logistic regression"
    )
    print(
        "Raw predictors      : 17"
    )
    print(
        "Encoded predictors  : 28"
    )
    print(
        "Excluded            : "
        f"{FINAL_EXCLUDED_PREDICTORS}"
    )
    print(
        "Calibration         : sigmoid"
    )
    print(
        "Operating threshold : 0.40"
    )
    print(
        "Exact break-even    : "
        f"{EXACT_BREAK_EVEN_THRESHOLD:.6f}"
    )

    print()
    print(
        "THE FINAL HOLDOUT HAS NOT BEEN LOADED."
    )

    print(
        "PREFLIGHT PASSED. PIPELINE IS READY "
        "FOR ONE-TIME HOLDOUT EVALUATION."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()