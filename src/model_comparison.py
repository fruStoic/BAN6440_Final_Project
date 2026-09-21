"""
Development-only matched model comparison.

Models
------
1. Majority-class baseline
2. Logistic regression
3. HistGradientBoostingClassifier
4. Incumbent Module 6 ANN

All four are evaluated on the exact same 5 x 3 repeated stratified
outer folds.

IMPORTANT
---------
The final holdout is never loaded by this script.
"""

import os

# Set before importing TensorFlow.
os.environ.setdefault(
    "TF_CPP_MIN_LOG_LEVEL",
    "2",
)
os.environ.setdefault(
    "TF_ENABLE_ONEDNN_OPTS",
    "0",
)
os.environ.setdefault(
    "TF_DETERMINISTIC_OPS",
    "1",
)

import gc
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
)
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
    train_test_split,
)

import tensorflow as tf

from .config import (
    OUTPUT_DIR,
    TABLES_DIR,
    CV_SPLITS,
    CV_REPEATS,
    CV_RANDOM_STATE,
    GLOBAL_RANDOM_SEED,
    LOGISTIC_MAX_ITER,
    LOGISTIC_SOLVER,
    HGB_LEARNING_RATE,
    HGB_MAX_ITER,
    HGB_MAX_LEAF_NODES,
    HGB_L2_REGULARIZATION,
    ANN_HIDDEN_1,
    ANN_HIDDEN_2,
    ANN_DROPOUT,
    ANN_LEARNING_RATE,
    ANN_BATCH_SIZE,
    ANN_MAX_EPOCHS,
    ANN_EARLY_STOPPING_PATIENCE,
    ANN_INNER_VALIDATION_SIZE,
    CLASSIFICATION_THRESHOLD,
)

from .data_loader import (
    load_development_data,
)

from .preprocessing import (
    split_features_target,
    build_preprocessor,
)


FOLD_RESULTS_PATH = (
    TABLES_DIR
    / "model_comparison_fold_metrics.csv"
)

SUMMARY_PATH = (
    TABLES_DIR
    / "model_comparison_summary.csv"
)

PAIRWISE_PATH = (
    TABLES_DIR
    / "model_comparison_paired_ap.csv"
)

SELECTION_PATH = (
    OUTPUT_DIR
    / "model_selection.json"
)

RUN_CONFIG_PATH = (
    OUTPUT_DIR
    / "model_comparison_run.json"
)


MODEL_ORDER = [
    "majority",
    "logistic_regression",
    "hist_gradient_boosting",
    "ann",
]

# Simplicity preference specified before results.
SIMPLICITY_ORDER = [
    "logistic_regression",
    "hist_gradient_boosting",
    "ann",
]


def set_tf_seed(seed: int) -> None:
    """
    Reset TensorFlow/Keras state for a reproducible ANN fit.
    """

    tf.keras.backend.clear_session()

    tf.keras.utils.set_random_seed(
        seed
    )

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_ann(
    input_dim: int,
    seed: int,
):
    """
    Recreate the incumbent Module 6 ANN.
    """

    set_tf_seed(seed)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(input_dim,)
        ),

        tf.keras.layers.Dense(
            ANN_HIDDEN_1,
            activation="relu",
        ),

        tf.keras.layers.Dropout(
            ANN_DROPOUT,
        ),

        tf.keras.layers.Dense(
            ANN_HIDDEN_2,
            activation="relu",
        ),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid",
        ),
    ])

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=ANN_LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
    )

    return model


def calculate_metrics(
    y_true,
    probabilities,
    predictions,
):
    """
    Return the model-comparison metrics for one outer fold.
    """

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


def evaluate_majority(
    y_train,
    y_test,
):
    """
    Majority class predicts non-churn.

    Probability score is the churn prevalence of the outer
    training fold, representing a no-feature prevalence model.
    """

    prevalence = float(
        np.mean(y_train)
    )

    probabilities = np.full(
        len(y_test),
        prevalence,
        dtype=float,
    )

    predictions = np.zeros(
        len(y_test),
        dtype=int,
    )

    return (
        probabilities,
        predictions,
        {},
    )


def evaluate_logistic(
    X_train_raw,
    y_train,
    X_test_raw,
):
    """
    Logistic regression with fold-local preprocessing.
    """

    preprocessor = (
        build_preprocessor(
            X_train_raw
        )
    )

    X_train = (
        preprocessor.fit_transform(
            X_train_raw
        )
    )

    X_test = (
        preprocessor.transform(
            X_test_raw
        )
    )

    model = LogisticRegression(
        solver=LOGISTIC_SOLVER,
        max_iter=LOGISTIC_MAX_ITER,
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities
        >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    extra = {
        "encoded_features": int(
            X_train.shape[1]
        ),
        "iterations": int(
            np.max(model.n_iter_)
        ),
    }

    return (
        probabilities,
        predictions,
        extra,
    )


def evaluate_hgb(
    X_train_raw,
    y_train,
    X_test_raw,
    fold_seed,
):
    """
    HistGradientBoosting with fold-local preprocessing.
    """

    preprocessor = (
        build_preprocessor(
            X_train_raw
        )
    )

    X_train = (
        preprocessor.fit_transform(
            X_train_raw
        )
    )

    X_test = (
        preprocessor.transform(
            X_test_raw
        )
    )

    model = HistGradientBoostingClassifier(
        learning_rate=HGB_LEARNING_RATE,
        max_iter=HGB_MAX_ITER,
        max_leaf_nodes=HGB_MAX_LEAF_NODES,
        l2_regularization=(
            HGB_L2_REGULARIZATION
        ),
        early_stopping=False,
        random_state=fold_seed,
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities
        >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    extra = {
        "encoded_features": int(
            X_train.shape[1]
        ),
        "iterations": int(
            model.n_iter_
        ),
    }

    return (
        probabilities,
        predictions,
        extra,
    )


def evaluate_ann(
    X_train_raw,
    y_train,
    X_test_raw,
    fold_seed,
):
    """
    Re-run the incumbent Module 6 ANN.

    Early stopping uses a stratified subset of the outer
    training fold only. The outer validation fold remains
    completely untouched until scoring.
    """

    preprocessor = (
        build_preprocessor(
            X_train_raw
        )
    )

    X_train_full = (
        preprocessor.fit_transform(
            X_train_raw
        )
        .astype("float32")
    )

    X_test = (
        preprocessor.transform(
            X_test_raw
        )
        .astype("float32")
    )

    y_train_array = np.asarray(
        y_train,
        dtype="int32",
    )

    (
        X_ann_train,
        X_ann_val,
        y_ann_train,
        y_ann_val,
    ) = train_test_split(
        X_train_full,
        y_train_array,
        test_size=ANN_INNER_VALIDATION_SIZE,
        stratify=y_train_array,
        random_state=fold_seed,
    )

    model = build_ann(
        input_dim=X_train_full.shape[1],
        seed=fold_seed,
    )

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=(
                ANN_EARLY_STOPPING_PATIENCE
            ),
            restore_best_weights=True,
        )
    )

    history = model.fit(
        X_ann_train,
        y_ann_train,
        validation_data=(
            X_ann_val,
            y_ann_val,
        ),
        epochs=ANN_MAX_EPOCHS,
        batch_size=ANN_BATCH_SIZE,
        callbacks=[
            early_stopping
        ],
        verbose=0,
        shuffle=True,
    )

    probabilities = (
        model.predict(
            X_test,
            verbose=0,
        )
        .reshape(-1)
    )

    predictions = (
        probabilities
        >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    val_losses = (
        history.history[
            "val_loss"
        ]
    )

    best_epoch = int(
        np.argmin(val_losses) + 1
    )

    extra = {
        "encoded_features": int(
            X_train_full.shape[1]
        ),
        "epochs_run": int(
            len(
                history.history[
                    "loss"
                ]
            )
        ),
        "best_epoch": (
            best_epoch
        ),
        "best_val_loss": float(
            np.min(
                val_losses
            )
        ),
    }

    del model
    gc.collect()

    return (
        probabilities,
        predictions,
        extra,
    )


def save_incremental(
    rows,
):
    pd.DataFrame(
        rows
    ).to_csv(
        FOLD_RESULTS_PATH,
        index=False,
    )


def make_summary(
    results: pd.DataFrame,
):
    metrics = [
        "average_precision",
        "roc_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "brier",
        "fit_seconds",
    ]

    records = []

    for model in MODEL_ORDER:

        subset = results.loc[
            results["model"] == model
        ]

        row = {
            "model": model,
            "folds": int(
                len(subset)
            ),
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

        records.append(
            row
        )

    return pd.DataFrame(
        records
    )


def paired_selection(
    results: pd.DataFrame,
    summary: pd.DataFrame,
):
    """
    Apply the pre-registered conservative practical-tie rule.

    Candidate predictive models whose paired AP difference
    from the highest-mean model is <= the SD of that paired
    difference are considered tied with the best.

    The simplest tied model is selected.
    """

    candidates = [
        model
        for model in SIMPLICITY_ORDER
    ]

    means = {
        model: float(
            summary.loc[
                summary["model"]
                == model,
                "average_precision_mean",
            ].iloc[0]
        )
        for model in candidates
    }

    best_model = max(
        candidates,
        key=means.get,
    )

    pivot = (
        results.pivot(
            index="fold_id",
            columns="model",
            values="average_precision",
        )
        .sort_index()
    )

    pairwise_records = []
    tie_set = [
        best_model
    ]

    for candidate in candidates:

        if candidate == best_model:
            continue

        differences = (
            pivot[best_model]
            - pivot[candidate]
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
            abs(
                mean_difference
            )
            <= sd_difference
        )

        if practical_tie:
            tie_set.append(
                candidate
            )

        pairwise_records.append({
            "best_mean_model": (
                best_model
            ),
            "comparison_model": (
                candidate
            ),
            "mean_ap_difference": (
                mean_difference
            ),
            "sd_paired_difference": (
                sd_difference
            ),
            "practical_tie": (
                practical_tie
            ),
        })

    selected_model = next(
        model
        for model in SIMPLICITY_ORDER
        if model in tie_set
    )

    pairwise = pd.DataFrame(
        pairwise_records
    )

    selection = {
        "highest_mean_average_precision_model": (
            best_model
        ),

        "highest_mean_average_precision": (
            means[
                best_model
            ]
        ),

        "models_practically_tied_with_best": (
            tie_set
        ),

        "selected_model": (
            selected_model
        ),

        "selection_reason": (
            "Pre-registered practical-tie rule. "
            "Among models tied with the highest "
            "mean PR-AUC, select the simplest "
            "model according to the locked order: "
            "logistic regression, gradient "
            "boosting, ANN."
        ),
    }

    return (
        pairwise,
        selection,
    )


def main():

    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Development data ONLY.
    # ---------------------------------------------------------

    development = (
        load_development_data()
    )

    X, y = (
        split_features_target(
            development
        )
    )

    # ---------------------------------------------------------
    # Create matched CV splits ONCE.
    # ---------------------------------------------------------

    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=CV_RANDOM_STATE,
    )

    folds = list(
        cv.split(
            X,
            y,
        )
    )

    assert len(folds) == (
        CV_SPLITS
        * CV_REPEATS
    )

    run_config = {
        "created_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "scope": (
            "development_only"
        ),

        "development_rows": int(
            len(X)
        ),

        "development_churn": int(
            y.sum()
        ),

        "cv": {
            "splits": CV_SPLITS,
            "repeats": CV_REPEATS,
            "total_outer_folds": (
                len(folds)
            ),
            "random_state": (
                CV_RANDOM_STATE
            ),
        },

        "classification_threshold": (
            CLASSIFICATION_THRESHOLD
        ),

        "primary_metric": (
            "average_precision"
        ),

        "models": {
            "majority": {
                "description": (
                    "Predict non-churn; "
                    "training-fold churn prevalence "
                    "used as constant probability."
                )
            },

            "logistic_regression": {
                "solver": (
                    LOGISTIC_SOLVER
                ),
                "max_iter": (
                    LOGISTIC_MAX_ITER
                ),
            },

            "hist_gradient_boosting": {
                "learning_rate": (
                    HGB_LEARNING_RATE
                ),
                "max_iter": (
                    HGB_MAX_ITER
                ),
                "max_leaf_nodes": (
                    HGB_MAX_LEAF_NODES
                ),
                "l2_regularization": (
                    HGB_L2_REGULARIZATION
                ),
                "early_stopping": False,
            },

            "ann": {
                "architecture": (
                    "30 -> 32 ReLU -> "
                    "Dropout(0.30) -> "
                    "16 ReLU -> 1 sigmoid"
                ),
                "optimizer": "SGD",
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
        },
    }

    with RUN_CONFIG_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            run_config,
            file,
            indent=2,
        )

    rows = []

    print()
    print("=" * 76)
    print(
        "DEVELOPMENT-ONLY MODEL COMPARISON"
    )
    print("=" * 76)

    print(
        f"Rows            : {len(X):,}"
    )

    print(
        f"Churn           : {int(y.sum()):,} "
        f"({y.mean():.5%})"
    )

    print(
        f"Outer CV        : "
        f"{CV_SPLITS} folds x "
        f"{CV_REPEATS} repeats "
        f"= {len(folds)} matched folds"
    )

    print(
        "Primary metric  : "
        "Average Precision / PR-AUC"
    )

    print()
    print(
        "FINAL HOLDOUT IS NOT LOADED "
        "BY THIS SCRIPT."
    )
    print()

    for fold_number, (
        train_idx,
        test_idx,
    ) in enumerate(
        folds,
        start=1,
    ):

        X_train = (
            X.iloc[
                train_idx
            ]
            .copy()
        )

        y_train = (
            y.iloc[
                train_idx
            ]
            .copy()
        )

        X_test = (
            X.iloc[
                test_idx
            ]
            .copy()
        )

        y_test = (
            y.iloc[
                test_idx
            ]
            .copy()
        )

        repeat_number = (
            (fold_number - 1)
            // CV_SPLITS
            + 1
        )

        split_number = (
            (fold_number - 1)
            % CV_SPLITS
            + 1
        )

        fold_seed = (
            GLOBAL_RANDOM_SEED
            + fold_number
        )

        print(
            f"[Fold {fold_number:02d}/"
            f"{len(folds)} | "
            f"repeat {repeat_number}, "
            f"split {split_number}]"
        )

        evaluators = [
            (
                "majority",
                lambda: evaluate_majority(
                    y_train,
                    y_test,
                ),
            ),

            (
                "logistic_regression",
                lambda: evaluate_logistic(
                    X_train,
                    y_train,
                    X_test,
                ),
            ),

            (
                "hist_gradient_boosting",
                lambda: evaluate_hgb(
                    X_train,
                    y_train,
                    X_test,
                    fold_seed,
                ),
            ),

            (
                "ann",
                lambda: evaluate_ann(
                    X_train,
                    y_train,
                    X_test,
                    fold_seed,
                ),
            ),
        ]

        for model_name, evaluator in evaluators:

            start = time.perf_counter()

            (
                probabilities,
                predictions,
                extra,
            ) = evaluator()

            elapsed = (
                time.perf_counter()
                - start
            )

            metrics = calculate_metrics(
                y_test,
                probabilities,
                predictions,
            )

            row = {
                "fold_id": (
                    fold_number
                ),
                "repeat": (
                    repeat_number
                ),
                "split": (
                    split_number
                ),
                "model": (
                    model_name
                ),
                "train_rows": int(
                    len(train_idx)
                ),
                "validation_rows": int(
                    len(test_idx)
                ),
                "train_churn_rate": float(
                    y_train.mean()
                ),
                "validation_churn_rate": float(
                    y_test.mean()
                ),
                **metrics,
                "fit_seconds": float(
                    elapsed
                ),
                **extra,
            }

            rows.append(
                row
            )

            save_incremental(
                rows
            )

            print(
                f"  {model_name:<25} "
                f"AP={metrics['average_precision']:.4f}  "
                f"AUC={metrics['roc_auc']:.4f}  "
                f"F1={metrics['f1']:.4f}  "
                f"time={elapsed:.1f}s"
            )

        print()

    # ---------------------------------------------------------
    # Final fold-level table.
    # ---------------------------------------------------------

    results = pd.DataFrame(
        rows
    )

    results.to_csv(
        FOLD_RESULTS_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Summary.
    # ---------------------------------------------------------

    summary = make_summary(
        results
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Pre-registered model selection.
    # ---------------------------------------------------------

    (
        pairwise,
        selection,
    ) = paired_selection(
        results,
        summary,
    )

    pairwise.to_csv(
        PAIRWISE_PATH,
        index=False,
    )

    with SELECTION_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            selection,
            file,
            indent=2,
        )

    # ---------------------------------------------------------
    # Console summary.
    # ---------------------------------------------------------

    print()
    print("=" * 76)
    print(
        "MODEL COMPARISON SUMMARY"
    )
    print("=" * 76)

    display_columns = [
        "model",
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
                f"{x:.4f}"
            ),
        )
    )

    print()
    print(
        "PAIRED PRACTICAL-TIE CHECK"
    )

    if len(pairwise):
        print(
            pairwise.to_string(
                index=False,
                float_format=lambda x: (
                    f"{x:.6f}"
                ),
            )
        )

    print()
    print(
        "Highest mean AP : "
        f"{selection['highest_mean_average_precision_model']}"
    )

    print(
        "Tied set       : "
        f"{selection['models_practically_tied_with_best']}"
    )

    print(
        "Selected model : "
        f"{selection['selected_model']}"
    )

    print()
    print("FILES CREATED")
    print(FOLD_RESULTS_PATH)
    print(SUMMARY_PATH)
    print(PAIRWISE_PATH)
    print(SELECTION_PATH)
    print(RUN_CONFIG_PATH)

    print()
    print(
        "FINAL HOLDOUT REMAINS SEALED."
    )
    print("=" * 76)


if __name__ == "__main__":
    main()