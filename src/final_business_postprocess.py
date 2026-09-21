import json

import pandas as pd

from .config import (
    OUTPUT_DIR,
    TABLES_DIR,
    OPERATIONAL_THRESHOLD,
    EXACT_BREAK_EVEN_THRESHOLD,
    CONTACT_COST,
    OFFER_COST,
    ACCEPTANCE_RATE,
    INCREMENTAL_RETENTION_UPLIFT,
    CONTRIBUTION_MARGIN_RATE,
    MONTHLY_REVENUE_PROXY,
    HORIZON_MONTHS,
)


PREDICTIONS_PATH = (
    TABLES_DIR
    / "final_holdout_predictions.csv"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "final_business_impact_operational.json"
)


def calculate(probabilities, threshold):

    selected = (
        probabilities >= threshold
    )

    n = int(
        selected.sum()
    )

    probability_sum = float(
        probabilities[selected].sum()
    )

    cost_per_contact = (
        CONTACT_COST
        + ACCEPTANCE_RATE
        * OFFER_COST
    )

    margin = (
        CONTRIBUTION_MARGIN_RATE
        * MONTHLY_REVENUE_PROXY
        * HORIZON_MONTHS
    )

    benefit_coefficient = (
        INCREMENTAL_RETENTION_UPLIFT
        * margin
    )

    cost = (
        n * cost_per_contact
    )

    benefit = (
        benefit_coefficient
        * probability_sum
    )

    net = (
        benefit - cost
    )

    roi = (
        net / cost
    )

    prevented = (
        INCREMENTAL_RETENTION_UPLIFT
        * probability_sum
    )

    return {
        "threshold": float(
            threshold
        ),
        "selected": n,
        "selected_percent": float(
            n / len(probabilities)
        ),
        "probability_sum": (
            probability_sum
        ),
        "expected_prevented_churn": float(
            prevented
        ),
        "campaign_cost": float(
            cost
        ),
        "expected_benefit": float(
            benefit
        ),
        "expected_net": float(
            net
        ),
        "roi": float(
            roi
        ),
    }


def main():

    df = pd.read_csv(
        PREDICTIONS_PATH
    )

    probabilities = (
        df[
            "calibrated_probability"
        ]
        .to_numpy()
    )

    operational = calculate(
        probabilities,
        OPERATIONAL_THRESHOLD,
    )

    exact = calculate(
        probabilities,
        EXACT_BREAK_EVEN_THRESHOLD,
    )

    output = {
        "note": (
            "Post-processing correction from immutable "
            "saved holdout predictions. No model was "
            "retrained and no threshold was changed."
        ),
        "operational_0_40": (
            operational
        ),
        "exact_break_even": (
            exact
        ),
        "rounding_effect": {
            "customer_difference": (
                operational["selected"]
                - exact["selected"]
            ),
            "net_value_difference": (
                operational["expected_net"]
                - exact["expected_net"]
            ),
        },
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print()
    print("OPERATIONAL BUSINESS RESULT @ 0.40")
    print("=" * 55)

    print(
        f"Selected           : "
        f"{operational['selected']}"
    )
    print(
        f"Selected %         : "
        f"{operational['selected_percent']:.4%}"
    )
    print(
        f"Expected prevented : "
        f"{operational['expected_prevented_churn']:.2f}"
    )
    print(
        f"Campaign cost      : "
        f"${operational['campaign_cost']:,.2f}"
    )
    print(
        f"Expected benefit   : "
        f"${operational['expected_benefit']:,.2f}"
    )
    print(
        f"Expected net       : "
        f"${operational['expected_net']:,.2f}"
    )
    print(
        f"ROI                : "
        f"{operational['roi']:.2%}"
    )

    print()
    print(
        "Exact threshold selected:",
        exact["selected"],
    )

    print(
        "Net difference from rounding:",
        f"${output['rounding_effect']['net_value_difference']:,.2f}",
    )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()