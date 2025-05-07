"""Tests for the exploration spec module."""

import geoflex.experiment_design
import geoflex.exploration_spec
import geoflex.metrics
import pytest


ExperimentType = geoflex.experiment_design.ExperimentType
GeoAssignment = geoflex.experiment_design.GeoAssignment
ExperimentDesignExplorationSpec = (
    geoflex.exploration_spec.ExperimentDesignExplorationSpec
)
GeoEligibility = geoflex.experiment_design.GeoEligibility
ExperimentBudget = geoflex.experiment_design.ExperimentBudget
ExperimentBudgetType = geoflex.experiment_design.ExperimentBudgetType
CellVolumeConstraint = geoflex.experiment_design.CellVolumeConstraint

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


@pytest.mark.parametrize(
    "invalid_args",
    [
        pytest.param(
            {"runtime_weeks_candidates": []},
            id="runtime_weeks_candidates_is_empty",
        ),
        pytest.param(
            {"n_cells": 1},
            id="n_cells_less_than_2",
        ),
        pytest.param(
            {
                "cell_volume_constraint_candidates": [
                    CellVolumeConstraint(values=[5, 1])
                ],
                "n_cells": 3,
            },
            id="cell_volume_constraint_not_match_n_cells",
        ),
        pytest.param(
            {
                "n_cells": 3,
                "geo_eligibility_candidates": [
                    GeoEligibility(treatment=[{"US"}])
                ],
            },
            id="geo_eligibility_does_not_match_n_cells",
        ),
        pytest.param(
            {"alternative_hypothesis": "something_else"},
            id="alternative_hypothesis_is_invalid",
        ),
        pytest.param(
            {"alpha": 1.1},
            id="alpha_is_greater_than_1",
        ),
        pytest.param(
            {"alpha": -0.1},
            id="alpha_is_less_than_0",
        ),
        pytest.param(
            {
                "fixed_geo_candidates": [
                    GeoAssignment(treatment=[["US", "UK"], ["CA"]])
                ]
            },
            id="fixed_geo_candidates_does_not_match_n_cells",
        ),
        pytest.param(
            {"secondary_metrics": ["revenue", "conversions"]},
            id="secondary_metrics_contains_duplicate_names",
        ),
        pytest.param(
            {
                "experiment_type": ExperimentType.HEAVY_UP,
                "experiment_budget_candidates": [
                    ExperimentBudget(
                        value=-10,
                        budget_type=ExperimentBudgetType.DAILY_BUDGET,
                    )
                ],
            },
            id="heavy_up_experiment_must_have_positive_budget",
        ),
        pytest.param(
            {
                "experiment_type": ExperimentType.HOLD_BACK,
                "experiment_budget_candidates": [
                    ExperimentBudget(
                        value=-10,
                        budget_type=ExperimentBudgetType.DAILY_BUDGET,
                    )
                ],
            },
            id="hold_back_experiment_must_have_positive_budget",
        ),
        pytest.param(
            {
                "experiment_type": ExperimentType.HOLD_BACK,
                "experiment_budget_candidates": [
                    ExperimentBudget(
                        value=0.5,
                        budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
                    )
                ],
            },
            id="hold_back_experiment_cannot_have_percentage_change_budget",
        ),
        pytest.param(
            {
                "experiment_type": ExperimentType.GO_DARK,
                "experiment_budget_candidates": [
                    ExperimentBudget(
                        value=0.5,
                        budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
                    )
                ],
            },
            id="go_dark_experiment_cannot_have_positive_budget",
        ),
        pytest.param(
            {
                "experiment_type": ExperimentType.GO_DARK,
                "experiment_budget_candidates": [
                    ExperimentBudget(
                        value=-0.5,
                        budget_type=ExperimentBudgetType.DAILY_BUDGET,
                    )
                ],
            },
            id="go_dark_experiment_cannot_have_daily_budget",
        ),
        pytest.param(
            {
                "experiment_type": ExperimentType.GO_DARK,
                "experiment_budget_candidates": [
                    ExperimentBudget(
                        value=-0.5,
                        budget_type=ExperimentBudgetType.TOTAL_BUDGET,
                    )
                ],
            },
            id="go_dark_experiment_cannot_have_total_budget",
        ),
        pytest.param(
            {
                "experiment_type": ExperimentType.AB_TEST,
                "experiment_budget_candidates": [
                    ExperimentBudget(
                        value=-0.5,
                        budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
                    )
                ],
            },
            id="ab_experiment_cannot_have_budget",
        ),
        pytest.param(
            {
                "experiment_type": ExperimentType.GO_DARK,
                "experiment_budget_candidates": [None],
            },
            id="go_dark_cannot_have_zero_budget",
        ),
        pytest.param(
            {
                "experiment_type": ExperimentType.HOLD_BACK,
                "experiment_budget_candidates": [None],
            },
            id="hold_back_cannot_have_zero_budget",
        ),
        pytest.param(
            {
                "experiment_type": ExperimentType.HEAVY_UP,
                "experiment_budget_candidates": [None],
            },
            id="heavy_up_cannot_have_zero_budget",
        ),
        pytest.param(
            {
                "experiment_type": ExperimentType.AB_TEST,
                "experiment_budget_candidates": [None],
                "secondary_metrics": [geoflex.metrics.ROAS()],
            },
            id="ab_test_cannot_have_cost_secondary_metrics",
        ),
        pytest.param(
            {
                "experiment_type": ExperimentType.AB_TEST,
                "experiment_budget_candidates": [None],
                "primary_metric": geoflex.metrics.ROAS(),
            },
            id="ab_test_cannot_have_cost_secondary_metrics",
        ),
    ],
)
def test_explore_spec_raise_exception_inputs_are_invalid(invalid_args):
  default_args = {
      "experiment_type": ExperimentType.GO_DARK,
      "primary_metric": "revenue",
      "experiment_budget_candidates": [
          ExperimentBudget(
              value=-0.1,
              budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
          )
      ],
  }
  default_args.update(invalid_args)
  with pytest.raises(ValueError):
    ExperimentDesignExplorationSpec(**default_args)


@pytest.mark.parametrize(
    "valid_args",
    [
        {
            "geo_eligibility_candidates": [
                GeoAssignment(treatment=[["US", "UK"]], control=["CA", "AU"])
            ],
            "runtime_weeks_candidates": [2, 4],
        },
        {},
        {"geo_eligibility_candidates": [None]},
        {
            "cell_volume_constraint_candidates": [
                CellVolumeConstraint(values=[2, 2]),
                CellVolumeConstraint(values=[1, 5]),
                None,
            ]
        },
        {"n_cells": 3},
        {"secondary_metrics": ["conversions"]},
        {
            "experiment_type": ExperimentType.GO_DARK,
            "experiment_budget_candidates": [
                ExperimentBudget(
                    value=-0.1,
                    budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
                )
            ],
        },
    ],
)
def test_explore_spec_can_be_created_with_valid_input(valid_args):
  default_args = {
      "experiment_type": ExperimentType.AB_TEST,
      "primary_metric": "revenue",
  }
  default_args.update(valid_args)

  ExperimentDesignExplorationSpec(**default_args)


def test_explore_spec_takes_first_budget_candidate_if_budget_is_irrelevant():
  # Budget is irrelevant because no cost metrics are used,
  explore_spec = ExperimentDesignExplorationSpec(
      experiment_type=ExperimentType.GO_DARK,
      primary_metric="revenue",
      secondary_metrics=["conversions"],
      experiment_budget_candidates=[
          ExperimentBudget(
              value=-0.1,
              budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
          ),
          ExperimentBudget(
              value=100,
              budget_type=ExperimentBudgetType.DAILY_BUDGET,
          ),
      ],
  )

  expected_budget_candidates = [
      ExperimentBudget(
          value=-0.1,
          budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
      )
  ]
  assert explore_spec.experiment_budget_candidates == expected_budget_candidates
