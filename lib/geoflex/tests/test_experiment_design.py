"""Tests for the experiment design module."""

import geoflex.experiment_design
import geoflex.metrics
import numpy as np
import pandas as pd
import pytest


ExperimentDesignSpec = geoflex.experiment_design.ExperimentDesignSpec
ExperimentType = geoflex.experiment_design.ExperimentType
GeoAssignment = geoflex.experiment_design.GeoAssignment
ExperimentDesign = geoflex.experiment_design.ExperimentDesign
GeoEligibility = geoflex.experiment_design.GeoEligibility
ExperimentBudget = geoflex.experiment_design.ExperimentBudget
ExperimentBudgetType = geoflex.experiment_design.ExperimentBudgetType
CellVolumeConstraint = geoflex.experiment_design.CellVolumeConstraint

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


@pytest.mark.parametrize(
    "treatment,control,exclude",
    [
        ([["US", "UK"], []], ["US", "AU"], ["FR"]),  # Treatment control overlap
        (
            [["US", "UK"], ["CA", "US"]],
            ["FR", "AU"],
            [],
        ),  # Treatment treatment overlap
        ([["US", "UK"], []], ["FR", "AU"], ["US"]),  # Treatment exclude overlap
        ([["US", "UK"], []], ["FR", "AU"], ["AU"]),  # Control exclude overlap
    ],
)
def test_geo_assignment_raises_exception_if_geos_in_multiple_groups(
    treatment, control, exclude
):
  treatment_sets = [set(arm) for arm in treatment]
  control_set = set(control)
  exclude_set = set(exclude)
  with pytest.raises(ValueError, match="multiple groups"):
    GeoAssignment(
        treatment=treatment_sets,
        control=control_set,
        exclude=exclude_set)


def test_geo_assignment_raises_exception_if_treatment_geos_are_single_list():
  with pytest.raises(ValueError):
    # Treatment must be a list of lists to be able to have multiple treatment
    # arms.
    GeoAssignment(treatment=["US", "UK"], control=["US", "AU"])


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
def test_design_spec_raise_exception_inputs_are_invalid(invalid_args):
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
    ExperimentDesignSpec(**default_args)


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
            "experiment_type": ExperimentType.AB_TEST,
            "experiment_budget_candidates": [None],
        },
    ],
)
def test_design_spec_can_be_created_with_valid_input(valid_args):
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
  default_args.update(valid_args)

  ExperimentDesignSpec(**default_args)


def test_design_spec_takes_first_budget_candidate_if_budget_is_irrelevant():
  # Budget is irrelevant because no cost metrics are used,
  design_spec = ExperimentDesignSpec(
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
  assert design_spec.experiment_budget_candidates == expected_budget_candidates


@pytest.mark.parametrize(
    "methodology_parameters,expected_pretest_weeks",
    [
        ({"pretest_weeks": 4}, 4),
        ({}, 0),
    ],
)
def test_experiment_design_sets_pretest_weeks_correctly(
    methodology_parameters, expected_pretest_weeks
):
  design = ExperimentDesign(
      experiment_type=ExperimentType.GO_DARK,
      primary_metric="revenue",
      experiment_budget=ExperimentBudget(
          value=-0.1,
          budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
      ),
      secondary_metrics=["conversions"],
      methodology="test_methodology",
      methodology_parameters=methodology_parameters,
      runtime_weeks=4,
      alpha=0.1,
      geo_eligibility=None,
  )
  assert design.pretest_weeks == expected_pretest_weeks


def test_metric_names_must_be_unique():
  with pytest.raises(ValueError):
    ExperimentDesign(
        experiment_type=ExperimentType.GO_DARK,
        primary_metric="revenue",
        experiment_budget=ExperimentBudget(
            value=-0.1,
            budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
        ),
        secondary_metrics=["revenue", "conversions"],
        methodology="test_methodology",
        runtime_weeks=4,
        alpha=0.1,
        geo_eligibility=None,
    )


def test_geoeligibilty_inflexible_2cell():
  eligibility = GeoEligibility(
      control={"g1", "g2"},
      treatment=[{"g3"}],  # 1 treatment arm, 2 cells total
      exclude={"g4"},
      all_geos={"g1", "g2", "g3", "g4", "g5"},
      flexible=False
  )
  data_frames = eligibility.data

  # check there is only 1 dataframe
  assert isinstance(data_frames, list)
  assert len(data_frames) == 1
  df = data_frames[0]

  assert isinstance(df, pd.DataFrame)
  assert df.index.name == "geo"
  assert df.columns.to_list() == ["control", "treatment", "exclude"]
  assert set(df.index) == {"g1", "g2", "g3", "g4", "g5"}

  # check values of specific geos
  assert df.loc["g1"].equals(
      pd.Series({"control": 1, "treatment": 0, "exclude": 0}, name="g1")
      )
  assert df.loc["g2"].equals(
      pd.Series({"control": 1, "treatment": 0, "exclude": 0}, name="g2")
      )
  assert df.loc["g3"].equals(
      pd.Series({"control": 0, "treatment": 1, "exclude": 0}, name="g3")
      )
  assert df.loc["g4"].equals(
      pd.Series({"control": 0, "treatment": 0, "exclude": 1}, name="g4")
      )
  # g5 is not explicitly defined and flexible=False -> should be all 0s
  assert df.loc["g5"].equals(
      pd.Series({"control": 0, "treatment": 0, "exclude": 0}, name="g5")
      )


def test_geoeligibility_flexible_2cell():
  eligibility = GeoEligibility(
      control={"g4", "g3"},  # out of order numbers
      treatment=[{"g2"}],
      exclude={"g5"},
      all_geos={"g1", "g2", "g3", "g4", "g5", "g6"},
      flexible=True
  )
  data_frames = eligibility.data

  # check there is only 1 dataframe
  assert isinstance(data_frames, list)
  assert len(data_frames) == 1
  df = data_frames[0]

  assert isinstance(df, pd.DataFrame)
  assert df.index.name == "geo"
  assert df.columns.to_list() == ["control", "treatment", "exclude"]
  assert set(df.index) == {"g1", "g2", "g3", "g4", "g5", "g6"}

  # check values of specific geos
  # g1 is not defined and flexible=True -> flexible control and treatment
  assert df.loc["g1"].equals(
      pd.Series({"control": 1, "treatment": 1, "exclude": 0}, name="g1")
      )
  assert df.loc["g2"].equals(
      pd.Series({"control": 0, "treatment": 1, "exclude": 0}, name="g2")
      )
  assert df.loc["g3"].equals(
      pd.Series({"control": 1, "treatment": 0, "exclude": 0}, name="g3")
      )
  assert df.loc["g4"].equals(
      pd.Series({"control": 1, "treatment": 0, "exclude": 0}, name="g4")
      )
  assert df.loc["g5"].equals(
      pd.Series({"control": 0, "treatment": 0, "exclude": 1}, name="g5")
      )
  # g6 is not defined and flexible=True -> flexible control and treatment
  assert df.loc["g6"].equals(
      pd.Series({"control": 1, "treatment": 1, "exclude": 0}, name="g6")
      )


def test_geoeligibility_multiarm_3cell():
  eligibililty = GeoEligibility(
      control={"c1"},
      treatment=[
          {"t1", "t3"},
          {"t2", "t3"}
      ],
      exclude={"e1"},
      all_geos={"c1", "t1", "t2", "t3", "e1", "u1"},
      flexible=False
  )
  data_frames = eligibililty.data

  # check there are 2 dataframes
  assert isinstance(data_frames, list)
  assert len(data_frames) == 2
  df_arm1 = data_frames[0]
  df_arm2 = data_frames[1]

  assert isinstance(df_arm1, pd.DataFrame)
  assert isinstance(df_arm2, pd.DataFrame)
  expected_geos = {"c1", "t1", "t2", "t3", "e1", "u1"}
  assert set(df_arm1.index) == expected_geos
  assert set(df_arm2.index) == expected_geos

  # check eligibility for t1 which is only eligible for treatment in arm1
  assert df_arm1.loc["t1"].equals(
      pd.Series({"control": 0, "treatment": 1, "exclude": 0}, name="t1")
      )
  assert df_arm2.loc["t1"].equals(
      pd.Series({"control": 0, "treatment": 0, "exclude": 0}, name="t1")
      )

  # check eligibility for t2 which is only eligible for treatment in arm2
  assert df_arm1.loc["t2"].equals(
      pd.Series({"control": 0, "treatment": 0, "exclude": 0}, name="t2")
      )
  assert df_arm2.loc["t2"].equals(
      pd.Series({"control": 0, "treatment": 1, "exclude": 0}, name="t2")
      )

  # check eligibility for t3 which is eligible for treatment in both arms
  assert df_arm1.loc["t3"].equals(
      pd.Series({"control": 0, "treatment": 1, "exclude": 0}, name="t3")
      )
  assert df_arm2.loc["t3"].equals(
      pd.Series({"control": 0, "treatment": 1, "exclude": 0}, name="t3")
      )

  # check eligibility for c1 which is only eligible for control
  assert df_arm1.loc["c1"].equals(
      pd.Series({"control": 1, "treatment": 0, "exclude": 0}, name="c1")
      )
  assert df_arm2.loc["c1"].equals(
      pd.Series({"control": 1, "treatment": 0, "exclude": 0}, name="c1")
      )

  # check eligibility for e1 which is only eligible for exclude
  assert df_arm1.loc["e1"].equals(
      pd.Series({"control": 0, "treatment": 0, "exclude": 1}, name="e1")
      )
  assert df_arm2.loc["e1"].equals(
      pd.Series({"control": 0, "treatment": 0, "exclude": 1}, name="e1")
      )

  # check eligibility for u1 which is not defined and flexible=False
  assert df_arm1.loc["u1"].equals(
      pd.Series({"control": 0, "treatment": 0, "exclude": 0}, name="u1")
      )
  assert df_arm2.loc["u1"].equals(
      pd.Series({"control": 0, "treatment": 0, "exclude": 0}, name="u1")
      )


def test_make_geo_assignment_array_returns_correct_array():
  assignment = GeoAssignment(
      treatment=[{"US", "UK"}, {"CA"}],
      control={"FR", "AU"},
      exclude={"DE", "JP"},
  ).make_geo_assignment_array(["US", "CA", "UK", "FR", "DE", "Other"])
  expected_assignment = np.array([1, 2, 1, 0, -1, -1])
  assert np.array_equal(assignment, expected_assignment)
