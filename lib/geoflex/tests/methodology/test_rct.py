"""Tests for the RCT methodology module."""

import geoflex.data
import geoflex.experiment_design
import geoflex.exploration_spec
import geoflex.methodology.rct
import geoflex.metrics
import pandas as pd
import pytest

RCT = geoflex.methodology.rct.RCT
ExperimentDesignExplorationSpec = (
    geoflex.exploration_spec.ExperimentDesignExplorationSpec
)
ExperimentType = geoflex.experiment_design.ExperimentType
GeoAssignment = geoflex.experiment_design.GeoAssignment
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
ExperimentDesign = geoflex.experiment_design.ExperimentDesign
GeoEligibility = geoflex.experiment_design.GeoEligibility
ExperimentBudget = geoflex.experiment_design.ExperimentBudget
ExperimentBudgetType = geoflex.experiment_design.ExperimentBudgetType
CellVolumeConstraint = geoflex.experiment_design.CellVolumeConstraint
CellVolumeConstraintType = geoflex.experiment_design.CellVolumeConstraintType

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


@pytest.fixture(name="performance_data")
def performance_data_fixture():
  """Fixture for historical data."""
  return GeoPerformanceDataset(
      data=pd.DataFrame({
          "geo_id": ["US", "UK", "CA", "AU", "US", "UK", "CA", "AU"],
          "date": [
              "2024-01-01",
              "2024-01-01",
              "2024-01-01",
              "2024-01-01",
              "2024-01-02",
              "2024-01-02",
              "2024-01-02",
              "2024-01-02",
          ],
          "revenue": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
          "cost": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
          "conversions": [
              8000.0,
              10000.0,
              12000.0,
              14000.0,
              17000.0,
              20000.0,
              100.0,
              1000.0,
          ],
      })
  )


@pytest.mark.parametrize(
    "design,expected_is_eligible",
    [
        (
            ExperimentDesign(
                experiment_type=ExperimentType.GO_DARK,
                primary_metric="revenue",
                experiment_budget=ExperimentBudget(
                    value=-0.1,
                    budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
                ),
                methodology="RCT",
                runtime_weeks=4,
                geo_eligibility=GeoEligibility(
                    control=["US", "UK"], treatment=[{"CA", "AU"}]
                ),
            ),
            False,
        ),
        (
            ExperimentDesign(
                experiment_type=ExperimentType.GO_DARK,
                primary_metric="revenue",
                experiment_budget=ExperimentBudget(
                    value=-0.1,
                    budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
                ),
                methodology="RCT",
                runtime_weeks=4,
                geo_eligibility=GeoEligibility(control={"US", "UK"}),
            ),
            False,
        ),
        (
            ExperimentDesign(
                experiment_type=ExperimentType.GO_DARK,
                primary_metric="revenue",
                experiment_budget=ExperimentBudget(
                    value=-0.1,
                    budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
                ),
                methodology="RCT",
                runtime_weeks=4,
                geo_eligibility=GeoEligibility(treatment=[{"CA", "AU"}]),
            ),
            False,
        ),
        (
            ExperimentDesign(
                experiment_type=ExperimentType.GO_DARK,
                primary_metric="revenue",
                experiment_budget=ExperimentBudget(
                    value=-0.1,
                    budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
                ),
                methodology="RCT",
                runtime_weeks=4,
                geo_eligibility=None,
            ),
            True,
        ),
        (
            ExperimentDesign(
                experiment_type=ExperimentType.GO_DARK,
                primary_metric="revenue",
                experiment_budget=ExperimentBudget(
                    value=-0.1,
                    budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
                ),
                methodology="RCT",
                runtime_weeks=4,
                geo_eligibility=GeoEligibility(exclude={"US"}),
            ),
            True,
        ),
        (
            ExperimentDesign(
                experiment_type=ExperimentType.GO_DARK,
                primary_metric="revenue",
                experiment_budget=ExperimentBudget(
                    value=-0.1,
                    budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
                ),
                methodology="Something Else",
                runtime_weeks=4,
            ),
            False,
        ),
    ],
)
def test_rct_is_eligible_for_design(design, expected_is_eligible):
  assert RCT().is_eligible_for_design(design) == expected_is_eligible


def test_rct_has_expected_methodology_parameter_candidates():
  assert RCT().default_methodology_parameter_candidates == {
      "trimming_quantile": [0.0, 0.05]
  }


@pytest.mark.parametrize(
    "n_geos_per_group,exclude_geos,n_cells,expected_geo_counts",
    [
        ([2, 2], [], 2, [2, 2]),
        ([2, 1], ["UK"], 2, [2, 1]),
        (None, [], 2, [2, 2]),
        (None, ["UK"], 2, [2, 1]),
        ([1, 2, 1], [], 3, [1, 2, 1]),
        ([1, 1, 1], ["UK"], 3, [1, 1, 1]),
        (None, [], 3, [2, 1, 1]),
        (None, ["UK"], 3, [1, 1, 1]),
    ],
)  # "US", "UK", "CA", "AU"
def test_rct_assign_geos(
    performance_data,
    n_geos_per_group,
    exclude_geos,
    n_cells,
    expected_geo_counts,
):
  if n_geos_per_group is None:
    cell_volume_constraint = None
  else:
    cell_volume_constraint = CellVolumeConstraint(
        values=n_geos_per_group,
        constraint_type=CellVolumeConstraintType.NUMBER_OF_GEOS,
    )
  experiment_design = ExperimentDesign(
      experiment_type=ExperimentType.GO_DARK,
      primary_metric="revenue",
      experiment_budget=ExperimentBudget(
          value=-0.1,
          budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
      ),
      methodology="RCT",
      runtime_weeks=4,
      n_cells=n_cells,
      alpha=0.1,
      geo_eligibility=GeoEligibility(exclude=set(exclude_geos)),
      cell_volume_constraint=cell_volume_constraint,
  )
  geo_assignment = RCT().assign_geos(experiment_design, performance_data)
  geo_counts = [len(geo_assignment.control)] + list(
      map(len, geo_assignment.treatment)
  )
  if n_geos_per_group is None:
    # When n_geos_per_group is not specified, the order is randomized.
    # Sorting for the test to be deterministic.
    assert sorted(geo_counts) == sorted(expected_geo_counts)
  else:
    assert geo_counts == expected_geo_counts

  eligible_geos = set(performance_data.geos) - set(exclude_geos)
  used_geos = set(geo_assignment.control) | set(
      sum(map(list, geo_assignment.treatment), [])
  )
  assert used_geos == eligible_geos


def test_rct_analyze_experiment(performance_data):
  experiment_design = ExperimentDesign(
      experiment_type=ExperimentType.GO_DARK,
      primary_metric=geoflex.metrics.ROAS(),
      experiment_budget=ExperimentBudget(
          value=-0.1,
          budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
      ),
      secondary_metrics=[geoflex.metrics.CPA(), "revenue", "conversions"],
      methodology="RCT",
      runtime_weeks=4,
      alpha=0.1,
      geo_eligibility=None,
  )
  experiment_design.geo_assignment = GeoAssignment(
      treatment=[["US", "UK"]], control=["CA", "AU"]
  )

  analysis_results = RCT().analyze_experiment(
      performance_data, experiment_design, "2024-01-01"
  )

  expected_results = pd.DataFrame({
      "cell": [1, 1, 1, 1],
      "metric": ["ROAS", "CPA", "revenue", "conversions"],
      "is_primary_metric": [True, False, False, False],
      "point_estimate": [10.0, -0.002867, -400.0, 13950.0],
      "lower_bound": [-0.323708, -0.010241, -812.948321, 3905.762437],
      "upper_bound": [20.323708, -0.001667, 12.948321, 23994.237563],
      "point_estimate_relative": [pd.NA, pd.NA, -0.363636, 1.029520],
      "lower_bound_relative": [pd.NA, pd.NA, -0.645294, 0.239739],
      "upper_bound_relative": [pd.NA, pd.NA, 0.014503, 2.470856],
      "p_value": [0.105573, 0.061079, 0.105573, 0.061079],
      "is_significant": [False, True, False, True],
  })
  pd.testing.assert_frame_equal(
      analysis_results, expected_results, check_like=True, atol=1e-6
  )
