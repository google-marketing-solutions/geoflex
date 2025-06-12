"""Tests for the TestingMethodology methodology module."""

import geoflex.data
import geoflex.experiment_design
import geoflex.exploration_spec
import geoflex.methodology.testing_methodology
import geoflex.metrics
import numpy as np
import pandas as pd
import pytest

TestingMethodology = geoflex.methodology.testing_methodology.TestingMethodology
ExperimentDesignExplorationSpec = (
    geoflex.exploration_spec.ExperimentDesignExplorationSpec
)
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


@pytest.fixture(name="big_performance_data")
def big_performance_data_fixture():
  """Fixture for a mock historical data with lots of geos."""
  rng = np.random.default_rng(seed=42)
  data = pd.DataFrame({
      "geo_id": [f"geo_{i}" for i in range(20) for _ in range(100)],  # pylint: disable=g-complex-comprehension
      "date": pd.date_range(start="2024-01-01", periods=100).tolist() * 20,
      "revenue": rng.random(size=2000),
      "cost": rng.random(size=2000),
      "conversions": rng.random(size=2000),
  })
  data["date"] = data["date"].dt.strftime("%Y-%m-%d")

  return geoflex.data.GeoPerformanceDataset(data=data)


@pytest.mark.parametrize(
    "design,expected_is_eligible",
    [
        (
            ExperimentDesign(
                primary_metric="revenue",
                experiment_budget=ExperimentBudget(
                    value=-0.1,
                    budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
                ),
                methodology="TestingMethodology",
                runtime_weeks=4,
                geo_eligibility=GeoEligibility(
                    control=["US", "UK"], treatment=[{"CA", "AU"}]
                ),
            ),
            False,
        ),
        (
            ExperimentDesign(
                primary_metric="revenue",
                experiment_budget=ExperimentBudget(
                    value=-0.1,
                    budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
                ),
                methodology="TestingMethodology",
                runtime_weeks=4,
                geo_eligibility=GeoEligibility(control={"US", "UK"}),
            ),
            False,
        ),
        (
            ExperimentDesign(
                primary_metric="revenue",
                experiment_budget=ExperimentBudget(
                    value=-0.1,
                    budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
                ),
                methodology="TestingMethodology",
                runtime_weeks=4,
                geo_eligibility=GeoEligibility(treatment=[{"CA", "AU"}]),
            ),
            False,
        ),
        (
            ExperimentDesign(
                primary_metric="revenue",
                experiment_budget=ExperimentBudget(
                    value=-0.1,
                    budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
                ),
                methodology="TestingMethodology",
                runtime_weeks=4,
                geo_eligibility=None,
            ),
            True,
        ),
        (
            ExperimentDesign(
                primary_metric="revenue",
                experiment_budget=ExperimentBudget(
                    value=-0.1,
                    budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
                ),
                methodology="TestingMethodology",
                runtime_weeks=4,
                geo_eligibility=GeoEligibility(exclude={"US"}),
            ),
            True,
        ),
        (
            ExperimentDesign(
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
def test_testing_methodology_is_eligible_for_design(
    design, expected_is_eligible, big_performance_data
):
  assert (
      TestingMethodology().is_eligible_for_design_and_data(
          design, big_performance_data
      )
      == expected_is_eligible
  )


def test_testing_methodology_has_expected_methodology_parameter_candidates():
  assert TestingMethodology().default_methodology_parameter_candidates == {
      "mock_parameter": [1, 2]
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
def test_testing_methodology_assign_geos(
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
        constraint_type=CellVolumeConstraintType.MAX_GEOS,
    )
  experiment_design = ExperimentDesign(
      primary_metric="revenue",
      experiment_budget=ExperimentBudget(
          value=-0.1,
          budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
      ),
      methodology="TestingMethodology",
      runtime_weeks=4,
      n_cells=n_cells,
      alpha=0.1,
      geo_eligibility=GeoEligibility(exclude=set(exclude_geos)),
      cell_volume_constraint=cell_volume_constraint,
  )
  geo_assignment, _ = TestingMethodology().assign_geos(
      experiment_design, performance_data
  )
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


def test_testing_methodology_analyze_experiment(performance_data):
  experiment_design = ExperimentDesign(
      primary_metric=geoflex.metrics.iROAS(),
      experiment_budget=ExperimentBudget(
          value=-0.1,
          budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
      ),
      secondary_metrics=[geoflex.metrics.CPiA(), "revenue", "conversions"],
      methodology="TestingMethodology",
      runtime_weeks=4,
      alpha=0.1,
      geo_eligibility=None,
  )
  experiment_design.geo_assignment = GeoAssignment(
      treatment=[["US", "UK"]], control=["CA", "AU"]
  )

  analysis_results, _ = TestingMethodology().analyze_experiment(
      performance_data, experiment_design, "2024-01-01"
  )

  expected_results = pd.DataFrame({
      "cell": [1, 1, 1, 1],
      "metric": ["iROAS", "CPiA", "revenue", "conversions"],
      "is_primary_metric": [True, False, False, False],
      "point_estimate": [
          0.1257302210933933,
          -0.1321048632913019,
          0.6404226504432821,
          0.10490011715303971,
      ],
      "lower_bound": [
          -1.5191234058580796,
          -1.7769584902427749,
          -1.004430976508191,
          -1.5399535097984332,
      ],
      "upper_bound": [
          1.7705838480448655,
          1.5127487636601704,
          2.2852762773947544,
          1.749753744104512,
      ],
      "point_estimate_relative": [
          pd.NA,
          pd.NA,
          0.6404226504432821,
          0.10490011715303971,
      ],
      "lower_bound_relative": [
          pd.NA,
          pd.NA,
          -1.004430976508191,
          -1.5399535097984332,
      ],
      "upper_bound_relative": [
          pd.NA,
          pd.NA,
          2.2852762773947544,
          1.749753744104512,
      ],
      "p_value": [
          0.8999454787169219,
          0.8949013492784883,
          0.521897861133398,
          0.9164550660076147,
      ],
      "is_significant": [False, False, False, False],
  })

  pd.testing.assert_frame_equal(
      analysis_results, expected_results, check_like=True, atol=1e-6
  )


@pytest.mark.parametrize(
    "alternative_hypothesis",
    ["two-sided", "greater", "less"],
)
def test_testing_methodology_analyze_experiment_works_for_any_alternative_hypothesis(
    performance_data, alternative_hypothesis
):
  experiment_design = ExperimentDesign(
      primary_metric=geoflex.metrics.iROAS(),
      experiment_budget=ExperimentBudget(
          value=-0.1,
          budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
      ),
      secondary_metrics=[geoflex.metrics.CPiA(), "revenue", "conversions"],
      methodology="TestingMethodology",
      runtime_weeks=4,
      alpha=0.1,
      geo_eligibility=None,
      alternative_hypothesis=alternative_hypothesis,
  )
  experiment_design.geo_assignment = GeoAssignment(
      treatment=[["US", "UK"]], control=["CA", "AU"]
  )

  analysis_results, _ = TestingMethodology().analyze_experiment(
      performance_data, experiment_design, "2024-01-01"
  )
  assert isinstance(analysis_results, pd.DataFrame)


@pytest.mark.parametrize(
    "methodology, expected_is_pseudo_experiment",
    [
        ("TestingMethodology", False),
        ("PseudoExperimentTestingMethodology", True),
    ],
)
def test_is_pseudo_experiment(methodology, expected_is_pseudo_experiment):
  design = ExperimentDesign(
      methodology=methodology, primary_metric="revenue", runtime_weeks=4
  )

  assert (
      geoflex.methodology.is_pseudo_experiment(design)
      == expected_is_pseudo_experiment
  )
