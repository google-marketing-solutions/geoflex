"""TBRMM specific tests to be run on top of the standard tests.

Focus is on ensuring the TBRMM wrapper is working. The original library is
tested in its own unit tests and the wrapper is dependent on the original
library working correctly.
"""

import logging

import geoflex.data
import geoflex.experiment_design
import geoflex.exploration_spec
import geoflex.methodology
import geoflex.methodology.tbrmm
import geoflex.metrics
import numpy as np
import pandas as pd
import pytest


TBRMM = geoflex.methodology.tbrmm.TBRMM
ExperimentDesignExplorationSpec = (
    geoflex.exploration_spec.ExperimentDesignExplorationSpec
)
GeoAssignment = geoflex.experiment_design.GeoAssignment
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
ExperimentDesign = geoflex.experiment_design.ExperimentDesign
GeoEligibility = geoflex.experiment_design.GeoEligibility
ExperimentBudget = geoflex.experiment_design.ExperimentBudget
ExperimentBudgetType = geoflex.experiment_design.ExperimentBudgetType

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


@pytest.fixture(name="performance_data")
def performance_data_fixture():
  """Fixture for historical data suitable for TBRMM."""
  rng = np.random.default_rng(seed=42)
  data = pd.DataFrame({
      "geo_id": np.repeat([f"geo_{i}" for i in range(20)], 100),
      "date": pd.date_range(start="2024-01-01", periods=100).tolist() * 20,
      "revenue": rng.random(size=2000) + 5,
      "cost": rng.random(size=2000) + 5,
      "conversions": rng.random(size=2000) + 5,
  })
  data["date"] = data["date"].dt.strftime("%Y-%m-%d")
  return GeoPerformanceDataset(data=data)


@pytest.fixture(name="few_geos_performance_data")
def few_geos_performance_data_fixture():
  """Fixture for historical data with few geos for TBRMM exhaustive search."""
  rng = np.random.default_rng(seed=42)
  # Only 6 geos, to test the exhaustive search path in TBRMM
  data = pd.DataFrame({
      "geo_id": np.repeat([f"geo_{i}" for i in range(6)], 100),
      "date": pd.date_range(start="2024-01-01", periods=100).tolist() * 6,
      "revenue": rng.random(size=600) + 5,
      "cost": rng.random(size=600) + 5,
      "conversions": rng.random(size=600) + 5,
  })
  data["date"] = data["date"].dt.strftime("%Y-%m-%d")
  return GeoPerformanceDataset(data=data)


@pytest.mark.parametrize(
    "design, expected_is_eligible",
    [
        (
            ExperimentDesign(
                primary_metric="revenue",
                methodology="TBRMM",
                runtime_weeks=2,
                n_cells=2,
                methodology_parameters={"pretest_weeks": 4},
            ),
            True,
        ),
        (
            ExperimentDesign(
                primary_metric="revenue",
                methodology="TBRMM",
                runtime_weeks=2,
                n_cells=3,  # n_cells must be 2
                methodology_parameters={"pretest_weeks": 4},
            ),
            False,
        ),
        (
            ExperimentDesign(
                primary_metric="revenue",
                methodology="TBRMM",
                runtime_weeks=2,
                n_cells=2,
                methodology_parameters={},
            ),
            True,
        ),
        (
            ExperimentDesign(
                primary_metric="revenue",
                methodology="TBRMM",
                runtime_weeks=2,
                n_cells=2,
                methodology_parameters={"pretest_weeks": 4},
                geo_eligibility=GeoEligibility(
                    control={"geo_0"}, treatment=[{"geo_1"}]
                ),
            ),
            True,  # tbrmm should be able to support geo constraints
        ),
    ],
)


def test_tbrmm_is_eligible_for_design(
    design, expected_is_eligible, performance_data
):
  assert (
      TBRMM().is_eligible_for_design_and_data(design, performance_data)
      == expected_is_eligible
  )


@pytest.mark.parametrize(
    "design_params, expected_geo_counts, should_raise_error",
    [
        (
            {
                "methodology_parameters": {
                    "treatment_geos_range": (1, 2),
                }
            },
            {"max_treatment": 2},
            False,
        ),
        (
            {
                "methodology_parameters": {
                    "min_corr": 0.999,  # Unattainable correlation
                }
            },
            None,
            True,
        ),
        (
            {
                "methodology_parameters": {
                    "rho_max": 0.001,  # Unattainable autocorrelation limit
                }
            },
            None,
            True,
        ),
        (
            {
                "methodology_parameters": {
                    "control_geos_range": (1, 5),
                }
            },
            {"max_control": 5},
            False,
        ),
    ],
)


def test_tbrmm_assign_geos_with_constraints(
    design_params,
    expected_geo_counts,
    should_raise_error,
    performance_data,
):
  """Tests that GeoFleX constraints are correctly passed to the original lib."""
  base_design_dict = {
      "primary_metric": "revenue",
      "methodology": "TBRMM",
      "runtime_weeks": 2,
      "n_cells": 2,
      "methodology_parameters": {"pretest_weeks": 4},
  }
  # Deep merge the design_params into the base_design_dict
  if "methodology_parameters" in design_params:
    base_design_dict["methodology_parameters"].update(
        design_params.pop("methodology_parameters")
    )
  base_design_dict.update(design_params)

  experiment_design = ExperimentDesign(**base_design_dict)

  if should_raise_error:
    with pytest.raises(
        (ValueError, RuntimeError), match="returned no suitable designs"
    ):
      TBRMM().assign_geos(experiment_design, performance_data)
  else:
    geo_assignment, _ = TBRMM().assign_geos(
        experiment_design, performance_data
    )
    assert isinstance(geo_assignment, GeoAssignment)
    if expected_geo_counts:
      if "max_treatment" in expected_geo_counts:
        assert (
            len(geo_assignment.treatment[0])
            <= expected_geo_counts["max_treatment"]
        )
      if "max_control" in expected_geo_counts:
        assert len(geo_assignment.control) <= expected_geo_counts["max_control"]


def test_tbrmm_assign_geos_with_geo_eligibility_constraints(performance_data):
  geo_eligibility = GeoEligibility(
      control={"geo_0", "geo_1"},
      treatment=[{"geo_1"}],
      # 'geo_0' can only be in control.
      # 'geo_1' can be in control or treatment, but not excluded.
      # other geos should all be flexible
  )
  experiment_design = ExperimentDesign(
      primary_metric="revenue",
      methodology="TBRMM",
      runtime_weeks=2,
      n_cells=2,
      methodology_parameters={"pretest_weeks": 4},
      geo_eligibility=geo_eligibility,
  )

  geo_assignment, _ = TBRMM().assign_geos(experiment_design, performance_data)

  assert "geo_0" in geo_assignment.control
  assert "geo_0" not in geo_assignment.treatment[0]
  assert "geo_0" not in geo_assignment.exclude
  assert "geo_1" not in geo_assignment.exclude


def test_tbrmm_assign_geos_with_inflexible_geo_eligibility(performance_data):
  # Only geo_0, geo_1, geo_2, geo_3 are eligible for anything.
  # All other geos in performance_data (up to geo_19) should be excluded.
  geo_eligibility = GeoEligibility(
      control={"geo_0", "geo_1"},
      treatment=[{"geo_2", "geo_3"}],
      flexible=False,
  )
  experiment_design = ExperimentDesign(
      primary_metric="revenue",
      methodology="TBRMM",
      runtime_weeks=2,
      n_cells=2,
      methodology_parameters={"pretest_weeks": 4},
      geo_eligibility=geo_eligibility,
  )

  geo_assignment, _ = TBRMM().assign_geos(experiment_design, performance_data)

  assigned_geos = geo_assignment.control | geo_assignment.treatment[0]
  explicitly_eligible_geos = {"geo_0", "geo_1", "geo_2", "geo_3"}

  # Check that only from the explicitly eligible geos were assigned.
  assert assigned_geos.issubset(explicitly_eligible_geos)

  # Check that a geo not in the eligibility list was excluded.
  assert "geo_5" in geo_assignment.exclude
  assert "geo_5" not in assigned_geos


def test_tbrmm_assign_geos_fails_without_pretest_weeks(performance_data):
  experiment_design = ExperimentDesign(
      primary_metric="revenue",
      methodology="TBRMM",
      runtime_weeks=2,
      n_cells=2,
      methodology_parameters={},  # Missing pretest_weeks
  )
  with pytest.raises(ValueError, match="Failed to adapt inputs"):
    TBRMM().assign_geos(experiment_design, performance_data)


@pytest.mark.parametrize(
    "performance_data_fixture_name",
    ["performance_data", "few_geos_performance_data"],
)
def test_tbrmm_assign_geos(performance_data_fixture_name, request):
  """Tests geo assignment for both greedy and exhaustive search paths."""
  performance_data = request.getfixturevalue(performance_data_fixture_name)
  experiment_design = ExperimentDesign(
      primary_metric="revenue",
      methodology="TBRMM",
      runtime_weeks=2,
      n_cells=2,
      methodology_parameters={"pretest_weeks": 4},
  )
  geo_assignment, _ = TBRMM().assign_geos(experiment_design, performance_data)
  assert isinstance(geo_assignment, GeoAssignment)
  assert geo_assignment.control
  assert len(geo_assignment.treatment) == 1
  assert geo_assignment.treatment[0]
  assert not (geo_assignment.control & geo_assignment.treatment[0])


def test_tbrmm_greedy_search_with_infeasible_budget_succeeds(
    performance_data, caplog
):
  # budget is effectively zero so it is not possible to find a suitable design
  # greedy search should still return a design but also a warning
  experiment_design_with_infeasible_budget = ExperimentDesign(
      primary_metric="revenue",
      methodology="TBRMM",
      runtime_weeks=2,
      n_cells=2,
      methodology_parameters={"pretest_weeks": 4, "cost_column": "cost"},
      experiment_budget=ExperimentBudget(
          budget_type=ExperimentBudgetType.TOTAL_BUDGET,
          value=[0.00001],
      ),
  )

  tbrmm_methodology = TBRMM()
  with caplog.at_level(logging.WARNING):
    assigned_geos, _ = tbrmm_methodology.assign_geos(
        experiment_design=experiment_design_with_infeasible_budget,
        historical_data=performance_data,
    )

  # Check that a design was still returned
  assert isinstance(assigned_geos, GeoAssignment)
  assert assigned_geos.control
  assert assigned_geos.treatment[0]

  # Check that a warning about the budget was logged
  assert "search is over budget" in caplog.text


def test_tbrmm_analyze_experiment(performance_data):
  experiment_design = ExperimentDesign(
      primary_metric="revenue",
      secondary_metrics=["conversions"],
      methodology="TBRMM",
      runtime_weeks=2,
      n_cells=2,
      methodology_parameters={"pretest_weeks": 4},
  )
  # A plausible geo assignment for the analysis step.
  geos = list(performance_data.geos)
  experiment_design.geo_assignment = GeoAssignment(
      control=set(geos[:10]), treatment=[set(geos[10:])]
  )

  analysis_results, _ = TBRMM().analyze_experiment(
      runtime_data=performance_data,
      experiment_design=experiment_design,
      experiment_start_date="2024-03-01",
      experiment_end_date="2024-03-15",
  )

  assert isinstance(analysis_results, pd.DataFrame)
  assert not analysis_results.empty
