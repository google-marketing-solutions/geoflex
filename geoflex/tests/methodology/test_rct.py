"""Tests for the RCT methodology module."""

import geoflex.data
import geoflex.experiment_design
import geoflex.methodology.rct
import numpy as np
import pandas as pd
import pytest
import vizier.pyvizier as vz

RCT = geoflex.methodology.rct.RCT
ExperimentDesignConstraints = (
    geoflex.experiment_design.ExperimentDesignConstraints
)
ExperimentType = geoflex.experiment_design.ExperimentType
GeoAssignment = geoflex.experiment_design.GeoAssignment
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
ExperimentDesign = geoflex.experiment_design.ExperimentDesign

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
          "clicks": [
              8000.0,
              10000.0,
              12000.0,
              14000.0,
              17000.0,
              20000.0,
              100.0,
              1000.0,
          ],
      }),
      response_columns=["revenue", "clicks"],
  )


@pytest.mark.parametrize(
    "fixed_control_geos,fixed_treatment_geos,expected_is_eligible",
    [
        (["US", "UK"], ["CA", "AU"], False),
        (["US", "UK"], [], False),
        ([], ["CA", "AU"], False),
        ([], [], True),
    ],
)
def test_rct_is_eligible_for_constraints(
    fixed_control_geos, fixed_treatment_geos, expected_is_eligible
):
  constraints = ExperimentDesignConstraints(
      experiment_type=ExperimentType.GO_DARK,
      max_runtime_weeks=4,
      min_runtime_weeks=2,
      fixed_geos=GeoAssignment(
          control=fixed_control_geos, treatment=fixed_treatment_geos
      ),
  )
  assert RCT().is_eligible_for_constraints(constraints) == expected_is_eligible


def test_rct_add_parameters_to_search_space():
  problem_statement = vz.ProblemStatement()
  RCT().add_parameters_to_search_space(
      ExperimentDesignConstraints(
          experiment_type=ExperimentType.GO_DARK,
          max_runtime_weeks=4,
          min_runtime_weeks=2,
      ),
      problem_statement.search_space.root,
  )
  assert set(problem_statement.search_space.parameter_names) == set(
      ["treatment_propensity"]
  )


@pytest.mark.parametrize(
    "treatment_propensity,expected_treatment_geos",
    [(0.5, 2), (0.25, 1), (0.75, 3), (0.99, 3), (0.01, 1)],
)
def test_rct_assign_geos(
    performance_data, treatment_propensity, expected_treatment_geos
):
  rng = np.random.default_rng(seed=42)
  experiment_design = ExperimentDesign(
      primary_response_metric="revenue",
      methodology="RCT",
      runtime_weeks=4,
      alpha=0.1,
      fixed_geos=None,
      methodology_parameters={"treatment_propensity": treatment_propensity},
  )
  geo_assignment = RCT().assign_geos(experiment_design, performance_data, rng)
  assert len(geo_assignment.treatment) == expected_treatment_geos
  assert len(geo_assignment.control) == 4 - expected_treatment_geos
  assert set(geo_assignment.treatment + geo_assignment.control) == set(
      performance_data.geos
  )


def test_rct_analyze_experiment(performance_data):
  experiment_design = ExperimentDesign(
      primary_response_metric="revenue",
      methodology="RCT",
      runtime_weeks=4,
      alpha=0.1,
      fixed_geos=None,
      methodology_parameters={"treatment_propensity": 0.5},
  )
  experiment_design.geo_assignment = GeoAssignment(
      treatment=["US", "UK"], control=["CA", "AU"]
  )

  analysis_results = RCT().analyze_experiment(
      performance_data, experiment_design, "2024-01-01"
  )

  expected_results = pd.DataFrame({
      "metric": ["revenue", "clicks"],
      "point_estimate": [-400.0, 13950.0],
      "lower_bound": [-812.948321, 3905.762437],
      "upper_bound": [12.948321, 23994.237563],
      "point_estimate_relative": [-0.363636, 1.029520],
      "lower_bound_relative": [-0.645294, 0.239739],
      "upper_bound_relative": [0.014503, 2.470856],
      "p_value": [0.105573, 0.061079],
      "is_significant": [False, True],
  })
  pd.testing.assert_frame_equal(
      analysis_results, expected_results, check_like=True, atol=1e-6
  )
