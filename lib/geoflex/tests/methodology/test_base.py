"""Tests for the base methodology module."""

from typing import Any
import geoflex.data
import geoflex.experiment_design
import geoflex.exploration_spec
import geoflex.methodology._base
import geoflex.metrics
import numpy as np
import optuna as op
import pandas as pd
import pytest


Methodology = geoflex.methodology._base.Methodology  # pylint: disable=protected-access
ExperimentType = geoflex.experiment_design.ExperimentType
GeoAssignment = geoflex.experiment_design.GeoAssignment
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
ExperimentDesign = geoflex.experiment_design.ExperimentDesign
ExperimentDesignExplorationSpec = (
    geoflex.exploration_spec.ExperimentDesignExplorationSpec
)

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


@pytest.fixture(name="historical_data")
def mock_historical_data_fixture():
  """Fixture for a mock historical data."""
  rng = np.random.default_rng(seed=42)
  data = pd.DataFrame({
      "geo_id": [f"geo_{i}" for i in range(20) for _ in range(100)],  # pylint: disable=g-complex-comprehension
      "date": pd.date_range(start="2024-01-01", periods=100).tolist() * 20,
      "revenue": rng.random(size=2000),
      "cost": rng.random(size=2000),
      "conversions": rng.random(size=2000),
  })
  data["date"] = data["date"].dt.strftime("%Y-%m-%d")

  return GeoPerformanceDataset(data=data)


@pytest.fixture(name="default_experiment_design")
def mock_experiment_design_fixture():
  """Fixture for a mock design spec."""
  return ExperimentDesign(
      experiment_type=ExperimentType.AB_TEST,
      primary_metric="revenue",
      methodology="MockMethodology",
      runtime_weeks=2,
      n_cells=3,
  )


@pytest.fixture(name="MockMethodology")
def mock_test_methodology_fixture():
  """Fixture for a mock test methodology. The methods don't do anything."""

  class MockMethodology(Methodology):
    """Mock methodology for testing.

    All the methods are no-ops, each test needs to define the methods it needs.
    """

    def _methodology_assign_geos(
        self,
        experiment_design: ExperimentDesign,
        historical_data: GeoPerformanceDataset,
    ) -> GeoAssignment:
      return GeoAssignment()

    def is_eligible_for_design(self, design: ExperimentDesign) -> bool:
      # Not used in this test
      return True

    def suggest_methodology_parameters(
        self,
        design_spec: ExperimentDesignExplorationSpec,
        trial: op.Trial,
    ) -> dict[str, Any]:
      # Not used in this test
      return {}

    def analyze_experiment(
        self,
        runtime_data: GeoPerformanceDataset,
        experiment_design: ExperimentDesign,
        experiment_start_date: str,
    ) -> pd.DataFrame:
      # Not used in this test
      return pd.DataFrame()

  return MockMethodology


def test_methodology_assign_geos_puts_missing_geos_into_exclude_list(
    historical_data, default_experiment_design, MockMethodology
):
  class AssignGeosMockMethodology(MockMethodology):

    def _methodology_assign_geos(
        self,
        experiment_design: ExperimentDesign,
        historical_data: GeoPerformanceDataset,
    ) -> GeoAssignment:
      del experiment_design, historical_data

      return GeoAssignment(
          treatment=[{"geo_2", "geo_3"}, {"geo_4", "geo_5"}],
          control={"geo_6", "geo_7"},
          exclude={"geo_0", "geo_1"},
      )

  geo_assignment = AssignGeosMockMethodology().assign_geos(
      default_experiment_design, historical_data
  )

  assert {"geo_0", "geo_1"}.issubset(geo_assignment.exclude)
  assert geo_assignment.treatment == [{"geo_2", "geo_3"}, {"geo_4", "geo_5"}]
  assert geo_assignment.control == {"geo_6", "geo_7"}

  all_assigned_geos = set().union(
      geo_assignment.control, geo_assignment.exclude, *geo_assignment.treatment
  )
  assert all_assigned_geos == set(historical_data.geos)


def test_methodology_assign_geos_raises_error_if_no_control_geos(
    historical_data, default_experiment_design, MockMethodology
):
  class AssignGeosMockMethodology(MockMethodology):

    def _methodology_assign_geos(
        self,
        experiment_design: ExperimentDesign,
        historical_data: GeoPerformanceDataset,
    ) -> GeoAssignment:
      del experiment_design, historical_data

      return GeoAssignment(
          treatment=[{"geo_2", "geo_3"}, {"geo_4", "geo_5"}],
          control=set(),
          exclude={"geo_0", "geo_1"},
      )

  with pytest.raises(ValueError):
    AssignGeosMockMethodology().assign_geos(
        default_experiment_design,
        historical_data,
    )


def test_methodology_assign_geos_raises_error_if_no_treatment_geos(
    historical_data, default_experiment_design, MockMethodology
):
  class AssignGeosMockMethodology(MockMethodology):

    def _methodology_assign_geos(
        self,
        experiment_design: ExperimentDesign,
        historical_data: GeoPerformanceDataset,
    ) -> GeoAssignment:
      del experiment_design, historical_data

      return GeoAssignment(
          treatment=[set(), {"geo_4", "geo_5"}],
          control={"geo_6", "geo_7"},
          exclude={"geo_0", "geo_1"},
      )

  with pytest.raises(ValueError):
    AssignGeosMockMethodology().assign_geos(
        default_experiment_design,
        historical_data,
    )


def test_methodology_assign_geos_raises_error_if_too_few_treatment_groups(
    historical_data, default_experiment_design, MockMethodology
):
  class AssignGeosMockMethodology(MockMethodology):

    def _methodology_assign_geos(
        self,
        experiment_design: ExperimentDesign,
        historical_data: GeoPerformanceDataset,
    ) -> GeoAssignment:
      del experiment_design, historical_data

      return GeoAssignment(
          treatment=[{"geo_2", "geo_3"}],
          control={"geo_6", "geo_7"},
          exclude={"geo_0", "geo_1"},
      )

  with pytest.raises(ValueError):
    AssignGeosMockMethodology().assign_geos(
        default_experiment_design,
        historical_data,
    )


def test_methodology_assign_geos_raises_error_if_too_many_treatment_groups(
    historical_data, default_experiment_design, MockMethodology
):
  class AssignGeosMockMethodology(MockMethodology):

    def _methodology_assign_geos(
        self,
        experiment_design: ExperimentDesign,
        historical_data: GeoPerformanceDataset,
    ) -> GeoAssignment:
      del experiment_design, historical_data

      return GeoAssignment(
          treatment=[
              {"geo_2", "geo_3"},
              {"geo_4", "geo_5"},
              {"geo_6", "geo_7"},
          ],
          control={"geo_6", "geo_7"},
          exclude={"geo_0", "geo_1"},
      )

  with pytest.raises(ValueError):
    AssignGeosMockMethodology().assign_geos(
        default_experiment_design,
        historical_data,
    )
