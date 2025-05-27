"""Tests for the base methodology module."""

import datetime as dt
from typing import Any
import geoflex.data
import geoflex.experiment_design
import geoflex.exploration_spec
import geoflex.methodology._base
import geoflex.metrics
import numpy as np
import pandas as pd
import pytest


Methodology = geoflex.methodology._base.Methodology  # pylint: disable=protected-access
GeoAssignment = geoflex.experiment_design.GeoAssignment
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
ExperimentDesign = geoflex.experiment_design.ExperimentDesign
ExperimentDesignExplorationSpec = (
    geoflex.exploration_spec.ExperimentDesignExplorationSpec
)
list_methodologies = geoflex.methodology.list_methodologies
register_methodology = geoflex.methodology.register_methodology
ExperimentBudget = geoflex.experiment_design.ExperimentBudget
ExperimentBudgetType = geoflex.experiment_design.ExperimentBudgetType
CellVolumeConstraintType = geoflex.experiment_design.CellVolumeConstraintType
CellVolumeConstraint = geoflex.experiment_design.CellVolumeConstraint

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
      return GeoAssignment(), {"intermediate_data": "from_assign_geos"}

    def is_eligible_for_design(self, design: ExperimentDesign) -> bool:
      # Not used in this test
      return True

    def _methodology_analyze_experiment(
        self,
        runtime_data: GeoPerformanceDataset,
        experiment_design: ExperimentDesign,
        experiment_start_date: pd.Timestamp,
        experiment_end_date: pd.Timestamp,
        pretest_period_end_date: pd.Timestamp,
    ) -> pd.DataFrame:
      # Not used in this test
      return pd.DataFrame(), {"intermediate_data": "from_analyze_experiment"}

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

      return (
          GeoAssignment(
              treatment=[{"geo_2", "geo_3"}, {"geo_4", "geo_5"}],
              control={"geo_6", "geo_7"},
              exclude={"geo_0", "geo_1"},
          ),
          {},
      )

  geo_assignment, _ = AssignGeosMockMethodology().assign_geos(
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

      return (
          GeoAssignment(
              treatment=[{"geo_2", "geo_3"}, {"geo_4", "geo_5"}],
              control=set(),
              exclude={"geo_0", "geo_1"},
          ),
          {},
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

      return (
          GeoAssignment(
              treatment=[set(), {"geo_4", "geo_5"}],
              control={"geo_6", "geo_7"},
              exclude={"geo_0", "geo_1"},
          ),
          {},
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

      return (
          GeoAssignment(
              treatment=[{"geo_2", "geo_3"}],
              control={"geo_6", "geo_7"},
              exclude={"geo_0", "geo_1"},
          ),
          {},
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

      return (
          GeoAssignment(
              treatment=[
                  {"geo_2", "geo_3"},
                  {"geo_4", "geo_5"},
                  {"geo_6", "geo_7"},
              ],
              control={"geo_6", "geo_7"},
              exclude={"geo_0", "geo_1"},
          ),
          {},
      )

  with pytest.raises(ValueError):
    AssignGeosMockMethodology().assign_geos(
        default_experiment_design,
        historical_data,
    )


@pytest.mark.parametrize(
    "missing_column",
    [
        "metric",
        "cell",
        "point_estimate",
        "lower_bound",
        "upper_bound",
        "point_estimate_relative",
        "lower_bound_relative",
        "upper_bound_relative",
    ],
)
def test_methodology_analyze_experiment_raises_error_if_missing_required_columns(
    historical_data, default_experiment_design, MockMethodology, missing_column
):
  class AnalyzeExperimentMockMethodology(MockMethodology):
    """Mock methodology for testing."""

    def _methodology_analyze_experiment(
        self,
        runtime_data: GeoPerformanceDataset,
        experiment_design: ExperimentDesign,
        experiment_start_date: pd.Timestamp,
        experiment_end_date: pd.Timestamp,
        pretest_period_end_date: pd.Timestamp,
    ) -> pd.DataFrame:
      del (
          runtime_data,
          experiment_design,
          experiment_start_date,
          experiment_end_date,
          pretest_period_end_date,
      )

      return (
          pd.DataFrame({
              "metric": ["revenue"],
              "cell": [1],
              "point_estimate": [1.0],
              "lower_bound": [0.5],
              "upper_bound": [1.5],
              "point_estimate_relative": [0.1],
              "lower_bound_relative": [0.05],
              "upper_bound_relative": [0.15],
              "p_value": [0.01],
          }).drop([missing_column], axis=1),
          {},
      )

  with pytest.raises(ValueError):
    AnalyzeExperimentMockMethodology().analyze_experiment(
        historical_data, default_experiment_design, "2024-01-01"
    )


def test_methodology_analyze_experiment_drops_extra_columns(
    historical_data, default_experiment_design, MockMethodology
):
  class AnalyzeExperimentMockMethodology(MockMethodology):
    """Mock methodology for testing."""

    def _methodology_analyze_experiment(
        self,
        runtime_data: GeoPerformanceDataset,
        experiment_design: ExperimentDesign,
        experiment_start_date: pd.Timestamp,
        experiment_end_date: pd.Timestamp,
        pretest_period_end_date: pd.Timestamp,
    ) -> pd.DataFrame:
      del (
          runtime_data,
          experiment_design,
          experiment_start_date,
          experiment_end_date,
          pretest_period_end_date,
      )

      return (
          pd.DataFrame({
              "metric": ["revenue"],
              "cell": [1],
              "point_estimate": [1.0],
              "lower_bound": [0.5],
              "upper_bound": [1.5],
              "point_estimate_relative": [0.1],
              "lower_bound_relative": [0.05],
              "upper_bound_relative": [0.15],
              "p_value": [0.01],
              "extra_column": [1],
          }),
          {},
      )

  results, _ = AnalyzeExperimentMockMethodology().analyze_experiment(
      historical_data, default_experiment_design, "2024-01-01"
  )
  assert "extra_column" not in results.columns


def test_methodology_analyze_experiment_raises_error_if_missing_metrics(
    historical_data, default_experiment_design, MockMethodology
):
  class AnalyzeExperimentMockMethodology(MockMethodology):
    """Mock methodology for testing."""

    def _methodology_analyze_experiment(
        self,
        runtime_data: GeoPerformanceDataset,
        experiment_design: ExperimentDesign,
        experiment_start_date: pd.Timestamp,
        experiment_end_date: pd.Timestamp,
        pretest_period_end_date: pd.Timestamp,
    ) -> pd.DataFrame:
      del (
          runtime_data,
          experiment_design,
          experiment_start_date,
          experiment_end_date,
          pretest_period_end_date,
      )

      return (
          pd.DataFrame({
              "metric": ["revenue"],
              "cell": [1],
              "point_estimate": [1.0],
              "lower_bound": [0.5],
              "upper_bound": [1.5],
              "point_estimate_relative": [0.1],
              "lower_bound_relative": [0.05],
              "upper_bound_relative": [0.15],
              "p_value": [0.01],
          }),
          {},
      )

  experiment_design = default_experiment_design.make_variation(
      secondary_metrics=["conversions"]
  )

  with pytest.raises(ValueError):
    AnalyzeExperimentMockMethodology().analyze_experiment(
        historical_data, experiment_design, "2024-01-01"
    )


def test_methodology_analyze_experiment_raises_error_if_missing_primary_metric(
    historical_data, default_experiment_design, MockMethodology
):
  class AnalyzeExperimentMockMethodology(MockMethodology):
    """Mock methodology for testing."""

    def _methodology_analyze_experiment(
        self,
        runtime_data: GeoPerformanceDataset,
        experiment_design: ExperimentDesign,
        experiment_start_date: pd.Timestamp,
        experiment_end_date: pd.Timestamp,
        pretest_period_end_date: pd.Timestamp,
    ) -> pd.DataFrame:
      del (
          runtime_data,
          experiment_design,
          experiment_start_date,
          experiment_end_date,
          pretest_period_end_date,
      )

      return (
          pd.DataFrame({
              "metric": ["conversions", "conversions", "revenue"],
              "cell": [1, 2, 1],
              "point_estimate": [1.0] * 3,
              "lower_bound": [0.5] * 3,
              "upper_bound": [1.5] * 3,
              "point_estimate_relative": [0.1] * 3,
              "lower_bound_relative": [0.05] * 3,
              "upper_bound_relative": [0.15] * 3,
              "p_value": [0.01] * 3,
          }),
          {},
      )

  experiment_design = default_experiment_design.make_variation(
      secondary_metrics=["conversions"]
  )

  with pytest.raises(
      ValueError,
      match="Some of the cells are missing the results for the primary metric",
  ):
    AnalyzeExperimentMockMethodology().analyze_experiment(
        historical_data, experiment_design, "2024-01-01"
    )


def test_methodology_analyze_experiment_forces_relative_effect_size_to_na_for_cost_per_metric_and_metric_per_cost_metrics(
    historical_data, default_experiment_design, MockMethodology
):
  class AnalyzeExperimentMockMethodology(MockMethodology):
    """Mock methodology for testing."""

    def _methodology_analyze_experiment(
        self,
        runtime_data: GeoPerformanceDataset,
        experiment_design: ExperimentDesign,
        experiment_start_date: pd.Timestamp,
        experiment_end_date: pd.Timestamp,
        pretest_period_end_date: pd.Timestamp,
    ) -> pd.DataFrame:
      del (
          runtime_data,
          experiment_design,
          experiment_start_date,
          experiment_end_date,
          pretest_period_end_date,
      )

      return (
          pd.DataFrame({
              "metric": ["revenue", "CPiA", "iROAS"],
              "cell": [1, 1, 1],
              "point_estimate": [1.0, 1.0, 1.0],
              "lower_bound": [0.5, 0.5, 0.5],
              "upper_bound": [1.5, 1.5, 1.5],
              "point_estimate_relative": [0.1, 0.1, 0.1],
              "lower_bound_relative": [0.05, 0.05, 0.05],
              "upper_bound_relative": [0.15, 0.15, 0.15],
              "p_value": [0.01, 0.01, 0.01],
          }),
          {},
      )

  experiment_design = default_experiment_design.make_variation(
      secondary_metrics=[geoflex.metrics.CPiA(), geoflex.metrics.iROAS()],
      experiment_budget=geoflex.experiment_design.ExperimentBudget(
          value=-0.1,
          budget_type=geoflex.experiment_design.ExperimentBudgetType.PERCENTAGE_CHANGE,
      ),
  )

  results, _ = AnalyzeExperimentMockMethodology().analyze_experiment(
      historical_data, experiment_design, "2024-01-01"
  )

  expected_results = pd.DataFrame({
      "metric": ["revenue", "CPiA", "iROAS"],
      "is_primary_metric": [True, False, False],
      "cell": [1, 1, 1],
      "point_estimate": [1.0, 1.0, 1.0],
      "lower_bound": [0.5, 0.5, 0.5],
      "upper_bound": [1.5, 1.5, 1.5],
      "point_estimate_relative": [0.1, np.nan, np.nan],
      "lower_bound_relative": [0.05, np.nan, np.nan],
      "upper_bound_relative": [0.15, np.nan, np.nan],
      "p_value": [0.01, 0.01, 0.01],
      "is_significant": [True, True, True],
  })
  pd.testing.assert_frame_equal(results, expected_results, check_like=True)


def test_methodology_analyze_experiment_infers_p_value_if_not_set(
    historical_data, default_experiment_design, MockMethodology
):
  class AnalyzeExperimentMockMethodology(MockMethodology):
    """Mock methodology for testing."""

    def _methodology_analyze_experiment(
        self,
        runtime_data: GeoPerformanceDataset,
        experiment_design: ExperimentDesign,
        experiment_start_date: pd.Timestamp,
        experiment_end_date: pd.Timestamp,
        pretest_period_end_date: pd.Timestamp,
    ) -> pd.DataFrame:
      del (
          runtime_data,
          experiment_design,
          experiment_start_date,
          experiment_end_date,
          pretest_period_end_date,
      )

      return (
          pd.DataFrame({
              "metric": ["revenue"],
              "cell": [1],
              "point_estimate": [1.0],
              "lower_bound": [0.5],
              "upper_bound": [1.5],
              "point_estimate_relative": [0.1],
              "lower_bound_relative": [0.05],
              "upper_bound_relative": [0.15],
          }),
          {},
      )

  results, _ = AnalyzeExperimentMockMethodology().analyze_experiment(
      historical_data, default_experiment_design, "2024-01-01"
  )
  assert "p_value" in results.columns


def test_methodology_analyze_experiment_sets_is_significant_correctly(
    historical_data, default_experiment_design, MockMethodology
):
  class AnalyzeExperimentMockMethodology(MockMethodology):
    """Mock methodology for testing."""

    def _methodology_analyze_experiment(
        self,
        runtime_data: GeoPerformanceDataset,
        experiment_design: ExperimentDesign,
        experiment_start_date: pd.Timestamp,
        experiment_end_date: pd.Timestamp,
        pretest_period_end_date: pd.Timestamp,
    ) -> pd.DataFrame:
      del (
          runtime_data,
          experiment_design,
          experiment_start_date,
          experiment_end_date,
          pretest_period_end_date,
      )

      return (
          pd.DataFrame({
              "metric": ["revenue", "revenue"],
              "cell": [1, 2],
              "point_estimate": [1.0, 1.0],
              "lower_bound": [0.5, 0.5],
              "upper_bound": [1.5, 1.5],
              "point_estimate_relative": [0.1, 0.1],
              "lower_bound_relative": [0.05, 0.05],
              "upper_bound_relative": [0.15, 0.15],
              "p_value": [0.01, 0.1],
          }),
          {},
      )

  results, _ = AnalyzeExperimentMockMethodology().analyze_experiment(
      historical_data, default_experiment_design, "2024-01-01"
  )
  assert results["is_significant"].tolist() == [True, False]


def test_list_methodologies_includes_registered_methodology(
    MockMethodology,
):
  register_methodology(MockMethodology)
  assert "MockMethodology" in list_methodologies()


def test_list_methodologies_excludes_testing_methodology():
  assert "TestingMethodology" not in list_methodologies()


def test_is_eligible_for_design_and_data_returns_true_for_good_design_and_data(
    MockMethodology, historical_data, default_experiment_design
):
  assert MockMethodology().is_eligible_for_design_and_data(
      default_experiment_design, historical_data
  )


def test_is_eligible_for_design_and_data_returns_false_if_design_name_does_not_match_methodology_name(
    MockMethodology, historical_data, default_experiment_design
):
  assert not MockMethodology().is_eligible_for_design_and_data(
      default_experiment_design.make_variation(methodology="OtherMethodology"),
      historical_data,
  )


def test_is_eligible_for_design_and_data_returns_false_if_not_enough_data_days(
    MockMethodology, historical_data, default_experiment_design
):
  assert not MockMethodology().is_eligible_for_design_and_data(
      default_experiment_design.make_variation(runtime_weeks=100),
      historical_data,
  )


def test_is_eligible_for_design_and_data_returns_false_if_zero_costs_and_percentage_change_budget(
    MockMethodology, historical_data, default_experiment_design
):
  zero_costs_data = historical_data.data.copy()
  zero_costs_data["cost"] = 0.0
  zero_costs_data = GeoPerformanceDataset(data=zero_costs_data)

  assert not MockMethodology().is_eligible_for_design_and_data(
      default_experiment_design.make_variation(
          experiment_budget=ExperimentBudget(
              budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
              value=-1.0,
          ),
          primary_metric=geoflex.metrics.iROAS(),
      ),
      zero_costs_data,
  )


def test_is_eligible_for_design_and_data_returns_false_if_cell_volume_metric_column_not_in_data(
    MockMethodology, historical_data, default_experiment_design
):
  assert not MockMethodology().is_eligible_for_design_and_data(
      default_experiment_design.make_variation(
          cell_volume_constraint=CellVolumeConstraint(
              constraint_type=CellVolumeConstraintType.MAX_PERCENTAGE_OF_METRIC,
              values=[None, 0.5, 0.2],
              metric_column="non_existent_metric",
          ),
      ),
      historical_data,
  )


def test_is_eligible_for_design_and_data_returns_false_if_cost_column_not_in_data(
    MockMethodology, historical_data, default_experiment_design
):
  assert not MockMethodology().is_eligible_for_design_and_data(
      default_experiment_design.make_variation(
          secondary_metrics=[
              geoflex.metrics.iROAS(cost_column="non_existent_cost")
          ],
          experiment_budget=ExperimentBudget(
              budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
              value=-1.0,
          ),
      ),
      historical_data,
  )


def test_is_eligible_for_design_and_data_returns_false_if_response_column_not_in_data(
    MockMethodology, historical_data, default_experiment_design
):
  assert not MockMethodology().is_eligible_for_design_and_data(
      default_experiment_design.make_variation(
          secondary_metrics=["non_existent_response"],
      ),
      historical_data,
  )


def test_is_eligible_for_design_and_data_returns_false_if_methodology_returns_false(
    MockMethodology, historical_data, default_experiment_design
):
  class MockMethodologyWithIneligibility(MockMethodology):

    def _methodology_is_eligible_for_design_and_data(
        self, design: ExperimentDesign, historical_data: GeoPerformanceDataset
    ) -> bool:
      del design, historical_data
      return False

  assert not MockMethodologyWithIneligibility().is_eligible_for_design_and_data(
      default_experiment_design,
      historical_data,
  )


@pytest.mark.parametrize(
    "runtime_weeks,user_experiment_end_date,expected_experiment_end_date",
    [
        (1, "2024-01-01", "2024-01-01"),
        (1, None, "2024-01-08"),
        (2, None, "2024-01-15"),
        (1000, None, "2024-04-10"),  # Last day in the data + 1
        (
            1000,
            "2024-04-11",
            "2024-04-10",
        ),  # User date is after the last day in the data, use the last day in
        # the data instead.
    ],
)
def test_experiment_end_date_is_set_correctly(
    MockMethodology,
    historical_data,
    default_experiment_design,
    runtime_weeks,
    user_experiment_end_date,
    expected_experiment_end_date,
):

  class MockMethodologyWithExperimentEndDate(MockMethodology):
    """Mock methodology for testing."""

    def _methodology_analyze_experiment(
        self,
        runtime_data: GeoPerformanceDataset,
        experiment_design: ExperimentDesign,
        experiment_start_date: pd.Timestamp,
        experiment_end_date: pd.Timestamp,
        pretest_period_end_date: pd.Timestamp,
    ) -> pd.DataFrame:
      del (
          runtime_data,
          experiment_design,
          experiment_start_date,
          pretest_period_end_date,
      )

      # Perform the assertion here to avoid having to mock the return value.
      assert experiment_end_date == dt.datetime.strptime(
          expected_experiment_end_date, "%Y-%m-%d"
      )

      return (
          pd.DataFrame({
              "metric": ["revenue", "revenue"],
              "cell": [1, 2],
              "point_estimate": [1.0, 1.0],
              "lower_bound": [0.5, 0.5],
              "upper_bound": [1.5, 1.5],
              "point_estimate_relative": [0.1, 0.1],
              "lower_bound_relative": [0.05, 0.05],
              "upper_bound_relative": [0.15, 0.15],
              "p_value": [0.01, 0.1],
          }),
          {},
      )

  MockMethodologyWithExperimentEndDate().analyze_experiment(
      historical_data,
      default_experiment_design.make_variation(runtime_weeks=runtime_weeks),
      "2024-01-01",
      user_experiment_end_date,
  )


@pytest.mark.parametrize(
    "pretest_period_end_date,expected_pretest_period_end_date",
    [
        ("2024-01-01", "2024-01-01"),
        (None, "2024-02-01"),
    ],
)
def test_pretest_period_end_date_is_set_correctly(
    MockMethodology,
    historical_data,
    default_experiment_design,
    pretest_period_end_date,
    expected_pretest_period_end_date,
):
  class MockMethodologyWithPretestPeriodEndDate(MockMethodology):
    """Mock methodology for testing."""

    def _methodology_analyze_experiment(
        self,
        runtime_data: GeoPerformanceDataset,
        experiment_design: ExperimentDesign,
        experiment_start_date: pd.Timestamp,
        experiment_end_date: pd.Timestamp,
        pretest_period_end_date: pd.Timestamp,
    ) -> pd.DataFrame:
      del (
          runtime_data,
          experiment_design,
          experiment_start_date,
          experiment_end_date,
      )

      # Perform the assertion here to avoid having to mock the return value.
      assert pretest_period_end_date == dt.datetime.strptime(
          expected_pretest_period_end_date, "%Y-%m-%d"
      )

      return (
          pd.DataFrame({
              "metric": ["revenue", "revenue"],
              "cell": [1, 2],
              "point_estimate": [1.0, 1.0],
              "lower_bound": [0.5, 0.5],
              "upper_bound": [1.5, 1.5],
              "point_estimate_relative": [0.1, 0.1],
              "lower_bound_relative": [0.05, 0.05],
              "upper_bound_relative": [0.15, 0.15],
              "p_value": [0.01, 0.1],
          }),
          {},
      )

  MockMethodologyWithPretestPeriodEndDate().analyze_experiment(
      historical_data,
      default_experiment_design,
      "2024-02-01",
      pretest_period_end_date=pretest_period_end_date,
  )


def test_analyze_experiment_raises_exception_if_pretest_period_end_date_after_experiment_start_date(
    MockMethodology,
    historical_data,
    default_experiment_design,
):
  with pytest.raises(ValueError):
    MockMethodology().analyze_experiment(
        historical_data,
        default_experiment_design,
        "2024-02-01",
        pretest_period_end_date="2024-02-02",
    )


@pytest.mark.parametrize(
    "return_intermediate_data,expected_intermediate_data",
    [(True, {"mock_intermediate_result": "analyze_experiment"}), (False, {})],
)
def test_analyze_experiment_returns_intermediate_data_if_requested(
    MockMethodology,
    historical_data,
    default_experiment_design,
    return_intermediate_data,
    expected_intermediate_data,
):

  class MockMethodologyWithIntermediateData(MockMethodology):
    """Mock methodology for testing."""

    def _methodology_analyze_experiment(
        self,
        runtime_data: GeoPerformanceDataset,
        experiment_design: ExperimentDesign,
        experiment_start_date: pd.Timestamp,
        experiment_end_date: pd.Timestamp,
        pretest_period_end_date: pd.Timestamp,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
      del (
          runtime_data,
          experiment_design,
          experiment_start_date,
          experiment_end_date,
          pretest_period_end_date,
      )

      return pd.DataFrame({
          "metric": ["revenue", "CPiA", "iROAS"],
          "cell": [1, 1, 1],
          "point_estimate": [1.0, 1.0, 1.0],
          "lower_bound": [0.5, 0.5, 0.5],
          "upper_bound": [1.5, 1.5, 1.5],
          "point_estimate_relative": [0.1, 0.1, 0.1],
          "lower_bound_relative": [0.05, 0.05, 0.05],
          "upper_bound_relative": [0.15, 0.15, 0.15],
          "p_value": [0.01, 0.01, 0.01],
      }), {"mock_intermediate_result": "analyze_experiment"}

  default_experiment_design.geo_assignment = GeoAssignment(
      treatment=[["US", "UK"]], control=["CA", "AU"]
  )
  (
      results,
      intermediate_data,
  ) = MockMethodologyWithIntermediateData().analyze_experiment(
      historical_data,
      default_experiment_design,
      "2024-02-01",
      return_intermediate_data=return_intermediate_data,
  )
  assert isinstance(results, pd.DataFrame)
  assert intermediate_data == expected_intermediate_data


@pytest.mark.parametrize(
    "return_intermediate_data,expected_intermediate_data",
    [(True, {"mock_intermediate_result": "assign_geos"}), (False, {})],
)
def test_assign_geos_returns_intermediate_data_if_requested(
    MockMethodology,
    historical_data,
    default_experiment_design,
    return_intermediate_data,
    expected_intermediate_data,
):

  class MockMethodologyWithIntermediateData(MockMethodology):
    """Mock methodology for testing."""

    def _methodology_assign_geos(
        self,
        experiment_design: ExperimentDesign,
        historical_data: GeoPerformanceDataset,
    ) -> tuple[GeoAssignment, dict[str, Any]]:
      del experiment_design, historical_data
      return (
          GeoAssignment(treatment=[["US"], ["UK"]], control=["CA", "AU"]),
          {"mock_intermediate_result": "assign_geos"},
      )

  (
      results,
      intermediate_data,
  ) = MockMethodologyWithIntermediateData().assign_geos(
      default_experiment_design,
      historical_data,
      return_intermediate_data=return_intermediate_data,
  )
  assert isinstance(results, GeoAssignment)
  assert intermediate_data == expected_intermediate_data
