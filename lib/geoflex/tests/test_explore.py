"""Tests for the explore module."""

import geoflex.data
import geoflex.experiment_design
import geoflex.exploration_spec
import geoflex.explore
import geoflex.metrics
import mock
import numpy as np
import pandas as pd
import pytest

ExperimentDesignExplorationSpec = (
    geoflex.exploration_spec.ExperimentDesignExplorationSpec
)
ExperimentDesignExplorer = geoflex.explore.ExperimentDesignExplorer
GeoEligibility = geoflex.experiment_design.GeoEligibility
ExperimentBudget = geoflex.experiment_design.ExperimentBudget
ExperimentBudgetType = geoflex.experiment_design.ExperimentBudgetType
EffectScope = geoflex.experiment_design.EffectScope
ExperimentDesign = geoflex.experiment_design.ExperimentDesign
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


@pytest.fixture(name="historical_data")
def mock_historical_data_fixture():
  """Fixture for a mock historical data."""
  rng = np.random.default_rng(seed=42)
  data = pd.DataFrame({
      "geo_id": ["US"] * 100 + ["CA"] * 100 + ["DE"] * 100 + ["FR"] * 100,
      "date": pd.date_range(start="2024-01-01", periods=100).tolist() * 4,
      "revenue": rng.random(size=400),
      "cost": rng.random(size=400),
      "conversions": rng.random(size=400),
  })
  data["date"] = data["date"].dt.strftime("%Y-%m-%d")

  return geoflex.data.GeoPerformanceDataset(data=data)


@pytest.fixture(name="historical_data_lots_of_geos")
def mock_historical_data_lots_of_geos_fixture():
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


@pytest.fixture(name="default_explore_spec")
def mock_explore_spec_fixture():
  """Fixture for a mock explore spec."""
  return ExperimentDesignExplorationSpec(
      primary_metric="revenue",
      secondary_metrics=[
          "conversions",
          geoflex.metrics.iROAS(),
          geoflex.metrics.CPiA(),
      ],
      experiment_budget_candidates=[
          ExperimentBudget(
              value=-1.0, budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE
          ),
          ExperimentBudget(
              value=-0.5, budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE
          ),
      ],
      eligible_methodologies=["TestingMethodology"],
      runtime_weeks_candidates=[2, 4],
      n_cells=3,
      geo_eligibility_candidates=[None],
      effect_scope=EffectScope.ALL_GEOS,
  )


def test_explorer_clear_designs(historical_data, default_explore_spec):
  explorer = ExperimentDesignExplorer(
      historical_data=historical_data,
      explore_spec=default_explore_spec,
  )

  explorer.explored_designs = {
      "test_design_id": mock.MagicMock(spec=ExperimentDesign)
  }

  explorer.clear_designs()

  assert explorer.explored_designs == {}  # pylint: disable=g-explicit-bool-comparison


def test_explorer_get_design_by_id_returns_correct_design(
    historical_data, default_explore_spec
):
  explorer = ExperimentDesignExplorer(
      historical_data=historical_data,
      explore_spec=default_explore_spec,
  )

  mock_design_1 = mock.MagicMock(spec=ExperimentDesign)
  mock_design_2 = mock.MagicMock(spec=ExperimentDesign)
  explorer.explored_designs = {
      "test_design_id_1": mock_design_1,
      "test_design_id_2": mock_design_2,
  }

  assert explorer.get_design_by_id("test_design_id_2") == mock_design_2


def test_explorer_get_design_by_id_returns_none_if_design_not_found(
    historical_data, default_explore_spec
):
  explorer = ExperimentDesignExplorer(
      historical_data=historical_data,
      explore_spec=default_explore_spec,
  )

  explorer.explored_designs = {
      "test_design_id_1": mock.MagicMock(spec=ExperimentDesign),
      "test_design_id_2": mock.MagicMock(spec=ExperimentDesign),
  }

  assert explorer.get_design_by_id("missing_design") is None


def test_explorer_exp_start_date_is_set_correctly(
    historical_data, default_explore_spec
):
  explorer = ExperimentDesignExplorer(
      historical_data=historical_data,
      explore_spec=default_explore_spec,
  )

  # This should be the latest date in the historical data minus the maximum
  # runtime weeks candidate.
  assert explorer.exp_start_date == (
      historical_data.parsed_data["date"].max() - pd.Timedelta(weeks=4)
  )


def test_explorer_explore_suggests_designs_within_spec(
    historical_data, default_explore_spec
):
  explorer = ExperimentDesignExplorer(
      historical_data=historical_data,
      explore_spec=default_explore_spec,
  )
  explorer._experiment_design_evaluator = mock.MagicMock()  # pylint: disable=protected-access
  explorer.explore(
      max_trials=20, aa_simulations_per_trial=3, ab_simulations_per_trial=3
  )

  for suggested_design in explorer.get_designs():
    assert isinstance(suggested_design, ExperimentDesign)
    assert (
        suggested_design.primary_metric == default_explore_spec.primary_metric
    )
    assert (
        suggested_design.experiment_budget
        in default_explore_spec.experiment_budget_candidates
    )
    assert suggested_design.secondary_metrics == (
        default_explore_spec.secondary_metrics
    )
    assert (
        suggested_design.methodology
        in default_explore_spec.eligible_methodologies
    )
    assert isinstance(suggested_design.methodology_parameters, dict)
    assert (
        suggested_design.runtime_weeks
        in default_explore_spec.runtime_weeks_candidates
    )
    assert suggested_design.n_cells == default_explore_spec.n_cells
    assert suggested_design.alpha == default_explore_spec.alpha
    assert (
        suggested_design.alternative_hypothesis
        == default_explore_spec.alternative_hypothesis
    )
    assert (
        suggested_design.geo_eligibility
        in default_explore_spec.geo_eligibility_candidates
    )
    assert (
        suggested_design.cell_volume_constraint
        in default_explore_spec.cell_volume_constraint_candidates
    )
    assert suggested_design.random_seed in default_explore_spec.random_seeds
    assert suggested_design.effect_scope == default_explore_spec.effect_scope


def test_explorer_explore_uses_default_methodology_parameter_candidates_if_not_set(
    historical_data, default_explore_spec
):
  explorer = ExperimentDesignExplorer(
      historical_data=historical_data,
      explore_spec=default_explore_spec,
  )
  explorer._experiment_design_evaluator = mock.MagicMock()  # pylint: disable=protected-access
  explorer.explore(
      max_trials=20, aa_simulations_per_trial=3, ab_simulations_per_trial=3
  )

  unique_methodology_parameters = {}
  for suggested_design in explorer.get_designs():
    for key in suggested_design.methodology_parameters:
      if key not in unique_methodology_parameters:
        unique_methodology_parameters[key] = set()

      unique_methodology_parameters[key].add(
          suggested_design.methodology_parameters[key]
      )

  # Based on the default_methodology_parameter_candidates in the
  # TestingMethodology methodology.
  expected_unique_methodology_parameters = {
      "mock_parameter": {1, 2},
  }
  assert unique_methodology_parameters == expected_unique_methodology_parameters


def test_explorer_explore_uses_spec_methodology_parameter_candidates_if_set(
    historical_data, default_explore_spec
):
  explore_spec = default_explore_spec.model_copy(
      update={
          "methodology_parameter_candidates": {
              "TestingMethodology": {"mock_parameter": [1, 2, 3]}
          }
      }
  )
  explorer = ExperimentDesignExplorer(
      historical_data=historical_data,
      explore_spec=explore_spec,
  )
  explorer._experiment_design_evaluator = mock.MagicMock()  # pylint: disable=protected-access
  explorer.explore(
      max_trials=20, aa_simulations_per_trial=3, ab_simulations_per_trial=3
  )

  unique_methodology_parameters = {}
  for suggested_design in explorer.get_designs():
    for key in suggested_design.methodology_parameters:
      if key not in unique_methodology_parameters:
        unique_methodology_parameters[key] = set()

      unique_methodology_parameters[key].add(
          suggested_design.methodology_parameters[key]
      )

  # Based on the spec methodology parameter candidates.
  expected_unique_methodology_parameters = {
      "mock_parameter": {1, 2, 3},
  }
  assert unique_methodology_parameters == expected_unique_methodology_parameters


def test_explorer_explore_assigns_geos_and_evaluates_designs(
    historical_data, default_explore_spec
):
  explorer = ExperimentDesignExplorer(
      historical_data=historical_data,
      explore_spec=default_explore_spec,
  )

  explorer.explore(
      max_trials=2,
      n_jobs=1,
      aa_simulations_per_trial=3,
      ab_simulations_per_trial=3,
  )
  for design in explorer.get_designs():
    assert isinstance(
        design.evaluation_results,
        geoflex.experiment_design.ExperimentDesignEvaluationResults,
    )
    assert isinstance(
        design.geo_assignment, geoflex.experiment_design.GeoAssignment
    )


def test_explorer_extend_top_n_designs_extends_designs_correctly(
    historical_data, default_explore_spec
):
  explorer = ExperimentDesignExplorer(
      historical_data=historical_data,
      explore_spec=default_explore_spec,
  )

  explorer.explore(
      max_trials=4,
      n_jobs=1,
      aa_simulations_per_trial=3,
      ab_simulations_per_trial=3,
  )

  summary_results_1 = explorer.get_design_summaries()

  explorer.extend_top_n_designs(
      top_n=2,
      n_aa_simulations=3,
      n_ab_simulations=3,
  )
  summary_results_2 = explorer.get_design_summaries()

  # Top designs are changed
  assert not summary_results_1.iloc[:2].equals(summary_results_2.iloc[:2])
  # Bottom designs are unchanged
  assert summary_results_1.iloc[2:].equals(summary_results_2.iloc[2:])


def test_explorer_extend_designs_by_ids_correctly(
    historical_data, default_explore_spec
):
  explorer = ExperimentDesignExplorer(
      historical_data=historical_data,
      explore_spec=default_explore_spec,
  )

  explorer.explore(
      max_trials=4,
      n_jobs=1,
      aa_simulations_per_trial=3,
      ab_simulations_per_trial=3,
  )

  summary_results_1 = explorer.get_design_summaries()

  design_ids_to_extend = [
      explorer.get_designs()[1].design_id,
      explorer.get_designs()[3].design_id,
  ]
  other_design_ids = [
      design_id
      for design_id in summary_results_1.index
      if design_id not in design_ids_to_extend
  ]
  explorer.extend_design_by_id(
      design_ids_to_extend,
      n_aa_simulations=3,
      n_ab_simulations=3,
  )
  summary_results_2 = explorer.get_design_summaries()

  # Selected designs are changed
  assert not summary_results_1.loc[design_ids_to_extend].equals(
      summary_results_2.loc[design_ids_to_extend]
  )
  # Other designs are unchanged
  assert summary_results_1.loc[other_design_ids].equals(
      summary_results_2.loc[other_design_ids]
  )


def test_explorer_extend_designs_by_ids_correctly_single_design(
    historical_data, default_explore_spec
):
  explorer = ExperimentDesignExplorer(
      historical_data=historical_data,
      explore_spec=default_explore_spec,
  )

  explorer.explore(
      max_trials=4,
      n_jobs=1,
      aa_simulations_per_trial=3,
      ab_simulations_per_trial=3,
  )

  summary_results_1 = explorer.get_design_summaries()

  design_id_to_extend = explorer.get_designs()[1].design_id
  other_design_ids = [
      design_id
      for design_id in summary_results_1.index
      if design_id != design_id_to_extend
  ]
  explorer.extend_design_by_id(
      design_id_to_extend,
      n_aa_simulations=3,
      n_ab_simulations=3,
  )
  summary_results_2 = explorer.get_design_summaries()

  # Selected designs are changed
  assert not summary_results_1.loc[design_id_to_extend].equals(
      summary_results_2.loc[design_id_to_extend]
  )
  # Other designs are unchanged
  assert summary_results_1.loc[other_design_ids].equals(
      summary_results_2.loc[other_design_ids]
  )


def test_get_design_summaries_returns_correct_data_relative_effects(
    historical_data_lots_of_geos, default_explore_spec
):
  explorer = ExperimentDesignExplorer(
      historical_data=historical_data_lots_of_geos,
      explore_spec=default_explore_spec,
  )
  explorer.explore(
      max_trials=3,
      n_jobs=1,
      aa_simulations_per_trial=3,
      ab_simulations_per_trial=3,
  )
  design_summaries = explorer.get_design_summaries(
      use_relative_effects_where_possible=True,
  )

  assert isinstance(design_summaries, pd.DataFrame)
  assert design_summaries.index.names == ["design_id"]
  assert design_summaries.dtypes.to_dict() == {
      "experiment_budget": "object",
      "primary_metric": "object",
      "secondary_metrics": "object",
      "methodology": "object",
      "methodology_parameters": "object",
      "runtime_weeks": "int64",
      "n_cells": "int64",
      "cell_volume_constraint": "object",
      "effect_scope": "object",
      "alpha": "float64",
      "alternative_hypothesis": "object",
      "random_seed": "int64",
      "geo_assignment_control": "object",
      "geo_assignment_treatment_1": "object",
      "geo_assignment_treatment_2": "object",
      "failing_checks": "object",
      "all_checks_pass": "bool",
      "representativeness_score": "float64",
      "MDE (CPiA)": "float64",
      "MDE (iROAS)": "float64",
      "Relative MDE (conversions)": "float64",
      "primary_metric_failing_checks": "object",
      "primary_metric_all_checks_pass": "bool",
      "primary_metric_standard_error": "float64",
      "Relative MDE (revenue, primary metric)": "float64",
      "actual_cell_volumes": "object",
      "warnings": "object",
      "sufficient_simulations": "bool",
  }
  assert len(design_summaries) == 3


def test_get_design_summaries_returns_correct_data_absolute_effects(
    historical_data_lots_of_geos, default_explore_spec
):
  explorer = ExperimentDesignExplorer(
      historical_data=historical_data_lots_of_geos,
      explore_spec=default_explore_spec,
  )
  explorer.explore(
      max_trials=3,
      n_jobs=1,
      aa_simulations_per_trial=3,
      ab_simulations_per_trial=3,
  )
  design_summaries = explorer.get_design_summaries(
      use_relative_effects_where_possible=False,
  )

  assert isinstance(design_summaries, pd.DataFrame)
  assert design_summaries.index.names == ["design_id"]
  assert design_summaries.dtypes.to_dict() == {
      "experiment_budget": "object",
      "primary_metric": "object",
      "secondary_metrics": "object",
      "methodology": "object",
      "methodology_parameters": "object",
      "runtime_weeks": "int64",
      "n_cells": "int64",
      "cell_volume_constraint": "object",
      "effect_scope": "object",
      "alpha": "float64",
      "alternative_hypothesis": "object",
      "random_seed": "int64",
      "geo_assignment_control": "object",
      "geo_assignment_treatment_1": "object",
      "geo_assignment_treatment_2": "object",
      "failing_checks": "object",
      "all_checks_pass": "bool",
      "representativeness_score": "float64",
      "MDE (CPiA)": "float64",
      "MDE (iROAS)": "float64",
      "MDE (conversions)": "float64",
      "primary_metric_failing_checks": "object",
      "primary_metric_all_checks_pass": "bool",
      "primary_metric_standard_error": "float64",
      "MDE (revenue, primary metric)": "float64",
      "actual_cell_volumes": "object",
      "warnings": "object",
      "sufficient_simulations": "bool",
  }
  assert len(design_summaries) == 3


def test_get_design_summaries_returns_correct_data_pareto_front_only(
    historical_data_lots_of_geos, default_explore_spec
):
  explorer = ExperimentDesignExplorer(
      historical_data=historical_data_lots_of_geos,
      explore_spec=default_explore_spec,
  )
  explorer.explore(
      max_trials=10,
      n_jobs=1,
      aa_simulations_per_trial=3,
      ab_simulations_per_trial=3,
  )
  design_summaries = explorer.get_design_summaries(
      pareto_front_only=True,
  )

  assert isinstance(design_summaries, pd.DataFrame)
  assert (
      len(design_summaries) == 1
  )  # 10 designs, but 9 are dominated by the others


@pytest.mark.parametrize("use_relative_effects_where_possible", [True, False])
@pytest.mark.parametrize("drop_constant_columns", [True, False])
@pytest.mark.parametrize("shorten_geo_assignments", [True, False])
def test_get_design_summaries_returns_styler_if_style_output_is_true(
    historical_data_lots_of_geos,
    default_explore_spec,
    use_relative_effects_where_possible,
    drop_constant_columns,
    shorten_geo_assignments,
):
  explorer = ExperimentDesignExplorer(
      historical_data=historical_data_lots_of_geos,
      explore_spec=default_explore_spec,
  )
  explorer.explore(
      max_trials=3,
      n_jobs=1,
      aa_simulations_per_trial=3,
      ab_simulations_per_trial=3,
  )
  design_summaries = explorer.get_design_summaries(
      style_output=True,
      use_relative_effects_where_possible=use_relative_effects_where_possible,
      drop_constant_columns=drop_constant_columns,
      shorten_geo_assignments=shorten_geo_assignments,
  )
  assert isinstance(design_summaries, pd.io.formats.style.Styler)


def test_count_all_eligible_designs_returns_correct_data(
    historical_data_lots_of_geos, default_explore_spec
):
  explorer = ExperimentDesignExplorer(
      historical_data=historical_data_lots_of_geos,
      explore_spec=default_explore_spec,
  )
  counts = explorer.count_all_eligible_designs()
  assert counts == {"TestingMethodology": 8}


def test_can_write_explorer_to_json(
    historical_data_lots_of_geos, default_explore_spec
):
  explorer = ExperimentDesignExplorer(
      historical_data=historical_data_lots_of_geos,
      explore_spec=default_explore_spec,
  )
  explorer.explore(
      max_trials=3,
      n_jobs=1,
      aa_simulations_per_trial=3,
      ab_simulations_per_trial=3,
  )
  json_string = explorer.model_dump_json()

  new_explorer = ExperimentDesignExplorer.model_validate_json(json_string)

  pd.testing.assert_frame_equal(
      new_explorer.get_design_summaries(),
      explorer.get_design_summaries(),
      check_like=True,
  )

  assert new_explorer.study is None  # Study cannot be written to JSON.
