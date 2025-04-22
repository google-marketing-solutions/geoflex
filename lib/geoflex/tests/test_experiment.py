"""Tests for the experiment module."""

import geoflex.data
import geoflex.experiment
import geoflex.experiment_design
import geoflex.metrics
import mock
import numpy as np
import optuna as op
import pandas as pd
import pytest

ExperimentDesignSpec = geoflex.experiment_design.ExperimentDesignSpec
ExperimentType = geoflex.experiment_design.ExperimentType
GeoEligibility = geoflex.experiment_design.GeoEligibility
ExperimentBudget = geoflex.experiment_design.ExperimentBudget
ExperimentBudgetType = geoflex.experiment_design.ExperimentBudgetType
EffectScope = geoflex.experiment_design.EffectScope
ExperimentDesign = geoflex.experiment_design.ExperimentDesign

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


def test_experiment_can_record_new_designs():
  experiment = geoflex.experiment.Experiment(
      name="test_experiment",
      historical_data=mock.MagicMock(),
      design_spec=mock.MagicMock(),
  )

  mock_design = mock.MagicMock()
  mock_design.design_id = "test_design_id"
  mock_raw_eval_metrics = mock.MagicMock()
  mock_primary_metric_standard_error = mock.MagicMock()
  mock_representiveness_score = mock.MagicMock()

  experiment.record_design(
      design=mock_design,
      raw_eval_metrics=mock_raw_eval_metrics,
      primary_metric_standard_error=mock_primary_metric_standard_error,
      representiveness_score=mock_representiveness_score,
  )

  assert experiment.n_experiment_designs == 1

  recorded_design_results = experiment.get_experiment_design_results(
      "test_design_id"
  )
  assert recorded_design_results == {
      "design": mock_design,
      "raw_eval_metrics": mock_raw_eval_metrics,
      "representiveness_score": mock_representiveness_score,
      "primary_metric_standard_error": mock_primary_metric_standard_error,
  }


def test_experiment_clear_designs():
  experiment = geoflex.experiment.Experiment(
      name="test_experiment",
      historical_data=mock.MagicMock(),
      design_spec=mock.MagicMock(),
  )
  experiment.record_design(
      design=mock.MagicMock(),
      raw_eval_metrics=mock.MagicMock(),
      primary_metric_standard_error=mock.MagicMock(),
      representiveness_score=mock.MagicMock(),
  )
  experiment.clear_designs()
  assert experiment.n_experiment_designs == 0


def test_experiment_get_all_raw_eval_metrics():
  experiment = geoflex.experiment.Experiment(
      name="test_experiment",
      historical_data=mock.MagicMock(),
      design_spec=mock.MagicMock(),
  )

  mock_design_1 = mock.MagicMock()
  mock_design_1.design_id = "test_design_id_1"
  mock_raw_eval_metrics_1 = pd.DataFrame({
      "col_1": [1, 2, 3],
      "col_2": [4, 5, 6],
  })
  mock_design_2 = mock.MagicMock()
  mock_design_2.design_id = "test_design_id_2"
  mock_raw_eval_metrics_2 = pd.DataFrame({
      "col_1": [7, 8, 9],
      "col_2": [10, 11, 12],
  })
  experiment.record_design(
      design=mock_design_1,
      raw_eval_metrics=mock_raw_eval_metrics_1,
      primary_metric_standard_error=mock.MagicMock(),
      representiveness_score=mock.MagicMock(),
  )
  experiment.record_design(
      design=mock_design_2,
      raw_eval_metrics=mock_raw_eval_metrics_2,
      primary_metric_standard_error=mock.MagicMock(),
      representiveness_score=mock.MagicMock(),
  )

  expected_all_raw_eval_metrics = pd.DataFrame({
      "design_id": ["test_design_id_1"] * 3 + ["test_design_id_2"] * 3,
      "col_1": [1, 2, 3, 7, 8, 9],
      "col_2": [4, 5, 6, 10, 11, 12],
  })
  pd.testing.assert_frame_equal(
      experiment.all_raw_eval_metrics,
      expected_all_raw_eval_metrics,
      check_like=True,
  )


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


@pytest.fixture(name="default_design_spec")
def mock_design_spec_fixture():
  """Fixture for a mock design spec."""
  return ExperimentDesignSpec(
      experiment_type=ExperimentType.GO_DARK,
      primary_metric="revenue",
      secondary_metrics=[
          "conversions",
          geoflex.metrics.ROAS(),
          geoflex.metrics.CPA(),
      ],
      experiment_budget_candidates=[
          ExperimentBudget(
              value=-1.0, budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE
          ),
          ExperimentBudget(
              value=-0.5, budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE
          ),
      ],
      eligible_methodologies=["RCT"],
      runtime_weeks_candidates=[2, 4],
      n_cells=3,
      geo_eligibility_candidates=[None],
      effect_scope=EffectScope.ALL_GEOS,
  )


def test_experiment_bootstrapper_is_initialized_correctly(
    historical_data, default_design_spec
):
  experiment = geoflex.experiment.Experiment(
      name="test_experiment",
      historical_data=historical_data,
      design_spec=default_design_spec,
  )
  assert experiment.bootstrapper.log_transform
  assert experiment.bootstrapper.seasonality == 7
  assert experiment.bootstrapper.seasons_per_block == 2


def test_experiment_representativeness_scorer_is_initialized_correctly(
    historical_data, default_design_spec
):
  experiment = geoflex.experiment.Experiment(
      name="test_experiment",
      historical_data=historical_data,
      design_spec=default_design_spec,
  )
  assert experiment.representativeness_scorer.historical_data.equals(
      historical_data.parsed_data
  )
  assert (
      experiment.representativeness_scorer.geo_column_name
      == historical_data.geo_id_column
  )
  assert experiment.representativeness_scorer.geos == historical_data.geos


@pytest.fixture(name="mock_trial")
def mock_trial_fixture():
  """Fixture for a mock trial."""
  study = op.create_study(direction="minimize")
  trial = study.ask()
  return trial


def test_experiment_suggest_experiment_design_returns_correct_design(
    historical_data, default_design_spec, mock_trial
):
  experiment = geoflex.experiment.Experiment(
      name="test_experiment",
      historical_data=historical_data,
      design_spec=default_design_spec,
  )
  suggested_design = experiment.suggest_experiment_design(mock_trial)

  assert isinstance(suggested_design, ExperimentDesign)
  assert suggested_design.experiment_type == default_design_spec.experiment_type
  assert suggested_design.primary_metric == default_design_spec.primary_metric
  assert (
      suggested_design.experiment_budget
      in default_design_spec.experiment_budget_candidates
  )
  assert suggested_design.secondary_metrics == (
      default_design_spec.secondary_metrics
  )
  assert (
      suggested_design.methodology in default_design_spec.eligible_methodologies
  )
  assert isinstance(suggested_design.methodology_parameters, dict)
  assert (
      suggested_design.runtime_weeks
      in default_design_spec.runtime_weeks_candidates
  )
  assert suggested_design.n_cells == default_design_spec.n_cells
  assert suggested_design.alpha == default_design_spec.alpha
  assert (
      suggested_design.alternative_hypothesis
      == default_design_spec.alternative_hypothesis
  )
  assert (
      suggested_design.geo_eligibility
      in default_design_spec.geo_eligibility_candidates
  )
  assert (
      suggested_design.n_geos_per_group
      in default_design_spec.n_geos_per_group_candidates
  )
  assert suggested_design.random_seed in default_design_spec.random_seeds
  assert suggested_design.effect_scope == default_design_spec.effect_scope


def test_simulate_experiments_returns_correct_data(
    historical_data_lots_of_geos, default_design_spec, mock_trial
):
  experiment = geoflex.experiment.Experiment(
      name="test_experiment",
      historical_data=historical_data_lots_of_geos,
      design_spec=default_design_spec,
  )
  suggested_design = experiment.suggest_experiment_design(mock_trial)
  simulated_data = experiment.simulate_experiments(
      suggested_design, simulations_per_trial=1
  )

  assert isinstance(simulated_data, pd.DataFrame)
  assert len(simulated_data) == 8  # 4 metrics, 2 treatment arms
  assert simulated_data.dtypes.to_dict() == {
      "cell": "int64",
      "metric": "object",
      "is_primary_metric": "bool",
      "point_estimate": "float64",
      "lower_bound": "float64",
      "upper_bound": "float64",
      "point_estimate_relative": "object",
      "lower_bound_relative": "object",
      "upper_bound_relative": "object",
      "p_value": "float64",
      "is_significant": "bool",
  }


def test_simulate_experiments_returns_none_for_invalid_design(
    historical_data_lots_of_geos, default_design_spec, mock_trial
):
  experiment = geoflex.experiment.Experiment(
      name="test_experiment",
      historical_data=historical_data_lots_of_geos,
      design_spec=default_design_spec,
  )
  suggested_design = experiment.suggest_experiment_design(mock_trial)

  # Mock the RCT methodology to always say invalid design.
  with mock.patch.object(
      geoflex.methodology.rct.RCT, "is_eligible_for_design", autospec=True
  ) as mock_is_eligible_for_design:
    mock_is_eligible_for_design.return_value = False
    simulated_data = experiment.simulate_experiments(
        suggested_design, simulations_per_trial=1
    )
    assert simulated_data is None


def test_evaluate_single_simulation_results_returns_correct_data_with_relative_effects(
    historical_data_lots_of_geos, default_design_spec, mock_trial
):
  experiment = geoflex.experiment.Experiment(
      name="test_experiment",
      historical_data=historical_data_lots_of_geos,
      design_spec=default_design_spec,
  )
  suggested_design = experiment.suggest_experiment_design(mock_trial)
  simulated_data = experiment.simulate_experiments(
      suggested_design, simulations_per_trial=1
  )

  # Revenue is a metric that should have relative effects
  simulated_data = simulated_data.loc[simulated_data["metric"] == "revenue"]

  evaluation_results = experiment.evaluate_single_simulation_results(
      simulated_data
  )

  assert isinstance(evaluation_results, pd.Series)
  assert evaluation_results.index.values.tolist() == [
      "avg_absolute_effect",
      "standard_error_absolute_effect",
      "coverage_absolute_effect",
      "absolute_effect_is_unbiased",
      "absolute_effect_has_coverage",
      "avg_relative_effect",
      "standard_error_relative_effect",
      "coverage_relative_effect",
      "relative_effect_is_unbiased",
      "relative_effect_has_coverage",
      "all_checks_pass",
      "failing_checks",
  ]
  # Check all numeric values are finite
  assert np.all(
      np.isfinite(
          evaluation_results[[
              "avg_absolute_effect",
              "standard_error_absolute_effect",
              "coverage_absolute_effect",
              "avg_relative_effect",
              "standard_error_relative_effect",
              "coverage_relative_effect",
          ]]
          .astype(float)
          .values
      )
  )


def test_evaluate_single_simulation_results_returns_correct_data_without_relative_effects(
    historical_data_lots_of_geos, default_design_spec, mock_trial
):
  experiment = geoflex.experiment.Experiment(
      name="test_experiment",
      historical_data=historical_data_lots_of_geos,
      design_spec=default_design_spec,
  )
  suggested_design = experiment.suggest_experiment_design(mock_trial)
  simulated_data = experiment.simulate_experiments(
      suggested_design, simulations_per_trial=1
  )

  # ROAS is a metric that should not have relative effects
  simulated_data = simulated_data.loc[simulated_data["metric"] == "ROAS"]

  evaluation_results = experiment.evaluate_single_simulation_results(
      simulated_data
  )

  assert isinstance(evaluation_results, pd.Series)
  assert evaluation_results.index.values.tolist() == [
      "avg_absolute_effect",
      "standard_error_absolute_effect",
      "coverage_absolute_effect",
      "absolute_effect_is_unbiased",
      "absolute_effect_has_coverage",
      "avg_relative_effect",
      "standard_error_relative_effect",
      "coverage_relative_effect",
      "relative_effect_is_unbiased",
      "relative_effect_has_coverage",
      "all_checks_pass",
      "failing_checks",
  ]
  # Check all absolute effect values are finite
  assert np.all(
      np.isfinite(
          evaluation_results[[
              "avg_absolute_effect",
              "standard_error_absolute_effect",
              "coverage_absolute_effect",
          ]]
          .astype(float)
          .values
      )
  )

  # Check all relative effect values are NA
  assert np.all((
      evaluation_results[[
          "avg_relative_effect",
          "standard_error_relative_effect",
          "coverage_relative_effect",
      ]].isna()
  ))


def test_explore_experiment_designs_records_design_results(
    historical_data_lots_of_geos, default_design_spec
):
  experiment = geoflex.experiment.Experiment(
      name="test_experiment",
      historical_data=historical_data_lots_of_geos,
      design_spec=default_design_spec,
  )
  experiment.explore_experiment_designs(
      max_trials=3, simulations_per_trial=5, n_jobs=1
  )

  assert experiment.all_raw_eval_metrics["design_id"].nunique() == 3


def test_get_all_design_summaries_returns_correct_data_relative_effects(
    historical_data_lots_of_geos, default_design_spec
):
  experiment = geoflex.experiment.Experiment(
      name="test_experiment",
      historical_data=historical_data_lots_of_geos,
      design_spec=default_design_spec,
  )
  experiment.explore_experiment_designs(
      max_trials=3, simulations_per_trial=5, n_jobs=1
  )
  design_summaries = experiment.get_all_design_summaries(
      target_power=0.8,
      target_primary_metric_mde=None,
      pareto_front_only=False,
      include_design_parameters=False,
      use_relative_effects_where_possible=True,
  )

  assert isinstance(design_summaries, pd.DataFrame)
  assert design_summaries.index.names == ["design_id"]
  assert design_summaries.dtypes.to_dict() == {
      "MDE (CPA)": "float64",
      "MDE (ROAS)": "float64",
      "Relative MDE (conversions)": "float64",
      "Relative MDE (revenue, primary metric)": "float64",
      "all_checks_pass": "bool",
      "failing_checks": "object",
      "primary_metric_all_checks_pass": "bool",
      "primary_metric_failing_checks": "object",
      "primary_metric_standard_error": "float64",
      "treatment_groups_representiveness_score": "float64",
  }
  assert len(design_summaries) == 3


def test_get_all_design_summaries_returns_correct_data_absolute_effects(
    historical_data_lots_of_geos, default_design_spec
):
  experiment = geoflex.experiment.Experiment(
      name="test_experiment",
      historical_data=historical_data_lots_of_geos,
      design_spec=default_design_spec,
  )
  experiment.explore_experiment_designs(
      max_trials=3, simulations_per_trial=5, n_jobs=1
  )
  design_summaries = experiment.get_all_design_summaries(
      target_power=0.8,
      target_primary_metric_mde=None,
      pareto_front_only=False,
      include_design_parameters=False,
      use_relative_effects_where_possible=False,
  )

  assert isinstance(design_summaries, pd.DataFrame)
  assert design_summaries.index.names == ["design_id"]
  assert design_summaries.dtypes.to_dict() == {
      "MDE (CPA)": "float64",
      "MDE (ROAS)": "float64",
      "MDE (conversions)": "float64",
      "MDE (revenue, primary metric)": "float64",
      "all_checks_pass": "bool",
      "failing_checks": "object",
      "primary_metric_all_checks_pass": "bool",
      "primary_metric_failing_checks": "object",
      "primary_metric_standard_error": "float64",
      "treatment_groups_representiveness_score": "float64",
  }
  assert len(design_summaries) == 3


def test_get_all_design_summaries_returns_correct_data_pareto_front_only(
    historical_data_lots_of_geos, default_design_spec
):
  experiment = geoflex.experiment.Experiment(
      name="test_experiment",
      historical_data=historical_data_lots_of_geos,
      design_spec=default_design_spec,
  )
  experiment.explore_experiment_designs(
      max_trials=3, simulations_per_trial=5, n_jobs=1
  )
  design_summaries = experiment.get_all_design_summaries(
      target_power=0.8,
      target_primary_metric_mde=None,
      pareto_front_only=True,
      include_design_parameters=False,
      use_relative_effects_where_possible=True,
  )

  assert isinstance(design_summaries, pd.DataFrame)
  assert design_summaries.index.names == ["design_id"]
  assert design_summaries.dtypes.to_dict() == {
      "MDE (CPA)": "float64",
      "MDE (ROAS)": "float64",
      "Relative MDE (conversions)": "float64",
      "Relative MDE (revenue, primary metric)": "float64",
      "all_checks_pass": "bool",
      "failing_checks": "object",
      "primary_metric_all_checks_pass": "bool",
      "primary_metric_failing_checks": "object",
      "primary_metric_standard_error": "float64",
      "treatment_groups_representiveness_score": "float64",
  }
  assert (
      len(design_summaries) == 1
  )  # Only one design is on the Pareto front in the test data.


def test_get_all_design_summaries_returns_correct_data_with_design_parameters(
    historical_data_lots_of_geos, default_design_spec
):
  experiment = geoflex.experiment.Experiment(
      name="test_experiment",
      historical_data=historical_data_lots_of_geos,
      design_spec=default_design_spec,
  )
  experiment.explore_experiment_designs(
      max_trials=3, simulations_per_trial=5, n_jobs=1
  )
  design_summaries = experiment.get_all_design_summaries(
      target_power=0.8,
      target_primary_metric_mde=None,
      pareto_front_only=False,
      include_design_parameters=True,
      use_relative_effects_where_possible=True,
  )

  assert isinstance(design_summaries, pd.DataFrame)
  assert design_summaries.index.names == ["design_id"]
  assert design_summaries.dtypes.to_dict() == {
      "n_geos_control": "int64",
      "n_geos_exclude": "int64",
      "n_geos_treatment_0": "int64",
      "n_geos_treatment_1": "int64",
      "experiment_budget": "object",
      "methodology": "object",
      "runtime_weeks": "int64",
      "methodology_parameters": "object",
      "random_seed": "int64",
      "MDE (CPA)": "float64",
      "MDE (ROAS)": "float64",
      "Relative MDE (conversions)": "float64",
      "Relative MDE (revenue, primary metric)": "float64",
      "all_checks_pass": "bool",
      "failing_checks": "object",
      "primary_metric_all_checks_pass": "bool",
      "primary_metric_failing_checks": "object",
      "primary_metric_standard_error": "float64",
      "treatment_groups_representiveness_score": "float64",
  }
  assert len(design_summaries) == 3


def test_count_all_eligible_designs_returns_correct_data(
    historical_data_lots_of_geos, default_design_spec
):
  experiment = geoflex.experiment.Experiment(
      name="test_experiment",
      historical_data=historical_data_lots_of_geos,
      design_spec=default_design_spec,
  )
  counts = experiment.count_all_eligible_designs()
  assert counts == {"RCT": 4}
