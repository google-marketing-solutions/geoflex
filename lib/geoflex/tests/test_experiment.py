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
      max_runtime_weeks=4,
      min_runtime_weeks=2,
      n_cells=3,
      geo_eligibility_candidates=[
          None,
          GeoEligibility(control=["geo_1"], treatment=[[], []], exclude=[]),
          GeoEligibility(control=[], treatment=[[], []], exclude=["geo_1"]),
      ],
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
  assert suggested_design.runtime_weeks in range(
      default_design_spec.min_runtime_weeks,
      default_design_spec.max_runtime_weeks + 1,
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
