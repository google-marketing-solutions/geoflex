"""Tests for the evaluation module."""

import geoflex.data
import geoflex.evaluation
import geoflex.experiment_design
import mock
import numpy as np
import pandas as pd
import pytest

ExperimentDesignEvaluator = geoflex.evaluation.ExperimentDesignEvaluator

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name
# pylint: disable=g-doc-return-or-yield


@pytest.fixture(name="mock_design")
def mock_design_fixture():
  """Fixture for a mock design."""
  return geoflex.experiment_design.ExperimentDesign(
      primary_metric="revenue",
      secondary_metrics=[
          "conversions",
          geoflex.metrics.iROAS(),
          geoflex.metrics.CPiA(),
      ],
      experiment_budget=geoflex.experiment_design.ExperimentBudget(
          value=-0.1,
          budget_type=(
              geoflex.experiment_design.ExperimentBudgetType.PERCENTAGE_CHANGE
          ),
      ),
      methodology="RCT",
      runtime_weeks=4,
      n_cells=3,
      alpha=0.1,
      geo_eligibility=None,
  )


@pytest.fixture(name="raw_data")
def raw_data_fixture():
  """Fixture for test data.

  Metrics are correlated between UK and US, and AU and NL.
  Therefore assignment should be good if UK and US are not together, and AU
  and NL are not together.
  """
  rng = np.random.default_rng(seed=42)

  UK_US_correlated_clicks = rng.uniform(size=100)
  UK_US_correlated_cost = rng.uniform(size=100)
  UK_US_correlated_revenue = rng.uniform(size=100)
  AU_NL_correlated_clicks = rng.uniform(size=100)
  AU_NL_correlated_cost = rng.uniform(size=100)
  AU_NL_correlated_revenue = rng.uniform(size=100)

  clicks = (
      np.concatenate([
          UK_US_correlated_clicks,
          UK_US_correlated_clicks,
          AU_NL_correlated_clicks,
          AU_NL_correlated_clicks,
      ])
      + rng.uniform(size=400) * 0.1
  )
  cost = (
      np.concatenate([
          UK_US_correlated_cost,
          UK_US_correlated_cost,
          AU_NL_correlated_cost,
          AU_NL_correlated_cost,
      ])
      + rng.uniform(size=400) * 0.1
  )
  revenue = (
      np.concatenate([
          UK_US_correlated_revenue,
          UK_US_correlated_revenue,
          AU_NL_correlated_revenue,
          AU_NL_correlated_revenue,
      ])
      + rng.uniform(size=400) * 0.1
  )

  return pd.DataFrame({
      "geo_id": ["US"] * 100 + ["UK"] * 100 + ["AU"] * 100 + ["NL"] * 100,
      "date": pd.date_range(start="2024-01-01", periods=100).tolist() * 4,
      "clicks": clicks,
      "cost": cost,
      "revenue": revenue,
  })


@pytest.fixture(name="historical_data")
def mock_historical_data_fixture():
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


def test_scorer_returns_score_in_correct_range(raw_data):
  scorer = geoflex.evaluation.GeoAssignmentRepresentivenessScorer(
      historical_data=raw_data,
      geo_column_name="geo_id",
      geos=["US", "UK", "AU", "NL"],
  )
  result, _ = scorer(np.array([0, 1, 0, 1]))
  assert result <= 1.0
  assert result >= -1.0


def test_scorer_returns_pvalue_in_correct_range_if_requested(raw_data):
  scorer = geoflex.evaluation.GeoAssignmentRepresentivenessScorer(
      historical_data=raw_data,
      geo_column_name="geo_id",
      geos=["US", "UK", "AU", "NL"],
  )
  _, pvalue = scorer(np.array([0, 1, 0, 1]), with_pvalue=True)
  assert pvalue <= 1.0
  assert pvalue >= 0.0


def test_scorer_returns_none_for_pvalue_if_not_requested(raw_data):
  scorer = geoflex.evaluation.GeoAssignmentRepresentivenessScorer(
      historical_data=raw_data,
      geo_column_name="geo_id",
      geos=["US", "UK", "AU", "NL"],
  )
  _, pvalue = scorer(np.array([0, 1, 0, 1]))
  assert pvalue is None


def test_scorer_returns_higher_score_for_representative_assignment(raw_data):
  scorer = geoflex.evaluation.GeoAssignmentRepresentivenessScorer(
      historical_data=raw_data,
      geo_column_name="geo_id",
      geos=["US", "UK", "AU", "NL"],
  )
  result_1, _ = scorer(np.array([0, 1, 0, 1]))
  result_2, _ = scorer(np.array([0, 0, 1, 1]))
  assert result_1 > result_2


def test_scorer_can_handle_assignment_with_multiple_treatment_groups(raw_data):
  scorer = geoflex.evaluation.GeoAssignmentRepresentivenessScorer(
      historical_data=raw_data,
      geo_column_name="geo_id",
      geos=["US", "UK", "AU", "NL"],
  )
  result, _ = scorer(np.array([0, 1, 1, 2]))
  assert result <= 1.0
  assert result >= -1.0


def test_scorer_can_handle_assignment_with_excluded_geos(raw_data):
  scorer = geoflex.evaluation.GeoAssignmentRepresentivenessScorer(
      historical_data=raw_data,
      geo_column_name="geo_id",
      geos=["US", "UK", "AU", "NL"],
  )
  result, _ = scorer(np.array([-1, 0, 1, 1]))
  assert result <= 1.0
  assert result >= -1.0


def test_calculate_minimum_detectable_effect_from_stats_raises_error_for_invalid_alternative():
  with pytest.raises(ValueError):
    geoflex.evaluation.calculate_minimum_detectable_effect_from_stats(
        standard_error=1.0,
        alternative="invalid_alternative",
    )


@pytest.mark.parametrize(
    "standard_error,alternative,power,alpha,expected_result",
    [
        (1.0, "two-sided", 0.8, 0.05, 2.801585218),
        (1.0, "greater", 0.8, 0.05, 2.48647486),
        (1.0, "less", 0.8, 0.05, 2.48647486),
        (1.0, "two-sided", 0.9, 0.05, 3.24151555),
        (1.0, "greater", 0.9, 0.05, 2.92640519),
        (1.0, "less", 0.9, 0.05, 2.92640519),
        (1.0, "two-sided", 0.8, 0.1, 2.486474860),
        (1.0, "greater", 0.8, 0.1, 2.123172799),
        (1.0, "less", 0.8, 0.1, 2.123172799),
        (2.0, "two-sided", 0.8, 0.05, 5.603170436),
        (2.0, "greater", 0.8, 0.05, 4.972949721),
        (2.0, "less", 0.8, 0.05, 4.972949721),
    ],
)
def test_calculate_minimum_detectable_effect_from_stats_returns_expected_values(
    standard_error, alternative, power, alpha, expected_result
):
  result = geoflex.evaluation.calculate_minimum_detectable_effect_from_stats(
      standard_error=standard_error,
      alternative=alternative,
      power=power,
      alpha=alpha,
  )
  assert np.isclose(result, expected_result)


def test_evaluator_bootstrapper_is_initialized_correctly(historical_data):
  evaluator = ExperimentDesignEvaluator(
      historical_data=historical_data,
  )
  assert evaluator.bootstrapper.log_transform
  assert evaluator.bootstrapper.seasonality == 7
  assert evaluator.bootstrapper.seasons_per_block == 2


def test_evaluator_representativeness_scorer_is_initialized_correctly(
    historical_data,
):
  explorer = ExperimentDesignEvaluator(
      historical_data=historical_data,
  )
  assert explorer.representativeness_scorer.historical_data.equals(
      historical_data.parsed_data
  )
  assert (
      explorer.representativeness_scorer.geo_column_name
      == historical_data.geo_id_column
  )
  assert explorer.representativeness_scorer.geos == historical_data.geos


def test_evaluator_simulate_experiment_results_returns_correct_data(
    historical_data, mock_design
):
  evaluator = ExperimentDesignEvaluator(
      historical_data=historical_data,
  )
  raw_results = evaluator.simulate_experiment_results(
      mock_design, n_simulations=3
  )

  assert raw_results.design_is_valid
  assert raw_results.design == mock_design

  assert isinstance(raw_results.representiveness_scores, list)
  assert len(raw_results.representiveness_scores) == 2
  assert isinstance(raw_results.representiveness_scores[0], float)
  assert isinstance(raw_results.representiveness_scores[1], float)

  assert isinstance(raw_results.simulation_results, pd.DataFrame)
  assert (
      len(raw_results.simulation_results) == 24
  )  # 4 metrics, 2 treatment arms, 3 simulations
  assert raw_results.simulation_results.dtypes.to_dict() == {
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
      "design_id": "object",
  }


def test_evaluator_simulate_experiment_results_for_invalid_design(
    historical_data, mock_design
):
  evaluator = ExperimentDesignEvaluator(
      historical_data=historical_data,
  )

  # Mock the RCT methodology to always say invalid design.
  with mock.patch.object(
      geoflex.methodology.rct.RCT, "is_eligible_for_design", autospec=True
  ) as mock_is_eligible_for_design:
    mock_is_eligible_for_design.return_value = False
    raw_results = evaluator.simulate_experiment_results(
        mock_design, n_simulations=3
    )

  assert not raw_results.design_is_valid
  assert raw_results.simulation_results.empty
  assert raw_results.representiveness_scores is None
  assert raw_results.design == mock_design


def test_raw_experiment_simulation_results_raises_error_if_missing_results(
    mock_design,
):
  # Mock design has 2 treatment cells, so we need 2 simulation results.
  with pytest.raises(ValueError):
    geoflex.evaluation.RawExperimentSimulationResults(
        design=mock_design,
        simulation_results=pd.DataFrame({"cell": [1]}),
        representiveness_scores=[1, 2],
        design_is_valid=True,
    )


def test_raw_experiment_simulation_results_does_not_raise_error_if_all_results_exist(
    mock_design,
):
  # Mock design has 2 treatment cells, so we need 2 simulation results.
  geoflex.evaluation.RawExperimentSimulationResults(
      design=mock_design,
      simulation_results=pd.DataFrame({"cell": [1, 2]}),
      representiveness_scores=[1, 2],
      design_is_valid=True,
  )


def test_raw_experiment_simulation_results_raises_error_if_missing_representiveness_scores(
    mock_design,
):
  # Mock design has 2 treatment cells, so we need 2 representiveness scores.
  with pytest.raises(ValueError):
    geoflex.evaluation.RawExperimentSimulationResults(
        design=mock_design,
        simulation_results=pd.DataFrame({"cell": [1, 2]}),
        representiveness_scores=[1],
        design_is_valid=True,
    )


def test_evaluator_evaluate_design_returns_correct_data_and_adds_results_to_design(
    historical_data, mock_design
):
  evaluator = ExperimentDesignEvaluator(
      historical_data=historical_data,
  )
  evaluation_results = evaluator.evaluate_design(mock_design)

  assert isinstance(
      evaluation_results, geoflex.evaluation.ExperimentDesignEvaluationResults
  )
  assert mock_design.evaluation_results is not None
  assert mock_design.evaluation_results == evaluation_results

  assert evaluation_results.is_valid_design
  assert (
      evaluation_results.primary_metric_name == mock_design.primary_metric.name
  )
  assert evaluation_results.alpha == mock_design.alpha
  assert (
      evaluation_results.alternative_hypothesis
      == mock_design.alternative_hypothesis
  )

  # Check number of cells matches design.
  assert len(evaluation_results.representiveness_scores_per_cell) == 2
  for metric_results in evaluation_results.all_metric_results_per_cell.values():
    assert len(metric_results) == 2
