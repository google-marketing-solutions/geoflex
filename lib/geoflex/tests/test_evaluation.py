"""Tests for the evaluation module."""

import geoflex.data
import geoflex.evaluation
import geoflex.experiment_design
import mock
import numpy as np
import pandas as pd
import pytest

ExperimentDesignEvaluator = geoflex.evaluation.ExperimentDesignEvaluator
CellVolumeConstraint = geoflex.experiment_design.CellVolumeConstraint
CellVolumeConstraintType = geoflex.experiment_design.CellVolumeConstraintType
GeoAssignment = geoflex.experiment_design.GeoAssignment

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
      methodology="TestingMethodology",
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
  raw_data_pivoted = geoflex.data.GeoPerformanceDataset(
      data=raw_data
  ).pivoted_data
  scorer = geoflex.evaluation.GeoAssignmentRepresentativenessScorer(
      historical_data=raw_data_pivoted,
      geos=["US", "UK", "AU", "NL"],
  )
  result = scorer(np.array([0, 1, 0, 1]))
  assert result <= 1.0
  assert result >= -1.0


def test_scorer_returns_higher_score_for_representative_assignment(raw_data):
  raw_data_pivoted = geoflex.data.GeoPerformanceDataset(
      data=raw_data
  ).pivoted_data
  scorer = geoflex.evaluation.GeoAssignmentRepresentativenessScorer(
      historical_data=raw_data_pivoted,
      geos=["US", "UK", "AU", "NL"],
  )
  result_1 = scorer(np.array([0, 1, 0, 1]))
  result_2 = scorer(np.array([0, 0, 1, 1]))
  assert result_1 > result_2


def test_scorer_can_handle_constant_column(raw_data):
  raw_data["revenue"] = 0.0
  raw_data_pivoted = geoflex.data.GeoPerformanceDataset(
      data=raw_data
  ).pivoted_data
  scorer = geoflex.evaluation.GeoAssignmentRepresentativenessScorer(
      historical_data=raw_data_pivoted,
      geos=["US", "UK", "AU", "NL"],
  )
  result = scorer(np.array([0, 1, 0, 1]))
  assert isinstance(result, float)


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
  assert evaluator.bootstrapper.sampling_type == "permutation"
  assert evaluator.bootstrapper.seasonality == 7
  assert evaluator.bootstrapper.seasons_per_block == 4
  assert evaluator.bootstrapper.model_type == "multiplicative"
  assert evaluator.bootstrapper.stl_params == {
      "seasonal_deg": 0,
      "trend_deg": 0,
      "low_pass_deg": 0,
      "robust": False,
  }


def test_evaluator_representativeness_scorer_is_initialized_correctly(
    historical_data,
):
  explorer = ExperimentDesignEvaluator(
      historical_data=historical_data,
  )
  assert explorer.representativeness_scorer.historical_data.equals(
      historical_data.pivoted_data
  )
  assert explorer.representativeness_scorer.geos == historical_data.geos


def test_evaluator_representativeness_scorer_is_initialized_correctly_with_metrics(
    historical_data,
):
  explorer = ExperimentDesignEvaluator(
      historical_data=historical_data,
      representativeness_scorer_metrics=["revenue", "cost"],
  )
  assert explorer.representativeness_scorer.historical_data.equals(
      historical_data.pivoted_data[["revenue", "cost"]]
  )
  assert explorer.representativeness_scorer.geos == historical_data.geos


def test_evaluator_evaluate_design_raw_results_have_correct_data(
    historical_data, mock_design
):
  evaluator = ExperimentDesignEvaluator(
      historical_data=historical_data,
  )
  _ = evaluator.evaluate_design(
      mock_design, n_aa_simulations=3, n_ab_simulations=3
  )
  raw_results = evaluator.raw_simulation_results[mock_design.design_id]

  assert raw_results.is_valid_design
  assert raw_results.design == mock_design

  assert isinstance(raw_results.representativeness_scores, list)
  assert len(raw_results.representativeness_scores) == 2
  assert isinstance(raw_results.representativeness_scores[0], float)
  assert isinstance(raw_results.representativeness_scores[1], float)

  assert isinstance(raw_results.aa_simulation_results, pd.DataFrame)
  assert isinstance(raw_results.ab_simulation_results, pd.DataFrame)
  assert (
      len(raw_results.aa_simulation_results) == 24
  )  # 4 metrics, 2 treatment arms, 3 simulations
  assert raw_results.aa_simulation_results.dtypes.to_dict() == {
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
      "true_point_estimate": "float64",
      "true_point_estimate_relative": "float64",
      "sample_id": "object",
  }
  assert (
      len(raw_results.ab_simulation_results) == 6
  )  # Only primary metric for a/b simulation), 2 treatment arms, 3 simulations
  assert raw_results.ab_simulation_results.dtypes.to_dict() == {
      "cell": "int64",
      "metric": "object",
      "is_primary_metric": "bool",
      "point_estimate": "float64",
      "lower_bound": "float64",
      "upper_bound": "float64",
      "point_estimate_relative": "float64",
      "lower_bound_relative": "float64",
      "upper_bound_relative": "float64",
      "p_value": "float64",
      "is_significant": "bool",
      "design_id": "object",
      "true_point_estimate": "float64",
      "true_point_estimate_relative": "float64",
      "sample_id": "object",
  }


def test_evaluator_evaluate_design_raw_results_have_correct_data_if_no_ab_simulations(
    historical_data, mock_design
):
  evaluator = ExperimentDesignEvaluator(
      historical_data=historical_data,
  )
  _ = evaluator.evaluate_design(
      mock_design, n_aa_simulations=3, n_ab_simulations=0
  )
  raw_results = evaluator.raw_simulation_results[mock_design.design_id]

  assert raw_results.is_valid_design
  assert raw_results.design == mock_design

  assert isinstance(raw_results.representativeness_scores, list)
  assert len(raw_results.representativeness_scores) == 2
  assert isinstance(raw_results.representativeness_scores[0], float)
  assert isinstance(raw_results.representativeness_scores[1], float)

  assert isinstance(raw_results.aa_simulation_results, pd.DataFrame)
  assert isinstance(raw_results.ab_simulation_results, pd.DataFrame)
  assert (
      len(raw_results.aa_simulation_results) == 24
  )  # 4 metrics, 2 treatment arms, 3 simulations
  assert raw_results.aa_simulation_results.dtypes.to_dict() == {
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
      "true_point_estimate": "float64",
      "true_point_estimate_relative": "float64",
      "sample_id": "object",
  }
  assert raw_results.ab_simulation_results.empty


def test_evaluator_evaluate_design_extends_results_correctly(
    historical_data, mock_design
):
  evaluator = ExperimentDesignEvaluator(
      historical_data=historical_data,
  )
  _ = evaluator.evaluate_design(
      mock_design, n_aa_simulations=3, n_ab_simulations=3
  )
  initial_raw_results = evaluator.raw_simulation_results[mock_design.design_id]
  _ = evaluator.evaluate_design(
      mock_design,
      n_aa_simulations=3,
      n_ab_simulations=3,
      overwrite_mode="extend",
  )
  extended_raw_results = evaluator.raw_simulation_results[mock_design.design_id]

  # Check that the new results are double the initial results, because we
  # extended the results by 3 simulations.
  assert (
      len(extended_raw_results.aa_simulation_results)
      == len(initial_raw_results.aa_simulation_results) * 2
  )
  assert (
      len(extended_raw_results.ab_simulation_results)
      == len(initial_raw_results.ab_simulation_results) * 2
  )


def test_evaluator_evaluate_design_skips_results_correctly(
    historical_data, mock_design
):
  evaluator = ExperimentDesignEvaluator(
      historical_data=historical_data,
  )
  _ = evaluator.evaluate_design(
      mock_design, n_aa_simulations=3, n_ab_simulations=3
  )
  initial_raw_results = evaluator.raw_simulation_results[mock_design.design_id]
  _ = evaluator.evaluate_design(
      mock_design,
      n_aa_simulations=3,
      n_ab_simulations=3,
      overwrite_mode="skip",
  )
  new_raw_results = evaluator.raw_simulation_results[mock_design.design_id]

  assert (
      new_raw_results.aa_simulation_results
      is initial_raw_results.aa_simulation_results
  )  # Skipped, so no change.
  assert (
      new_raw_results.ab_simulation_results
      is initial_raw_results.ab_simulation_results
  )  # Skipped, so no change.


def test_evaluator_overwrites_results_if_overwrite_mode_is_overwrite(
    historical_data, mock_design
):
  evaluator = ExperimentDesignEvaluator(
      historical_data=historical_data,
  )
  _ = evaluator.evaluate_design(
      mock_design, n_aa_simulations=3, n_ab_simulations=3
  )
  initial_raw_results = evaluator.raw_simulation_results[mock_design.design_id]
  _ = evaluator.evaluate_design(
      mock_design,
      n_aa_simulations=3,
      n_ab_simulations=3,
      overwrite_mode="overwrite",
  )
  overwritten_raw_results = evaluator.raw_simulation_results[
      mock_design.design_id
  ]

  assert (
      overwritten_raw_results.aa_simulation_results
      is not initial_raw_results.aa_simulation_results
  )
  assert (
      overwritten_raw_results.ab_simulation_results
      is not initial_raw_results.ab_simulation_results
  )
  assert len(overwritten_raw_results.aa_simulation_results) == 24
  assert len(overwritten_raw_results.ab_simulation_results) == 6


def test_evaluator_evaluate_design_overwrites_results_if_raw_results_are_not_available(
    historical_data, mock_design
):
  evaluator = ExperimentDesignEvaluator(
      historical_data=historical_data,
  )
  _ = evaluator.evaluate_design(
      mock_design, n_aa_simulations=5, n_ab_simulations=5
  )
  # Clear the raw simulation results.
  del evaluator.raw_simulation_results[mock_design.design_id]

  evaluator.raw_simulation_results = {}
  _ = evaluator.evaluate_design(
      mock_design,
      n_aa_simulations=3,
      n_ab_simulations=3,
      overwrite_mode="extend",
  )
  overwritten_raw_results = evaluator.raw_simulation_results[
      mock_design.design_id
  ]

  # Expected results are 24 because the design has 2 treatment cells, 4 metrics,
  # and we ran 3 simulations for each, and the results were overwritten.
  assert len(overwritten_raw_results.aa_simulation_results) == 24
  # A/B results should be 6 because they only exist for the primary metric, and
  # we ran 3 simulations for 2 cells.
  assert len(overwritten_raw_results.ab_simulation_results) == 6


def test_evaluator_evaluate_design_for_invalid_design(
    historical_data, mock_design
):
  evaluator = ExperimentDesignEvaluator(
      historical_data=historical_data,
  )

  # Mock the TestingMethodology methodology to always say invalid design.
  with mock.patch.object(
      geoflex.methodology.testing_methodology.TestingMethodology,
      "is_eligible_for_design_and_data",
      autospec=True,
  ) as mock_is_eligible_for_design_and_data:
    mock_is_eligible_for_design_and_data.return_value = False
    results = evaluator.evaluate_design(
        mock_design, n_aa_simulations=5, n_ab_simulations=5
    )

  raw_results = evaluator.raw_simulation_results[mock_design.design_id]

  assert not results.is_valid_design
  assert results.representativeness_scores_per_cell is None
  assert raw_results.aa_simulation_results.empty
  assert raw_results.ab_simulation_results.empty
  assert raw_results.design == mock_design


def test_raw_experiment_simulation_results_raises_error_if_missing_results(
    mock_design,
):
  # Mock design has 2 treatment cells, so we need 2 simulation results.
  with pytest.raises(ValueError):
    geoflex.evaluation.RawExperimentSimulationResults(
        design=mock_design,
        aa_simulation_results=pd.DataFrame({"cell": [1]}),
        ab_simulation_results=pd.DataFrame(),
        representativeness_scores=[1, 2],
        is_valid_design=True,
        sufficient_simulations=True,
    )


def test_raw_experiment_simulation_results_does_not_raise_error_if_all_results_exist(
    mock_design,
):
  # Mock design has 2 treatment cells, so we need 2 simulation results.
  geoflex.evaluation.RawExperimentSimulationResults(
      design=mock_design,
      aa_simulation_results=pd.DataFrame({"cell": [1, 2]}),
      ab_simulation_results=pd.DataFrame(),
      representativeness_scores=[1, 2],
      is_valid_design=True,
      sufficient_simulations=True,
  )


def test_raw_experiment_simulation_results_raises_error_if_missing_representativeness_scores(
    mock_design,
):
  # Mock design has 2 treatment cells, so we need 2 representativeness scores.
  with pytest.raises(ValueError):
    geoflex.evaluation.RawExperimentSimulationResults(
        design=mock_design,
        aa_simulation_results=pd.DataFrame({"cell": [1, 2]}),
        ab_simulation_results=pd.DataFrame(),
        representativeness_scores=[1],
        is_valid_design=True,
        sufficient_simulations=True,
    )


@pytest.mark.parametrize(
    "methodology", ["TestingMethodology", "PseudoExperimentTestingMethodology"]
)
def test_evaluator_evaluate_design_returns_correct_data_and_adds_results_to_design(
    historical_data, mock_design, methodology
):
  evaluator = ExperimentDesignEvaluator(
      historical_data=historical_data,
  )

  mock_design = mock_design.make_variation(methodology=methodology)
  evaluation_results = evaluator.evaluate_design(
      mock_design, n_aa_simulations=5, n_ab_simulations=5
  )

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
  assert len(evaluation_results.representativeness_scores_per_cell) == 2
  for metric_results in evaluation_results.all_metric_results_per_cell.values():
    assert len(metric_results) == 2


def test_evaluator_evaluate_design_correctly_sets_automatic_sample_size(
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
  assert len(evaluation_results.representativeness_scores_per_cell) == 2
  for metric_results in evaluation_results.all_metric_results_per_cell.values():
    assert len(metric_results) == 2


def test_evaluator_evaluate_design_returns_correct_data_for_invalid_design(
    historical_data, mock_design
):
  evaluator = ExperimentDesignEvaluator(
      historical_data=historical_data,
  )
  # Mock is_valid_for_design_and_data to always return False.
  with mock.patch.object(
      geoflex.methodology.testing_methodology.TestingMethodology,
      "is_eligible_for_design_and_data",
      autospec=True,
  ) as mock_is_eligible_for_design_and_data:
    mock_is_eligible_for_design_and_data.return_value = False
    evaluation_results = evaluator.evaluate_design(
        mock_design, n_aa_simulations=5, n_ab_simulations=5
    )

  assert isinstance(
      evaluation_results, geoflex.evaluation.ExperimentDesignEvaluationResults
  )
  assert mock_design.evaluation_results is not None
  assert mock_design.evaluation_results == evaluation_results

  assert not evaluation_results.is_valid_design
  assert (
      evaluation_results.primary_metric_name == mock_design.primary_metric.name
  )
  assert evaluation_results.alpha == mock_design.alpha
  assert (
      evaluation_results.alternative_hypothesis
      == mock_design.alternative_hypothesis
  )

  assert evaluation_results.representativeness_scores_per_cell is None

  # Results are None because the design is invalid.
  assert evaluation_results.all_metric_results_per_cell is None
  assert evaluation_results.all_metric_results is None
  assert evaluation_results.primary_metric_results is None
  assert evaluation_results.primary_metric_results_per_cell is None


def test_evaluator_evaluate_design_does_not_reassign_geos_for_pseudo_experiments(
    historical_data, mock_design
):
  evaluator = ExperimentDesignEvaluator(
      historical_data=historical_data,
  )
  mock_design = mock_design.make_variation(
      methodology="PseudoExperimentTestingMethodology"
  )

  geoflex.methodology.assign_geos(mock_design, historical_data)

  # check that assign geos is not called
  with mock.patch.object(
      geoflex.methodology.Methodology,
      "assign_geos",
      autospec=True,
  ) as mock_assign_geos:
    mock_assign_geos.return_value = None, None
    evaluator.evaluate_design(
        mock_design, n_aa_simulations=5, n_ab_simulations=5
    )

  mock_assign_geos.assert_not_called()


@pytest.fixture(name="historical_data_with_fixed_values")
def historical_data_with_fixed_values_fixture():
  """Fixture for a mock historical data with fixed values."""
  return geoflex.data.GeoPerformanceDataset(
      data=pd.DataFrame({
          "geo_id": ["geo_1", "geo_2", "geo_3", "geo_4"] * 2,
          "date": ["2024-01-01"] * 4 + ["2024-01-02"] * 4,
          "clicks": np.linspace(100, 800, 8),
          "cost": np.linspace(20, 27, 8),
          "revenue": np.linspace(50, 36, 8),
      })
  )


@pytest.mark.parametrize(
    "constraint, geo_assignment, expected_values,expected_is_valid,"
    " expected_error_message",
    [
        (  # No cell volume constraint
            CellVolumeConstraint(
                values=[None, None],
                constraint_type=CellVolumeConstraintType.MAX_GEOS,
            ),
            GeoAssignment(
                control=["geo_1", "geo_2"], treatment=[["geo_3", "geo_4"]]
            ),
            None,
            True,
            "",
        ),
        (  # Good max cells constraint
            CellVolumeConstraint(
                values=[2, 2],
                constraint_type=CellVolumeConstraintType.MAX_GEOS,
            ),
            GeoAssignment(control=["geo_1"], treatment=[["geo_3", "geo_4"]]),
            [1, 2],
            True,
            "",
        ),
        (  # Good max revenue constraint
            CellVolumeConstraint(
                values=[0.5, 0.5],
                constraint_type=CellVolumeConstraintType.MAX_PERCENTAGE_OF_METRIC,
                metric_column="revenue",
            ),
            GeoAssignment(control=["geo_1"], treatment=[["geo_3", "geo_4"]]),
            [0.26744186046511625, 0.47674418604651164],
            True,
            "",
        ),
        (  # Too many geos in control
            CellVolumeConstraint(
                values=[1, None],
                constraint_type=CellVolumeConstraintType.MAX_GEOS,
            ),
            GeoAssignment(
                control=["geo_1", "geo_2", "geo_3"], treatment=[["geo_4"]]
            ),
            [3, 1],
            False,
            (
                "Cell volume constraint is not respected. Target = control: 1"
                " geos, treatment_1: No Constraint, Actual = control: 3 geos,"
                " treatment_1: 1 geos."
            ),
        ),
        (  # Too many geos in treatment
            CellVolumeConstraint(
                values=[None, 1],
                constraint_type=CellVolumeConstraintType.MAX_GEOS,
            ),
            GeoAssignment(
                control=["geo_1"], treatment=[["geo_2", "geo_3", "geo_4"]]
            ),
            [1, 3],
            False,
            (
                "Cell volume constraint is not respected. Target = control:"
                " No Constraint, treatment_1: 1 geos, Actual = control: 1 geos,"
                " treatment_1: 3 geos."
            ),
        ),
        (  # Too much revenue in control
            CellVolumeConstraint(
                values=[0.1, None],
                constraint_type=CellVolumeConstraintType.MAX_PERCENTAGE_OF_METRIC,
                metric_column="revenue",
            ),
            GeoAssignment(
                control=["geo_1", "geo_2"], treatment=[["geo_3", "geo_4"]]
            ),
            [0.5232558139534884, 0.47674418604651164],
            False,
            (
                "Cell volume constraint is not respected. Target = control:"
                " 10.0% of revenue, treatment_1: No Constraint, Actual ="
                " control: 52.3% of revenue, treatment_1: 47.7% of revenue."
            ),
        ),
        (  # Too much revenue in treatment
            CellVolumeConstraint(
                values=[None, 0.1],
                constraint_type=CellVolumeConstraintType.MAX_PERCENTAGE_OF_METRIC,
                metric_column="revenue",
            ),
            GeoAssignment(
                control=["geo_1"], treatment=[["geo_2", "geo_3", "geo_4"]]
            ),
            [0.26744186046511625, 0.7325581395348837],
            False,
            (
                "Cell volume constraint is not respected. Target = control: No"
                " Constraint, treatment_1: 10.0% of revenue, Actual = control:"
                " 26.7% of revenue, treatment_1: 73.3% of revenue."
            ),
        ),
    ],
)
def test_validate_cell_volume_constraint_is_respected_returns_expected_results(
    constraint,
    geo_assignment,
    historical_data_with_fixed_values,
    expected_values,
    expected_is_valid,
    expected_error_message,
):
  actual_cell_volumes, actual_is_valid, actual_error_message = (
      geoflex.evaluation.validate_cell_volume_constraint_is_respected(
          geo_assignment=geo_assignment,
          cell_volume_constraint=constraint,
          historical_data=historical_data_with_fixed_values,
      )
  )

  expected_cell_volumes = None
  if expected_values is not None:
    expected_cell_volumes = CellVolumeConstraint(
        constraint_type=constraint.constraint_type,
        metric_column=constraint.metric_column,
        values=expected_values,
    )

  assert actual_cell_volumes == expected_cell_volumes
  assert actual_is_valid == expected_is_valid
  assert actual_error_message == expected_error_message
