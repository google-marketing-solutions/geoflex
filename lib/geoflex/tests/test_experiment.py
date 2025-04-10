"""Tests for the experiment module."""

import geoflex.experiment
import mock
import pandas as pd

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
