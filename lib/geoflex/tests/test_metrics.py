"""Tests for the metrics module."""

import geoflex.metrics
import pytest

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


def test_can_instantiate_metric():
  """Tests that geoflex can be imported."""
  metric = geoflex.metrics.Metric(name="test_metric", column="test_column")
  assert metric.name == "test_metric"
  assert metric.column == "test_column"
  assert not metric.metric_per_cost
  assert not metric.cost_per_metric


def test_metric_column_defaults_to_name():
  """Tests that the metric column defaults to the name."""
  metric = geoflex.metrics.Metric(name="test_metric")
  assert metric.name == "test_metric"
  assert metric.column == "test_metric"


def test_roas_metric_has_correct_defaults():
  """Tests that the ROAS metric has the correct defaults."""
  roas = geoflex.metrics.ROAS()
  assert roas.name == "ROAS"
  assert roas.column == "revenue"
  assert roas.metric_per_cost
  assert not roas.cost_per_metric


def test_cpa_metric_has_correct_defaults():
  """Tests that the CPA metric has the correct defaults."""
  cpa = geoflex.metrics.CPA()
  assert cpa.name == "CPA"
  assert cpa.column == "conversions"
  assert not cpa.metric_per_cost
  assert cpa.cost_per_metric


@pytest.mark.parametrize(
    "metric_cls",
    [geoflex.metrics.ROAS, geoflex.metrics.CPA],
)
@pytest.mark.parametrize(
    "invalid_args",
    [
        {"name": "another_name"},
        {"metric_per_cost": True},
        {"cost_per_metric": True},
    ],
)
def test_can_only_set_column_for_roas_and_cpa(metric_cls, invalid_args):
  """Tests that the column can only be set for ROAS and CPA."""
  metric_cls(column="my_column")  # Should not raise an error.
  with pytest.raises(TypeError):
    metric_cls(
        column="my_column", **invalid_args
    )  # Should not raise an error.
