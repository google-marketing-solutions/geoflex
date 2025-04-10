"""Tests for the metrics module."""

import geoflex.metrics
import pytest

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


def test_can_instantiate_metric():
  """Tests that a metric can be instantiated."""
  metric = geoflex.metrics.Metric(name="test_metric", column="test_column")
  assert metric.name == "test_metric"
  assert metric.column == "test_column"
  assert not metric.metric_per_cost
  assert not metric.cost_per_metric
  assert not metric.cost_column


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
  assert roas.cost_column == "cost"


def test_cpa_metric_has_correct_defaults():
  """Tests that the CPA metric has the correct defaults."""
  cpa = geoflex.metrics.CPA()
  assert cpa.name == "CPA"
  assert cpa.column == "conversions"
  assert not cpa.metric_per_cost
  assert cpa.cost_per_metric
  assert cpa.cost_column == "cost"


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
  metric_cls()  # Should not raise an error.
  with pytest.raises(TypeError):
    metric_cls(**invalid_args)  # Should not raise an error.


def test_metric_per_cost_and_cost_per_metric_are_mutually_exclusive():
  """Tests that metric per cost and cost per metric are mutually exclusive."""
  with pytest.raises(ValueError):
    geoflex.metrics.Metric(
        name="test_metric",
        metric_per_cost=True,
        cost_per_metric=True,
        cost_column="cost",
    )


@pytest.mark.parametrize(
    "metric_cls, args",
    [
        (geoflex.metrics.ROAS, {"cost_column": ""}),
        (geoflex.metrics.CPA, {"cost_column": ""}),
        (
            geoflex.metrics.Metric,
            {"metric_per_cost": True, "name": "test_metric"},
        ),
        (
            geoflex.metrics.Metric,
            {"cost_per_metric": True, "name": "test_metric"},
        ),
    ],
)
def test_cost_column_must_be_set_if_metric_needs_cost(metric_cls, args):
  """Tests that cost column is set if metric is metric per cost or cost per metric."""
  with pytest.raises(ValueError):
    metric_cls(**args)


def test_invert_metric_per_cost_round_trip():
  """Tests that invert works for metric per cost."""
  metric = geoflex.metrics.Metric(
      name="test_metric",
      metric_per_cost=True,
      cost_column="cost",
  )
  inverted_metric = metric.invert()

  expected_inverted_metric = geoflex.metrics.Metric(
      name="test_metric __INVERTED__",
      column="test_metric",
      metric_per_cost=False,
      cost_per_metric=True,
      cost_column="cost",
  )
  assert inverted_metric == expected_inverted_metric

  twice_inverted_metric = inverted_metric.invert()
  assert twice_inverted_metric == metric


def test_invert_cost_per_metric_round_trip():
  """Tests that invert works for metric per cost."""
  metric = geoflex.metrics.Metric(
      name="test_metric",
      cost_per_metric=True,
      cost_column="cost",
  )
  inverted_metric = metric.invert()

  expected_inverted_metric = geoflex.metrics.Metric(
      name="test_metric __INVERTED__",
      column="test_metric",
      metric_per_cost=True,
      cost_per_metric=False,
      cost_column="cost",
  )
  assert inverted_metric == expected_inverted_metric

  twice_inverted_metric = inverted_metric.invert()
  assert twice_inverted_metric == metric


def test_invert_raises_error_for_non_cost_metric():
  """Tests that invert raises an error for a non-cost metric."""
  metric = geoflex.metrics.Metric(
      name="test_metric",
      column="test_column",
  )
  with pytest.raises(ValueError):
    metric.invert()
