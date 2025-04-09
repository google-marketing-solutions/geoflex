"""Tests for the evaluation module."""

import geoflex.evaluation
import numpy as np
import pandas as pd
import pytest

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name
# pylint: disable=g-doc-return-or-yield


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


def test_scorer_returns_lower_score_for_representative_assignment(raw_data):
  scorer = geoflex.evaluation.GeoAssignmentRepresentivenessScorer(
      historical_data=raw_data,
      geo_column_name="geo_id",
      geos=["US", "UK", "AU", "NL"],
  )
  result_1, _ = scorer(np.array([0, 1, 0, 1]))
  result_2, _ = scorer(np.array([0, 0, 1, 1]))
  assert result_1 < result_2


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
