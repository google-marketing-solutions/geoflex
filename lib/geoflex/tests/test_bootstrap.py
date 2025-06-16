"""Tests for the bootstrap module."""

import geoflex.bootstrap
import numpy as np
import pandas as pd
import pytest


# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name
@pytest.fixture(name="raw_data")
def raw_data_fixture() -> pd.DataFrame:
  """Fixture for test data."""
  rng = np.random.default_rng(seed=42)
  raw_data = pd.DataFrame(
      rng.random((200, 2)),
      columns=["col1", "col2"],
      index=pd.date_range(start="2024-01-01", periods=200),
  )
  return raw_data


ELIGIBLE_PARAMS = [
    {},
    {"seasonality": 14},
    {"model_type": "additive"},
    {"model_type": "multiplicative"},
    {"seasons_per_block": 3},
    {"sampling_type": "random"},
    {"sampling_type": "permutation"},
    {"stl_params": {"robust": True}},
]


@pytest.mark.parametrize("params", ELIGIBLE_PARAMS)
def test_bootstrap_sample_returns_unique_samples(raw_data, params):
  bootstrap = geoflex.bootstrap.MultivariateTimeseriesBootstrap(**params)
  bootstrap.fit(raw_data)
  unique_samples = set()
  for bootstrap_sample in bootstrap.sample_dataframes(n_bootstraps=10):
    bootstrap_sample = tuple(bootstrap_sample.values.flatten().tolist())
    assert bootstrap_sample not in unique_samples
    unique_samples.add(bootstrap_sample)


@pytest.mark.parametrize("params", ELIGIBLE_PARAMS)
def test_sample_dataframes_returns_correct_shape_and_finite_values(
    raw_data, params
):
  bootstrap = geoflex.bootstrap.MultivariateTimeseriesBootstrap(**params)
  bootstrap.fit(raw_data)
  for sample in bootstrap.sample_dataframes(n_bootstraps=10):
    assert sample.shape == raw_data.shape
    assert np.all(np.isfinite(sample.values))
    assert sample.index.values.tolist() == raw_data.index.values.tolist()
    assert sample.columns.values.tolist() == raw_data.columns.values.tolist()


def test_fit_fails_for_negative_data_with_multiplicative_model(raw_data):
  raw_data["col1"] = raw_data["col1"] * -1
  bootstrap = geoflex.bootstrap.MultivariateTimeseriesBootstrap(
      model_type="multiplicative"
  )
  with pytest.raises(ValueError):
    bootstrap.fit(raw_data)
