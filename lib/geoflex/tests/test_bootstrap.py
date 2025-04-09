"""Tests for the bootstrap module."""

import geoflex.bootstrap
import numpy as np
import pandas as pd
import pytest


# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name
@pytest.fixture(name="raw_data")
def raw_data_fixture():
  """Fixture for test data."""
  rng = np.random.default_rng(seed=42)
  return rng.random((200, 2))


ELIGIBLE_PARAMS = [
    {},
    {"seasonality": 14},
    {"log_transform": True},
    {"seasons_per_block": 3},
    {"verbose": True},
    {"full_blocks_only": True},
]


@pytest.mark.parametrize("params", ELIGIBLE_PARAMS)
def test_bootstrap_sample_returns_correct_shape_and_finite_values(
    raw_data, params
):
  bootstrap = geoflex.bootstrap.MultivariateTimeseriesBootstrap(**params)
  bootstrap.fit(raw_data)
  bootstrap_samples = bootstrap.sample(n_bootstraps=10)
  expected_shape = (10, 200, 2)
  if params.get("full_blocks_only", False):
    # 189 because the block size is 21 and this is the largest multiple of 21
    # that is smaller than 200.
    expected_shape = (10, 189, 2)
  assert bootstrap_samples.shape == expected_shape
  assert np.all(np.isfinite(bootstrap_samples))


@pytest.mark.parametrize("params", ELIGIBLE_PARAMS)
def test_bootstrap_sample_returns_unique_samples(raw_data, params):
  bootstrap = geoflex.bootstrap.MultivariateTimeseriesBootstrap(**params)
  bootstrap.fit(raw_data)
  unique_samples = set()
  for bootstrap_sample in bootstrap.sample(n_bootstraps=10):
    bootstrap_sample = tuple(bootstrap_sample.flatten().tolist())
    assert bootstrap_sample not in unique_samples
    unique_samples.add(bootstrap_sample)


def test_remove_remainder_if_not_full_blocks_only(
    raw_data,
):
  bootstrap = geoflex.bootstrap.MultivariateTimeseriesBootstrap()
  bootstrap.fit(raw_data)
  remainder_removed = bootstrap.remove_remainder(raw_data)
  np.testing.assert_array_equal(remainder_removed, raw_data)


def test_remove_remainder_if_full_blocks_only(
    raw_data,
):
  bootstrap = geoflex.bootstrap.MultivariateTimeseriesBootstrap(
      full_blocks_only=True
  )
  bootstrap.fit(raw_data)
  remainder_removed = bootstrap.remove_remainder(raw_data)
  # 189 because the block size is 21 and this is the largest multiple of 21
  # that is smaller than 200.
  np.testing.assert_array_equal(remainder_removed, raw_data[:189, :])
  assert 189 % bootstrap.block_size == 0


def test_sample_dataframes_returns_correct_shape_and_finite_values(
    raw_data,
):
  bootstrap = geoflex.bootstrap.MultivariateTimeseriesBootstrap()
  bootstrap.fit(raw_data)
  for sample in bootstrap.sample_dataframes(n_bootstraps=10):
    assert sample.shape == (200, 2)
    assert np.all(np.isfinite(sample.values))


def test_sample_dataframes_returns_columns_and_index(
    raw_data,
):
  bootstrap = geoflex.bootstrap.MultivariateTimeseriesBootstrap()
  bootstrap.fit(raw_data)
  for sample in bootstrap.sample_dataframes(n_bootstraps=10):
    assert sample.columns.tolist() == ["Column 0", "Column 1"]
    assert sample.index.tolist() == list(range(200))


def test_sample_dataframes_keeps_columns_and_index_when_dataframe_is_passed(
    raw_data,
):
  bootstrap = geoflex.bootstrap.MultivariateTimeseriesBootstrap()
  bootstrap.fit(
      pd.DataFrame(
          raw_data, columns=["Column A", "Column B"], index=10 + np.arange(200)
      )
  )
  for sample in bootstrap.sample_dataframes(n_bootstraps=10):
    assert sample.columns.tolist() == ["Column A", "Column B"]
    assert sample.index.tolist() == (10 + np.arange(200)).tolist()
