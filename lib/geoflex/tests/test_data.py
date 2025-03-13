"""Tests for the data module."""

import geoflex.data
import pandas as pd
import pytest

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name

GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset


@pytest.fixture(name="raw_data")
def raw_data_fixture():
  """Fixture for test data."""
  return pd.DataFrame({
      "geo_id": ["US", "US", "CA", "CA"],
      "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
      "revenue": [100, 200, 300, 400],
      "cost": [10, 20, 30, 40],
      "clicks": [1000, 2000, 3000, 4000],
      "redundant_column": ["a", "b", "c", "d"],
  })


@pytest.fixture(name="expected_parsed_data")
def expected_parsed_data_fixture():
  """Fixture for parsed test data."""
  return (
      pd.DataFrame({
          "geo_id": ["US", "US", "CA", "CA"],
          "date": [
              pd.to_datetime("2024-01-01"),
              pd.to_datetime("2024-01-02"),
              pd.to_datetime("2024-01-01"),
              pd.to_datetime("2024-01-02"),
          ],
          "revenue": [100.0, 200.0, 300.0, 400.0],
          "cost": [10.0, 20.0, 30.0, 40.0],
          "clicks": [1000.0, 2000.0, 3000.0, 4000.0],
          "geo_id_date": [
              "US__2024-01-01",
              "US__2024-01-02",
              "CA__2024-01-01",
              "CA__2024-01-02",
          ],
      })
      .set_index(["geo_id_date"])
      .copy()
  )


@pytest.mark.parametrize(
    "missing_column",
    ["geo_id", "date"],
)
def test_geo_performance_dataset_raises_exception_if_data_missing_required_columns(
    raw_data, missing_column
):
  with pytest.raises(ValueError):
    GeoPerformanceDataset(data=raw_data.drop(columns=[missing_column]))


def test_geo_performance_dataset_raises_exception_if_data_has_no_response_columns(
    raw_data,
):
  with pytest.raises(ValueError):
    GeoPerformanceDataset(
        data=raw_data.drop(columns=["clicks", "cost", "revenue"])
    )


def test_geo_performance_dataset_raises_exception_if_data_has_duplicate_geos_and_dates(
    raw_data,
):
  duplicate_data = pd.concat([raw_data, raw_data], ignore_index=True)
  with pytest.raises(ValueError):
    GeoPerformanceDataset(data=duplicate_data)


def test_geo_performance_dataset_parses_data_correctly(
    raw_data, expected_parsed_data
):
  geo_dataset = GeoPerformanceDataset(data=raw_data)
  pd.testing.assert_frame_equal(
      geo_dataset.parsed_data, expected_parsed_data, check_like=True
  )


def test_geo_performance_dataset_fills_missing_data_with_zeros(
    raw_data, expected_parsed_data
):
  missing_index = expected_parsed_data.index[0]
  expected_parsed_data.loc[missing_index, "revenue"] = 0.0
  expected_parsed_data.loc[missing_index, "cost"] = 0.0
  expected_parsed_data.loc[missing_index, "clicks"] = 0.0

  geo_dataset = GeoPerformanceDataset(data=raw_data.iloc[1:])
  pd.testing.assert_frame_equal(
      geo_dataset.parsed_data, expected_parsed_data, check_like=True
  )


def test_geo_performance_dataset_returns_geos_correctly(raw_data):
  geo_dataset = GeoPerformanceDataset(data=raw_data)
  assert geo_dataset.geos == ["US", "CA"]


def test_geo_performance_dataset_returns_dates_correctly(raw_data):
  geo_dataset = GeoPerformanceDataset(data=raw_data)
  assert geo_dataset.dates == [
      pd.to_datetime("2024-01-01"),
      pd.to_datetime("2024-01-02"),
  ]
