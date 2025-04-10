"""Tests for the data module."""

import re
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


@pytest.fixture(name="raw_data_na_geo")
def raw_data_na_geo_fixture():
  """Fixture for test data with NA geo IDs."""
  return pd.DataFrame({
      "geo_id": ["US", None, "CA", "CA"],
      "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
      "revenue": [100, 200, 300, 400]
  })


@pytest.fixture(name="raw_data_gaps")
def raw_data_gaps_fixture():
  """Fixture for test data with gaps."""
  return pd.DataFrame({
      "geo_id": ["US", "US", "US", "CA", "CA", "CA"],
      "date": [
          "2024-01-01",
          "2024-01-02",  # day 2
          "2024-01-04",  # day 4 (gap after Day 2)
          "2024-01-01",
          "2024-01-02",
          "2024-01-04",
      ],
      "revenue": [100, 200, 300, 400, 500, 600],
  })


@pytest.fixture(name="raw_data_weekly")
def raw_data_weekly_fixture():
  """Fixture for test data with weekly frequency."""
  return pd.DataFrame({
      "geo_id": ["US", "US", "CA", "CA"],
      "date": [
          "2024-01-01",  # week 1
          "2024-01-08",  # week 2
          "2024-01-01",
          "2024-01-08",
          ],
      "revenue": [100, 200, 300, 400],
  })


@pytest.fixture(name="raw_data_int_geos")
def raw_data_int_geos_fixture():
  """Fixture for test data with integer geo IDs."""
  return pd.DataFrame({
      "geo_id": [10, 10, 20, 20],  # integer geo IDs
      "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
      "revenue": [100, 200, 300, 400],
  })


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


def test_geo_performance_dataset_raises_exception_if_geo_id_has_na(
    raw_data_na_geo
):
  with pytest.raises(
      ValueError, match=re.escape("non-NA values in the geo_id column")
  ):
    GeoPerformanceDataset(data=raw_data_na_geo)


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
  assert geo_dataset.geos == ["CA", "US"]


def test_geo_performance_dataset_returns_dates_correctly(raw_data):
  geo_dataset = GeoPerformanceDataset(data=raw_data)
  assert geo_dataset.dates == [
      pd.to_datetime("2024-01-01"),
      pd.to_datetime("2024-01-02"),
  ]


def test_geo_performance_dataset_handles_integer_geo_ids(raw_data_int_geos):
  geo_dataset = GeoPerformanceDataset(data=raw_data_int_geos)
  assert geo_dataset.geos == ["10", "20"]  # check conversion to str and sorted
  assert "10" in geo_dataset.parsed_data["geo_id"].unique()
  assert "20" in geo_dataset.parsed_data["geo_id"].unique()
  assert geo_dataset.geomapping_str_to_int == {"10": 1, "20": 2}


def test_data_frequency_days_detects_daily(raw_data):
  geo_dataset = GeoPerformanceDataset(data=raw_data)
  assert geo_dataset.data_frequency_days == 1


def test_data_frequency_days_detects_weekly(raw_data_weekly):
  geo_dataset = GeoPerformanceDataset(data=raw_data_weekly)
  assert geo_dataset.data_frequency_days == 7


def test_data_frequency_days_raises_error_for_less_than_two_dates(raw_data):
  one_date_data = raw_data[raw_data["date"] == "2024-01-01"]
  geo_dataset = GeoPerformanceDataset(data=one_date_data)
  with pytest.raises(ValueError, match=re.escape("at least 2 unique dates")):
    _ = geo_dataset.data_frequency_days


def test_data_frequency_days_raises_error_for_ambiguous_freq(raw_data_gaps):
  with pytest.raises(
      ValueError, match=re.escape("Cannot determine data frequency")
  ):
    geo_dataset = GeoPerformanceDataset(data=raw_data_gaps)
    _ = geo_dataset.data_frequency_days


def test_check_date_gaps_raises_error_on_gaps_by_default(raw_data_gaps):
  with pytest.raises(
      ValueError,
      match=re.escape("Cannot determine data frequency"),
  ):
    GeoPerformanceDataset(data=raw_data_gaps, allow_missing_dates=False)


def test_geo_mappings_are_correct(raw_data):
  geo_dataset = GeoPerformanceDataset(data=raw_data)
  # geos should be sorted alphabetically: CA, US
  expected_str_to_int = {"CA": 1, "US": 2}
  expected_int_to_str = {1: "CA", 2: "US"}
  expected_geos_int = {1, 2}

  assert geo_dataset.geomapping_str_to_int == expected_str_to_int
  assert geo_dataset.geomapping_int_to_str == expected_int_to_str
  assert geo_dataset.geos_int == expected_geos_int


def test_parsed_data_int_geos_correct(raw_data, expected_parsed_data):
  geo_dataset = GeoPerformanceDataset(data=raw_data)
  parsed_int = geo_dataset.parsed_data_int_geos

  # create expected parsed data with integer geo IDs
  expected_int = expected_parsed_data.copy()
  geo_map = {"CA": 1, "US": 2}  # explicitly define what mapping should be
  expected_int["geo_id"] = expected_int["geo_id"].map(geo_map).astype(int)
  expected_int = expected_int.sort_values(by=["geo_id", "date"])

  # check geo_id column is an integer and has expected values
  assert pd.api.types.is_integer_dtype(parsed_int["geo_id"])
  assert set(parsed_int["geo_id"].unique()) == {1, 2}

  # check the entire parsed data frame is correct
  pd.testing.assert_frame_equal(
      parsed_int.sort_values(by=["geo_id", "date"]),
      expected_int,  # already sorted above
      check_like=True  # handles potential column order differences
  )


def test_geo_performance_dataset_returns_pivoted_data_correctly(raw_data):
  geo_dataset = GeoPerformanceDataset(data=raw_data)
  expected_pivoted_data = pd.DataFrame(
      {
          ("clicks", "CA"): [3000.0, 4000.0],
          ("clicks", "US"): [1000.0, 2000.0],
          ("cost", "CA"): [30.0, 40.0],
          ("cost", "US"): [10.0, 20.0],
          ("revenue", "CA"): [300.0, 400.0],
          ("revenue", "US"): [100.0, 200.0],
      },
      index=pd.Series(
          pd.to_datetime(["2024-01-01", "2024-01-02"]), name="date"
      ),
  )
  expected_pivoted_data.columns.names = [None, "geo_id"]
  pd.testing.assert_frame_equal(
      geo_dataset.pivoted_data, expected_pivoted_data, check_like=True
  )


def test_geo_performance_dataset_from_pivoted_data_returns_correct_data(
    raw_data, expected_parsed_data
):
  first_geo_dataset = GeoPerformanceDataset(
      data=raw_data,
      geo_id_column="geo_id",
      date_column="date",
  )
  geo_dataset = GeoPerformanceDataset.from_pivoted_data(
      first_geo_dataset.pivoted_data,
      geo_id_column="geo_id",
      date_column="date",
  )
  pd.testing.assert_frame_equal(
      geo_dataset.parsed_data, expected_parsed_data, check_like=True
  )
