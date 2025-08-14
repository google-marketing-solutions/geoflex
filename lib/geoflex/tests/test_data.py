# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the data module."""

import datetime as dt
import re
import geoflex.data
import numpy as np
import pandas as pd
import pytest

# Tests don't need docstrings.
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name

GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
ExperimentBudget = geoflex.experiment_design.ExperimentBudget
ExperimentBudgetType = geoflex.experiment_design.ExperimentBudgetType
ExperimentDesign = geoflex.experiment_design.ExperimentDesign
GeoAssignment = geoflex.experiment_design.GeoAssignment
Metric = geoflex.metrics.Metric


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


@pytest.fixture(name="default_design")
def default_design_fixture():
  """Fixture for a default experiment design."""
  return ExperimentDesign(
      geo_assignment=GeoAssignment(
          treatment=[["US", "CA"], ["DE", "FR"]],
          control=["UK", "NL"],
      ),
      experiment_budget=ExperimentBudget(
          budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
          value=0.1,
      ),
      primary_metric=Metric(
          name="ROAS",
          column="revenue",
          cost_column="cost",
          metric_per_cost=True,
      ),
      secondary_metrics=[
          Metric(
              name="Campaign ROAS",
              column="revenue",
              cost_column="campaign_cost",
              metric_per_cost=True,
          )
      ],
      runtime_weeks=4,
      methodology="test_methodology",
      n_cells=3,
  )


@pytest.fixture(name="experiment_start_date")
def default_start_date_fixture():
  """Fixture for a default experiment start date."""
  return pd.to_datetime("2024-02-01")


@pytest.fixture(name="big_raw_data")
def big_raw_data_fixture():
  """Fixture for a big raw data."""
  rng = np.random.default_rng(seed=42)
  data = pd.DataFrame({
      "geo_id": (
          ["US"] * 100
          + ["UK"] * 100
          + ["NL"] * 100
          + ["CA"] * 100
          + ["DE"] * 100
          + ["FR"] * 100
      ),
      "date": pd.date_range(start="2024-01-01", periods=100).tolist() * 6,
      "revenue": rng.random(size=600),
      "campaign_cost": rng.random(size=600),
      "cost": rng.random(size=600),
  })
  data["date"] = data["date"].dt.strftime("%Y-%m-%d")
  return data


@pytest.fixture(name="big_raw_data_zero_costs")
def big_raw_data_fixture_zero_costs(big_raw_data):
  """Fixture for a big raw data with zero costs for testing hold back."""
  big_raw_data_zero_cost = big_raw_data.copy()
  big_raw_data_zero_cost["cost"] = 0.0
  big_raw_data_zero_cost["campaign_cost"] = 0.0
  return big_raw_data_zero_cost


def test_simulate_experiment_returns_correct_data_with_treatment_effect_iroas(
    big_raw_data, default_design, experiment_start_date
):
  geo_dataset = GeoPerformanceDataset(data=big_raw_data)
  simulated_dataset = geo_dataset.simulate_experiment(
      experiment_start_date=experiment_start_date,
      design=default_design,
      treatment_effect_sizes=[1.5, 0.5],  # iROAS
  )

  experiment_end_date = experiment_start_date + dt.timedelta(
      weeks=default_design.runtime_weeks
  )
  is_experiment_date = (
      geo_dataset.parsed_data["date"] >= experiment_start_date
  ) & (geo_dataset.parsed_data["date"] < experiment_end_date)
  expected_data = geo_dataset.parsed_data.copy()

  expected_data.loc[
      expected_data.geo_id.isin(["US", "CA"]) & is_experiment_date,
      "revenue",
  ] += (
      expected_data.loc[
          expected_data.geo_id.isin(["US", "CA"]) & is_experiment_date,
          "cost",
      ].values
      * 0.1
      * 1.5
  )
  expected_data.loc[
      expected_data.geo_id.isin(["DE", "FR"]) & is_experiment_date,
      "revenue",
  ] += (
      expected_data.loc[
          expected_data.geo_id.isin(["DE", "FR"]) & is_experiment_date,
          "cost",
      ].values
      * 0.1
      * 0.5
  )

  expected_data.loc[
      expected_data.geo_id.isin(["US", "CA", "DE", "FR"]) & is_experiment_date,
      ["cost", "campaign_cost"],
  ] *= 1.1

  pd.testing.assert_frame_equal(
      simulated_dataset.parsed_data, expected_data, check_like=True
  )


def test_simulate_experiment_returns_correct_data_with_treatment_effect_cpia(
    big_raw_data, default_design, experiment_start_date
):
  geo_dataset = GeoPerformanceDataset(data=big_raw_data)
  design = default_design.make_variation(
      primary_metric=geoflex.metrics.CPiA(conversions_column="revenue"),
  )
  design.geo_assignment = default_design.geo_assignment.model_copy()
  simulated_dataset = geo_dataset.simulate_experiment(
      experiment_start_date=experiment_start_date,
      design=design,
      treatment_effect_sizes=[1.5, 0.5],
  )

  experiment_end_date = experiment_start_date + dt.timedelta(
      weeks=design.runtime_weeks
  )
  is_experiment_date = (
      geo_dataset.parsed_data["date"] >= experiment_start_date
  ) & (geo_dataset.parsed_data["date"] < experiment_end_date)
  expected_data = geo_dataset.parsed_data.copy()

  expected_data.loc[
      expected_data.geo_id.isin(["US", "CA"]) & is_experiment_date,
      "revenue",
  ] += (
      expected_data.loc[
          expected_data.geo_id.isin(["US", "CA"]) & is_experiment_date,
          "cost",
      ].values
      * 0.1
      / 1.5
  )
  expected_data.loc[
      expected_data.geo_id.isin(["DE", "FR"]) & is_experiment_date,
      "revenue",
  ] += (
      expected_data.loc[
          expected_data.geo_id.isin(["DE", "FR"]) & is_experiment_date,
          "cost",
      ].values
      * 0.1
      / 0.5
  )

  expected_data.loc[
      expected_data.geo_id.isin(["US", "CA", "DE", "FR"]) & is_experiment_date,
      ["cost", "campaign_cost"],
  ] *= 1.1

  pd.testing.assert_frame_equal(
      simulated_dataset.parsed_data, expected_data, check_like=True
  )


def test_simulate_experiment_returns_correct_data_with_treatment_effect_regular_metric(
    big_raw_data, default_design, experiment_start_date
):
  geo_dataset = GeoPerformanceDataset(data=big_raw_data)
  design = default_design.make_variation(
      primary_metric="revenue",
  )
  design.geo_assignment = default_design.geo_assignment.model_copy()
  simulated_dataset = geo_dataset.simulate_experiment(
      experiment_start_date=experiment_start_date,
      design=design,
      treatment_effect_sizes=[1.5, 0.5],
  )

  experiment_end_date = experiment_start_date + dt.timedelta(
      weeks=design.runtime_weeks
  )
  is_experiment_date = (
      geo_dataset.parsed_data["date"] >= experiment_start_date
  ) & (geo_dataset.parsed_data["date"] < experiment_end_date)
  expected_data = geo_dataset.parsed_data.copy()

  expected_data.loc[
      expected_data.geo_id.isin(["US", "CA"]) & is_experiment_date,
      "revenue",
  ] += 1.5 / (
      2 * design.runtime_weeks * 7
  )  # Absolute effect shared over geos and dates
  expected_data.loc[
      expected_data.geo_id.isin(["DE", "FR"]) & is_experiment_date,
      "revenue",
  ] += 0.5 / (
      2 * design.runtime_weeks * 7
  )  # Absolute effect shared over geos and dates
  expected_data.loc[
      expected_data.geo_id.isin(["US", "CA", "DE", "FR"]) & is_experiment_date,
      ["campaign_cost"],
  ] *= 1.1

  pd.testing.assert_frame_equal(
      simulated_dataset.parsed_data, expected_data, check_like=True
  )


def test_simulate_experiment_raises_error_for_percentage_change_zero_historical_cost(
    big_raw_data_zero_costs, default_design, experiment_start_date
):
  geo_dataset = GeoPerformanceDataset(data=big_raw_data_zero_costs)
  with pytest.raises(ValueError):
    geo_dataset.simulate_experiment(
        experiment_start_date=experiment_start_date,
        design=default_design,
        treatment_effect_sizes=None,
    )


def test_simulate_experiment_returns_correct_data_with_different_budgets_per_cell(
    big_raw_data, default_design, experiment_start_date
):
  design = default_design.make_variation(
      experiment_budget=ExperimentBudget(
          budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
          value=[0.1, -0.1],
      ),
      n_cells=3,
  )
  design.geo_assignment = default_design.geo_assignment.model_copy()

  geo_dataset = GeoPerformanceDataset(data=big_raw_data)
  simulated_dataset = geo_dataset.simulate_experiment(
      experiment_start_date=experiment_start_date,
      design=design,
      treatment_effect_sizes=None,
  )

  experiment_end_date = experiment_start_date + dt.timedelta(
      weeks=default_design.runtime_weeks
  )
  is_experiment_date = (
      geo_dataset.parsed_data["date"] >= experiment_start_date
  ) & (geo_dataset.parsed_data["date"] < experiment_end_date)
  expected_data = geo_dataset.parsed_data.copy()
  expected_data.loc[
      expected_data.geo_id.isin(["US", "CA"]) & is_experiment_date,
      ["cost", "campaign_cost"],
  ] *= 1.1
  expected_data.loc[
      expected_data.geo_id.isin(["DE", "FR"]) & is_experiment_date,
      ["cost", "campaign_cost"],
  ] *= 0.9

  pd.testing.assert_frame_equal(
      simulated_dataset.parsed_data, expected_data, check_like=True
  )


def test_simulate_experiment_returns_correct_data_for_heavy_up_percentage_change(
    big_raw_data, default_design, experiment_start_date
):
  geo_dataset = GeoPerformanceDataset(data=big_raw_data)
  simulated_dataset = geo_dataset.simulate_experiment(
      experiment_start_date=experiment_start_date,
      design=default_design,
      treatment_effect_sizes=None,
  )

  experiment_end_date = experiment_start_date + dt.timedelta(
      weeks=default_design.runtime_weeks
  )
  is_experiment_date = (
      geo_dataset.parsed_data["date"] >= experiment_start_date
  ) & (geo_dataset.parsed_data["date"] < experiment_end_date)
  expected_data = geo_dataset.parsed_data.copy()
  expected_data.loc[
      expected_data.geo_id.isin(["US", "CA", "DE", "FR"]) & is_experiment_date,
      ["cost", "campaign_cost"],
  ] *= 1.1

  pd.testing.assert_frame_equal(
      simulated_dataset.parsed_data, expected_data, check_like=True
  )


def test_simulate_experiment_returns_correct_data_for_heavy_up_total_budget(
    big_raw_data, default_design, experiment_start_date
):
  geo_dataset = GeoPerformanceDataset(data=big_raw_data)
  design = default_design.make_variation(
      experiment_budget=ExperimentBudget(
          budget_type=ExperimentBudgetType.TOTAL_BUDGET,
          value=100,
      ),
  )
  design.geo_assignment = default_design.geo_assignment.model_copy()

  simulated_dataset = geo_dataset.simulate_experiment(
      experiment_start_date=experiment_start_date,
      design=design,
      treatment_effect_sizes=None,
  )

  experiment_end_date = experiment_start_date + dt.timedelta(
      weeks=design.runtime_weeks
  )
  is_experiment_date = (
      geo_dataset.parsed_data["date"] >= experiment_start_date
  ) & (geo_dataset.parsed_data["date"] < experiment_end_date)
  expected_data = geo_dataset.parsed_data.copy()

  cost_diff = (
      simulated_dataset.parsed_data[["cost", "campaign_cost"]]
      - expected_data[["cost", "campaign_cost"]]
  )
  runtime_cost_diff = cost_diff.loc[is_experiment_date]
  non_runtime_cost_diff = cost_diff.loc[~is_experiment_date]

  # Assert that the cost is unchanhged outside of the experiment.
  assert (non_runtime_cost_diff == 0.0).all().all()

  # Assert that the total cost equals the expected budget per cell.
  total_cost_cell_1 = runtime_cost_diff.loc[
      expected_data.geo_id.isin(["US", "CA"])
  ].sum()
  assert np.isclose(total_cost_cell_1["cost"], 100)
  assert np.isclose(total_cost_cell_1["campaign_cost"], 100)
  total_cost_cell_2 = runtime_cost_diff.loc[
      expected_data.geo_id.isin(["DE", "FR"])
  ].sum()
  assert np.isclose(total_cost_cell_2["cost"], 100)
  assert np.isclose(total_cost_cell_2["campaign_cost"], 100)

  # Assert that the cost is spread evenly per day
  for geo_id in ["US", "CA", "DE", "FR"]:
    cost_diff_per_day = runtime_cost_diff.loc[expected_data.geo_id == geo_id]
    np.testing.assert_array_almost_equal(
        cost_diff_per_day["cost"].values,
        cost_diff_per_day["cost"].values[0],
        decimal=7,
    )
    np.testing.assert_array_almost_equal(
        cost_diff_per_day["campaign_cost"].values,
        cost_diff_per_day["campaign_cost"].values[0],
        decimal=7,
    )


def test_simulate_experiment_returns_correct_data_for_heavy_up_daily_budget(
    big_raw_data, default_design, experiment_start_date
):
  geo_dataset = GeoPerformanceDataset(data=big_raw_data)
  design = default_design.make_variation(
      experiment_budget=ExperimentBudget(
          budget_type=ExperimentBudgetType.DAILY_BUDGET,
          value=100,
      ),
  )
  design.geo_assignment = default_design.geo_assignment.model_copy()

  simulated_dataset = geo_dataset.simulate_experiment(
      experiment_start_date=experiment_start_date,
      design=design,
      treatment_effect_sizes=None,
  )

  experiment_end_date = experiment_start_date + dt.timedelta(
      weeks=design.runtime_weeks
  )
  is_experiment_date = (
      geo_dataset.parsed_data["date"] >= experiment_start_date
  ) & (geo_dataset.parsed_data["date"] < experiment_end_date)
  expected_data = geo_dataset.parsed_data.copy()

  cost_diff = (
      simulated_dataset.parsed_data[["cost", "campaign_cost"]]
      - expected_data[["cost", "campaign_cost"]]
  )
  runtime_cost_diff = cost_diff.loc[is_experiment_date].copy()
  non_runtime_cost_diff = cost_diff.loc[~is_experiment_date].copy()

  # Assert that the cost is unchanhged outside of the experiment.
  assert (non_runtime_cost_diff == 0.0).all().all()

  # Assert that the cost per day is equal to the expected daily budget.
  runtime_cost_diff["date"] = expected_data["date"].copy()
  for geo_ids in [["US", "CA"], ["DE", "FR"]]:
    cost_diff_per_day = (
        runtime_cost_diff.loc[expected_data.geo_id.isin(geo_ids)]
        .groupby("date")[["cost", "campaign_cost"]]
        .sum()
    )
    np.testing.assert_array_almost_equal(
        cost_diff_per_day["cost"].values,
        100.0,
        decimal=7,
    )
    np.testing.assert_array_almost_equal(
        cost_diff_per_day["campaign_cost"].values,
        100.0,
        decimal=7,
    )


@pytest.mark.parametrize(
    "budget_value,budget_type",
    [
        (0.0, ExperimentBudgetType.PERCENTAGE_CHANGE),
        (-0.1, ExperimentBudgetType.PERCENTAGE_CHANGE),
        (100, ExperimentBudgetType.TOTAL_BUDGET),
        (100, ExperimentBudgetType.DAILY_BUDGET),
    ],
)
def test_simulate_experiment_returns_unchanged_data_if_no_cost_metrics(
    big_raw_data,
    default_design,
    experiment_start_date,
    budget_value,
    budget_type,
):
  geo_dataset = GeoPerformanceDataset(data=big_raw_data)
  design = default_design.make_variation(
      experiment_budget=ExperimentBudget(
          budget_type=budget_type,
          value=budget_value,
      ),
      primary_metric="conversions",
      secondary_metrics=[],
  )
  design.geo_assignment = default_design.geo_assignment.model_copy()

  simulated_dataset = geo_dataset.simulate_experiment(
      experiment_start_date=experiment_start_date,
      design=design,
      treatment_effect_sizes=None,
  )

  pd.testing.assert_frame_equal(
      simulated_dataset.parsed_data, geo_dataset.parsed_data, check_like=True
  )


def test_simulate_experiment_returns_correct_data_for_go_dark(
    big_raw_data, default_design, experiment_start_date
):
  geo_dataset = GeoPerformanceDataset(data=big_raw_data)
  design = default_design.make_variation(
      experiment_budget=ExperimentBudget(
          budget_type=ExperimentBudgetType.PERCENTAGE_CHANGE,
          value=-0.1,
      ),
  )
  design.geo_assignment = default_design.geo_assignment.model_copy()

  simulated_dataset = geo_dataset.simulate_experiment(
      experiment_start_date=experiment_start_date,
      design=design,
      treatment_effect_sizes=None,
  )

  experiment_end_date = experiment_start_date + dt.timedelta(
      weeks=design.runtime_weeks
  )
  is_experiment_date = (
      geo_dataset.parsed_data["date"] >= experiment_start_date
  ) & (geo_dataset.parsed_data["date"] < experiment_end_date)
  expected_data = geo_dataset.parsed_data.copy()
  expected_data.loc[
      expected_data.geo_id.isin(["US", "CA", "DE", "FR"]) & is_experiment_date,
      ["cost", "campaign_cost"],
  ] *= 0.9

  pd.testing.assert_frame_equal(
      simulated_dataset.parsed_data, expected_data, check_like=True
  )


def test_simulate_experiment_returns_correct_data_for_hold_back_total_budget(
    big_raw_data_zero_costs, default_design, experiment_start_date
):
  geo_dataset = GeoPerformanceDataset(data=big_raw_data_zero_costs)
  design = default_design.make_variation(
      experiment_budget=ExperimentBudget(
          budget_type=ExperimentBudgetType.TOTAL_BUDGET,
          value=100,
      ),
  )
  design.geo_assignment = default_design.geo_assignment.model_copy()

  simulated_dataset = geo_dataset.simulate_experiment(
      experiment_start_date=experiment_start_date,
      design=design,
      treatment_effect_sizes=None,
  )

  experiment_end_date = experiment_start_date + dt.timedelta(
      weeks=design.runtime_weeks
  )
  is_experiment_date = (
      geo_dataset.parsed_data["date"] >= experiment_start_date
  ) & (geo_dataset.parsed_data["date"] < experiment_end_date)
  expected_data = geo_dataset.parsed_data.copy()

  cost_diff = (
      simulated_dataset.parsed_data[["cost", "campaign_cost"]]
      - expected_data[["cost", "campaign_cost"]]
  )
  runtime_cost_diff = cost_diff.loc[is_experiment_date]
  non_runtime_cost_diff = cost_diff.loc[~is_experiment_date]

  # Assert that the cost is unchanhged outside of the experiment.
  assert (non_runtime_cost_diff == 0.0).all().all()

  # Assert that the total cost equals the expected budget per cell.
  total_cost_cell_1 = runtime_cost_diff.loc[
      expected_data.geo_id.isin(["US", "CA"])
  ].sum()
  assert np.isclose(total_cost_cell_1["cost"], 100)
  assert np.isclose(total_cost_cell_1["campaign_cost"], 100)
  total_cost_cell_2 = runtime_cost_diff.loc[
      expected_data.geo_id.isin(["DE", "FR"])
  ].sum()
  assert np.isclose(total_cost_cell_2["cost"], 100)
  assert np.isclose(total_cost_cell_2["campaign_cost"], 100)

  # Assert that the cost is spread evenly per day
  for geo_id in ["US", "CA", "DE", "FR"]:
    cost_diff_per_day = runtime_cost_diff.loc[expected_data.geo_id == geo_id]
    np.testing.assert_array_almost_equal(
        cost_diff_per_day["cost"].values,
        cost_diff_per_day["cost"].values[0],
        decimal=7,
    )
    np.testing.assert_array_almost_equal(
        cost_diff_per_day["campaign_cost"].values,
        cost_diff_per_day["campaign_cost"].values[0],
        decimal=7,
    )


def test_simulate_experiment_returns_correct_data_for_hold_back_daily_budget(
    big_raw_data_zero_costs, default_design, experiment_start_date
):
  geo_dataset = GeoPerformanceDataset(data=big_raw_data_zero_costs)
  design = default_design.make_variation(
      experiment_budget=ExperimentBudget(
          budget_type=ExperimentBudgetType.DAILY_BUDGET,
          value=100,
      ),
  )
  design.geo_assignment = default_design.geo_assignment.model_copy()

  simulated_dataset = geo_dataset.simulate_experiment(
      experiment_start_date=experiment_start_date,
      design=design,
      treatment_effect_sizes=None,
  )

  experiment_end_date = experiment_start_date + dt.timedelta(
      weeks=design.runtime_weeks
  )
  is_experiment_date = (
      geo_dataset.parsed_data["date"] >= experiment_start_date
  ) & (geo_dataset.parsed_data["date"] < experiment_end_date)
  expected_data = geo_dataset.parsed_data.copy()

  cost_diff = (
      simulated_dataset.parsed_data[["cost", "campaign_cost"]]
      - expected_data[["cost", "campaign_cost"]]
  )
  runtime_cost_diff = cost_diff.loc[is_experiment_date].copy()
  non_runtime_cost_diff = cost_diff.loc[~is_experiment_date].copy()

  # Assert that the cost is unchanhged outside of the experiment.
  assert (non_runtime_cost_diff == 0.0).all().all()

  # Assert that the cost per day is equal to the expected daily budget.
  runtime_cost_diff["date"] = expected_data["date"].copy()
  for geo_ids in [["US", "CA"], ["DE", "FR"]]:
    cost_diff_per_day = (
        runtime_cost_diff.loc[expected_data.geo_id.isin(geo_ids)]
        .groupby("date")[["cost", "campaign_cost"]]
        .sum()
    )
    np.testing.assert_array_almost_equal(
        cost_diff_per_day["cost"].values,
        100.0,
        decimal=7,
    )
    np.testing.assert_array_almost_equal(
        cost_diff_per_day["campaign_cost"].values,
        100.0,
        decimal=7,
    )


def test_geo_performance_dataset_check_date_format_raises_error_for_non_iso_string_date(
    raw_data,
):
  raw_data["date"] = pd.to_datetime(raw_data["date"]).dt.strftime("%Y/%m/%d")
  with pytest.raises(ValueError):
    GeoPerformanceDataset(data=raw_data)


def test_geo_performance_dataset_check_date_format_raises_error_for_non_string_or_date_column(
    raw_data,
):
  raw_data["date"] = pd.to_datetime(raw_data["date"]).astype(int)
  with pytest.raises(ValueError):
    GeoPerformanceDataset(data=raw_data)


def test_geo_performance_dataset_check_date_format_does_not_raise_error_for_valid_datetime_column(
    raw_data,
):
  raw_data["date"] = pd.to_datetime(raw_data["date"])
  GeoPerformanceDataset(data=raw_data)


def test_geo_performance_dataset_check_date_format_does_not_raise_error_for_valid_date_column(
    raw_data,
):
  raw_data["date"] = pd.to_datetime(raw_data["date"]).dt.date
  dataset = GeoPerformanceDataset(data=raw_data)
  assert isinstance(dataset.parsed_data, pd.DataFrame)
