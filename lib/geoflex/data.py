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

"""Data for a geoflex experiment design."""

from collections.abc import Mapping
import datetime as dt
import functools
import itertools
import logging
import warnings

import geoflex.experiment_design
import geoflex.utils
import numpy as np
import pandas as pd
import pydantic

ExperimentDesign = geoflex.experiment_design.ExperimentDesign
ExperimentBudgetType = geoflex.experiment_design.ExperimentBudgetType
ParquetDataFrame = geoflex.utils.ParquetDataFrame

logger = logging.getLogger(__name__)


class GeoPerformanceDataset(pydantic.BaseModel):
  """Performance data for each geo and day."""

  data: ParquetDataFrame
  geo_id_column: str = "geo_id"
  date_column: str = "date"
  allow_missing_dates: bool = False

  model_config = pydantic.ConfigDict(
      arbitrary_types_allowed=True, extra="forbid"
  )

  @pydantic.model_validator(mode="after")
  def check_input_data_column(self) -> "GeoPerformanceDataset":
    """Checks that input data has required columns."""
    required_columns = [self.geo_id_column, self.date_column]
    missing_cols = [
        col for col in required_columns if col not in self.data.columns
    ]

    if missing_cols:
      error_message = (
          f"Data must have the following columns: {', '.join(required_columns)}"
      )
      logger.error(error_message)
      raise ValueError(error_message)

    return self

  @pydantic.model_validator(mode="after")
  def check_input_geos_na(self) -> "GeoPerformanceDataset":
    """Checks that input geos are not NA."""
    if self.data[self.geo_id_column].isna().any():
      error_message = (
          f"Data must have non-NA values in the {self.geo_id_column} column."
      )
      logger.error(error_message)
      raise ValueError(error_message)
    return self

  @functools.cached_property
  def parsed_data(self) -> pd.DataFrame:
    """Returns the parsed data with unique index.

    The parsed data is a copy of the data with the date column converted to
    a date object, the cost and all response columns converted to floats,
    the geo_id converted to a string, and a geo_id_date column added as a unique
    identifier for the dataset.
    """
    # convert geo to str first to ensure consistent merging
    input_data = self.data.copy()
    input_data[self.geo_id_column] = input_data[self.geo_id_column].astype(str)

    unique_geos = input_data[self.geo_id_column].unique().tolist()
    unique_dates = input_data[self.date_column].unique().tolist()

    if not unique_geos:
      error_message = "No unique geos found in the data."
      logger.error(error_message)
      raise ValueError(error_message)
    if not unique_dates:
      error_message = "No unique dates found in the data."
      logger.error(error_message)
      raise ValueError(error_message)

    index = pd.DataFrame(
        itertools.product(unique_geos, unique_dates),
        columns=[self.geo_id_column, self.date_column],
    )

    all_metric_columns = (  # Automatically find all metric columns.
        input_data.select_dtypes("number").columns.values.tolist()
    )
    if not all_metric_columns:
      error_message = "No metric columns found in the data."
      logger.error(error_message)
      raise ValueError(error_message)

    all_columns_to_keep = list(
        {self.geo_id_column, self.date_column} | set(all_metric_columns)
    )

    parsed_data = (
        index.merge(
            input_data[all_columns_to_keep],
            on=[self.geo_id_column, self.date_column],
            how="left",
        )
        .copy()
        # create unique id for each geo-date pair
        .assign(
            geo_id_date=lambda df: df[self.geo_id_column].astype(str)
            + "__"
            + df[self.date_column].astype(str)
        )
        .astype(
            {
                self.geo_id_column: "object",
                self.date_column: "datetime64[ns]",
            }
            | {column: "float64" for column in all_metric_columns}
        )
        .set_index("geo_id_date")
        .sort_values(by=[self.geo_id_column, self.date_column])
    )

    parsed_data[all_metric_columns] = (
        parsed_data[all_metric_columns].fillna(0.0)
    )

    if not parsed_data.index.is_unique:
      duplicated_indices = parsed_data.index[
          parsed_data.index.duplicated()
      ].unique().tolist()
      error_message = (
          "Data processing resulted in non-unique 'geo_id_date' index values. "
          f"Duplicated indices start with: {', '.join(duplicated_indices)}"
      )
      logger.error(error_message)
      raise ValueError(error_message)

    return parsed_data

  @functools.cached_property
  def geos(self) -> list[str]:
    """Returns the geos as string sorted alphabetically."""
    return sorted(self.parsed_data[self.geo_id_column].unique().tolist())

  @functools.cached_property
  def dates(self) -> list[dt.datetime]:
    """Returns the dates in the data sorted chronologically."""
    return sorted(self.parsed_data[self.date_column].unique().tolist())

  @functools.cached_property
  def geomapping_str_to_int(self) -> dict[str, int]:
    """Maps sorted string geo IDs to unique, 1-indexed integers."""
    return {geo_str: i + 1 for i, geo_str in enumerate(self.geos)}

  @functools.cached_property
  def geomapping_int_to_str(self) -> Mapping[int, str]:
    """Reversed mapping of integer geo ids back to strings."""
    return {v: k for k, v in self.geomapping_str_to_int.items()}

  @functools.cached_property
  def geos_int(self) -> set[int]:
    """Returns set of unique geo ids as 1-indexed integers."""
    return set(self.geomapping_str_to_int.values())

  @functools.cached_property
  def data_frequency_days(self) -> int:
    """Determines frequency of data in days.

    Calculated based on most common difference between consecutive dates.

    Returns:
      1 for daily data.
      7 for weekly data.

    Raises:
      ValueError: If fewer than 2 unique dates exist or frequency cannot be
      reliably determined.
    """
    unique_dates = self.dates
    if len(unique_dates) < 2:
      error_message = "Data must have at least 2 unique dates."
      logger.error(error_message)
      raise ValueError(error_message)

    date_diffs = pd.Series(unique_dates).diff().dt.days.dropna()

    if date_diffs.empty:
      warning_message = "Cannot determine data frequency."
      logger.warning(warning_message)
      warnings.warn(warning_message)
      return 1  # default to daily

    mode_freq = date_diffs.mode()

    if len(mode_freq) == 1 and mode_freq[0] in [1, 7]:
      frequency = int(mode_freq[0])
      return frequency
    else:
      median_freq = date_diffs.median()
      if median_freq in [1.0, 7.0]:
        frequency = int(median_freq)
        return frequency
      else:
        error_message = (
            "Cannot determine data frequency. "
            "Most common and median frequencies are not 1, 7, or integer."
        )
        logger.error(error_message)
        raise ValueError(error_message)

  @functools.cached_property
  def parsed_data_int_geos(self) -> pd.DataFrame:
    """Returns copy of parsed data with geo ids mapped to integers."""
    data_copy = self.parsed_data.copy()
    data_copy[self.geo_id_column] = (
        data_copy[self.geo_id_column].map(self.geomapping_str_to_int)
    )
    data_copy[self.geo_id_column] = data_copy[self.geo_id_column].astype(int)
    return data_copy

  @pydantic.model_validator(mode="after")
  def check_date_format(self) -> "GeoPerformanceDataset":
    """Checks that the input date column is either a string or a date object."""
    if pd.api.types.is_datetime64_any_dtype(self.data[self.date_column]):
      # If it's already a datetime object, then it's valid.
      return self

    if (
        self.data[self.date_column]
        .apply(lambda x: isinstance(x, dt.date))
        .all()
    ):
      # If it's a date object, then it's valid.
      return self

    if pd.api.types.is_string_dtype(self.data[self.date_column]):
      try:
        # Check that the string is in iso format.
        pd.to_datetime(self.data[self.date_column], format="%Y-%m-%d")
      except ValueError as e:
        error_message = (
            "Date column is a string but is not in ISO format (YYYY-MM-DD)."
        )
        logger.error(error_message)
        raise ValueError(error_message) from e
      return self

    error_message = (
        f"Date column ({self.date_column}) must be either a string in ISO"
        " format (YYYY-MM-DD) or a datetime object, but got"
        f" {self.data[self.date_column].dtype}"
    )
    logger.error(error_message)
    raise ValueError(error_message)

  @pydantic.model_validator(mode="after")
  def check_date_gaps(self) -> "GeoPerformanceDataset":
    """Checks that the data has no date gaps."""
    if len(self.dates) < 2:
      return self

    frequency = self.data_frequency_days
    expected_diff = pd.Timedelta(days=frequency)
    date_series = pd.Series(self.dates)
    actual_diffs = date_series.diff().dropna()  # exclude first na

    # get dates before gaps
    gap_indices = actual_diffs[actual_diffs != expected_diff].index
    dates_before_gaps = [date_series[i-1]for i in gap_indices]

    if dates_before_gaps:
      gap_dates_str = ", ".join(
          [date.strftime("%Y-%m-%d") for date in dates_before_gaps]
      )
      message = (
          f"Gaps found in data based on detected frequency of {frequency} days."
          f"Check these dates preceding a gap: {gap_dates_str}."
      )
      if self.allow_missing_dates:
        logger.warning(message)
        warnings.warn(message, UserWarning)
      else:
        error_message = message + " Set allow_missing_dates=True to ignore."
        logger.error(error_message)
        raise ValueError(error_message)

    return self

  @functools.cached_property
  def pivoted_data(self) -> pd.DataFrame:
    """Returns the data pivoted by geo and date."""
    return pd.pivot_table(
        self.parsed_data,
        index=self.date_column,
        columns=self.geo_id_column,
    ).sort_index()

  @classmethod
  def from_pivoted_data(
      cls,
      pivoted_data: pd.DataFrame,
      geo_id_column: str = "geo_id",
      date_column: str = "date",
  ) -> "GeoPerformanceDataset":
    """Returns a GeoPerformanceDataset from a pivoted data frame."""
    melted_data = (
        pivoted_data.stack(level=geo_id_column, future_stack=True)
        .reset_index()
        .sort_values([date_column, geo_id_column])
        .copy()
    )
    melted_data[date_column] = melted_data[date_column].dt.strftime("%Y-%m-%d")
    return cls(
        data=melted_data,
        geo_id_column=geo_id_column,
        date_column=date_column,
    )

  def get_pivoted_treatment_data(
      self, design: ExperimentDesign, experiment_start_date: dt.date
  ) -> list[pd.DataFrame]:
    """Returns the pivoted data for the treatment geos and experiment dates.

    Args:
      design: The experiment design.
      experiment_start_date: The start date of the experiment.

    Returns:
      A list of pivoted data frames, one for each treatment cell.
    """
    pivoted_data = self.pivoted_data.copy()

    experiment_end_date = experiment_start_date + dt.timedelta(
        weeks=design.runtime_weeks
    )
    treatment_dates_mask = (
        pivoted_data.index.values >= experiment_start_date
    ) & (pivoted_data.index.values < experiment_end_date)

    treatment_data = []
    for treatment_cell in design.geo_assignment.treatment:
      treatment_data.append(
          pivoted_data.loc[
              treatment_dates_mask, (slice(None), list(treatment_cell))
          ]
      )
    return treatment_data

  def simulate_experiment(
      self,
      experiment_start_date: dt.date,
      design: ExperimentDesign,
      treatment_effect_sizes: list[float] | None = None,
  ) -> "GeoPerformanceDataset":
    """Simulates a geo experiment in the data.

    This applies the experiment budget, as specified in the experiment design,
    to the cost metrics in the data. For now it only supports A/A simulations
    with no treatment effect.

    Args:
      experiment_start_date: The start date of the experiment. This is the date
        that the treatment geos were first exposed to the treatment.
      design: The experiment design.
      treatment_effect_sizes: The absolute treatment effect sizes for the
        primary metric. One value per treatment cell. If None, then the
        treatment effect size is 0.0 for all treatment cells.

    Returns:
      The simulated experiment data.

    Raises:
      ValueError: If the experiment type is hold back and the cost metric is
        not 0.0.
    """
    treatment_effect_sizes = treatment_effect_sizes or [0.0] * (
        design.n_cells - 1
    )
    if len(treatment_effect_sizes) != design.n_cells - 1:
      error_message = (
          "Treatment effect sizes must be specified for all treatment cells,"
          f" got {len(treatment_effect_sizes)} treatment effect sizes for"
          f" {design.n_cells - 1} treatment cells."
      )
      logger.error(error_message)
      raise ValueError(error_message)

    # Get all cost metrics.
    all_metrics = [design.primary_metric] + design.secondary_metrics
    cost_columns = set([
        metric.cost_column
        for metric in all_metrics
        if metric.cost_per_metric or metric.metric_per_cost
    ])

    data = self.pivoted_data.copy()
    experiment_end_date = experiment_start_date + dt.timedelta(
        weeks=design.runtime_weeks
    )
    treatment_dates_mask = (data.index.values >= experiment_start_date) & (
        data.index.values < experiment_end_date
    )

    is_percent_change = (
        design.experiment_budget.budget_type
        == ExperimentBudgetType.PERCENTAGE_CHANGE
    )
    is_total_budget = (
        design.experiment_budget.budget_type
        == ExperimentBudgetType.TOTAL_BUDGET
    )
    has_cost_metrics = bool(cost_columns)
    has_existing_cost = data[list(cost_columns)].abs().sum().sum() > 0

    if has_cost_metrics and is_percent_change and not has_existing_cost:
      error_message = (
          "Cost metrics exist and are zero, cannot use a percentage change"
          " budget."
      )
      logger.error(error_message)
      raise ValueError(error_message)

    # Apply the experiment budget to the cost metrics.
    for cost_metric in cost_columns:
      budget_values = design.experiment_budget.value
      if not isinstance(budget_values, list):
        budget_values = [budget_values] * (design.n_cells - 1)

      for treatment_cell, cell_budget_value in zip(
          design.geo_assignment.treatment, budget_values
      ):
        if np.isclose(cell_budget_value, 0.0):
          # If there is no budget change then there is no cost to simulate.
          continue

        is_spend_increase = cell_budget_value > 0.0
        is_heavy_up = is_spend_increase and has_existing_cost

        for treatment_geo in treatment_cell:
          if is_percent_change:
            # Simply apply the percentage change
            data.loc[treatment_dates_mask, (cost_metric, treatment_geo)] *= (
                1 + cell_budget_value
            )
            continue

          budget_value = cell_budget_value
          if is_total_budget:
            # Total budget is divided across runtime
            budget_value /= design.runtime_weeks * 7

          if is_heavy_up:
            # For a heavy up experiment, the budget is shared proportional to
            # the existing cost across geos.
            total_cost = (
                self.pivoted_data[cost_metric]
                .loc[treatment_dates_mask, list(treatment_cell)]
                .abs()  # Take abs() just in case there are negative costs.
                .sum()
                .sum()
            )
            geo_cost = (
                self.pivoted_data[cost_metric]
                .loc[treatment_dates_mask, treatment_geo]
                .abs()  # Take abs() just in case there are negative costs.
                .sum()
            )

            budget_frac = geo_cost / total_cost
          else:  # is_hold_back
            # For a hold back experiment, the budget is shared proportional to
            # the primary response metric across geos. This is because the
            # cost does not exist yet for a hold back experiment.
            total_primary_response = (
                self.pivoted_data[design.primary_metric.column]
                .loc[treatment_dates_mask, list(treatment_cell)]
                .abs()  # Take abs() just in case there are negative responses.
                .sum()
                .sum()
            )
            geo_primary_response = (
                self.pivoted_data[design.primary_metric.column]
                .loc[treatment_dates_mask, treatment_geo]
                .abs()  # Take abs() just in case there are negative responses.
                .sum()
            )
            budget_frac = geo_primary_response / total_primary_response

          data.loc[treatment_dates_mask, (cost_metric, treatment_geo)] += (
              budget_value * budget_frac
          )

          # Finally make sure the spend does not go negative for any geo
          data.loc[treatment_dates_mask, (cost_metric, treatment_geo)] = (
              data.loc[treatment_dates_mask, (cost_metric, treatment_geo)].clip(
                  lower=0
              )
          )

    # Now if there is a non zero treatment effect size, then apply it to the
    # primary metric.
    for treatment_cell, treatment_effect_size in zip(
        design.geo_assignment.treatment, treatment_effect_sizes
    ):
      if not np.isclose(treatment_effect_size, 0.0):
        # For a cost per metric or metric per cost, we need to calculate the
        # delta in cost and use that to calculate the delta in the metric.
        if (
            design.primary_metric.cost_per_metric
            or design.primary_metric.metric_per_cost
        ):
          cost_delta = (
              data.loc[
                  treatment_dates_mask,
                  (design.primary_metric.cost_column, list(treatment_cell)),
              ]
              - self.pivoted_data.loc[
                  treatment_dates_mask,
                  (design.primary_metric.cost_column, list(treatment_cell)),
              ]
          )
        else:
          cost_delta = 0.0

        if design.primary_metric.metric_per_cost:
          data.loc[
              treatment_dates_mask,
              (design.primary_metric.column, list(treatment_cell)),
          ] += (
              cost_delta.values * treatment_effect_size
          )
        elif design.primary_metric.cost_per_metric:
          data.loc[
              treatment_dates_mask,
              (design.primary_metric.column, list(treatment_cell)),
          ] += (
              cost_delta.values / treatment_effect_size
          )
        else:
          # For a regular metric, it's just the absolute lift in the metric. The
          # absolute lift is the total lift across the geos, so we will divide
          # it equally across the geos and the days.
          n_geos = len(treatment_cell)
          n_days = treatment_dates_mask.sum()
          data.loc[
              treatment_dates_mask,
              (design.primary_metric.column, list(treatment_cell)),
          ] += treatment_effect_size / (n_geos * n_days)

    return self.from_pivoted_data(
        data,
        geo_id_column=self.geo_id_column,
        date_column=self.date_column,
    )
