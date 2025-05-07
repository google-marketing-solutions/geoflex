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
ExperimentType = geoflex.experiment_design.ExperimentType
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

  def simulate_experiment(
      self,
      experiment_start_date: dt.date,
      design: ExperimentDesign,
      treatment_effect_size: float = 0.0,
  ) -> "GeoPerformanceDataset":
    """Simulates a geo experiment in the data.

    This applies the experiment budget, as specified in the experiment design,
    to the cost metrics in the data. For now it only supports A/A simulations
    with no treatment effect.

    Args:
      experiment_start_date: The start date of the experiment. This is the date
        that the treatment geos were first exposed to the treatment.
      design: The experiment design.
      treatment_effect_size: The treatment effect size. This is the percentage
        change in the response metric that the treatment geos will experience
        compared to the control geos in all response metrics.

    Returns:
      The simulated experiment data.

    Raises:
      NotImplementedError: If the treatment effect size is not 0.0.
      ValueError: If the experiment type is hold back and the cost metric is
        not 0.0.
    """
    if not np.isclose(treatment_effect_size, 0.0):
      error_message = (
          "Simulate experiment is currently only supported for A/A simulations"
          " with no treatment effect."
      )
      logger.error(error_message)
      raise NotImplementedError(error_message)

    # Get all cost metrics.
    all_metrics = [design.primary_metric] + design.secondary_metrics
    cost_columns = set(
        [metric.cost_column for metric in all_metrics if metric.cost_column]
    )

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
    is_heavy_up = design.experiment_type == ExperimentType.HEAVY_UP
    is_hold_back = design.experiment_type == ExperimentType.HOLD_BACK

    # Apply the experiment budget to the cost metrics.
    for cost_metric in cost_columns:
      has_cost = data[cost_metric].sum().sum() > 0
      if is_hold_back and has_cost:
        error_message = (
            "Cost metric found in a hold back experiment. This is not"
            " supported."
        )
        logger.error(error_message)
        raise ValueError(error_message)
      for treatment_cell in design.geo_assignment.treatment:
        for treatment_geo in treatment_cell:
          if is_percent_change:
            # Simply apply the percentage change
            data.loc[treatment_dates_mask, (cost_metric, treatment_geo)] *= (
                1 + design.experiment_budget.value
            )
            continue

          budget_value = design.experiment_budget.value
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
          elif is_hold_back:
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
          else:
            error_message = (
                "This shouldn't have happened, but somehow got an experiment"
                " type that is neither heavy up, hold back, but a budget that"
                " is not a percentage change."
            )
            logger.error(error_message)
            raise RuntimeError(error_message)  # pylint: disable=g-doc-exception

          data.loc[treatment_dates_mask, (cost_metric, treatment_geo)] += (
              budget_value * budget_frac
          )

          # Finally make sure the spend does not go negative for any geo
          data.loc[treatment_dates_mask, (cost_metric, treatment_geo)] = (
              data.loc[treatment_dates_mask, (cost_metric, treatment_geo)].clip(
                  lower=0
              )
          )

      # Create the runtime dataset and analyse it
    return self.from_pivoted_data(
        data,
        geo_id_column=self.geo_id_column,
        date_column=self.date_column,
    )
