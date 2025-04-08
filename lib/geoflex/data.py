"""Data for a geoflex experiment design."""

from collections.abc import Mapping
import datetime as dt
import functools
import itertools
import warnings

import pandas as pd
import pydantic


class GeoPerformanceDataset(pydantic.BaseModel):
  """Performance data for each geo and day."""

  data: pd.DataFrame
  geo_id_column: str = "geo_id"
  date_column: str = "date"
  allow_missing_dates: bool = False

  model_config = pydantic.ConfigDict(
      arbitrary_types_allowed=True, extra="forbid"
  )

  @pydantic.model_validator(mode="before")
  @classmethod
  def check_input_data_column(cls, values):
    """Checks that input data has required columns."""
    data = values.get("data")
    geo_col = values.get("geo_id_column", "geo_id")
    date_col = values.get("date_column", "date")

    if data is None:
      return values

    if not isinstance(data, pd.DataFrame):
      return values

    required_columns = [geo_col, date_col]
    missing_cols = [col for col in required_columns if col not in data.columns]

    if missing_cols:
      raise ValueError(
          f"Data must have the following columns: {', '.join(required_columns)}"
      )

    return values

  @pydantic.model_validator(mode="before")
  @classmethod
  def check_input_geos_na(cls, values):
    """Checks that input geos are not NA."""
    data = values.get("data")
    geo_col = values.get("geo_id_column", "geo_id")

    if data is None or geo_col not in data.columns:
      return values

    if data[geo_col].isna().any():
      raise ValueError(
          f"Data must have non-NA values in the {geo_col} column."
      )
    return values

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
    unique_dates = pd.to_datetime(
        input_data[self.date_column]).unique().tolist()

    if not unique_geos:
      raise ValueError("No unique geos found in the data.")
    if not unique_dates:
      raise ValueError("No unique dates found in the data.")

    index = pd.DataFrame(
        itertools.product(unique_geos, unique_dates),
        columns=[self.geo_id_column, self.date_column],
    )

    all_metric_columns = (  # Automatically find all metric columns.
        input_data.select_dtypes("number").columns.values.tolist()
    )
    if not all_metric_columns:
      raise ValueError("No metric columns found in the data.")

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
      raise ValueError(
          "Data processing resulted in non-unique 'geo_id_date' index values. "
          f"Duplicated indices start with: {', '.join(duplicated_indices)}"
      )

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
      raise ValueError("Data must have at least 2 unique dates.")

    date_diffs = pd.Series(unique_dates).diff().dt.days.dropna()

    if date_diffs.empty:
      warnings.warn("Cannot determine data frequency")
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
        raise ValueError(
            "Cannot determine data frequency. "
            "Most common and median frequencies are not 1, 7, or integer."
        )

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
        warnings.warn(message, UserWarning)
      else:
        raise ValueError(message + " Set allow_missing_dates=True to ignore.")

    return self

