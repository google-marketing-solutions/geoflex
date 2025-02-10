"""Data for a geoflex experiment design."""

import datetime as dt
import functools
import itertools
import pandas as pd
import pydantic


class GeoPerformanceDataset(pydantic.BaseModel):
  """Performance data for each geo and day."""

  data: pd.DataFrame
  geo_id_column: str = "geo_id"
  date_column: str = "date"
  cost_column: str = "cost"
  response_columns: list[str] = ["revenue"]

  model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

  @functools.cached_property
  def parsed_data(self) -> pd.DataFrame:
    """Returns the parsed data.

    The parsed data is a copy of the data with the date column converted to
    a date object, the cost and all response columns converted to floats,
    the geo_id converted to a string, and a geo_id_date column added as a unique
    identifier for the dataset.
    """
    unique_geos = self.data[self.geo_id_column].unique().tolist()
    unique_dates = self.data[self.date_column].unique().tolist()
    index = pd.DataFrame(
        itertools.product(unique_geos, unique_dates),
        columns=[self.geo_id_column, self.date_column],
    )

    all_metric_columns = [self.cost_column] + self.response_columns
    all_columns = [
        self.geo_id_column,
        self.date_column,
    ] + all_metric_columns

    parsed_data = (
        index.merge(
            self.data[all_columns],
            on=[self.geo_id_column, self.date_column],
            how="left",
        )
        .copy()
        .assign(
            geo_id_date=lambda df: df[self.geo_id_column]
            + "__"
            + df[self.date_column]
        )
        .astype(
            {
                self.geo_id_column: "object",
                self.date_column: "datetime64[ns]",
            }
            | {column: "float64" for column in all_metric_columns}
        )
        .set_index("geo_id_date")
    )

    parsed_data[all_metric_columns] = (
        parsed_data[all_metric_columns].fillna(0.0).copy()
    )
    return parsed_data

  @functools.cached_property
  def geos(self) -> list[str]:
    """Returns the geos in the data."""
    return self.parsed_data[self.geo_id_column].unique().tolist()

  @functools.cached_property
  def dates(self) -> list[dt.datetime]:
    """Returns the dates in the data."""
    return self.parsed_data[self.date_column].unique().tolist()

  @pydantic.model_validator(mode="after")
  def check_data_has_required_columns(self) -> "GeoPerformanceDataset":
    """Checks that the data has the required columns."""
    required_columns = [
        self.geo_id_column,
        self.date_column,
        self.cost_column,
    ] + self.response_columns
    if not all(column in self.data.columns for column in required_columns):
      required_columns_str = ", ".join(required_columns)
      raise ValueError(
          f"Data must have the following columns: {required_columns_str}. "
          "You can override these column names by setting the corresponding "
          "attributes in the GeoPerformanceDataset class."
      )
    return self

  @pydantic.model_validator(mode="after")
  def check_data_has_unique_geos_and_dates(
      self,
  ) -> "GeoPerformanceDataset":
    """Checks that the data has unique geos and dates."""
    if not self.parsed_data.index.is_unique:
      raise ValueError("Data must have unique geos and dates. ")
    return self
