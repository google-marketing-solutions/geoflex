"""Utils for GeoFleX."""

import base64
import io
import logging
from typing import Annotated
import numpy as np
import pandas as pd
import pydantic
from scipy import stats


logger = logging.getLogger(__name__)


def serialize_dataframe_to_parquet_b64(data: pd.DataFrame) -> str:
  """Serializes a DataFrame to a Base64 encoded Parquet string.

  Used with pydantic for serialization of DataFrames.

  Args:
    data: The DataFrame to serialize.

  Returns:
    The Base64 encoded Parquet string.
  """
  # if data.empty and not data.columns.tolist():
  #   pass
  parquet_bytes = data.to_parquet()
  return base64.b64encode(parquet_bytes).decode("utf-8")


def deserialize_df_from_parquet_b64_if_string(
    value: str | pd.DataFrame, handler: pydantic.ValidatorFunctionWrapHandler
) -> pd.DataFrame:
  """Deserializes a DataFrame from a Base64 encoded Parquet string.

  Used with pydantic for deserialization of DataFrames from json.

  Args:
    value: The Base64 encoded Parquet string.
    handler: The handler which represents all the other validators. This is
      called after the serialization to pandas.

  Returns:
    The deserialized DataFrame.
  """
  if not isinstance(value, str):  # Not a string
    return handler(value)

  if not value:  # Empty string
    return handler(pd.DataFrame())

  try:
    base64_decoded_bytes = base64.b64decode(value.encode("utf-8"))
    if not base64_decoded_bytes:  # Empty bytes after decoding
      return handler(pd.DataFrame())
    parquet_file_like = io.BytesIO(base64_decoded_bytes)
    return handler(pd.read_parquet(parquet_file_like))
  except Exception as e:
    error_message = f"Failed to deserialize DataFrame from Parquet string: {e}"
    logger.error(error_message)
    raise ValueError(error_message) from e


# Use this type annotation instead of pd.DataFrame to serialize and deserialize
# DataFrames to and from Parquet strings.
ParquetDataFrame = Annotated[
    pd.DataFrame,
    pydantic.PlainSerializer(
        serialize_dataframe_to_parquet_b64, when_used="json"
    ),
    pydantic.WrapValidator(deserialize_df_from_parquet_b64_if_string),
]


def infer_p_value(
    mean: float,
    confidence_interval: tuple[float, float],
    alpha: float,
    alternative_hypothesis: str = "two-sided",
):
  """Infers a p-value from a sample mean and confidence interval.

  Assumes a normal distribution and a null hypothesis mean of 0.

  Args:
      mean: The sample mean.
      confidence_interval: A tuple containing the lower and upper bounds of the
        confidence interval. For two-sided tests, this is a standard (lower,
        upper) interval. For one-sided tests: - If alternative_hypothesis is
        "greater", confidence_interval is (lower_bound, np.inf) - If
        alternative_hypothesis is "less", confidence_interval is (-np.inf,
        upper_bound)
      alpha: The alpha level used to construct the confidence interval (e.g.,
        0.05 for a 95% CI).
      alternative_hypothesis:  The type of alternative hypothesis. Must be one
        of "two-sided", "greater", or "less". Defaults to "two-sided".

  Returns:
      float: The p-value. Returns 0.0 if the confidence interval
             is invalid (e.g., bounds are the same in a two-sided test).

  Raises:
      ValueError: If alpha is not between 0 and 1, or if alternative_hypothesis
                  is not one of the allowed values.
      TypeError:  If confidence_interval is not a tuple.
      ValueError: If confidence_interval is a tuple of the wrong length.
  """
  if not 0 < alpha < 1:
    error_message = f"Alpha must be between 0 and 1, but got {alpha}"
    logger.error(error_message)
    raise ValueError(error_message)

  if alternative_hypothesis not in ["two-sided", "greater", "less"]:
    error_message = (
        "Alternative hypothesis must be one of 'two-sided', 'greater', or"
        " 'less'."
    )
    logger.error(error_message)
    raise ValueError(error_message)

  lower_bound, upper_bound = confidence_interval

  if alternative_hypothesis == "two-sided":
    if lower_bound == upper_bound:
      logger.warning(
          "Confidence interval is invalid for two-sided test: lower bound"
          " (%s) is equal to upper bound (%s). Returning"
          " p-value of 0.0.",
          lower_bound,
          upper_bound,
      )
      return 0.0

    z_critical = stats.norm.ppf(1 - alpha / 2)
    standard_error = (mean - lower_bound) / z_critical

  elif alternative_hypothesis == "greater":
    if upper_bound != np.inf:
      error_message = (
          "For alternative='greater', upper bound must be np.inf, but got"
          f" {upper_bound}"
      )
      logger.error(error_message)
      raise ValueError(error_message)

    z_critical = stats.norm.ppf(1 - alpha)  # Use one-sided alpha
    standard_error = (mean - lower_bound) / z_critical

  else:  # alternative_hypothesis == "less"
    if lower_bound != -np.inf:
      error_message = (
          "For alternative='less', lower bound must be -np.inf, but got"
          f" {lower_bound}"
      )
      logger.error(error_message)
      raise ValueError(error_message)

    z_critical = stats.norm.ppf(1 - alpha)  # Use one-sided alpha
    standard_error = (
        upper_bound - mean
    ) / z_critical  # changed to (upper - mean)

  # Calculate the z-statistic.
  z_statistic = mean / standard_error

  # Calculate the p-value based on the alternative hypothesis
  if alternative_hypothesis == "two-sided":
    p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
  elif alternative_hypothesis == "greater":
    p_value = 1 - stats.norm.cdf(z_statistic)
  else:  # alternative_hypothesis == "less"
    p_value = stats.norm.cdf(z_statistic)

  return p_value
