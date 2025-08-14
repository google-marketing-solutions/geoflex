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

"""Utils for GeoFleX."""

import base64
import io
import logging
from typing import Annotated
from feedx import statistics
import numpy as np
import pandas as pd
import pydantic
from scipy import stats

relative_difference_confidence_interval = (
    statistics.relative_difference_confidence_interval
)
absolute_difference_confidence_interval = (
    statistics.absolute_difference_confidence_interval
)
ttest_from_stats = statistics.ttest_from_stats


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


def assign_geos_randomly(
    geo_ids: list[str],
    n_groups: int,
    rng: np.random.Generator,
    metric_values: list[float] | None = None,
    max_metric_per_group: list[float] | None = None,
    pre_assigned_geos: dict[str, int] | None = None,
) -> tuple[list[list[str]], list[float]]:
  """Randomly assigns geo units into up to N separate groups.

  This will first assign the pre-assigned geos to their target groups, and then
  randomly assign the remaining geos to the groups by sorting them randomly
  and assigning them one by one until the groups are full.

  Args:
      geo_ids: A list of unique string identifiers for each geo unit.
      n_groups: The number of groups to assign geos into.
      rng: A numpy random number generator.
      metric_values: An optional list of float values, where metric[i] is the
        metric for geo_ids[i].
      max_metric_per_group: A list of float values, where
        max_metric_per_group[j] is the maximum total metric allowed for group j.
        The length of this list must be equal to n_groups.
      pre_assigned_geos: An optional dictionary where keys are geo_ids (str) and
        values are group_indices (int) for geos that must be placed in a
        specific group. Defaults to None.

  Returns:
      A tuple:
        First element is a list of lists, where each inner list contains the
          geo_ids assigned to that group. The outer list will have a length of
          n_groups.
        Second element is a list of the total metric values for each group.

  Raises:
      ValueError: If input constraints are violated (e.g., mismatched lengths,
        invalid group index, pre-assignment exceeds capacity, or empty group
        after random assignment).
  """
  if metric_values is None:
    metric_values = [1.0] * len(geo_ids)  # Ensure float for consistency

  if n_groups <= 0:
    error_message = "n_groups must be greater than 0."
    logger.error(error_message)
    raise ValueError(error_message)

  if not geo_ids:
    # If no geos are provided, return empty groups.
    return [[]] * n_groups, [0.0] * n_groups

  if max_metric_per_group is None:
    max_metric_per_group = [np.inf] * n_groups

  if len(geo_ids) != len(metric_values):
    error_message = (
        "geo_ids and metric must have the same length. Got"
        f" {len(geo_ids)} geo_ids and {len(metric_values)} metric values."
    )
    logger.error(error_message)
    raise ValueError(error_message)

  if n_groups != len(max_metric_per_group):
    error_message = (
        "n_groups must be equal to the length of max_metric_per_group."
    )
    logger.error(error_message)
    raise ValueError(error_message)

  assigned_geos_in_groups: list[list[str]] = [[] for _ in range(n_groups)]
  current_group_metrics = np.zeros(n_groups, dtype=float)

  geo_to_metric_map = {gid: met for gid, met in zip(geo_ids, metric_values)}
  geo_ids_for_assignment = geo_ids.copy()

  if pre_assigned_geos:
    for geo_to_preassign, target_group_idx in pre_assigned_geos.items():
      if geo_to_preassign not in geo_to_metric_map:
        raise ValueError(
            f"Pre-assigned geo_id '{geo_to_preassign}' not found in main"
            " geo_ids list."
        )

      if not (0 <= target_group_idx < n_groups):
        raise ValueError(
            f"Invalid target_group_idx {target_group_idx} for pre-assigned geo"
            f" '{geo_to_preassign}'. Must be between 0 and {n_groups-1}."
        )

      geo_metric_val = geo_to_metric_map[geo_to_preassign]
      assigned_geos_in_groups[target_group_idx].append(geo_to_preassign)
      current_group_metrics[target_group_idx] += geo_metric_val
      if geo_to_preassign in geo_ids_for_assignment:
        geo_ids_for_assignment.remove(geo_to_preassign)

  rng.shuffle(geo_ids_for_assignment)

  for geo_id in geo_ids_for_assignment:
    geo_metric = geo_to_metric_map[geo_id]
    eligible_groups_indices = []
    for i in range(n_groups):
      if (
          current_group_metrics[i] + geo_metric
          <= max_metric_per_group[i] + 1e-9  # Tolerance check
      ):
        eligible_groups_indices.append(i)

    if not eligible_groups_indices:
      continue

    chosen_group_idx = rng.choice(eligible_groups_indices)
    assigned_geos_in_groups[chosen_group_idx].append(geo_id)
    current_group_metrics[chosen_group_idx] += geo_metric

  for i, group in enumerate(assigned_geos_in_groups):
    if not group:
      raise ValueError(f"Group {i} is empty after assigning all geos.")

  return assigned_geos_in_groups, current_group_metrics.tolist()


def get_summary_statistics_from_standard_errors(
    impact_estimate: float,
    impact_standard_error: float,
    degrees_of_freedom: int,
    alternative_hypothesis: str,
    alpha: float,
    baseline_estimate: float | None = None,
    baseline_standard_error: float | None = None,
    impact_baseline_corr: float | None = None,
    invert_result: bool = False,
) -> dict[str, float]:
  """Calculates the summary statistics from the standard errors.

  This calculates the summary statistics from the standard errors. It calculates
  the point estimate, confidence intervals, and p-value. It also calculates the
  relative difference confidence interval if the impact is not cost per metric
  or metric per cost, and if the baseline and treatment groups are both
  positive.

  Args:
    impact_estimate: The estimated impact of the treatment on the metric.
    impact_standard_error: The standard error of the estimated impact.
    degrees_of_freedom: The degrees of freedom of the estimate.
    alternative_hypothesis: The alternative hypothesis of the test.
    alpha: The alpha level of the confidence interval.
    baseline_estimate: The estimated baseline metric, used to calculate the
      relative difference. If None, the relative difference is not calculated.
    baseline_standard_error: The standard error of the baseline metric.
    impact_baseline_corr: The correlation coefficient between the impact and the
      baseline.
    invert_result: Whether to invert the result. If True, the resulting summary
      statistics are for 1/y rather than y. This is used for cost per metric
      style metrics, like CPiA.

  Returns:
    A dictionary containing the summary statistics, including the point
    estimate, confidence intervals, p-value, and relative difference point
    estimate and confidence interval if applicable.
  """

  p_value = ttest_from_stats(
      point_estimate=impact_estimate,
      standard_error=impact_standard_error,
      degrees_of_freedom=degrees_of_freedom,
      alternative=alternative_hypothesis,
  )[1]

  if invert_result:
    # If we want to invert the result, we do this by using the relative
    # difference confidence interval. This gives us the esimtate of 1/impact.
    impact_sign = np.sign(impact_estimate)
    abs_impact_estimate = np.abs(impact_estimate)

    point_estimate = 1.0 / abs_impact_estimate
    lower_bound, upper_bound = relative_difference_confidence_interval(
        mean_1=1.0,
        mean_2=abs_impact_estimate,
        standard_error_1=0.0,
        standard_error_2=impact_standard_error,
        corr=0.0,
        degrees_of_freedom=degrees_of_freedom,
        alternative=alternative_hypothesis,
        alpha=alpha,
    )

    # relative_difference_confidence_interval calculates the CI on (y2-y1)/y1
    # but we want y2/y1 (in out case y2=1), so we add 1 to the results.
    lower_bound += 1
    upper_bound += 1

    if impact_sign == -1:
      # If it was originally a negative impact we need to make it negative again
      lower_bound *= -1
      upper_bound *= -1
      point_estimate *= -1

  else:
    # If it's not cost per metric we can use the absolute difference confidence
    # interval
    point_estimate = impact_estimate
    lower_bound, upper_bound = absolute_difference_confidence_interval(
        mean_1=impact_estimate,
        mean_2=0.0,
        standard_error_1=impact_standard_error,
        standard_error_2=0.0,
        corr=0.0,
        degrees_of_freedom=degrees_of_freedom,
        alternative=alternative_hypothesis,
        alpha=alpha,
    )

  if (
      invert_result
      or (baseline_estimate is None)
      or ((baseline_estimate + impact_estimate) <= 0.0)
      or (baseline_estimate <= 0.0)
  ):
    # Do not calculate the relative effect if inverting the result, if the
    # baseline is not provided, or if either the baseline or treatment group is
    # not positive
    point_estimate_relative = pd.NA
    lower_bound_relative = pd.NA
    upper_bound_relative = pd.NA
  else:
    point_estimate_relative = impact_estimate / baseline_estimate

    lower_bound_relative, upper_bound_relative = (
        relative_difference_confidence_interval(
            mean_1=baseline_estimate + impact_estimate,
            mean_2=baseline_estimate,
            standard_error_1=impact_standard_error,
            standard_error_2=baseline_standard_error,
            corr=impact_baseline_corr,
            degrees_of_freedom=degrees_of_freedom,
            alternative=alternative_hypothesis,
            alpha=alpha,
        )
    )

  return {
      "point_estimate": point_estimate,
      "lower_bound": lower_bound,
      "upper_bound": upper_bound,
      "point_estimate_relative": point_estimate_relative,
      "lower_bound_relative": lower_bound_relative,
      "upper_bound_relative": upper_bound_relative,
      "p_value": p_value,
  }
