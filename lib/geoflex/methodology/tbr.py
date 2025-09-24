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

"""Time-Based Regression (TBR) methodology for GeoFleX."""

import datetime as dt
import logging
from typing import Any

import geoflex.data
import geoflex.experiment_design
from geoflex.methodology import _base
import geoflex.utils

from matched_markets.methodology import semantics
from matched_markets.methodology import tbr
from matched_markets.methodology import tbr_iroas
from matched_markets.methodology import tbrdiagnostics

import numpy as np
import pandas as pd
import pydantic


GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
ExperimentDesign = geoflex.experiment_design.ExperimentDesign
GeoAssignment = geoflex.experiment_design.GeoAssignment
Metric = geoflex.metrics.Metric
register_methodology = _base.register_methodology

logger = logging.getLogger(__name__)


class TBRParameters(pydantic.BaseModel):
  """Parameters specific to the TBR methodology for GeoFleX.

  These parameters control the pre-analysis diagnostics and the analysis itself.
  """
  pretest_weeks: int = pydantic.Field(
      ...,
      gt=0,
      description="Number of weeks in pretest period used for model training.")
  use_cooldown: bool = pydantic.Field(
      default=True,
      description="Whether to include the cooldown period in the analysis.")
  model_config = pydantic.ConfigDict(extra="forbid")


@register_methodology(default=True)
class TBR(_base.Methodology):
  """Time-Based Regression (TBR) methodology for GeoFleX.

  This class acts as a wrapper around an original TBR library.

  The methodology first assigns geos randomly with stratification. It then uses
  the original library's diagnostic tools to ensure data quality before
  performing the final analysis using a time-based regression model.
  """

  is_pseudo_experiment = False
  default_methodology_parameter_candidates: dict[str, list[Any]] = {
      "pretest_weeks": [8, 12, 16, 20, 24],
      "use_cooldown": [True],
  }

  def _methodology_is_eligible_for_design_and_data(
      self,
      design: ExperimentDesign,
      pretest_data: GeoPerformanceDataset
  ) -> bool:
    """Checks if the design is eligible for the TBR methodology.

    Original TBR only accepts 2 cell designs as the model structure only
    has one predictor (control) and one outcome (treatment). It also does not
    support pre-specified control or treatment geos, and only supports iROAS
    as a cost-based metric.

    Args:
      design: The experiment design to check.se
      pretest_data: The pretest data (unused in this check).

    Returns:
      True if the design is eligible, False otherwise.
    """
    if design.n_cells != 2:
      logger.info(
          "Original TBR methodology only supports 2-cell (control/treatment)"
          " designs. Got %s cells.", design.n_cells)
      return False

    if design.geo_eligibility.control or any(design.geo_eligibility.treatment):
      logger.info(
          "TBR methodology does not support pre-specified control or treatment"
          " geos when paired with randomized assignment. Please remove them"
          " from the geo_eligibility settings."
      )
      return False

    all_metrics = [design.primary_metric] + design.secondary_metrics
    for metric in all_metrics:
      if metric.cost_column and metric.cost_per_metric:
        logger.info(
            "TBR methodology does not support inverse cost metrics (like CPA)."
            " The metric '%s' is not eligible.",
            metric.name,
        )
        return False

    return True

  def _methodology_assign_geos(
      self,
      experiment_design: ExperimentDesign,
      historical_data: GeoPerformanceDataset,
  ) -> tuple[GeoAssignment, dict[str, Any]]:
    """Assigns geos randomly into control and treatment groups.

    This method randomly assigns geos to control and treatment groups, while
    respecting any constraints on cell volume. It also respects the
    geo eligibility criteria for exclude.

    Args:
      experiment_design: The design to assign geos for.
      historical_data: The dataset used for assignment.
    Raises:
      RuntimeError: If the cell volume constraint type is invalid.
    Returns:
      A tuple containing the GeoAssignment and an empty dictionary.
    """
    eligible_geos = list(
        set(historical_data.geos) - experiment_design.geo_eligibility.exclude
    )

    if (
        experiment_design.cell_volume_constraint.constraint_type
        == geoflex.experiment_design.CellVolumeConstraintType.MAX_GEOS
    ):
      metric_values = None
    elif (
        experiment_design.cell_volume_constraint.constraint_type
        == geoflex.experiment_design.CellVolumeConstraintType.MAX_PERCENTAGE_OF_METRIC
    ):
      metric_values = historical_data.parsed_data.groupby(
          historical_data.geo_id_column
      )[experiment_design.cell_volume_constraint.metric_column].sum()
      metric_values /= metric_values.sum()
    else:
      raise RuntimeError(
          "Got an invalid cell volume constraint type"
          f" {experiment_design.cell_volume_constraint.constraint_type}"
      )

    if metric_values is not None:
      metric_values_dict = metric_values.to_dict()
      metric_values = [
          metric_values_dict.get(geo_id, 0.0) for geo_id in eligible_geos
      ]

    max_metric_per_group = [
        value if value is not None else np.inf
        for value in experiment_design.cell_volume_constraint.values
    ]
    raw_assignments, _ = geoflex.utils.assign_geos_randomly(
        geo_ids=eligible_geos,
        metric_values=metric_values,
        n_groups=experiment_design.n_cells,
        max_metric_per_group=max_metric_per_group,
        rng=experiment_design.get_rng(),
    )

    return (
        GeoAssignment(
            control=raw_assignments[0], treatment=raw_assignments[1:]
        ),
        {},
    )

  def _methodology_analyze_experiment(
      self,
      runtime_data: GeoPerformanceDataset,
      experiment_design: ExperimentDesign,
      experiment_start_date: pd.Timestamp,
      experiment_end_date: pd.Timestamp,
      pretest_period_end_date: pd.Timestamp,
  ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Analyzes the experiment by wrapping the original TBR library.

    Data format from GeoFleX is adapted to format expected by the original
    library. Then typical TBR pre-analysis diagnostics are run to ensure data
    quality. Finally, the TBR model is fitted and the results are adapted back
    to the GeoFleX format.

    Args:
      runtime_data: The runtime data for the experiment.
      experiment_design: The design of the experiment being analyzed.
      experiment_start_date: The start date of the experiment.
      experiment_end_date: The end date of the experiment.
      pretest_period_end_date: The end date of the pretest period.

    Returns:
      A DataFrame with the analysis results and an empty dictionary.
    """
    logger.info("Starting TBR analysis for design %s using original library.",
                experiment_design.design_id)

    try:
      params = TBRParameters.model_validate(
          experiment_design.methodology_parameters)
    except pydantic.ValidationError as e:
      raise ValueError(f"Invalid TBR parameters for analysis: {e}") from e

    analysis_results = []
    all_metrics = ([experiment_design.primary_metric] +
                   experiment_design.secondary_metrics)

    for metric in all_metrics:
      logger.debug("Analyzing metric: %s", metric.name)
      try:
        # adapt data for original tbr library
        adapted_df, tbr_kwargs = self._adapt_data_for_tbr(
            runtime_data, experiment_design, metric, params,
            experiment_start_date, pretest_period_end_date)

        control_total_runtime = self._get_control_total_for_relative(
            adapted_df, metric, tbr_kwargs
        )

        # run diagnostics from the original library
        # removes noisy geos, outlier dates and does final correlation check
        # performing final correlation test between control and treatment
        try:
          diagnostics = tbrdiagnostics.TBRDiagnostics()
          diagnostics.fit(
              adapted_df, target=tbr_kwargs["key_response"],
              **tbr_kwargs
              )

          if not diagnostics.tests_passed():
            test_results = diagnostics.get_test_results()
            # only log warning if test failed but do not stop analysis so
            # user can see results from evaluation layer of GeoFleX
            logger.warning(
                "TBR diagnostics failed for metric '%s'. Results: %s."
                " Proceeding with analysis without diagnostics.",
                metric.name, test_results)
            clean_data = adapted_df
          else:
            logger.debug("TBR diagnostics passed for metric '%s'.", metric.name)
            clean_data = diagnostics.get_data()

        except (ValueError, RuntimeError) as diag_error:
          logger.warning(
              "TBR diagnostics failed with an exception for metric '%s': %s."
              " Proceeding with analysis without diagnostics.",
              metric.name, diag_error)
          clean_data = adapted_df

        # fit model and get summary
        if metric.cost_column:
          model = tbr_iroas.TBRiROAS(use_cooldown=params.use_cooldown)
          model.fit(
              clean_data,
              **tbr_kwargs
              )
          summary = model.summary(level=1.0 - experiment_design.alpha)
        else:
          model = tbr.TBR(use_cooldown=params.use_cooldown)
          model.fit(
              clean_data,
              target=tbr_kwargs["key_response"],
              **tbr_kwargs
              )
          summary = model.summary(level=1.0 - experiment_design.alpha)

        summary_row = summary.iloc[0]
        if metric.cost_column:
          df = summary_row.get("df_resid")
        else:
          df = model.pre_period_model.df_resid

        if pd.isna(df) or df <= 0:
          logger.warning(
              "Could not obtain valid degrees of freedom for metric '%s'."
              " Using an approximation for p-value calculation.",
              metric.name,
          )
          df = 1000  # Default degrees of freedom for approximation
        # adapt results back to GeoFleX format
        result_row = self._adapt_summary_to_geoflex(
            summary_row,
            metric,
            experiment_design.alternative_hypothesis,
            experiment_design.alpha,
            control_total_runtime,
            df
        )
        analysis_results.append(result_row)

      except (ValueError, KeyError, IndexError, RuntimeError) as e:
        logger.error(
            "Analysis for metric '%s' failed: %s",
            metric.name, e, exc_info=True)

        # add placeholder row for failed metric as required by _base.py
        analysis_results.append({
            "metric": metric.name,
            "cell": 1,
            "point_estimate": pd.NA,
            "lower_bound": pd.NA,
            "upper_bound": pd.NA,
            "p_value": pd.NA,
            "point_estimate_relative": pd.NA,
            "lower_bound_relative": pd.NA,
            "upper_bound_relative": pd.NA,
        })

    return pd.DataFrame(analysis_results), {}

  def _get_control_total_for_relative(
      self,
      adapted_df: pd.DataFrame,
      metric: Metric,
      tbr_kwargs: dict[str, Any]
  ) -> float:
    """Calculates total response of the control group during the test period.

    This is the baseline for calculating relative metrics. The GeoFleX
    evaluation layer expects MDE to be provided as a percentage so calculating
    relative metrics instead of absolute metrics is required.

    Args:
      adapted_df: The adapted DataFrame for the original TBR library.
      metric: The metric that was analyzed.
      tbr_kwargs: A dictionary of parameters for the original library.

    Returns:
      The total response of the control group during the test period.
    """
    control_test_df = adapted_df[
        (adapted_df[tbr_kwargs["key_group"]] == tbr_kwargs["group_control"]) &
        (adapted_df[tbr_kwargs["key_period"]] == tbr_kwargs["period_test"])
    ]
    # The response column name is stored in key_response, which is metric.column
    return control_test_df[tbr_kwargs["key_response"]].sum()

  def _adapt_data_for_tbr(
      self,
      runtime_data: GeoPerformanceDataset,
      design: ExperimentDesign,
      metric: Metric,
      params: TBRParameters,
      exp_start_date: pd.Timestamp,
      pre_end_date: pd.Timestamp,
  ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Adapts GeoFleX data structures to a DataFrame for the original library.

    Original TBR library expects a DataFrame with the following columns:
    - date: Date of the data point.
    - geo: Geo ID.
    - group: Group (control or treatment) ID.
    - period: Period (pre-test or test) ID.
    - response: Response metric.
    - cost: Cost metric (optional).

    Args:
      runtime_data: The runtime data for the experiment.
      design: The design of the experiment being analyzed.
      metric: The metric to analyze.
      params: The parameters of the TBR methodology.
      exp_start_date: The start date of the experiment.
      pre_end_date: The end date of the pretest period.

    Returns:
      A tuple containing the adapted DataFrame and a dictionary of
      parameters for the original library.
    """
    pre_start_date = pre_end_date - dt.timedelta(weeks=params.pretest_weeks)
    data_slice = runtime_data.parsed_data[
        (runtime_data.parsed_data[runtime_data.date_column] >= pre_start_date)
    ].copy()

    # define semantics and column names for the original library
    df_names = semantics.DataFrameNameMapping(
        response=metric.column,
        cost=metric.cost_column
        )
    groups = semantics.GroupSemantics()
    periods = semantics.PeriodSemantics()
    tbr_kwargs = {
        "key_date": df_names.date,
        "key_geo": df_names.geo,
        "key_group": df_names.group,
        "key_period": df_names.period,
        "key_response": df_names.response,
        "key_cost": df_names.cost,
        "group_control": groups.control,
        "group_treatment": groups.treatment,
        "period_pre": periods.pre,
        "period_test": periods.test,
        "period_cooldown": periods.cooldown
    }

    # Rename columns to match original library's expectation
    rename_map = {
        runtime_data.geo_id_column: df_names.geo,
        runtime_data.date_column: df_names.date
    }
    data_slice.rename(columns=rename_map, inplace=True)

    # Add 'group' column based on geo assignment
    assignment_map = {
        geo: groups.control for geo in design.geo_assignment.control
        }
    assignment_map.update(
        {geo: groups.treatment for geo in design.geo_assignment.treatment[0]}
        )
    data_slice[df_names.group] = data_slice[df_names.geo].map(assignment_map)
    data_slice = data_slice.dropna(subset=[df_names.group])
    data_slice[df_names.group] = data_slice[df_names.group].astype(int)

    # Add 'period' column
    data_slice[df_names.period] = np.where(
        data_slice[df_names.date] < exp_start_date, periods.pre, periods.test)

    # Handle cooldown period if specified
    if params.use_cooldown and design.evaluation_results:
      cooldown_start = exp_start_date + dt.timedelta(
          weeks=design.runtime_weeks
          )
      data_slice.loc[data_slice[df_names.date] >= cooldown_start,
                     df_names.period] = periods.cooldown

    return data_slice, tbr_kwargs

  def _adapt_summary_to_geoflex(
      self,
      summary_row: pd.Series,
      metric: Metric,
      alternative: str,
      alpha: float,
      control_total_runtime: float,
      df: int,
  ) -> dict[str, Any]:
    """Adapts a result row from the original library to the GeoFleX format.

    Original library returns a summary row as a Series. This function adapts
    that row to the GeoFleX format.

    Args:
      summary_row: A Series containing the summary row from the original
        library.
      metric: The metric that was analyzed.
      alternative: The alternative hypothesis of the experiment design.
      alpha: The alpha level for the confidence interval.
      control_total_runtime: The total response of the control group during the
        test period, used for calculating relative metrics.
      df: The number of degrees of freedom in the posterior distribution.
    Returns:
      A dictionary containing the adapted summary row in the GeoFleX format.
    """
    point_estimate = summary_row.get("estimate", pd.NA)

    # if the analysis failed to produce an estimate, return a null row.
    if pd.isna(point_estimate):
      return {
          "metric": metric.name,
          "cell": 1,
          "point_estimate": pd.NA,
          "lower_bound": pd.NA,
          "upper_bound": pd.NA,
          "p_value": pd.NA,
          "point_estimate_relative": pd.NA,
          "lower_bound_relative": pd.NA,
          "upper_bound_relative": pd.NA,
      }

    if metric.cost_column:
      # For iROAS, bounds are in the summary. p-value is not available.
      lower_bound = summary_row.get("lower", pd.NA)
      upper_bound = summary_row.get("upper", pd.NA)
      p_value = pd.NA
    else:
      scale = summary_row.get("scale", pd.NA)
      if pd.isna(scale):
        return {
            "metric": metric.name,
            "cell": 1,
            "point_estimate": point_estimate,
            "lower_bound": pd.NA,
            "upper_bound": pd.NA,
            "p_value": pd.NA,
            "point_estimate_relative": pd.NA,
            "lower_bound_relative": pd.NA,
            "upper_bound_relative": pd.NA,
        }
      # reconstruct the posterior t-distribution from the model results
      # scipy is imported as sp in the original tbr.py
      dist = tbr.sp.stats.t(df, loc=point_estimate, scale=scale)

      # calculate confidence bounds and p-value based on the hypothesis
      if alternative == "greater":
        lower_bound = dist.ppf(alpha)
        upper_bound = np.inf
        p_value = 1.0 - dist.cdf(0)
      elif alternative == "less":
        lower_bound = -np.inf
        upper_bound = dist.ppf(1.0 - alpha)
        p_value = dist.cdf(0)
      # remaining cases are two-sided
      else:
        alpha_half = alpha / 2.0
        lower_bound = dist.ppf(alpha_half)
        upper_bound = dist.ppf(1.0 - alpha_half)
        p_value = 2.0 * min(dist.cdf(0), 1.0 - dist.cdf(0))

    # For iROAS, relative metrics are not meaningful
    pe_rel, lb_rel, ub_rel = (pd.NA, pd.NA, pd.NA)
    if not metric.cost_column:
      if control_total_runtime > 0:
        pe_rel = (
            point_estimate / control_total_runtime
            if pd.notna(point_estimate)
            else pd.NA
        )
        lb_rel = (
            lower_bound / control_total_runtime
            if pd.notna(lower_bound)
            else pd.NA
        )
        ub_rel = (
            upper_bound / control_total_runtime
            if pd.notna(upper_bound)
            else pd.NA
        )
      else:
        logger.warning(
            "Control total is not positive (%s), cannot calculate relative"
            " metrics for metric '%s'.",
            control_total_runtime,
            metric.name,
        )
    else:
      # For iROAS, return absolute estimates which are already relative
      pe_rel = point_estimate
      lb_rel = lower_bound
      ub_rel = upper_bound

    return {
        "metric": metric.name,
        "cell": 1,
        "point_estimate": point_estimate,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "p_value": p_value,
        "point_estimate_relative": pe_rel,
        "lower_bound_relative": lb_rel,
        "upper_bound_relative": ub_rel
    }
