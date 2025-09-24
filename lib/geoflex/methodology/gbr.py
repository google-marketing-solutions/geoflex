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

"""Our implementation of the GBR methodology."""

import datetime as dt
import logging
from typing import Any
import geoflex.data
import geoflex.experiment_design
from geoflex.methodology import _base
import geoflex.metrics
import geoflex.utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


ExperimentDesign = geoflex.experiment_design.ExperimentDesign
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
GeoAssignment = geoflex.experiment_design.GeoAssignment
register_methodology = _base.register_methodology
Metric = geoflex.metrics.Metric


logger = logging.getLogger(__name__)

# We use mathematical notation in variables, disabling the linter.
# pylint: disable=invalid-name


@register_methodology(default=True)
class GBR(_base.Methodology):
  """The Geo Based Regression (GBR) methodology.

  This is a simple geo testing methodology which aggregates all geos to have
  a pre-test and post-test value, and then performs a regression to estimate
  the impact. It works well with lots of geos.

  https://research.google/pubs/measuring-ad-effectiveness-using-geo-experiments/

  Design:
    Geos are split with stratification on the primary response metric.

  Evaluation:
    The evaluation is done with a linear regression. The pre-test period is
    taken to be the same length as the experiment runtime, so an 8 week
    experiment will use an 8-week pre-period.
  """

  default_methodology_parameter_candidates = {
      "linear_model_type": ["wls", "robust_ols"]
  }

  def _methodology_is_eligible_for_design_and_data(
      self, design: ExperimentDesign, pretest_data: GeoPerformanceDataset
  ) -> bool:
    """Checks if the methodology is eligible for the given design.

    Not eligible if geos are forced into a control
    or treatment group, because this methodology will randomly assign the
    geos.

    Not eligible if the metric values are zero or negative.

    Args:
      design: The design to check against.
      pretest_data: The dataset to check against.

    Returns:
      True if the methodology is eligible for the design and data
    """

    has_geo_eligibility_constraints = design.geo_eligibility is not None and (
        design.geo_eligibility.control or any(design.geo_eligibility.treatment)
    )
    if has_geo_eligibility_constraints:
      logger.info(
          "GBR ineligible for design %s: There are constraints on the geos "
          "in control or treatment, and GBR must assign them randomly.",
          design.design_id,
      )
      return False

    metrics = [design.primary_metric] + design.secondary_metrics
    metric_columns = list(set([metric.column for metric in metrics]))
    cost_columns = list(
        set([
            metric.cost_column
            for metric in metrics
            if metric.cost_per_metric or metric.metric_per_cost
        ])
    )

    # If using WLS, then the costs and metrics must be positive everywhere
    # because they will be used for weights and negative weights break the
    # method.
    if design.methodology_parameters["linear_model_type"] == "wls":
      # If the costs are zero everywhere, then it's fine, and if they are
      # positive everywhere it's fine.
      # However if they are sometimes positive but sometimes zero or negative,
      # then this will fail
      if cost_columns:
        costs_all_zero = (
            (pretest_data.pivoted_data[cost_columns] == 0).sum().all()
        )
        costs_all_positive = (
            (pretest_data.pivoted_data[cost_columns] > 0).sum().all()
        )
        if not (costs_all_positive or costs_all_zero):
          logger.info(
              "GBR ineligible for design %s: Cost column totals contain a mix"
              " of some positive and some zero or negative values in the"
              " pretest data. For WLS, costs must be all positive or all zero,"
              " but not a mix.",
              design.design_id,
          )
          return False

      # If the metrics are ever zero or negative it also fails
      if not (pretest_data.pivoted_data[metric_columns] > 0).sum().all():
        logger.info(
            "GBR ineligible for design %s: Metric column totals contain some"
            " zero or negative values in the pretest data. For WLS, metrics"
            " must be all positive.",
            design.design_id,
        )
        return False

    # If there are too few geos it won't work
    geos_per_cell = len(pretest_data.geos) / design.n_cells
    if geos_per_cell < 4:
      logger.info(
          "GBR is ineligible for design %s: GBR requires at least 4 geos per"
          " cell on average, but got %s geos and %s cells",
          design.design_id,
          len(pretest_data.geos),
          design.n_cells,
      )
      return False

    return True

  def _methodology_assign_geos(
      self,
      experiment_design: ExperimentDesign,
      historical_data: GeoPerformanceDataset,
  ) -> tuple[GeoAssignment, dict[str, Any]]:
    """Assigns all geos to control or treatment.

    This assigns the geos randomly, while making sure to meet the constraints.

    Args:
      experiment_design: The design to assign geos for.
      historical_data: The dataset to assign geos for.

    Returns:
      The geo assignment.

    Raises:
      RuntimeError: If the cell volume constraint type is invalid.
    """
    eligible_geos = list(
        set(historical_data.geos) - experiment_design.geo_eligibility.exclude
    )

    if (
        experiment_design.cell_volume_constraint.constraint_type
        == geoflex.CellVolumeConstraintType.MAX_GEOS
    ):
      metric_values = None
    elif (
        experiment_design.cell_volume_constraint.constraint_type
        == geoflex.CellVolumeConstraintType.MAX_PERCENTAGE_OF_METRIC
    ):
      metric_values = historical_data.parsed_data.groupby(
          historical_data.geo_id_column
      )[experiment_design.cell_volume_constraint.metric_column].sum()
      metric_values /= metric_values.sum()
    else:
      error_message = (
          "Got an invalid cell volume constraint type"
          f" {experiment_design.cell_volume_constraint.constraint_type}"
      )
      logger.error(error_message)
      raise RuntimeError(error_message)

    if metric_values is not None:
      metric_values_dict = metric_values.to_dict()
      metric_values = [metric_values_dict[geo_id] for geo_id in eligible_geos]

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

  def _split_data_into_pretest_and_experiment(
      self,
      runtime_data: GeoPerformanceDataset,
      experiment_start_date: pd.Timestamp,
      experiment_end_date: pd.Timestamp,
      runtime_weeks: int,
      pretest_period_end_date: pd.Timestamp,
  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    pretest_start_date = pretest_period_end_date - dt.timedelta(
        weeks=runtime_weeks
    )

    is_pretest = (
        runtime_data.parsed_data[runtime_data.date_column] >= pretest_start_date
    ) & (
        runtime_data.parsed_data[runtime_data.date_column]
        < pretest_period_end_date
    )
    is_runtime = (
        runtime_data.parsed_data[runtime_data.date_column]
        >= experiment_start_date
    ) & (
        runtime_data.parsed_data[runtime_data.date_column] < experiment_end_date
    )

    if not is_pretest.any() or not is_runtime.any():
      error_message = (
          "No data in the pretest or runtime period for GBR. This is likely"
          " because the experiment start date is too close to the start or the"
          " end of the provided data."
      )
      logger.error(error_message)
      raise RuntimeError(error_message)

    pretest_data = runtime_data.parsed_data[is_pretest].copy()
    experiment_data = runtime_data.parsed_data[is_runtime].copy()

    n_pretest_dates = pretest_data[runtime_data.date_column].nunique()
    n_experiment_dates = experiment_data[runtime_data.date_column].nunique()
    if n_pretest_dates != n_experiment_dates:
      logger.warning(
          "The pretest and experiment data have different numbers of dates."
          " %s pretest dates != %s experiment dates. This is not necessarily"
          " a problem as long as both have complete weeks.",
          n_pretest_dates,
          n_experiment_dates,
      )

    return pretest_data, experiment_data

  def _construct_geo_level_data(
      self,
      all_metrics: list[Metric],
      pretest_data: pd.DataFrame,
      experiment_data: pd.DataFrame,
      geo_id_column: str,
      all_geos: list[str],
      linear_model_type: str,
      geo_assignment: GeoAssignment,
  ) -> pd.DataFrame:
    all_metric_columns = set()
    for metric in all_metrics:
      all_metric_columns.add(metric.column)
      if metric.cost_per_metric or metric.metric_per_cost:
        all_metric_columns.add(metric.cost_column)
    all_metric_columns = list(all_metric_columns)

    all_cost_columns = set()
    for metric in all_metrics:
      if metric.cost_per_metric or metric.metric_per_cost:
        all_cost_columns.add(metric.cost_column)
    all_cost_columns = list(all_cost_columns)

    geo_data = pd.DataFrame({
        geo_id_column: all_geos,
        "geo_assignment": geo_assignment.make_geo_assignment_array(all_geos),
    }).set_index(geo_id_column)

    geo_data = geo_data.loc[
        geo_data["geo_assignment"] >= 0.0
    ]  # Exclude excluded geos

    for column in all_metric_columns:
      geo_data[f"{column}_pretest"] = pretest_data.groupby(geo_id_column)[
          column
      ].sum()
      geo_data[f"{column}_runtime"] = experiment_data.groupby(geo_id_column)[
          column
      ].sum()

      if (geo_data[f"{column}_pretest"] <= 0.0).any():
        # If any of the metrics are non-positive, then we can't use WLS because
        # it assumes everything is positive. This shouldn't happen but in
        # case we fall back to robust ols.
        linear_model_type = "robust_ols"

    for cost_column in all_cost_columns:
      geo_data[f"{cost_column}_pretest"] = pretest_data.groupby(geo_id_column)[
          cost_column
      ].sum()
      geo_data[f"{cost_column}_runtime"] = experiment_data.groupby(
          geo_id_column
      )[cost_column].sum()

      some_costs_not_positive = (
          geo_data[f"{cost_column}_pretest"] <= 0.0
      ).any() and not (geo_data[f"{cost_column}_pretest"] == 0.0).all()

      if some_costs_not_positive:
        # If any of the costs are non-positive (but not all), then we can't use
        # WLS because it assumes everything is positive. This shouldn't happen
        # but in case we fall back to robust ols.
        linear_model_type = "robust_ols"

    # Missing values are assumed to be 0.0
    if geo_data.isna().any().any():
      logger.warning(
          "After aggregating to geo level, the data has missing values in the"
          " metrics or the costs per geo, filling with 0.0."
      )
      geo_data = geo_data.fillna(0.0)

    # Add all the cost differentials. If the costs are all zero in the pretest
    # period, then
    for cost_column in all_cost_columns:
      geo_data = self._add_cost_differential(
          geo_data, cost_column, linear_model_type
      )

    return geo_data, linear_model_type

  def _add_cost_differential(
      self, geo_data: pd.DataFrame, cost_column: str, model_type: str
  ) -> pd.DataFrame:
    """Get the cost differential."""
    pre_period_cost_always_0 = (geo_data[f"{cost_column}_pretest"] == 0).all()

    if pre_period_cost_always_0:
      geo_data[f"{cost_column}_cost_differential"] = geo_data[
          f"{cost_column}_runtime"
      ].copy()
      return geo_data

    is_control = geo_data["geo_assignment"] == 0
    is_treatment = ~is_control
    x = geo_data.loc[is_control, f"{cost_column}_pretest"]
    y = geo_data.loc[is_control, f"{cost_column}_runtime"]

    if model_type == "wls":
      w = 1.0 / x
    elif model_type == "robust_ols":
      w = np.ones_like(x)
    else:
      error_message = f"Invalid model type {model_type}"
      logger.error(error_message)
      raise RuntimeError(error_message)

    x_is_constant = np.all(x == x[0])
    if x_is_constant:
      y_mean = (y * w).sum() / w.sum()
      coefs = [0.0, y_mean]
    else:
      coefs = np.polyfit(x, y, deg=1, w=w)

    geo_data[f"{cost_column}_cost_differential"] = 0.0
    geo_data.loc[is_treatment, f"{cost_column}_cost_differential"] = (
        geo_data.loc[is_treatment, f"{cost_column}_runtime"]
        - coefs[0] * geo_data.loc[is_treatment, f"{cost_column}_pretest"]
        - coefs[1]
    )

    return geo_data

  def _fit_linear_model(
      self, cell_data: pd.DataFrame, metric: Metric, model_type: str
  ) -> tuple[pd.Series, pd.DataFrame, float]:
    """Fits the linear model and returns the parameters and cov matrix."""
    X = cell_data.copy()
    y = X[f"{metric.column}_runtime"].copy()

    if model_type == "wls":

      w = 1.0 / X[f"{metric.column}_pretest"].copy()

      if metric.cost_per_metric or metric.metric_per_cost:
        X["cost_differential"] = X[
            f"{metric.cost_column}_cost_differential"
        ].copy()
      else:
        X["cost_differential"] = X["geo_assignment"].copy()

      X = sm.add_constant(X[[f"{metric.column}_pretest", "cost_differential"]])

      # De-mean the pretest metric
      pretest_mean = X[f"{metric.column}_pretest"].mean()
      X[f"{metric.column}_pretest"] -= pretest_mean

      wls_results = sm.WLS(y, X, weights=w).fit()
      covariance = (
          wls_results.cov_params()
      )  # Non robust - robustness handled by weights
      params = wls_results.params
      degrees_of_freedom = float(wls_results.df_resid)

    elif model_type == "robust_ols":

      if metric.cost_per_metric or metric.metric_per_cost:
        X["cost_differential"] = X[
            f"{metric.cost_column}_cost_differential"
        ].copy()
      else:
        X["cost_differential"] = X["geo_assignment"].copy()

      X = sm.add_constant(X[[f"{metric.column}_pretest", "cost_differential"]])

      # De-mean the pretest metric
      pretest_mean = X[f"{metric.column}_pretest"].mean()
      X[f"{metric.column}_pretest"] -= pretest_mean

      ols_results = sm.OLS(y, X).fit()
      covariance = pd.DataFrame(
          ols_results.cov_HC3,  # Robust standard errors used.
          index=ols_results.params.index.values,
          columns=ols_results.params.index.values,
      )
      params = ols_results.params
      degrees_of_freedom = float(ols_results.df_resid)

    else:
      error_message = f"Invalid model type {model_type}"
      logger.error(error_message)
      raise RuntimeError(error_message)

    # Const demean corrected is just used for plotting
    # The regular const is used for analysis because it accurately captures the
    # baseline metrics inside the constant. The demean corrected const is just
    # used for plotting to make the linear regression line easy to plot with
    # the original non-demeaned data.
    params["const_demean_corrected"] = (
        params["const"] - pretest_mean * params[f"{metric.column}_pretest"]
    )
    return params, covariance, degrees_of_freedom

  def _get_summary_statistics(
      self,
      metric: Metric,
      n_treatment_geos: int,
      params: pd.Series,
      covariance: pd.DataFrame,
      degrees_of_freedom: int,
      alternative_hypothesis: str,
      alpha: float,
  ) -> dict[str, float]:
    """Calculates the summary statistics from the result of the linear model."""

    impact_estimate = params["cost_differential"]
    impact_standard_error = np.sqrt(
        covariance.loc["cost_differential", "cost_differential"]
    )

    if metric.cost_per_metric or metric.metric_per_cost:
      # Don't calculate the relative effect for cost per metric or metric per
      # cost. The relative effect is not meaningful for these metrics.
      baseline_estimate = None
      baseline_standard_error = None
      impact_baseline_corr = None
    else:
      # If not cost per metric or metric per cost, then calculate the relative
      # effect.
      baseline_estimate = params["const"]
      baseline_standard_error = np.sqrt(covariance.loc["const", "const"])
      impact_baseline_corr = covariance.loc[
          "cost_differential", "const"
      ] / np.sqrt(
          covariance.loc["const", "const"]
          * covariance.loc["cost_differential", "cost_differential"]
      )

      # If not cost per metric or metric per cost, then the absolute impact
      # is the impact per geo. We need the total so multiply by the number of
      # geos.
      impact_estimate *= n_treatment_geos
      impact_standard_error *= n_treatment_geos
      baseline_estimate *= n_treatment_geos
      baseline_standard_error *= n_treatment_geos

    summary_statistics = (
        geoflex.utils.get_summary_statistics_from_standard_errors(
            impact_estimate=impact_estimate,
            impact_standard_error=impact_standard_error,
            degrees_of_freedom=degrees_of_freedom,
            alternative_hypothesis=alternative_hypothesis,
            alpha=alpha,
            invert_result=metric.cost_per_metric,
            baseline_estimate=baseline_estimate,
            baseline_standard_error=baseline_standard_error,
            impact_baseline_corr=impact_baseline_corr,
        )
    )

    summary_statistics["metric"] = metric.name
    return summary_statistics

  def _methodology_analyze_experiment(
      self,
      runtime_data: GeoPerformanceDataset,
      experiment_design: ExperimentDesign,
      experiment_start_date: pd.Timestamp,
      experiment_end_date: pd.Timestamp,
      pretest_period_end_date: pd.Timestamp,
  ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Analyzes the experiment with GBR.

    Returns a dataframe with the analysis results. Each row represents each
    metric provided in the experiment data. The columns are the following:

    - metric: The metric name.
    - cell: The cell number.
    - point_estimate: The point estimate of the treatment effect.
    - lower_bound: The lower bound of the confidence interval.
    - upper_bound: The upper bound of the confidence interval.
    - point_estimate_relative: The relative effect size of the treatment.
    - lower_bound_relative: The relative lower bound of the confidence interval.
    - upper_bound_relative: The relative upper bound of the confidence interval.
    - p_value: The p-value of the null hypothesis.

    Args:
      runtime_data: The runtime data for the experiment.
      experiment_design: The design of the experiment being analyzed.
      experiment_start_date: The start date of the experiment.
      experiment_end_date: The end date of the experiment.
      pretest_period_end_date: The end date of the pretest period.

    Returns:
      A dataframe with the analysis results.
    """
    intermediate_data = {}

    pretest_data, experiment_data = (
        self._split_data_into_pretest_and_experiment(
            runtime_data=runtime_data,
            experiment_start_date=experiment_start_date,
            experiment_end_date=experiment_end_date,
            runtime_weeks=experiment_design.runtime_weeks,
            pretest_period_end_date=pretest_period_end_date,
        )
    )

    all_metrics = [
        experiment_design.primary_metric
    ] + experiment_design.secondary_metrics

    # Construct geo level data (one row per geo)
    geo_data, adjusted_linear_model_type = self._construct_geo_level_data(
        all_metrics=all_metrics,
        pretest_data=pretest_data,
        experiment_data=experiment_data,
        all_geos=runtime_data.geos,
        linear_model_type=experiment_design.methodology_parameters[
            "linear_model_type"
        ],
        geo_assignment=experiment_design.geo_assignment,
        geo_id_column=runtime_data.geo_id_column,
    )
    intermediate_data["geo_data"] = geo_data

    results = []
    intermediate_data["params"] = {}
    intermediate_data["covariance"] = {}
    intermediate_data["degrees_of_freedom"] = {}
    for cell in range(1, experiment_design.n_cells):
      # Create the data for the current treatment cell, containing only that
      # cell and the control group.
      cell_data = geo_data.loc[
          geo_data["geo_assignment"].isin([0, cell])
      ].copy()
      cell_data["geo_assignment"] = (
          cell_data["geo_assignment"] == cell
      ).astype(float)
      if cell_data.shape[0] == 0:
        # If the cell is empty, skip
        continue

      for metric in all_metrics:
        params, covariance, degrees_of_freedom = self._fit_linear_model(
            cell_data,
            metric,
            adjusted_linear_model_type,
        )

        key = f"{metric.name}__cell_{cell}"

        intermediate_data["params"][key] = params
        intermediate_data["covariance"][key] = covariance
        intermediate_data["degrees_of_freedom"][key] = degrees_of_freedom

        results_i = self._get_summary_statistics(
            metric=metric,
            n_treatment_geos=cell_data["geo_assignment"].sum(),
            params=params,
            covariance=covariance,
            degrees_of_freedom=degrees_of_freedom,
            alternative_hypothesis=experiment_design.alternative_hypothesis,
            alpha=experiment_design.alpha,
        )

        results_i["cell"] = cell
        results.append(results_i)

    return pd.DataFrame(results), intermediate_data

  def plot_analysis_results(
      self,
      analysis_results: pd.DataFrame,
      intermediate_data: dict[str, Any],
      design: ExperimentDesign,
  ) -> None:
    """Produces custom plots for the analysis results.

    Args:
      analysis_results: The analysis results to plot.
      intermediate_data: The intermediate data from the analysis.
      design: The experiment design.
    """

    del analysis_results  # Unused
    for cell in range(1, design.n_cells):
      for metric in [design.primary_metric] + design.secondary_metrics:
        _plot_gbr_plot(
            intermediate_data=intermediate_data,
            metric=metric,
            treatment_cell=cell,
        )


def _plot_gbr_plot(
    intermediate_data: dict[str, Any],
    metric: Metric,
    treatment_cell: int = 1,
) -> np.ndarray:
  """Plots the GBR plot for a given metric and treatment cell."""
  fig, ax = plt.subplots(ncols=2, figsize=(12, 4), constrained_layout=True)

  if metric.cost_column:
    metric_name = f"{metric.column} (metric = {metric.name})"
  else:
    metric_name = metric.name

  params = intermediate_data["params"][f"{metric.name}__cell_{treatment_cell}"]
  geo_data = intermediate_data["geo_data"].sort_values(
      metric.column + "_pretest"
  )

  is_treated = geo_data["geo_assignment"] == 1

  treated_x = geo_data.loc[is_treated, metric.column + "_pretest"]
  treated_y = geo_data.loc[is_treated, metric.column + "_runtime"]
  control_x = geo_data.loc[~is_treated, metric.column + "_pretest"]
  control_y = geo_data.loc[~is_treated, metric.column + "_runtime"]

  if metric.cost_column:
    treated_cost_diff = geo_data.loc[
        is_treated, metric.cost_column + "_cost_differential"
    ]
  else:
    treated_cost_diff = np.ones_like(treated_x)

  control_y_fit = (
      params["const_demean_corrected"]
      + params[metric.column + "_pretest"] * control_x
  )
  treated_y_fit_baseline = (
      params["const_demean_corrected"]
      + params[metric.column + "_pretest"] * treated_x
  )
  treated_y_fit = (
      treated_y_fit_baseline + params["cost_differential"] * treated_cost_diff
  )

  control_residual = control_y - control_y_fit
  treated_residual = treated_y - treated_y_fit_baseline

  ax[0].plot(
      control_x, control_y, label="Control Geos", marker=".", lw=0, color="C0"
  )
  ax[0].plot(
      treated_x, treated_y, label="Treated Geos", marker=".", lw=0, color="C1"
  )
  ax[0].plot(control_x, control_y_fit, lw=1, color="C0")
  ax[0].plot(treated_x, treated_y_fit, lw=1, color="C1")

  ax[0].set_xlabel("Pretest Value")
  ax[0].set_ylabel("Experiment Value")
  ax[0].set_title("Experiment vs Pre-test")

  ax[0].legend()

  ax[1].plot(
      control_x,
      control_residual,
      label="Control Geos",
      marker=".",
      lw=0,
      color="C0",
  )
  ax[1].plot(
      treated_x,
      treated_residual,
      label="Treated Geos",
      marker=".",
      lw=0,
      color="C1",
  )
  ax[1].axhline(0.0, lw=1, color="C0")
  ax[1].plot(
      treated_x,
      params["cost_differential"] * treated_cost_diff,
      lw=1,
      color="C1",
  )

  ax[1].set_xlabel("Pretest Value")
  ax[1].set_ylabel("Experiment Value - Expected Value")
  ax[1].set_title("Residuals vs Pre-test")
  ax[1].legend()

  fig.suptitle(f"GBR Plot | {metric_name} | Cell {treatment_cell}")

  return ax
