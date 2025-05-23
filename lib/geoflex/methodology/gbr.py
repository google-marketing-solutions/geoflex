"""Our implementation of the GBR methodology."""

import datetime as dt
import logging
import geoflex.data
import geoflex.experiment_design
from geoflex.methodology import _base
import geoflex.utils
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


@register_methodology
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
      self, design: ExperimentDesign, historical_data: GeoPerformanceDataset
  ) -> bool:
    """Checks if the methodology is eligible for the given design.

    Not eligible if geos are forced into a control
    or treatment group, because this methodology will randomly assign the
    geos.

    Not eligible if the metric values are zero or negative.

    Args:
      design: The design to check against.
      historical_data: The dataset to check against.

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
            (historical_data.parsed_data[cost_columns] == 0).all().all()
        )
        costs_all_positive = (
            (historical_data.parsed_data[cost_columns] > 0).all().all()
        )
        if not (costs_all_positive or costs_all_zero):
          logger.info(
              "GBR ineligible for design %s: Cost columns contain a mix of some"
              " positive and some zero or negative values. Costs must be all"
              " positive or all zero, but not a mix.",
              design.design_id,
          )
          return False

      # If the metrics are ever zero or negative it also fails
      if not (historical_data.parsed_data[metric_columns] > 0).all().all():
        logger.info(
            "GBR ineligible for design %s: Metric columns contain some zero or"
            " negative values. Metrics must be all positive.",
            design.design_id,
        )
        return False

    # If there are too few geos it won't work
    geos_per_cell = len(historical_data.geos) / design.n_cells
    if geos_per_cell < 4:
      logger.info(
          "GBR is ineligible for design %s: GBR requires at least 4 geos per"
          " cell on average, but got %s geos and %s cells",
          design.design_id,
          len(historical_data.geos),
          design.n_cells,
      )
      return False

    return True

  def _methodology_assign_geos(
      self,
      experiment_design: ExperimentDesign,
      historical_data: GeoPerformanceDataset,
  ) -> GeoAssignment:
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
        == geoflex.CellVolumeConstraintType.MAX_PERCENTAGE_OF_TOTAL_RESPONSE
    ):
      metric_values = historical_data.parsed_data.groupby("geo_id")[
          experiment_design.primary_metric.column
      ].sum()
      metric_values /= metric_values.sum()
    elif (
        experiment_design.cell_volume_constraint.constraint_type
        == geoflex.CellVolumeConstraintType.MAX_PERCENTAGE_OF_TOTAL_COST
    ):

      if experiment_design.main_cost_column is None:
        error_message = (
            "Trying to use max_percentage_of_total_cost constraint without a"
            " cost metric in the design."
        )
        logger.error(error_message)
        raise RuntimeError(error_message)

      metric_values = historical_data.parsed_data.groupby("geo_id")[
          experiment_design.main_cost_column
      ].sum()
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
    else:
      max_metric_per_group = None

    raw_assignments, _ = geoflex.utils.assign_geos_randomly(
        geo_ids=eligible_geos,
        metric_values=metric_values,
        n_groups=experiment_design.n_cells,
        max_metric_per_group=max_metric_per_group,
        rng=experiment_design.get_rng(),
    )

    return GeoAssignment(
        control=raw_assignments[0], treatment=raw_assignments[1:]
    )

  def _split_data_into_pretest_and_experiment(
      self,
      runtime_data: GeoPerformanceDataset,
      experiment_start_date: pd.Timestamp,
      experiment_end_date: pd.Timestamp,
      runtime_weeks: int,
  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    experiment_start_date = pd.to_datetime(experiment_start_date)
    pretest_start_date = experiment_start_date - dt.timedelta(
        weeks=runtime_weeks
    )

    is_pretest = (
        runtime_data.parsed_data[runtime_data.date_column] >= pretest_start_date
    ) & (
        runtime_data.parsed_data[runtime_data.date_column]
        < experiment_start_date
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

    n_pretest_dates = pretest_data["date"].nunique()
    n_experiment_dates = experiment_data["date"].nunique()
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
        "geo_id": all_geos,
        "geo_assignment": geo_assignment.make_geo_assignment_array(all_geos),
    }).set_index("geo_id")

    geo_data = geo_data.loc[
        geo_data["geo_assignment"] >= 0.0
    ]  # Exclude excluded geos

    for column in all_metric_columns:
      geo_data[f"{column}_pretest"] = pretest_data.groupby("geo_id")[
          column
      ].sum()
      geo_data[f"{column}_runtime"] = experiment_data.groupby("geo_id")[
          column
      ].sum()

    for cost_column in all_cost_columns:
      geo_data[f"{cost_column}_pretest"] = pretest_data.groupby("geo_id")[
          cost_column
      ].sum()
      geo_data[f"{cost_column}_runtime"] = experiment_data.groupby("geo_id")[
          cost_column
      ].sum()

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

    return geo_data

  def _add_cost_differential(
      self, geo_data: pd.DataFrame, cost_column: str, model_type: str
  ) -> pd.DataFrame:
    """Get the cost differential."""
    pre_period_cost_always_0 = (geo_data[f"{cost_column}_pretest"] == 0).all()

    if pre_period_cost_always_0:
      geo_data[f"{cost_column}_cost_differential"] = geo_data[
          f"{cost_column}_runtime"
      ].copy()

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
      X[f"{metric.column}_pretest"] -= (
          1.0 / (1.0 / X[f"{metric.column}_pretest"]).mean()
      )

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
      X[f"{metric.column}_pretest"] -= X[f"{metric.column}_pretest"].mean()

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

    return params, covariance, degrees_of_freedom

  def _get_summary_statistics(
      self,
      metric: Metric,
      params: pd.Series,
      covariance: pd.DataFrame,
      degrees_of_freedom: int,
      alternative_hypothesis: str,
      alpha: float,
  ) -> dict[str, float]:
    """Calculates the summary statistics from the result of the linear model."""

    if metric.cost_per_metric or metric.metric_per_cost:
      # Don't calculate the relative effect for cost per metric or metric per
      # cost. The relative effect is not meaningful for these metrics.
      baseline_estimate = None
      baseline_standard_error = None
      impact_baseline_corr = None
    else:
      baseline_estimate = params["const"]
      baseline_standard_error = np.sqrt(covariance.loc["const", "const"])
      impact_baseline_corr = covariance.loc[
          "cost_differential", "const"
      ] / np.sqrt(
          covariance.loc["const", "const"]
          * covariance.loc["cost_differential", "cost_differential"]
      )

    summary_statistics = (
        geoflex.utils.get_summary_statistics_from_standard_errors(
            impact_estimate=params["cost_differential"],
            impact_standard_error=np.sqrt(
                covariance.loc["cost_differential", "cost_differential"]
            ),
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
  ) -> pd.DataFrame:
    """Analyzes the experiment with GBR.

    Returns a dataframe with the analysis results. Each row represents each
    metric provided in the experiment data. The columns are the following:

    - metric: The metric name.
    - is_primary_metric: Whether the metric is a primary metric.
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

    Returns:
      A dataframe with the analysis results.
    """
    pretest_data, experiment_data = (
        self._split_data_into_pretest_and_experiment(
            runtime_data=runtime_data,
            experiment_start_date=experiment_start_date,
            experiment_end_date=experiment_end_date,
            runtime_weeks=experiment_design.runtime_weeks,
        )
    )

    all_metrics = [
        experiment_design.primary_metric
    ] + experiment_design.secondary_metrics

    # Construct geo level data (one row per geo)
    geo_data = self._construct_geo_level_data(
        all_metrics,
        pretest_data,
        experiment_data,
        runtime_data.geos,
        experiment_design.methodology_parameters["linear_model_type"],
        experiment_design.geo_assignment,
    )

    results = []
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
            experiment_design.methodology_parameters["linear_model_type"],
        )
        results_i = self._get_summary_statistics(
            metric,
            params,
            covariance,
            degrees_of_freedom,
            experiment_design.alternative_hypothesis,
            experiment_design.alpha,
        )

        results_i["cell"] = cell
        results_i["is_primary_metric"] = (
            metric == experiment_design.primary_metric
        )
        results.append(results_i)

    return pd.DataFrame(results)
