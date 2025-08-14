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

"""The Synthetic Controls methodology for GeoFleX."""

import logging
from typing import Any
import geoflex.data
import geoflex.experiment_design
from geoflex.methodology import _base
import geoflex.utils
import numpy as np
import pandas as pd
from pysyncon import Dataprep
from pysyncon import Synth


logger = logging.getLogger(__name__)


# pylint:disable=protected-access

ExperimentDesign = geoflex.experiment_design.ExperimentDesign
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
GeoAssignment = geoflex.experiment_design.GeoAssignment
CellVolumeConstraintType = geoflex.experiment_design.CellVolumeConstraintType

register_methodology = _base.register_methodology


@register_methodology
class SyntheticControls(_base.Methodology):
  """The Synthetic Control methodology for GeoFleX.

  The methodology uses a synthetic control linear model to predict the
  counterfactual of the test geos based on the control geos.

  Design:
    Geos are split into treatment and control groups based on the
    maximization of predictive power of the test by the control.

  Evaluation:
    The evaluation is done with a t-test based on the difference between
    the prediction and actual values.
  """

  is_pseudo_experiment = True

  default_methodology_parameter_candidates = {
      "num_iterations": [10],
  }

  def _aggregate_treatment(
      self,
      df: pd.DataFrame,
      treatment_geos: list[str],
      control_geos: list[str],
      geo_var: str,
      time_var: str,
      dependent: str
  ) -> pd.DataFrame:
    """Aggregates the treatment geos into a single time series.

    This function calculates the mean of the dependent variable across all
    treatment geos for each time period, creating a new entity named
    'Aggregated_Treatment'.

    Args:
      df: The input DataFrame containing historical performance data.
      treatment_geos: A list of geo identifiers for the treatment group.
      control_geos: A list of geo identifiers for the control group.
      geo_var: The name of the column identifying the geos.
      time_var: The name of the column identifying the time periods.
      dependent: The name of the column for the performance metric.

    Returns:
      A new DataFrame with the treatment geos replaced by a single
      'Aggregated_Treatment' entity.
    """

    test_geos_df = df[df[geo_var].isin(treatment_geos)]
    control_geos_df = df[df[geo_var].isin(control_geos)]
    test_geos_df = test_geos_df.groupby(
        time_var)[dependent].mean().reset_index()
    test_geos_df[geo_var] = ["Aggregated_Treatment"]*len(test_geos_df)

    return pd.concat([test_geos_df, control_geos_df])

  def _fit_model(
      self,
      df_agg: pd.DataFrame,
      geo_var: str,
      time_var: str,
      dependent: str,
      control_geos: list[str],
      test_geo: str,
      treatment_geos: list[str],
      train_start_date: str,
      train_end_date: str,
      predictor_start_date: str,
      predictor_end_date: str,
      train_fraction: float = 0.75,
      is_analysis_phase: bool = False
  ) -> dict[str, Any]:
    """Fits a synthetic control model and evaluates its predictive power.

    This function takes aggregated historical data, splits it into training
    and validation periods, and fits a PySyncon model. It then calculates
    the out-of-sample R-squared (R²) on the validation period to measure
    how well the control geos predict the treatment geo's counterfactual.

    Args:
      df_agg: The aggregated DataFrame with a single treatment unit.
      geo_var: The name of the column identifying the geos.
      time_var: The name of the column identifying the time periods.
      dependent: The name of the column for the performance metric.
      control_geos: A list of geo identifiers for the control pool.
      test_geo: The identifier for the single aggregated treatment unit.
      treatment_geos: The original list of treatment geos before aggregation.
      train_start_date: The start date for the training period.
      train_end_date: The end date for the training period. If None, it's
        inferred from the `train_fraction`.
      predictor_start_date: The start date for the predictors to use.
      predictor_end_date: The end date for the predictors to use.
      train_fraction: The proportion of data to use for training if
        `train_end_date` is not specified.
      is_analysis_phase: Is analysis or assignment phase.

    Returns:
      A dictionary containing the results of the model fit, including:
      - 'validation_r2': The out-of-sample R-squared score.
      - 'synth_model': The fitted PySynCon Synth object.
      - 'weights': The weights assigned to each control geo.
      - 'ssr': The Sum of Squared Residuals (in-sample error).
      - and other metadata about the model run.
    """
    # Sort & pick absolute defaults
    all_dates = sorted(df_agg[time_var].unique())
    first, last = all_dates[0], all_dates[-1]

    # Normalize & default
    train_start = pd.to_datetime(train_start_date or first)

    # Guardrail: Check if there's enough data for a meaningful split.
    # We need at least 3 data points to have a training and a
    # validation set of >1 point.
    if len(all_dates) < 3:
      return {
          "treatment_geos": treatment_geos,
          "control_geos": control_geos,
          "ssr": None,
          "weights": None,
          "synth_model": None,
          "dataprep": None,
          "validation_r2": None,
          "train_start_date": None,
          "train_end_date": None,
          "train_fraction": train_fraction
      }

    # To guarantee a validation set of at least 2 points, the training
    # data must end at or before the date at index `len(all_dates) - 3`.
    max_train_idx = len(all_dates) - 3
    train_end_idx = int(len(all_dates) * train_fraction)

    # Clamp the index to ensure the validation set is large enough.
    train_end_idx = min(train_end_idx, max_train_idx)

    train_end = pd.to_datetime(train_end_date or all_dates[train_end_idx])

    predictor_start = pd.to_datetime(predictor_start_date or first)
    predictor_end = pd.to_datetime(predictor_end_date or last)

    # String versions
    train_start_str = train_start.strftime("%Y-%m-%d")
    train_end_str = train_end.strftime("%Y-%m-%d")

    # Slice data
    train_dates = pd.date_range(start=train_start, end=train_end)
    predictor_dates = pd.date_range(start=predictor_start, end=predictor_end)

    foo_data = df_agg if is_analysis_phase else df_agg[
        df_agg[time_var].isin(train_dates)
    ]

    # Build & fit synthetic control
    dataprep = Dataprep(
        foo=foo_data,
        predictors=[],
        dependent=dependent,
        unit_variable=geo_var,
        time_variable=time_var,
        treatment_identifier=test_geo,
        controls_identifier=control_geos,
        time_predictors_prior=list(predictor_dates),
        special_predictors=[(dependent, list(train_dates), "mean")],
        time_optimize_ssr=list(train_dates),
        predictors_op="mean",
    )
    synth = Synth()
    synth.fit(dataprep)
    ssr = synth.loss_V

    # Calculate validation R²
    val_df = df_agg[
        (df_agg[time_var] > train_end) & (df_agg[time_var] <= predictor_end)
    ]

    if val_df.empty:
      val_r2 = None
    else:
      val_r2 = self._calculate_r2(
          synth_model=synth,
          df=val_df,
          unit_var=geo_var,
          time_var=time_var,
          dependent=dependent,
          control_geos=control_geos,
          test_geo=test_geo
      )

    # Return results
    return {
        "treatment_geos": treatment_geos,
        "control_geos": control_geos,
        "ssr": ssr,
        "weights": synth.weights(),
        "synth_model": synth,
        "dataprep": dataprep,
        "validation_r2": val_r2,
        "train_start_date": train_start_str,
        "train_end_date": train_end_str,
        "train_fraction": train_fraction,
    }

  @staticmethod
  def _calculate_r2(
      synth_model: Synth,
      df: pd.DataFrame,
      unit_var: str,
      time_var: str,
      dependent: str,
      control_geos: list[str],
      test_geo: str
    ) -> float | None:
    """Calculates the out-of-sample R-squared (R²) for the synthetic control.

    This function measures the predictive accuracy of the fitted model on a
    holdout (validation) dataset. R² is calculated as 1 - (MSPE / Variance),
    where MSPE is the Mean Squared Prediction Error.

    Args:
      synth_model: The fitted PySynCon Synth object.
      df: The validation DataFrame.
      unit_var: The name of the column identifying the geos.
      time_var: The name of the column identifying the time periods.
      dependent: The name of the column for the performance metric.
      control_geos: A list of geo identifiers for the control group.
      test_geo: The identifier for the treatment unit (e.g.,
        'Aggregated_Treatment').

    Returns:
      The calculated R-squared value, or None if the variance of the
      actuals is zero.
    """
    val_data = df[df[unit_var].isin(control_geos + [test_geo])]
    z_val = val_data[val_data[unit_var].isin(control_geos)].pivot_table(
        index=time_var, columns=unit_var, values=dependent)
    z_one_val = val_data[
        val_data[unit_var] == test_geo].set_index(time_var)[dependent]

    mspe = synth_model.mspe(Z0=z_val, Z1=z_one_val)

    variance_actuals = z_one_val.var()
    if variance_actuals == 0:
      return None

    r2 = 1 - (mspe / variance_actuals)
    return r2

  def _methodology_assign_geos(
      self,
      experiment_design: ExperimentDesign,
      historical_data: GeoPerformanceDataset,
  ) -> tuple[GeoAssignment, dict[str, Any]]:
    """Assigns geos to control and test based on the synthetic controls method.

    Args:
      experiment_design: The experiment design to assign geos for.
      historical_data: The historical data for the experiment. Can be used to
        choose geos that are similar to geos that have been used in the past.

    Returns:
      A GeoAssignment object containing the lists of geos for the control and
      treatment groups, and optionally a list of geos that should be ignored.
    """
    num_iters = experiment_design.methodology_parameters["num_iterations"]
    geo_eligibility = experiment_design.geo_eligibility
    exclude = set(geo_eligibility.exclude or [])
    rng = experiment_design.get_rng()
    n_cells = experiment_design.n_cells

    best_partition = None
    max_min_r2 = -float("inf")

    # Handle pre-assigned geos from GeoEligibility
    pre_assigned = {geo: 0 for geo in (geo_eligibility.control or [])}
    for i, treatment_geos in enumerate(geo_eligibility.treatment or []):
      for geo in treatment_geos:
        pre_assigned[geo] = i + 1

    all_geos = set(historical_data.geos)
    assigned_geos = set(pre_assigned.keys())
    eligible_geos_for_random = list(all_geos - exclude - assigned_geos)

    # Check if there are enough geos to assign at least one to each cell
    if len(all_geos - exclude) < n_cells:
      raise ValueError(
          f"Not enough eligible geos ({len(all_geos - exclude)}) to assign "
          f"at least one to each of the {n_cells} cells."
      )

    for _ in range(num_iters):
      # 1. Create a temporary pre-assignment map for this iteration to ensure
      #    every cell gets at least one geo.
      temp_pre_assigned = pre_assigned.copy()

      # 2. Shuffle the available geos for random seeding.
      temp_eligible_geos = eligible_geos_for_random.copy()
      rng.shuffle(temp_eligible_geos)

      # 3. Find which cells are currently empty.
      assigned_cells = set(temp_pre_assigned.values())
      empty_cells = set(range(n_cells)) - assigned_cells

      # 4. "Seed" each empty cell with one geo from the eligible pool.
      #    This guarantees that the assign_geos_randomly function won't fail.
      for cell_idx in empty_cells:
        if not temp_eligible_geos:
          raise ValueError(
              f"Cannot find an eligible geo to assign to empty cell {cell_idx}"
          )
        geo_to_seed = temp_eligible_geos.pop()
        temp_pre_assigned[geo_to_seed] = cell_idx

      assignments, _ = geoflex.utils.assign_geos_randomly(
          geo_ids=list(all_geos - exclude),
          n_groups=n_cells,
          rng=rng,
          pre_assigned_geos=temp_pre_assigned,
      )

      current_partition = {
          "control": assignments[0],
          "treatment": assignments[1:],
      }

      r2_scores_for_this_partition = []
      for treatment_cell in current_partition["treatment"]:
        if not treatment_cell:
          continue

        df_agg = self._aggregate_treatment(
            historical_data.parsed_data,
            treatment_geos=treatment_cell,
            control_geos=current_partition["control"],
            geo_var=historical_data.geo_id_column,
            time_var=historical_data.date_column,
            dependent=experiment_design.primary_metric.column,
        )
        if df_agg.empty:
          continue

        model_result = self._fit_model(
            df_agg,
            historical_data.geo_id_column,
            historical_data.date_column,
            experiment_design.primary_metric.column,
            control_geos=current_partition["control"],
            test_geo="Aggregated_Treatment",
            treatment_geos=treatment_cell,
            train_start_date=historical_data.parsed_data[
                historical_data.date_column
            ].min(),
            train_end_date=None,
            predictor_start_date=historical_data.parsed_data[
                historical_data.date_column
            ].min(),
            predictor_end_date=historical_data.parsed_data[
                historical_data.date_column
            ].max(),
        )

        if model_result and model_result.get("validation_r2") is not None:
          r2_scores_for_this_partition.append(model_result["validation_r2"])

      if not r2_scores_for_this_partition:
        continue

      current_min_r2 = min(r2_scores_for_this_partition)

      # Step 2c: Update the Best Result
      if current_min_r2 > max_min_r2:
        max_min_r2 = current_min_r2
        best_partition = current_partition

    # Finalization
    return (
        GeoAssignment(
            control=set(best_partition["control"]),
            treatment=[set(cell) for cell in best_partition["treatment"]],
            exclude=exclude,
        ),
        {},
    )

  def _validate_data_periods(
      self,
      runtime_data: GeoPerformanceDataset,
      experiment_start_date: pd.Timestamp,
      experiment_end_date: pd.Timestamp,
      pretest_period_end_date: pd.Timestamp,
  ) -> None:
    """Checks for data presence in pretest and runtime periods."""
    is_pretest = (
        runtime_data.parsed_data[runtime_data.date_column]
        < pretest_period_end_date
    )
    is_runtime = (
        runtime_data.parsed_data[runtime_data.date_column]
        >= experiment_start_date
    ) & (
        runtime_data.parsed_data[runtime_data.date_column]
        < experiment_end_date
    )

    if not is_pretest.any() or not is_runtime.any():
      error_message = (
          "No data in the pretest or runtime period for SyntheticControls. "
          "This is likely because the experiment start date is too close to "
          "the start or end of the provided data."
      )
      logger.error(error_message)
      raise RuntimeError(error_message)

  def _methodology_analyze_experiment(
      self,
      runtime_data: GeoPerformanceDataset,
      experiment_design: ExperimentDesign,
      experiment_start_date: pd.Timestamp,
      experiment_end_date: pd.Timestamp,
      pretest_period_end_date: pd.Timestamp,
  ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Analyzes the experiment with SyntheticControls.

    Returns a dataframe with the analysis results. Each row represents each
    metric provided in the experiment data. The columns are the following:

    - metric: The metric name.
    - point_estimate: The point estimate of the treatment effect.
    - lower_bound: The lower bound of the confidence interval.
    - upper_bound: The upper bound of the confidence interval.
    - point_estimate_relative: The relative effect size of the treatment.
    - lower_bound_relative: The relative lower bound of the confidence
      interval.
    - upper_bound_relative: The relative upper bound of the confidence
      interval.
    - p_value: The p-value of the null hypothesis.
    - is_significant: Whether the null hypothesis is rejected.

    Args:
      runtime_data: The runtime data for the experiment.
      experiment_design: The design of the experiment being analyzed.
      experiment_start_date: The start date of the experiment.
      experiment_end_date: The end date of the experiment.
      pretest_period_end_date: The end date of the pretest period.

    Returns:
      A dataframe with the analysis results.
    """
    last_available_date = runtime_data.parsed_data[
        runtime_data.date_column
    ].max()
    if experiment_end_date > last_available_date:
      logger.warning(
          "The provided experiment_end_date (%s) is after the last"
          " available data point (%s). Adjusting experiment_end_date to"
          " match the last available date.",
          experiment_end_date.strftime("%Y-%m-%d"),
          last_available_date.strftime("%Y-%m-%d"),
      )
      experiment_end_date = last_available_date
      self._validate_data_periods(
          runtime_data,
          experiment_start_date,
          experiment_end_date,
          pretest_period_end_date,
      )

    intermediate_data = {}
    results = []
    control_geos = list(experiment_design.geo_assignment.control)

    for cell_index, treatment_geos_set in enumerate(
        experiment_design.geo_assignment.treatment
    ):
      treatment_geos = list(treatment_geos_set)
      if not treatment_geos:
        continue

      for metric in (
          [experiment_design.primary_metric]
          + experiment_design.secondary_metrics
      ):
        df_agg = self._aggregate_treatment(
            runtime_data.parsed_data,
            treatment_geos,
            control_geos,
            runtime_data.geo_id_column,
            runtime_data.date_column,
            metric.column,
        )
        if df_agg.empty:
          continue

        model_results = self._fit_model(
            df_agg,
            runtime_data.geo_id_column,
            runtime_data.date_column,
            metric.column,
            control_geos=control_geos,
            test_geo="Aggregated_Treatment",
            treatment_geos=treatment_geos,
            train_start_date=runtime_data.parsed_data[
                runtime_data.date_column
            ].min(),
            train_end_date=pretest_period_end_date,
            predictor_start_date=runtime_data.parsed_data[
                runtime_data.date_column
            ].min(),
            predictor_end_date=experiment_end_date,
            is_analysis_phase=True,
        )

        synth_model = model_results.get("synth_model")
        if not synth_model:
          continue

        # Create the ideal time period for the analysis.
        ideal_time_period = pd.date_range(
            start=experiment_start_date,
            end=experiment_end_date,
            inclusive="left"
        )
        # Find the dates that are actually available in the runtime data.
        available_runtime_dates = runtime_data.parsed_data[
            runtime_data.parsed_data[runtime_data.date_column].isin(
                ideal_time_period
            )
        ][runtime_data.date_column].unique()
        # The final time period is the intersection of the ideal and available.
        time_period = ideal_time_period.intersection(available_runtime_dates)

        att_results = synth_model.att(time_period=time_period)

        # 1. Get the essential time series: actuals and predictions
        z0_exp, z1_exp = synth_model.dataprep.make_outcome_mats(
            time_period=time_period
        )
        treatment_ts = z1_exp.squeeze()
        baseline_ts = synth_model._synthetic(Z0=z0_exp)

        # Call the utils function to get the parts it does correctly:
        # p-value and the absolute confidence interval.
        # The relative CI from this call will be WRONG.
        st = geoflex.utils.get_summary_statistics_from_standard_errors(
            impact_estimate=att_results["att"],
            impact_standard_error=att_results["se"],
            degrees_of_freedom=len(time_period) - 1,
            alternative_hypothesis=experiment_design.alternative_hypothesis,
            alpha=experiment_design.alpha,
            invert_result=metric.cost_per_metric,
            baseline_estimate=None,
        )
        # 2. Manually calculate the relative CI.
        # Check if we can calculate a relative effect
        if (
            metric.cost_per_metric
            or (baseline_ts.mean() <= 0.0)
            or (treatment_ts.mean() <= 0.0)
        ):
          # If not, set relative metrics to NA
          st["point_estimate_relative"] = pd.NA
          st["lower_bound_relative"] = pd.NA
          st["upper_bound_relative"] = pd.NA
        else:
          # 3. Calculate the correct inputs for the relative difference
          n = len(time_period)
          se_treatment_mean = treatment_ts.std() / np.sqrt(n)
          se_baseline_mean = baseline_ts.std() / np.sqrt(n)
          correlation = treatment_ts.corr(baseline_ts)
          if pd.isna(correlation):
            correlation = 0.0

          # 4. Call relative_difference_confidence_interval
          lower_bound_relative, upper_bound_relative = (
              geoflex.utils.relative_difference_confidence_interval(
                  mean_1=treatment_ts.mean(),
                  mean_2=baseline_ts.mean(),
                  standard_error_1=se_treatment_mean,
                  standard_error_2=se_baseline_mean,
                  corr=correlation,
                  degrees_of_freedom=len(time_period) - 1,
                  alternative=experiment_design.alternative_hypothesis,
                  alpha=experiment_design.alpha,
              )
          )

          # 5. Add the correct relative values to our results dictionary
          point_est_rel = (treatment_ts.mean() / baseline_ts.mean()) - 1
          st["point_estimate_relative"] = point_est_rel
          st["lower_bound_relative"] = lower_bound_relative
          st["upper_bound_relative"] = upper_bound_relative

        st["metric"] = metric.name
        st["cell"] = cell_index + 1
        results.append(st)

    return pd.DataFrame(results), intermediate_data
