"""The Synthetic Controls methodology for GeoFleX."""

from typing import Any
import geoflex.data
import geoflex.experiment_design
from geoflex.methodology import _base
import geoflex.utils
import pandas as pd
from pysyncon import Dataprep
from pysyncon import Synth


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
    exclude = set(experiment_design.geo_eligibility.exclude or [])
    rng = experiment_design.get_rng()
    n_cells = experiment_design.n_cells

    best_partition = None
    max_min_r2 = -float("inf")

    eligible_geos = list(set(historical_data.geos) - exclude)

    # Check if there are enough geos to pre-assign one to each cell
    if len(eligible_geos) < n_cells:
      raise ValueError(
          f"Not enough eligible geos ({len(eligible_geos)}) to assign "
          f"at least one to each of the {n_cells} cells."
      )

    for _ in range(num_iters):
      # 1. Shuffle the geos to ensure the pre-assignment is random each time.
      rng.shuffle(eligible_geos)

      # 2. Select one "anchor" geo for each cell to pre-assign.
      anchor_geos = eligible_geos[:n_cells]
      pre_assigned = {geo: i for i, geo in enumerate(anchor_geos)}

      # 3. Call the utility function with the pre_assigned parameter.
      # This guarantees that no group will be empty.
      assignments, _ = geoflex.utils.assign_geos_randomly(
          geo_ids=eligible_geos,
          n_groups=n_cells,
          rng=rng,
          pre_assigned_geos=pre_assigned,
      )

      current_partition = {
          "control": assignments[0],
          "treatment": assignments[1:],
      }

      # Step 2b: Evaluate the Partition
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

        time_period = pd.date_range(
            start=experiment_start_date, end=experiment_end_date
        )
        att_results = synth_model.att(time_period=time_period)

        # 1. Get control data (Z0) to calculate the baseline
        exp_data_controls = df_agg[
            (df_agg[runtime_data.date_column].isin(time_period)) &
            (df_agg[runtime_data.geo_id_column].isin(control_geos))
        ]
        z0_exp = exp_data_controls.pivot_table(
            index=runtime_data.date_column,
            columns=runtime_data.geo_id_column,
            values=metric.column
        )

        # 2. Calculate the baseline estimate (the counterfactual)
        baseline_ts = synth_model._synthetic(Z0=z0_exp)
        baseline_estimate = baseline_ts.mean()

        # 3. Estimate baseline_standard_error from pre-treatment residuals
        z_zero_pre, z_one_pre = synth_model.dataprep.make_outcome_mats(
            time_period=synth_model.dataprep.time_optimize_ssr
        )
        pre_treatment_gaps = synth_model._gaps(Z0=z_zero_pre, Z1=z_one_pre)
        baseline_standard_error = pre_treatment_gaps.std()

        # 4. Get summary statistics, now including baseline estimates
        st = geoflex.utils.get_summary_statistics_from_standard_errors(
            impact_estimate=att_results["att"],
            impact_standard_error=att_results["se"],
            baseline_estimate=baseline_estimate,
            baseline_standard_error=baseline_standard_error,
            impact_baseline_corr=0,
            degrees_of_freedom=len(time_period) - 1,
            alternative_hypothesis=experiment_design.alternative_hypothesis,
            alpha=experiment_design.alpha,
            invert_result=metric.cost_per_metric,
        )
        st["metric"] = metric.name
        st["cell"] = cell_index + 1
        results.append(st)

    return pd.DataFrame(results), intermediate_data
