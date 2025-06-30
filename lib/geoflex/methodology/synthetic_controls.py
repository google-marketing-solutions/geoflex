"""The Synthetic Controls methodology for GeoFleX."""

from typing import Any
import geoflex.data
import geoflex.experiment_design
from geoflex.methodology import _base
import numpy as np
import pandas as pd
import pysyncon


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
      "min_treatment_geos": [1],
      "num_iterations": [10],
  }

  def _randomly_assign_geos(
      self,
      experiment_design: ExperimentDesign,
      historical_data: GeoPerformanceDataset,
      rng: np.random.Generator,
  ) -> tuple[list[str], list[str]] | None:
    """This function randomly assigns geos.

    Assign treatment and control groups.

    Args:
      experiment_design: The design parameters for the experiment.
      historical_data: Historical geo performance data.
      rng: Random number generator for reproducible sampling.

    Returns:
      A tuple (treatment_geos, control_geos) or None if constraints unmet.
    """

    # Params
    params = experiment_design.methodology_parameters
    min_treatment_geos = params["min_treatment_geos"]

    treatment_list = experiment_design.geo_eligibility.treatment
    force_test = treatment_list[0] if treatment_list else []
    force_control = experiment_design.geo_eligibility.control or []
    exclude = experiment_design.geo_eligibility.exclude or []
    all_geos = historical_data.geos

    num_forced_control = len(set(force_control) - set(force_test))
    required_controls = max(1, num_forced_control)
    max_treatment_geos = len(all_geos) - required_controls
    if (
        experiment_design.cell_volume_constraint.constraint_type
        == geoflex.CellVolumeConstraintType.MAX_GEOS
    ):
      max_treatment_geos = experiment_design.cell_volume_constraint.values[1]

    # Sample Geos
    return self._sample_geos(
        all_geos,
        force_test,
        force_control,
        exclude,
        min_treatment_geos,
        max_treatment_geos,
        rng
    )

  def _aggregate_and_fit(
      self,
      experiment_design: ExperimentDesign,
      historical_data: GeoPerformanceDataset,
      sample: tuple[list[str], list[str]]
  ) -> tuple[GeoAssignment, dict[str, Any]] | None:
    """Aggregates treatment geos and fits a synthetic control model.

    This method takes a sample of treatment and control geos, aggregates the
    treatment group into a single unit, and then fits a synthetic control
    model to the historical data.

    Args:
      experiment_design: The design parameters for the experiment.
      historical_data: Historical geo performance data.
      sample: A tuple containing two lists: the treatment geos and the
        control geos for this iteration.

    Returns:
      A tuple containing the GeoAssignment object for the sample and a
      dictionary of model results, or None if the process fails.
    """

    test_geos, control_geos = sample
    exclude = experiment_design.geo_eligibility.exclude or []

    # Get Data
    df = historical_data.parsed_data
    dep = experiment_design.primary_metric.column

    # Aggregate
    df_agg = self.aggregate_treatment(
        df,
        test_geos,
        control_geos,
        historical_data.geo_id_column,
        historical_data.date_column,
        dep
        )
    if df_agg.empty:
      return None

    # Fit Model
    mr = self._fit_model(
        df_agg,
        historical_data.geo_id_column,
        historical_data.date_column,
        dep,
        control_geos=control_geos,
        test_geo="Aggregated_Treatment",
        treatment_geos=test_geos,
        train_start_date=df[historical_data.date_column].min(),
        train_end_date=None,
        predictor_start_date=df[historical_data.date_column].min(),
        predictor_end_date=df[historical_data.date_column].max(),
        )

    # Build the GeoAssignment
    treat = [set(test_geos)]
    ctrl = set(control_geos)
    excl = set(exclude)

    geo_assignment = GeoAssignment(
        treatment=treat,
        control=ctrl,
        exclude=excl
        )

    return geo_assignment, mr

  def aggregate_treatment(
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

  def _sample_geos(
      self,
      all_geos: list[str],
      force_test: list[str],
      force_control: list[str],
      exclude: list[str],
      min_treatment_geos: int,
      max_treatment_geos: int,
      rng: np.random.Generator,
  ) -> tuple[list[str], list[str]] | None:
    """Randomly sample treatment and control geos given constraints.

    Args:
      all_geos: List of all candidate geo identifiers.
      force_test: Geos that must be included in treatment.
      force_control: Geos that must be included in control.
      exclude: Geos to exclude entirely.
      min_treatment_geos: Minimum number of treatment geos.
      max_treatment_geos: Maximum number of treatment geos.
      rng: Random number generator for reproducible sampling.

    Returns:
      A tuple (treatment_geos, control_geos) or None if constraints unmet.
    """

    pool = set(all_geos) - set(exclude)
    f_test = set(force_test)
    f_control = set(force_control) - f_test

    available = list(pool - f_test - f_control)
    need_min = max(0, min_treatment_geos - len(f_test))
    max_extra = max(0, max_treatment_geos - len(f_test))

    if len(available) < need_min:
      return None

    extra = int(rng.integers(need_min, min(max_extra, len(available)) + 1))
    new_test = (
        list(rng.choice(available, size=extra, replace=False))
        if extra else []
    )

    test_geos = list(f_test | set(new_test))
    control_geos = list(f_control | (set(available) - set(new_test)))

    if not test_geos or not control_geos:
      return None

    return test_geos, control_geos

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
      train_fraction: float = 0.75
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
    train_df = df_agg[df_agg[time_var].isin(train_dates)]

    # Build & fit synthetic control
    dataprep = pysyncon.Dataprep(
        foo=train_df,
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
    synth = pysyncon.Synth()
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
      synth_model: pysyncon.Synth,
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
    exclude = experiment_design.geo_eligibility.exclude or []

    # Run all iterations
    best_iter = None
    best_validation = float("-inf")
    rng = experiment_design.get_rng()

    for _ in range(num_iters):
      sample = self._randomly_assign_geos(
          experiment_design,
          historical_data,
          rng
      )
      if not sample:
        continue

      iter_output = self._aggregate_and_fit(
          experiment_design,
          historical_data,
          sample
      )
      if not iter_output:
        continue

      _, result = iter_output
      # Pull out the validation R²; skip if missing
      val_r2 = result.get("validation_r2")
      if val_r2 is None:
        continue

      # Keep the iteration with highest validation_r2
      if val_r2 > best_validation:
        best_validation = val_r2
        best_iter = result
    # Build the GeoAssignment from the best iteration
    treat = [set(best_iter["treatment_geos"])]
    ctrl = set(best_iter["control_geos"])
    excl = set(exclude)

    return GeoAssignment(
        treatment=treat,
        control=ctrl,
        exclude=excl,
    ), {}

  def analyze_experiment(
      self,
      runtime_data: GeoPerformanceDataset,
      experiment_design: ExperimentDesign,
      experiment_start_date: str,
  ) -> pd.DataFrame:
    """Analyzes a Synthetic Control experiment.

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

    Returns:
      A dataframe with the analysis results.
    """
    pass

  def _methodology_analyze_experiment(self):
    pass
