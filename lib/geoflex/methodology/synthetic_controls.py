"""The Synthetic Controls methodology for GeoFleX."""

from typing import Any
import geoflex.data
import geoflex.experiment_design
from geoflex.methodology import _base
import numpy as np
import pandas as pd


ExperimentDesign = geoflex.experiment_design.ExperimentDesign
ExperimentDesignSpec = geoflex.experiment_design.ExperimentDesignSpec
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
ExperimentDesignEvaluation = (
    geoflex.experiment_design.ExperimentDesignEvaluation
)
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
    ) -> tuple[GeoAssignment, dict[str, Any]] | None:
    """This function randomly assigns geos.

    Assign treatment and control groups, fits a synthetic control model
    using historical data, and calculates the out-of-sample R² as a measure
    of the model's predictive power.

    Args:
      experiment_design: The design parameters for the experiment.
      historical_data: Historical geo performance data.
      rng: Random number generator for reproducible sampling.

    Returns:
      A tuple with the geo assignment and a dictionary of results for this
      iteration, or None if invalid.
    """

    # Get Data
    df = historical_data.parsed_data
    dep = experiment_design.primary_metric.column

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
    sample = self._sample_geos(
        all_geos,
        force_test,
        force_control,
        exclude,
        min_treatment_geos,
        max_treatment_geos,
        rng
    )
    if sample is None:
      return None
    test_geos, control_geos = sample

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
        predictor_start_date=df[historical_data.date_column].min(),
        predictor_end_date=df[historical_data.date_column].max(),
        )

    # build the GeoAssignment
    treat = set(test_geos)
    ctrl = set(control_geos)
    excl = set(exclude)

    geo_assignment = GeoAssignment(
        treatment=treat,
        control=ctrl,
        exclude=excl,
        all_geos=treat | ctrl | excl
    )

    return geo_assignment, mr

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

  # pylint: disable=unused-argument
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
      predictor_start_date: str,
      predictor_end_date: str,
  ) -> dict[str, Any]:
    """Fit the synthetic control model on aggregated data.

    Args:
      df_agg: Aggregated treatment/control data.
      geo_var: Column name for geo identifiers.
      time_var: Column name for time identifiers.
      dependent: Column name for the outcome variable.
      control_geos: List of control geos.
      test_geo: Identifier for the synthetic treatment geo.
      treatment_geos: List of treatment geos.
      train_start_date: Starting date for training the model.
      predictor_start_date: Start date for predictors.
      predictor_end_date: End date for predictors.

    Returns:
      A dictionary of model fit results.
    """
    pass

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

    # run all iterations
    best_iter = None
    best_validation = float("-inf")
    rng = experiment_design.get_rng()

    for _ in range(num_iters):
      iter_output = self._randomly_assign_geos(
          experiment_design,
          historical_data,
          rng
      )
      if not iter_output:
        continue

      _, result = iter_output

      # pull out the validation R²; skip if missing
      val_r2 = result.get("validation_r2")
      if val_r2 is None:
        continue

      # keep the iteration with highest validation_r2
      if val_r2 > best_validation:
        best_validation = val_r2
        best_iter = result

    # build the GeoAssignment from the best iteration
    treat = set(best_iter["treatment_geos"])
    ctrl = set(best_iter["control_geos"])
    excl = set(exclude)

    return GeoAssignment(
        treatment=treat,
        control=ctrl,
        exclude=excl,
        all_geos=treat | ctrl | excl
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
