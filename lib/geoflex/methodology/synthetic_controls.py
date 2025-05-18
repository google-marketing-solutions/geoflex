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

  def assign_geos(
      self,
      experiment_design: ExperimentDesign,
      historical_data: GeoPerformanceDataset,
      rng: np.random.Generator,
  ) -> GeoAssignment:
    """Assigns geos to control and test based on the synthetic controls method.

    Args:
      experiment_design: The experiment design to assign geos for.
      historical_data: The historical data for the experiment. Can be used to
        choose geos that are similar to geos that have been used in the past.
      rng: The random number generator to use for randomization, if needed.

    Returns:
      A GeoAssignment object containing the lists of geos for the control and
      treatment groups, and optionally a list of geos that should be ignored.
    """

    params = {}
    if experiment_design.methodology_parameters:
      params = experiment_design.methodology_parameters

    exclude_geos_input = list(self._get_param(params, "exclude_geos"))

    results_df, _ = run_experiment_simulations(
        df=historical_data.parsed_data,
        geo_var=historical_data.geo_id_column,
        time_var=historical_data.date_column,
        dependent=experiment_design.primary_metric.column,
        num_iterations=int(self._get_param(params, "num_iterations")),
        min_treatment_geos=int(self._get_param(params, "min_treatment_geos")),
        max_treatment_geos=int(self._get_param(params, "max_treatment_geos")),
        force_test_geos=list(self._get_param(params, "force_test_geos")),
        force_control_geos=list(self._get_param(params, "force_control_geos")),
        exclude_geos=exclude_geos_input,
        effect_size=float(self._get_param(params, "effect_size")),
        test_duration=int(self._get_param(params, "test_duration")),
        alpha=float(self._get_param(params, "alpha")),
        target_power=float(self._get_param(params, "target_power"))
    )

    best_assignment_series = results_df.iloc[0]

    final_treatment_geos_list = best_assignment_series.get("treatment_geos", [])
    final_control_geos_list = best_assignment_series.get("control_geos", [])

    if final_treatment_geos_list:
      treatment_as_set = final_treatment_geos_list
    else:
      treatment_as_set = []
    treatment_as_set = set(treatment_as_set)

    if final_control_geos_list:
      control_as_set = final_control_geos_list
    else:
      control_as_set = []
    control_as_set = set(control_as_set)

    exclude_as_set = set(exclude_geos_input if exclude_geos_input else [])

    all_assigned_geos_set = treatment_as_set | control_as_set | exclude_as_set

    return GeoAssignment(
        treatment=[treatment_as_set],
        control=control_as_set,
        exclude=exclude_as_set,
        all_geos=all_assigned_geos_set
    )

  def _get_param(
      self,
      methodology_parameters: dict[str, Any],
      param_name: str,
      required: bool = False
  ) -> Any:
    """Assigns geos to control and test based on the synthetic controls method.

    Args:
      methodology_parameters: a dictionary with the methodology parameters.
      param_name: the name of the param to get.
      required: boolean - is it required or not.

    Returns:
      The value of the parameter that was required.
    """

    val = methodology_parameters.get(
        param_name, self.default_params.get(param_name)
    )
    if val is None and required:
      if param_name in self.default_params:
        return self.default_params[param_name]
      raise ValueError(
          f"""Required parameter '{param_name}'
          not found in methodology_parameters or defaults.""")
    return val

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


# pylint: disable=unused-argument]
def run_experiment_simulations(
    df: pd.DataFrame,
    geo_var: str,
    time_var: str,
    dependent: str,
    num_iterations: int,
    min_treatment_geos: int,
    max_treatment_geos: int,
    force_test_geos: list[str],
    force_control_geos: list[str],
    exclude_geos: list[str],
    effect_size: float,
    test_duration: int,
    alpha: float,
    target_power: float,
) -> pd.DataFrame:
  """Wraps the provided script for running synthetic control simulations.

  Args:
    df: DataFrame containing the historical data.
    geo_var: Name of the column identifying the geo units.
    time_var: Name of the column identifying the time periods.
    dependent: Name of the dependent variable column.
    num_iterations: Number of simulation iterations.
    min_treatment_geos: Minimum number of geos in the treatment group.
    max_treatment_geos: Maximum number of geos in the treatment group.
    force_test_geos: List of geos to always include in the treatment group.
    force_control_geos: List of geos to always include in the control group.
    exclude_geos: List of geos to exclude from any group.
    effect_size: Assumed effect size for power calculations.
    test_duration: Duration of the hypothetical test (e.g., in days) for
      power calculations.
    alpha: Significance level for power calculations.
    target_power: Target power for MDE calculations.

  Returns:
    A DataFrame containing the results of the simulations, sorted by power.
  """
  pass
