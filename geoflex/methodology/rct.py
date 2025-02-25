"""The Randomized Controlled Trial (RCT) methodology for GeoFleX."""

from typing import Any
from feedx import statistics
import geoflex.data
import geoflex.experiment_design
from geoflex.methodology import _base
import numpy as np
import optuna as op
import pandas as pd
import pydantic


ExperimentDesign = geoflex.experiment_design.ExperimentDesign
ExperimentDesignConstraints = (
    geoflex.experiment_design.ExperimentDesignConstraints
)
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
ExperimentDesignEvaluation = (
    geoflex.experiment_design.ExperimentDesignEvaluation
)
GeoAssignment = geoflex.experiment_design.GeoAssignment


class RCTParameters(pydantic.BaseModel):
  """The parameters for the RCT methodology.

  Attributes:
    treatment_propensity: The propensity of geos to be in the treatment group,
      out of all the non-ignored geos. Defaults to 0.5 to maximize the power.
    n_geos_ignored: The number of geos to ignore. Defaults to 0 to maximize the
      power.
    trimming_quantile: The quantile to use for trimming. Higher trimming can
      lead to more power, but will lead to more geos being ignored. Cannot be
      greater than 0.5. Defaults to 0.0 to not trim any data.
  """

  treatment_propensity: float = pydantic.Field(default=0.5, gt=0.0, lt=1.0)
  n_geos_ignored: int = pydantic.Field(default=0, ge=0)
  trimming_quantile: float = pydantic.Field(default=0.0, ge=0.0, lt=0.5)


class RCT(_base.Methodology):
  """The Randomized Controlled Trial (RCT) methodology for GeoFleX.

  It is a very simple methodolgy based on a simple A/B test. It is not
  recommended for most experiments, but can be used as a baseline for
  comparison.

  Design:
    Geos are split randomly into treatment and control groups. The treatment
    group will receive the treatment, and the control group will not. We will
    try multiple random splits and pick the best one with the lowest variance.

  Evaluation:
    The evaluation is done with a simple t-test on each test statistic.
  """

  def is_eligible_for_constraints(
      self, design_constraints: ExperimentDesignConstraints
  ) -> bool:
    """Checks if an RCT is eligible for the given design constraints.

    For a RCT, the only constraints that matter are the fixed geos, which
    cannot be used in an RCT.

    Args:
      design_constraints: The design constraints to check against.

    Returns:
      True if an RCT is eligible for the given design constraints, False
        otherwise.
    """
    return not design_constraints.fixed_geos

  def suggest_methodology_parameters(
      self,
      design_constraints: ExperimentDesignConstraints,
      trial: op.Trial,
  ) -> dict[str, Any]:
    """Suggests the parameters for this trial.

    Args:
      design_constraints: The design constraints for the experiment.
      trial: The Optuna trial to use to suggest the parameters.

    Returns:
      A dictionary of the suggested parameters.
    """
    parameters = {}
    if design_constraints.geo_treatment_propensity_range is not None:
      parameters["treatment_propensity"] = trial.suggest_float(
          "treatment_propensity",
          design_constraints.geo_treatment_propensity_range[0],
          design_constraints.geo_treatment_propensity_range[1],
      )
    if design_constraints.n_geos_ignored_range is not None:
      parameters["n_geos_ignored"] = trial.suggest_int(
          "n_geos_ignored",
          design_constraints.n_geos_ignored_range[0],
          design_constraints.n_geos_ignored_range[1],
      )
    if design_constraints.trimming_quantile_range is not None:
      parameters["trimming_quantile"] = trial.suggest_float(
          "trimming_quantile",
          design_constraints.trimming_quantile_range[0],
          design_constraints.trimming_quantile_range[1],
      )
    return parameters

  def assign_geos(
      self,
      experiment_design: ExperimentDesign,
      historical_data: GeoPerformanceDataset,
      rng: np.random.Generator,
  ) -> GeoAssignment:
    """Randomly assigns all geos to the treatment and control groups.

    The treatment propensity and number of ignored geos is used to determine
    how many geos should be in the treatment group. The number of geos in the
    treatment group will be rounded to the nearest integer and clipped to
    ensure that at least 1 geo in both the treatment and control groups.

    Args:
      experiment_design: The experiment design to assign geos for.
      historical_data: The historical data for the experiment. Can be used to
        choose geos that are similar to geos that have been used in the past.
      rng: The random number generator to use for randomization, if needed.

    Returns:
      A GeoAssignment object containing the lists of geos for the control and
      treatment groups, and optionally a list of geos that should be ignored.
    """
    parameters = RCTParameters.model_validate(
        experiment_design.methodology_parameters
    )

    n_non_ignored_geos = max(
        2, len(historical_data.geos) - parameters.n_geos_ignored
    )
    n_treatment_geos = int(
        np.round(n_non_ignored_geos * parameters.treatment_propensity)
    )
    n_treatment_geos = max(min(n_treatment_geos, n_non_ignored_geos - 1), 1)

    non_ignored_geos = rng.choice(
        historical_data.geos,
        n_non_ignored_geos,
        replace=False,
    ).tolist()
    treatment_geos = rng.choice(
        non_ignored_geos,
        n_treatment_geos,
        replace=False,
    ).tolist()

    control_geos = list(set(non_ignored_geos) - set(treatment_geos))
    return GeoAssignment(treatment=treatment_geos, control=control_geos)

  def analyze_experiment(
      self,
      runtime_data: GeoPerformanceDataset,
      experiment_design: ExperimentDesign,
      experiment_start_date: str,
  ) -> pd.DataFrame:
    """Analyzes a RCT experiment.

    Returns a dataframe with the analysis results. Each row represents each
    metric provided in the experiment data. The columns are the following:

    - metric: The metric name.
    - point_estimate: The point estimate of the treatment effect.
    - lower_bound: The lower bound of the confidence interval.
    - upper_bound: The upper bound of the confidence interval.
    - point_estimate_relative: The relative effect size of the treatment.
    - lower_bound_relative: The relative lower bound of the confidence interval.
    - upper_bound_relative: The relative upper bound of the confidence interval.
    - p_value: The p-value of the null hypothesis.
    - is_significant: Whether the null hypothesis is rejected.

    Args:
      runtime_data: The runtime data for the experiment.
      experiment_design: The design of the experiment being analyzed.
      experiment_start_date: The start date of the experiment.

    Returns:
      A dataframe with the analysis results.
    """
    parameters = RCTParameters.model_validate(
        experiment_design.methodology_parameters
    )

    grouped_data = runtime_data.parsed_data.groupby("geo_id")[
        runtime_data.response_columns
    ].sum()

    treatment_data = grouped_data.loc[
        grouped_data.index.isin(experiment_design.geo_assignment.treatment)
    ]
    control_data = grouped_data.loc[
        grouped_data.index.isin(experiment_design.geo_assignment.control)
    ]

    results = []
    for metric in runtime_data.response_columns:
      statistical_results = statistics.yuens_t_test_ind(
          treatment_data[metric].values,
          control_data[metric].values,
          trimming_quantile=parameters.trimming_quantile,
          alpha=experiment_design.alpha,
          alternative=experiment_design.alternative_hypothesis,
      )
      results.append({
          "metric": metric,
          "point_estimate": statistical_results.absolute_difference,
          "lower_bound": statistical_results.absolute_difference_lower_bound,
          "upper_bound": statistical_results.absolute_difference_upper_bound,
          "point_estimate_relative": statistical_results.relative_difference,
          "lower_bound_relative": (
              statistical_results.relative_difference_lower_bound
          ),
          "upper_bound_relative": (
              statistical_results.relative_difference_upper_bound
          ),
          "p_value": statistical_results.p_value,
          "is_significant": statistical_results.is_significant,
      })
    return pd.DataFrame(results)
