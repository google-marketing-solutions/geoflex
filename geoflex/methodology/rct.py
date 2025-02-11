"""The Randomized Controlled Trial (RCT) methodology for GeoFleX."""

from feedx import statistics
import geoflex.data
import geoflex.experiment_design
from geoflex.methodology import _base
import numpy as np
import pandas as pd
from vizier import pyvizier as vz

ExperimentDesign = geoflex.experiment_design.ExperimentDesign
ExperimentDesignConstraints = (
    geoflex.experiment_design.ExperimentDesignConstraints
)
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
ExperimentDesignEvaluation = (
    geoflex.experiment_design.ExperimentDesignEvaluation
)
GeoAssignment = geoflex.experiment_design.GeoAssignment


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

  def add_parameters_to_search_space(
      self,
      design_constraints: ExperimentDesignConstraints,
      search_space_root: vz.SearchSpaceSelector,
  ) -> None:
    """Defines the parameter search space for the RCT methodology.

    Args:
      design_constraints: The design constraints for the experiment.
      search_space_root: The root of the search space to add the parameters to.
    """
    search_space_root.add_float_param(
        name="treatment_propensity",
        min_value=0.25,
        max_value=0.75,
    )

  def assign_geos(
      self,
      experiment_design: ExperimentDesign,
      historical_data: GeoPerformanceDataset,
      rng: np.random.Generator,
  ) -> GeoAssignment:
    """Randomly assigns all geos to the treatment and control groups.

    The treatment propensity is used to determine how many geos should be in
    the treatment group. The number of geos in the treatment group will be
    rounded up to the nearest integer, to ensure that the treatment group always
    has at least one geo, but is capped to ensure there is also always at least
    one geo in the control group.

    Args:
      experiment_design: The experiment design to assign geos for.
      historical_data: The historical data for the experiment. Can be used to
        choose geos that are similar to geos that have been used in the past.
      rng: The random number generator to use for randomization, if needed.

    Returns:
      A GeoAssignment object containing the lists of geos for the control and
      treatment groups, and optionally a list of geos that should be ignored.
    """
    treatment_propensity = experiment_design.methodology_parameters[
        "treatment_propensity"
    ]
    n_treatment_geos = int(
        np.min([
            np.ceil(len(historical_data.geos) * treatment_propensity),
            len(historical_data.geos) - 1,
        ])
    )

    treatment_geos = rng.choice(
        historical_data.geos,
        n_treatment_geos,
        replace=False,
    ).tolist()
    control_geos = list(set(historical_data.geos) - set(treatment_geos))
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
          trimming_quantile=0.0,
          alpha=experiment_design.alpha,
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
