"""The Randomized Controlled Trial (RCT) methodology for GeoFleX."""

import geoflex.data
import geoflex.experiment_design
from geoflex.methodology import _base
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
    raise NotImplementedError()

  def assign_geos(
      self,
      experiment_design: ExperimentDesign,
      historical_data: GeoPerformanceDataset,
  ) -> GeoAssignment:
    """Randomly assigns all geos to the treatment and control groups.

    Args:
      experiment_design: The experiment design to assign geos for.
      historical_data: The historical data for the experiment. Can be used to
        choose geos that are similar to geos that have been used in the past.

    Returns:
      A GeoAssignment object containing the lists of geos for the control and
      treatment groups, and optionally a list of geos that should be ignored.
    """
    raise NotImplementedError()

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
    """
    raise NotImplementedError()
