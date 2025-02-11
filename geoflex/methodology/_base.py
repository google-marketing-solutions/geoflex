"""The base class for all methodologies, to ensure a unified interface."""

import abc
import geoflex.data
import geoflex.experiment_design
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


class Methodology(abc.ABC):
  """Base class for all methodologies.

  This contains a unified interface to design and analyse experiments
  using different methodologies.
  """

  @abc.abstractmethod
  def is_eligible_for_constraints(
      self, design_constraints: ExperimentDesignConstraints
  ) -> bool:
    """Checks if this methodology is eligible for the given design constraints.

    Args:
      design_constraints: The design constraints to check against.

    Returns:
      True if this methodology is eligible for the given design constraints,
      False otherwise.
    """
    pass

  @abc.abstractmethod
  def add_parameters_to_search_space(
      self,
      design_constraints: ExperimentDesignConstraints,
      search_space_root: vz.SearchSpaceSelector,
  ) -> None:
    """Defines the parameter search space for this methodology.

    This is done by adding the parameters to the search space root. It must
    consider the design constraints, so that the parameters are within the
    allowed ranges and are compatible with each other.

    This should only add the parameters that are specific to this methodology,
    that will be placed in the ExperimentDesign.methodology_parameters dict.
    The parameter names must not overlap with any of the other parameter names
    in the ExperimentDesign object.

    For more information on how to define the search space, see
    https://oss-vizier.readthedocs.io/en/latest/guides/user/search_spaces.html

    Args:
      design_constraints: The design constraints for the experiment.
      search_space_root: The root of the search space to add the parameters to.
    """
    pass

  @abc.abstractmethod
  def assign_geos(
      self,
      experiment_design: ExperimentDesign,
      historical_data: GeoPerformanceDataset,
  ) -> GeoAssignment:
    """Assigns geos to the control and treatment groups.

    This should return two lists of geos, one for the treatment and one for the
    control. The geos should be chosen based on the parameters in the experiment
    design.

    Args:
      experiment_design: The experiment design to assign geos for.
      historical_data: The historical data for the experiment. Can be used to
        choose geos that are similar to geos that have been used in the past.

    Returns:
      A GeoAssignment object containing the lists of geos for the control and
      treatment groups, and optionally a list of geos that should be ignored.
    """
    pass

  @abc.abstractmethod
  def analyze_experiment(
      self,
      runtime_data: GeoPerformanceDataset,
      experiment_design: ExperimentDesign,
      experiment_start_date: str,
  ) -> pd.DataFrame:
    """Analyzes an experiment using this methodology.

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
    pass
