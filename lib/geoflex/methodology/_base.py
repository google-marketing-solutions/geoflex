"""The base class for all methodologies, to ensure a unified interface."""

import abc
from typing import Any
import geoflex.data
import geoflex.experiment_design
import numpy as np
import optuna as op
import pandas as pd


ExperimentDesign = geoflex.experiment_design.ExperimentDesign
ExperimentDesignSpec = geoflex.experiment_design.ExperimentDesignSpec
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
ExperimentDesignEvaluation = (
    geoflex.experiment_design.ExperimentDesignEvaluation
)
GeoAssignment = geoflex.experiment_design.GeoAssignment

_METHODOLOGIES = {}


class Methodology(abc.ABC):
  """Base class for all methodologies.

  This contains a unified interface to design and analyse experiments
  using different methodologies.
  """

  @abc.abstractmethod
  def is_eligible_for_design(self, design: ExperimentDesign) -> bool:
    """Checks if this methodology is eligible for the given design.

    Args:
      design: The design to check against.

    Returns:
      True if this methodology is eligible for the given design,
      False otherwise.
    """
    pass

  @abc.abstractmethod
  def suggest_methodology_parameters(
      self,
      design_spec: ExperimentDesignSpec,
      trial: op.Trial,
  ) -> dict[str, Any]:
    """Suggests the parameters for this trial.

    It must consider the design specification, so that the parameters are within
    the allowed ranges and are compatible with each other.

    This should only add the parameters that are specific to this methodology,
    that will be placed in the ExperimentDesign.methodology_parameters dict.
    The parameter names must not overlap with any of the other parameter names
    in the ExperimentDesign object.

    For more information on how to define the search space, see
    https://oss-vizier.readthedocs.io/en/latest/guides/user/search_spaces.html

    Args:
      design_spec: The design specification for the experiment.
      trial: The Optuna trial to use to suggest the parameters.

    Returns:
      A dictionary of the suggested parameters.
    """
    pass

  @abc.abstractmethod
  def assign_geos(
      self,
      experiment_design: ExperimentDesign,
      historical_data: GeoPerformanceDataset,
      rng: np.random.Generator,
  ) -> GeoAssignment:
    """Assigns geos to the control and treatment groups.

    This should return two lists of geos, one for the treatment and one for the
    control. The geos should be chosen based on the parameters in the experiment
    design.

    Args:
      experiment_design: The experiment design to assign geos for.
      historical_data: The historical data for the experiment. Can be used to
        choose geos that are similar to geos that have been used in the past.
      rng: The random number generator to use for randomization, if needed.

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

    Returns:
      A dataframe with the analysis results.
    """
    pass


def register_methodology(
    methodology_class: type[Methodology],
) -> type[Methodology]:
  """Registers a methodology so it can be retrieved by name."""
  _METHODOLOGIES[methodology_class.__name__] = methodology_class
  return methodology_class


def get_methodology(methodology_name: str) -> Methodology:
  """Returns the methodology with the given name."""
  return _METHODOLOGIES[methodology_name]()


def list_methodologies() -> list[str]:
  """Returns a list of all methodologies."""
  return list(_METHODOLOGIES.keys())
