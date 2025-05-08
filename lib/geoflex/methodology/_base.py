"""The base class for all methodologies, to ensure a unified interface."""

import abc
import logging
from typing import Any
import geoflex.data
import geoflex.experiment_design
import geoflex.exploration_spec
import pandas as pd


ExperimentDesign = geoflex.experiment_design.ExperimentDesign
ExperimentDesignExplorationSpec = (
    geoflex.exploration_spec.ExperimentDesignExplorationSpec
)
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
GeoAssignment = geoflex.experiment_design.GeoAssignment

_METHODOLOGIES = {}
logger = logging.getLogger(__name__)


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

  @property
  def default_methodology_parameter_candidates(self) -> dict[str, list[Any]]:
    """All the parameters that are specific to this methodology.

    This should add the parameters that are specific to this methodology,
    that will be placed in the ExperimentDesign.methodology_parameters dict.
    The parameter names must not overlap with any of the other parameter names
    in the ExperimentDesign object.

    For each parameter, the list of valid values should be provided. The first
    value in the list will be used as the default value for the parameter, if
    an experiment design does not specify a value for it.

    Returns:
      A dictionary of parameter names and a list of valid values. Default is an
      empty dictionary.
    """
    return {}

  @abc.abstractmethod
  def _methodology_assign_geos(
      self,
      experiment_design: ExperimentDesign,
      historical_data: GeoPerformanceDataset,
  ) -> GeoAssignment:
    """How the methodology assigns geos to the control and treatment groups.

    This should return a Geo Assignment object containing the assignment to
    control and treatment groups. It must respect the geo eligibility, cell
    volume and number of cells from the experiment design.

    Make sure to use experiment_design.get_rng() to get a random number
    generator if your methodology requires any randomization, so that the
    assignment is reproducible.

    Args:
      experiment_design: The experiment design to assign geos for.
      historical_data: The historical data for the experiment. Can be used to
        choose geos that are similar to geos that have been used in the past.

    Returns:
      A GeoAssignment object containing the lists of geos for the control and
      treatment groups, and optionally a list of geos that should be ignored.
    """
    pass

  def assign_geos(
      self,
      experiment_design: ExperimentDesign,
      historical_data: GeoPerformanceDataset,
  ) -> GeoAssignment:
    """Assigns geos to the control and treatment groups.

    This will call the _methodology_assign_geos() method to do the actual
    assignment, and it will ensure that the assignment is valid and contains
    all the geos in the historical data.

    Args:
      experiment_design: The experiment design to assign geos for.
      historical_data: The historical data for the experiment. Can be used to
        choose geos that are similar to geos that have been used in the past.

    Returns:
      A GeoAssignment object containing the lists of geos for the control and
      treatment groups, and optionally a list of geos that should be ignored.
    """
    geo_assignment = self._methodology_assign_geos(
        experiment_design.model_copy(deep=True), historical_data
    )

    # Put missing geos into the exclude list.
    missing_geos = (
        set(historical_data.geos)
        - set().union(*geo_assignment.treatment)
        - set(geo_assignment.control)
    )
    if missing_geos:
      excluded_geos = geo_assignment.exclude | missing_geos
      geo_assignment = geo_assignment.model_copy(
          update={"exclude": excluded_geos}
      )

    # Check that the number of treatment groups is correct.
    if len(geo_assignment.treatment) != (experiment_design.n_cells - 1):
      error_message = (
          f"Assign_geos created {len(geo_assignment.treatment)} treatment"
          " groups, but the experiment design requires"
          f" {experiment_design.n_cells - 1}."
      )
      logger.error(error_message)
      raise ValueError(error_message)

    # Check that the control group is not empty.
    if not geo_assignment.control:
      error_message = "Assign_geos assigned no geos to the control group."
      logger.error(error_message)
      raise ValueError(error_message)

    # Check that the treatment groups are not empty.
    for i, treatment_group in enumerate(geo_assignment.treatment):
      if not treatment_group:
        error_message = (
            "Assign_geos assigned no geos to treatment"
            f" group {i+1}, but at least 1 geo is required."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    return geo_assignment

  @abc.abstractmethod
  def _methodology_analyze_experiment(
      self,
      runtime_data: GeoPerformanceDataset,
      experiment_design: ExperimentDesign,
      experiment_start_date: str,
  ) -> pd.DataFrame:
    """How the methodology analyzes the experiment.

    Must return a dataframe with the analysis results. Each row represents each
    metric provided in the experiment data. The columns are the following:

    - metric: The metric name.
    - cell: The cell number.
    - point_estimate: The point estimate of the treatment effect.
    - lower_bound: The lower bound of the confidence interval.
    - upper_bound: The upper bound of the confidence interval.
    - point_estimate_relative: The relative effect size of the treatment. This
    should be NA if the metric is a cost per metric or metric per cost.
    - lower_bound_relative: The relative lower bound of the confidence interval.
    This should be NA if the metric is a cost per metric or metric per cost.
    - upper_bound_relative: The relative upper bound of the confidence interval.
    This should be NA if the metric is a cost per metric or metric per cost.

    Optionally you can also provide:
    - p_value: The p-value of the null hypothesis.
    - is_significant: Whether the null hypothesis is rejected.

    If the p_value and is_significant columns are not provided, they will be
    inferred based on the point estimates and confidence intervals.

    This will be wrapped by the analyze_experiment() method, which will apply
    some validation to the inputs and outputs.

    Args:
      runtime_data: The runtime data for the experiment.
      experiment_design: The design of the experiment being analyzed.
      experiment_start_date: The start date of the experiment.

    Returns:
      A dataframe with the analysis results.
    """
    pass

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
    - cell: The cell number.
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
    validated_experiment_design = experiment_design.model_copy()

    # If a parameter is missing, assign the first value in the list of
    # candidates as the default value.
    for (
        param_name,
        param_values,
    ) in self.default_methodology_parameter_candidates.items():
      if param_name not in validated_experiment_design.methodology_parameters:
        validated_experiment_design.methodology_parameters[param_name] = (
            param_values[0]
        )

    raw_results = self._methodology_analyze_experiment(
        runtime_data, validated_experiment_design, experiment_start_date
    )

    return raw_results


def register_methodology(
    methodology_class: type[Methodology],
) -> type[Methodology]:
  """Registers a methodology so it can be retrieved by name."""
  logger.info("Registering methodology: %s", methodology_class.__name__)
  _METHODOLOGIES[methodology_class.__name__] = methodology_class
  return methodology_class


def get_methodology(methodology_name: str) -> Methodology:
  """Returns the methodology with the given name."""
  if methodology_name not in _METHODOLOGIES:
    error_message = f"Methodology {methodology_name} not registered."
    logger.error(error_message)
    raise ValueError(error_message)

  return _METHODOLOGIES[methodology_name]()


def list_methodologies() -> list[str]:
  """Returns a list of all methodologies."""
  return list(_METHODOLOGIES.keys())


def assign_geos(
    experiment_design: ExperimentDesign,
    historical_data: GeoPerformanceDataset,
    add_to_design: bool = True,
) -> GeoAssignment | None:
  """Assigns geos to the control and treatment groups.

  Args:
    experiment_design: The experiment design to assign geos for.
    historical_data: The historical data for the experiment. Geos may be
      assigned based on this data to maximize power, depending on the
      methodology set in the experiment design.
    add_to_design: Whether to add the geo assignment to the experiment design.

  Returns:
    The geo assignment for the experiment, or None if the design is not valid
    for the methodology.
  """
  if not design_is_valid(experiment_design):
    logger.warning(
        "Design is not valid for methodology %s, skipping geo assignment.",
        experiment_design.methodology,
    )
    return None

  methodology = get_methodology(experiment_design.methodology)
  geo_assignment = methodology.assign_geos(experiment_design, historical_data)

  if add_to_design:
    experiment_design.geo_assignment = geo_assignment

  return geo_assignment


def analyze_experiment(
    experiment_design: ExperimentDesign,
    runtime_data: GeoPerformanceDataset,
    experiment_start_date: str,
) -> pd.DataFrame | None:
  """Analyzes an experiment using the methodology set in the design.

  Returns a dataframe with the analysis results. Each row represents each
  metric provided in the experiment data, and a different treatment cell.

  The columns are the following:

  - metric: The metric name.
  - cell: The cell number.
  - point_estimate: The point estimate of the treatment effect.
  - lower_bound: The lower bound of the confidence interval.
  - upper_bound: The upper bound of the confidence interval.
  - point_estimate_relative: The relative effect size of the treatment.
  - lower_bound_relative: The relative lower bound of the confidence interval.
  - upper_bound_relative: The relative upper bound of the confidence interval.
  - p_value: The p-value of the null hypothesis.
  - is_significant: Whether the null hypothesis is rejected.

  Args:
    experiment_design: The design of the experiment being analyzed.
    runtime_data: The runtime data for the experiment.
    experiment_start_date: The start date of the experiment, as a string in the
      format YYYY-MM-DD.

  Returns:
    A dataframe with the analysis results, or None if the design is not valid
    for the methodology.
  """
  if not design_is_valid(experiment_design):
    logger.warning(
        "Design is not valid for methodology %s, skipping analysis.",
        experiment_design.methodology,
    )
    return None

  methodology = get_methodology(experiment_design.methodology)
  return methodology.analyze_experiment(
      runtime_data, experiment_design, experiment_start_date
  )


def design_is_valid(experiment_design: ExperimentDesign) -> bool:
  """Checks if the experiment design valid.

  This will check if the specified methodology is eligible for the other
  parameters in the design. Not all methodologies will be able to use all
  parameters, so this will check that the values make sense together.

  Args:
    experiment_design: The experiment design to check.

  Returns:
    True if the design is valid, False otherwise.
  """
  methodology = get_methodology(experiment_design.methodology)
  return methodology.is_eligible_for_design(experiment_design)
