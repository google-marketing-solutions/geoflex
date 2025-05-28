"""The base class for all methodologies, to ensure a unified interface."""

import abc
import datetime as dt
import logging
from typing import Any
import geoflex.data
import geoflex.experiment_design
import geoflex.exploration_spec
import geoflex.utils
import pandas as pd


ExperimentDesign = geoflex.experiment_design.ExperimentDesign
ExperimentDesignExplorationSpec = (
    geoflex.exploration_spec.ExperimentDesignExplorationSpec
)
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
GeoAssignment = geoflex.experiment_design.GeoAssignment
ExperimentBudgetType = geoflex.experiment_design.ExperimentBudgetType

_METHODOLOGIES = {}
logger = logging.getLogger(__name__)


class Methodology(abc.ABC):
  """Base class for all methodologies.

  This contains a unified interface to design and analyse experiments
  using different methodologies.
  """

  def _fill_missing_methodology_parameters(
      self, experiment_design: ExperimentDesign
  ) -> ExperimentDesign:
    """Fills in missing methodology parameters with default values."""
    validated_experiment_design = experiment_design.model_copy(deep=True)
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
    return validated_experiment_design

  def _methodology_is_eligible_for_design_and_data(
      self, design: ExperimentDesign, historical_data: GeoPerformanceDataset
  ) -> bool:
    """Methodology specific checks for eligibility for the design and dataset.

    This checks that the methodology can both design and analyze the given
    dataset, based on the given design.

    Args:
      design: The design to check against.
      historical_data: The historical data to check against.

    Returns:
      True if this methodology is eligible for the given design and dataset,
      False otherwise.
    """
    del design, historical_data  # Unused
    return True

  def is_eligible_for_design_and_data(
      self, design: ExperimentDesign, historical_data: GeoPerformanceDataset
  ) -> bool:
    """Checks if this methodology is eligible for the given design and dataset.

    This will check:

    1. The methodology in the design is the same as this methodology
    2. The data has sufficient time steps
    3. For a percentage change budget, the costs must be non-zero.
    4. All checks defined in _methodology_is_eligible_for_design_and_data()

    Args:
      design: The design to check against.
      historical_data: The historical data to check against.

    Returns:
      True if this methodology is eligible for the given design and dataset,
      False otherwise.
    """
    if design.methodology != self.__class__.__name__:
      logger.error(
          "Design methodology %s does not match called methodology %s.",
          design.methodology,
          self.__class__.__name__,
      )
      return False

    n_days_in_data = len(historical_data.dates)
    n_full_weeks_in_data = n_days_in_data // 7

    if n_full_weeks_in_data < 2 * design.runtime_weeks:
      # We need double the runtime weeks, to ensure there is enough runtime and
      # pretest data.
      logger.error(
          "Data has %d days, which is not enough for a %d week experiment, need"
          " double the runtime weeks.",
          n_days_in_data,
          design.runtime_weeks,
      )
      return False

    # Check that the cell volume column constraint metric column is in the data.
    if (
        design.cell_volume_constraint.constraint_type
        == geoflex.CellVolumeConstraintType.MAX_PERCENTAGE_OF_METRIC
    ):
      if (
          design.cell_volume_constraint.metric_column
          not in historical_data.parsed_data.columns
      ):
        logger.error(
            "The cell volume metric column %s is not in the data.",
            design.cell_volume_constraint.metric_column,
        )
        return False

    # Check that all of the metric columns are in the data.
    for metric in [design.primary_metric] + design.secondary_metrics:
      if (
          metric.cost_column
          and metric.cost_column not in historical_data.parsed_data.columns
      ):
        logger.error(
            "The cost column %s is not in the data.", metric.cost_column
        )
        return False

      if metric.column not in historical_data.parsed_data.columns:
        logger.error(
            "The response column %s is not in the data.", metric.column
        )
        return False

    # Check that the costs are non-zero for a percentage change budget.
    if (
        design.experiment_budget.budget_type
        == ExperimentBudgetType.PERCENTAGE_CHANGE
    ):
      for metric in [design.primary_metric] + design.secondary_metrics:
        if metric.cost_per_metric or metric.metric_per_cost:
          if historical_data.parsed_data[metric.cost_column].sum() <= 0:
            # The costs are zero, so a percentage change budget is not possible.
            logger.error(
                "Costs are zero or negative for metric %s, which is a cost per"
                " metric or metric per cost. A percentage change budget is not"
                " possible with zero costs.",
                metric.name,
            )
            return False

    # Add default parameters to the design.
    design = self._fill_missing_methodology_parameters(design)

    return self._methodology_is_eligible_for_design_and_data(
        design, historical_data
    )

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
  ) -> tuple[GeoAssignment, dict[str, Any]]:
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
      return_intermediate_data: bool = False,
  ) -> GeoAssignment | tuple[GeoAssignment, dict[str, Any]]:
    """Assigns geos to the control and treatment groups.

    This will call the _methodology_assign_geos() method to do the actual
    assignment, and it will ensure that the assignment is valid and contains
    all the geos in the historical data.

    Args:
      experiment_design: The experiment design to assign geos for.
      historical_data: The historical data for the experiment. Can be used to
        choose geos that are similar to geos that have been used in the past.
      return_intermediate_data: Whether to return the intermediate data. This
        will be different for each methodology, and it can be used to debug the
        assignment. It is a dict with custom keys and values for each
        methodology.

    Returns:
      A GeoAssignment object containing the lists of geos for the control and
      treatment groups, and optionally a list of geos that should be ignored.
    """

    # Add default parameters to the design.
    experiment_design = self._fill_missing_methodology_parameters(
        experiment_design
    )

    geo_assignment, intermediate_data = self._methodology_assign_geos(
        experiment_design, historical_data
    )
    if not return_intermediate_data:
      intermediate_data = {}

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

    # Check that the geo eligibility is respected.
    geo_eligibility = experiment_design.geo_eligibility.create_inflexible_geo_eligibility_from_geos(
        set(historical_data.geos)
    )

    bad_control_geos = set(geo_assignment.control) - geo_eligibility.control
    if bad_control_geos:
      error_message = (
          "Assign_geos assigned geos to the control group that are not eligible"
          f" for control: {bad_control_geos}."
      )
      logger.error(error_message)
      raise ValueError(error_message)

    bad_exclude_geos = set(geo_assignment.exclude) - geo_eligibility.exclude
    if bad_exclude_geos:
      error_message = (
          "Assign_geos assigned geos to the exclude group that are not eligible"
          f" for exclude: {bad_exclude_geos}."
      )
      logger.error(error_message)
      raise ValueError(error_message)

    for i, treatment_group in enumerate(geo_assignment.treatment):
      bad_treatment_geos = set(treatment_group) - geo_eligibility.treatment[i]
      if bad_treatment_geos:
        error_message = (
            "Assign_geos assigned geos to treatment group"
            f" {i+1} that are not eligible for treatment: {bad_treatment_geos}."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    return geo_assignment, intermediate_data

  @abc.abstractmethod
  def _methodology_analyze_experiment(
      self,
      runtime_data: GeoPerformanceDataset,
      experiment_design: ExperimentDesign,
      experiment_start_date: pd.Timestamp,
      experiment_end_date: pd.Timestamp,
      pretest_period_end_date: pd.Timestamp,
  ) -> tuple[pd.DataFrame, dict[str, Any]]:
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

    If the p_value column is not provided, it will be
    inferred based on the point estimates and confidence intervals.

    This will be wrapped by the analyze_experiment() method, which will apply
    some validation to the inputs and outputs.

    Args:
      runtime_data: The runtime data for the experiment.
      experiment_design: The design of the experiment being analyzed.
      experiment_start_date: The start date of the experiment.
      experiment_end_date: The end date of the experiment, or the date to end
        the analysis (not inclusive).
      pretest_period_end_date: The end date of the pretest period (not
        inclusive). This will always be less than or equal to the
        experiment_start_date.

    Returns:
      A dataframe with the analysis results.
    """
    pass

  def analyze_experiment(
      self,
      runtime_data: GeoPerformanceDataset,
      experiment_design: ExperimentDesign,
      experiment_start_date: str,
      experiment_end_date: str | None = None,
      pretest_period_end_date: str | None = None,
      return_intermediate_data: bool = False,
  ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Analyzes an experiment using this methodology.

    Returns a dataframe with the analysis results. Each row represents each
    metric provided in the experiment data. The columns are the following:

    - metric: The metric name.
    - is_primary_metric: Whether the metric is a primary metric.
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
      experiment_end_date: The end date of the experiment, or the date to end
        the analysis (not inclusive). If not provided, the analysis will infer
        this based on the start date and the runtime weeks, or the last date in
        the data, whichever is earlier.
      pretest_period_end_date: The end date of the pretest period (not
        inclusive). If not provided, it will be assumed to be the same as the
        experiment_start_date. This is useful to allow for a washout period
        before the experiment starts.
      return_intermediate_data: Whether to return the intermediate data. This
        will be different for each methodology, and it can be used to debug the
        analysis. It is a dict with custom keys and values for each methodology.

    Returns:
      A dataframe with the analysis results.
    """
    experiment_start_date = pd.to_datetime(experiment_start_date)
    data_end_date = runtime_data.parsed_data[
        runtime_data.date_column
    ].max() + dt.timedelta(days=1)

    # Infer the experiment end date if not provided.
    if experiment_end_date is None:
      runtime_end_date = experiment_start_date + dt.timedelta(
          weeks=experiment_design.runtime_weeks
      )
      experiment_end_date = min(runtime_end_date, data_end_date)
    else:
      experiment_end_date = pd.to_datetime(experiment_end_date)

    if experiment_end_date > data_end_date:
      logger.warning(
          "The experiment end date is after the last date in the data. The"
          " analysis will actually be for the last date in the data."
      )
      experiment_end_date = data_end_date

    if pretest_period_end_date is None:
      pretest_period_end_date = experiment_start_date
    else:
      pretest_period_end_date = pd.to_datetime(pretest_period_end_date)

    if pretest_period_end_date > experiment_start_date:
      error_message = (
          "The pretest period end date is after the experiment start date."
      )
      logger.error(error_message)
      raise ValueError(error_message)

    days_in_runtime = (experiment_end_date - experiment_start_date).days
    days_in_design_runtime = experiment_design.runtime_weeks * 7
    if days_in_runtime > days_in_design_runtime:
      logger.warning(
          "The analysis is using %s days of data, but the experiment was"
          " designed for %s days of data (%s weeks). This is not a problem if"
          " you are doing this intentionally, for example to include a cooldown"
          " period.",
          days_in_runtime,
          days_in_design_runtime,
          experiment_design.runtime_weeks,
      )
    elif days_in_runtime < days_in_design_runtime:
      logger.warning(
          "The analysis is using %s days of data, but the experiment was"
          " designed for %s days of data (%s weeks). This analysis is"
          " incomplete and you should wait for the full runtime before making"
          " any decisions.",
          days_in_runtime,
          days_in_design_runtime,
          experiment_design.runtime_weeks,
      )

    # Add default parameters to the design.
    validated_experiment_design = self._fill_missing_methodology_parameters(
        experiment_design
    )

    raw_results, intermediate_data = self._methodology_analyze_experiment(
        runtime_data,
        validated_experiment_design,
        experiment_start_date,
        experiment_end_date,
        pretest_period_end_date,
    )
    if not return_intermediate_data:
      intermediate_data = {}

    # Check that the required columns are present.
    required_columns = {
        "metric",
        "cell",
        "point_estimate",
        "lower_bound",
        "upper_bound",
        "point_estimate_relative",
        "lower_bound_relative",
        "upper_bound_relative",
    }
    optional_columns = set(["p_value"])
    missing_columns = required_columns - set(raw_results.columns)
    if missing_columns:
      error_message = (
          "The analysis results are missing the following columns:"
          f" {missing_columns}"
      )
      logger.error(error_message)
      raise ValueError(error_message)

    extra_columns = (
        set(raw_results.columns) - required_columns - optional_columns
    )
    if extra_columns:
      logger.warning(
          "The analysis results contain the following extra columns, which will"
          " be dropped: %s",
          extra_columns,
      )
      raw_results = raw_results.drop(extra_columns, axis=1)

    # Check that all metrics are present in the results.
    all_metrics = [
        validated_experiment_design.primary_metric
    ] + validated_experiment_design.secondary_metrics
    all_metrics_names = [metric.name for metric in all_metrics]
    missing_metrics = set(all_metrics_names) - set(raw_results["metric"])
    if missing_metrics:
      error_message = (
          "The analysis results are missing the following metrics:"
          f" {missing_metrics}"
      )
      logger.error(error_message)
      raise ValueError(error_message)

    # Check that for cost_per_metric and metric_per_cost metrics, the relative
    # effect size columns are NA.
    metrics_without_relative = [
        metric.name
        for metric in all_metrics
        if metric.cost_per_metric or metric.metric_per_cost
    ]
    is_metric_without_relative = raw_results["metric"].isin(
        metrics_without_relative
    )
    has_non_na_relative = raw_results["point_estimate_relative"].notna()
    if (is_metric_without_relative & has_non_na_relative).any():
      logger.warning(
          "The analysis results contain non-NA relative effect sizes for"
          " metrics that are cost per metric or metric per cost. These will be"
          " forced to NA."
      )

    raw_results.loc[is_metric_without_relative, "point_estimate_relative"] = (
        pd.NA
    )
    raw_results.loc[is_metric_without_relative, "lower_bound_relative"] = pd.NA
    raw_results.loc[is_metric_without_relative, "upper_bound_relative"] = pd.NA

    if "p_value" not in raw_results.columns:
      # Infer p-value if not provided.
      raw_results["p_value"] = raw_results.apply(
          lambda row: geoflex.utils.infer_p_value(
              row["point_estimate"],
              (row["lower_bound"], row["upper_bound"]),
              validated_experiment_design.alpha,
              validated_experiment_design.alternative_hypothesis,
          ),
          axis=1,
      )

    # Calculate is_significant based on p-value and alpha.
    raw_results["is_significant"] = (
        raw_results["p_value"] < validated_experiment_design.alpha
    )

    # Flag the primary metric.
    raw_results["is_primary_metric"] = (
        raw_results["metric"] == validated_experiment_design.primary_metric.name
    )
    every_cell_has_primary = (
        raw_results.groupby("cell")["is_primary_metric"].any().all()
    )
    if not every_cell_has_primary:
      error_message = (
          "Some of the cells are missing the results for the primary metric."
      )
      logger.error(error_message)
      raise ValueError(error_message)

    return raw_results, intermediate_data


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
  return [
      methodology_name
      for methodology_name in _METHODOLOGIES.keys()
      if methodology_name != "TestingMethodology"
  ]


def assign_geos(
    experiment_design: ExperimentDesign,
    historical_data: GeoPerformanceDataset,
    add_to_design: bool = True,
    return_intermediate_data: bool = False,
) -> GeoAssignment | None | tuple[GeoAssignment, dict[str, Any]]:
  """Assigns geos to the control and treatment groups.

  Args:
    experiment_design: The experiment design to assign geos for.
    historical_data: The historical data for the experiment. Geos may be
      assigned based on this data to maximize power, depending on the
      methodology set in the experiment design.
    add_to_design: Whether to add the geo assignment to the experiment design.
    return_intermediate_data: Whether to return the intermediate data. This will
      be different for each methodology, and it can be used to debug the geo
      assignment. It is a dict with custom keys and values for each methodology.

  Returns:
    The geo assignment for the experiment, or None if the design is not valid
    for the methodology.
  """
  if not design_is_eligible_for_data(experiment_design, historical_data):
    logger.warning(
        "Design or data are not valid for methodology %s, skipping geo"
        " assignment.",
        experiment_design.methodology,
    )
    return None

  methodology = get_methodology(experiment_design.methodology)
  geo_assignment, intermediate_data = methodology.assign_geos(
      experiment_design,
      historical_data,
      return_intermediate_data=return_intermediate_data,
  )

  if add_to_design:
    experiment_design.geo_assignment = geo_assignment

  if return_intermediate_data:
    return geo_assignment, intermediate_data
  else:
    return geo_assignment


def analyze_experiment(
    experiment_design: ExperimentDesign,
    runtime_data: GeoPerformanceDataset,
    experiment_start_date: str,
    experiment_end_date: str | None = None,
    pretest_period_end_date: str | None = None,
    return_intermediate_data: bool = False,
) -> pd.DataFrame | None | tuple[pd.DataFrame, dict[str, Any]]:
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
    experiment_end_date: The end date of the experiment, as a string in the
      format YYYY-MM-DD. If not provided, the analysis will infer this based on
      the start date and the runtime weeks, or the last date in the data,
      whichever is earlier.
    pretest_period_end_date: The end date of the pretest period (not inclusive),
      as a string in the format YYYY-MM-DD. If not provided, it will be assumed
      to be the same as the experiment_start_date. This is useful to allow for a
      washout period before the experiment starts.
    return_intermediate_data: Whether to return the intermediate data. This will
      be different for each methodology, and it can be used to debug the
      analysis. It is a dict with custom keys and values for each methodology.

  Returns:
    A dataframe with the analysis results, or None if the design is not valid
    for the methodology.
  """
  if pretest_period_end_date is None:
    pretest_period_end_date = experiment_start_date
  pretest_period_end_date = pd.to_datetime(pretest_period_end_date)

  pretest_data = GeoPerformanceDataset(
      data=runtime_data.parsed_data[
          runtime_data.parsed_data[runtime_data.date_column]
          < pretest_period_end_date
      ],
      geo_id_column=runtime_data.geo_id_column,
      date_column=runtime_data.date_column,
  )

  if not design_is_eligible_for_data(experiment_design, pretest_data):
    logger.warning(
        "Design or data are not valid for methodology %s, skipping analysis.",
        experiment_design.methodology,
    )
    return None, {}

  methodology = get_methodology(experiment_design.methodology)
  results, intermediate_data = methodology.analyze_experiment(
      runtime_data,
      experiment_design,
      experiment_start_date,
      experiment_end_date,
      pretest_period_end_date,
      return_intermediate_data=return_intermediate_data,
  )

  if return_intermediate_data:
    return results, intermediate_data
  else:
    return results


def design_is_eligible_for_data(
    experiment_design: ExperimentDesign, data: GeoPerformanceDataset
) -> bool:
  """Checks if the experiment design valid.

  This will check if the specified methodology is eligible for the other
  parameters in the design and the data. Not all methodologies will be able to
  use all parameters, so this will check that the values make sense together.

  Args:
    experiment_design: The experiment design to check.
    data: The data for the experiment.

  Returns:
    True if the design is valid, False otherwise.
  """
  methodology = get_methodology(experiment_design.methodology)
  return methodology.is_eligible_for_design_and_data(experiment_design, data)
