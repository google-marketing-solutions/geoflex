"""A testing methodology for GeoFleX. Used for testing purposes only."""

from typing import Any
import geoflex.data
import geoflex.experiment_design
import geoflex.exploration_spec
from geoflex.methodology import _base
import numpy as np
import pandas as pd
from scipy import stats


ExperimentDesign = geoflex.experiment_design.ExperimentDesign
ExperimentDesignExplorationSpec = (
    geoflex.exploration_spec.ExperimentDesignExplorationSpec
)
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
GeoAssignment = geoflex.experiment_design.GeoAssignment
CellVolumeConstraintType = geoflex.experiment_design.CellVolumeConstraintType

register_methodology = _base.register_methodology


@register_methodology
class TestingMethodology(_base.Methodology):
  """A testing methodology for GeoFleX. Used for testing purposes only.

  Design:
    Geos are split randomly into treatment and control groups.

  Evaluation:
    Always returns a fixed set of results.
  """

  default_methodology_parameter_candidates = {"mock_parameter": [1, 2]}

  def _methodology_is_eligible_for_design_and_data(
      self, design: ExperimentDesign, historical_data: GeoPerformanceDataset
  ) -> bool:
    """Checks if the testing methodology is eligible for the given design.

    Returns true as long as there are no geos that are forced into a control
    or treatment group, because this methodology will randomly assign the
    geos.

    Args:
      design: The design to check against.
      historical_data: The dataset to check against.

    Returns:
      True if an RCT is eligible for the given design, False
        otherwise.
    """

    has_geo_eligibility_constraints = design.geo_eligibility is not None and (
        design.geo_eligibility.control or any(design.geo_eligibility.treatment)
    )
    return not has_geo_eligibility_constraints

  def _methodology_assign_geos(
      self,
      experiment_design: ExperimentDesign,
      historical_data: GeoPerformanceDataset,
  ) -> tuple[GeoAssignment, dict[str, Any]]:
    """Randomly assigns all geos to the treatment and control groups.

    The treatment propensity and number of ignored geos is used to determine
    how many geos should be in the treatment group. The number of geos in the
    treatment group will be rounded to the nearest integer and clipped to
    ensure that at least 1 geo in both the treatment and control groups.

    Args:
      experiment_design: The experiment design to assign geos for.
      historical_data: The historical data for the experiment. Can be used to
        choose geos that are similar to geos that have been used in the past.

    Returns:
      A GeoAssignment object containing the lists of geos for the control and
      treatment groups, and optionally a list of geos that should be ignored.
    """
    rng = experiment_design.get_rng()

    if experiment_design.geo_eligibility is None:
      exclude_geos = set()
    else:
      exclude_geos = set(experiment_design.geo_eligibility.exclude)
    all_geos = set(historical_data.geos) - exclude_geos

    if (
        experiment_design.cell_volume_constraint.constraint_type
        != CellVolumeConstraintType.MAX_GEOS
    ):
      raise ValueError(
          "Unsupported cell volume constraint type:"
          f" {experiment_design.cell_volume_constraint.constraint_type}"
      )

    n_geos_per_group = experiment_design.cell_volume_constraint.values

    # If any of the groups are set to None, make an approximate equal split
    # For example, splitting 10 into 3 groups will give 3, 3, 4.
    # The order will be randomized, so it could be 4, 3, 3 etc.
    n_non_specified_cells = sum(1 for n in n_geos_per_group if n is None)
    n_remaining_geos = len(all_geos) - sum(
        n for n in n_geos_per_group if n is not None
    )

    if n_non_specified_cells:
      if n_remaining_geos == 0:
        n_geos_per_group = [n if n is not None else 0 for n in n_geos_per_group]
      else:
        base = n_remaining_geos // n_non_specified_cells
        remainder = n_remaining_geos % n_non_specified_cells
        n_geos_per_group_non_specified = [base] * n_non_specified_cells
        for i in range(remainder):
          n_geos_per_group_non_specified[i] += 1
        rng.shuffle(n_geos_per_group_non_specified)

        for i in range(len(n_geos_per_group)):
          if n_geos_per_group[i] is None:
            n_geos_per_group[i] = n_geos_per_group_non_specified.pop()

        if n_geos_per_group_non_specified:
          raise RuntimeError(  # pylint: disable=g-doc-exception
              "n_geos_per_group_non_specified should be empty at this point."
          )

    geo_groups = []
    for n_geos in n_geos_per_group:
      new_geo_group = rng.choice(
          list(all_geos),
          n_geos,
          replace=False,
      ).tolist()
      all_geos -= set(new_geo_group)
      geo_groups.append(new_geo_group)
    return GeoAssignment(treatment=geo_groups[1:], control=geo_groups[0]), {
        "mock_intermediate_result": "assign_geos"
    }

  def _methodology_analyze_experiment(
      self,
      runtime_data: GeoPerformanceDataset,
      experiment_design: ExperimentDesign,
      experiment_start_date: pd.Timestamp,
      experiment_end_date: pd.Timestamp,
      pretest_period_end_date: pd.Timestamp,
  ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Analyzes a RCT experiment.

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

    Args:
      runtime_data: The runtime data for the experiment.
      experiment_design: The design of the experiment being analyzed.
      experiment_start_date: The start date of the experiment.
      experiment_end_date: The end date of the experiment, or the date to end
        the analysis (not inclusive).
      pretest_period_end_date: The end date of the pretest period (not
        inclusive).

    Returns:
      A dataframe with the analysis results.
    """
    all_metrics = [
        experiment_design.primary_metric
    ] + experiment_design.secondary_metrics

    results = []

    rng = experiment_design.get_rng()

    for cell in range(1, experiment_design.n_cells):
      for metric in all_metrics:
        point_estimate = rng.normal()

        if experiment_design.alternative_hypothesis == "two-sided":
          lower_bound, upper_bound = stats.norm.interval(
              confidence=1.0 - experiment_design.alpha,
              loc=point_estimate,
              scale=1.0,
          )
        elif experiment_design.alternative_hypothesis == "greater":
          lower_bound = stats.norm.ppf(
              q=experiment_design.alpha, loc=point_estimate, scale=1.0
          )
          upper_bound = np.inf
        elif experiment_design.alternative_hypothesis == "less":
          upper_bound = stats.norm.ppf(
              q=1.0 - experiment_design.alpha, loc=point_estimate, scale=1.0
          )
          lower_bound = -np.inf
        else:
          raise ValueError(
              "Unsupported alternative hypothesis:"
              f" {experiment_design.alternative_hypothesis}"
          )

        if metric.cost_per_metric or metric.metric_per_cost:
          point_estimate_relative = pd.NA
          lower_bound_relative = pd.NA
          upper_bound_relative = pd.NA
        else:
          point_estimate_relative = point_estimate
          lower_bound_relative = lower_bound
          upper_bound_relative = upper_bound

        results.append({
            "cell": cell,
            "metric": metric.name,
            "point_estimate": point_estimate,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "point_estimate_relative": point_estimate_relative,
            "lower_bound_relative": lower_bound_relative,
            "upper_bound_relative": upper_bound_relative,
        })

    return pd.DataFrame(results), {
        "mock_intermediate_result": "analyze_experiment"
    }
