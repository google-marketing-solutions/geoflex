"""The Randomized Controlled Trial (RCT) methodology for GeoFleX."""

from feedx import statistics
import geoflex.data
import geoflex.experiment_design
import geoflex.exploration_spec
from geoflex.methodology import _base
import pandas as pd


ExperimentDesign = geoflex.experiment_design.ExperimentDesign
ExperimentDesignExplorationSpec = (
    geoflex.exploration_spec.ExperimentDesignExplorationSpec
)
GeoPerformanceDataset = geoflex.data.GeoPerformanceDataset
GeoAssignment = geoflex.experiment_design.GeoAssignment
CellVolumeConstraintType = geoflex.experiment_design.CellVolumeConstraintType

register_methodology = _base.register_methodology


@register_methodology
class RCT(_base.Methodology):
  """The Randomized Controlled Trial (RCT) methodology for GeoFleX.

  It is a very simple methodolgy based on a simple A/B test. It is not
  recommended for most experiments, but can be used as a baseline for
  comparison.

  Design:
    Geos are split randomly into treatment and control groups.

  Evaluation:
    The evaluation is done with a simple t-test on each test statistic.
  """

  default_methodology_parameter_candidates = {"trimming_quantile": [0.0, 0.05]}

  def is_eligible_for_design(self, design: ExperimentDesign) -> bool:
    """Checks if an RCT is eligible for the given design.

    For a RCT, the only constraints that matter are the fixed geos. Because the
    assignment is random, we can only run it if there are no fixed geos in the
    control or treatment groups. There can be fixed geos in the exclude group,
    these are excluded pre-randomization.

    Args:
      design: The design to check against.

    Returns:
      True if an RCT is eligible for the given design, False
        otherwise.
    """
    if design.methodology != "RCT":
      return False

    has_geo_eligibility_constraints = design.geo_eligibility is not None and (
        design.geo_eligibility.control or any(design.geo_eligibility.treatment)
    )
    return not has_geo_eligibility_constraints

  def _methodology_assign_geos(
      self,
      experiment_design: ExperimentDesign,
      historical_data: GeoPerformanceDataset,
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
        != CellVolumeConstraintType.NUMBER_OF_GEOS
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
    return GeoAssignment(treatment=geo_groups[1:], control=geo_groups[0])

  def _methodology_analyze_experiment(
      self,
      runtime_data: GeoPerformanceDataset,
      experiment_design: ExperimentDesign,
      experiment_start_date: str,
  ) -> pd.DataFrame:
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

    Returns:
      A dataframe with the analysis results.
    """
    is_during_runtime = (
        runtime_data.parsed_data[runtime_data.date_column]
        >= experiment_start_date
    )

    all_metrics = [
        experiment_design.primary_metric
    ] + experiment_design.secondary_metrics

    all_metric_columns = set()
    for metric in all_metrics:
      all_metric_columns.add(metric.column)
      if metric.cost_column:
        all_metric_columns.add(metric.cost_column)
    all_metric_columns = list(all_metric_columns)

    grouped_data = (
        runtime_data.parsed_data.loc[is_during_runtime]
        .groupby("geo_id")[all_metric_columns]
        .sum()
    )

    control_data = grouped_data.loc[
        grouped_data.index.isin(experiment_design.geo_assignment.control)
    ]

    results = []

    for cell, treatment_group in enumerate(
        experiment_design.geo_assignment.treatment, 1
    ):
      treatment_data = grouped_data.loc[
          grouped_data.index.isin(treatment_group)
      ]
      for metric in all_metrics:
        statistical_results = statistics.yuens_t_test_ind(
            treatment_data[metric.column].values,
            control_data[metric.column].values,
            trimming_quantile=experiment_design.methodology_parameters[
                "trimming_quantile"
            ],
            alpha=experiment_design.alpha,
            alternative=experiment_design.alternative_hypothesis,
        )

        point_estimate = statistical_results.absolute_difference
        lower_bound = statistical_results.absolute_difference_lower_bound
        upper_bound = statistical_results.absolute_difference_upper_bound
        point_estimate_relative = statistical_results.relative_difference
        lower_bound_relative = (
            statistical_results.relative_difference_lower_bound
        )
        upper_bound_relative = (
            statistical_results.relative_difference_upper_bound
        )

        # To estimate the cost change I just look at the point estimate of the
        # cost change. This is not accurate because it means we don't take into
        # account the variance of the cost change.
        # DO NOT USE THIS IN REAL EXPERIMENTS.
        if metric.metric_per_cost:
          absolute_cost_change = (
              treatment_data[metric.cost_column].mean()
              - control_data[metric.cost_column].mean()
          )

          point_estimate = point_estimate / absolute_cost_change
          lower_bound = lower_bound / absolute_cost_change
          upper_bound = upper_bound / absolute_cost_change

          # Cannot calculate relative differences if it's metric per cost.
          point_estimate_relative = pd.NA
          lower_bound_relative = pd.NA
          upper_bound_relative = pd.NA

        elif metric.cost_per_metric:
          absolute_cost_change = (
              treatment_data[metric.cost_column].mean()
              - control_data[metric.cost_column].mean()
          )

          point_estimate = absolute_cost_change / point_estimate
          lower_bound = absolute_cost_change / lower_bound

          upper_bound = absolute_cost_change / upper_bound
          # Cannot calculate relative differences if it's cost per metric.
          point_estimate_relative = pd.NA
          lower_bound_relative = pd.NA
          upper_bound_relative = pd.NA

        # For cost per metric or metric per cost, if the cost and/or metric is
        # negative, then the lower and upper bound can be flipped. Here I just
        # force them to be in the correct order.
        if lower_bound > upper_bound:
          lower_bound, upper_bound = upper_bound, lower_bound

        results.append({
            "cell": cell,
            "metric": metric.name,
            "is_primary_metric": metric == experiment_design.primary_metric,
            "point_estimate": point_estimate,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "point_estimate_relative": point_estimate_relative,
            "lower_bound_relative": lower_bound_relative,
            "upper_bound_relative": upper_bound_relative,
            "p_value": statistical_results.p_value,
        })
    return pd.DataFrame(results)
